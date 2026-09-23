from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from importlib.resources import files
from os import PathLike as OSPathLike
from pathlib import Path
from typing import Any, get_args, get_type_hints

from omegaconf import DictConfig, OmegaConf

from ftgspp.config.schema import AppConfig
from ftgspp.config.validate import (
    apply_world_size_training_policy,
    compute_derived_fields,
    validate_static_config,
)

type PathLike = str | OSPathLike[str]
RESOLVED_SCHEMA = "ftgspp-resolved-v1"


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("ftgspp_min"):
        OmegaConf.register_new_resolver(
            "ftgspp_min",
            lambda left, right: min(int(left), int(right)),
        )


def _absolute(path: Path, base_dir: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_paths(config: AppConfig, *, base_dir: Path | None = None) -> None:
    base_dir = Path.cwd() if base_dir is None else Path(base_dir).resolve()
    data = config.data
    data.root = _absolute(data.root, base_dir)
    data.video_path = _absolute(data.video_path, base_dir)
    data.calibration_path = _absolute(data.calibration_path, base_dir)
    data.extracted_path = _absolute(data.extracted_path, base_dir)
    if data.view_manifest is not None:
        data.view_manifest = _absolute(data.view_manifest, base_dir)
    if data.test_manifest is not None:
        data.test_manifest = _absolute(data.test_manifest, base_dir)
    if data.masks.root is not None:
        data.masks.root = _absolute(data.masks.root, base_dir)
    if data.masks.manifest_path is not None:
        data.masks.manifest_path = _absolute(data.masks.manifest_path, base_dir)
    if data.masks.validity_root is not None:
        data.masks.validity_root = _absolute(data.masks.validity_root, base_dir)
    if config.init.points_path is not None:
        config.init.points_path = _absolute(config.init.points_path, base_dir)
    if config.init.da3.model_path is not None:
        config.init.da3.model_path = _absolute(config.init.da3.model_path, base_dir)
    if config.init.da3.cache_path is not None:
        config.init.da3.cache_path = _absolute(config.init.da3.cache_path, base_dir)
    if config.train.resume_checkpoint is not None:
        config.train.resume_checkpoint = _absolute(config.train.resume_checkpoint, base_dir)
    config.run.experiment_root = _absolute(config.run.experiment_root, base_dir)
    config.run.output_path = _absolute(config.run.output_path, base_dir)


def _pipeline_profile_defaults(config: DictConfig) -> DictConfig:
    """Return mode defaults below explicit user settings in merge priority.

    The profile is deliberately applied between package defaults and the user
    node. A community user only selects ``pipeline.mode``; controlled
    background-policy and initializer overrides remain explicit. Foreground
    uses carving by default with the composite_v1 / foreground_v1 contract.
    """

    mode = str(OmegaConf.select(config, "pipeline.mode", default="scene"))
    if mode == "scene":
        return OmegaConf.create({
            "model": {"marginal_gating": True},
            "train": {
                "color_correction": True,
                "hard_separation": True,
                "relocation": {"score_mode": "gate_included"},
            },
        })
    if mode != "foreground":
        return OmegaConf.create({})
    raw_mask_source = OmegaConf.select(config, "data.masks.source", default=None)
    raw_mask_root = OmegaConf.select(config, "data.masks.root", default=None)
    raw_mask_manifest = OmegaConf.select(
        config, "data.masks.manifest_path", default=None,
    )
    masks: dict[str, object] = {"required": True}
    if raw_mask_source is None:
        masks["source"] = "external"
    if raw_mask_source in {None, "external"} and (
        raw_mask_root is None and raw_mask_manifest is None
    ):
        masks["manifest_path"] = "${data.root}/manifests/masks.json"
    init_profile: dict[str, object] = {
        "backend": "carving",
        "points_path": (
            "${data.root}/points_vhull_ibr_stride"
            "${init.keyframe_stride}"
        ),
        "da3": {
            "cache_path": (
                "${data.root}/points_da3_foreground_stride"
                "${init.keyframe_stride}"
            ),
            "foreground_only": True,
            "min_consistent_views": None,
        },
    }
    profile: dict[str, object] = {
        "model": {"marginal_gating": True},
        "data": {"masks": masks},
        # Carving consumes only a provenance-checked visual-hull point set.
        "init": init_profile,
        "train": {
            "background": {"mode": "random_solid"},
            "relocation": {"score_mode": "gate_included"},
        },
        "eval": {
            "selector": "foreground_psnr",
        },
    }
    return OmegaConf.create(profile)


@dataclass(frozen=True)
class ResolvedRunPlan:
    """Immutable effective plan; callers receive isolated working config copies."""

    yaml: str

    @property
    def config(self) -> AppConfig:
        return OmegaConf.to_object(OmegaConf.merge(
            OmegaConf.structured(AppConfig), OmegaConf.create(self.yaml),
        ))


def resolve_config(
    config: DictConfig, *, base_dir: Path | None = None, effective: bool = False,
) -> ResolvedRunPlan:
    _register_resolvers()
    raw = OmegaConf.to_container(config, resolve=False)
    if isinstance(raw, dict):
        eval_node = raw.get("eval")
        if isinstance(eval_node, dict):
            eval_node.pop("full_validation_on_export", None)
        config = OmegaConf.create(raw)
    default_node = OmegaConf.load(str(files("ftgspp.config").joinpath("defaults.yaml")))
    profile_node = _pipeline_profile_defaults(config)
    merged = OmegaConf.merge(
        OmegaConf.structured(AppConfig), default_node, profile_node, config,
    )
    OmegaConf.resolve(merged)
    result = OmegaConf.to_object(merged)
    _resolve_paths(result, base_dir=base_dir)
    compute_derived_fields(result)
    if not effective:
        apply_world_size_training_policy(
            result, world_size=len(result.distributed.device_ids)
            if result.distributed.mode == "sync" else 1,
        )
    validate_static_config(result)
    node = _yaml_safe(OmegaConf.to_container(OmegaConf.structured(result), resolve=True))
    return ResolvedRunPlan(OmegaConf.to_yaml(OmegaConf.create(node)))

def _yaml_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    return value


def _nested_dataclass_type(annotation: Any) -> type | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    for candidate in get_args(annotation):
        if isinstance(candidate, type) and is_dataclass(candidate):
            return candidate
    return None


def _prune_resolved_fields(value: object, schema_type: type) -> object:
    """Drop retired fields only while loading a versioned run artifact.

    User YAML still merges directly into the structured schema and rejects
    unknown keys. Historical resolved configs remain readable without keeping
    removed strategy names in the runtime.
    """

    if not isinstance(value, dict):
        return value
    hints = get_type_hints(schema_type)
    result: dict[str, object] = {}
    for schema_field in fields(schema_type):
        name = schema_field.name
        if name not in value:
            continue
        child = value[name]
        child_type = _nested_dataclass_type(hints.get(name, schema_field.type))
        result[name] = (
            _prune_resolved_fields(child, child_type)
            if child_type is not None else child
        )
    return result


def load_config(path: PathLike) -> AppConfig:
    _register_resolvers()
    path = Path(path).expanduser().resolve()
    node = OmegaConf.load(path)
    if node.get("schema") == RESOLVED_SCHEMA:
        raw = OmegaConf.to_container(node.config, resolve=False)
        pruned = _prune_resolved_fields(raw, AppConfig)
        return resolve_config(
            OmegaConf.create(pruned), base_dir=path.parent, effective=True,
        ).config
    return resolve_config(node, base_dir=path.parent).config


def save_resolved_config(config: AppConfig, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = OmegaConf.structured(config)
    plain = _yaml_safe(OmegaConf.to_container(node, resolve=True, enum_to_str=True))
    OmegaConf.save(config=OmegaConf.create({"schema": RESOLVED_SCHEMA, "config": plain}), f=path, resolve=True)


def load_run_config(run_path: PathLike) -> AppConfig:
    run_path = Path(run_path)
    resolved = run_path / "config.resolved.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"missing config.resolved.yaml in {run_path}")
    return load_config(resolved)
