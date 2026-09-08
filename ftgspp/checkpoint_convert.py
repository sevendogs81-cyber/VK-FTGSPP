"""Bidirectional ``.pt`` (Torch migration) / standalone native ``.npz`` conversion.

Torch is imported only when converting a ``.pt`` checkpoint. Native Vulkan
training and evaluation stay Torch-free. There is no Torch training path in
this repository; ``.pt`` support exists only for migration. ``latest.json``
is accepted as a source and resolved to its immutable numbered V4 NPZ.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from ftgspp.native_runtime.checkpoint import (
    FORMAT, NATIVE_ABI, load_checkpoint_full, resolve_checkpoint_path,
)
from ftgspp.native_runtime.model import NativeGaussians


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            ".pt conversion requires the optional 'convert' dependencies "
            "(pip install -e '.[convert]'); not needed for native Vulkan training"
        ) from exc
    return torch


def _legacy_sidecars(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(".vk.npz"), path.with_suffix(".runtime.pt")


def _native_arrays(
    path: Path,
) -> tuple[
    NativeGaussians, dict[str, np.ndarray], int, dict,
    dict[str, object] | None,
]:
    with np.load(path, allow_pickle=False) as archive:
        if "runtime__format" not in archive.files:
            model = {key: np.array(archive[key], copy=True) for key in archive.files}
            optimizer, iteration = {}, -1
            rng = np.random.default_rng(0).bit_generator.state
            return (
                NativeGaussians.from_mapping(model), optimizer, iteration, rng,
                None,
            )
    # Route versioned checkpoints through the validating loader. V4 is accepted
    # only through this explicit migration entry point.
    snapshot = load_checkpoint_full(path, allow_v4_migration=True)
    return (
        snapshot.model, dict(snapshot.optimizer), snapshot.iteration,
        dict(snapshot.rng_state), dict(snapshot.metadata),
    )


def _legacy_to_native(
    path: Path, field_mode: str, *, marginal_gating: bool | None = None,
) -> tuple[
    NativeGaussians, dict[str, np.ndarray], int, dict,
    dict[str, object] | None, list[dict[str, object]],
]:
    torch = _torch()
    from ftgspp.compat.torch_models import Gaussians
    from ftgspp.compat.torch_models.velocity import VelocityField

    gs = Gaussians.load(path, map_location="cpu")
    with torch.no_grad():
        is_field = isinstance(gs.velocity_model, VelocityField)
        if is_field:
            if field_mode == "reject":
                raise ValueError("legacy checkpoint contains HashGrid+HyperMLP")
            velocities = gs.velocity_model(gs.means, gs.times).detach().cpu().numpy()
        else:
            velocities = gs.velocity_model.detach().cpu().numpy()
        sh = torch.cat((gs.sh_0, gs.sh_n), dim=1).detach().cpu().numpy()
        if sh.shape[1] > 16:
            raise ValueError(f"native ABI supports at most degree-3 SH, got {sh.shape[1]} coefficients")
        if sh.shape[1] < 16:
            sh = np.pad(sh, ((0,0),(0,16-sh.shape[1]),(0,0)))
        gate_raw = gs.marginal_gates.detach().cpu().numpy()
        if marginal_gating is False:
            gate_raw = np.full_like(gate_raw, -100.0, dtype=np.float32)
        model = NativeGaussians(
            means=gs.means.detach().cpu().numpy(),
            log_scales=gs.scales.detach().cpu().numpy(),
            quaternions=gs.quats.detach().cpu().numpy(),
            opacity_logits=gs.opacities.detach().cpu().numpy(),
            sh_coefficients=sh,
            times=gs.times.detach().cpu().numpy(),
            duration_raw=gs.durations.detach().cpu().numpy(),
            velocities=velocities,
            gate_raw=gate_raw,
            max_duration=float(gs.max_duration),
        )

    vk_path, runtime_path = _legacy_sidecars(path)
    optimizer: dict[str, np.ndarray] = {}
    source_metadata: dict[str, object] | None = None
    transformations: list[dict[str, object]] = []
    iteration = -1
    rng = np.random.default_rng(0).bit_generator.state
    if vk_path.is_file() != runtime_path.is_file():
        raise FileNotFoundError(f"incomplete legacy checkpoint sidecars beside {path}")
    if vk_path.is_file():
        with np.load(vk_path, allow_pickle=False) as archive:
            optimizer = {key: np.array(archive[key], copy=True) for key in archive.files}
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        iteration = int(runtime["iteration"])
        rng = runtime.get("relocation_rng", np.random.default_rng(0).bit_generator.state)
        saved_metadata = runtime.get("checkpoint_metadata")
        if isinstance(saved_metadata, dict):
            source_metadata = copy.deepcopy(saved_metadata)
    if is_field and field_mode == "sample":
        # A checkpoint produced by an earlier conversion may already carry
        # native field arrays in its Vulkan sidecar.  Sampling explicitly
        # requests that none of them leak into the destination checkpoint.
        optimizer = {key: value for key, value in optimizer.items() if not key.startswith("field_")}
        transformations.append({
            "kind": "velocity_field",
            "operation": "sample",
            "from": "field",
            "to": "explicit",
        })
    if vk_path.is_file():
        cc = runtime.get("color_correctors")
        if cc is not None:
            optimizer.update(_flatten_legacy_color_correctors(cc))
    if marginal_gating is False:
        transformations.append({
            "kind": "marginal_gating",
            "operation": "disable",
            "from": (
                dict(source_metadata.get("model_config", {})).get(
                    "marginal_gating", "unknown"
                ) if source_metadata is not None else "unknown"
            ),
            "to": False,
        })
    return model, optimizer, iteration, rng, source_metadata, transformations


def _flatten_legacy_color_correctors(state_dict: dict) -> dict[str, np.ndarray]:
    """Pack ColorCorrectors state_dict into native N×12 color_params (+ zero moments)."""
    indices = sorted(
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("_correctors.") and key.endswith(".mat")
    )
    if not indices:
        return {}
    n = max(indices) + 1
    params = np.zeros((n, 12), dtype=np.float32)
    for i in indices:
        mat = np.asarray(state_dict[f"_correctors.{i}.mat"], np.float32).reshape(9)
        vec = np.asarray(state_dict[f"_correctors.{i}.vec"], np.float32).reshape(3)
        params[i, :9] = mat
        params[i, 9:] = vec
    flat = params.reshape(-1)
    return {
        "color_params": flat,
        "color_m": np.zeros_like(flat),
        "color_v": np.zeros_like(flat),
    }


def _field_parameter_arrays(field) -> tuple[list, list]:
    main = [layer.weight for layer in field.hyper_mlp.layers]
    bias = [parameter for net in field.hyper_mlp.bias_nets for layer in net.layers for parameter in (layer.weight, layer.bias)]
    return main, bias


def _flatten_legacy_field(gs, field) -> dict[str, np.ndarray]:
    main, bias = _field_parameter_arrays(field)
    mlp = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in (*main, *bias)]).astype(np.float32)
    distill = gs.__dict__.get("_velocity_distill")
    if distill is None:
        distill = (gs.means.detach(), gs.times.detach(), gs.velocities_t(gs.times).detach())
    xyz, times, target = (value.detach().cpu().numpy().astype(np.float32) for value in distill)
    count_hash = field.enc_xyz.table.numel()
    count_mlp = mlp.size
    return {
        "field_hash_params": field.enc_xyz.table.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "field_hash_m": np.zeros(count_hash, np.float32),
        "field_hash_v": np.zeros(count_hash, np.float32),
        "field_mlp_params": mlp,
        "field_mlp_m": np.zeros(count_mlp, np.float32),
        "field_mlp_v": np.zeros(count_mlp, np.float32),
        "field_bounds_xyz": field.bounds_xyz._bounds.detach().cpu().numpy().astype(np.float32),
        "field_bounds_t": field.bounds_t._bounds.detach().cpu().numpy().astype(np.float32),
        "field_distill_xyz": xyz,
        "field_distill_times": np.pad(times, ((0,0),(0,1))).astype(np.float32),
        "field_distill_target": target,
        "field_distill_scale": np.asarray([max(float(np.linalg.norm(target, axis=1).mean()), 1e-6)], np.float32),
    }


def _copy_legacy_field_moments(field, state_dict: dict, output: dict[str, np.ndarray]) -> None:
    named = list(field.named_parameters())
    ids = [item for group in state_dict["param_groups"] for item in group["params"]]
    if len(ids) != len(named):
        return
    moments = {name: state_dict["state"].get(identifier) for (name, _), identifier in zip(named, ids)}
    hash_state = moments.get("enc_xyz.table")
    if hash_state:
        output["field_hash_m"] = np.asarray(hash_state["exp_avg"], np.float32).reshape(-1)
        output["field_hash_v"] = np.asarray(hash_state["exp_avg_sq"], np.float32).reshape(-1)
    ordered_names = [f"hyper_mlp.layers.{i}.weight" for i in range(4)]
    ordered_names += [
        f"hyper_mlp.bias_nets.{net}.layers.{layer}.{kind}"
        for net in range(4) for layer in range(4) for kind in ("weight", "bias")
    ]
    for source_key, destination in (("exp_avg", "field_mlp_m"), ("exp_avg_sq", "field_mlp_v")):
        values = [np.asarray(moments[name][source_key], np.float32).reshape(-1) for name in ordered_names if moments.get(name)]
        if len(values) == len(ordered_names):
            output[destination] = np.concatenate(values)


def _restore_native_field(torch, optimizer: dict[str, np.ndarray]):
    from ftgspp.compat.torch_models.velocity import Bounds, VelocityField

    field = VelocityField(
        Bounds(torch.from_numpy(optimizer["field_bounds_xyz"])),
        Bounds(torch.from_numpy(optimizer["field_bounds_t"])),
    )
    with torch.no_grad():
        field.enc_xyz.table.copy_(torch.from_numpy(optimizer["field_hash_params"]).reshape_as(field.enc_xyz.table))
        cursor = 0
        main, bias = _field_parameter_arrays(field)
        flat = optimizer["field_mlp_params"]
        for parameter in (*main, *bias):
            size = parameter.numel()
            parameter.copy_(torch.from_numpy(flat[cursor:cursor+size]).reshape_as(parameter))
            cursor += size
        if cursor != len(flat):
            raise ValueError("native HyperMLP checkpoint has a mismatched flattened layout")
    return field


def _array_layout(values: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        str(key): {
            "shape": [int(size) for size in np.asarray(value).shape],
            "dtype": np.asarray(value).dtype.str,
        }
        for key, value in sorted(values.items())
    }


def _converted_metadata(
    model: NativeGaussians,
    optimizer: dict[str, np.ndarray],
    phase: str,
) -> dict[str, object]:
    """Create an explicitly incomplete manifest for a non-V5 source."""

    return {
        "checkpoint_schema": "ftgspp-vulkan-v5-converted",
        "legacy_migration": True,
        "phase": str(phase),
        "phase_step": -1,
        "training": {
            "world_size": "unknown",
            "global_batch_size": "unknown",
            "sampling_policy": "unknown",
            "sampler_abi": "unknown",
        },
        "dataset": {
            "fingerprint": "unknown",
            "camera_dense_mapping": "unknown",
            "frames": "unknown",
            "fps": "unknown",
            "time_offsets": "unknown",
        },
        "model_config": {
            "velocity_model": "explicit",
            "marginal_gating": "unknown",
            "max_duration": float(model.max_duration),
            "sh_degree": 3,
            "color_correction": "unknown",
        },
        "runtime": {
            "native_abi": NATIVE_ABI,
            "native_reported_abi": "unknown",
            "devices": "unknown",
            "shaders": {"set_sha256": "unknown", "files": {}},
            "git": {"commit": "unknown", "dirty": "unknown"},
        },
    }


def _metadata_for_conversion(
    source_metadata: dict[str, object] | None,
    model: NativeGaussians,
    optimizer: dict[str, np.ndarray],
    *,
    phase: str,
    transformations: list[dict[str, object]],
) -> dict[str, object]:
    """Build explicit-only V5 metadata while preserving source provenance."""

    field_keys = sorted(
        str(key) for key in optimizer if str(key).startswith("field_")
    )
    if field_keys:
        raise ValueError(
            "V5 output is explicit-only; use --field-mode sample before "
            f"serialization: {field_keys}"
        )

    manifest = (
        copy.deepcopy(source_metadata)
        if source_metadata is not None
        else _converted_metadata(model, optimizer, phase)
    )
    source = copy.deepcopy(manifest)
    manifest["phase"] = str(phase)
    if any(item.get("kind") == "phase" for item in transformations):
        # A phase override cannot inherit the old phase-local counter.
        manifest["phase_step"] = -1

    model_config = dict(manifest.get("model_config", {}))
    if source_metadata is None:
        model_config["max_duration"] = float(model.max_duration)
    model_config["velocity_model"] = "explicit"
    if any(item.get("kind") == "marginal_gating" for item in transformations):
        model_config["marginal_gating"] = False
    manifest["model_config"] = model_config

    changes_algorithm = any(
        item.get("kind") in {"velocity_field", "marginal_gating"}
        for item in transformations
    )
    if changes_algorithm:
        algorithm = dict(manifest.get("algorithm", {}))
        config = algorithm.get("config")
        if isinstance(config, dict):
            config = copy.deepcopy(config)
            algorithm_model = dict(config.get("model", {}))
            algorithm_model["velocity_model"] = model_config["velocity_model"]
            algorithm_model["marginal_gating"] = model_config.get(
                "marginal_gating", "unknown"
            )
            config["model"] = algorithm_model
            canonical = json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
            updated_algorithm = copy.deepcopy(algorithm)
            updated_algorithm["sha256"] = hashlib.sha256(canonical).hexdigest()
            updated_algorithm["config"] = config
            manifest["algorithm"] = updated_algorithm
        elif source_metadata is not None:
            # Do not retain a source hash that describes different semantics.
            updated_algorithm = copy.deepcopy(algorithm)
            updated_algorithm["sha256"] = "unknown"
            updated_algorithm["config"] = "unknown"
            manifest["algorithm"] = updated_algorithm

    if transformations:
        history = list(manifest.get("migration_history", []))
        history.append({
            "schema": "ftgspp-checkpoint-transform-v1",
            "tool": "ftgspp-convert-checkpoint",
            "source": {
                "checkpoint_schema": source.get("checkpoint_schema", "unknown"),
                "phase": source.get("phase", "unknown"),
                "phase_step": source.get("phase_step", -1),
                "model_config": source.get("model_config", {}),
                "algorithm": source.get("algorithm", {}),
                # Runtime is the device/shader/Git provenance that produced
                # the source parameters. Keep it verbatim after conversion.
                "runtime": source.get("runtime", {}),
            },
            "transforms": copy.deepcopy(transformations),
        })
        manifest["migration_history"] = history

    model_arrays = model.arrays()
    layout = dict(manifest.get("layout", {}))
    layout.update({
        "model": _array_layout(model_arrays),
        "optimizer": _array_layout(optimizer),
        "color": _array_layout({
            key: value for key, value in optimizer.items()
            if str(key).startswith("color_")
        }),
    })
    manifest["layout"] = layout
    manifest["checkpoint_schema"] = "ftgspp-vulkan-v5-converted"
    runtime = dict(manifest.get("runtime", {}))
    runtime["native_abi"] = NATIVE_ABI
    runtime["native_reported_abi"] = "unknown"
    runtime["shaders"] = {"set_sha256": "unknown", "files": {}}
    manifest["runtime"] = runtime
    return manifest


def _write_native(
    path: Path, model: NativeGaussians, optimizer: dict[str, np.ndarray],
    iteration: int, rng: dict,
    *, metadata: dict[str, object] | None = None,
    phase: str = "unknown",
    transformations: list[dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if optimizer or iteration >= 0:
        arrays = {f"model__{key}": value for key, value in model.arrays().items()}
        arrays.update({f"optimizer__{key}": value for key, value in optimizer.items()})
        manifest = _metadata_for_conversion(
            metadata, model, optimizer, phase=str(phase),
            transformations=list(transformations or []),
        )
        arrays.update({
            "runtime__format": np.asarray(FORMAT, np.int32),
            "runtime__iteration": np.asarray(iteration, np.int64),
            "runtime__rng": np.asarray(json.dumps(rng)),
            "runtime__metadata": np.asarray(json.dumps(
                manifest, sort_keys=True, allow_nan=False,
            )),
        })
    else:
        arrays = model.arrays()
    np.savez(path, **arrays)


def _write_legacy(
    path: Path, model: NativeGaussians, optimizer: dict[str, np.ndarray],
    iteration: int, rng: dict,
    *, metadata: dict[str, object] | None = None,
) -> None:
    torch = _torch()
    from ftgspp.compat.torch_models import Gaussians

    t = lambda value: torch.from_numpy(np.ascontiguousarray(value)).float()
    velocity_model = (
        _restore_native_field(torch, optimizer)
        if "field_hash_params" in optimizer else t(model.velocities)
    )
    gs = Gaussians(
        means=t(model.means), scales=t(model.log_scales),
        quats=t(model.quaternions), opacities=t(model.opacity_logits),
        sh_0=t(model.sh_coefficients[:,:1]), sh_n=t(model.sh_coefficients[:,1:]),
        times=t(model.times), durations=t(model.duration_raw),
        velocity_model=velocity_model, max_duration=model.max_duration,
        marginal_gates=t(model.gate_raw),
    )
    if "field_distill_xyz" in optimizer:
        gs.__dict__["_velocity_distill"] = (
            t(optimizer["field_distill_xyz"]),
            t(optimizer["field_distill_times"][:, :1]),
            t(optimizer["field_distill_target"]),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    gs.save(path)
    if optimizer or iteration >= 0:
        vk_path, runtime_path = _legacy_sidecars(path)
        np.savez(vk_path, **optimizer)
        color_correctors = None
        if "color_params" in optimizer:
            from ftgspp.compat.torch_models.colorcorrect import ColorCorrectors
            flat = np.asarray(optimizer["color_params"], np.float32).reshape(-1)
            n = flat.size // 12
            color_correctors = ColorCorrectors(n)
            with torch.no_grad():
                for i in range(n):
                    row = flat[i * 12:(i + 1) * 12]
                    color_correctors._correctors[i].mat.copy_(
                        torch.from_numpy(row[:9].reshape(3, 3)))
                    color_correctors._correctors[i].vec.copy_(
                        torch.from_numpy(row[9:12]))
            color_correctors = color_correctors.state_dict()
        runtime = {
            "format": 2, "iteration": int(iteration),
            "python_rng": random.Random(0).getstate(),
            "numpy_rng": np.random.RandomState(0).get_state(),
            "relocation_rng": rng, "torch_rng": torch.get_rng_state(),
            "field_optimizer": None, "field_scheduler": None,
            "color_optimizer": None, "color_correctors": color_correctors,
            # Preserve V4 provenance through the optional Torch interchange
            # format. A subsequent .pt -> .npz conversion restores it instead
            # of fabricating an all-unknown converted manifest.
            "checkpoint_metadata": copy.deepcopy(metadata),
        }
        torch.save(runtime, runtime_path)


def convert(
    source: Path, destination: Path, field_mode: str = "reject",
    *, marginal_gating: bool | None = None, phase: str | None = None,
) -> None:
    source = resolve_checkpoint_path(source)
    if field_mode not in {"sample", "reject"}:
        raise ValueError(f"unsupported field conversion mode: {field_mode}")
    source_suffix = source.suffix.lower()
    if source_suffix == ".pt":
        (
            model, optimizer, iteration, rng, source_metadata, transformations,
        ) = _legacy_to_native(
            source, field_mode, marginal_gating=marginal_gating,
        )
    elif source_suffix == ".npz":
        model, optimizer, iteration, rng, source_metadata = _native_arrays(source)
        transformations = []
        if (
            source_metadata is not None
            and str(source_metadata.get("checkpoint_schema", "")).startswith(
                "ftgspp-vulkan-v4"
            )
        ):
            transformations.append({
                "kind": "checkpoint_schema",
                "operation": "migrate",
                "from": source_metadata.get("checkpoint_schema"),
                "to": "ftgspp-vulkan-v5",
            })
        if marginal_gating is False:
            source_gating = (
                dict(source_metadata.get("model_config", {})).get(
                    "marginal_gating", "unknown"
                ) if source_metadata is not None else "unknown"
            )
            model.gate_raw = np.full_like(
                model.gate_raw, -100.0, dtype=np.float32,
            )
            transformations.append({
                "kind": "marginal_gating",
                "operation": "disable",
                "from": source_gating,
                "to": False,
            })
        if "field_hash_params" in optimizer:
            if field_mode == "reject":
                raise ValueError("native checkpoint contains HashGrid+HyperMLP")
            if field_mode == "sample":
                torch = _torch()
                field = _restore_native_field(torch, optimizer)
                with torch.no_grad():
                    model.velocities = np.ascontiguousarray(
                        field(
                            torch.from_numpy(model.means),
                            torch.from_numpy(model.times),
                        ).cpu().numpy(),
                        dtype=np.float32,
                    )
                optimizer = {
                    key: value for key, value in optimizer.items()
                    if not key.startswith("field_")
                }
                transformations.append({
                    "kind": "velocity_field",
                    "operation": "sample",
                    "from": "field",
                    "to": "explicit",
                })
    else:
        raise ValueError("source must be latest.json or end in .pt/.npz")

    source_phase = (
        str(source_metadata.get("phase", "unknown"))
        if source_metadata is not None else "unknown"
    )
    native_phase = source_phase if phase is None else str(phase)
    if native_phase not in {
        "unknown", "main", "main_complete", "main_all_dead",
    }:
        raise ValueError(f"unsupported checkpoint phase: {native_phase}")
    if phase is not None and native_phase != source_phase:
        transformations.append({
            "kind": "phase",
            "operation": "override",
            "from": source_phase,
            "to": native_phase,
        })

    values = model, optimizer, iteration, rng
    if destination.suffix.lower() == ".pt":
        output_metadata = _metadata_for_conversion(
            source_metadata, model, optimizer, phase=native_phase,
            transformations=transformations,
        )
        _write_legacy(destination, *values, metadata=output_metadata)
    elif destination.suffix.lower() == ".npz":
        _write_native(
            destination, *values, metadata=source_metadata,
            phase=native_phase, transformations=transformations,
        )
    else:
        raise ValueError("destination must end in .pt or .npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--field-mode", choices=("sample", "reject"), default="reject",
        help="sample an explicit approximation or reject legacy field input",
    )
    parser.add_argument(
        "--no-marginal-gating", action="store_true",
        help="force gate_raw to -100 (native init / train when model.marginal_gating=false)",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "main", "main_complete", "main_all_dead",
        ),
        help=(
            "declare the source optimizer phase; required before resuming a "
            "legacy checkpoint whose iteration is at/after the main boundary"
        ),
    )
    args = parser.parse_args()
    convert(
        args.source, args.destination, args.field_mode,
        marginal_gating=False if args.no_marginal_gating else None,
        phase=args.phase,
    )
    print(f"converted {args.source} -> {args.destination}")


if __name__ == "__main__":
    main()
