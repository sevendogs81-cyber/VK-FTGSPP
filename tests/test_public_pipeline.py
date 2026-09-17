import dataclasses
import inspect
import logging
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ftgspp.config.compose import load_config, resolve_config, save_resolved_config
from ftgspp.io.checkpoint import ResumeCompatibilityError, validate_training_clocks
from ftgspp.training.distributed import DistributedPlan
from ftgspp.training.sampler import sample_rank_views
from ftgspp.training.runner import NativeRunner, _TrainingReporter


def node(**distributed):
    return OmegaConf.create({"data": {"root": "scene"}, "distributed": distributed})


@pytest.mark.parametrize("removed", [
    "train.batch_size=1", "train.batch_size=2", "distributed.global_batch_size=2",
    "distributed.K=1", "distributed.microsteps=1", "init.roma=true",
    "train.K=1", "pipeline.stages=[init,train]", "pipeline.check_only=true",
    "run.log_dir=logs", "train.image_objective=scene_rgb_v1",
    "eval.protocol=scene_rgb_v1",
    "pipeline.foreground_version=v1", "pipeline.foreground_preset=auto",
    "init.algorithm_version=v2", "init.budget_policy=adaptive",
    "init.sampling_policy=spatial_grid", "init.replica_opacity_policy=inherit",
    "init.da3.evidence_version=v1", "init.da3.camera_chunking=sequential",
    "init.da3.cache_sampling=spatial_grid",
    "init.da3.depth_conflict_policy=zero_trusted",
    "init.da3.silhouette_filter=off", "init.da3.silhouette_min_views=2",
    "init.da3.silhouette_support_fraction=0.9",
    "init.da3.boundary_width_px=2",
    "init.da3.target_confidence_percentile=20",
    "init.da3.enable_l1_cache=false", "init.da3.roi_mode=off",
    "init.da3.roi_margin=0.15", "init.da3.roi_min_side=64",
    "init.da3.roi_max_image_fraction=0.85",
    "init.da3.cache_coverage_fraction=0.5",
    "init.da3.fusion_retry_limit=0",
    "init.da3.voxel_query_full_tree=true",
    "init.sparse_point_policy=jitter", "init.sparse_jitter_scale=0.25",
    "init.velocity_estimator=median", "init.coverage_fraction=0.5",
    "init.isolate_source_scales=true", "init.time_anchor=true",
    "train.foreground_loss.alpha_weight=0.1",
])
def test_removed_config_fields_are_rejected(removed):
    with pytest.raises(Exception, match="(not in|Key)"):
        resolve_config(OmegaConf.merge(node(), OmegaConf.from_dotlist([removed])))


def test_retired_full_validation_on_export_is_stripped(tmp_path):
    raw = OmegaConf.merge(node(), OmegaConf.from_dotlist([
        "eval.full_validation_on_export=false",
    ]))
    plan = resolve_config(raw, base_dir=tmp_path)
    assert not hasattr(plan.config.eval, "full_validation_on_export")


def test_versioned_resolved_artifact_ignores_retired_fields(tmp_path):
    path = tmp_path / "config.resolved.yaml"
    save_resolved_config(resolve_config(node()).config, path)
    artifact = OmegaConf.load(path)
    artifact.config.pipeline.retired_strategy = "historical-value"
    artifact.config.init.da3.retired_strategy = {"nested": True}
    OmegaConf.save(artifact, path)

    loaded = load_config(path)
    assert not hasattr(loaded.pipeline, "retired_strategy")
    assert not hasattr(loaded.init.da3, "retired_strategy")


@pytest.mark.parametrize("policy,factor", [
    ("none", 1), ("linear_lr", 2), ("sqrt_lr", 2**0.5),
    ("effective_batch_adam", 2**0.5),
])
def test_resolved_plan_scales_once_and_is_immutable(tmp_path, policy, factor):
    raw = node(mode="sync", device_ids=[0, 2], world_size_lr_scaling=policy)
    plan = resolve_config(raw, base_dir=tmp_path)
    assert plan.config.train.lrs["means"] == pytest.approx(0.00016 * factor)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.yaml = ""
    copy = plan.config
    copy.train.lrs["means"] = 7
    assert plan.config.train.lrs["means"] != 7
    saved = tmp_path / "resolved.yaml"
    save_resolved_config(plan.config, saved)
    assert load_config(saved).train.lrs == plan.config.train.lrs
    assert plan.config.data.root == tmp_path / "scene"


def test_step_scaling_round_trip(tmp_path):
    plan = resolve_config(node(mode="sync", device_ids=[0, 2], world_size_lr_scaling="steps_div_p"))
    saved = tmp_path / "resolved.yaml"
    save_resolved_config(plan.config, saved)
    assert load_config(saved).train.iterations == plan.config.train.iterations == 15000


def test_one_view_per_rank_has_no_accumulation_api():
    assert "microsteps" not in inspect.signature(sample_rank_views).parameters
    fields = {field.name for field in dataclasses.fields(DistributedPlan)}
    assert not fields & {"microsteps", "per_device_batch_size", "global_batch_size"}
    for world_size in (1, 2):
        assert len(sample_rank_views(seed=4, step=2, world_size=world_size, num_views=17)) == world_size


def test_v6_requires_explicit_three_clocks():
    record = {"training": {"clocks": {
        "abi": "ftgspp-training-clocks-v2", "optimizer_steps": 8,
        "views_seen": 16, "wall_seconds": 0.2,
    }}}
    clocks, warning = validate_training_clocks(record, iteration=7)
    assert clocks["views_seen"] == 16 and warning is None
    record["training"]["clocks"]["microsteps"] = 8
    with pytest.raises(ResumeCompatibilityError, match="unknown"):
        validate_training_clocks(record, iteration=7)
    with pytest.raises(ResumeCompatibilityError):
        validate_training_clocks({"training": {}}, iteration=7)


def test_both_phases_use_local_adam_steps():
    from types import SimpleNamespace
    runner = NativeRunner(resolve_config(node()).config)
    calls = []
    backend = SimpleNamespace(train_view=lambda view, step, **kw: calls.append((view, step)))
    runner._optimizer_update(backend, None, (3,), 0, False, (), phase="hard_separation")
    assert calls == [(3, 0)]


def test_peer_error_aborts_before_joining_waiting_rank():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace
    from ftgspp.training.distributed import SynchronousBackendGroup
    peer_started, aborted = Event(), Event()
    leader = SimpleNamespace(abort_pair_session=lambda reason: aborted.set())
    group = object.__new__(SynchronousBackendGroup)
    group.backends = (leader, object())
    group._closed = False
    def operation(rank, backend):
        if rank == 0:
            peer_started.set()
            assert aborted.wait(2), "coordinator joined before aborting peer"
        else:
            assert peer_started.wait(2)
            raise ValueError("injected peer failure")
    with ThreadPoolExecutor(max_workers=2) as executor:
        group._executor = executor
        with pytest.raises(ValueError, match="injected peer failure"):
            group._parallel(operation)
    assert aborted.is_set()


def test_doctor_does_not_create_directories(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ftgspp.cli import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ftgspp.training.backend.NativeVulkanBackend",
                        lambda *a, **kw: SimpleNamespace(
                            device_metadata=lambda: {}, close=lambda: None))
    main(["doctor", "--devices", "0"])
    assert list(tmp_path.iterdir()) == []


def test_train_cli_pipeline_switch_resolves_foreground_profile(tmp_path, monkeypatch):
    from ftgspp.cli import main

    config_path = tmp_path / "scene.yaml"
    config_path.write_text("data:\n  root: scene\n", encoding="utf-8")
    captured = {}

    class Runner:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return None

    monkeypatch.setattr("ftgspp.training.runner.NativeRunner", Runner)
    main([
        "train", "--config", str(config_path),
        "--pipeline", "foreground",
    ])
    config = captured["config"]
    assert config.pipeline.mode == "foreground"
    assert config.init.backend == "auto"
    assert config.init.da3.foreground_only
    assert config.pipeline.image_objective == "composite_v1"
    assert config.pipeline.eval_protocol == "foreground_v1"


def test_train_cli_two_devices_warns_and_still_resolves(tmp_path, monkeypatch, capsys):
    from ftgspp.cli import main

    config_path = tmp_path / "scene.yaml"
    config_path.write_text("data:\n  root: scene\n", encoding="utf-8")
    captured = {}

    class Runner:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            return None

    monkeypatch.setattr("ftgspp.training.runner.NativeRunner", Runner)
    main(["train", "--config", str(config_path), "--devices", "0", "2"])
    assert "two-device training is in development" in capsys.readouterr().err
    config = captured["config"]
    assert config.distributed.mode == "sync"
    assert list(config.distributed.device_ids) == [0, 2]


def test_train_cli_rejects_more_than_two_devices(tmp_path, monkeypatch):
    from ftgspp.cli import main

    config_path = tmp_path / "scene.yaml"
    config_path.write_text("data:\n  root: scene\n", encoding="utf-8")
    monkeypatch.setattr(
        "ftgspp.training.runner.NativeRunner",
        lambda config: (_ for _ in ()).throw(AssertionError("should not train")),
    )
    with pytest.raises(SystemExit) as raised:
        main(["train", "--config", str(config_path), "--devices", "0", "1", "2"])
    assert raised.value.code == 2


def test_failed_overwrite_keeps_existing_config(tmp_path):
    cfg = resolve_config(node(), base_dir=tmp_path).config
    cfg.run.output_path.mkdir(parents=True)
    (cfg.run.output_path / "gaussians.npz").write_bytes(b"existing model")
    saved = cfg.run.output_path / "config.resolved.yaml"
    saved.write_text("original configuration")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        NativeRunner(cfg).run()
    assert saved.read_text() == "original configuration"
    assert not (cfg.run.output_path / "run.log").exists()


@pytest.mark.parametrize("resume", [False, True])
def test_fixed_training_pipeline(tmp_path, monkeypatch, resume):
    cfg = resolve_config(node(), base_dir=tmp_path).config
    cfg.train.resume = resume
    calls = []
    model = object()
    monkeypatch.setattr(NativeRunner, "_preflight", lambda self: object())
    def initialize(*args):
        calls.append("init")
        return model
    def train(self, dataset, initial):
        assert initial is (None if resume else model)
        calls.append("train")
        return model
    monkeypatch.setattr("ftgspp.training.runner.initialize_explicit", initialize)
    monkeypatch.setattr(NativeRunner, "_train", train)
    monkeypatch.setattr("ftgspp.training.runner.export_4dgs",
                        lambda path, model: calls.append("export"))
    run_log = cfg.run.output_path / "run.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    run_log.write_text("previous session\n", encoding="utf-8")
    before_handlers = tuple(logging.getLogger("ftgspp").handlers)
    NativeRunner(cfg).run()
    assert calls == (["train", "export"] if resume else ["init", "train", "export"])
    log_text = run_log.read_text(encoding="utf-8")
    assert (log_text.startswith("previous session\n") is resume)
    assert f"mode={'resume' if resume else 'train'}" in log_text
    assert "pipeline completed" in log_text
    assert tuple(logging.getLogger("ftgspp").handlers) == before_handlers


def test_preflight_failure_is_recorded_in_run_log(tmp_path, monkeypatch):
    cfg = resolve_config(node(), base_dir=tmp_path).config
    package_logger = logging.getLogger("ftgspp")
    before_handlers = tuple(package_logger.handlers)
    before_level = package_logger.level
    monkeypatch.setattr(
        NativeRunner, "_preflight",
        lambda self: (_ for _ in ()).throw(ValueError("bad fixture")),
    )

    with pytest.raises(ValueError, match="bad fixture"):
        NativeRunner(cfg).run()

    log_text = (cfg.run.output_path / "run.log").read_text(encoding="utf-8")
    assert "pipeline failed" in log_text
    assert "ValueError: bad fixture" in log_text
    assert tuple(package_logger.handlers) == before_handlers
    assert package_logger.level == before_level


def test_tty_reporter_tracks_progress_metrics_and_checkpoint(monkeypatch):
    class FakeBar:
        def __init__(self, *, total, initial, **kwargs):
            self.total = total
            self.n = initial
            self.postfix = {}
            self.closed = False

        def set_postfix(self, values, *, refresh):
            self.postfix = dict(values)

        def update(self, amount):
            self.n += amount

        def close(self):
            self.closed = True

    fake_stderr = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(sys, "stderr", fake_stderr)
    monkeypatch.setattr("ftgspp.training.runner.tqdm", FakeBar)
    reporter = _TrainingReporter()
    reporter.start_phase("main", total=10, initial=3)
    bar = reporter._bar
    reporter.observe_eval({"psnr": 28.125, "ssim": 0.91234})
    reporter.checkpoint_started("000004.npz")
    reporter.advance(4, {"total_loss": 0.0123456})

    assert bar.n == 5
    assert bar.postfix == {
        "loss": "0.012346", "PSNR": "28.12", "SSIM": "0.9123",
        "state": "saving 000004.npz",
    }
    reporter.checkpoint_finished()
    assert "state" not in bar.postfix
    reporter.close()
    assert bar.closed
