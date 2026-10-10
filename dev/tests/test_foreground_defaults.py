import sys

import pytest
from omegaconf import OmegaConf

from ftgspp.config.compose import load_config, resolve_config, save_resolved_config
from ftgspp.config.training_schedule import TRAINING_PRESETS
from ftgspp.optional.da3_schedule import resolve_da3_schedule


def foreground(frames=121, **train):
    return OmegaConf.create({
        "pipeline": {"mode": "foreground"},
        "data": {"root": "capture", "frames": {"start": 10, "stop": 10 + frames}},
        "train": train,
    })


@pytest.mark.parametrize("preset,frames,steps", [
    ("balanced", 121, 70000), ("balanced", 300, 174000),
    ("balanced", 1, 35000), ("balanced", 1000, 579000),
    ("preview", 121, 35000), ("preview", 300, 87000),
    ("extended", 121, 105000), ("extended", 300, 261000),
])
def test_adaptive_horizon_and_model_budget(preset, frames, steps):
    config = resolve_config(foreground(frames, preset=preset)).config
    assert config.train.iterations == steps
    assert config.init.num_gaussians == 2_000_000
    assert config.train.relocation.stop == steps // 2
    assert config.train.relocation.start == round(steps / 35)
    assert config.train.relocation.every == round(steps / 140)
    assert config.train.checkpoint_interval == round(steps / 35)
    assert config.train.lr_schedules["means"] == 0.03


def test_temporal_exposures_stable_without_coupling_camera_count_or_stride():
    short = resolve_config(foreground()).config
    raw = foreground(300)
    raw.data.train_cameras = list(range(48))
    raw.init = {"keyframe_stride": 7}
    raw.data.fps = 15
    long = resolve_config(raw).config
    assert long.train.iterations / 300 == pytest.approx(short.train.iterations / 121, rel=0.003)
    assert long.init.duration == pytest.approx(7 / 15)
    assert long.model.max_duration == 20


@pytest.mark.parametrize("steps", [1, 2, 100, 10000, 200000])
def test_explicit_horizons_override_preset_with_valid_default_clocks(steps):
    config = resolve_config(foreground(preset="preview", iterations=steps)).config
    assert config.train.iterations == steps
    assert 0 <= config.train.relocation.start < config.train.relocation.stop <= steps
    assert config.train.relocation.every >= 1


def test_manual_clocks_win_and_null_restores_auto():
    raw = foreground(
        iterations=None, checkpoint_interval=321, log_interval=123,
        relocation={"start": 5, "stop": 200, "every": 10},
    )
    config = resolve_config(raw).config
    assert config.train.iterations == 70000
    assert config.train.relocation.stop == 200
    assert config.train.checkpoint_interval == 321
    assert config.train.log_interval == 123
    assert raw.train.iterations is None  # Input is not mutated.


def test_saved_plans_never_rerun_budget_selection(tmp_path, monkeypatch):
    original = resolve_config(foreground()).config
    path = tmp_path / "config.resolved.yaml"
    save_resolved_config(original, path)
    monkeypatch.setitem(TRAINING_PRESETS["foreground"], "balanced", 999000)
    assert load_config(path) == original
    # Artifacts before train.preset existed still keep all effective clocks.
    node = OmegaConf.load(path)
    del node.config.train.preset
    OmegaConf.save(node, path)
    assert load_config(path) == original
    node.config.train.iterations = None
    OmegaConf.save(node, path)
    with pytest.raises(ValueError, match="new run"):
        load_config(path)


@pytest.mark.parametrize("raw,message", [
    (foreground(preset="unknown", iterations=70000), "train.preset"),
    (foreground(0), "frames.stop"),
])
def test_invalid_presets_and_auto_inputs_rejected(raw, message):
    with pytest.raises(Exception, match=message):
        resolve_config(raw)


def test_da3_dense_cap_adapts_to_actual_keyframes_and_explicit_override():
    config = resolve_config(foreground()).config
    for keyframes, expected in [(121, 120000), (300, 53328), (44, 120000)]:
        schedule = resolve_da3_schedule(
            config.init.da3, native_hw=(1080, 1920), num_gaussians=2_000_000,
            n_keyframes=keyframes, n_cameras=4,
        )
        assert schedule.max_points == expected
    config.init.da3.max_points = 150000
    schedule = resolve_da3_schedule(
        config.init.da3, native_hw=(1080, 1920), num_gaussians=2_000_000,
        n_keyframes=300, n_cameras=4,
    )
    assert schedule.max_points == 150000


def test_print_config_is_read_only_and_does_not_import_runner(tmp_path, capsys, monkeypatch):
    from ftgspp.cli import main

    path = tmp_path / "capture.yaml"
    OmegaConf.save(foreground(), path)
    # Importing the runner would transitively import the native-facing runtime.
    monkeypatch.setitem(sys.modules, "ftgspp.training.runner", None)
    main(["train", "--config", str(path), "--print-config",
          "--set", "train.preset=preview"])
    printed = OmegaConf.create(capsys.readouterr().out)
    assert printed.train.iterations == 35000
    assert printed.train.relocation.stop == 17500
    assert not (tmp_path / "runs").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["capture.yaml"]


def test_print_resume_config_keeps_saved_horizon(tmp_path, capsys, monkeypatch):
    from ftgspp.cli import main

    config = resolve_config(foreground(iterations=12345)).config
    run = tmp_path / "run"
    save_resolved_config(config, run / "config.resolved.yaml")
    monkeypatch.setitem(sys.modules, "ftgspp.training.runner", None)
    main(["train", "--resume", str(run), "--print-config"])
    assert OmegaConf.create(capsys.readouterr().out).train.iterations == 12345
    assert sorted(p.name for p in run.iterdir()) == ["config.resolved.yaml"]
