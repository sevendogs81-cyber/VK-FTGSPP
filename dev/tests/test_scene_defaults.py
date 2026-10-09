import math
import sys

import pytest
from omegaconf import OmegaConf

from ftgspp.config.compose import load_config, resolve_config, save_resolved_config
from ftgspp.config.training_schedule import TRAINING_PRESETS
from ftgspp.optional.da3_schedule import resolve_da3_schedule


def scene(frames=60, fps=60, **train):
    return OmegaConf.create({
        "data": {"root": "capture", "fps": fps,
                 "frames": {"start": 10, "stop": 10 + frames}},
        "train": train,
    })


@pytest.mark.parametrize("preset,frames,steps", [
    ("balanced", 60, 30000), ("balanced", 121, 61000),
    ("balanced", 300, 150000), ("balanced", 1, 15000),
    ("preview", 60, 15000), ("preview", 300, 75000),
    ("extended", 60, 45000), ("extended", 300, 225000),
])
def test_scene_schedule_preserves_its_own_two_phase_recipe(preset, frames, steps):
    config = resolve_config(scene(frames, preset=preset)).config
    assert config.pipeline.mode == "scene"
    assert config.pipeline.image_objective == "scene_rgb_v1"
    assert config.data.masks.source == "none"
    assert not config.init.da3.foreground_only
    assert config.train.iterations == steps
    assert config.init.num_gaussians == 2_000_000
    assert config.train.color_correction
    assert config.train.hard_separation
    assert config.train.color_correction_start == round(steps / 10)
    assert config.train.hard_separation_steps == round(steps / 10)
    assert list(config.train.hard_relocation_steps) == [round(steps / 10) // 2]
    assert config.train.relocation.start == round(steps / 60)
    assert config.train.relocation.stop == math.ceil(steps * 5 / 6)
    assert config.train.relocation.every == round(steps / 300)
    assert config.train.relocation.score_mode == "gate_included"
    assert config.train.relocation.score_mode_start == round(steps / 10)
    assert config.train.checkpoint_interval == round(steps / 15)
    assert config.train.background.mode == "fixed"
    assert config.train.lr_schedules["means"] == 0.01
    assert config.init.num_velocity_nns == 0


@pytest.mark.parametrize("fps,frames,stride", [
    (60, 300, 10), (30, 300, 5), (24, 300, 4), (120, 300, 20),
    (60, 2, 2), (60, 1, 1), (1, 10, 1),
])
def test_scene_stride_follows_seconds_and_short_clips(fps, frames, stride):
    config = resolve_config(scene(frames, fps)).config
    assert config.init.keyframe_stride == stride
    assert config.init.duration == pytest.approx(stride / fps)
    assert config.init.points_path.name == f"points_stride{stride}"
    assert config.init.da3.cache_path.name == f"points_da3_stride{stride}"


def test_explicit_clocks_stride_and_disabled_hs_are_preserved():
    raw = scene(iterations=None, preset="preview", hard_separation=False,
                hard_separation_steps=11, hard_relocation_steps=[2, 7],
                color_correction_start=20, checkpoint_interval=77,
                relocation={"start": 10, "stop": 1000, "every": 11, "score_mode_start": 30})
    raw.init = {"keyframe_stride": 3}
    config = resolve_config(raw).config
    assert config.train.iterations == 15000
    assert config.init.keyframe_stride == 3
    assert not config.train.hard_separation
    assert config.train.hard_separation_steps == 11
    assert list(config.train.hard_relocation_steps) == [2, 7]
    assert config.train.color_correction_start == 20
    assert config.train.checkpoint_interval == 77
    assert config.train.relocation.score_mode_start == 30
    assert config.train.relocation.stop == 1000


@pytest.mark.parametrize("steps", [1, 2, 5, 10, 20, 100])
def test_tiny_debug_horizons_have_valid_main_and_hs_clocks(steps):
    config = resolve_config(scene(iterations=steps)).config
    assert config.train.iterations == steps
    assert 0 <= config.train.relocation.start < config.train.relocation.stop <= steps
    assert config.train.color_correction_start < steps
    h = config.train.hard_separation_steps
    assert all(0 < value < h for value in config.train.hard_relocation_steps)


def test_saved_scene_plan_keeps_old_explicit_recipe(tmp_path, monkeypatch):
    raw = scene(300, iterations=30000, hard_separation_steps=3000,
                hard_relocation_steps=[1500], color_correction_start=3000,
                checkpoint_interval=2000, log_interval=2000,
                relocation={"start": 500, "stop": 25000, "every": 100, "score_mode_start": 3000})
    raw.init = {"keyframe_stride": 10}
    original = resolve_config(raw).config
    path = tmp_path / "config.resolved.yaml"
    save_resolved_config(original, path)
    artifact = OmegaConf.load(path)
    del artifact.config.train.preset
    OmegaConf.save(artifact, path)
    monkeypatch.setitem(TRAINING_PRESETS["scene"], "balanced", 999000)
    assert load_config(path) == original


def test_scene_da3_cap_uses_keyframes_not_total_frames():
    config = resolve_config(scene(300, 30)).config
    assert config.init.keyframe_stride == 5
    keys = math.ceil((300 - 1) / 5) + 1
    schedule = resolve_da3_schedule(config.init.da3, native_hw=(1080, 1920),
                                    num_gaussians=2_000_000, n_keyframes=keys, n_cameras=4)
    assert keys == 61
    assert schedule.max_points == 120000


def test_scene_print_config_has_both_effective_phases(tmp_path, capsys, monkeypatch):
    from ftgspp.cli import main

    path = tmp_path / "scene.yaml"
    OmegaConf.save(scene(300, 30), path)
    monkeypatch.setitem(sys.modules, "ftgspp.training.runner", None)
    main(["train", "--config", str(path), "--print-config"])
    printed = OmegaConf.create(capsys.readouterr().out)
    assert printed.train.iterations == 150000
    assert printed.train.hard_separation_steps == 15000
    assert printed.init.keyframe_stride == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scene.yaml"]
