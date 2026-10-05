import copy
import numpy as np
import pytest
from omegaconf import OmegaConf
from ftgspp.config.compose import resolve_config
from ftgspp.training.sampler import sample_batch_views, sample_rank_views
from ftgspp.training.background import sample_rank_backgrounds
from ftgspp.io.checkpoint import (
    LOCAL_BATCH_ABI, capture_checkpoint_snapshot, publish_checkpoint,
    latest_checkpoint, load_checkpoint_full, validate_training_clocks,
    validate_resume_compatibility, ResumeCompatibilityError,
)
from tests.test_checkpoint_v6 import _metadata, _SnapshotBackend


def config(train=None, **extra):
    return resolve_config(OmegaConf.create({
        'pipeline': {'mode': 'foreground'},
        'data': {'root': 'fixture', 'masks': {'source': 'alpha'}},
        'train': train or {}, **extra,
    })).config


@pytest.mark.parametrize('value', [True, False, 0, -1, 2, 3, 8, 4.0, '4'])
def test_batch_rejects_invalid_or_coerced_values(value):
    with pytest.raises((ValueError, TypeError)):
        config({'batch_size': value})


@pytest.mark.parametrize('change', [
    {'train': {'color_correction': True}}, {'train': {'hard_separation': True}},
    {'distributed': {'sampling_policy': 'co_frame'}},
    {'distributed': {'world_size_lr_scaling': 'linear_lr'}},
    {'pipeline': {'mode': 'scene'}},
])
def test_batch_rejects_unsupported_combinations(change):
    change = copy.deepcopy(change)
    train = {'batch_size': 4, **change.pop('train', {})}
    with pytest.raises(ValueError, match='batch_size=4'):
        config(train, **change)


def test_flattened_exposures_backgrounds_and_resume_match_b1():
    c = config({'batch_size': 4})
    count, views = 52, 13  # Four-image batches cross odd-sized epoch seams.
    actual = [v for s in range(count//4) for v in
              sample_batch_views(seed=3, step=s, batch_size=4, num_views=views)]
    expected = [sample_rank_views(seed=3, step=s, world_size=1, num_views=views)[0]
                for s in range(count)]
    assert actual == expected
    assert all(len(set(actual[k:k+views])) == views for k in range(0,count,views))
    resumed = [v for s in range(5,count//4) for v in
               sample_batch_views(seed=3, step=s, batch_size=4, num_views=views)]
    assert resumed == actual[20:]
    def bg(s):
        return sample_rank_backgrounds(c.train.background, phase='main',
                                       logical_step=s, world_size=1)[0].color
    np.testing.assert_array_equal([bg(4*s+i) for s in range(13) for i in range(4)],
                                  [bg(s) for s in range(52)])


def test_v7_batch_checkpoint_roundtrip_and_wrong_clock_rejection(tmp_path):
    metadata = _metadata(world_size=1)
    metadata['training']['local_batch'] = {'abi': LOCAL_BATCH_ABI, 'size': 4}
    metadata['training']['clocks']['views_seen'] = 32
    snapshot = capture_checkpoint_snapshot(_SnapshotBackend(), 7, metadata=metadata)
    path = publish_checkpoint(tmp_path, snapshot)
    assert latest_checkpoint(tmp_path) == path
    with np.load(path) as f:
        assert int(f['runtime__format']) == 7
    restored = load_checkpoint_full(path)
    assert validate_training_clocks(restored.metadata, iteration=7)[0]['views_seen'] == 32
    changed = copy.deepcopy(metadata)
    changed['training'].pop('local_batch')
    with pytest.raises(ResumeCompatibilityError, match='local_batch'):
        validate_resume_compatibility(metadata, changed, allow_world_size_change=False)
    metadata['training']['clocks']['views_seen'] = 8
    with pytest.raises(ResumeCompatibilityError, match='four views'):
        validate_training_clocks(metadata, iteration=7)


def test_default_batch_preserves_single_view():
    assert config().train.batch_size == 1


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize('should_log', [False, True])
def test_shared_runner_dispatch_preserves_b1_and_batch_backgrounds(
    monkeypatch, batch_size, should_log,
):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from ftgspp.training.runner import NativeRunner

    monkeypatch.setenv('FTGSPP_TRAIN_LOG_GRAD', '1')
    runner = object.__new__(NativeRunner)
    runner.config = config({'batch_size': batch_size})
    backend = SimpleNamespace(train_view=Mock(), train_batch=Mock(),
        last_loss=Mock(return_value={'total_loss': 0.25}),
        set_collect_geometry_grad_stats=Mock(),
        geometry_gradient_stats=Mock(return_value={'geometry_gradient': 0.5}))
    views = tuple(range(batch_size))
    loss, timing, audit = runner._optimizer_update(
        backend, None, views, 7, should_log, None,
    )
    expected_backgrounds = tuple(sample_rank_backgrounds(
        runner.config.train.background, phase='main',
        logical_step=7 * batch_size + i, world_size=1,
    )[0].color for i in range(batch_size))
    if batch_size == 1:
        backend.train_view.assert_called_once_with(
            0, 7, collect_loss=should_log, background=expected_backgrounds[0],
        )
        backend.train_batch.assert_not_called()
    else:
        backend.train_batch.assert_called_once_with(
            views, 7, collect_loss=should_log, backgrounds=expected_backgrounds,
        )
        backend.train_view.assert_not_called()
    expected_loss = {'total_loss': 0.25} if should_log else None
    if batch_size == 1 and should_log:
        expected_loss['geometry_gradient'] = 0.5
        assert [call.args for call in backend.set_collect_geometry_grad_stats.call_args_list] == [(True,), (False,)]
        backend.geometry_gradient_stats.assert_called_once_with()
    else:
        backend.set_collect_geometry_grad_stats.assert_not_called()
        backend.geometry_gradient_stats.assert_not_called()
    assert loss == expected_loss
    assert backend.last_loss.call_count == int(should_log)
    assert timing == {}
    assert len(audit['background_contexts']) == batch_size
    assert all(row['rank'] == 0 for row in audit['background_contexts'])


@pytest.mark.parametrize('persistent', [False, True])
def test_batch_allows_each_view_to_grow_tile_capacity_before_final_replay(persistent):
    from ftgspp.training.backend import NativeVulkanBackend

    views = (4, 2, 6, 1)
    colors = np.arange(12, dtype=np.float32).reshape(4, 3) / 12

    class Native:
        active = False
        commits = 0

        def __init__(self):
            self.attempts = []

        def begin_batch(self, step, collect_loss):
            assert not self.active and step == 12 and collect_loss
            self.active = True
            self.attempts.append([])

        def accumulate_batch_view(self, view, color):
            assert self.active
            slot = len(self.attempts[-1])
            assert view == views[slot]
            np.testing.assert_array_equal(color, colors[slot])
            self.attempts[-1].append(view)
            if persistent or slot == len(self.attempts) - 1:
                self.active = False  # native rolls back before returning False
                return False
            return True

        def commit_batch(self):
            assert self.active and len(self.attempts[-1]) == 4
            self.active = False
            self.commits += 1

        def abort_batch(self):
            self.active = False

    backend = object.__new__(NativeVulkanBackend)
    backend.native = Native()
    backend._bind_train_rgb = lambda view: None
    backend._maybe_log_pipeline_telemetry = lambda step: None
    if persistent:
        with pytest.raises(RuntimeError, match='five attempts'):
            backend.train_batch(views, 12, backgrounds=colors)
        assert backend.native.commits == 0
    else:
        backend.train_batch(views, 12, backgrounds=colors)
        assert [len(x) for x in backend.native.attempts] == [1, 2, 3, 4, 4]
        assert backend.native.commits == 1
    assert len(backend.native.attempts) == 5
    assert not backend.native.active
