from types import SimpleNamespace
import sys

import pytest

from ftgspp.training.evaluation import Evaluator


@pytest.mark.parametrize("scope,count,filename", [
    ("validation", 20, "full_validation_metrics.jsonl"),
    ("candidate", 7, "metrics.jsonl"),
    ("final_test", 20, "final_test_metrics.jsonl"),
])
def test_scope_controls_coverage_and_result_destination(tmp_path, scope, count, filename):
    config = SimpleNamespace(
        pipeline=SimpleNamespace(mode="scene", eval_protocol="scene_rgb_v1"),
        run=SimpleNamespace(output_path=tmp_path),
        eval=SimpleNamespace(frame_interval=2, candidate_max_views=7, selector="psnr",
                             background_colors=[[0., 0., 0.]]),
    )
    dataset = SimpleNamespace(
        frames=list(range(40)),
        eval_views=[SimpleNamespace(frame=i, camera_id=i % 4) for i in range(40)],
        evaluation_identity={"kind": "final_test"} if scope == "final_test" else None,
    )
    seen = []

    class Backend:
        def evaluate_holdout_view(self, index, **kwargs):
            seen.append(index)
            return dict(squared_error_sum=3., valid_pixels=100., ssim_sum=270.,
                        valid_windows=100., foreground_ssim_sum=270.,
                        foreground_valid_windows=100., foreground_squared_error_sum=3.,
                        foreground_samples=100.), None

    result = Evaluator(config).evaluate(Backend(), dataset, 9, scope=scope)
    assert result["views"] == len(seen) == count
    assert result["eligible_views"] == 20
    assert all(index % 2 == 0 for index in seen)
    assert result["evaluation_scope"] == ("candidate_subset" if scope == "candidate" else "full_validation")
    assert [path.name for path in tmp_path.iterdir()] == [filename]


@pytest.mark.parametrize("scope,identity", [
    ("candidate", {"kind": "final_test"}),
    ("validation", {"kind": "final_test"}),
    ("final_test", None),
    ("typo", None),
])
def test_scope_must_match_locked_dataset(scope, identity):
    with pytest.raises(ValueError):
        Evaluator(None).evaluate(None, SimpleNamespace(evaluation_identity=identity, eval_views=[]), 0, scope=scope)


@pytest.mark.parametrize("saved", [False, True])
def test_cli_rejects_quick_locked_test_before_loading_data(tmp_path, monkeypatch, saved):
    from ftgspp import cli

    monkeypatch.setitem(sys.modules, "ftgspp.training.runner", None)
    config = SimpleNamespace(
        run=SimpleNamespace(output_path=tmp_path),
        data=SimpleNamespace(test_manifest=tmp_path / "test.json" if saved else None),
    )
    monkeypatch.setattr(cli, "load_run_config", lambda path: config)
    monkeypatch.setattr(
        "ftgspp.training.evaluation.load_evaluation_scene",
        lambda config: pytest.fail("invalid scope must not load a dataset"),
    )
    arguments = ["eval", "--run", str(tmp_path), "--quick"]
    if not saved:
        arguments += ["--test-manifest", str(tmp_path / "test.json")]
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
