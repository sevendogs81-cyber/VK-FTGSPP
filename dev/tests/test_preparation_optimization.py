"""Preparation budgets, conservative bounds, occlusion and first-use contracts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ftgspp.optional.carving_bounds import silhouette_bounds, SilhouetteBoundsUnavailable
from ftgspp.optional.point_color import color_points_ibr, project_points
from ftgspp.optional.preparation_views import PreparationView
from ftgspp.optional.visual_hull import carve_visual_hull, preparation_point_cap
from ftgspp.optional import visual_hull
from ftgspp.training.initialize import _valid_prepared_point_manifest


def box_rig(center=(0, 0, 0), count=8):
    center = np.asarray(center, float)
    corners = np.array([[x, y, z] for x in (-.3, .3) for y in (-.5, .5)
                        for z in (-.3, .3)]) + center
    views = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        eye = np.array([4*np.cos(angle), 1, 4*np.sin(angle)])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.stack([right, down, forward])
        pose[:3, 3] = -pose[:3, :3] @ eye
        view = SimpleNamespace(w2c=pose, intrinsic=np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]]),
                               size=(128, 128), camera_id=index, frame=0, timestamp=0.)
        u, v, _ = project_points(corners, view)
        mask = np.zeros(view.size, np.float32)
        mask[max(0, int(np.floor(v.min()))):min(128, int(np.ceil(v.max()))+1),
             max(0, int(np.floor(u.min()))):min(128, int(np.ceil(u.max()))+1)] = 1
        view.load_mask = lambda mask=mask: mask
        view.load_validity = lambda: np.ones((128, 128), np.float32)
        view.load_rgb = lambda: np.full((128, 128, 3), 150, np.uint8)
        views.append(view)
    return views, corners


def test_generated_budget_adapts_to_actual_keyframes_and_user_clouds_are_preserved():
    assert preparation_point_cap(2_000_000, 121) == 66116
    assert preparation_point_cap(2_000_000, 300) == 26668
    assert preparation_point_cap(2_000_000, 121, quality="fast") == 33058
    assert preparation_point_cap(2_000_000, 121, quality="high") == 132232
    assert preparation_point_cap(2_000_000, 121, quality="full") is None
    assert preparation_point_cap(2_000_000, 121, generated_geometry=False) is None
    assert preparation_point_cap(2_000_000, 300, max_points=150000) == 150000


@pytest.mark.parametrize("center", [(0, 0, 0), (1., .2, 0)])
def test_bounds_follow_motion_and_tolerate_one_bad_silhouette(center):
    views, corners = box_rig(center)
    # A single incorrectly tiny mask must not collapse the object's envelope.
    mask = np.zeros((128, 128), np.float32)
    mask[62:66, 62:66] = 1
    views[0].load_mask = lambda: mask
    bounds, report = silhouette_bounds(views, support_fraction=.8)
    assert report["groups"] == 2
    assert np.all(corners >= bounds[0]) and np.all(corners <= bounds[1])


def test_bounds_exclude_clipped_views_without_tightening_other_constraints():
    views, corners = box_rig()
    views[0].load_mask = lambda: np.ones((128, 128), np.float32)
    bounds, report = silhouette_bounds(views)
    assert report["excluded_views"] == [{"camera_id": 0, "reason": "clipped_silhouette"}]
    assert np.all(corners >= bounds[0]) and np.all(corners <= bounds[1])
    for view in views:
        view.load_mask = lambda: np.ones((128, 128), np.float32)
    with pytest.raises(SilhouetteBoundsUnavailable, match="too few"):
        silhouette_bounds(views)


def test_unbounded_camera_rig_requires_depth_fallback():
    views, _ = box_rig()
    for view in views:
        view.w2c = views[0].w2c
    with pytest.raises(SilhouetteBoundsUnavailable, match="unbounded"):
        silhouette_bounds(views)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["visibility_robust", "depth_rank_top3"])
def test_query_subset_matches_full_coloring_with_all_occluders(device, mode):
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    views, _ = box_rig()
    # Add a deliberate occlusion along camera 0's principal ray.
    view = views[0]
    center = np.linalg.inv(view.w2c)[:3, 3]
    points = np.array([[0, 0, 0], center * .2, [.1, .2, .1], [30, 0, 0]], np.float32)
    query = np.array([3, 2, 0])
    full = color_points_ibr(points, [view], device=device, mode=mode)
    subset = color_points_ibr(points, [view], device=device, mode=mode, query_indices=query)
    np.testing.assert_array_equal(subset.keep, full.keep[query])
    np.testing.assert_array_equal(subset.rgb, full.rgb[query])
    np.testing.assert_array_equal(subset.source_count, full.source_count[query])
    np.testing.assert_allclose(subset.disagreement, full.disagreement[query], atol=0, equal_nan=True)
    if mode == "visibility_robust":
        assert not subset.keep[-1]
        alone = color_points_ibr(points[[0]], [view], device=device)
        assert alone.keep[0]  # Dropping the occluder would introduce wrong color.


@pytest.mark.parametrize("query", [[0, 0], [-1], [100], [0.5], [[0]]])
def test_invalid_query_indices_are_rejected(query):
    views, _ = box_rig()
    with pytest.raises(ValueError, match="query_indices"):
        color_points_ibr(np.zeros((2, 3), np.float32), views, query_indices=np.asarray(query))


def test_coarse_fine_and_color_load_each_mask_once():
    views, _ = box_rig()
    counts = {"mask": 0, "validity": 0}
    for view in views:
        for kind in counts:
            original = getattr(view, "load_" + kind)
            def counted(original=original, kind=kind):
                counts[kind] += 1
                return original()
            setattr(view, "load_" + kind, counted)
    prepared = [PreparationView(view) for view in views]
    bounds, _ = silhouette_bounds(prepared)
    points, _ = carve_visual_hull(prepared, bounds, resolution=24, coarse_resolution=6, device="cpu")
    colored = color_points_ibr(points, prepared, device="cpu", query_indices=np.arange(min(20, len(points))))
    assert colored.keep.any()
    assert counts == {"mask": len(views), "validity": len(views)}


@pytest.mark.parametrize("retry_color", [False, True])
def test_calibrated_carving_skips_depth_and_publishes_budgeted_valid_artifacts(tmp_path, monkeypatch, retry_color):
    views, _ = box_rig()
    scene = SimpleNamespace(frames=[0], train_cameras=list(range(8)), fps=30., calibration_available=True,
                            view=lambda *args: views[0], views_nearest_timestamp=lambda *a, **k: views,
                            checkpoint_metadata=lambda: {"fingerprint": "test-rig"})
    config = SimpleNamespace(init=SimpleNamespace(num_gaussians=100, keyframe_stride=1,
                             da3=SimpleNamespace(sync_tolerance_fraction=.25)), train=SimpleNamespace(seed=4))
    monkeypatch.setattr(visual_hull, "load_config", lambda _: config)
    monkeypatch.setattr(visual_hull.Scene, "from_config", lambda _: scene)
    def forbid(*args, **kwargs):
        raise AssertionError("closed calibrated rig must not infer DA3 depth")
    monkeypatch.setattr("ftgspp.optional.da3.generate_da3_points", forbid)
    if retry_color:
        from ftgspp.optional.point_color import PointColorResult
        original = visual_hull.color_points_ibr
        def sparse_queries(points, views, **kwargs):
            result = original(points, views, **kwargs)
            if kwargs.get("query_indices") is not None:
                return PointColorResult(result.rgb, np.zeros_like(result.keep),
                                        result.source_count, result.disagreement)
            return result
        monkeypatch.setattr(visual_hull, "color_points_ibr", sparse_queries)
    out = tmp_path / "points"
    path = visual_hull.prepare_visual_hull_points(tmp_path / "config", out, None,
                                                 resolution=24, coarse_resolution=6, device="cpu")
    manifest = json.loads(path.read_text())
    assert manifest["schema"] == "ftgspp-visual-hull-ibr-v4"
    assert manifest["rows"][0]["bounds_source"]["kind"] == "silhouette_frusta_v1"
    assert manifest["preparation_plan"]["resolved_max_points"] == 400
    assert 0 < manifest["rows"][0]["colored_points"] <= 400
    assert manifest["rows"][0]["geometry_diagnostics"]["occlusion_reference_count"] > 400
    if retry_color:
        diagnostics = manifest["rows"][0]["geometry_diagnostics"]
        assert diagnostics["color_budget_fallback"]
        assert diagnostics["color_evaluations"] > diagnostics["occlusion_reference_count"]
    assert _valid_prepared_point_manifest(out, scene, [0], expected_method="carving")
    with (out / "f000000.ply").open("ab") as stream:
        stream.write(b"corruption")
    assert not _valid_prepared_point_manifest(out, scene, [0], expected_method="carving")


def test_prefetch_preserves_order_and_propagates_decode_failure():
    from ftgspp.optional.preparation_views import prefetch_rgb
    views = [SimpleNamespace(load_rgb=lambda i=i: np.array([i])) for i in range(7)]
    assert [int(rgb[0]) for _, _, rgb in prefetch_rgb(views)] == list(range(7))
    def fail():
        raise FileNotFoundError("missing frame")
    views[2].load_rgb = fail
    with pytest.raises(FileNotFoundError, match="missing frame"):
        list(prefetch_rgb(views))


def test_raw_enclosing_rig_only_estimates_first_keyframe_and_reloads(tmp_path, monkeypatch):
    import cv2
    from omegaconf import OmegaConf
    from ftgspp.config.compose import load_config
    from ftgspp.data.prepare_dataset import main as prepare_dataset
    from ftgspp.data.scene import Scene
    from ftgspp.optional import da3
    from ftgspp.training.initialize import initialize_explicit
    views, _ = box_rig(count=4)
    for camera, view in enumerate(views):
        image_root = tmp_path / 'images' / str(camera)
        mask_root = tmp_path / 'masks' / str(camera)
        image_root.mkdir(parents=True)
        mask_root.mkdir(parents=True)
        for frame in (0, 1):
            cv2.imwrite(str(image_root / f'{frame:06d}.png'), view.load_rgb())
            cv2.imwrite(str(mask_root / f'{frame:06d}.png'), (view.load_mask()*255).astype(np.uint8))
    prepare_dataset(['--images', str(tmp_path/'images'), '--fps', '30', '--pipeline', 'foreground',
                     '--mask-root', str(tmp_path/'masks')])
    path = tmp_path/'ftgspp.yaml'
    node = OmegaConf.load(path)
    node.init.num_gaussians = 100
    node.init.da3 = {'device': 'cpu', 'process_res': 128, 'max_points': 256, 'min_consistent_views': 1}
    OmegaConf.save(node, path)
    class Model:
        def inference(self, images, **kwargs):
            assert len(images) == 4
            return SimpleNamespace(depth=np.full((4,128,128), 4, np.float32),
                                   conf=np.ones((4,128,128), np.float32), sky=None,
                                   processed_images=np.stack(images),
                                   extrinsics=np.stack([v.w2c for v in views]),
                                   intrinsics=np.stack([v.intrinsic for v in views]))
    monkeypatch.setattr(da3, '_load_model', lambda *a, **k: (Model(), 'cpu'))
    original = da3.generate_da3_points
    requests = []
    def generate(config, scene, positions):
        requests.append(list(positions))
        return original(config, scene, positions)
    monkeypatch.setattr(da3, 'generate_da3_points', generate)
    manifest_path = visual_hull.prepare_visual_hull_points(
        path, tmp_path/'points_vhull_ibr_stride1', None,
        resolution=24, coarse_resolution=6, device='cpu',
    )
    manifest = json.loads(manifest_path.read_text())
    assert requests == [[0]]
    assert all(row['bounds_source']['kind'] == 'silhouette_frusta_v1' for row in manifest['rows'])
    def forbid(*a, **k):
        raise AssertionError('reload must not infer depth or cameras')
    monkeypatch.setattr(da3, '_load_model', forbid)
    config = load_config(path)
    reloaded = Scene.from_config(config)
    assert reloaded.calibration_available
    config.run.output_path.mkdir(parents=True)
    assert len(initialize_explicit(config, reloaded)) == 100


def test_real_view_reuses_mask_inside_rgba_decode_and_preserves_premultiplication(tmp_path, monkeypatch):
    import cv2
    from ftgspp.data.scene import View
    image = tmp_path / 'rgb.png'
    mask_path = tmp_path / 'mask.png'
    cv2.imwrite(str(image), np.full((8, 8, 3), 50, np.uint8))
    cv2.imwrite(str(mask_path), np.full((8, 8), 128, np.uint8))
    k = np.array([[8,0,4], [0,8,4], [0,0,1]], np.float32)
    view = View(path=image, frame=0, camera_id=0, timestamp=0,
                w2c=np.eye(4, dtype=np.float32), source_intrinsic=k, intrinsic=k,
                distortion=np.zeros(5), size=(8,8), mask_path=mask_path,
                mask_source='external', rgb_encoding='premultiplied')
    expected = view.load_rgb()
    count = 0
    original = View.load_mask
    def load_mask(self):
        nonlocal count
        count += 1
        return original(self)
    monkeypatch.setattr(View, 'load_mask', load_mask)
    cached = PreparationView(view)
    cached.load_mask()
    np.testing.assert_array_equal(cached.load_rgb(), expected)
    np.testing.assert_array_equal(cached.load_rgb(), expected)
    assert count == 1
