"""Conservative per-frame bounds for calibrated, enclosing foreground rigs."""

import numpy as np
from scipy.optimize import linprog


class SilhouetteBoundsUnavailable(ValueError):
    """The masks/rig cannot safely provide a finite foreground envelope."""


def silhouette_bounds(views, *, mask_threshold=0.5, mask_dilation_pixels=2,
                      support_fraction=0.9):
    """Bound mask-box frusta without depth inference or temporal subsampling.

    This path requires complete silhouettes in valid images. Split the rig into
    disjoint groups, intersect each group's frusta, and take the UNION of their
    bounds. With fewer bad silhouettes than groups, one group is unaffected.
    Every group must bound all three axes; otherwise use the depth fallback.
    Actual silhouettes, not boxes, still decide fine carving occupancy.
    """
    count = len(views)
    # The same ceil support rule as carving. A small epsilon avoids inventing
    # an extra tolerated dropout for decimal roundoff (e.g. 10 * 0.9).
    dropouts = count - int(np.ceil(count * float(support_fraction) - 1e-12))
    groups = dropouts + 1
    if count < 2 or count // groups < 2:
        raise SilhouetteBoundsUnavailable("insufficient views for support-aware bounds")
    constraints = []
    excluded = []
    for view in views:
        mask = view.load_mask() >= float(mask_threshold)
        validity = view.load_validity() >= 0.5
        if not bool(np.all(validity)):
            excluded.append({"camera_id": int(view.camera_id), "reason": "validity_holes"})
            continue
        rows = np.flatnonzero(np.any(mask, axis=1))
        columns = np.flatnonzero(np.any(mask, axis=0))
        if not len(rows) or not len(columns):
            excluded.append({"camera_id": int(view.camera_id), "reason": "empty_silhouette"})
            continue
        h, w = view.size
        guard = int(mask_dilation_pixels) + 2
        if (rows[0] < guard or columns[0] < guard
                or rows[-1] >= h - guard or columns[-1] >= w - guard):
            excluded.append({"camera_id": int(view.camera_id), "reason": "clipped_silhouette"})
            continue
        # Include complete pixel footprints, mask dilation and a small image
        # margin. No averaging or triangulation of silhouette centers is used.
        margin_x = guard + 0.02 * (columns[-1] - columns[0] + 1)
        margin_y = guard + 0.02 * (rows[-1] - rows[0] + 1)
        k = np.asarray(view.intrinsic, np.float64)
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise SilhouetteBoundsUnavailable("non-positive camera focal length")
        left = (columns[0] - margin_x - k[0, 2]) / k[0, 0]
        right = (columns[-1] + 1 + margin_x - k[0, 2]) / k[0, 0]
        top = (rows[0] - margin_y - k[1, 2]) / k[1, 1]
        bottom = (rows[-1] + 1 + margin_y - k[1, 2]) / k[1, 1]
        camera_planes = np.array([
            [-1, 0, left], [1, 0, -right],
            [0, -1, top], [0, 1, -bottom], [0, 0, -1],
        ], np.float64)
        w2c = np.asarray(view.w2c, np.float64)
        constraints.append((camera_planes @ w2c[:3, :3], -camera_planes @ w2c[:3, 3]))
    if len(constraints) // groups < 2:
        raise SilhouetteBoundsUnavailable("too few complete, valid silhouettes to bound the rig")
    boxes = []
    for group in range(groups):
        selected = constraints[group::groups]
        a = np.concatenate([value[0] for value in selected])
        b = np.concatenate([value[1] for value in selected])
        limits = np.empty((2, 3), np.float64)
        for axis in range(3):
            for side, sign in enumerate((1.0, -1.0)):
                objective = np.eye(3)[axis] * sign
                solved = linprog(objective, A_ub=a, b_ub=b,
                                 bounds=[(None, None)] * 3, method="highs")
                if not solved.success:
                    raise SilhouetteBoundsUnavailable(
                        "silhouette frusta are unbounded or inconsistent"
                    )
                limits[side, axis] = solved.x[axis]
        boxes.append(limits)
    bounds = np.stack((np.min([box[0] for box in boxes], axis=0),
                       np.max([box[1] for box in boxes], axis=0)))
    extent = bounds[1] - bounds[0]
    if not np.isfinite(bounds).all() or np.any(extent <= 1e-8):
        raise SilhouetteBoundsUnavailable("degenerate silhouette envelope")
    centers = np.array([np.linalg.inv(view.w2c)[:3, 3] for view in views])
    rig_extent = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-8)
    if float(np.linalg.norm(extent)) > 4 * rig_extent:
        raise SilhouetteBoundsUnavailable("poorly constrained silhouette envelope")
    padding = np.maximum(extent * 0.05, np.max(extent) * 0.005)
    bounds += np.stack((-padding, padding))
    return bounds, {
        "kind": "silhouette_frusta_v1", "groups": groups,
        "tolerated_silhouette_dropouts": dropouts, "views": count,
        "bounds_views": len(constraints), "excluded_views": excluded,
        "padding_fraction": 0.05,
    }
