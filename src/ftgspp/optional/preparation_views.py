"""Frame-scoped scalar-map reuse and bounded RGB decode prefetch."""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

import numpy as np

from ftgspp.data.scene import View


class PreparationView:
    """Share masks across bounds, coarse/fine carving and coloring.

    The caller releases these wrappers after each keyframe. RGB is streamed,
    never retained for the whole rig or sequence.
    """

    def __init__(self, view):
        self.view = view

    def __getattr__(self, name):
        return getattr(self.view, name)

    @cached_property
    def mask(self):
        if isinstance(self.view, View) and self.view.mask_source == "none":
            return np.broadcast_to(np.float32(1), self.view.size)
        return self.view.load_mask()

    @cached_property
    def validity(self):
        if isinstance(self.view, View) and self.view.validity_path is None:
            return np.broadcast_to(np.bool_(True), self.view.size)
        return self.view.load_validity() >= 0.5

    def load_mask(self):
        return self.mask

    def load_validity(self):
        return self.validity

    def load_rgb(self):
        if isinstance(self.view, View):
            # View.load_rgb also reads the mask to attach reference alpha and
            # undo premultiplication. Bind it to this proxy so that internal
            # read uses the same transformed mask as carving and IBR.
            return View.load_rgb(self)
        return self.view.load_rgb()


def prefetch_rgb(views):
    """Overlap two image decodes with projection, preserving view order.

    At most two future images and the current image are live. Exceptions are
    propagated on consumption; closing the generator joins its workers.
    """
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ftgspp-rgb") as pool:
        pending = {}
        for index in range(min(2, len(views))):
            pending[index] = pool.submit(views[index].load_rgb)
        try:
            for index, view in enumerate(views):
                rgb = pending.pop(index).result()
                if index + 2 < len(views):
                    pending[index + 2] = pool.submit(views[index + 2].load_rgb)
                yield index, view, rgb
        finally:
            for future in pending.values():
                future.cancel()
