# FreeTimeGS++ checkpoint compatibility model. Not used by native training.
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import torch
from torch import Tensor, nn


class ColorCorrector(nn.Module):
    # Affine model
    mat: nn.Parameter
    vec: nn.Parameter
    eye: nn.Buffer

    def __init__(self):
        super().__init__()
        self.mat = nn.Parameter(torch.zeros((3, 3)))
        self.vec = nn.Parameter(torch.zeros(3))
        self.eye = nn.Buffer(torch.eye(3))

    def forward(
        self,
        rgb: Tensor,  # (..., 3)
    ):
        return rgb @ (self.mat + self.eye) + self.vec


class ColorCorrectors(nn.Module):
    _correctors: nn.ModuleList

    def __init__(self, num_views: int):
        super().__init__()
        self._correctors = nn.ModuleList(ColorCorrector() for _ in range(num_views))

    def forward(
        self,
        view_idxs: list[int],
        rgb: Tensor,  # (..., 3)
    ) -> Tensor:
        rgb_corrected = torch.empty_like(rgb)

        for i, idx in enumerate(view_idxs):
            rgb_corrected[i] = self._correctors[idx](rgb[i])

        return rgb_corrected

    def regularize(self) -> Tensor:
        device = self._correctors[0].mat.device if len(self._correctors) else torch.device("cpu")
        sum = torch.zeros((), device=device)
        for corrector in self._correctors:
            sum += corrector.mat.square().sum() + corrector.vec.square().sum()  # type: ignore

        return sum

    def __len__(self) -> int:
        return len(self._correctors)
