"""DCT-based frequency decomposition for AI-generated image artifact extraction.

Vendored from AIDE (data/submodules/AIDE/data/dct.py). Splits an image into
frequency-domain patches and selects the highest/lowest energy blocks for
downstream high-pass filtering and ResNet analysis.
"""

import numpy as np
import torch
import torch.nn as nn


def _dct_mat(size: int) -> list[list[float]]:
    """Build a Type-II DCT basis matrix of the given size."""
    return [
        [
            (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
            * np.cos((j + 0.5) * np.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]


def _generate_filter(start: float, end: float, size: int) -> list[list[float]]:
    """Binary band-pass mask: 1 where ``start <= i+j <= end``, else 0."""
    return [
        [0.0 if i + j > end or i + j < start else 1.0 for j in range(size)]
        for i in range(size)
    ]


def _norm_sigma(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sigmoid(x) - 1.0


class _Filter(nn.Module):
    """Learnable (or fixed) triangular band-pass filter applied in DCT domain."""

    def __init__(
        self,
        size: int,
        band_start: float,
        band_end: float,
        use_learnable: bool = False,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_learnable = use_learnable
        self.norm = norm

        self.base = nn.Parameter(
            torch.tensor(_generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0.0, 0.1)
        if self.norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(_generate_filter(band_start, band_end, size))),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt = (
            self.base + _norm_sigma(self.learnable) if self.use_learnable else self.base
        )
        return x * filt / self.ft_num if self.norm else x * filt


class DCTBaseRecModule(nn.Module):
    """Decompose a single image tensor into four DCT-reconstructed views.

    Given ``[C, H, W]``, the module extracts overlapping patches, transforms
    them to the DCT domain, applies a full-band filter, then selects the
    *two highest-energy* and *two lowest-energy* patches. Each selected patch
    is inverse-transformed and folded back to produce a spatial map of shape
    ``[level_N * C, win, win]``.

    Returns:
        Tuple of four tensors ``(x_minmin, x_maxmax, x_minmin1, x_maxmax1)``.
    """

    def __init__(
        self,
        window_size: int = 32,
        stride: int = 16,
        output: int = 256,
        grade_n: int = 6,
        level_filter: list[int] | None = None,
    ) -> None:
        super().__init__()
        if level_filter is None:
            level_filter = [0]

        assert output % window_size == 0
        assert len(level_filter) > 0

        self.window_size = window_size
        self.grade_N = grade_n
        self.level_N = len(level_filter)
        self.N = (output // window_size) * (output // window_size)

        self._DCT_patch = nn.Parameter(
            torch.tensor(_dct_mat(window_size)).float(), requires_grad=False
        )
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(_dct_mat(window_size)).float(), 0, 1),
            requires_grad=False,
        )

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
        self.fold = nn.Fold(
            output_size=(window_size, window_size),
            kernel_size=(window_size, window_size),
            stride=window_size,
        )

        level_f = [_Filter(window_size, 0, window_size * 2)]
        self.level_filters = nn.ModuleList([level_f[i] for i in level_filter])
        self.grade_filters = nn.ModuleList(
            [
                _Filter(
                    window_size,
                    window_size * 2.0 / grade_n * i,
                    window_size * 2.0 / grade_n * (i + 1),
                    norm=True,
                )
                for i in range(grade_n)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, _W, _H = x.shape

        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)
        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        y_list = []
        for i in range(level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)

        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for idx in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[idx](_x)
            _x = torch.sum(_x, dim=[1, 2, 3])
            grade += w * _x
            w *= k

        _, idx_sorted = torch.sort(grade)
        max_idx = torch.flip(idx_sorted, dims=[0])[:N]
        min_idx = idx_sorted[:N]

        maxmax_idx = max_idx[0]
        maxmax_idx1 = max_idx[1] if len(max_idx) > 1 else max_idx[0]
        minmin_idx = min_idx[0]
        minmin_idx1 = min_idx[1] if len(min_idx) > 1 else min_idx[0]

        def _fold_select(sel_idx: int) -> torch.Tensor:
            patch = torch.index_select(level_x_unfold, 0, sel_idx)
            return self.fold(
                patch.reshape(1, level_N * C * window_size * window_size).transpose(
                    0, 1
                )
            )

        return (
            _fold_select(minmin_idx),
            _fold_select(maxmax_idx),
            _fold_select(minmin_idx1),
            _fold_select(maxmax_idx1),
        )
