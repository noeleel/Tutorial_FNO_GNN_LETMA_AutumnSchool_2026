"""FNO and FFNO in ANY dimension (1D, 2D, 3D, ...) -- SELF-CONTAINED file.

Generic version, mostly useful in 3D and beyond (appendix A4). For 1D and 2D, the
explicit files fno1d.py / fno2d.py / ffno.py are easier to read; the tests
(tests/test_models.py) check that these generic versions reproduce the explicit ones
exactly.

Direct use
----------
    from fno_tuto.models.fno_nd import FNO, FFNO

    fno3d  = FNO(modes=(8, 8, 8), in_channels=1, out_channels=1, width=16)
    ffno3d = FFNO(modes=(16, 16, 16), width=32, n_layers=8)
    y = fno3d(x)        # x: (B, C_in, N1, N2, N3)

    python -m fno_tuto.models.fno_nd       # quick test

Number of parameters per spectral layer
---------------------------------------
    full FNO : 2^(d-1) * prod(M_i) * C^2 complex    (2^(d-1) "corners" of the rfftn)
    FFNO     :      sum(M_i)       * C^2 complex
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConvNd(nn.Module):
    """Spectral convolution in dimension d = len(modes).

    Input (B, C_in, N_1, ..., N_d) -> Output (B, C_out, N_1, ..., N_d).
    The first d-1 axes get a full FFT (positive AND negative frequencies), the last one
    an rfft (positive only): there are 2^(d-1) corners, hence 2^(d-1) weight tensors.
    """

    def __init__(self, in_channels, out_channels, modes, fft_norm="ortho"):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes = tuple(modes)
        self.d = len(self.modes)
        self.fft_norm = fft_norm
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_channels, out_channels, *self.modes,
                                             dtype=torch.cfloat))
            for _ in range(2 ** (self.d - 1))
        ])
        letters = "xyzwv"[: self.d]
        self.eq = f"bi{letters},io{letters}->bo{letters}"

    def _corners(self, kshape):
        """For each corner: (slices in the spectrum, slices in the weights)."""
        per_axis = []
        for i in range(self.d - 1):                        # axes with a full FFT
            m = min(self.modes[i], kshape[i] // 2)
            per_axis.append([
                (slice(0, m), slice(0, m)),                                      # k >= 0
                (slice(kshape[i] - m, kshape[i]), slice(self.modes[i] - m, self.modes[i])),  # k < 0
            ])
        m = min(self.modes[-1], kshape[-1])                # last axis: rfft
        per_axis.append([(slice(0, m), slice(0, m))])
        for combo in itertools.product(*per_axis):
            yield tuple(c[0] for c in combo), tuple(c[1] for c in combo)

    def forward(self, x):
        dims = tuple(range(-self.d, 0))
        spatial = x.shape[-self.d:]
        x_hat = torch.fft.rfftn(x, dim=dims, norm=self.fft_norm)
        kshape = x_hat.shape[-self.d:]
        out = torch.zeros(x.shape[0], self.out_channels, *kshape,
                          dtype=x_hat.dtype, device=x.device)
        head = (slice(None), slice(None))
        for w, (ds, ws) in zip(self.weights, self._corners(kshape)):
            out[head + ds] = torch.einsum(self.eq, x_hat[head + ds], w[head + ws])
        return torch.fft.irfftn(out, s=spatial, dim=dims, norm=self.fft_norm)

    def extra_repr(self):
        return f"{self.in_channels}->{self.out_channels}, modes={self.modes}, corners={len(self.weights)}"


class FactorizedSpectralConvNd(nn.Module):
    """Factorized spectral convolution: one 1D convolution per axis, then sum."""

    def __init__(self, channels, modes, fft_norm="ortho"):
        super().__init__()
        self.channels = channels
        self.modes = tuple(modes)
        self.d = len(self.modes)
        self.fft_norm = fft_norm
        scale = 1.0 / (channels * channels)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(channels, channels, m, dtype=torch.cfloat))
            for m in self.modes
        ])

    def forward(self, x):
        out = torch.zeros_like(x)
        for i, w in enumerate(self.weights):
            dim = -self.d + i
            n = x.shape[dim]
            # move the processed axis to the last position to keep a simple einsum
            xh = torch.fft.rfft(x, dim=dim, norm=self.fft_norm).movedim(dim, -1)
            shp = xh.shape
            flat = xh.reshape(shp[0], shp[1], -1, shp[-1])           # (B, C, rest, K)
            m = min(self.modes[i], shp[-1])
            yh = torch.zeros_like(flat)
            yh[..., :m] = torch.einsum("birk,iok->bork", flat[..., :m], w[..., :m])
            yh = yh.reshape(shp).movedim(-1, dim)
            out = out + torch.fft.irfft(yh, n=n, dim=dim, norm=self.fft_norm)
        return out

    def extra_repr(self):
        return f"channels={self.channels}, modes={self.modes} (factorized, {self.d} axes)"


class PointwiseNd(nn.Module):
    """1x1 convolution in any dimension (space is flattened)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        b, c = x.shape[:2]
        return self.lin(x.reshape(b, c, -1)).reshape(b, -1, *x.shape[2:])


class FNOBlockNd(nn.Module):
    def __init__(self, width, modes, use_norm=False):
        super().__init__()
        self.spectral = SpectralConvNd(width, width, modes)
        self.pointwise = PointwiseNd(width, width)
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.pointwise(x)))


class FFNOBlockNd(nn.Module):
    def __init__(self, width, modes, mlp_ratio=2, use_norm=True, spectral=None):
        super().__init__()
        self.spectral = spectral if spectral is not None else FactorizedSpectralConvNd(width, modes)
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()
        hidden = width * mlp_ratio
        self.ff = nn.Sequential(PointwiseNd(width, hidden), nn.GELU(), PointwiseNd(hidden, width))

    def forward(self, x):
        return x + self.ff(F.gelu(self.spectral(self.norm(x))))


class _OperatorNd(nn.Module):
    """Shared skeleton: grid -> lift -> padding -> blocks -> unpad -> projection."""

    def _init_common(self, modes, in_channels, out_channels, width, padding, grid, proj_hidden):
        self.modes = tuple(modes)
        self.d = len(self.modes)
        self.padding = padding
        self.grid = grid
        self.lift = PointwiseNd(in_channels + (self.d if grid else 0), width)
        self.proj = nn.Sequential(PointwiseNd(width, proj_hidden), nn.GELU(),
                                  PointwiseNd(proj_hidden, out_channels))

    def make_grid(self, x):
        axes = [torch.linspace(0, 1, n, device=x.device, dtype=x.dtype) for n in x.shape[2:]]
        g = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0).unsqueeze(0)
        return g.expand(x.shape[0], *g.shape[1:])

    def forward(self, x):
        if self.grid:
            x = torch.cat([x, self.make_grid(x)], dim=1)
        x = self.lift(x)
        p = self.padding
        if p > 0:
            # F.pad supports "replicate" only up to 3D
            x = F.pad(x, [0, p] * self.d, mode="replicate" if self.d <= 3 else "constant")
        for block in self.blocks:
            x = block(x)
        if p > 0:
            x = x[(Ellipsis,) + (slice(None, -p),) * self.d]
        return self.proj(x)


class FNO(_OperatorNd):
    """FNO in dimension d = len(modes)."""

    def __init__(self, modes, in_channels=1, out_channels=1, width=32, n_layers=4,
                 padding=0, grid=True, proj_hidden=128, use_norm=False):
        super().__init__()
        self.config = dict(modes=tuple(modes), in_channels=in_channels, out_channels=out_channels,
                           width=width, n_layers=n_layers, padding=padding, grid=grid,
                           proj_hidden=proj_hidden, use_norm=use_norm)
        self._init_common(modes, in_channels, out_channels, width, padding, grid, proj_hidden)
        self.blocks = nn.ModuleList(
            [FNOBlockNd(width, self.modes, use_norm=use_norm) for _ in range(n_layers)])


class FFNO(_OperatorNd):
    """FFNO in dimension d = len(modes), with optional sharing of the spectral weights."""

    def __init__(self, modes, in_channels=1, out_channels=1, width=32, n_layers=8,
                 padding=0, grid=True, proj_hidden=128, mlp_ratio=2, share_weights=False,
                 use_norm=True):
        super().__init__()
        self.config = dict(modes=tuple(modes), in_channels=in_channels, out_channels=out_channels,
                           width=width, n_layers=n_layers, padding=padding, grid=grid,
                           proj_hidden=proj_hidden, mlp_ratio=mlp_ratio,
                           share_weights=share_weights, use_norm=use_norm)
        self._init_common(modes, in_channels, out_channels, width, padding, grid, proj_hidden)
        shared = FactorizedSpectralConvNd(width, self.modes) if share_weights else None
        self.blocks = nn.ModuleList([
            FFNOBlockNd(width, self.modes, mlp_ratio=mlp_ratio, use_norm=use_norm, spectral=shared)
            for _ in range(n_layers)
        ])


if __name__ == "__main__":
    def n_real(m):
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())

    for modes, shp in (((16,), (2, 1, 64)), ((12, 12), (2, 1, 32, 32)),
                       ((8, 8, 8), (2, 1, 16, 16, 16))):
        f, g = FFNO(modes, width=24, n_layers=4), FNO(modes, width=24, n_layers=4)
        y = f(torch.randn(*shp))
        print(f"d={len(modes)}: output {tuple(y.shape)} | FNO {n_real(g):10,d} params | "
              f"FFNO {n_real(f):8,d} params")
