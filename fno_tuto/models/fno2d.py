"""2D FNO -- SELF-CONTAINED file (depends on torch only).

Direct use
----------
    from fno_tuto.models.fno2d import FNO2d

    model = FNO2d(in_channels=1, out_channels=1, width=32, modes=(12, 12), n_layers=4,
                  padding=8)
    y = model(x)        # x: (B, C_in, H, W)  ->  y: (B, C_out, H, W)

Quick test:
    python -m fno_tuto.models.fno2d

Difference from 1D: the "corner" trap
-------------------------------------
rfft2 performs a FULL FFT along the x axis (second to last) and an rfft along the y axis
(last). The spectrum has shape (H, W//2 + 1), and along the x axis the indices are:

    index  :  0   1   2  ...  H/2  ...  H-2  H-1
    k_x    :  0  +1  +2  ... +-H/2 ...   -2   -1

The low k_x frequencies therefore sit at BOTH ENDS of the axis. Two weight blocks are
needed: one for k_x >= 0 (start of the axis), one for k_x < 0 (end of the axis).
Along the y axis (rfft), there are only k_y >= 0: one side is enough.

                 k_y ->  0 .. my-1
    k_x = 0      +-----------+
          ...    | weight_pos|    <- rows 0 .. mx-1 of the spectrum
          mx-1   +-----------+
                 |  (zeros)  |    <- high frequencies in x: cut
    k_x = -mx    +-----------+
          ...    | weight_neg|    <- rows H-mx .. H-1 of the spectrum
          -1     +-----------+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(m):
    return (m, m) if isinstance(m, int) else tuple(m)


class SpectralConv2d(nn.Module):
    """2D spectral convolution with two weight blocks (k_x >= 0 and k_x < 0).

    Input  (B, C_in,  H, W)   ->   Output (B, C_out, H, W)
    Parameters: 2 * C_in * C_out * mx * my complex numbers.
    """

    def __init__(self, in_channels: int, out_channels: int, modes=(12, 12),
                 fft_norm: str = "ortho"):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes = _pair(modes)
        self.fft_norm = fft_norm
        mx, my = self.modes
        scale = 1.0 / (in_channels * out_channels)
        # weight_pos: k_x in [0, mx)        weight_neg: k_x in [-mx, 0)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, mx, my, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, mx, my, dtype=torch.cfloat))

    def forward(self, x):
        B, _, H, W = x.shape
        x_hat = torch.fft.rfft2(x, norm=self.fft_norm)              # (B, C_in, H, W//2+1)
        # H // 2: the two corners must not overlap at low resolution
        mx = min(self.modes[0], H // 2)
        my = min(self.modes[1], W // 2 + 1)
        out = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                          dtype=x_hat.dtype, device=x.device)
        # corner k_x >= 0: first rows of the spectrum, first rows of the weights
        out[:, :, :mx, :my] = torch.einsum(
            "bixy,ioxy->boxy", x_hat[:, :, :mx, :my], self.weight_pos[:, :, :mx, :my])
        # corner k_x < 0: last rows of the spectrum. We take the LAST rows of the
        # weights (frequencies closest to 0) so that evaluating at a lower resolution
        # than training stays consistent.
        mx_full = self.modes[0]
        out[:, :, H - mx:, :my] = torch.einsum(
            "bixy,ioxy->boxy", x_hat[:, :, H - mx:, :my],
            self.weight_neg[:, :, mx_full - mx:, :my])
        return torch.fft.irfft2(out, s=(H, W), norm=self.fft_norm)

    def extra_repr(self):
        return f"{self.in_channels}->{self.out_channels}, modes={self.modes}"


class FNOBlock2d(nn.Module):
    """v -> GELU( K v + W v ), with K a 2D spectral convolution and W a 1x1 convolution."""

    def __init__(self, width: int, modes=(12, 12), use_norm: bool = False):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.pointwise(x)))


class FNO2d(nn.Module):
    """Complete 2D FNO. Same parameters as FNO1d, with `modes` = (mx, my) or an integer.

    Modes can be ANISOTROPIC: modes=(24, 6) if the field varies quickly along x and
    slowly along y. The cost is proportional to mx * my.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32,
                 modes=(12, 12), n_layers: int = 4, padding: int = 0, grid: bool = True,
                 proj_hidden: int = 128, use_norm: bool = False):
        super().__init__()
        modes = _pair(modes)
        self.config = dict(in_channels=in_channels, out_channels=out_channels, width=width,
                           modes=modes, n_layers=n_layers, padding=padding, grid=grid,
                           proj_hidden=proj_hidden, use_norm=use_norm)
        self.padding = padding
        self.grid = grid
        self.lift = nn.Conv2d(in_channels + (2 if grid else 0), width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes, use_norm=use_norm) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv2d(width, proj_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(proj_hidden, out_channels, kernel_size=1),
        )

    @staticmethod
    def make_grid(x):
        """Two coordinate channels (x, y) in [0, 1]^2."""
        B, _, H, W = x.shape
        gx = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        mx, my = torch.meshgrid(gx, gy, indexing="ij")
        return torch.stack([mx, my], 0).unsqueeze(0).expand(B, 2, H, W)

    def forward(self, x):
        if self.grid:
            x = torch.cat([x, self.make_grid(x)], dim=1)
        x = self.lift(x)                                         # (B, width, H, W)
        p = self.padding
        if p > 0:
            # F.pad: paddings from the LAST axis to the first -> (y_left, y_right, x_top, x_bottom)
            x = F.pad(x, (0, p, 0, p), mode="replicate")
        for block in self.blocks:
            x = block(x)
        if p > 0:
            x = x[..., :-p, :-p]
        return self.proj(x)                                      # (B, C_out, H, W)


if __name__ == "__main__":
    model = FNO2d(in_channels=1, out_channels=1, width=24, modes=(12, 12), n_layers=4,
                  padding=8)
    n_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    print(model)
    print(f"\n{n_real:,} real parameters")
    for hw in ((32, 32), (64, 64), (48, 80)):
        y = model(torch.randn(2, 1, *hw))
        print(f"input (2, 1, {hw[0]}, {hw[1]}) -> output {tuple(y.shape)}")
