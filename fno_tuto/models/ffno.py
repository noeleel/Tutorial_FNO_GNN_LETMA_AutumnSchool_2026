"""2D FFNO (Factorized Fourier Neural Operator) -- SELF-CONTAINED file (torch only).

Tran, Mathews, Xie, Ong, "Factorized Fourier Neural Operators", ICLR 2023.

Direct use
----------
    from fno_tuto.models.ffno import FFNO2d

    model = FFNO2d(in_channels=1, out_channels=1, width=32, modes=(16, 16), n_layers=8)
    y = model(x)        # x: (B, C_in, H, W)  ->  y: (B, C_out, H, W)

Quick test:
    python -m fno_tuto.models.ffno

The three differences from the 2D FNO
-------------------------------------
1. FACTORIZED spectral convolution: instead of one weight R[k_x, k_y] for each pair of
   modes (mx * my matrices), we apply one 1D spectral convolution along x, another one
   along y, and sum them:

       K v = IFFT_x( R_x[k_x] . FFT_x v )  +  IFFT_y( R_y[k_y] . FFT_y v )

   Parameters: (mx + my) * C^2 instead of 2 * mx * my * C^2.
   The price: a factorized filter cannot couple k_x and k_y (no "diagonal" filter).
   Depth and nonlinearities partly make up for it.

2. Transformer-style RESIDUAL block:  v -> v + FF( GELU( K( Norm(v) ) ) )
   with FF a small pointwise MLP. The identity path makes 8 to 24 layers trainable.

3. share_weights option: all blocks share the SAME spectral convolution; the number
   of spectral parameters no longer depends on the depth.

In 1D factorization brings nothing: use FNO1d (or FFNO from fno_nd.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(m):
    return (m, m) if isinstance(m, int) else tuple(m)


class FactorizedSpectralConv2d(nn.Module):
    """Sum of two 1D spectral convolutions: one along x, one along y.

    Input/Output (B, C, H, W). Parameters: C * C * (mx + my) complex numbers.
    """

    def __init__(self, channels: int, modes=(16, 16), fft_norm: str = "ortho"):
        super().__init__()
        self.channels = channels
        self.modes = _pair(modes)
        self.fft_norm = fft_norm
        scale = 1.0 / (channels * channels)
        mx, my = self.modes
        self.weight_x = nn.Parameter(scale * torch.randn(channels, channels, mx, dtype=torch.cfloat))
        self.weight_y = nn.Parameter(scale * torch.randn(channels, channels, my, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        # --- x branch: rfft along the x axis (dim -2), each y column independent
        xh = torch.fft.rfft(x, dim=-2, norm=self.fft_norm)          # (B, C, H//2+1, W)
        mx = min(self.modes[0], xh.shape[-2])
        out_x = torch.zeros_like(xh)
        out_x[:, :, :mx, :] = torch.einsum("bixy,iox->boxy", xh[:, :, :mx, :],
                                           self.weight_x[..., :mx])
        out_x = torch.fft.irfft(out_x, n=H, dim=-2, norm=self.fft_norm)
        # --- y branch: rfft along the y axis (dim -1), each x row independent
        yh = torch.fft.rfft(x, dim=-1, norm=self.fft_norm)          # (B, C, H, W//2+1)
        my = min(self.modes[1], yh.shape[-1])
        out_y = torch.zeros_like(yh)
        out_y[..., :my] = torch.einsum("bixy,ioy->boxy", yh[..., :my],
                                       self.weight_y[..., :my])
        out_y = torch.fft.irfft(out_y, n=W, dim=-1, norm=self.fft_norm)
        return out_x + out_y

    def extra_repr(self):
        return f"channels={self.channels}, modes={self.modes} (factorized)"


class FFNOBlock2d(nn.Module):
    """Pre-norm residual block:  v + FF( GELU( K( Norm(v) ) ) ).

    `spectral` can be passed in from outside: this is how a single spectral
    convolution is shared between all blocks.
    """

    def __init__(self, width: int, modes=(16, 16), mlp_ratio: int = 2,
                 use_norm: bool = True, spectral: nn.Module = None):
        super().__init__()
        self.spectral = spectral if spectral is not None else FactorizedSpectralConv2d(width, modes)
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()
        hidden = width * mlp_ratio
        self.ff = nn.Sequential(
            nn.Conv2d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, width, kernel_size=1),
        )

    def forward(self, x):
        return x + self.ff(F.gelu(self.spectral(self.norm(x))))


class FFNO2d(nn.Module):
    """Complete 2D FFNO: lift -> L factorized residual blocks -> projection.

    Specific parameters (the others are those of FNO2d)
    ---------------------------------------------------
    mlp_ratio     : feedforward width = mlp_ratio * width
    share_weights : a single spectral convolution shared by all blocks
    use_norm      : GroupNorm(1, width) at the start of each block (useful beyond ~8 layers)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32,
                 modes=(16, 16), n_layers: int = 8, padding: int = 0, grid: bool = True,
                 proj_hidden: int = 128, mlp_ratio: int = 2, share_weights: bool = False,
                 use_norm: bool = True):
        super().__init__()
        modes = _pair(modes)
        self.config = dict(in_channels=in_channels, out_channels=out_channels, width=width,
                           modes=modes, n_layers=n_layers, padding=padding, grid=grid,
                           proj_hidden=proj_hidden, mlp_ratio=mlp_ratio,
                           share_weights=share_weights, use_norm=use_norm)
        self.padding = padding
        self.grid = grid
        self.lift = nn.Conv2d(in_channels + (2 if grid else 0), width, kernel_size=1)
        shared = FactorizedSpectralConv2d(width, modes) if share_weights else None
        self.blocks = nn.ModuleList([
            FFNOBlock2d(width, modes, mlp_ratio=mlp_ratio, use_norm=use_norm, spectral=shared)
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(width, proj_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(proj_hidden, out_channels, kernel_size=1),
        )

    @staticmethod
    def make_grid(x):
        B, _, H, W = x.shape
        gx = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        mx, my = torch.meshgrid(gx, gy, indexing="ij")
        return torch.stack([mx, my], 0).unsqueeze(0).expand(B, 2, H, W)

    def forward(self, x):
        if self.grid:
            x = torch.cat([x, self.make_grid(x)], dim=1)
        x = self.lift(x)
        p = self.padding
        if p > 0:
            x = F.pad(x, (0, p, 0, p), mode="replicate")
        for block in self.blocks:
            x = block(x)
        if p > 0:
            x = x[..., :-p, :-p]
        return self.proj(x)


if __name__ == "__main__":
    def n_real(m):
        # parameters() deduplicates shared weights: each tensor is counted once
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())

    for share in (False, True):
        model = FFNO2d(width=32, modes=(16, 16), n_layers=8, share_weights=share)
        y = model(torch.randn(2, 1, 64, 64))
        print(f"FFNO2d(L=8, w=32, M=16, share_weights={share!s:5s}): "
              f"{n_real(model):9,d} parameters, output {tuple(y.shape)}")
