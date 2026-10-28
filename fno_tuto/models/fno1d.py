"""1D FNO -- SELF-CONTAINED file (depends on torch only).

You can copy this file alone into another project: it imports nothing from the rest
of the package.

Direct use
----------
    from fno_tuto.models.fno1d import FNO1d

    model = FNO1d(in_channels=1, out_channels=1, width=32, modes=16, n_layers=4)
    y = model(x)        # x: (B, C_in, N)  ->  y: (B, C_out, N), any N

Quick test from the command line
--------------------------------
    python -m fno_tuto.models.fno1d

Architecture (Li et al., ICLR 2021)
-----------------------------------
    a(x) --P--> v0 --[block 1]--> v1 --> ... --[block L]--> vL --Q--> u(x)

    P     : "lifting", 1x1 convolution: C_in (+1 coordinate channel) -> width
    block : v_{l+1} = GELU( K_l v_l  +  W_l v_l )
            K_l : SPECTRAL convolution -- global, only sees the first `modes`
                  frequencies (SpectralConv1d below)
            W_l : 1x1 convolution -- local, acts on all frequencies
    Q     : "projection", small pointwise MLP: width -> proj_hidden -> C_out

Shape convention: channels-first, (B, C, N). The spatial axis is the LAST one, because
torch.fft acts on the last axis by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D spectral convolution: the first `modes` Fourier coefficients are multiplied by
    a learned complex matrix (one C_in x C_out matrix PER MODE).

        (K v)^[k] = R[k] . v^[k]   for k < modes,   0 beyond

    Input  (B, C_in,  N)   ->   Output (B, C_out, N)
    Parameters: C_in * C_out * modes complex numbers (= 2x as many reals).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int,
                 fft_norm: str = "ortho"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.fft_norm = fft_norm
        # Deliberately small initialization: the layer starts almost at zero and the
        # local path W carries the signal at first (see appendix A2).
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        # 1) to Fourier. rfft (not fft): a real signal has a Hermitian spectrum, the
        #    negative frequencies are redundant. We get n//2 + 1 modes.
        x_hat = torch.fft.rfft(x, norm=self.fft_norm)               # (B, C_in, n//2+1)
        # 2) truncation. min(): at low resolution there are fewer modes than weights --
        #    this is what allows evaluating the model at any resolution.
        m = min(self.modes, x_hat.shape[-1])
        # 3) INDEPENDENT matrix product for each mode k (no mixing between k).
        #    b = batch, i = input channel, o = output channel, k = mode
        out_hat = torch.zeros(x.shape[0], self.out_channels, x_hat.shape[-1],
                              dtype=x_hat.dtype, device=x.device)
        out_hat[..., :m] = torch.einsum("bik,iok->bok", x_hat[..., :m], self.weight[..., :m])
        # 4) back to physical space. n=n is mandatory: otherwise irfft assumes an even
        #    size and is off by one point when n is odd.
        return torch.fft.irfft(out_hat, n=n, norm=self.fft_norm)

    def extra_repr(self):
        return f"{self.in_channels}->{self.out_channels}, modes={self.modes}"


class FNOBlock1d(nn.Module):
    """One FNO block: v -> GELU( K v + W v ).

    The two paths play complementary roles:
      * spectral (K) : non-local, carries information from one end of the domain to the
                       other, but produces NOTHING beyond `modes`;
      * pointwise (W): local (1x1), lets all frequencies through.
    The GELU nonlinearity is what prevents L blocks from collapsing into a single filter.
    """

    def __init__(self, width: int, modes: int, use_norm: bool = False):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        # No normalization by default: it erases the amplitude of each sample
        # (see appendix A2, "internal normalization").
        self.norm = nn.GroupNorm(1, width) if use_norm else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.spectral(x) + self.pointwise(x)))


class FNO1d(nn.Module):
    """Complete 1D FNO.

    Parameters
    ----------
    in_channels  : number of physical input channels (without the coordinate)
    out_channels : number of output channels
    width        : number of hidden channels d_v (32 is a good start)
    modes        : number of Fourier modes kept per layer (M)
    n_layers     : number of FNO blocks (4 is standard)
    padding      : points appended on the right before the spectral layers, removed at
                   the end. 0 if the problem is periodic, ~5-15 % of N otherwise.
    grid         : adds the coordinate x as an input channel (breaks translation
                   invariance -- almost always useful)
    proj_hidden  : hidden width of the projection MLP Q
    use_norm     : GroupNorm in each block (not recommended by default)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32,
                 modes: int = 16, n_layers: int = 4, padding: int = 0, grid: bool = True,
                 proj_hidden: int = 128, use_norm: bool = False):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, width=width,
                           modes=modes, n_layers=n_layers, padding=padding, grid=grid,
                           proj_hidden=proj_hidden, use_norm=use_norm)
        self.padding = padding
        self.grid = grid
        self.lift = nn.Conv1d(in_channels + (1 if grid else 0), width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock1d(width, modes, use_norm=use_norm) for _ in range(n_layers)])
        self.proj = nn.Sequential(
            nn.Conv1d(width, proj_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(proj_hidden, out_channels, kernel_size=1),
        )

    @staticmethod
    def make_grid(x):
        """Coordinate x in [0, 1], same shape as x on a single channel."""
        n = x.shape[-1]
        g = torch.linspace(0, 1, n, device=x.device, dtype=x.dtype)
        return g.view(1, 1, n).expand(x.shape[0], 1, n)

    def forward(self, x):
        if self.grid:
            x = torch.cat([x, self.make_grid(x)], dim=1)
        x = self.lift(x)                                        # (B, width, N)
        if self.padding > 0:
            # "replicate" rather than zeros: avoids creating a step at the junction
            x = F.pad(x, (0, self.padding), mode="replicate")
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., : -self.padding]
        return self.proj(x)                                     # (B, C_out, N)


if __name__ == "__main__":
    model = FNO1d(in_channels=1, out_channels=1, width=32, modes=16, n_layers=4)
    n_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    print(model)
    print(f"\n{n_real:,} real parameters (a complex one counts as 2)")
    for n in (64, 128, 777):
        y = model(torch.randn(2, 1, n))
        print(f"input (2, 1, {n:3d}) -> output {tuple(y.shape)}   <- same weights")
