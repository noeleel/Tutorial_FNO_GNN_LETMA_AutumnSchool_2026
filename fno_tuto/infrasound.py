"""Educational c_eff(z) -> near-ground infrasound waveforms (relative pressure).

A transparent split-step FOURIER narrow-angle parabolic-equation (PE)
illustration for a range-independent effective sound speed, constant density,
reflecting ground and absorbing upper boundary. It is NOT a validated full
moving-atmosphere, wide-angle, 3-D, or regional infrasound solver.

The domain is a single ATMOSPHERIC BOUNDARY LAYER: altitude z in [0, 2 km],
receivers on the ground between 2 and 11 km. There is no stratosphere here:
nothing reflects back from above (the sponge absorbs it), so every recorded
arrival is a ground echo of the boundary-layer refraction.

Governing approximation, for positive angular frequency omega:
    (2 i k_ref) d_r u + d_zz u + (k(z)^2-k_ref^2) u = 0,
    p_hat(r,z,omega) = exp(i k_ref r) u(r,z,omega) S(omega),
    k(z) = omega/c_eff(z), k_ref=omega/c_ref.
z=0 uses a mirrored extension of u (Neumann reflecting ground); a sponge
above 1.4 km suppresses artificial upper-boundary reflections.

Returns waveforms on a RETARDED time axis tau=t-r/c_ref, meaning that the
phase exp(i omega r/c_ref) is deliberately removed from frequency response.
A fixed source and fixed receiver layout are used for operator learning.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from fno_tuto.models.fno1d import FNO1d


@dataclass(frozen=True)
class Configuration:
    n_alt: int = 64
    z_max_m: float = 2000.0            # the boundary layer only
    c_ref_mps: float = 330.0
    receiver_alt_m: float = 0.0        # receivers on the ground
    source_alt_m: float = 20.0         # a near-ground source (H6)
    source_vertical_width_m: float = 60.0
    station_ranges_km: tuple = (2.0, 5.0, 8.0, 11.0)
    dr_m: float = 50.0
    n_time: int = 256
    dt_s: float = 0.0625            # 16 s window, 8 Hz Nyquist: the [0.3, 4] Hz band fits in 64 modes
    source_center_tau_s: float = 3.0
    source_f0_hz: float = 1.5       # simple Ricker wavelet (H4, H6)
    f_min_hz: float = 0.3
    f_max_hz: float = 4.0


def sample_ceff(n: int, config: Configuration, seed: int = 42):
    """Diverse effective celerities, NOT observed atmospheric profiles.

    c_eff is used only for in-plane one-way propagation inside the boundary
    layer (0-2 km). Each profile combines: a random surface value, a
    large-scale vertical trend (inversions bend rays down, lapse bends them
    up), a low-level bend, random Gaussian perturbations concentrated in the
    boundary layer (2-5 bumps of amplitude ~N(0, 3.5 m/s), widths 0.1-0.5 km)
    representing shear layers and thermals, and a small short-correlation
    jitter (~0.5 m/s, ~100 m scale) representing frozen turbulence.
    """
    rng = np.random.default_rng(seed)
    z_m = np.linspace(0.0, config.z_max_m, config.n_alt)
    z_km = z_m / 1000.0
    profiles = []
    for _ in range(n):
        surface = rng.uniform(325.0, 343.0)
        trend = rng.uniform(-8.0, 50.0) * (z_km / 2.0)      # c(2 km) - c(0): lapse or inversion
        bend = rng.uniform(-12.0, 30.0) * np.exp(-((z_km - 0.8) / 0.5)**2)   # low-level jet
        bl = np.zeros_like(z_km)
        for _ in range(int(rng.integers(2, 6))):             # 2-5 Gaussian bumps
            bl = bl + rng.normal(0.0, 4.0) * np.exp(
                -((z_km - rng.uniform(0.0, 2.0)) / rng.uniform(0.1, 0.5))**2)
        jitter = rng.normal(0.0, 1.0, z_km.shape)            # smoothed white noise
        kernel = np.exp(-0.5 * (np.arange(-3, 4) / 1.2)**2)    # ~100 m correlation
        kernel /= kernel.sum()
        bl = bl + 0.5 * np.convolve(jitter, kernel, mode='same')
        profiles.append(surface + trend + bend + bl)
    return z_m.astype(np.float32), np.asarray(profiles, dtype=np.float32)


def ricker_source(config: Configuration):
    """Ricker wavelet on the retarded time axis; fixed source for the dataset."""
    tau = np.arange(config.n_time) * config.dt_s
    arg = np.pi * config.source_f0_hz * (tau - config.source_center_tau_s)
    s = (1.0 - 2.0 * arg*arg) * np.exp(-arg*arg)
    return tau.astype(np.float32), s.astype(np.float64)


def parabolic_waveforms(profile_mps, z_m, config: Configuration):
    """One atmospheric profile -> (stations, retarded-time) RELATIVE pressure.

    This uses a split-step narrow-angle PE and finite one-sided frequency
    sampling. The reduced equation ignores atmospheric density stratification,
    absorption in physical units, 3-D spreading and reflections from above.
    High-frequency/steep-angle accuracy is NOT guaranteed.
    """
    c = np.asarray(profile_mps, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if c.shape != z.shape or len(z) < 8 or np.any(c <= 0):
        raise ValueError('Profile must be positive and sampled on z grid')
    ranges_m = np.asarray(config.station_ranges_km) * 1000.0
    if np.max(np.abs(ranges_m/config.dr_m - np.rint(ranges_m/config.dr_m))) > 1e-8:
        raise ValueError('All station ranges must be multiples of dr_m')
    n_steps = int(round(ranges_m.max() / config.dr_m))
    sample_steps = {int(round(r/config.dr_m)): i for i, r in enumerate(ranges_m)}
    # Mirrored vertical grid: u'(0)=0; the top of the boundary layer is also
    # reflecting but strongly damped by the sponge layer: nothing comes back
    # from above 2 km (this is a boundary layer, not the stratosphere).
    z_ext = np.concatenate((z, -z[-2:0:-1]))
    c_ext = np.concatenate((c, c[-2:0:-1]))
    n_ext = z_ext.size
    kz = 2*np.pi*np.fft.fftfreq(n_ext, d=float(z[1]-z[0]))
    freq = np.fft.rfftfreq(config.n_time, d=config.dt_s)
    active_ids = np.flatnonzero((freq >= config.f_min_hz) & (freq <= config.f_max_hz))
    active_freq = freq[active_ids]
    omega = 2*np.pi*active_freq
    k0 = omega/config.c_ref_mps
    k = omega[:, None] / c_ext[None, :]
    # Strang splitting: half vertical diffraction, full refractive phase,
    # half diffraction. Numerically convenient, but ONLY a paraxial solver.
    diffraction_half = np.exp(-0.25j * config.dr_m * kz[None, :]**2 / k0[:, None])
    refraction_full = np.exp(0.5j * config.dr_m * (k*k - k0[:, None]**2) / k0[:, None])
    # soft sponge starts at 1.4 km, stronger towards 2 km (nonphysical absorber)
    sponge = np.exp(-0.6 * np.maximum(np.abs(z_ext)/1000.0 - 1.4, 0.0)**3)
    refraction_full *= sponge[None, :]
    source_z = np.exp(-0.5*((np.abs(z_ext) - config.source_alt_m)/config.source_vertical_width_m)**2)
    source_z /= source_z.max()
    u = np.broadcast_to(source_z[None, :], (len(active_ids), n_ext)).astype(complex).copy()
    # Source injected in the initial vertical field, near ground.
    pressure_spectrum = np.zeros((len(ranges_m), len(freq)), dtype=complex)
    z_rec_id = int(np.argmin(np.abs(z - config.receiver_alt_m)))
    for step in range(1, n_steps+1):
        u = np.fft.ifft(np.fft.fft(u, axis=1) * diffraction_half, axis=1)
        u *= refraction_full
        u = np.fft.ifft(np.fft.fft(u, axis=1) * diffraction_half, axis=1)
        if step in sample_steps:
            station = sample_steps[step]
            # PE uses exp(-i omega t) with outgoing exp(+i k r), whereas
            # NumPy irfft reconstructs with exp(+i omega t). Conjugate the
            # complex envelope to obtain the correct temporal sign.
            pressure_spectrum[station, active_ids] = np.conj(u[:, z_rec_id])
    tau, src = ricker_source(config)
    src_fft = np.fft.rfft(src)
    pressure_spectrum *= src_fft[None, :]
    # No carrier phase exp(i*k0*r): waveforms use retarded tau=t-r/c_ref.
    traces = np.fft.irfft(pressure_spectrum, n=config.n_time, axis=-1)
    # Paraxial 2-D scheme has relative amplitude only: normalize no individual traces.
    return np.asarray(traces, dtype=np.float32)


def make_dataset(n: int = 60, seed: int = 123, config: Configuration | None = None):
    config = Configuration() if config is None else config
    z, ceff = sample_ceff(n, config, seed=seed)
    p = np.stack([parabolic_waveforms(prof, z, config) for prof in ceff], axis=0)
    tau, source = ricker_source(config)
    return dict(z_m=z, ceff_mps=ceff, p=p, tau_s=tau,
                ranges_km=np.asarray(config.station_ranges_km, dtype=np.float32),
                source=source.astype(np.float32))


class ProfileToWaveformsFNO(nn.Module):
    """FNO altitude encoder plus fixed-receiver/time pressure readout.

    Input: (B,Nz) effective celerity, already standardized.
    Output: (B,Nreceiver,Ntau) pressure, in a global normalized scale.
    FIXED receiver locations and fixed time axis, not arbitrary query stations.
    The spectral blocks act on altitude, not on physical acoustic frequency.
    """
    def __init__(self, n_receivers=4, n_time=256, width=16, modes=12, blocks=2):
        super().__init__()
        self.encoder = FNO1d(in_channels=1, out_channels=12,
                             width=width, modes=modes, n_layers=blocks,
                             proj_hidden=24, grid=True)
        self.decoder = nn.Sequential(
            nn.Linear(12*3, 64), nn.GELU(),
            nn.Linear(64, n_receivers*n_time),
        )
        self.n_receivers, self.n_time = n_receivers, n_time

    def forward(self, ceff_z):
        x = self.encoder(ceff_z[:, None, :])
        n = x.shape[-1]
        h = torch.cat((x[..., :n//3].mean(-1),
                       x[..., n//3:2*n//3].mean(-1),
                       x[..., 2*n//3:].mean(-1)), dim=-1)
        return self.decoder(h).reshape(-1, self.n_receivers, self.n_time)
