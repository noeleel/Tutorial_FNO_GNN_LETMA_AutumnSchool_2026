"""Tutorial datasets -- all GENERATED locally, nothing to download.

    load_burgers(...)   viscous 1D Burgers, u(., 0) -> u(., T)        [notebooks 02, 03, 06]
    load_darcy(...)     2D Darcy, binary coefficient a -> solution u  [notebooks 04, 05, 06]
    diffusion_1d(...)   1D diffusion, EXACT in Fourier (known answer) [notebooks 01, 03]
    advdiff_3d(...)     exact 3D advection-diffusion                  [appendix A4]
    load_kh(...)        2D Kelvin-Helmholtz mixing layer,
                        initial perturbation -> (vorticity, scalar)   [notebook 07]

The expensive generations (Burgers, Darcy, Kelvin-Helmholtz) are cached in the `data/`
folder at the project root (or in $FNO_TUTO_DATA): the first call takes a few tens of
seconds (Kelvin-Helmholtz: ~15-20 min), the following ones are instantaneous.

All functions return float32 tensors in channels-first layout:
    1D: (n_samples, 1, N)        2D: (n_samples, 1, H, W)
"""
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

try:                                    # only needed by darcy_solve
    import scipy.sparse as sp_sparse
    import scipy.sparse.linalg as sp_linalg
except ImportError:                     # pragma: no cover
    sp_sparse = sp_linalg = None

DATA_DIR = Path(os.environ.get("FNO_TUTO_DATA", Path(__file__).resolve().parents[1] / "data"))


def _cached(name: str, params: dict, builder, cache: bool = True, verbose: bool = True):
    """Calls builder() or reloads the result from data/<name>_<hash>.pt."""
    key = hashlib.md5(repr(sorted(params.items())).encode()).hexdigest()[:10]
    path = DATA_DIR / f"{name}_{key}.pt"
    if cache and path.exists():
        if verbose:
            print(f"[data] {name} loaded from cache ({path.name})")
        return torch.load(path)
    t0 = time.time()
    out = builder()
    if verbose:
        print(f"[data] {name} generated in {time.time() - t0:.1f} s")
    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
    return out


# ============================================================================
#  1D: Gaussian random fields, Burgers, diffusion
# ============================================================================
def grf_1d(n_samples, n, alpha=3.0, tau=7.0, gen=None, dtype=torch.float64):
    """PERIODIC Gaussian random field on [0,1), unit variance, zero mean.

    Matern-type spectrum: amplitude ~ (4 pi^2 k^2 + tau^2)^(-alpha/2).
    Large alpha -> smooth field; large tau -> short correlation length.
    """
    k = torch.fft.rfftfreq(n, d=1.0 / n)
    amp = (4.0 * math.pi**2 * k**2 + tau**2) ** (-alpha / 2.0) * n
    var = (4.0 * (amp[1:-1] ** 2).sum() + amp[-1] ** 2) / n**2        # analytical variance
    amp = amp / var.sqrt()
    xi = torch.randn(n_samples, k.numel(), 2, generator=gen, dtype=dtype)
    coef = torch.complex(xi[..., 0], xi[..., 1]) * amp.to(dtype)
    coef[:, 0] = 0.0                                                   # zero mean
    return torch.fft.irfft(coef, n=n)


def solve_burgers(u0, nu=0.02, T=0.5, dt=2.5e-4):
    """Periodic Burgers  u_t + u u_x = nu u_xx  up to t = T.

    Pseudo-spectral, ETDRK2 exponential integrator (viscous part treated exactly) and
    2/3 de-aliasing of the nonlinear term. u0: (B, N) float64.
    """
    B, N = u0.shape
    kk = torch.fft.rfftfreq(N, d=1.0 / N).to(u0.dtype)
    k = 2.0 * math.pi * kk
    z = -nu * k**2 * dt
    E = torch.exp(z)
    small = z.abs() < 1e-12
    zs = torch.where(small, torch.ones_like(z), z)
    phi1 = torch.where(small, torch.ones_like(z), (E - 1.0) / zs)
    phi2 = torch.where(small, torch.full_like(z, 0.5), (E - 1.0 - z) / zs**2)
    mask = (kk <= (N // 2) * (2.0 / 3.0)).to(u0.dtype)

    def nonlinear(uh):
        u = torch.fft.irfft(uh, n=N)
        ux = torch.fft.irfft(1j * k * uh, n=N)
        return -torch.fft.rfft(u * ux) * mask

    uh = torch.fft.rfft(u0) * mask
    for _ in range(int(round(T / dt))):
        nu_h = nonlinear(uh)
        a = E * uh + dt * phi1 * nu_h
        uh = (a + dt * phi2 * (nonlinear(a) - nu_h)) * mask
    return torch.fft.irfft(uh, n=N)


def burgers_fine(n_samples=1200, n_fine=512, nu=0.008, T=0.3, alpha=2.5, tau=12.0,
                 seed=0, cache=True, verbose=True):
    """Pairs (u0, uT) at the FINE resolution n_fine, float64, shapes (n_samples, n_fine).

    Serves as the reference for super-resolution: subsample it afterwards.
    """
    params = dict(n=n_samples, nf=n_fine, nu=nu, T=T, a=alpha, t=tau, s=seed)

    def build():
        g = torch.Generator().manual_seed(seed)
        u0 = grf_1d(n_samples, n_fine, alpha=alpha, tau=tau, gen=g)
        return {"u0": u0, "uT": solve_burgers(u0, nu=nu, T=T)}

    d = _cached("burgers", params, build, cache=cache, verbose=verbose)
    return d["u0"], d["uT"]


def load_burgers(n_train=1000, n_test=200, resolution=128, n_fine=512, nu=0.008, T=0.3,
                 seed=0, cache=True, verbose=True):
    """1D Burgers, ready for training: (x_train, y_train, x_test, y_test).

    x = u(., 0), y = u(., T), shapes (B, 1, resolution), float32. The samples are
    independent trajectories: a train/test split by index is legitimate.
    `resolution` must divide n_fine (64, 128, 256, 512 with n_fine=512).
    """
    assert n_fine % resolution == 0, "resolution must divide n_fine"
    u0, uT = burgers_fine(n_train + n_test, n_fine, nu, T, seed=seed, cache=cache,
                          verbose=verbose)
    s = n_fine // resolution
    X = u0[:, ::s].unsqueeze(1).float()
    Y = uT[:, ::s].unsqueeze(1).float()
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def diffusion_1d(n_samples, n, nu=0.03, T=0.1, gen=None, amp=None):
    """Diffusion  u_t = nu u_xx, periodic, solved EXACTLY in Fourier.

    The operator is linear and diagonal in Fourier:
        u^(k, T) = exp(-nu (2 pi k)^2 T) . u^(k, 0)
    Returns (u0, uT, multiplier) with u0, uT of shape (n_samples, 1, n).
    amp (optional, (n_samples,)): per-sample amplitude.
    """
    u0 = grf_1d(n_samples, n, alpha=2.0, tau=4.0, gen=gen)
    if amp is not None:
        u0 = u0 * amp.view(-1, 1)
    k = torch.fft.rfftfreq(n, d=1.0 / n).to(u0.dtype)
    mult = torch.exp(-nu * (2 * math.pi * k) ** 2 * T)
    uT = torch.fft.irfft(torch.fft.rfft(u0) * mult, n=n)
    return u0.unsqueeze(1).float(), uT.unsqueeze(1).float(), mult


# ============================================================================
#  2D: Darcy
# ============================================================================
def grf_2d(n_samples, n, alpha=2.0, tau=5.0, gen=None):
    """Periodic 2D Gaussian random field, unit variance."""
    kx = torch.fft.fftfreq(n, d=1.0 / n)
    ky = torch.fft.rfftfreq(n, d=1.0 / n)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    amp = (4 * math.pi**2 * K2 + tau**2) ** (-alpha / 2.0) * n
    xi = torch.randn(n_samples, n, ky.numel(), 2, generator=gen, dtype=torch.float64)
    coef = torch.complex(xi[..., 0], xi[..., 1]) * amp
    coef[:, 0, 0] = 0.0
    f = torch.fft.irfft2(coef, s=(n, n))
    return f / f.std()


def darcy_solve(a):
    """-div(a grad u) = 1 on (0,1)^2, u = 0 on the boundary.

    5-point finite volumes, face transmissivity by HARMONIC MEAN (the right choice for a
    discontinuous coefficient). a: numpy array (n, n).
    """
    if sp_sparse is None:
        raise ImportError("darcy_solve requires scipy")
    n = a.shape[0]
    h = 1.0 / n
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    diag = np.zeros(n * n)
    for ax in (0, 1):
        s1 = (slice(0, n - 1), slice(None)) if ax == 0 else (slice(None), slice(0, n - 1))
        s2 = (slice(1, n), slice(None)) if ax == 0 else (slice(None), slice(1, n))
        T = 2 * a[s1] * a[s2] / (a[s1] + a[s2])                        # harmonic mean
        i1, i2, Tv = idx[s1].ravel(), idx[s2].ravel(), T.ravel()
        rows += [i1, i2]
        cols += [i2, i1]
        vals += [-Tv, -Tv]
        np.add.at(diag, i1, Tv)
        np.add.at(diag, i2, Tv)
    for sb in [(0, slice(None)), (n - 1, slice(None)), (slice(None), 0), (slice(None), n - 1)]:
        np.add.at(diag, idx[sb].ravel(), (2.0 * a[sb]).ravel())        # boundary face at h/2
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag)
    A = sp_sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n)).tocsc()
    return sp_linalg.spsolve(A, np.full(n * n, h * h)).reshape(n, n)


def load_darcy(n_train=600, n_test=100, n=64, low=3.0, high=12.0, seed=101, cache=True,
               verbose=True):
    """2D Darcy: (x_train, y_train, x_test, y_test), shapes (B, 1, n, n), float32.

    x = coefficient a, binary in {low, high} (thresholded Gaussian field),
    y = solution u of -div(a grad u) = 1, u = 0 on the boundary (NON-periodic domain).
    """
    params = dict(ntr=n_train, nte=n_test, n=n, lo=low, hi=high, s=seed)

    def build():
        g = torch.Generator().manual_seed(seed)
        fields = grf_2d(n_train + n_test, n, alpha=2.0, tau=5.0, gen=g).numpy()
        A = np.where(fields > 0, high, low)
        U = np.stack([darcy_solve(A[i]) for i in range(len(A))])
        return {"a": torch.from_numpy(A).unsqueeze(1).float(),
                "u": torch.from_numpy(U).unsqueeze(1).float()}

    d = _cached("darcy", params, build, cache=cache, verbose=verbose)
    X, Y = d["a"], d["u"]
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


# ============================================================================
#  3D: exact advection-diffusion (appendix A4)
# ============================================================================
def advdiff_3d(n_samples, n=16, nu=0.004, c=(0.6, -0.3, 0.2), T=0.4, tau=3.0, gen=None):
    """u_t + c . grad u = nu lap u, 3D periodic, solved exactly in Fourier.

    Returns (u0, uT) of shape (n_samples, 1, n, n, n).
    """
    kx = torch.fft.fftfreq(n, d=1.0 / n)
    ky = torch.fft.fftfreq(n, d=1.0 / n)
    kz = torch.fft.rfftfreq(n, d=1.0 / n)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    amp = (4 * math.pi**2 * K2 + tau**2) ** (-1.5) * n ** 1.5
    xi = torch.randn(n_samples, n, n, kz.numel(), 2, generator=gen, dtype=torch.float64)
    coef = torch.complex(xi[..., 0], xi[..., 1]) * amp
    coef[:, 0, 0, 0] = 0.0
    u0 = torch.fft.irfftn(coef, s=(n, n, n), dim=(-3, -2, -1))
    u0 = u0 / u0.std()
    phase = torch.exp(-1j * 2 * math.pi * (c[0] * KX + c[1] * KY + c[2] * KZ) * T)
    mult = torch.exp(-nu * (2 * math.pi) ** 2 * K2 * T) * phase
    uT = torch.fft.irfftn(torch.fft.rfftn(u0, dim=(-3, -2, -1)) * mult, s=(n, n, n),
                          dim=(-3, -2, -1))
    return u0.unsqueeze(1).float(), uT.unsqueeze(1).float()


# ============================================================================
#  2D: Kelvin-Helmholtz mixing layer (notebook 07)
# ============================================================================
#  2D incompressible Navier-Stokes, vorticity / stream-function formulation, on the
#  periodic square [0,1)^2, with a passive scalar c (tracer concentration):
#
#      d_t w + u . grad w = nu  lap w          w = d_x v - d_y u  (vorticity)
#      d_t c + u . grad c = kap lap c          u = d_y psi, v = -d_x psi, -lap psi = w
#
#  x axis = second-to-last dimension of the arrays, y axis = last dimension.
def _kh_wavenumbers(n, dtype):
    kx = 2 * math.pi * torch.fft.fftfreq(n, d=1.0 / n, dtype=dtype)
    ky = 2 * math.pi * torch.fft.rfftfreq(n, d=1.0 / n, dtype=dtype)
    return torch.meshgrid(kx, ky, indexing="ij")


def kh_base_flow(n, delta=0.04, dtype=torch.float64):
    """Periodic double shear layer, a function of y only (arrays of size n).

    u(y) = tanh((y - 1/4)/delta) - tanh((y - 3/4)/delta) - 1: +1 in the central band,
    -1 outside; two shear layers of thickness delta at y = 1/4 and 3/4.
    Returns (u, w, c): velocity, vorticity w = -du/dy, and scalar c = (1 + u)/2, which is
    1 in the central stream and 0 in the outer stream.
    """
    y = torch.arange(n, dtype=dtype) / n
    s1, s2 = (y - 0.25) / delta, (y - 0.75) / delta
    u = torch.tanh(s1) - torch.tanh(s2) - 1.0
    w = -(torch.cosh(s1) ** -2 - torch.cosh(s2) ** -2) / delta
    return u, w, 0.5 * (1.0 + u)


def kh_initial(n_samples, n=128, delta=0.04, eps=0.02, kmax=4, gen=None, dtype=torch.float64):
    """Initial conditions: base flow + small RANDOM perturbation.

    The perturbation is a stream function localized on each layer (Gaussian envelope of
    width 2*delta), a sum of modes k = 1..kmax in x with Gaussian amplitudes and uniform
    phases, independent for the two layers. It is scaled so that its maximum velocity
    equals `eps`.
    Returns (w0, c0, w_pert), each of shape (n_samples, n, n).
    """
    x = torch.arange(n, dtype=dtype) / n
    X, Y = torch.meshgrid(x, x, indexing="ij")
    psi = torch.zeros(n_samples, n, n, dtype=dtype)
    for y_layer in (0.25, 0.75):
        env = torch.exp(-((Y - y_layer) / (2 * delta)) ** 2)
        amp = torch.randn(n_samples, kmax, generator=gen, dtype=dtype)
        phase = 2 * math.pi * torch.rand(n_samples, kmax, generator=gen, dtype=dtype)
        for k in range(1, kmax + 1):
            psi += (amp[:, k - 1, None, None] * env
                    * torch.cos(2 * math.pi * k * X + phase[:, k - 1, None, None]))
    KX, KY = _kh_wavenumbers(n, dtype)
    psih = torch.fft.rfft2(psi)
    up = torch.fft.irfft2(1j * KY * psih, s=(n, n))
    vp = torch.fft.irfft2(-1j * KX * psih, s=(n, n))
    scale = eps / torch.sqrt(up**2 + vp**2).amax(dim=(1, 2), keepdim=True)
    w_pert = torch.fft.irfft2((KX**2 + KY**2) * psih, s=(n, n)) * scale      # -lap psi
    _, wb, cb = kh_base_flow(n, delta, dtype)
    w0 = wb.view(1, 1, n) + w_pert
    c0 = cb.view(1, 1, n).expand(n_samples, n, n).clone()
    return w0, c0, w_pert


def solve_kh(w0, c0, nu=3e-4, kappa=3e-4, t_out=(1.5, 3.0), dt=0.005):
    """Integrates (w, c) from t = 0 to the times t_out. w0, c0: (B, n, n).

    Pseudo-spectral (FFT), 2/3 de-aliasing of the nonlinear terms, 4th-order Runge-Kutta.
    Returns a tensor (B, len(t_out), 2, n, n): channel 0 = w, channel 1 = c.
    Computed in the dtype of w0 (float32: twice as fast, more than accurate enough here).
    """
    B, n, _ = w0.shape
    KX, KY = _kh_wavenumbers(n, w0.dtype)
    K2 = KX**2 + KY**2
    K2inv = torch.where(K2 > 0, 1.0 / K2, torch.zeros_like(K2))
    kcut = (2.0 / 3.0) * math.pi * n                     # 2/3 of the maximum wavenumber pi*n
    mask = ((KX.abs() < kcut) & (KY.abs() < kcut)).to(w0.dtype)
    ifft = lambda fh: torch.fft.irfft2(fh, s=(n, n))     # noqa: E731

    def rhs(wh, ch):
        psih = wh * K2inv
        u, v = ifft(1j * KY * psih), ifft(-1j * KX * psih)
        adv_w = u * ifft(1j * KX * wh) + v * ifft(1j * KY * wh)
        adv_c = u * ifft(1j * KX * ch) + v * ifft(1j * KY * ch)
        return (-torch.fft.rfft2(adv_w) * mask - nu * K2 * wh,
                -torch.fft.rfft2(adv_c) * mask - kappa * K2 * ch)

    steps_out = {int(round(t / dt)): i for i, t in enumerate(t_out)}
    out = torch.zeros(B, len(t_out), 2, n, n, dtype=w0.dtype)
    wh, ch = torch.fft.rfft2(w0), torch.fft.rfft2(c0)
    for s in range(1, max(steps_out) + 1):
        k1 = rhs(wh, ch)
        k2 = rhs(wh + 0.5 * dt * k1[0], ch + 0.5 * dt * k1[1])
        k3 = rhs(wh + 0.5 * dt * k2[0], ch + 0.5 * dt * k2[1])
        k4 = rhs(wh + dt * k3[0], ch + dt * k3[1])
        wh = wh + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ch = ch + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if s in steps_out:
            out[:, steps_out[s], 0], out[:, steps_out[s], 1] = ifft(wh), ifft(ch)
    return out


KH_HORIZONS = (1.5, 3.0)


def kh_fine(n_samples=700, n_test=100, n_fine=128, delta=0.04, eps=0.02, nu=3e-4, dt=0.005,
            seed=7, chunk=100, cache=True, verbose=True):
    """Simulates n_samples mixing layers at resolution n_fine, at the times KH_HORIZONS.

    The cache keeps everything at resolution 64 (every other point), and the LAST n_test
    samples also at resolution n_fine (super-resolution).
    First call: 20 to 30 min on 2 CPU cores (depending on the load), then instantaneous.
    """
    params = dict(n=n_samples, nt=n_test, nf=n_fine, d=delta, e=eps, nu=nu, dt=dt, s=seed,
                  t=KH_HORIZONS)

    def build():
        g = torch.Generator().manual_seed(seed)
        _, _, wp = kh_initial(n_samples, n_fine, delta, eps, gen=g)
        s = n_fine // 64
        d = {"w0p_64": torch.empty(n_samples, 64, 64),
             "y_64": torch.empty(n_samples, len(KH_HORIZONS), 2, 64, 64),
             "w0p_fine": torch.empty(n_test, n_fine, n_fine),
             "y_fine": torch.empty(n_test, len(KH_HORIZONS), 2, n_fine, n_fine)}
        _, wb, cb = kh_base_flow(n_fine, delta)
        for i0 in range(0, n_samples, chunk):
            t0 = time.time()
            sl = slice(i0, min(i0 + chunk, n_samples))
            w0 = (wb.view(1, 1, -1) + wp[sl]).float()
            c0 = cb.view(1, 1, -1).expand_as(w0).float().clone()
            y = solve_kh(w0, c0, nu=nu, kappa=nu, t_out=KH_HORIZONS, dt=dt)
            d["w0p_64"][sl] = wp[sl, ::s, ::s].float()
            d["y_64"][sl] = y[..., ::s, ::s]
            for j in range(sl.start, sl.stop):
                if j >= n_samples - n_test:
                    d["w0p_fine"][j - (n_samples - n_test)] = wp[j].float()
                    d["y_fine"][j - (n_samples - n_test)] = y[j - sl.start]
            if verbose:
                print(f"[data] kelvin-helmholtz: samples {sl.start}-{sl.stop - 1} "
                      f"simulated in {time.time() - t0:.0f} s")
        return d

    return _cached("kh", params, build, cache=cache, verbose=verbose)


def load_kh(horizon=1.5, n_train=600, n_test=100, test_resolution=64, cache=True,
            verbose=True):
    """Kelvin-Helmholtz mixing layer: (x_train, y_train, x_test, y_test).

    x = initial vorticity perturbation w0 - w_base, shape (B, 1, 64, 64);
    y = (w, c) at time `horizon` (1.5 or 3.0), shape (B, 2, 64, 64).
    test_resolution=128 returns the test set at the solver's resolution (super-resolution).
    """
    assert horizon in KH_HORIZONS, f"horizon must be one of {KH_HORIZONS}"
    d = kh_fine(n_samples=n_train + n_test, n_test=n_test, cache=cache, verbose=verbose)
    it = KH_HORIZONS.index(horizon)
    X, Y = d["w0p_64"].unsqueeze(1), d["y_64"][:, it]
    xtr, ytr = X[:n_train], Y[:n_train]
    if test_resolution == 64:
        return xtr, ytr, X[n_train:], Y[n_train:]
    assert test_resolution == d["w0p_fine"].shape[-1], "test_resolution: 64 or 128"
    return xtr, ytr, d["w0p_fine"].unsqueeze(1), d["y_fine"][:, it]
