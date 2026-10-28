"""Self-checks for the FNO lab (notebooks/labs/lab_fno_student.ipynb).

Each function compares the student's class with the package's reference version, using the
SAME weights, and prints a readable diagnostic:

    from fno_tuto.td import check_spectral_conv1d
    check_spectral_conv1d(MySpectralConv1d)      # -> [OK] ... or [FAIL] ... + hint

The functions return True when everything is correct. They never print the solution.
"""
import functools

import torch

from .models import FNO1d, SpectralConv1d, SpectralConv2d
from .models.ffno import FactorizedSpectralConv2d

_TOL = 1e-5


def _keep_rng(fn):
    """A check must not disturb the notebook's random draws."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.random.fork_rng():
            return fn(*args, **kwargs)
    return wrapper


def _report(ok, msg, hint=None):
    print(("[OK]     " if ok else "[FAIL]   ") + msg)
    if not ok and hint:
        print("         hint: " + hint)
    return ok


def _same(e):
    return "matches the reference" if e < _TOL else "DIFFERS from the reference"


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _bandlimited(n, n_modes=10, channels=3, seed=0):
    """The same continuous function (k < n_modes), sampled on n points."""
    g = torch.Generator().manual_seed(seed)
    coef = torch.zeros(1, channels, n // 2 + 1, dtype=torch.cfloat)
    a = torch.randn(1, channels, n_modes, 2, generator=g)
    coef[..., :n_modes] = torch.complex(a[..., 0], a[..., 1])
    return torch.fft.irfft(coef, n=n) * n


def _run(fn, *args):
    """Calls fn(*args); returns (output, None) or (None, error message)."""
    try:
        with torch.no_grad():
            out = fn(*args)
    except Exception as exc:                       # student code may raise anything
        return None, f"{type(exc).__name__}: {exc}"
    if out is Ellipsis:
        return None, "the output is ...: the cell is not completed yet"
    if not isinstance(out, torch.Tensor):
        return None, f"the output is a {type(out).__name__}, not a tensor: missing return?"
    return out, None


# ----------------------------------------------------------------------------- Q1
@_keep_rng
def check_spectral_conv1d(cls):
    """Question 1: cls(in_channels, out_channels, modes) must behave like
    fno_tuto.models.SpectralConv1d (rfft 'ortho', truncation, per-mode product, irfft)."""
    torch.manual_seed(0)
    mine = cls(3, 5, 12)
    ref = SpectralConv1d(3, 5, 12)
    if not hasattr(mine, "weight") or tuple(mine.weight.shape) != (3, 5, 12):
        return _report(False, "attribute self.weight (C_in, C_out, modes) not found",
                       "do not modify __init__")
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
    ok = True
    for n, what in ((64, "N = 64"), (65, "N = 65 (odd)"), (16, "N = 16 (fewer modes than M)")):
        x = torch.randn(4, 3, n)
        y, err = _run(mine, x)
        if err:
            return _report(False, f"{what}: computation fails ({err})",
                           "at low resolution there are fewer modes (N//2+1) than weights: "
                           "m = min(...)" if n == 16 else
                           "follow the 4 steps; every '...' must be replaced")
        if tuple(y.shape) != (4, 5, n):
            ok &= _report(False, f"{what}: output shape {tuple(y.shape)}, expected (4, 5, {n})",
                          "irfft needs n=N to return N points"
                          if n % 2 else "the output must have C_out channels and N points")
            continue
        e = _rel(y, ref(x))
        hint = ("at low resolution there are fewer modes (N//2+1) than weights: m = min(...)"
                if n == 16 else "check the einsum and the required norm='ortho' convention")
        ok &= _report(e < _TOL, f"{what}: output {_same(e)} (rel. diff {e:.1e})", hint)
    return ok


# ----------------------------------------------------------------------------- Q3
def same_function(n, n_modes=10, seed=0):
    """The SAME continuous function (no energy beyond n_modes), sampled on n points:
    shape (1, 1, n)."""
    return _bandlimited(n, n_modes=n_modes, channels=1, seed=seed)


def spectral_layer_conventions(x, R, norm_fwd, norm_inv):
    """A spectral layer (weights R: (C_in, C_out, M)) with a chosen `norm` convention for
    the forward (norm_fwd) and the inverse (norm_inv) transform."""
    x_hat = torch.fft.rfft(x, norm=norm_fwd)
    m = min(R.shape[-1], x_hat.shape[-1])
    out = torch.zeros(x.shape[0], R.shape[1], x_hat.shape[-1], dtype=x_hat.dtype)
    out[..., :m] = torch.einsum("bik,iok->bok", x_hat[..., :m], R[..., :m])
    return torch.fft.irfft(out, n=x.shape[-1], norm=norm_inv)


# ----------------------------------------------------------------------------- Q4
@_keep_rng
def check_fno1d(model_cls):
    """Question 4: model_cls(width, modes, n_layers) must reproduce fno_tuto.models.FNO1d
    (no padding, grid as an input channel) with the same weights."""
    torch.manual_seed(0)
    ref = FNO1d(in_channels=1, out_channels=1, width=8, modes=6, n_layers=2)
    mine = model_cls(width=8, modes=6, n_layers=2)
    try:
        mine.load_state_dict(ref.state_dict())
    except Exception as exc:
        return _report(False, f"structure differs from the reference ({type(exc).__name__})",
                       "do not modify the __init__ methods: lift, blocks[i].spectral, "
                       "blocks[i].pointwise, proj")
    ok = True
    blk_in = torch.randn(2, 8, 40)
    y, err = _run(mine.blocks[0], blk_in)
    if err:
        return _report(False, f"block: computation fails ({err})", "a block returns GELU(K v + W v)")
    e = _rel(y, ref.blocks[0](blk_in))
    ok &= _report(e < _TOL, f"FNO block output {_same(e)} (rel. diff {e:.1e})",
                  "v -> GELU(K v + W v): the two paths are SUMMED before the activation; "
                  "if the difference persists, check question 1 first")
    for n in (64, 100):
        a = torch.randn(3, 1, n)
        y, err = _run(mine, a)
        if err:
            return _report(False, f"model, N = {n}: computation fails ({err})",
                           "concatenate (a, x) along the channel axis (dim=1) before the lifting")
        if tuple(y.shape) != (3, 1, n):
            ok &= _report(False, f"model, N = {n}: shape {tuple(y.shape)}, expected (3, 1, {n})")
            continue
        e = _rel(y, ref(a))
        ok &= _report(e < _TOL, f"full model, N = {n}: output {_same(e)} (rel. diff {e:.1e})",
                      "lift(concat(a, x)) -> blocks -> proj")
    return ok


# ----------------------------------------------------------------------------- Q8
@_keep_rng
def check_spectral_conv2d(cls):
    """Question 8: cls must reproduce fno_tuto.models.SpectralConv2d (two weight corners)."""
    torch.manual_seed(0)
    mine = cls(3, 4, (6, 5))
    ref = SpectralConv2d(3, 4, (6, 5))
    with torch.no_grad():
        ref.weight_pos.copy_(mine.weight_pos)
        ref.weight_neg.copy_(mine.weight_neg)
    ok = True
    for hw, what in (((32, 32), "32 x 32"), ((24, 40), "24 x 40"),
                     ((10, 10), "10 x 10 (fewer rows than 2 x modes)")):
        x = torch.randn(2, 3, *hw)
        y, err = _run(mine, x)
        if err:
            return _report(False, f"{what}: computation fails ({err})",
                           "the k_x < 0 corner occupies the LAST mx rows of the spectrum")
        e = _rel(y, ref(x))
        hint = ("at low resolution, k_x = -1 is the LAST row of weight_neg: "
                "take its last mx rows" if hw == (10, 10)
                else "rows H - mx to H - 1 of the spectrum, multiplied by weight_neg")
        ok &= _report(e < _TOL, f"{what}: output {_same(e)} (rel. diff {e:.1e})", hint)
    return ok


# ----------------------------------------------------------------------------- going further
@_keep_rng
def check_factorized_conv2d(cls):
    """Going further: cls(channels, modes) must reproduce
    fno_tuto.models.ffno.FactorizedSpectralConv2d (sum of a 1D spectral convolution along x
    and one along y) with the same weights."""
    torch.manual_seed(0)
    ref = FactorizedSpectralConv2d(3, (6, 5))
    try:
        mine = cls(3, (6, 5))
        mine.load_state_dict(ref.state_dict())
    except Exception as exc:
        return _report(False, f"structure differs from the reference ({type(exc).__name__})",
                       "do not modify __init__ (weight_x, weight_y)")
    ok = True
    for hw, what in (((32, 32), "32 x 32"), ((24, 40), "24 x 40"),
                     ((8, 8), "8 x 8 (fewer modes on the grid than weights)")):
        x = torch.randn(2, 3, *hw)
        with torch.no_grad():
            y_ref = ref(x)
        y, err = _run(mine, x.clone())
        if err:
            return _report(False, f"{what}: computation fails ({err})",
                           "the y branch is the x branch with the roles of the two axes swapped: "
                           "rfft along dim=-1, einsum over the last axis, irfft with n=W")
        if y.shape != y_ref.shape:
            return _report(False, f"{what}: shape {tuple(y.shape)}, expected {tuple(y_ref.shape)}")
        e = _rel(y, y_ref)
        ok &= _report(e < _TOL, f"{what}: output {_same(e)} (rel. diff {e:.1e})",
                      "both branches use norm=self.fft_norm; keep min(modes, grid modes); "
                      "the output is the SUM of the two branches")
    return ok
