"""Generate data/burgers_jax.npz: the Lab 1 (FNO) Burgers dataset as plain NumPy arrays.

Run ONCE, in the PyTorch environment (torch + fno_tuto):

    python scripts/make_burgers_npz.py

The JAX extension notebook (notebooks/extensions/jax_fno_*.ipynb) reads the result and needs
NEITHER torch NOR any other tutorial package: the same Burgers samples as the PyTorch Lab 1
(same seeds), at the training resolution N=64 (192 samples) and the super-resolution test
N=256 (32 samples).

Layout:
    X64   (192, 1, 64)   float32  initial conditions  u(x, 0)
    Y64   (192, 1, 64)   float32  targets             u(x, T), T = 0.3, nu = 0.008
    X256  (32, 1, 256)   float32  initial conditions at N=256 (seed 5, unseen)
    Y256  (32, 1, 256)   float32  targets at N=256
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fno_tuto.data import load_burgers

X64, Y64, _, _ = load_burgers(n_train=192, n_test=0, resolution=64, seed=0, cache=False)
X256, Y256, _, _ = load_burgers(n_train=32, n_test=0, resolution=256, seed=5, cache=False)
out = dict(X64=X64.numpy(), Y64=Y64.numpy(), X256=X256.numpy(), Y256=Y256.numpy())
path = ROOT / 'data' / 'burgers_jax.npz'
np.savez_compressed(path, **out)
print('wrote', path, f'({path.stat().st_size / 1e3:.0f} kB)')
