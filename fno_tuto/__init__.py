"""fno_tuto -- Fourier Neural Operators (FNO 1D, FNO 2D, FFNO), from tutorial to practice.

Common imports:

    from fno_tuto.models import FNO1d, FNO2d, FFNO2d
    from fno_tuto.losses import get_loss
    from fno_tuto.data import load_burgers, load_darcy
    from fno_tuto.training import ChannelNormalizer, train, predict, save_run, load_run

Each file in fno_tuto/models/ is self-contained (torch only) and can be copied as is.
"""
from .losses import AbsL1Loss, AbsL2Loss, LpLoss, RelL1Loss, RelL2Loss, get_loss
from .models import FFNO, FFNO2d, FNO, FNO1d, FNO2d, build_model
from .training import (DEVICE, ChannelNormalizer, load_run, n_params, predict, save_run,
                       set_seed, train)

__version__ = "1.0.0"
