"""The models. Each file is self-contained (torch only) and can be copied as is.

    fno1d.py   SpectralConv1d, FNOBlock1d, FNO1d
    fno2d.py   SpectralConv2d, FNOBlock2d, FNO2d
    ffno.py    FactorizedSpectralConv2d, FFNOBlock2d, FFNO2d
    fno_nd.py  generic versions in dimension d: SpectralConvNd, FNO, FFNO

Construction by name (used by the command-line scripts):

    from fno_tuto.models import build_model
    model = build_model("fno1d", width=32, modes=16)
"""
from .fno1d import FNO1d, FNOBlock1d, SpectralConv1d
from .fno2d import FNO2d, FNOBlock2d, SpectralConv2d
from .ffno import FFNO2d, FFNOBlock2d, FactorizedSpectralConv2d
from .fno_nd import (FFNO, FNO, FactorizedSpectralConvNd, FFNOBlockNd, FNOBlockNd,
                     PointwiseNd, SpectralConvNd)

MODELS = {
    "fno1d": FNO1d,
    "fno2d": FNO2d,
    "ffno2d": FFNO2d,
    "fno": FNO,        # any dimension, modes=(m1, ..., md)
    "ffno": FFNO,      # any dimension, modes=(m1, ..., md)
}


def build_model(name: str, **kwargs):
    """Instantiates a model from its name and its hyperparameters."""
    try:
        cls = MODELS[name.lower()]
    except KeyError as err:
        raise ValueError(f"unknown model {name!r}, choices: {sorted(MODELS)}") from err
    return cls(**kwargs)


def model_name(model) -> str:
    """Registry name of an instance (inverse of build_model)."""
    for name, cls in MODELS.items():
        if type(model) is cls:
            return name
    raise ValueError(f"{type(model).__name__} is not in the MODELS registry")


__all__ = [
    "SpectralConv1d", "FNOBlock1d", "FNO1d",
    "SpectralConv2d", "FNOBlock2d", "FNO2d",
    "FactorizedSpectralConv2d", "FFNOBlock2d", "FFNO2d",
    "SpectralConvNd", "FactorizedSpectralConvNd", "PointwiseNd", "FNOBlockNd", "FFNOBlockNd",
    "FNO", "FFNO", "MODELS", "build_model", "model_name",
]
