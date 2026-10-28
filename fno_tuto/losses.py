"""The tutorial's four losses: L1 and L2, absolute and relative.

Notation: e = pred - y, for one sample b, over its N points (and its channels).

    name       formula (per sample)                   reduction over the batch
    ---------  -------------------------------------  ------------------------
    "l1"       mean_x |e(x)|                (MAE)     mean
    "l2"       mean_x e(x)^2                (MSE)     mean
    "rel_l1"   ||e||_1 / ||y||_1                      mean of the ratios
    "rel_l2"   ||e||_2 / ||y||_2                      mean of the ratios

Usage:

    from fno_tuto.losses import get_loss
    loss_fn = get_loss("rel_l2")
    loss = loss_fn(pred, target)            # scalar
    per_sample = RelL2Loss(reduction="none")(pred, target)   # (B,)

Three remarks that matter (detailed in notebook 03):

* The ABSOLUTE versions give more weight to large-amplitude samples. The RELATIVE
  versions treat every sample equally, whatever its scale -- this is the standard
  metric of the FNO papers.
* The relative versions are computed PER SAMPLE, then averaged. Dividing the error norm
  of the whole batch by the norm of the whole batch is a different loss, which gives
  weight back to the large samples.
* We use means (and ratios of norms): all these losses are independent of the number
  of points N, hence comparable across resolutions. A SUM of squared errors, on the
  other hand, grows with N.
"""
import torch
import torch.nn as nn


def _flat(t):
    return t.reshape(t.shape[0], -1)


class AbsL1Loss(nn.Module):
    """Mean absolute error (MAE): mean |pred - y|."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        per_sample = _flat(pred - target).abs().mean(dim=1)
        return per_sample.mean() if self.reduction == "mean" else per_sample


class AbsL2Loss(nn.Module):
    """Mean squared error (MSE): mean (pred - y)^2."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        per_sample = _flat(pred - target).pow(2).mean(dim=1)
        return per_sample.mean() if self.reduction == "mean" else per_sample


class LpLoss(nn.Module):
    """RELATIVE L^p error per sample:  ||pred - y||_p / ||y||_p.

    This is the loss of the FNO paper (Li et al.). p=1 or p=2.
    eps guards against an identically zero target.
    """

    def __init__(self, p: int = 2, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.p, self.eps, self.reduction = p, eps, reduction

    def forward(self, pred, target):
        diff = _flat(pred - target).norm(p=self.p, dim=1)
        norm = _flat(target).norm(p=self.p, dim=1).clamp_min(self.eps)
        ratio = diff / norm
        return ratio.mean() if self.reduction == "mean" else ratio


class RelL1Loss(LpLoss):
    """||pred - y||_1 / ||y||_1 per sample."""

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__(p=1, eps=eps, reduction=reduction)


class RelL2Loss(LpLoss):
    """||pred - y||_2 / ||y||_2 per sample (norm NOT squared)."""

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__(p=2, eps=eps, reduction=reduction)


LOSSES = {
    "l1": AbsL1Loss,
    "l2": AbsL2Loss,
    "mse": AbsL2Loss,
    "rel_l1": RelL1Loss,
    "rel_l2": RelL2Loss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """Instantiates a loss by name: 'l1', 'l2' (= 'mse'), 'rel_l1', 'rel_l2'."""
    try:
        return LOSSES[name.lower()](**kwargs)
    except KeyError as err:
        raise ValueError(f"unknown loss {name!r}, choices: {sorted(LOSSES)}") from err


def all_metrics(pred, target) -> dict:
    """The four metrics at once (averaged over the batch), as Python floats."""
    with torch.no_grad():
        return {name: LOSSES[name]()(pred, target).item()
                for name in ("l1", "l2", "rel_l1", "rel_l2")}
