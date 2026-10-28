"""Training: normalization, training loop, prediction, saving.

Typical usage:

    from fno_tuto.training import ChannelNormalizer, train, predict, save_run, load_run

    x_norm, y_norm = ChannelNormalizer(x_train), ChannelNormalizer(y_train)
    hist = train(model, x_norm.encode(x_train), y_norm.encode(y_train),
                 x_norm.encode(x_test), y_norm.encode(y_test),
                 epochs=50, lr=2e-3, loss_fn=get_loss("rel_l2"), y_norm=y_norm)
    y_pred = predict(model, x_test, x_norm, y_norm)          # in physical units
    save_run("runs/burgers_fno1d", model, x_norm, y_norm, hist)
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .losses import LpLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0


def set_seed(seed: int = SEED):
    """Seeds the python, numpy and torch generators (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def n_params(model) -> int:
    """Number of REAL parameters: a complex parameter counts as 2."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


class ChannelNormalizer:
    """Standardization (x - mean) / std, computed PER CHANNEL on the training set.

    Means/standard deviations are taken over every dimension except the channel axis
    (axis 1), then FROZEN: the same ones are reused for the test set. Why per channel
    rather than global: see appendix A1.
    """

    def __init__(self, x: torch.Tensor = None, eps: float = 1e-8):
        if x is not None:
            dims = [0] + list(range(2, x.ndim))              # everything except channels
            self.mean = x.mean(dim=dims, keepdim=True)
            self.std = x.std(dim=dims, keepdim=True).clamp_min(eps)

    def encode(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        obj.mean, obj.std = state["mean"], state["std"]
        return obj

    def __repr__(self):
        m = [round(v, 4) for v in self.mean.flatten().tolist()[:4]]
        s = [round(v, 4) for v in self.std.flatten().tolist()[:4]]
        return f"ChannelNormalizer(mean={m}, std={s})"


def make_param_groups(model, weight_decay: float):
    """Weight decay on the real matrices only; NEVER on the complex spectral weights
    (it would push the non-local path towards zero) nor on biases/norms.
    Demonstrated in appendix A2."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.is_complex() or p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def train(model, xtr, ytr, xte, yte, epochs=40, batch_size=32, lr=1e-3, weight_decay=1e-4,
          loss_fn=None, y_norm=None, sched="cosine", device=DEVICE, log_every=10,
          extra_metric=None, seed=SEED, verbose=True, grad_clip=None):
    """Minimal training loop: AdamW + cosine scheduler, validation at every epoch.

    Main parameters
    ---------------
    xtr, ytr, xte, yte : tensors (B, C, ...) -- usually ALREADY normalized
    loss_fn            : training loss (default: relative L2)
    y_norm             : if given, the validation metric is computed AFTER decoding,
                         in physical units (this is what we report)
    extra_metric       : function (pred, target) -> value, logged at every epoch
    grad_clip          : max gradient norm (None = no clipping)

    Returns a dict `hist`: epoch, train, val (relative L2), lr, extra, time, params.
    """
    set_seed(seed)
    model = model.to(device)
    loss_fn = loss_fn or LpLoss(2)
    opt = torch.optim.AdamW(make_param_groups(model, weight_decay), lr=lr)
    steps = max(1, math.ceil(len(xtr) / batch_size)) * epochs
    if sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 4), gamma=0.5)
    else:
        scheduler = None

    xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
    hist = {"epoch": [], "train": [], "val": [], "lr": [], "extra": []}
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=gen)
        running, n_batches = 0.0, 0
        for i in range(0, len(xtr), batch_size):
            idx = perm[i: i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(xte[i: i + 256]) for i in range(0, len(xte), 256)])
            if y_norm is not None:
                pred, target = y_norm.decode(pred), y_norm.decode(yte)
            else:
                target = yte
            val = LpLoss(2)(pred, target).item()
            extra = extra_metric(pred, target) if extra_metric else None

        hist["epoch"].append(ep)
        hist["train"].append(running / n_batches)
        hist["val"].append(val)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["extra"].append(extra)
        if verbose and (ep % log_every == 0 or ep == 1):
            msg = f"  epoch {ep:3d}  train {running / n_batches:.4f}  val L2rel {val:.4f}"
            if extra is not None:
                msg += f"  {extra}"
            print(msg)

    hist["time"] = time.time() - t0
    hist["params"] = n_params(model)
    if verbose:
        print(f"  -> {hist['time']:.1f} s, {hist['params']:,} parameters, "
              f"final val {hist['val'][-1]:.4f} (best {min(hist['val']):.4f})")
    return hist


@torch.no_grad()
def predict(model, x, x_norm=None, y_norm=None, batch_size=256, device=DEVICE):
    """Prediction in PHYSICAL units: encodes x, applies the model, decodes y. Output on CPU."""
    model.eval()
    model = model.to(device)
    outs = []
    for i in range(0, len(x), batch_size):
        xb = x[i: i + batch_size].to(device)
        if x_norm is not None:
            xb = x_norm.encode(xb)
        yb = model(xb)
        if y_norm is not None:
            yb = y_norm.decode(yb)
        outs.append(yb.cpu())
    return torch.cat(outs)


def save_run(path, model, x_norm=None, y_norm=None, hist=None, extra: dict = None):
    """Saves a model TOGETHER WITH its normalizers (a model without them is unusable).

    Creates path/model.pt (weights + config + normalizers) and path/history.json.
    """
    from .models import model_name

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_name": model_name(model),
        "config": model.config,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "x_norm": x_norm.state_dict() if x_norm is not None else None,
        "y_norm": y_norm.state_dict() if y_norm is not None else None,
        "extra": extra or {},
    }
    torch.save(ckpt, path / "model.pt")
    if hist is not None:
        with open(path / "history.json", "w") as f:
            json.dump({k: v for k, v in hist.items() if k != "extra"}, f, indent=1)
    return path


def load_run(path, device="cpu"):
    """Reloads (model, x_norm, y_norm, extra) from a folder created by save_run."""
    from .models import build_model

    ckpt = torch.load(Path(path) / "model.pt", map_location=device)
    model = build_model(ckpt["model_name"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x_norm = ChannelNormalizer.from_state_dict(ckpt["x_norm"]) if ckpt["x_norm"] else None
    y_norm = ChannelNormalizer.from_state_dict(ckpt["y_norm"]) if ckpt["y_norm"] else None
    return model, x_norm, y_norm, ckpt.get("extra", {})
