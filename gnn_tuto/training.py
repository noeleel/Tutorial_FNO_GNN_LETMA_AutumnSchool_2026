"""Minimal training for PyG graphs (AdamW, cosine learning rate).

    hist = train_graphs(model, train_graphs, val_graphs, epochs=20)
    pred = predict_graphs(model, graphs)

By default the target is data.y (one row per graph, or one per node). With node_mask="mask",
the loss only covers the nodes where data.mask is 1 (per-node tasks, masked nodes).
"""
import math
import time

import torch
from torch_geometric.loader import DataLoader


def set_seed(seed=0):
    torch.manual_seed(seed)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _loss(pred, data, node_mask):
    err = (pred - data.y) ** 2
    if node_mask is not None:
        m = getattr(data, node_mask).bool()
        if not m.any():                                  # no node to predict in this batch
            return err.sum() * 0.0
        err = err[m]
    return err.mean()


def train_graphs(model, train, val, epochs=20, batch_size=32, lr=2e-3, weight_decay=1e-4,
                 node_mask=None, log_every=5, verbose=True, seed=0, transform=None, device=None):
    """Train `model` (graph -> prediction) with the squared error on data.y.
    transform: function applied to each training batch (not to validation), e.g. a noise
    redrawn at every batch."""
    device = torch.device(device) if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=g)
    vloader = DataLoader(val, batch_size=256)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    hist = {"epoch": [], "train": [], "val": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            if transform is not None:
                batch = transform(batch)
            loss = _loss(model(batch), batch, node_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * batch.num_graphs, n + batch.num_graphs
        model.eval()
        with torch.no_grad():
            vl = []
            for b in vloader:
                b = b.to(device)
                vl.append((_loss(model(b), b, node_mask).item(), b.num_graphs))
        val_loss = sum(a * b for a, b in vl) / sum(b for _, b in vl)
        hist["epoch"].append(ep)
        hist["train"].append(tot / n)
        hist["val"].append(val_loss)
        if verbose and (ep == 1 or ep % log_every == 0 or ep == epochs):
            print(f"  epoch {ep:3d}  train {tot / n:.4f}  val {val_loss:.4f}")
    hist["time"] = time.time() - t0
    if verbose:
        print(f"  -> {hist['time']:.1f} s, {n_params(model):,} parameters")
    return hist


@torch.no_grad()
def predict_graphs(model, graphs, batch_size=256, device=None):
    """Predict graph targets and always return a CPU tensor for analysis/plotting."""
    model.eval()
    device = torch.device(device) if device is not None else next(model.parameters()).device
    return torch.cat([model(b.to(device)).detach().cpu()
                      for b in DataLoader(graphs, batch_size=batch_size)])


def train_tensors(model, xtr, ytr, xva, yva, epochs=20, batch_size=32, lr=2e-3,
                  weight_decay=1e-4, log_every=5, verbose=True, seed=0):
    """The same loop for a model on tensors (the MLP baseline)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = math.ceil(len(xtr) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    hist = {"epoch": [], "train": [], "val": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        tot = 0.0
        for i in range(0, len(xtr), batch_size):
            idx = perm[i:i + batch_size]
            loss = ((model(xtr[idx]) - ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = ((model(xva) - yva) ** 2).mean().item()
        hist["epoch"].append(ep)
        hist["train"].append(tot / len(xtr))
        hist["val"].append(val_loss)
        if verbose and (ep == 1 or ep % log_every == 0 or ep == epochs):
            print(f"  epoch {ep:3d}  train {tot / len(xtr):.4f}  val {val_loss:.4f}")
    hist["time"] = time.time() - t0
    if verbose:
        print(f"  -> {hist['time']:.1f} s, {n_params(model):,} parameters")
    return hist


def predict_hypo(model, graphs, batch_size=256):
    """Predicted hypocentres (n, 3) in km, for location graphs."""
    from .graphs import pred_to_hypo
    ref = torch.cat([g.ref for g in graphs])
    return pred_to_hypo(predict_graphs(model, graphs, batch_size), ref)
