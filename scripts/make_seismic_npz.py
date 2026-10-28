"""Generate data/seismic_jax.npz: the Lab 3 (GNN) seismic dataset, as plain NumPy arrays.

Run ONCE, in the PyTorch environment (torch + gnn_tuto):

    python scripts/make_seismic_npz.py

The JAX extension notebook (notebooks/extensions/jax_gnn_*.ipynb) reads the result and
needs NEITHER torch NOR torch_geometric — the same events as the PyTorch Lab 3 (same
seeds) are shipped as padded fixed-shape arrays, which is also the idiomatic JAX batching
strategy (vmap over a stacked array, masks for the varying node counts).

Layout (N_MAX = 30 stations, K = 6 neighbours, padded with zeros/-1):
    {split}_x      (n, 30, 5)   node features (rel. position /20km, dP/10s, dS/10s, S-pick flag)
    {split}_mask  (n, 30)       1 = real station (padded rows are 0)
    {split}_pos   (n, 30, 2)    relative positions /20km (= x[:, :2])
    {split}_esrc  (n, 30, 6)    edge j -> i: esrc[i, s] = the s-th neighbour of i
    {split}_edst  (n, 30, 6)    edst[i, s] = i (each node receives K messages)
    {split}_k      (n,)         effective k = min(6, node count - 1) for the event
    {split}_y      (n, 3)       normalized target (epicentre rel. /20km, depth - 15km /20km)
    {split}_ref   (n, 2)        reference station position (km)
    {split}_hypo  (n, 3)        TRUE hypocentre (km) — evaluation only, never an input
    {split}_sids  (n, 30)       station indices in the network (-1 = padding)
    {split}_vec   (n, 180)      the graph-free MLP baseline input (fixed 30-station vector)
    {split}_gap   (n,)          azimuthal gap of the event (degrees)
    {split}_n     (n,)          true node count
Splits: train (200, seed 101), val (40, seed 202), test (50, seed 303), stats (500, seed 999).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gnn_tuto.seismic import make_network, make_events, azimuthal_gap
from gnn_tuto.graphs import event_to_graph, event_to_vector, hypo_target

N_MAX, K = 30, 6

NETWORK = make_network(30, seed=2)          # same network as Lab 3


def pad_events(events):
    n = len(events)
    x = np.zeros((n, N_MAX, 5), np.float32)
    mask = np.zeros((n, N_MAX), np.float32)
    pos = np.zeros((n, N_MAX, 2), np.float32)
    esrc = np.zeros((n, N_MAX, K), np.int32)
    edst = np.zeros((n, N_MAX, K), np.int32)
    k_eff = np.zeros(n, np.int32)
    y = np.zeros((n, 3), np.float32)
    ref = np.zeros((n, 2), np.float32)
    hypo = np.zeros((n, 3), np.float32)
    sids = np.full((n, N_MAX), -1, np.int32)
    vec = np.zeros((n, N_MAX * 6), np.float32)
    gap = np.zeros(n, np.float32)
    counts = np.zeros(n, np.int32)
    for i, e in enumerate(events):
        g = event_to_graph(e, k=K)
        m = int(g.num_nodes)
        x[i, :m] = g.x.numpy()
        mask[i, :m] = 1.0
        pos[i, :m] = g.pos.numpy()
        src, dst = g.edge_index
        kk = src.numel() // m                    # effective k = min(K, m - 1), same for all nodes
        esrc[i, :m, :kk] = src.numpy().reshape(m, kk)
        edst[i, :m, :kk] = dst.numpy().reshape(m, kk)
        k_eff[i] = kk
        y[i] = g.y.numpy()[0]
        ref[i] = g.ref.numpy()[0]
        hypo[i] = e.hypo.numpy()
        sids[i, :m] = e.station_ids.numpy()
        vec[i] = event_to_vector(e, N_MAX).numpy()
        gap[i] = azimuthal_gap(e)
        counts[i] = m
    return dict(x=x, mask=mask, pos=pos, esrc=esrc, edst=edst, k=k_eff, y=y, ref=ref,
                hypo=hypo, sids=sids, vec=vec, gap=gap, n=counts)


def main():
    out = {'network': NETWORK[:, :2].numpy().astype(np.float64)}
    for split, (n_ev, seed) in [('train', (200, 101)), ('val', (40, 202)),
                                ('test', (50, 303)), ('stats', (500, 999))]:
        ev = make_events(NETWORK, n_ev, seed=seed)
        d = pad_events(ev)
        if split == 'stats':                   # the distributions cell needs less
            d = {k2: d[k2] for k2 in ('x', 'mask', 'hypo', 'n')}
        for k2, arr in d.items():
            out[f'{split}_{k2}'] = arr
        print(f'{split}: {n_ev} events '
              f'(mean node count {d["n"].mean() if "n" in d else float("nan"):.1f})')
    path = ROOT / 'data' / 'seismic_jax.npz'
    np.savez_compressed(path, **out)
    print('wrote', path, f'({path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
