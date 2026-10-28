"""Dynamic graphs: the neighbourhood of each node is recomputed, at every layer, from the
current node features instead of being fixed by geography.

    knn_feature_graph(h, batch, k, metric)   k nearest neighbours in feature space, per graph
    DynamicLocGNN                            LocGNN whose layers rebuild their graph from h
    regraph(graphs, mode)                    the same events with other static edges
                                             (geographic, arrival-time similarity, or both)

This is the idea of EdgeConv / Dynamic Graph CNN (Wang et al., 2019): after a few layers,
two nodes are "neighbours" because they carry similar information, not because they are
close in space. The similarity metric decides what "similar" means:

    "euclidean"  d(i, j) = |h_i - h_j|            sensitive to the norm of the features
    "cosine"     d(i, j) = 1 - cos(h_i, h_j)      only the direction of the features counts
"""
import functools

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from .graphs import knn_edge_index
from .models import MessageLayer, mlp

METRICS = ("euclidean", "cosine")


def feature_distance(h, metric="euclidean"):
    """Pairwise distances (n, n) between the rows of h, for the chosen similarity metric."""
    if metric == "euclidean":
        return torch.cdist(h, h)
    if metric == "cosine":
        hn = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return 1.0 - hn @ hn.t()
    raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}")


def knn_feature_graph(h, batch, k, metric="euclidean"):
    """Edges j -> i for the k nearest neighbours j of each node i in FEATURE space, computed
    separately inside each graph of the batch (no edge between two earthquakes).

    h (n, d), batch (n,) graph index of each node. Returns edge_index (2, n * k') with
    k' = min(k, smallest graph size - 1). The selection (top-k) is not differentiable: the
    gradient flows through the messages, not through the choice of neighbours."""
    with torch.no_grad():
        d = feature_distance(h, metric)
        d = d.masked_fill(batch[:, None] != batch[None, :], float("inf"))
        d.fill_diagonal_(float("inf"))
        k = min(k, int(torch.bincount(batch).min()) - 1)
        nbr = d.topk(k, largest=False).indices                    # (n, k)
    src = nbr.reshape(-1)
    dst = torch.arange(h.shape[0], device=h.device).repeat_interleave(k)
    return torch.stack([src, dst])


class DynamicLocGNN(nn.Module):
    """Same encoder, layers, readout and decoder as LocGNN, but each message-passing layer
    uses a graph rebuilt from the current features (static=False), or the geographic graph
    stored in data.edge_index (static=True, which is exactly LocGNN)."""

    def __init__(self, in_dim=5, hidden=32, n_layers=3, out_dim=3, k=6, metric="euclidean",
                 static=False, knn_fn=None):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden=hidden, n_layers=n_layers, out_dim=out_dim, k=k,
                           metric=metric, static=static, readout="mean")
        self.k, self.metric, self.static = k, metric, static
        # knn_fn(h, batch, k) -> edge_index ; by default the reference implementation
        self.knn_fn = knn_fn or functools.partial(knn_feature_graph, metric=metric)
        self.encoder = mlp(in_dim, hidden, hidden)
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(n_layers)])
        self.decoder = mlp(hidden, hidden, out_dim)

    def graphs(self, data):
        """The edge_index used by each layer (list of length n_layers), and the final states."""
        h = self.encoder(data.x)
        batch = data.batch if data.batch is not None else torch.zeros(h.shape[0], dtype=torch.long)
        used = []
        for layer in self.layers:
            ei = data.edge_index if self.static else self.knn_fn(h, batch, self.k)
            used.append(ei)
            h = layer(h, data.pos, ei)
        return used, h, batch

    def forward(self, data):
        _, h, batch = self.graphs(data)
        return self.decoder(global_mean_pool(h, batch))


def neighbour_overlap(ei_a, ei_b, n):
    """Fraction of the edges of ei_a that also belong to ei_b (both (2, E), on n nodes)."""
    a = set((ei_a[0] * n + ei_a[1]).tolist())
    b = set((ei_b[0] * n + ei_b[1]).tolist())
    return len(a & b) / max(len(a), 1)


def regraph(graphs, mode, k=6):
    """Copies of the location graphs with other (static) edges:
    "geographic"   the k nearest stations (the default of event_to_graph);
    "arrival_time" the k stations whose P arrival time is the closest (similar |t_P,i - t_P,j|),
                   i.e. stations at nearly the same distance from the source;
    "union"        both sets of edges."""
    out = []
    for g in graphs:
        g = g.clone()
        geo = knn_edge_index(g.pos, k)
        tim = knn_edge_index(g.x[:, 2:3], k)                 # column 2 = P time (normalized)
        g.edge_index = {"geographic": geo, "arrival_time": tim,
                        "union": torch.unique(torch.cat([geo, tim], 1), dim=1)}[mode]
        out.append(g)
    return out
