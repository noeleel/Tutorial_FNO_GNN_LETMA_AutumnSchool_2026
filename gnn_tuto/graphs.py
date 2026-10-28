"""Graph construction.

PyTorch Geometric convention: edge_index is a (2, E) tensor; an edge (j, i) goes from the
SOURCE j (row 0) to the TARGET i (row 1), and it is the target i that receives j's message.

The graphs are built by hand (a few lines of torch) rather than with
torch_geometric.nn.knn_graph, which needs a compiled extension (pyg-lib or torch-cluster).
"""
import torch
from torch_geometric.data import Data

from .seismic import ZMAX


def knn_edge_index(pos, k):
    """Edges j -> i for the k nearest neighbours j of each node i (no self-loops).

    pos: (n, d). Returns (2, n * min(k, n - 1)). Each node receives exactly k messages; the
    graph is not necessarily symmetric (j can be a neighbour of i but not the reverse)."""
    n = pos.shape[0]
    k = min(k, n - 1)
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    nbr = d.topk(k, largest=False).indices                  # (n, k): the neighbours of each i
    src = nbr.reshape(-1)
    dst = torch.arange(n).repeat_interleave(k)
    return torch.stack([src, dst])


def delaunay_edge_index(points):
    """Edges (in both directions) of a Delaunay triangulation of 2D points (n, 2)."""
    from scipy.spatial import Delaunay
    tri = torch.as_tensor(Delaunay(points.numpy()).simplices, dtype=torch.long)
    e = torch.cat([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e = torch.cat([e, e.flip(1)])
    e = torch.unique(e, dim=0)
    return e.t().contiguous()


# ----------------------------------------------------------------------------- location
LENGTH_SCALE = 20.0                   # km: positions and hypocentres are divided by 20 km
TIME_SCALE = 10.0                     # s: relative times are divided by 10 s


def reference_station(event):
    """The station with the first P arrival: usually the closest to the epicentre. Everything
    is expressed relative to it (the problem is invariant to horizontal translation)."""
    return event.stations[event.t_p.argmin(), :2]


def station_features(event, ref=None):
    """Features of each station (n, 5):
    position relative to the reference station (/ 20 km), P and S times relative to the
    first P arrival (/ 10 s; the origin time is unknown), S-pick indicator
    (the S time is 0 when it is missing)."""
    ref = reference_station(event) if ref is None else ref
    t_ref = event.t_p.min()
    rel = (event.stations[:, :2] - ref) / LENGTH_SCALE
    tp = (event.t_p - t_ref) / TIME_SCALE
    ts = (event.t_s - t_ref) / TIME_SCALE * event.s_mask
    return torch.cat([rel, tp[:, None], ts[:, None], event.s_mask[:, None]], 1).float()


def hypo_target(event, ref=None):
    """Target (3,): epicentre relative to the reference station and depth centred on
    ZMAX / 2, all divided by 20 km (a single scale: the loss counts km the same way on the
    three axes)."""
    ref = reference_station(event) if ref is None else ref
    return torch.cat([(event.hypo[:2] - ref), event.hypo[2:] - ZMAX / 2]).float() / LENGTH_SCALE


def pred_to_hypo(pred, ref):
    """Normalised prediction (B, 3) and reference stations (B, 2) -> hypocentres (B, 3) in km."""
    out = pred * LENGTH_SCALE
    return torch.cat([out[:, :2] + ref, out[:, 2:] + ZMAX / 2], 1)


def event_to_graph(event, k=6, edges=True, origin=None):
    """One earthquake -> one PyG graph: nodes = stations, edges = k nearest neighbours.

    x (n, 5) features, pos (n, 2) relative positions, y (1, 3) target, ref (1, 2) origin of
    the positions (km). edges=False gives the same graph WITHOUT edges.
    origin: by default, the first-arrival station (translation invariance); a fixed point
    (2,), e.g. the centre of the region, gives ABSOLUTE positions."""
    ref = reference_station(event) if origin is None else torch.as_tensor(origin, dtype=torch.float64)
    x = station_features(event, ref)
    pos = x[:, :2].clone()
    ei = knn_edge_index(pos, k) if edges else torch.zeros(2, 0, dtype=torch.long)
    return Data(x=x, pos=pos, edge_index=ei, y=hypo_target(event, ref)[None],
                ref=ref.float()[None], station_ids=event.station_ids)


def event_to_vector(event, n_stations):
    """The same earthquake for an MLP: a FIXED-size vector, one slot per network station (in
    network order), with zeros and a 0 indicator for missing stations."""
    feats = torch.zeros(n_stations, 6)
    feats[event.station_ids, :5] = station_features(event)
    feats[event.station_ids, 5] = 1.0
    return feats.reshape(-1)


# ----------------------------------------------------------------------------- shortcuts
def events_to_graphs(events, k=6, edges=True, origin=None):
    return [event_to_graph(e, k=k, edges=edges, origin=origin) for e in events]


def events_to_vectors(events, n_stations):
    """MLP inputs (n_events, n_stations * 6) and targets (n_events, 3)."""
    x = torch.stack([event_to_vector(e, n_stations) for e in events])
    y = torch.stack([hypo_target(e) for e in events])
    return x, y


def permute_graph(data, perm):
    """The same graph with renumbered nodes: new node k is old node perm[k]."""
    inv = torch.argsort(perm)
    out = data.clone()
    for key in ("x", "pos", "station_ids"):
        if getattr(data, key, None) is not None:
            out[key] = data[key][perm]
    out.edge_index = inv[data.edge_index]
    return out


def reference_stations(events):
    return torch.stack([reference_station(e) for e in events]).float()


def hypo_errors(hypo_pred, events):
    """Absolute errors (n_events, 3) in km, on x, y and z."""
    true = torch.stack([e.hypo for e in events]).float()
    return (hypo_pred.float() - true).abs()


def error_summary(err):
    """'x 0.86  y 0.77  z 0.95 km': median error per component."""
    m = err.median(0).values
    return f"x {m[0]:5.2f}  y {m[1]:5.2f}  z {m[2]:5.2f} km"
