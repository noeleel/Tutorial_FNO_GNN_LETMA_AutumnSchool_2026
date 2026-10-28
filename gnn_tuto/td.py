"""Self-checks for the GNN lab (notebooks/labs/lab_gnn_student.ipynb).

Each function loads the weights of the reference version into the class written by the
student, compares the outputs on random graphs, and prints [OK] or [FAIL] with a hint. It
never gives the solution, never crashes on wrong code, and does not disturb the random
draws. The reference outputs are computed BEFORE calling the student's code, and the student
code receives copies: code that modifies its inputs in place does not distort the
comparison.
"""
import functools

import torch
from torch_geometric.data import Batch, Data

from .graphs import knn_edge_index
from .models import LocGNN, MessageLayer

_TOL = 1e-5


def _keep_rng(fn):
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


def _run(fn, *args):
    try:
        with torch.no_grad():
            out = fn(*args)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if out is Ellipsis:
        return None, "the output is ...: the cell is not completed yet"
    if not isinstance(out, torch.Tensor):
        return None, f"the output is a {type(out).__name__}, not a tensor: missing return?"
    return out, None


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def _random_graph(n, hidden=None, in_dim=5, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 2, generator=g)
    x = torch.randn(n, hidden or in_dim, generator=g)
    return Data(x=x, pos=pos, edge_index=knn_edge_index(pos, k))


def _build(cls, ref, *args, **kwargs):
    """Instantiate the student's class and load the reference weights into it."""
    try:
        mine = cls(*args, **kwargs)
        mine.load_state_dict(ref.state_dict())
        return mine
    except Exception as exc:
        _report(False, f"structure differs from the reference ({type(exc).__name__})",
                "do not modify __init__: the check loads the reference weights into it")
        return None


_HINT_LAYER = ("message input: [h_i, h_j, rel, |rel|] in this order, with rel = pos_j - pos_i; "
               "residual update: h + U([h, a])")


@_keep_rng
def check_message_layer(cls, hidden=8):
    """Message-passing layer question: cls(hidden) must reproduce gnn_tuto.models.MessageLayer."""
    torch.manual_seed(0)
    ref = MessageLayer(hidden)
    mine = _build(cls, ref, hidden)
    if mine is None:
        return False
    ok = True
    for n, what in ((7, "7-node graph"), (12, "12-node graph")):
        d = _random_graph(n, hidden=hidden, seed=n)
        with torch.no_grad():
            y_ref = ref(d.x, d.pos, d.edge_index)
        y, err = _run(mine, d.x.clone(), d.pos.clone(), d.edge_index.clone())
        if err:
            return _report(False, f"{what}: forward fails ({err})",
                           "message() receives h_i, h_j, pos_i, pos_j; forward() calls "
                           "self.propagate(edge_index, h=h, pos=pos); concatenate along dim=-1, "
                           "and keep the dimension of the norm (keepdim=True)")
        if tuple(y.shape) != tuple(d.x.shape):
            return _report(False, f"{what}: shape {tuple(y.shape)}, expected {tuple(d.x.shape)}",
                           "one row per NODE: return the updated state, not the messages")
        e = _rel(y, y_ref)
        ok &= _report(e < _TOL, f"{what}: output {_same(e)} (diff {e:.1e})", _HINT_LAYER)
    # equivariance: permuting the nodes permutes the outputs
    d = _random_graph(9, hidden=hidden, seed=3)
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(1))
    inv = torch.argsort(perm)
    y, err = _run(mine, d.x.clone(), d.pos.clone(), d.edge_index.clone())
    yp, err2 = _run(mine, d.x[perm].clone(), d.pos[perm].clone(), inv[d.edge_index])
    if not (err or err2) and y.shape == yp.shape == d.x.shape:
        e = _rel(yp, y[perm])
        ok &= _report(e < _TOL, f"permuting the nodes permutes the outputs (diff {e:.1e})",
                      "the message must depend only on nodes i and j, not on their index")
    return ok


@_keep_rng
def check_loc_gnn(cls, hidden=8, n_layers=2):
    """Model question: cls(hidden=..., n_layers=...) must reproduce gnn_tuto.models.LocGNN."""
    torch.manual_seed(0)
    ref = LocGNN(hidden=hidden, n_layers=n_layers)
    mine = _build(cls, ref, hidden=hidden, n_layers=n_layers)
    if mine is None:
        return False
    graphs = [_random_graph(n, seed=n) for n in (6, 9, 14)]
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        y_ref = ref(batch)
        y1_ref = ref(Batch.from_data_list(graphs[1:2]))
    # if the message-passing layer is wrong, the whole model is wrong
    layers = getattr(mine, "layers", None)
    if layers is not None and len(layers):
        d = _random_graph(7, hidden=hidden, seed=11)
        with torch.no_grad():
            h_ref = ref.layers[0](d.x, d.pos, d.edge_index)
        h, err = _run(layers[0], d.x.clone(), d.pos.clone(), d.edge_index.clone())
        if err or h.shape != h_ref.shape or _rel(h, h_ref) >= _TOL:
            return _report(False, "your message-passing layer differs from the reference",
                           "finish the message-passing layer question (or run its FALLBACK cell), "
                           "then re-run this cell")
    y, err = _run(mine, batch.clone())
    if err:
        return _report(False, f"batch of 3 graphs: forward fails ({err})",
                       "encoder -> layers (h, pos, edge_index) -> global_mean_pool(h, batch) -> decoder")
    if tuple(y.shape) != (3, 3):
        return _report(False, f"batch of 3 graphs: shape {tuple(y.shape)}, expected (3, 3)",
                       "one row per GRAPH: the readout pools the nodes of each graph "
                       "using data.batch")
    ok = True
    e = _rel(y, y_ref)
    ok &= _report(e < _TOL, f"batch of 3 graphs: output {_same(e)} (diff {e:.1e})",
                  "MEAN readout (global_mean_pool), then the decoder")
    y1, err = _run(mine, Batch.from_data_list(graphs[1:2]))
    if not err and y1.shape == y1_ref.shape:
        e = _rel(y1, y[1:2])
        ok &= _report(e < _TOL, f"one graph alone or in a batch: same prediction (diff {e:.1e})",
                      "the readout must use data.batch, not average over the whole batch")
    return ok


def _neighbour_sets(edge_index, n):
    """For each target node i, the set of its source nodes j."""
    out = [set() for _ in range(n)]
    for j, i in edge_index.t().tolist():
        out[i].add(j)
    return out


@_keep_rng
def check_knn_feature_graph(fn, k=3):
    """Going further: fn(h, batch, k) must return the edges j -> i of the k nearest
    neighbours j of each node i in FEATURE space (Euclidean distance between the rows of h),
    computed inside each graph of the batch, without self-loops."""
    from .dynamic import knn_feature_graph
    g = torch.Generator().manual_seed(0)
    h = torch.randn(11, 4, generator=g)
    batch = torch.tensor([0] * 5 + [1] * 6)
    ref = knn_feature_graph(h, batch, k)
    out, err = _run(fn, h.clone(), batch.clone(), k)
    if err:
        return _report(False, f"the computation fails ({err})",
                       "distances with torch.cdist(h, h); forbid pairs from different graphs and "
                       "i = j with +inf; then topk(k, largest=False)")
    if out.dtype != torch.long or out.dim() != 2 or out.shape[0] != 2:
        return _report(False, f"edge_index must be a (2, E) tensor of indices, got {tuple(out.shape)} "
                              f"{out.dtype}", "torch.stack([src, dst])")
    ok = _report(out.shape[1] == ref.shape[1], f"number of edges {out.shape[1]} (expected {ref.shape[1]} = "
                 f"11 nodes x k)", "each node receives exactly k edges")
    mine, theirs = _neighbour_sets(out, 11), _neighbour_sets(ref, 11)
    cross = sum(1 for j, i in out.t().tolist() if batch[i] != batch[j])
    ok &= _report(cross == 0, f"edges between two different graphs: {cross}",
                  "set the distance between nodes of different graphs to +inf")
    loops = int((out[0] == out[1]).sum())
    ok &= _report(loops == 0, f"self-loops: {loops}", "set the diagonal of the distance matrix to +inf")
    same = sum(a == b for a, b in zip(mine, theirs))
    ok &= _report(same == 11, f"neighbour sets equal to the reference for {same}/11 nodes",
                  "row i of edge_index[1] is the TARGET i; edge_index[0] holds its neighbours j "
                  "(the k smallest distances in row i)")
    return ok
