"""GNN models (PyTorch Geometric).

    MessageLayer   one message-passing layer: h_i <- h_i + U(h_i, mean_j M(h_i, h_j, p_j - p_i))
    LocGNN         location: encoder -> L layers -> readout (mean over stations) -> hypocentre
    StationMLP     the graph-free baseline: an MLP on a fixed-size vector (one slot per station)
    NodeGNN        one output per node (ground motion at stations)
    MeshGNN        mesh simulator: encoder -> L MeshGraphNet-style blocks -> decoder (per node)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool


def mlp(d_in, d_hidden, d_out, n_hidden=1):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.GELU()]
        d = d_hidden
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def edge_geometry(pos, edge_index):
    """Geometric features of an edge j -> i: (p_j - p_i, |p_j - p_i|)."""
    src, dst = edge_index
    rel = pos[src] - pos[dst]
    return torch.cat([rel, rel.norm(dim=-1, keepdim=True)], -1)


class MessageLayer(MessagePassing):
    """Message passing, mean aggregation, residual update:

        m_ij = M( h_i, h_j, p_j - p_i, |p_j - p_i| )        message from j to i
        a_i  = mean of m_ij over the neighbours j of i      aggregation (permutation invariant)
        h_i <- h_i + U( h_i, a_i )                          update
    """

    def __init__(self, hidden, pos_dim=2, aggr="mean"):
        super().__init__(aggr=aggr)
        self.msg = mlp(2 * hidden + pos_dim + 1, hidden, hidden)
        self.upd = mlp(2 * hidden, hidden, hidden)

    def forward(self, h, pos, edge_index):
        a = self.propagate(edge_index, h=h, pos=pos)
        return h + self.upd(torch.cat([h, a], -1))

    def message(self, h_i, h_j, pos_i, pos_j):
        rel = pos_j - pos_i
        return self.msg(torch.cat([h_i, h_j, rel, rel.norm(dim=-1, keepdim=True)], -1))


POOL = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}


class LocGNN(nn.Module):
    """Location: one hypocentre per graph (= per earthquake)."""

    def __init__(self, in_dim=5, hidden=64, n_layers=4, out_dim=3, readout="mean"):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden=hidden, n_layers=n_layers, out_dim=out_dim,
                           readout=readout)
        self.encoder = mlp(in_dim, hidden, hidden)
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(n_layers)])
        self.readout = POOL[readout]
        self.decoder = mlp(hidden, hidden, out_dim)

    def forward(self, data):
        h = self.encoder(data.x)
        for layer in self.layers:
            h = layer(h, data.pos, data.edge_index)
        batch = data.batch if data.batch is not None else torch.zeros(h.shape[0], dtype=torch.long)
        return self.decoder(self.readout(h, batch))


class StationMLP(nn.Module):
    """The graph-free baseline: the network is a fixed-size vector, station by station."""

    def __init__(self, n_stations, in_dim=6, hidden=256, out_dim=3):
        super().__init__()
        self.config = dict(n_stations=n_stations, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        self.net = mlp(n_stations * in_dim, hidden, out_dim, n_hidden=2)

    def forward(self, v):
        return self.net(v)


class NodeGNN(nn.Module):
    """One prediction per node (per station)."""

    def __init__(self, in_dim, hidden=64, n_layers=4, out_dim=1):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden=hidden, n_layers=n_layers, out_dim=out_dim)
        self.encoder = mlp(in_dim, hidden, hidden)
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(n_layers)])
        self.decoder = mlp(hidden, hidden, out_dim)

    def forward(self, data):
        h = self.encoder(data.x)
        for layer in self.layers:
            h = layer(h, data.pos, data.edge_index)
        return self.decoder(h)


class GraphNetBlock(MessagePassing):
    """MeshGraphNet-style block: edges have their own latent state, updated before the
    nodes.   e_ij <- e_ij + E(e_ij, h_i, h_j) ;   h_i <- h_i + N(h_i, sum_j e_ij)."""

    def __init__(self, hidden):
        super().__init__(aggr="sum")
        self.edge_mlp = mlp(3 * hidden, hidden, hidden)
        self.node_mlp = mlp(2 * hidden, hidden, hidden)

    def forward(self, h, e, edge_index):
        src, dst = edge_index
        e = e + self.edge_mlp(torch.cat([e, h[dst], h[src]], -1))
        agg = self.propagate(edge_index, e=e, size=(h.shape[0], h.shape[0]))
        return h + self.node_mlp(torch.cat([h, agg], -1)), e

    def message(self, e):
        return e


class MeshGNN(nn.Module):
    """Mesh simulator (encoder - processor - decoder), after MeshGraphNets."""

    def __init__(self, node_in, edge_in=3, hidden=64, n_layers=6, out_dim=1):
        super().__init__()
        self.config = dict(node_in=node_in, edge_in=edge_in, hidden=hidden, n_layers=n_layers,
                           out_dim=out_dim)
        self.node_enc = mlp(node_in, hidden, hidden)
        self.edge_enc = mlp(edge_in, hidden, hidden)
        self.blocks = nn.ModuleList([GraphNetBlock(hidden) for _ in range(n_layers)])
        self.decoder = mlp(hidden, hidden, out_dim)

    def forward(self, data):
        h = self.node_enc(data.x)
        e = self.edge_enc(data.edge_attr)
        for block in self.blocks:
            h, e = block(h, e, data.edge_index)
        return self.decoder(h)
