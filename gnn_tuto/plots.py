"""Figures for the GNN part: station networks, message passing, model diagrams.

    plot_event_graph(ax, event, graph, network)   an earthquake seen by the network, and its graph
    draw_message_passing()                        one message-passing layer, seen from node i
    draw_loc_gnn(model, example)                  the location model, with MEASURED shapes and parameters
    draw_node_gnn(model, example)                 a GNN with one output per node
    draw_mesh_gnn(model, example)                 the mesh simulator (encoder-processor-decoder)
    error_hist(ax, errors, labels)                error distributions (km), log scale
    plot_graph(ax, pos, edge_index)               any graph: nodes, edges, node colours
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fno_tuto.plotting import C_AQUA, C_BLUE, C_ORANGE, C_PURPLE, C_RED, INK, INK2, INK3, SURFACE


def _fmt(n):
    return f"{n:,}"


def _count(m):
    return sum(p.numel() for p in m.parameters())


def plot_event_graph(ax, event, graph=None, network=None, title=None, cmap="viridis",
                     colorbar=True):
    """The network, the stations that recorded (colour = relative P time), the graph edges,
    the epicentre (star) and the reference station (first P arrival)."""
    if network is not None:
        ax.scatter(network[:, 0], network[:, 1], marker="^", s=40, facecolor="none",
                   edgecolor=INK3, lw=0.8, label="missing station", zorder=2)
    sta = event.stations
    if graph is not None and graph.edge_index.numel():
        for j, i in graph.edge_index.t().tolist():
            ax.plot([sta[j, 0], sta[i, 0]], [sta[j, 1], sta[i, 1]], color=INK3, lw=0.6,
                    alpha=0.6, zorder=1)
    tp = (event.t_p - event.t_p.min()).numpy()
    sc = ax.scatter(sta[:, 0], sta[:, 1], c=tp, cmap=cmap, marker="^", s=70, edgecolor=INK,
                    lw=0.6, zorder=3, label="station (colour: relative P time)")
    i0 = int(event.t_p.argmin())
    ax.scatter(sta[i0, 0], sta[i0, 1], marker="^", s=170, facecolor="none", edgecolor=C_RED,
               lw=1.6, zorder=4, label="reference station")
    ax.scatter(event.hypo[0], event.hypo[1], marker="*", s=260, color=C_ORANGE, edgecolor=INK,
               lw=0.7, zorder=5, label="epicentre")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    if colorbar:
        plt.colorbar(sc, ax=ax, fraction=0.046, label="relative P time (s)")
    return sc


def _box(ax, x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.8", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def _arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.2, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, text, ha="center", va="bottom", fontsize=7.5,
                color=color)


def draw_message_passing(figsize=(12, 4.2)):
    """Diagram of one message-passing layer, from the point of view of a node i."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 36)
    ax.axis("off")
    ci = (16, 18)
    nbrs = [(5, 30), (4, 9), (16, 4), (28, 29), (29, 10)]
    for k, (x, y) in enumerate(nbrs):
        ax.add_patch(plt.Circle((x, y), 2.2, fc="#dbe9fb", ec=C_BLUE, lw=1.2))
        ax.text(x, y, f"j{k + 1}", ha="center", va="center", fontsize=8)
        _arrow(ax, x + (ci[0] - x) * 0.2, y + (ci[1] - y) * 0.2, x + (ci[0] - x) * 0.78,
               y + (ci[1] - y) * 0.78, color=C_BLUE)
    ax.add_patch(plt.Circle(ci, 2.8, fc="#fde3d6", ec=C_ORANGE, lw=1.5))
    ax.text(*ci, "i", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(16, -3.6, "the neighbours j of i (its k nearest stations)\neach send a message",
            ha="center", fontsize=8, color=INK2)
    _box(ax, 51, 27, 27, 7.5, "1. message, for each edge j → i\n"
                              "m_ij = M( h_i, h_j, p_j − p_i, |p_j − p_i| )", "#dbe9fb", C_BLUE, fs=8)
    _box(ax, 51, 14, 27, 7.5, "2. aggregation\na_i = mean of m_ij over the neighbours j",
         "#d3f1e5", C_AQUA, fs=8)
    _box(ax, 84, 14, 26, 7.5, "3. update\nh_i ← h_i + U( h_i, a_i )", "#fde3d6", C_ORANGE, fs=8)
    _arrow(ax, 33, 22, 37.2, 25.5, rad=-0.2)
    _arrow(ax, 51, 23.2, 51, 17.8)
    _arrow(ax, 64.6, 14, 70.8, 14)
    ax.text(51, 5.2, "The mean depends neither on the order nor on the number of neighbours:\n"
                     "the same layer applies to a network of any size.",
            ha="center", fontsize=8, color=INK2)
    ax.text(84, 5.2, "M and U are two small MLPs SHARED\nby all edges and all nodes.",
            ha="center", fontsize=8, color=INK2)
    plt.tight_layout()
    return fig


def _pipeline(stages, title, footer, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24)
    ax.axis("off")
    k = len(stages)
    w = 92 / k - 1.6
    xs = [4 + w / 2 + i * (w + 1.6) for i in range(k)]
    for (txt, shape, p, fc, ec), x in zip(stages, xs):
        _box(ax, x, 14, w, 8, txt, fc, ec, fs=7.8)
        ax.text(x, 8.4, shape, ha="center", fontsize=7, color=INK2, family="monospace")
        if p:
            ax.text(x, 6.2, f"{_fmt(p)} params", ha="center", fontsize=7, color=INK3)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, x0 + w / 2, 14, x1 - w / 2, 14)
    ax.text(4, 22.5, title, fontsize=10.5, fontweight="bold", color=INK)
    ax.text(4, 1.5, footer, fontsize=8, color=INK2)
    plt.tight_layout()
    return fig


def draw_loc_gnn(model, example, figsize=(13, 3.4), title=None):
    """Diagram of the location model, with the shapes and parameter counts MEASURED on an
    example graph."""
    n = example.x.shape[0]
    hidden = model.config["hidden"]
    stages = [
        (f"Input\n{n} stations", f"({n}, {example.x.shape[1]})", None, "#f0efec", INK3),
        ("Encoder\nper-station MLP", f"({n}, {hidden})", _count(model.encoder), "#dbe9fb", C_BLUE),
    ]
    for i, layer in enumerate(model.layers):
        stages.append((f"Layer {i + 1}\nmessages", f"({n}, {hidden})", _count(layer), "#fde3d6",
                       C_ORANGE))
    stages += [
        (f"Readout\n{model.config['readout']} over stations", f"(1, {hidden})", None, "#d3f1e5", C_AQUA),
        ("Decoder\nMLP", f"(1, {model.config['out_dim']})", _count(model.decoder), "#dbe9fb", C_BLUE),
        ("Hypocentre\n(x, y, z)", f"(1, {model.config['out_dim']})", None, "#f0efec", INK3),
    ]
    return _pipeline(stages, title or f"{type(model).__name__}: {_fmt(_count(model))} parameters",
                     "The number of stations n changes from one earthquake to the next: the weights "
                     "depend only on the hidden width, not on n.", figsize)


def draw_node_gnn(model, example, figsize=(13, 3.4), title=None, out_label="Output\nper station"):
    """Diagram of a GNN with one output per node (NodeGNN)."""
    n = example.x.shape[0]
    hidden = model.config["hidden"]
    stages = [(f"Input\n{n} nodes", f"({n}, {example.x.shape[1]})", None, "#f0efec", INK3),
              ("Encoder\nper-node MLP", f"({n}, {hidden})", _count(model.encoder), "#dbe9fb", C_BLUE)]
    for i, layer in enumerate(model.layers):
        stages.append((f"Layer {i + 1}\nmessages", f"({n}, {hidden})", _count(layer), "#fde3d6",
                       C_ORANGE))
    stages += [("Decoder\nper-node MLP", f"({n}, {model.config['out_dim']})", _count(model.decoder),
                "#dbe9fb", C_BLUE),
               (out_label, f"({n}, {model.config['out_dim']})", None, "#f0efec", INK3)]
    return _pipeline(stages, title or f"{type(model).__name__}: {_fmt(_count(model))} parameters",
                     "No global readout: each node keeps its own state and gets its own "
                     "prediction.", figsize)


def draw_mesh_gnn(model, example, figsize=(13, 3.6), title=None):
    """Diagram of the mesh simulator (MeshGNN): nodes AND edges carry a latent state."""
    n, e = example.x.shape[0], example.edge_index.shape[1]
    hidden = model.config["hidden"]
    stages = [(f"Input\n{n} nodes\n{e} edges", f"({n}, {example.x.shape[1]})", None, "#f0efec", INK3),
              ("Encoders\nnodes + edges", f"({n}, {hidden})",
               _count(model.node_enc) + _count(model.edge_enc), "#dbe9fb", C_BLUE)]
    nb = len(model.blocks)
    shown = range(nb) if nb <= 4 else [0, 1, nb - 1]
    for i in shown:
        if nb > 4 and i == nb - 1:
            stages.append(("…", "", None, "#ffffff", INK3))
        stages.append((f"Block {i + 1}\nedges, then\nnodes", f"({n}, {hidden})", _count(model.blocks[i]),
                       "#fde3d6", C_ORANGE))
    stages += [("Decoder\nper-node MLP", f"({n}, {model.config['out_dim']})", _count(model.decoder),
                "#dbe9fb", C_BLUE),
               ("Acceleration\nu(t+Δt) − 2u(t)\n+ u(t−Δt)", f"({n}, {model.config['out_dim']})", None, "#f0efec", INK3)]
    return _pipeline(stages, title or f"{type(model).__name__}: {_fmt(_count(model))} parameters",
                     "Each block extends the receptive field by one edge: after L blocks, a node "
                     "\"sees\" up to L edges away.", figsize)


def error_hist(ax, errors_km, labels, colors=(C_BLUE, C_ORANGE, C_AQUA, C_PURPLE, C_RED),
               title=None, lo=-2, hi=2):
    """Error distributions (km), log scale."""
    bins = np.logspace(lo, hi, 40)
    for e, lab, c in zip(errors_km, labels, colors):
        ax.hist(np.clip(np.asarray(e), 10.0**lo, 10.0**hi), bins=bins, histtype="step", lw=1.6,
                color=c, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("error (km)")
    ax.set_ylabel("number of earthquakes")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    ax.legend(fontsize=8)


def plot_graph(ax, pos, edge_index, values=None, cmap="viridis", node_size=40, title=None,
               marker="o", arrows=False, vmin=None, vmax=None, colorbar=False, label=None,
               highlight=None):
    """Any graph: nodes at positions `pos` (n, 2), edges `edge_index` (2, E).
    values: node colours; arrows: draw the direction j -> i; highlight: indices of nodes
    circled in red."""
    pos = np.asarray(pos)
    for j, i in np.asarray(edge_index).T:
        if arrows:
            ax.annotate("", xy=pos[i], xytext=pos[j],
                        arrowprops=dict(arrowstyle="-|>", color=INK3, lw=0.6, shrinkA=4, shrinkB=4))
        else:
            ax.plot(*pos[[j, i]].T, color=INK3, lw=0.6, alpha=0.7, zorder=1)
    c = INK2 if values is None else np.asarray(values)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=c, cmap=cmap, s=node_size, marker=marker,
                    edgecolor=INK, lw=0.5, zorder=3, vmin=vmin, vmax=vmax)
    if highlight is not None:
        h = pos[np.atleast_1d(highlight)]
        ax.scatter(h[:, 0], h[:, 1], s=node_size * 4, facecolor="none", edgecolor=C_RED, lw=1.6,
                   zorder=4)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    if colorbar and values is not None:
        plt.colorbar(sc, ax=ax, fraction=0.046, label=label)
    return sc
