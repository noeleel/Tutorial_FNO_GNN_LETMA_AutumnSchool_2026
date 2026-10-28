"""gnn_tuto: graph neural networks (PyTorch Geometric) for seismic applications.

    seismic       station networks, travel times, P/S picks, classical location
    graphs        graph construction (k nearest neighbours, Delaunay)
    models        message-passing layers and models (location, per-node, mesh)
    training      minimal training loop for PyG graphs
    ground_motion ground motion at stations: GMPE, correlated residuals, kriging
    waves         2D SH waves: finite differences, irregular mesh, GNN rollout
    dynamic       dynamic graphs: neighbourhoods rebuilt from the node features
    plots         figures: networks, message passing, model diagrams
    td            self-checks for the GNN lab
"""
import warnings

# torch_geometric 2.8 imports torch.jit.script, deprecated in torch 2.14: irrelevant warning
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
# torch_geometric imports tqdm.auto: without ipywidgets, tqdm warns in Jupyter (irrelevant here)
warnings.filterwarnings("ignore", message=".*IProgress not found.*")
