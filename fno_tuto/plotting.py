"""Figure style, shared by all notebooks.

    from fno_tuto.plotting import setup_style, finish, C_BLUE, C_ORANGE, CMAP_SEQ
    setup_style()
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Categorical palette in a fixed order: one entity = one colour, always the same.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_PURPLE, C_RED = "#7b5cd6", "#8f1f1e"
SERIES = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_PURPLE]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#fcfcfb"

# Magnitude -> a single hue, from light to dark.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Signed values (residuals, activations) -> diverging, grey at the centre.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1f1e"])


def setup_style():
    """Applies the style (call once at the top of a notebook)."""
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 100,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK2,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.6,
        "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "font.size": 9, "image.cmap": "seq_blue",
    })
    for cmap in (CMAP_SEQ, CMAP_DIV):
        try:
            mpl.colormaps.register(cmap, force=True)
        except TypeError:                                   # older versions
            pass


def finish(ax, title=None, xlabel=None, ylabel=None, legend=None):
    """Left-aligned title, labels, optional legend ('best' if legend=True)."""
    if title:
        ax.set_title(title, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend if isinstance(legend, str) else "best")
    return ax


def show_field(ax, data, title=None, signed=False, colorbar=True, extent=(0, 1, 0, 1),
               vmin=None, vmax=None):
    """Displays a 2D field. signed=True: diverging colormap centred on 0."""
    kw = {}
    if signed:
        lim = float(abs(data).max()) or 1.0
        kw["norm"] = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
        cmap = CMAP_DIV
    else:
        cmap = CMAP_SEQ
        kw.update(vmin=vmin, vmax=vmax)
    im = ax.imshow(data, cmap=cmap, origin="lower", extent=extent, **kw)
    ax.grid(False)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    if title:
        ax.set_title(title, loc="left", pad=6)
    return im
