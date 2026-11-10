"""
style.py: the whatsthepoint style engine
-----------------------------------------------------------
Core style engine for whatsthepoint plots.

Style modes:
- 'house'     → the default: black, white and greys, no gridlines, boxed bold
                legend below the panels, layout measured so nothing overlaps
- 'viridis'   → perceptually uniform viridis colours
- 'reviewer3' → black, white and greys, earlier layout
- 'sentiment' → green for positive, red for negative
- 'fiery'     → reds, oranges and purples (high contrast)
- 'plainjane' → light red/blue academic contrast
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# I. STYLE CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_STYLES = ["house", "viridis", "reviewer3", "sentiment", "fiery", "plainjane"]
ACTIVE_STYLE = "house"
BW_MODES = ("house", "reviewer3")

# Shared rcParams for every mode. 'house' adds HOUSE_OVERRIDES on top.
HOUSE_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#000000',
    'axes.linewidth': 1.0,
    'axes.grid': False,          # no gridlines by default; add only when requested
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.fancybox': False,
    'legend.edgecolor': '#CCCCCC',
    'legend.fontsize': 9,
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'lines.markeredgewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# Fonts are looked up by name and never bundled; DejaVu Sans ships with
# matplotlib, so the fallback is always present.
HOUSE_OVERRIDES = {
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'axes.titlesize': 15,
    'legend.framealpha': 1.0,
    'legend.facecolor': 'white',
    'legend.edgecolor': '0.15',
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'legend.handlelength': 2.5,
    'legend.handletextpad': 0.8,
    'legend.columnspacing': 2.0,
    'legend.borderpad': 0.8,
    'hatch.linewidth': 0.8,
    'svg.hashsalt': 'whatsthepoint',   # stable element ids across renders
}

# Line styles, markers and marker fills for groups drawn as lines or points
# in the black-and-white modes: the group is read from shape, not colour.
BW_LINESTYLES = ['-', '--', ':', '-.']
BW_MARKERS = ['o', 's', '^', 'D']
BW_MARKER_FILLS = ['#000000', '#FFFFFF', '#888888', '#CCCCCC']

# B&W (house, reviewer3): black first; 2 groups = black + white with black edge
# (never gray); 3+ groups add grays, which carry hatching.
BW_PALETTE = ['#000000', '#FFFFFF', '#888888', '#CCCCCC']
BW_TWO_GROUP = ['#000000', '#FFFFFF']
BW_HATCHES = ['', '', '//', '..', 'xx', '\\\\', '++', 'oo']
BW_COLORS = {
    'negative': '#000000',
    'positive': '#FFFFFF',
    'neutral': '#888888',
    'highlight': '#000000',
    'threshold': '#000000',
}
BW_CMAP = 'Greys'

VIRIDIS_PALETTE = ['#440154', '#31688e', '#7ad151', '#fde724']
VIRIDIS_COLORS = {
    'negative': '#d62728',
    'positive': '#2ca02c',
    'primary': '#440154',
    'secondary': '#fde724',
}
VIRIDIS_CMAP = 'viridis'

STYLES = {
    'sentiment': {
        'discrete': ['#2ca02c', '#7f7f7f', '#d62728', '#8b0000'],
        'cmap': 'RdYlGn',
        'colors': {'positive': '#2ca02c', 'neutral': '#7f7f7f', 'negative': '#d62728'},
    },
    'fiery': {
        'discrete': ['#3b0f70', '#8c2d8e', '#e65d2f', '#fcffa4'],
        'cmap': 'Reds',
        'colors': {'positive': '#ff7f0e', 'neutral': '#8c564b', 'negative': '#d62728'},
    },
    'plainjane': {
        'discrete': ['#1f4e79', '#5b9bd5', '#d6dce4', '#ffffff'],
        'cmap': 'RdBu_r',
        'colors': {'positive': '#1f77b4', 'neutral': '#7f7f7f', 'negative': '#d62728'},
    },
}

DISCRETE_PALETTES = {
    'house': BW_PALETTE,
    'reviewer3': BW_PALETTE,
    'viridis': VIRIDIS_PALETTE,
    'sentiment': STYLES['sentiment']['discrete'],
    'fiery': STYLES['fiery']['discrete'],
    'plainjane': STYLES['plainjane']['discrete'],
}

DATA_ELEMENT_KWARGS = {'edgecolor': 'black', 'linewidth': 1.0}


def is_bw(mode=None):
    """True for the black-and-white modes ('house' and 'reviewer3')."""
    return (mode or ACTIVE_STYLE).lower() in BW_MODES


def is_house(mode=None):
    """True for the default 'house' mode."""
    return (mode or ACTIVE_STYLE).lower() == "house"


def set_style(mode: str = "house"):
    """
    Apply the rcParams and record the active whatsthepoint style.

    Parameters
    ----------
    mode : str
        One of {'house','viridis','reviewer3','sentiment','fiery','plainjane'}.
    """
    mode = (mode or "house").lower()
    if mode not in AVAILABLE_STYLES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {AVAILABLE_STYLES}.")

    plt.style.use("default")
    mpl.rcParams['svg.fonttype'] = 'none'  # editable text in SVG (Typst)
    plt.rcParams.update(HOUSE_RCPARAMS)
    plt.rcParams['savefig.facecolor'] = 'white'
    if mode == "house":
        plt.rcParams.update(HOUSE_OVERRIDES)

    global ACTIVE_STYLE
    ACTIVE_STYLE = mode


def get_palette(items, mode=None):
    """Discrete palette for 1-4 items, colormap sampling for 5+."""
    mode = (mode or ACTIVE_STYLE).lower()
    items = list(items)
    if len(items) <= 4:
        colors = DISCRETE_PALETTES.get(mode, VIRIDIS_PALETTE)
        return {item: colors[i] for i, item in enumerate(items)}
    cmap_name = {'house': BW_CMAP, 'reviewer3': BW_CMAP, 'viridis': VIRIDIS_CMAP}.get(
        mode, STYLES.get(mode, {}).get('cmap', VIRIDIS_CMAP))
    cmap = mpl.colormaps[cmap_name]
    return {item: cmap(0.3 + 0.7 * i / (len(items) - 1))
            for i, item in enumerate(items)}


def get_hatches(n, mode=None):
    """
    Hatch patterns for n filled groups. Only the black-and-white modes hatch,
    and only the gray groups of a 3+ group figure; black and white stay plain.
    """
    mode = (mode or ACTIVE_STYLE).lower()
    if mode not in BW_MODES or n < 3:
        return [''] * n
    return [BW_HATCHES[i % len(BW_HATCHES)] if i >= 2 else '' for i in range(n)]


def smart_format(value, is_percentage=False):
    """Format numbers: integers no decimals, auto-detect 1 vs 2, thousands sep."""
    if value != value:  # NaN check
        return ''
    if float(value) == int(value):
        formatted = f'{int(value):,}'
    elif round(value, 1) == round(value, 2):
        formatted = f'{value:,.1f}'
    else:
        formatted = f'{value:,.2f}'
    return f'{formatted}%' if is_percentage else formatted


def save_figure(fig, path, dpi=300):
    """
    Save `fig` as SVG and PNG side by side (path with or without extension).
    Returns the two paths written.
    """
    from pathlib import Path
    path = Path(path)
    stem = path.with_suffix('') if path.suffix.lower() in ('.svg', '.png', '.pdf', '.jpg', '.jpeg') else path
    svg = stem.with_suffix('.svg')
    png = stem.with_suffix('.png')
    # metadata stripped so a re-render of the same figure is byte-stable
    fig.savefig(svg, bbox_inches='tight', facecolor='white', metadata={'Date': None})
    fig.savefig(png, bbox_inches='tight', facecolor='white', dpi=dpi,
                metadata={'Software': None})
    return svg, png


def zero_baseline(ax, values, axis="y"):
    """Start the value axis at 0 when every value is non-negative."""
    vals = np.asarray(pd_to_numeric(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size and vals.min() >= 0:
        if axis == "y":
            ax.set_ylim(bottom=0)
        else:
            ax.set_xlim(left=0)
    return ax


def pd_to_numeric(values):
    """Coerce a sequence to floats (non-numeric entries become NaN)."""
    out = []
    for v in np.ravel(np.asarray(values, dtype=object)):
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return out


def style_colorbar(cbar, label=None):
    """House colorbar label: rotation=270, labelpad=15, bold."""
    if label is not None:
        cbar.set_label(label, rotation=270, labelpad=15, weight='bold')
    return cbar


def format_tick_labels(ax, rotation_x=0, bold=True):
    """
    Apply whatsthepoint tick label formatting: bold text with optional rotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format
    rotation_x : int or float
        Rotation angle for x-tick labels (default 0° for horizontal)
    bold : bool
        Whether to make labels bold (default True)
    """
    weight = 'bold' if bold else 'normal'

    # Format x-tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight(weight)
        label.set_rotation(rotation_x)
        if rotation_x != 0:
            label.set_ha('right')

    # Format y-tick labels
    for label in ax.get_yticklabels():
        label.set_fontweight(weight)


# ══════════════════════════════════════════════════════════════════════════════
# II. SEMANTIC COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

def generate_semantic_palette(groups: dict, mode: str = None):
    """
    Generate a semantic color palette depending on the active or specified style.

    For small numbers of items (2-4), uses explicitly defined high-contrast colors
    to ensure maximum visual distinction. For 5+ items, uses colormap sampling.

    Parameters
    ----------
    groups : dict
        e.g., {'positive': [...], 'neutral': [...], 'negative': [...]}
    mode : str, optional
        Override style manually.

    Returns
    -------
    palette : dict  {variable: RGBA tuple}
    """
    style = (mode or globals().get("ACTIVE_STYLE", "house")).lower()
    palette = {}

    # Setup colormaps for 5+ items (legacy behavior)
    if style == "fiery":
        cmaps = {"positive": cm.plasma, "neutral": cm.magma, "negative": cm.inferno}
        ranges = {"positive": (0.0, 1.0), "neutral": (0.0, 1.0), "negative": (0.0, 1.0)}
    elif style == "sentiment":
        cmaps = {"positive": cm.Greens, "neutral": cm.Greys, "negative": cm.Reds_r}
        ranges = {"positive": (0.3, 1.0), "neutral": (0.2, 0.8), "negative": (0.3, 1.0)}
    elif style == "plainjane":
        cmaps = {"positive": cm.Blues, "neutral": cm.Greys, "negative": cm.Reds}
        ranges = {"positive": (0.2, 1.0), "neutral": (0.2, 0.8), "negative": (0.2, 1.0)}
    elif style in BW_MODES:
        cmaps = {"positive": cm.Greys, "neutral": cm.Greys, "negative": cm.Greys}
        ranges = {"positive": (1.0, 0.3), "neutral": (1.0, 0.3), "negative": (1.0, 0.3)}
    else:  # viridis
        cmaps = {"positive": cm.viridis, "neutral": cm.viridis, "negative": cm.viridis}
        ranges = {"positive": (0.0, 1.0), "neutral": (0.0, 1.0), "negative": (0.0, 1.0)}

    # Build palette
    for group, items in groups.items():
        if not items:
            continue

        n_items = len(items)

        # For 1-4 items, use explicit discrete palette for maximum contrast
        if n_items <= 4:
            colors = DISCRETE_PALETTES.get(style, VIRIDIS_PALETTE)[:n_items]
            for item, color in zip(items, colors):
                palette[item] = mpl.colors.to_rgba(color)
        else:
            # For 5+ items, use colormap sampling
            g = group.lower()
            key = "positive" if g.startswith("pos") else \
                  "neutral" if g.startswith("neu") else \
                  "negative" if g.startswith("neg") else "positive"
            cmap, (low, high) = cmaps[key], ranges[key]

            vals = np.linspace(low, high, n_items)
            for item, v in zip(items, vals):
                palette[item] = cmap(v)

    return palette


# ══════════════════════════════════════════════════════════════════════════════
# II.B. CONTINUOUS COLORMAPS FOR HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

# Dictionary mapping style modes to their primary colormaps
COLORS_DICT = {
    "fiery": cm.inferno,
    "viridis": cm.viridis,
    "sentiment": cm.RdYlGn,
    "plainjane": cm.RdBu_r,
    "reviewer3": cm.Greys,
    "house": cm.Greys,
}


def get_continuous_cmap(mode: str = None):
    """
    Get continuous colormap for heatmaps, correlation matrices, etc.

    Parameters
    ----------
    mode : str, optional
        Style mode. If None, uses global ACTIVE_STYLE.

    Returns
    -------
    cmap_name : str
        Matplotlib colormap name suitable for continuous data.

    Examples
    --------
    Fiery: Dark red → white (heat aesthetic)
    Viridis: Standard perceptually uniform viridis
    Sentiment: Red-white-green diverging (correlation)
    Plainjane: Blue-white-red diverging
    Reviewer3: Grayscale white-to-black
    """
    style = (mode or globals().get("ACTIVE_STYLE", "house")).lower()

    cmap_mapping = {
        "fiery": "Reds",           # Dark red → white for "heat" aesthetic
        "viridis": "viridis",      # Perceptually uniform
        "sentiment": "RdYlGn",     # Red-yellow-green diverging
        "plainjane": "RdBu_r",     # Blue-white-red diverging
        "reviewer3": "Greys",      # Grayscale for publication
        "house": "Greys",
    }

    return cmap_mapping.get(style, "viridis")


# ══════════════════════════════════════════════════════════════════════════════
# III. TITLING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def _main_axes(fig):
    """Return the figure's plotting axes, leaving out colorbar axes."""
    return [a for a in fig.axes if a.get_label() != "<colorbar>" and a.get_visible()]


def apply_titles(fig, title=None, subtitle=None, n=None, ax=None):
    """
    Place the title (and optional subtitle) centred over the plotting axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    title, subtitle : str, optional
        Title in bold; subtitle in grey on the line below it.
    n : int, optional
        Sample size, printed as "(n = ...)" in the bottom right corner.
    ax : matplotlib.axes.Axes, optional
        Axes to centre the title over. If omitted and the figure has a single
        plotting axes (colorbars are ignored) without its own title, that axes
        is used. Figures with several panels get a figure title instead.
    """
    if title is None:
        return

    if ax is None:
        main = _main_axes(fig)
        if len(main) == 1 and not main[0].get_title():
            ax = main[0]

    if ax is not None:
        ax.set_title(
            f"{title}",
            fontsize=15,
            weight="bold",
            color="black",
            loc="center",
            pad=30 if subtitle else 12,
        )
        if subtitle:
            ax.annotate(
                f"{subtitle}",
                xy=(0.5, 1.0),
                xycoords="axes fraction",
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=11,
                color="grey",
                ha="center",
                va="bottom",
            )
    else:
        fig.suptitle(
            f"{title}",
            fontsize=15,
            weight="bold",
            color="black",
            y=0.98 if subtitle else 0.96,
            ha="center",
        )
        if subtitle:
            fig.text(
                0.5,
                0.93,
                f"{subtitle}",
                fontsize=11,
                color="grey",
                ha="center",
                va="center",
            )

    if n is not None:
        fig.text(
            0.99,
            0.01,
            f"(n = {n:,})",
            fontsize=9,
            color="grey",
            ha="right",
            va="bottom",
        )


# ══════════════════════════════════════════════════════════════════════════════
# IV. DATA ELEMENT DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

def get_data_element_kwargs():
    """Return default kwargs for whatsthepoint data elements."""
    return dict(DATA_ELEMENT_KWARGS)


# ══════════════════════════════════════════════════════════════════════════════
# V. LEGEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def legend_below(ax, handles=None, labels=None, title=None, ncol=None,
                 fontsize=11, gap=8, max_cols=4):
    """
    Draw one legend centred below the axes, under the tick labels and x-axis label.

    Call it after tick labels are final (rotation included) and before
    ``fig.tight_layout``; tight_layout then leaves room for the legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    handles, labels : list, optional
        Legend entries. Taken from ``ax.get_legend_handles_labels()`` if omitted.
    title : str, optional
        Legend title. None by default.
    ncol : int, optional
        Number of columns; defaults to ``min(len(labels), max_cols)``.
    fontsize : int, default 11
    gap : float, default 8
        Space in points between the x-axis label and the legend.
    max_cols : int, default 4

    Returns
    -------
    matplotlib.legend.Legend or None
    """
    from matplotlib.transforms import offset_copy

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    old = ax.get_legend()
    if old is not None:
        old.remove()

    fig = ax.figure
    renderer = fig._get_renderer()
    axes_bottom = ax.get_window_extent(renderer).y0
    tight_bottom = ax.get_tightbbox(renderer).y0
    drop = max(axes_bottom - tight_bottom, 0) * 72.0 / fig.dpi + gap
    anchor = offset_copy(ax.transAxes, fig=fig, y=-drop, units="points")

    n_items = len(labels)
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=anchor,
        ncol=ncol or min(n_items, max_cols),
        frameon=False,
        fontsize=fontsize,
        title=title,
        title_fontsize=fontsize + 1,
        handlelength=1.6,
        columnspacing=1.6,
    )
    if title:
        legend.get_title().set_fontweight("bold")
    return legend


def draw_legend_group(ax, fig, title, var_list, palette,
                      start_x, start_y, line_height=0.04, spacing=0.04):
    """Draw structured legend group (bars/lines)."""
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        rect = Rectangle((start_x, y - 0.015), 0.015, 0.025,
                         facecolor=palette[var], transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        fig.text(start_x + 0.02, y, var,
                 transform=ax.transAxes, fontsize=10, color="#333333",
                 ha="left", va="center")
        y -= line_height
    return y - spacing


def draw_scatter_legend(ax, fig, title, var_list, palette,
                        start_x, start_y, line_height=0.04, spacing=0.04):
    """Draw structured legend group (scatter)."""
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        ax.add_line(Line2D([start_x], [y - 0.01],
                           transform=ax.transAxes,
                           marker="o", color="w",
                           markerfacecolor=palette[var],
                           markersize=8, markeredgecolor="grey"))
        fig.text(start_x + 0.02, y, var,
                 transform=ax.transAxes, fontsize=10, color="#333333",
                 ha="left", va="center")
        y -= line_height
    return y - spacing


# ══════════════════════════════════════════════════════════════════════════════
# VI. SEMANTIC COLOR RETRIEVAL (THEME-AWARE)
# ══════════════════════════════════════════════════════════════════════════════

def get_color(semantic: str, mode: str = None) -> str:
    """
    Get theme-aware color for semantic use cases.

    For reviewer3, all colors map to grayscale (black/white/gray).
    For other themes, returns appropriate colors.

    Parameters
    ----------
    semantic : str
        One of: 'negative', 'positive', 'neutral', 'warning', 'highlight',
                'line', 'threshold', 'reference', 'primary', 'secondary'
    mode : str, optional
        Override style. If None, uses ACTIVE_STYLE.

    Returns
    -------
    color : str
        Color string (hex, rgb, or named color).

    Examples
    --------
    >>> get_color('warning')  # Returns 'red' in most themes, '#333333' in reviewer3
    >>> get_color('positive', mode='reviewer3')  # Returns '#666666'
    """
    style = (mode or globals().get("ACTIVE_STYLE", "house")).lower()

    if style in BW_MODES:
        # Pure grayscale mapping for publication
        color_map = {
            'negative': '#000000',      # Black for negative/warnings
            'positive': '#666666',      # Medium gray for positive
            'neutral': '#999999',       # Light gray for neutral
            'warning': '#000000',       # Black for warnings/thresholds
            'highlight': '#333333',     # Dark gray for highlights
            'line': '#666666',          # Medium gray for lines
            'threshold': '#000000',     # Black for thresholds
            'reference': '#000000',     # Black for reference lines
            'primary': '#333333',       # Dark gray primary
            'secondary': '#999999',     # Light gray secondary
            'increasing': '#666666',    # Medium gray
            'decreasing': '#333333',    # Dark gray
        }
    elif style == "sentiment":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#2ca02c',      # Green
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#1f77b4',       # Blue
            'secondary': '#ff7f0e',     # Orange
            'increasing': '#2ca02c',    # Green
            'decreasing': '#d62728',    # Red
        }
    elif style == "fiery":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#ff7f0e',      # Orange
            'neutral': '#8c564b',       # Brown
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#e377c2',          # Pink
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#d62728',       # Red
            'secondary': '#ff7f0e',     # Orange
            'increasing': '#ff7f0e',    # Orange
            'decreasing': '#8c564b',    # Brown
        }
    elif style == "plainjane":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#1f77b4',      # Blue
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#1f77b4',       # Blue
            'secondary': '#d62728',     # Red
            'increasing': '#1f77b4',    # Blue
            'decreasing': '#d62728',    # Red
        }
    else:  # viridis
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#2ca02c',      # Green
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#440154',       # Viridis dark purple
            'secondary': '#fde724',     # Viridis yellow
            'increasing': '#2ca02c',    # Green
            'decreasing': '#d62728',    # Red
        }

    return color_map.get(semantic, '#333333')
