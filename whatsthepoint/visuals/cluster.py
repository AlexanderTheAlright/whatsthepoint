"""
cluster.py: whatsthepoint Visualization Module
------------------------------------------------
Hierarchical clustering dendrogram for numeric data.

Features:
- Ward, average, complete, or single linkage
- Distance metric choice
- Optional color threshold and truncation
- whatsthepoint styling and semantic color support
- Interactive Plotly dendrogram counterpart
- Heatmap-cluster hybrid visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram
from ..utils.style import set_style, apply_titles, generate_semantic_palette, get_continuous_cmap, format_tick_labels


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════

def cluster(
    df,
    method="ward",
    metric="euclidean",
    title=None,
    subtitle=None,
    style_mode="house",
    color_threshold=None,
    truncate_mode=None,
    p=None,
    orientation="top",
    leaf_rotation=90,
    leaf_font_size=10,
    show_leaf_counts=True,
):
    """
    whatsthepoint hierarchical cluster dendrogram.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Select numeric columns
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found for clustering.")

    # Compute linkage matrix
    Z = linkage(data, method=method, metric=metric)

    # --- Assemble keyword args cleanly ---
    dendro_kwargs = dict(
        ax=ax,
        color_threshold=color_threshold,
        orientation=orientation,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        show_leaf_counts=show_leaf_counts,
    )

    if truncate_mode is not None:
        dendro_kwargs["truncate_mode"] = truncate_mode
        dendro_kwargs["p"] = int(p or 12)

    # --- Draw dendrogram ---
    dendrogram(Z, **dendro_kwargs)

    # whatsthepoint style cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_titles(fig, title, subtitle)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════

def cluster_interactive(
    df,
    method="ward",
    metric="euclidean",
    title=None,
    subtitle=None,
    style_mode="house",
):
    """
    Interactive Plotly dendrogram for whatsthepoint clustering.

    Parameters
    ----------
    df : DataFrame
        Numeric dataset to cluster.
    method : str, default "ward"
        Linkage method ('ward', 'average', 'complete', 'single').
    metric : str, default "euclidean"
        Distance metric.
    """
    set_style(style_mode)
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found for clustering.")

    fig = ff.create_dendrogram(
        data.values,
        labels=data.index.astype(str),
        linkagefun=lambda x: linkage(x, method=method, metric=metric),
        orientation="bottom",
    )

    fig.update_layout(
        title=f"<b>{title or 'Hierarchical Cluster Dendrogram'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        template="plotly_white",
        height=600,
        margin=dict(t=80, b=60, l=60, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP-CLUSTER HYBRID
# ══════════════════════════════════════════════════════════════════════════════

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..utils.style import set_style, get_continuous_cmap

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..utils.style import set_style, get_continuous_cmap

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ..utils.style import set_style, get_continuous_cmap

def heatmap_cluster(
    df,
    method="ward",
    metric="euclidean",
    title=None,
    subtitle=None,
    style_mode="house",
    cmap=None,
    annot=False,
    show_values=False,
    figsize=(14, 12),
):
    """
    whatsthepoint clustered heatmap with dendrograms and precise whatsthepoint layout.
    - Title/subtitle anchored at top-left corner
    - Optional display of all cell values in bold black
    """
    set_style(style_mode)

    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    # Ensure numeric data
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found for clustering.")

    # If show_values=True, override annot to True and force black text
    annot_opt = True if show_values else annot
    annot_kwargs = {"annot_kws": {"color": "black", "weight": "bold"}} if annot_opt else {}

    # Create clustermap
    g = sns.clustermap(
        data,
        method=method,
        metric=metric,
        cmap=cmap,
        annot=annot_opt,
        fmt=".2f" if annot_opt else "",
        figsize=figsize,
        cbar_kws={"shrink": 0.8, "label": "Value"},
        dendrogram_ratio=(0.15, 0.15),
        linewidths=0.5,
        linecolor="white",
        center=0 if "Rd" in cmap else None,
        **annot_kwargs,
    )

    fig = g.fig
    fig.subplots_adjust(top=0.90)
    if g.cax is not None:
        g.cax.set_ylabel("Value", rotation=270, labelpad=15, fontsize=11, weight="bold")

    # Clear default seaborn title
    g.fig.suptitle("")

    # Title block
    title_text = title or "Clustered Heatmap"
    subtitle_text = subtitle or ""

    left_x = 0.02
    base_y = 0.985

    if subtitle_text:
        # Center-left placement
        fig.text(
            left_x, 0.97,
            title_text,
            fontsize=18,
            fontweight="bold",
            color="black",
            ha="left",
            va="center",
        )
        fig.text(
            left_x, 0.945,
            subtitle_text,
            fontsize=12,
            color="grey",
            ha="left",
            va="center",
        )
    else:
        # Center title when no subtitle
        fig.text(
            0.5, base_y,
            title_text,
            fontsize=18,
            fontweight="bold",
            color="black",
            ha="center",
            va="top",
        )

    return fig, g.ax_heatmap



