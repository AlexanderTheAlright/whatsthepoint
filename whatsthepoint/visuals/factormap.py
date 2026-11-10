"""
factormap.py: whatsthepoint Visualization Module
------------------------------------------------
Factor map plots for MCA, PCA, CA, or any 2D reduction.

Features:
- Active vs supplementary variable mapping
- Smart text labeling with overlap avoidance
- Supports whatsthepoint styles: fiery, viridis, sentiment, plainjane, reviewer3
- Interactive Plotly version
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..utils.style import set_style, apply_titles, get_color, format_tick_labels, legend_below
from ..utils.optional import require

def factormap(
    active_coords,
    sup_coords=None,
    title=None,
    subtitle=None,
    style_mode="house",
    label_filter=30,
    figsize=(7, 6),
    label_suffix_map=None,
    active_label="Active Variables",
    sup_label="Supplementary Variables",
    annotate=True,
    dim_labels=("Dim 1", "Dim 2"),
    perc_var=None,
):
    """
    Plot a 2D factor map (e.g., from MCA, PCA, or correspondence analysis).

    Parameters
    ----------
    active_coords : pd.DataFrame
        DataFrame with columns [0, 1] representing first two dimensions.
    sup_coords : pd.DataFrame, optional
        DataFrame with same structure for supplementary variables.
    title, subtitle : str
        Plot titles.
    style_mode : str
        One of {'house', 'viridis', 'reviewer3', 'sentiment', 'fiery', 'plainjane'}.
    label_filter : int
        Number of top-magnitude points to label.
    figsize : tuple
        Figure size in inches.
    label_suffix_map : dict, optional
        Mapping for cleaning suffixes from variable names.
    active_label, sup_label : str
        Legend labels for groups.
    annotate : bool
        Whether to show text labels.
    dim_labels : tuple
        Names for the two plotted dimensions.
    perc_var : tuple, optional
        Percentage of variance explained for Dim1, Dim2.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Prepare data
    active = active_coords.copy()
    active["r"] = np.sqrt(active[0] ** 2 + active[1] ** 2)
    sup = sup_coords.copy() if sup_coords is not None else pd.DataFrame()

    # Determine which points to label
    n_label = min(label_filter, len(active))
    label_index = set(active.nlargest(n_label, "r").index)
    if not sup.empty:
        label_index = label_index.union(set(sup.index))

    # Define color scheme by style
    style_colors = {
        "fiery": {"active": "#C53E3E", "sup": "#FFB703"},
        "viridis": {"active": "#3B528B", "sup": "#5DC863"},
        "sentiment": {"active": "#2E8B57", "sup": "#D62828"},
        "plainjane": {"active": "#4682B4", "sup": "#CD5C5C"},
        "reviewer3": {"active": "#4D4D4D", "sup": "#111111"},
        "house": {"active": "#4D4D4D", "sup": "#111111"},
    }
    colors = style_colors.get(style_mode, style_colors["viridis"])

    # Plot active
    ax.scatter(
        active[0], active[1],
        color=colors["active"], alpha=0.4, s=50, label=active_label,
        edgecolor="white", linewidth=0.5, zorder=2
    )

    # Plot supplementary
    if not sup.empty:
        ax.scatter(
            sup[0], sup[1],
            color=colors["sup"], alpha=0.8, s=120,
            edgecolor="black" if style_mode not in ("reviewer3", "house") else "white",
            linewidth=0.6, label=sup_label, zorder=4
        )

    # Text labels with backgrounds and dynamic offset from y-axis
    if annotate:
        texts = []
        all_coords = pd.concat([active, sup]) if not sup.empty else active
        y_axis_left = ax.get_xlim()[0]  # Get left edge of plot

        for idx, row in all_coords.iterrows():
            if idx in label_index:
                display_text = idx
                display_text = re.sub(r"^kw_|^jq_|^score_", "", display_text)
                display_text = re.sub(r"_uniq_high$", " (high)", display_text)
                display_text = re.sub(r"_\d+$", "", display_text)
                display_text = display_text.replace("_", " ")
                if label_suffix_map:
                    for k, v in label_suffix_map.items():
                        display_text = display_text.replace(k, v)

                # Apply dynamic offset if label is too close to y-axis
                x_pos, y_pos = row[0], row[1]
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                offset_threshold = x_range * 0.05  # 5% of x-range from left edge

                if x_pos < (y_axis_left + offset_threshold):
                    # Too close to y-axis, shift right
                    x_pos = y_axis_left + offset_threshold + 0.05 * x_range

                texts.append(ax.text(
                    x_pos, y_pos,
                    display_text.lower(),
                    fontsize=11 if "score_" not in idx else 13,
                    fontweight="bold",
                    color="#111111",
                    ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="#333333", linewidth=1.5,
                             boxstyle="round,pad=0.4", alpha=0.95),
                    zorder=6,
                ))

        adjust_text = require("adjustText", "labels").adjust_text
        adjust_text(
            texts, ax=ax,
            expand_points=(1.3, 1.3), expand_text=(1.2, 1.2),
            force_points=(1.0, 1.0), force_text=(1.5, 1.5),
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.7),
            # Force labels away from y-axis
            only_move={'points':'', 'text':'xy', 'objects':'xy'},
        )

    # Axis + decorations
    ax.axhline(0, color="grey", lw=0.6, linestyle="--", alpha=0.7)
    ax.axvline(0, color="grey", lw=0.6, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", colors="black", length=4, width=1)

    # Axis labels with explained variance
    if perc_var is not None:
        ax.set_xlabel(f"{dim_labels[0]} ({perc_var[0]*100:.2f}%)", weight="bold", fontsize=12, color="black")
        ax.set_ylabel(f"{dim_labels[1]} ({perc_var[1]*100:.2f}%)", weight="bold", fontsize=12, color="black")
    else:
        ax.set_xlabel(dim_labels[0], weight="bold", fontsize=12, color="black")
        ax.set_ylabel(dim_labels[1], weight="bold", fontsize=12, color="black")

    # Title and layout
    apply_titles(fig, title, subtitle, n=len(active))

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    # One legend centred below the axes
    legend_below(ax)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


def factormap_interactive(
    active_coords,
    sup_coords=None,
    title=None,
    subtitle=None,
    style_mode="house",
    label_filter=30,
    active_label="Active Variables",
    sup_label="Supplementary Variables",
    dim_labels=("Dim 1", "Dim 2"),
    perc_var=None,
):
    """Interactive Plotly factor map."""
    set_style(style_mode)

    # Prepare data
    active = active_coords.copy()
    active["r"] = np.sqrt(active[0] ** 2 + active[1] ** 2)
    sup = sup_coords.copy() if sup_coords is not None else pd.DataFrame()

    # Determine which points to label
    n_label = min(label_filter, len(active))
    label_index = set(active.nlargest(n_label, "r").index)
    if not sup.empty:
        label_index = label_index.union(set(sup.index))

    # Define color scheme
    style_colors = {
        "fiery": {"active": "#C53E3E", "sup": "#FFB703"},
        "viridis": {"active": "#3B528B", "sup": "#5DC863"},
        "sentiment": {"active": "#2E8B57", "sup": "#D62828"},
        "plainjane": {"active": "#4682B4", "sup": "#CD5C5C"},
        "reviewer3": {"active": "#4D4D4D", "sup": "#111111"},
        "house": {"active": "#4D4D4D", "sup": "#111111"},
    }
    colors = style_colors.get(style_mode, style_colors["viridis"])

    fig = go.Figure()

    # Plot active
    fig.add_trace(go.Scatter(
        x=active[0], y=active[1],
        mode="markers",
        marker=dict(size=8, color=colors["active"], opacity=0.5,
                   line=dict(color="white", width=0.5)),
        name=active_label,
        text=[f"<b>{idx}</b>" for idx in active.index],
        hovertemplate='%{text}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>',
    ))

    # Plot supplementary
    if not sup.empty:
        fig.add_trace(go.Scatter(
            x=sup[0], y=sup[1],
            mode="markers",
            marker=dict(size=14, color=colors["sup"], opacity=0.9,
                       line=dict(color="black", width=1)),
            name=sup_label,
            text=[f"<b>{idx}</b>" for idx in sup.index],
            hovertemplate='%{text}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<extra></extra>',
        ))

    # Add zero lines
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dash"), opacity=0.5)
    fig.add_vline(x=0, line=dict(color="grey", width=1, dash="dash"), opacity=0.5)

    # Axis labels
    xaxis_title = f"{dim_labels[0]} ({perc_var[0]*100:.2f}%)" if perc_var is not None else dim_labels[0]
    yaxis_title = f"{dim_labels[1]} ({perc_var[1]*100:.2f}%)" if perc_var is not None else dim_labels[1]

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        height=700,
        width=900,
        hovermode="closest",
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
    )

    return fig


def factormap_3d(
    coords,
    sup_coords=None,
    title=None,
    subtitle=None,
    style_mode="house",
    label_filter=20,
    active_label="Active Variables",
    sup_label="Supplementary Variables",
    dim_labels=("Dim 1", "Dim 2", "Dim 3"),
    perc_var=None,
    figsize=(12, 10),
):
    """
    Plot a 3D factor map from PCA, MCA, or other dimensionality reduction.

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame with columns [0, 1, 2] for first three dimensions.
    sup_coords : pd.DataFrame, optional
        Supplementary variables coordinates.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    label_filter : int, default 20
        Number of top points to label.
    active_label : str
        Legend label for active variables.
    sup_label : str
        Legend label for supplementary variables.
    dim_labels : tuple
        Names for the three dimensions.
    perc_var : tuple, optional
        Percentage of variance explained for each dimension.
    figsize : tuple, default (12, 10)
        Figure size.

    Returns
    -------
    fig, ax : matplotlib 3D figure and axes

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=3)
    >>> coords = pd.DataFrame(pca.fit_transform(X))
    >>> perc_var = pca.explained_variance_ratio_
    >>> factormap_3d(coords, perc_var=perc_var, title='3D PCA Projection')
    """
    from mpl_toolkits.mplot3d import Axes3D

    set_style(style_mode)
    fig = plt.figure(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("white")

    # Prepare data
    active = coords.copy()
    active["r"] = np.sqrt(active[0]**2 + active[1]**2 + active[2]**2)
    sup = sup_coords.copy() if sup_coords is not None else pd.DataFrame()

    # Define colors
    style_colors = {
        "fiery": {"active": "#C53E3E", "sup": "#FFB703"},
        "viridis": {"active": "#3B528B", "sup": "#5DC863"},
        "sentiment": {"active": "#2E8B57", "sup": "#D62828"},
        "plainjane": {"active": "#4682B4", "sup": "#CD5C5C"},
        "reviewer3": {"active": "#4D4D4D", "sup": "#111111"},
        "house": {"active": "#4D4D4D", "sup": "#111111"},
    }
    colors = style_colors.get(style_mode, style_colors["viridis"])

    # Plot active
    ax.scatter(
        active[0], active[1], active[2],
        color=colors["active"], alpha=0.5, s=50, label=active_label,
        edgecolor="white", linewidth=0.5
    )

    # Plot supplementary
    if not sup.empty:
        ax.scatter(
            sup[0], sup[1], sup[2],
            color=colors["sup"], alpha=0.9, s=150,
            edgecolor="black", linewidth=1, label=sup_label
        )

    # Add labels for top points with white-bordered backgrounds
    n_label = min(label_filter, len(active))
    label_index = set(active.nlargest(n_label, "r").index)
    if not sup.empty:
        label_index = label_index.union(set(sup.index))

    all_coords = pd.concat([active, sup]) if not sup.empty else active
    for idx in label_index:
        if idx in all_coords.index:
            row = all_coords.loc[idx]
            display_text = str(idx).replace("_", " ")
            ax.text(row[0], row[1], row[2], display_text,
                   fontsize=9,
                   color='#111111',
                   weight='bold',
                   ha='center',
                   va='center',
                   bbox=dict(facecolor='white', edgecolor='#333333', linewidth=1.5,
                            boxstyle='round,pad=0.4', alpha=0.95),
                   zorder=10)

    # Axis labels
    if perc_var is not None:
        ax.set_xlabel(f"{dim_labels[0]} ({perc_var[0]*100:.2f}%)",
                     weight="bold", fontsize=11, color="black")
        ax.set_ylabel(f"{dim_labels[1]} ({perc_var[1]*100:.2f}%)",
                     weight="bold", fontsize=11, color="black")
        ax.set_zlabel(f"{dim_labels[2]} ({perc_var[2]*100:.2f}%)",
                     weight="bold", fontsize=11, color="black")
    else:
        ax.set_xlabel(dim_labels[0], weight="bold", fontsize=11, color="black")
        ax.set_ylabel(dim_labels[1], weight="bold", fontsize=11, color="black")
        ax.set_zlabel(dim_labels[2], weight="bold", fontsize=11, color="black")

    # Grid

    # Title - reduce space, increase plot area
    apply_titles(fig, title, subtitle)

    # Adjust layout - increase plot space, reduce margins
    # Format tick labels: bold (no rotation for 3D plots - labels are on axes)
    format_tick_labels(ax, rotation_x=0)

    # One legend centred below the axes
    legend_below(ax)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


def factormap_3d_interactive(
    coords,
    sup_coords=None,
    title=None,
    subtitle=None,
    style_mode="house",
    active_label="Active Variables",
    sup_label="Supplementary Variables",
    dim_labels=("Dim 1", "Dim 2", "Dim 3"),
    perc_var=None,
):
    """
    Interactive 3D factor map using Plotly.

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame with columns [0, 1, 2] for first three dimensions.
    sup_coords : pd.DataFrame, optional
        Supplementary variables coordinates.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    active_label : str
        Legend label for active variables.
    sup_label : str
        Legend label for supplementary variables.
    dim_labels : tuple
        Names for the three dimensions.
    perc_var : tuple, optional
        Percentage of variance explained.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> factormap_3d_interactive(coords, perc_var=perc_var,
    ...                          title='Interactive 3D PCA')
    """
    import plotly.graph_objects as go

    set_style(style_mode)

    # Prepare data
    active = coords.copy()
    sup = sup_coords.copy() if sup_coords is not None else pd.DataFrame()

    # Define colors
    style_colors = {
        "fiery": {"active": "#C53E3E", "sup": "#FFB703"},
        "viridis": {"active": "#3B528B", "sup": "#5DC863"},
        "sentiment": {"active": "#2E8B57", "sup": "#D62828"},
        "plainjane": {"active": "#4682B4", "sup": "#CD5C5C"},
        "reviewer3": {"active": "#4D4D4D", "sup": "#111111"},
        "house": {"active": "#4D4D4D", "sup": "#111111"},
    }
    colors = style_colors.get(style_mode, style_colors["viridis"])

    fig = go.Figure()

    # Plot active
    fig.add_trace(go.Scatter3d(
        x=active[0], y=active[1], z=active[2],
        mode="markers",
        marker=dict(size=4, color=colors["active"], opacity=0.6,
                   line=dict(color="white", width=0.5)),
        name=active_label,
        text=[f"<b>{idx}</b>" for idx in active.index],
        hovertemplate='%{text}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<br>Dim3: %{z:.2f}<extra></extra>',
    ))

    # Plot supplementary
    if not sup.empty:
        fig.add_trace(go.Scatter3d(
            x=sup[0], y=sup[1], z=sup[2],
            mode="markers",
            marker=dict(size=10, color=colors["sup"], opacity=0.9,
                       line=dict(color="black", width=1)),
            name=sup_label,
            text=[f"<b>{idx}</b>" for idx in sup.index],
            hovertemplate='%{text}<br>Dim1: %{x:.2f}<br>Dim2: %{y:.2f}<br>Dim3: %{z:.2f}<extra></extra>',
        ))

    # Axis labels
    xaxis_title = f"{dim_labels[0]} ({perc_var[0]*100:.2f}%)" if perc_var is not None else dim_labels[0]
    yaxis_title = f"{dim_labels[1]} ({perc_var[1]*100:.2f}%)" if perc_var is not None else dim_labels[1]
    zaxis_title = f"{dim_labels[2]} ({perc_var[2]*100:.2f}%)" if perc_var is not None else dim_labels[2]

    # Layout
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            xaxis=dict(backgroundcolor="white", gridcolor="rgba(180,180,180,0.3)"),
            yaxis=dict(backgroundcolor="white", gridcolor="rgba(180,180,180,0.3)"),
            zaxis=dict(backgroundcolor="white", gridcolor="rgba(180,180,180,0.3)"),
        ),
        template="plotly_white",
        height=700,
        width=900,
        hovermode="closest",
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
    )

    return fig


def variance_explained(
    variance_ratios,
    n_components=None,
    title=None,
    subtitle=None,
    style_mode="house",
    figsize=(10, 6),
    output_path=None,
):
    """
    Plot variance explained by principal components with bar + cumulative curve.

    Creates a scree plot showing individual and cumulative variance explained,
    useful for determining optimal number of components to retain.

    Parameters
    ----------
    variance_ratios : array-like
        Variance explained ratios for each component (from PCA, MCA, etc.).
    n_components : int, optional
        Highlight optimal number of components with vertical line.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 6)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    fig, ax : matplotlib figure and axes

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> pca = PCA().fit(X)
    >>> variance_explained(pca.explained_variance_ratio_, title='PCA Scree Plot')

    With optimal components highlighted:
    >>> variance_explained(pca.explained_variance_ratio_, n_components=5,
    ...                   title='PCA - 5 Components Explain 80% Variance')
    """
    set_style(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Prepare data
    variance_ratios = np.array(variance_ratios)
    n = len(variance_ratios)
    components = np.arange(1, n + 1)
    cumulative = np.cumsum(variance_ratios)

    # Get colors based on style
    if style_mode == 'fiery':
        bar_color = '#C53E3E'
        line_color = '#8B0000'
    elif style_mode == 'sentiment':
        bar_color = '#2E8B57'
        line_color = '#D62828'
    elif style_mode == 'plainjane':
        bar_color = '#4682B4'
        line_color = '#2F4F4F'
    elif style_mode in ("reviewer3", "house"):
        bar_color = '#4D4D4D'
        line_color = '#111111'
    else:  # viridis or default
        bar_color = plt.cm.viridis(0.6)
        line_color = plt.cm.viridis(0.3)

    # Bar chart for individual variance
    bars = ax.bar(components, variance_ratios * 100, color=bar_color,
                  alpha=0.7, edgecolor='black', linewidth=1.5,
                  label='Individual Variance')

    # Create secondary y-axis for cumulative
    ax2 = ax.twinx()
    line = ax2.plot(components, cumulative * 100, color=line_color,
                    marker='o', linewidth=2.5, markersize=6,
                    label='Cumulative Variance')

    # Highlight optimal components if specified
    if n_components is not None and n_components <= n:
        warning_color = get_color('warning', style_mode)
        ax.axvline(n_components, color=warning_color, linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Selected: {n_components} components')
        # Add annotation
        cum_var = cumulative[n_components - 1] * 100
        ax2.annotate(f'{cum_var:.2f}%',
                    xy=(n_components, cum_var),
                    xytext=(n_components + 0.5, cum_var - 5),
                    fontsize=11, weight='bold', color=warning_color,
                    arrowprops=dict(arrowstyle='->', color=warning_color, lw=1.5))

    # Labels and styling
    ax.set_xlabel('Component Number', fontsize=12, weight='bold', color='black')
    ax.set_ylabel('Individual Variance Explained (%)', fontsize=12, weight='bold', color='black')
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, weight='bold', color='black')

    # Set x-axis to integers
    ax.set_xticks(components)
    ax.set_xlim(0.5, n + 0.5)

    # Y-axis limits
    ax.set_ylim(0, max(variance_ratios * 100) * 1.15)
    ax2.set_ylim(0, 105)

    # Grid
    ax.set_axisbelow(True)

    # Combine both axes' entries into one legend, placed below the axes at layout time
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Spines
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Title
    apply_titles(fig, title, subtitle)

    # One legend centred below the axes
    legend_below(ax, handles=lines1 + lines2, labels=labels1 + labels2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig, ax


def variance_explained_interactive(
    variance_ratios,
    n_components=None,
    title=None,
    subtitle=None,
    style_mode="house",
):
    """
    Interactive variance explained plot using Plotly.

    Parameters
    ----------
    variance_ratios : array-like
        Variance explained ratios.
    n_components : int, optional
        Highlight optimal components.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> variance_explained_interactive(pca.explained_variance_ratio_,
    ...                                title='Interactive Scree Plot')
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    set_style(style_mode)

    # Prepare data
    variance_ratios = np.array(variance_ratios)
    n = len(variance_ratios)
    components = np.arange(1, n + 1)
    cumulative = np.cumsum(variance_ratios)

    # Get colors based on style
    if style_mode == 'fiery':
        bar_color_str = 'rgba(197, 62, 62, 0.7)'
        line_color_str = 'rgba(139, 0, 0, 1.0)'
    elif style_mode == 'sentiment':
        bar_color_str = 'rgba(46, 139, 87, 0.7)'
        line_color_str = 'rgba(214, 40, 40, 1.0)'
    elif style_mode == 'plainjane':
        bar_color_str = 'rgba(70, 130, 180, 0.7)'
        line_color_str = 'rgba(47, 79, 79, 1.0)'
    elif style_mode in ("reviewer3", "house"):
        bar_color_str = 'rgba(77, 77, 77, 0.7)'
        line_color_str = 'rgba(17, 17, 17, 1.0)'
    else:  # viridis or default
        bar_color = plt.cm.viridis(0.6)
        line_color = plt.cm.viridis(0.3)
        bar_color_str = f"rgba({int(bar_color[0]*255)},{int(bar_color[1]*255)},{int(bar_color[2]*255)},0.7)"
        line_color_str = f"rgba({int(line_color[0]*255)},{int(line_color[1]*255)},{int(line_color[2]*255)},1.0)"

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=components,
            y=variance_ratios * 100,
            name='Individual Variance',
            marker=dict(color=bar_color_str, line=dict(color='black', width=1.5)),
            hovertemplate='Component %{x}<br>Variance: %{y:.2f}%<extra></extra>',
        ),
        secondary_y=False,
    )

    # Line chart
    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative * 100,
            name='Cumulative Variance',
            mode='lines+markers',
            line=dict(color=line_color_str, width=3),
            marker=dict(size=8, color=line_color_str),
            hovertemplate='Component %{x}<br>Cumulative: %{y:.2f}%<extra></extra>',
        ),
        secondary_y=True,
    )

    # Add vertical line for optimal components
    if n_components is not None and n_components <= n:
        warning_color = get_color('warning', style_mode)
        fig.add_vline(x=n_components, line=dict(color=warning_color, dash='dash', width=2),
                     annotation_text=f'{n_components} components',
                     annotation_position='top')

    # Layout
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=50, l=60, r=60),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        xaxis_title=dict(text="Component Number", font=dict(size=12, color="black")),
        plot_bgcolor="white",
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
    )

    # Set axis titles
    fig.update_yaxes(title_text="Individual Variance Explained (%)",
                     secondary_y=False, showgrid=True,
                     gridcolor="rgba(180,180,180,0.3)")
    fig.update_yaxes(title_text="Cumulative Variance Explained (%)",
                     secondary_y=True, showgrid=False)

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)",
                     tickmode='linear', tick0=1, dtick=1)

    return fig
