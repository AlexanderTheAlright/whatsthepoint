"""
apc.py: whatsthepoint Visualization Module
------------------------------------------
Age-Period-Cohort (APC) visualizations in the whatsthepoint style
modes (fiery, viridis, sentiment,
plainjane, reviewer3).

Features:
- Rectangular heatmap layout
- Hexagonal Lexis diagram layout
- Intra-period and intra-cohort highlighting
- Value annotations with smart sizing
- Interactive Plotly version
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch, Rectangle, RegularPolygon
from matplotlib import cm

# Visuals utils
from ..utils.style import (
    set_style,
    apply_titles,
    generate_semantic_palette,
    format_tick_labels,
    legend_below,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_data_element_kwargs():
    """
    Returns default kwargs for data elements (heatmap cells, patches, etc.).

    Returns
    -------
    dict
        Keyword arguments for matplotlib data elements.
    """
    return {}


def get_color(element_type, style_mode):
    """
    Returns appropriate color for UI elements based on style mode.

    Parameters
    ----------
    element_type : str
        Type of element ('threshold', 'border', 'highlight', etc.).
    style_mode : str
        Style preset ('fiery', 'viridis', 'sentiment', 'plainjane', 'reviewer3').

    Returns
    -------
    str
        Color as hex string or named color.
    """
    color_map = {
        "fiery": {"threshold": "#d62728", "border": "#8B0000"},
        "viridis": {"threshold": "#000000", "border": "#440154"},
        "sentiment": {"threshold": "#000000", "border": "#2ca02c"},
        "plainjane": {"threshold": "#1f77b4", "border": "#08519c"},
        "reviewer3": {"threshold": "#333333", "border": "#000000"},
        "house": {"threshold": "#333333", "border": "#000000"},
    }

    style_colors = color_map.get(style_mode, {"threshold": "#000000", "border": "#333333"})
    return style_colors.get(element_type, "#000000")


# ══════════════════════════════════════════════════════════════════════════════
# APC HEATMAP FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def apc_heatmap(
        data,
        ages,
        periods,
        cohorts,
        layout="rectangular",
        title=None,
        subtitle=None,
        n=None,
        show_values=True,
        show_legend=True,
        style_mode="house",
        highlight_diagonals=None,
        highlight_periods=None,
        vmin=None,
        vmax=None,
):
    """
    whatsthepoint styled Age-Period-Cohort heatmap with dynamic theming.

    Parameters
    ----------
    data : array-like
        Matrix of values with shape (n_ages, n_periods) for rectangular layout,
        or flattened array for hexagonal layout.
    ages : array-like
        Age values for y-axis.
    periods : array-like
        Period values for x-axis.
    cohorts : array-like
        Cohort values for secondary axis.
    layout : str, default "rectangular"
        Visualization layout. One of {'rectangular', 'hexagonal'}.
        - 'rectangular': Standard heatmap grid
        - 'hexagonal': Lexis diagram with hexagonal tiles
    title, subtitle : str, optional
        Titles displayed above plot.
    n : int, optional
        Observation count for title suffix.
    show_values : bool, default True
        Display numeric values in cells/hexagons.
    show_legend : bool, default True
        Display legend box with variable info.
    style_mode : str, default "house"
        Visualization style preset. One of
        {'house','viridis','reviewer3','sentiment','fiery','plainjane'}.
    highlight_diagonals : list of int, optional
        Diagonal indices to highlight with thick borders (rectangular only).
        Example: [3, 7] highlights diagonals where row + col equals these values.
    highlight_periods : int, optional
        Column index after which to draw vertical separator line (rectangular only).
    vmin, vmax : float, optional
        Min and max values for color scale. Auto-detected if None.

    Returns
    -------
    fig, ax : matplotlib figure and axes
        The generated plot objects.
    """
    set_style(style_mode)

    # Route to appropriate layout function
    if layout == "rectangular":
        return _apc_rectangular(
            data, ages, periods, cohorts, title, subtitle, n,
            show_values, show_legend, style_mode,
            highlight_diagonals, highlight_periods, vmin, vmax
        )
    elif layout == "hexagonal":
        return _apc_hexagonal(
            data, ages, periods, cohorts, title, subtitle, n,
            show_values, show_legend, style_mode, vmin, vmax
        )
    else:
        raise ValueError(f"Invalid layout '{layout}'. Must be 'rectangular' or 'hexagonal'.")


# ══════════════════════════════════════════════════════════════════════════════
# RECTANGULAR LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def _apc_rectangular(
        data, ages, periods, cohorts, title, subtitle, n,
        show_values, show_legend, style_mode,
        highlight_diagonals, highlight_periods, vmin, vmax
):
    """Create rectangular APC heatmap."""

    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)

    data = np.array(data)

    # Auto-detect color range
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # ─── Color selection based on style ────────────────────────────────────────
    if style_mode == "fiery":
        cmap = "YlOrRd"
    elif style_mode == "viridis":
        cmap = "viridis"
    elif style_mode == "sentiment":
        cmap = "RdYlGn"
    elif style_mode == "plainjane":
        cmap = "Blues"
    elif style_mode in ("reviewer3", "house"):
        cmap = "Greys"
    else:
        cmap = "RdBu_r"  # Default

    # ─── Plot heatmap ──────────────────────────────────────────────────────────
    im = ax.imshow(
        data,
        cmap=cmap,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        origin='upper',
        **get_data_element_kwargs()
    )

    # ─── Axis configuration ────────────────────────────────────────────────────
    ax.set_xticks(np.arange(len(periods) + 1) - 0.5)
    ax.set_yticks(np.arange(len(ages) + 1) - 0.5)

    ax.set_xticks(np.arange(len(periods)), minor=True)
    ax.set_yticks(np.arange(len(ages)), minor=True)
    ax.set_xticklabels([], minor=False)
    ax.set_yticklabels([], minor=False)
    ax.set_xticklabels(periods, minor=True, fontsize=10, weight="bold", color="black")
    ax.set_yticklabels(ages, minor=True, fontsize=10, weight="bold", color="black")

    ax.set_xlabel('Period', fontsize=12, weight="bold", color="black")
    ax.set_ylabel('Age', fontsize=12, weight="bold", color="black")

    # Turn off all gridlines and tick marks
    ax.grid(False)
    ax.tick_params(which='both', length=0)

    # ─── Value annotations with white-bordered background ─────────────────────
    if show_values:
        fontsize = min(10, 140 // max(len(ages), len(periods)))
        for i in range(len(ages)):
            for j in range(len(periods)):
                value = data[i, j]
                ax.text(
                    j, i, f'{value:.2f}',
                    ha="center", va="center",
                    color="black",
                    fontsize=fontsize,
                    weight="bold",
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='#333333',
                        linewidth=1.5,
                        alpha=0.95
                    )
                )

    # ─── Highlight period boundaries ───────────────────────────────────────────
    if highlight_periods is not None:
        threshold_color = get_color('threshold', style_mode)
        ax.axvline(
            x=highlight_periods + 0.5,
            color=threshold_color,
            linewidth=3,
            alpha=0.9,
            zorder=4
        )

    # ─── Highlight diagonal trends ─────────────────────────────────────────────
    if highlight_diagonals:
        threshold_color = get_color('threshold', style_mode)
        for i in range(len(ages)):
            for j in range(len(periods)):
                if (i + j) in highlight_diagonals:
                    rect = Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False,
                        edgecolor=threshold_color,
                        linewidth=3,
                        zorder=5
                    )
                    ax.add_patch(rect)

    # ─── Secondary axis for cohorts ────────────────────────────────────────────
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(cohorts)))
    ax2.set_yticklabels(cohorts, fontsize=10, weight="bold", color="black")
    ax2.set_ylabel('Cohort', fontsize=12, weight="bold", color="black")

    # DISABLE GRID ON SECONDARY AXIS
    ax2.grid(False)

    # Remove spines
    for side in ["top"]:
        ax.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)

    # ─── Titles ────────────────────────────────────────────────────────────────
    apply_titles(fig, title, subtitle, n=n)

    # ─── Legend ────────────────────────────────────────────────────────────────
    if show_legend:
        legend_lines = [
            f"Ages: {ages[0]}–{ages[-1]}",
            f"Periods: {periods[0]}–{periods[-1]}",
            f"Layout: Rectangular"
        ]
        if n:
            legend_lines.append(f"N: {n:,}")


    # ─── Layout ────────────────────────────────────────────────────────────────

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    # Info line centred below the axes
    if show_legend:
        legend_below(ax, handles=[Patch(alpha=0)], labels=["  |  ".join(legend_lines)], fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# HEXAGONAL LAYOUT (LEXIS DIAGRAM)
# ══════════════════════════════════════════════════════════════════════════════

def _apc_hexagonal(
        data, ages, periods, cohorts, title, subtitle, n,
        show_values, show_legend, style_mode, vmin, vmax
):
    """Create hexagonal Lexis diagram for APC analysis."""

    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)

    # Flatten data if needed
    if isinstance(data, np.ndarray) and len(data.shape) == 2:
        data_flat = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_flat.append(data[i, j])
        data = data_flat

    # Auto-detect color range
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)

    # ─── Color selection based on style ────────────────────────────────────────
    if style_mode == "fiery":
        cmap_name = "YlOrRd"
    elif style_mode == "viridis":
        cmap_name = "viridis"
    elif style_mode == "sentiment":
        cmap_name = "RdYlGn"
    elif style_mode == "plainjane":
        cmap_name = "Blues"
    elif style_mode in ("reviewer3", "house"):
        cmap_name = "Greys"
    else:
        cmap_name = "RdBu_r"

    cmap_obj = matplotlib.colormaps[cmap_name]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ─── Hexagon parameters ────────────────────────────────────────────────────
    hex_radius = 0.5
    hex_width = np.sqrt(3) * hex_radius
    hex_height = 2 * hex_radius

    # ─── Create hexagonal grid ─────────────────────────────────────────────────
    texts = []
    idx = 0
    n_periods = len(periods)
    n_ages = len(ages)

    for age_idx in range(n_ages):
        for period_idx in range(n_periods):
            if idx >= len(data):
                break

            # Position hexagons diagonally
            x = period_idx * hex_width * 0.75
            y = -age_idx * hex_height * 0.75 - period_idx * hex_height * 0.375

            # Create hexagon
            hexagon = RegularPolygon(
                (x, y), 6,
                radius=hex_radius,
                orientation=0,
                facecolor=cmap_obj(norm(data[idx])),
                edgecolor='white',
                linewidth=1.5,
                **get_data_element_kwargs()
            )
            ax.add_patch(hexagon)

            # Store text position
            if show_values:
                texts.append((x, y, data[idx]))

            idx += 1

    # ─── Value annotations with white-bordered background ─────────────────────
    if show_values:
        fontsize = min(9, 120 // max(len(ages), len(periods)))
        for x, y, val in texts:
            ax.text(
                x, y, f'{val:.2f}',
                ha='center', va='center',
                fontsize=fontsize,
                color='black',
                weight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='white',
                    edgecolor='#333333',
                    linewidth=1.5,
                    alpha=0.95
                )
            )

    # ─── Axis properties ───────────────────────────────────────────────────────
    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')
    ax.margins(0.15)  # Add padding around hexagons

    # Remove any grid or tick marks
    ax.grid(False)

    # ─── Period labels (top) ───────────────────────────────────────────────────
    if texts:
        period_y = max([t[1] for t in texts]) + 1.5
        for i, period in enumerate(periods):
            x = i * hex_width * 0.75
            ax.text(
                x, period_y, str(period),
                ha='center', va='bottom',
                fontsize=10, weight='bold', color='black'
            )

        # "Period" label
        mid_x = (len(periods) - 1) * hex_width * 0.75 / 2
        ax.text(
            mid_x, period_y + 0.8, 'Period',
            ha='center', va='bottom',
            fontsize=14, weight='bold', color='black'
        )

        # ─── Age labels (left) ─────────────────────────────────────────────────
        for i, age in enumerate(ages):
            y = -i * hex_height * 0.75
            ax.text(
                -1.5, y, str(age),
                ha='right', va='center',
                fontsize=10, weight='bold', color='black'
            )

        # "Age" label (rotated)
        mid_y = -(len(ages) - 1) * hex_height * 0.75 / 2
        ax.text(
            -2.5, mid_y, 'Age',
            ha='center', va='center',
            fontsize=14, weight='bold', color='black',
            rotation=90
        )

        # ─── Cohort labels (bottom) ────────────────────────────────────────────
        cohort_y = min([t[1] for t in texts]) - 1.5
        for i, cohort in enumerate(cohorts):
            if i < len(periods):  # Only show as many as fit
                x = i * hex_width * 0.75
                ax.text(
                    x, cohort_y, str(cohort),
                    ha='center', va='top',
                    fontsize=10, weight='bold', color='black',
                    rotation=45
                )

        # "Cohort" label
        ax.text(
            mid_x, cohort_y - 1.5, 'Cohort',
            ha='center', va='top',
            fontsize=14, weight='bold', color='black'
        )

    # ─── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors='black')

    # ─── Titles ────────────────────────────────────────────────────────────────
    apply_titles(fig, title, subtitle, n=n)

    # ─── Legend ────────────────────────────────────────────────────────────────
    if show_legend:
        legend_lines = [
            f"Ages: {ages[0]}–{ages[-1]}",
            f"Periods: {periods[0]}–{periods[-1]}",
            f"Layout: Hexagonal"
        ]
        if n:
            legend_lines.append(f"N: {n:,}")

    # ─── Layout ────────────────────────────────────────────────────────────────

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    # Info line centred below the axes
    if show_legend:
        legend_below(ax, handles=[Patch(alpha=0)], labels=["  |  ".join(legend_lines)], fontsize=9)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))

    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLOTLY VERSION
# ══════════════════════════════════════════════════════════════════════════════

def apc_heatmap_interactive(
        data,
        ages,
        periods,
        cohorts,
        title=None,
        subtitle=None,
        style_mode="house",
        show_values=True,
):
    """
    Interactive Plotly APC heatmap (rectangular layout only).

    Parameters
    ----------
    data : array-like
        Matrix of values with shape (n_ages, n_periods).
    ages, periods, cohorts : array-like
        Labels for each axis.
    title, subtitle : str, optional
        Plot titles.
    style_mode : str, default "house"
        Color scheme.
    show_values : bool, default True
        Display values in cells.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plot object.
    """
    set_style(style_mode)

    data = np.array(data)

    # ─── Color selection ───────────────────────────────────────────────────────
    if style_mode == "fiery":
        colorscale = "YlOrRd"
    elif style_mode == "viridis":
        colorscale = "Viridis"
    elif style_mode == "sentiment":
        colorscale = "RdYlGn"
    elif style_mode == "plainjane":
        colorscale = "Blues"
    elif style_mode in ("reviewer3", "house"):
        colorscale = "Greys"
    else:
        colorscale = "RdBu_r"

    # ─── Create heatmap ────────────────────────────────────────────────────────
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=periods,
        y=ages,
        colorscale=colorscale,
        text=data if show_values else None,
        texttemplate='%{text:.2f}' if show_values else None,
        textfont={"size": 10, "color": "black"},
        hovertemplate='Period: %{x}<br>Age: %{y}<br>Value: %{z:.2f}<extra></extra>',
    ))

    # ─── Build title ───────────────────────────────────────────────────────────
    default_title = "Age-Period-Cohort Analysis"
    full_title = f"<b>{title or default_title}</b>"
    if subtitle:
        full_title += f"<br><span style='color:grey'>{subtitle}</span>"

    # ─── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=full_title,
        xaxis_title="Period",
        yaxis_title="Age",
        template="plotly_white",
        height=700,
        width=900,
    )

    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")

    return fig