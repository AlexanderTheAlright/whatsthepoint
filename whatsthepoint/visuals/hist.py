"""
hist.py: whatsthepoint Visualization Module
-------------------------------------------
Distribution histograms in the whatsthepoint style
modes (fiery, viridis, sentiment,
plainjane, reviewer3).

Features:
- Trace outline around distribution
- Threshold lines
- Two-variable coloring
- Interactive Plotly version
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm
from scipy.interpolate import make_interp_spline

# Correct relative import (visuals → utils)
from matplotlib.patches import Patch
from ..utils.style import legend_below
from ..utils.style import set_style, apply_titles, get_data_element_kwargs, get_color, format_tick_labels, generate_semantic_palette, get_hatches


# ══════════════════════════════════════════════════════════════════════════════
# HISTOGRAM FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def histogram(
    df,
    x,
    bins=20,
    title=None,
    subtitle=None,
    color=None,
    color_by=None,
    n=None,
    show_legend=True,
    style_mode="house",
    trace_outline=False,
    thresholds=None,
    figsize=(10, 6),
):
    """
    whatsthepoint styled histogram with dynamic theming.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing variable `x`.
    x : str
        Column name to plot.
    bins : int, default 20
        Number of histogram bins.
    title, subtitle : str, optional
        Titles displayed above plot.
    color : str or RGBA, optional
        Bar color; defaults to theme midtone.
    color_by : str, optional
        Column name to color bars by (creates two-variable visualization).
    n : int, optional
        Observation count for title suffix.
    show_legend : bool, default True
        Whether to display a legend box with variable info.
    style_mode : str, default "house"
        Visualization style preset. One of
        {'house','viridis','reviewer3','sentiment','fiery','plainjane'}.
    trace_outline : bool, default False
        Add smooth curve tracing the distribution top.
    thresholds : list, optional
        List of x-values to mark with vertical red dashed lines.
    figsize : tuple
        Figure size (width, height) in inches
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    vals = df[x].dropna()
    n = n or len(vals)

    # ─── Color logic ────────────────────────────────────────────────────────────
    if color_by and color_by in df.columns:
        # Two-variable coloring: split by color_by values
        color_var_vals = df[color_by].dropna().unique()
        colors_map = generate_semantic_palette({"positive": list(color_var_vals)}, mode=style_mode)
        hatch_map = dict(zip(color_var_vals, get_hatches(len(color_var_vals), style_mode)))

        # Plot stacked histogram
        for i, cv in enumerate(color_var_vals):
            subset = df[df[color_by] == cv][x].dropna()
            ax.hist(
                subset,
                bins=bins,
                color=colors_map[cv],
                hatch=hatch_map[cv],
                alpha=0.7,
                label=f"{color_by}={cv}",
                **get_data_element_kwargs()
            )
    else:
        # Single color
        color = color or (
            cm.plasma(0.7) if style_mode == "fiery"
            else matplotlib.colormaps["viridis"](0.6) if style_mode == "viridis"
            else cm.Greens(0.6) if style_mode == "sentiment"
            else cm.Blues(0.6) if style_mode == "plainjane"
            else "#000000"       # reviewer3: one group is black
        )

        # ─── Plot ───────────────────────────────────────────────────────────────
        counts, bin_edges, patches = ax.hist(
            vals,
            bins=bins,
            color=color,
            **get_data_element_kwargs()
        )

        # ─── Trace outline ──────────────────────────────────────────────────────
        if trace_outline:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Smooth curve with safety check for minimum points
            if len(bin_centers) >= 4:
                # Use spline smoothing for 4+ points
                k = min(3, len(bin_centers) - 1)
                spl = make_interp_spline(bin_centers, counts, k=k)
                x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
                y_smooth = spl(x_smooth)
                ax.plot(x_smooth, y_smooth, color="grey", linewidth=2, alpha=0.8, zorder=3)
            elif len(bin_centers) >= 2:
                # Simple line for 2-3 points
                ax.plot(bin_centers, counts, color="grey", linewidth=2, alpha=0.8, zorder=3, marker='o')

    # ─── Threshold lines (without annotated number blocks) ────────────────────────────────────────────────────────
    if thresholds:
        for thresh in thresholds:
            ax.axvline(thresh, color=get_color('threshold', style_mode), linestyle="--", linewidth=1.5, alpha=0.8, zorder=4)

    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Add 20% padding on top for better context (keep bottom at natural value)
    padding = y_range * 0.20
    ax.set_ylim(0, y_max + padding)

    # ─── Axis styling ───────────────────────────────────────────────────────────
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.set_ylabel("Frequency", fontsize=12, weight="bold", color="black")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # ─── Titles ────────────────────────────────────────────────────────────────
    apply_titles(fig, title, subtitle, n=n)

    # ─── Legend ────────────────────────────────────────────────────────────────
    # ─── Layout ────────────────────────────────────────────────────────────────
    # Format tick labels: bold and angled
    format_tick_labels(ax)

    # One legend centred below the axes
    if show_legend:
        if color_by:
            legend_below(ax)
        else:
            info = f"{x.replace('_', ' ').title()}  |  Bins: {bins}"
            if n:
                info += f"  |  N: {n:,}"
            legend_below(ax, handles=[Patch(alpha=0)], labels=[info])

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


def histogram_interactive(
    df,
    x,
    bins=20,
    title=None,
    subtitle=None,
    color_by=None,
    style_mode="house",
    thresholds=None,
):
    """Interactive Plotly histogram."""
    set_style(style_mode)

    vals = df[x].dropna()
    n = len(vals)

    fig = go.Figure()

    if color_by and color_by in df.columns:
        # Multi-variable histogram
        color_var_vals = df[color_by].dropna().unique()
        cmap = matplotlib.colormaps["viridis"]

        for i, cv in enumerate(color_var_vals):
            subset = df[df[color_by] == cv][x].dropna()
            rgba = cmap(i / len(color_var_vals))
            color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"

            fig.add_trace(go.Histogram(
                x=subset,
                nbinsx=bins,
                name=f"{color_by}={cv}",
                marker_color=color_str,
                opacity=0.7,
            ))
    else:
        # Single histogram
        cmap = matplotlib.colormaps["viridis"]
        rgba = cmap(0.6)
        color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"

        fig.add_trace(go.Histogram(
            x=vals,
            nbinsx=bins,
            marker_color=color_str,
            marker_line=dict(color="white", width=0.5),
        ))

    # Threshold lines
    if thresholds:
        for thresh in thresholds:
            thresh_color = get_color('threshold', style_mode)
            fig.add_vline(x=thresh, line=dict(color=thresh_color, width=2, dash="dash"),
                         annotation_text=f"{thresh}", annotation_position="top")

    # Build title
    default_title = f"Distribution of {x.replace('_', ' ').title()}"
    full_title = f"<b>{title or default_title}</b><br><span style='color:grey'>{subtitle or ''}</span>"

    fig.update_layout(
        title=full_title,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title="Frequency",
        template="plotly_white",
        height=600,
        width=900,
        barmode="overlay" if color_by else "group",
        showlegend=bool(color_by),
    )

    return fig
