"""
density.py: whatsthepoint Visualization Module
--------------------------------------------------
Density plots including:
- kernel density estimation (KDE)
- ridgeline plots (joyplots)
- raincloud plots (combination of violin, box, and scatter)
- consistent whatsthepoint styling
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from ..utils.style import (
    set_style,
    generate_semantic_palette,
    apply_titles,
    get_data_element_kwargs,
    format_tick_labels,
    legend_below,
    is_house,
    BW_LINESTYLES,
)
from ..utils.layout import finish, sentence_label
from matplotlib.patches import Patch


# ══════════════════════════════════════════════════════════════════════════════
# KERNEL DENSITY PLOT
# ══════════════════════════════════════════════════════════════════════════════

def kde(
    df,
    x,
    group=None,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    fill=True,
    bw_adjust=1.0,
    figsize=(10, 6),
    legend_title=None,
):
    """
    Kernel density estimation plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Variable for density estimation
    group : str, optional
        Grouping variable for multiple densities
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    palette : dict, optional
        Color mapping
    n : int, optional
        Sample size annotation
    style_mode : str
        whatsthepoint style mode
    fill : bool
        Fill under curves
    bw_adjust : float
        Bandwidth adjustment factor
    figsize : tuple
        Figure size (width, height) in inches
    legend_title : str, optional
        Title for the group legend. None by default.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    house = is_house(style_mode)
    house_handles = []
    house_fills = ["#CCCCCC", "none", "#888888", "none"]

    def curve(data, kde_obj):
        # the curve runs three bandwidths past the data so it ends at the
        # baseline, not in a vertical wall at the smallest and largest value
        reach = 3 * np.sqrt(kde_obj.covariance[0, 0])
        return np.linspace(data.min() - reach, data.max() + reach, 500)

    if group is None:
        # Single density
        data = df[x].dropna()
        kde_obj = stats.gaussian_kde(data, bw_method=bw_adjust)
        x_range = curve(data, kde_obj)
        density = kde_obj(x_range)

        color = "#000000" if house else matplotlib.colormaps["viridis"](0.6)
        ax.plot(x_range, density, color=color, linewidth=2.5)
        if fill:
            ax.fill_between(x_range, density, alpha=0.3, color="#888888" if house else color, linewidth=0)
    else:
        # Multiple densities by group
        # Filter out groups with no meaningful data
        def has_meaningful_data(vals):
            if len(vals) < 2:  # Need at least 2 points for density
                return False
            if (vals.abs() < 1e-10).all():
                return False
            if vals.std() < 1e-10:
                return vals.abs().mean() > 1e-10
            return True

        all_groups = df[group].unique()
        groups = [g for g in all_groups if has_meaningful_data(df[df[group] == g][x].dropna())]

        if palette is None:
            groups_dict = {"positive": list(groups)}
            palette = generate_semantic_palette(groups_dict, mode=style_mode)

        for k, g in enumerate(groups):
            data = df[df[group] == g][x].dropna()
            if len(data) > 1:
                kde_obj = stats.gaussian_kde(data, bw_method=bw_adjust)
                x_range = curve(data, kde_obj)
                density = kde_obj(x_range)

                if house:
                    # black curves told apart by dash pattern; grey fill on alternate groups
                    dash = BW_LINESTYLES[k % len(BW_LINESTYLES)]
                    shade = house_fills[k % len(house_fills)] if fill else "none"
                    ax.plot(x_range, density, color="#000000", linewidth=2.5, linestyle=dash, label=str(g),
                            zorder=3)
                    if shade != "none":
                        ax.fill_between(x_range, density, color=shade, alpha=0.6, linewidth=0, zorder=1)
                    house_handles.append(Patch(facecolor=shade if shade != "none" else "white",
                                               edgecolor="#000000", linestyle=dash, linewidth=1.8,
                                               label=str(g)))
                    continue
                color = palette.get(g, matplotlib.colormaps["viridis"](0.6))
                ax.plot(x_range, density, color=color, linewidth=2.5, label=str(g))
                if fill:
                    ax.fill_between(x_range, density, alpha=0.3, color=color)


    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Add 20% padding on top for better context (keep bottom at 0)
    padding = y_range * 0.20
    ax.set_ylim(0 if house else y_min, y_max + padding)

    name = sentence_label if house else (lambda s: s.replace("_", " ").title())
    ax.set_xlabel(name(x), fontsize=12 if house else 14, weight="bold", color="black")
    ax.set_ylabel("Density", fontsize=12 if house else 14, weight="bold", color="black")
    ax.tick_params(axis='both', which='major', labelsize=10 if house else 12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_titles(fig, title, subtitle, n=None if house else n)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    if house:
        finish(fig, handles=house_handles or None,
               labels=[h.get_label() for h in house_handles] or None,
               legend=bool(group), legend_title=legend_title, pad="lrt")
        return fig, ax

    if group:
        legend_below(ax, title=legend_title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# RIDGELINE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def ridgeline(
    df,
    x,
    group,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    overlap=0.5,
    figsize=None,
):
    """
    Ridgeline plot (joyplot) for comparing distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Variable for density estimation
    group : str
        Grouping variable (creates separate ridges)
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    palette : dict, optional
        Color mapping
    n : int, optional
        Sample size annotation
    style_mode : str
        whatsthepoint style mode
    overlap : float
        Vertical overlap between ridges (0=no overlap, 1=full overlap)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    set_style(style_mode)

    groups = sorted(df[group].unique(), reverse=True)
    n_groups = len(groups)

    # Use provided figsize or calculate dynamically based on number of groups
    if figsize is None:
        figsize = (10, 2 + n_groups * 0.8)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if palette is None:
        groups_dict = {"positive": list(groups)}
        palette = generate_semantic_palette(groups_dict, mode=style_mode)

    # Get global x range
    x_min = df[x].min()
    x_max = df[x].max()
    x_range = np.linspace(x_min, x_max, 500)

    max_density = 0
    for g in groups:
        data = df[df[group] == g][x].dropna()
        if len(data) > 1:
            kde_obj = stats.gaussian_kde(data)
            density = kde_obj(x_range)
            max_density = max(max_density, density.max())

    # Vertical spacing
    spacing = max_density * (1 - overlap)

    for i, g in enumerate(groups):
        data = df[df[group] == g][x].dropna()
        if len(data) > 1:
            kde_obj = stats.gaussian_kde(data)
            density = kde_obj(x_range)

            # Shift vertically
            y_offset = i * spacing
            density_shifted = density + y_offset

            color = palette.get(g, matplotlib.colormaps["viridis"](0.6))
            ax.plot(x_range, density_shifted, color='black', linewidth=1.5)
            ax.fill_between(x_range, y_offset, density_shifted, alpha=0.7, color=color)

            # Label with white-bordered background
            ax.text(x_min - (x_max - x_min) * 0.02, y_offset + max_density * 0.3,
                   str(g), va='center', ha='right', fontsize=10, weight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_titles(fig, title, subtitle, n=n)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# RAINCLOUD PLOT
# ══════════════════════════════════════════════════════════════════════════════

def raincloud(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    point_size=20,
    point_alpha=0.5,
    figsize=(10, 6),
):
    """
    Raincloud plot: combination of violin, box, and scatter plots.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Grouping variable
    y : str
        Value variable
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    palette : dict, optional
        Color mapping
    n : int, optional
        Sample size annotation
    style_mode : str
        whatsthepoint style mode
    point_size : float
        Size of data points
    point_alpha : float
        Transparency of points

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    groups = df[x].unique()

    if palette is None:
        groups_dict = {"positive": list(groups)}
        palette = generate_semantic_palette(groups_dict, mode=style_mode)

    colors = [palette.get(g, matplotlib.colormaps["viridis"](0.6)) for g in groups]

    for i, (g, color) in enumerate(zip(groups, colors)):
        data = df[df[x] == g][y].dropna().values
        pos = i * 3  # Spacing between groups

        # 1. Violin plot (half, on the left)
        parts = ax.violinplot([data], positions=[pos], showmeans=False,
                              showmedians=False, orientation="vertical", widths=1.2)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)
            pc.set_linewidth(1.5)

        # 2. Box plot (narrow, in the middle)
        bp = ax.boxplot([data], positions=[pos + 0.6], widths=0.3,
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                      markeredgecolor='black', markersize=5),
                        medianprops=dict(color='black', linewidth=2),
                        boxprops=dict(facecolor=color, edgecolor='black',
                                     linewidth=1.5, alpha=0.8))

        # 3. Scatter points (jittered, on the right)
        x_jitter = np.random.normal(pos + 1.2, 0.08, size=len(data))
        ax.scatter(x_jitter, data, alpha=point_alpha, s=point_size,
                  color=color, edgecolors='black', linewidth=0.5, zorder=3)

    # Set x-ticks at the violin positions
    ax.set_xticks([i * 3 for i in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_titles(fig, title, subtitle, n=n)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# RYDER PLOT (ALIAS FOR RIDGELINE)
# ══════════════════════════════════════════════════════════════════════════════

def ryder(
    df,
    x,
    group,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    overlap=0.5,
):
    """
    Ryder plot (alias for ridgeline/joyplot) for comparing distributions.

    This is a convenience wrapper around ridgeline() with the same signature.
    Named after "ryder plots" for stacked density visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Variable for density estimation
    group : str
        Grouping variable (creates separate ridges)
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    palette : dict, optional
        Color mapping
    n : int, optional
        Sample size annotation
    style_mode : str
        whatsthepoint style mode
    overlap : float
        Vertical overlap between ridges (0=no overlap, 1=full overlap)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    return ridgeline(
        df=df,
        x=x,
        group=group,
        title=title,
        subtitle=subtitle,
        palette=palette,
        n=n,
        style_mode=style_mode,
        overlap=overlap,
    )


def ryder_interactive(
    df,
    x,
    group,
    title=None,
    subtitle=None,
    style_mode="house",
    overlap=0.5,
):
    """
    Interactive ryder plot (ridgeline) using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Variable for density estimation
    group : str
        Grouping variable (creates separate ridges)
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    style_mode : str
        whatsthepoint style mode
    overlap : float
        Vertical overlap between ridges

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    set_style(style_mode)

    groups = sorted(df[group].unique(), reverse=True)

    # Generate palette
    groups_dict = {"positive": list(groups)}
    palette = generate_semantic_palette(groups_dict, mode=style_mode)

    # Get global x range
    x_min = df[x].min()
    x_max = df[x].max()
    x_range = np.linspace(x_min, x_max, 500)

    # Calculate max density for spacing
    max_density = 0
    for g in groups:
        data = df[df[group] == g][x].dropna()
        if len(data) > 1:
            kde_obj = stats.gaussian_kde(data)
            density = kde_obj(x_range)
            max_density = max(max_density, density.max())

    spacing = max_density * (1 - overlap)

    fig = go.Figure()

    for i, g in enumerate(groups):
        data = df[df[group] == g][x].dropna()
        if len(data) > 1:
            kde_obj = stats.gaussian_kde(data)
            density = kde_obj(x_range)

            # Shift vertically
            y_offset = i * spacing
            density_shifted = density + y_offset

            color = palette.get(g, matplotlib.colormaps["viridis"](0.6))
            color_str = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.7)"

            # Add filled area
            fig.add_trace(go.Scatter(
                x=x_range,
                y=density_shifted,
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=color_str,
                line=dict(color='black', width=1.5),
                name=str(g),
                hovertemplate=f"<b>{g}</b><br>Value: %{{x:.2f}}<br>Density: %{{y:.2f}}<extra></extra>",
            ))

            # Add baseline for this group
            if i == 0:
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[y_offset] * len(x_range),
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False,
                    hoverinfo='skip',
                ))

    # Layout
    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{title or f'Ryder Plot: {x} by {group}'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02, xanchor="left", yanchor="top", y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or f'Ryder Plot: {x} by {group}'}</b>",
            x=0.5, xanchor="center", yanchor="top", y=0.96,
        )

    fig.update_layout(
        template="plotly_white",
        height=400 + len(groups) * 60,
        margin=dict(t=90, b=50, l=60, r=30),
        title=title_dict,
        xaxis_title=dict(text=x.replace("_", " ").title(),
                        font=dict(size=12, color="black")),
        yaxis_title=dict(text="",
                        font=dict(size=12, color="black")),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="white",
        hovermode='closest',
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSIONS
# ══════════════════════════════════════════════════════════════════════════════

def kde_interactive(
    df,
    x,
    group=None,
    title=None,
    subtitle=None,
    style_mode="house",
    fill=True,
):
    """Interactive KDE plot."""
    set_style(style_mode)

    fig = go.Figure()

    if group is None:
        data = df[x].dropna()
        color = matplotlib.colormaps["viridis"](0.6)
        color_str = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]:.2f})"

        fig.add_trace(go.Violin(
            x=data,
            name=x,
            marker_color=color_str,
            box_visible=False,
            meanline_visible=False,
            showlegend=False,
            orientation='h',
        ))
    else:
        groups = df[group].unique()
        groups_dict = {"positive": list(groups)}
        palette = generate_semantic_palette(groups_dict, mode=style_mode)

        for g in groups:
            data = df[df[group] == g][x].dropna()
            color = palette.get(g, matplotlib.colormaps["viridis"](0.6))
            color_str = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]:.2f})"

            fig.add_trace(go.Violin(
                x=data,
                name=str(g),
                marker_color=color_str,
                box_visible=False,
                meanline_visible=False,
                orientation='h',
            ))

    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{title or f'Density Plot: {x}'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02, xanchor="left", yanchor="top", y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or f'Density Plot: {x}'}</b>",
            x=0.5, xanchor="center", yanchor="top", y=0.96,
        )

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=50, l=60, r=30),
        title=title_dict,
        xaxis_title=dict(text=x.replace("_", " ").title(),
                        font=dict(size=12, color="black")),
        plot_bgcolor="white",
    )

    return fig
