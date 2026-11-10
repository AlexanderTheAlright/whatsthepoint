"""
pair.py: whatsthepoint Visualization Module
---------------------------------------------------
Pair plots for exploring relationships between multiple variables
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
    format_tick_labels,
)


def pair(
    df,
    vars=None,
    hue=None,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    diag_kind="hist",  # 'hist' or 'kde'
    alpha=0.6,
):
    """
    Pair plot showing relationships between multiple variables.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    vars : list, optional
        Variables to include (if None, uses all numeric columns)
    hue : str, optional
        Grouping variable for coloring
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
    diag_kind : str
        Type of plot on diagonal ('hist' or 'kde')
    alpha : float
        Point transparency

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    set_style(style_mode)

    if vars is None:
        vars = df.select_dtypes(include=[np.number]).columns.tolist()

    n_vars = len(vars)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(3*n_vars, 3*n_vars), dpi=130)
    fig.set_facecolor("white")

    if hue is None:
        color = matplotlib.colormaps["viridis"](0.6)
        groups = [None]
        colors = [color]
    else:
        groups = df[hue].unique()
        if palette is None:
            groups_dict = {"positive": list(groups)}
            palette = generate_semantic_palette(groups_dict, mode=style_mode)
        colors = [palette.get(g, matplotlib.colormaps["viridis"](0.6)) for g in groups]

    for i, var1 in enumerate(vars):
        for j, var2 in enumerate(vars):
            ax = axes[i, j] if n_vars > 1 else axes
            ax.set_facecolor("white")

            if i == j:
                # Diagonal: histogram or KDE
                if hue is None:
                    data = df[var1].dropna()
                    if diag_kind == "kde" and len(data) > 1:
                        kde_obj = stats.gaussian_kde(data)
                        x_range = np.linspace(data.min(), data.max(), 200)
                        density = kde_obj(x_range)
                        ax.plot(x_range, density, color=colors[0], linewidth=2)
                        ax.fill_between(x_range, density, alpha=0.3, color=colors[0])
                    else:
                        ax.hist(data, bins=20, color=colors[0], alpha=0.7, edgecolor='black')
                else:
                    for g, color in zip(groups, colors):
                        data = df[df[hue] == g][var1].dropna()
                        if diag_kind == "kde" and len(data) > 1:
                            kde_obj = stats.gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 200)
                            density = kde_obj(x_range)
                            ax.plot(x_range, density, color=color, linewidth=2, label=str(g))
                            ax.fill_between(x_range, density, alpha=0.3, color=color)
                        else:
                            ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black', label=str(g))

                ax.set_yticks([])
            else:
                # Off-diagonal: scatter
                if hue is None:
                    ax.scatter(df[var2], df[var1], color=colors[0], alpha=alpha, s=20, edgecolors='black', linewidth=0.3)
                else:
                    for g, color in zip(groups, colors):
                        subset = df[df[hue] == g]
                        ax.scatter(subset[var2], subset[var1], color=color, alpha=alpha, s=20,
                                  edgecolors='black', linewidth=0.3, label=str(g))

            # Labels
            if i == n_vars - 1:
                ax.set_xlabel(var2.replace("_", " ").title(), fontsize=9, weight="bold")
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(var1.replace("_", " ").title(), fontsize=9, weight="bold")
            else:
                ax.set_ylabel("")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Add legend if grouped
    if hue is not None:
        handles, labels = axes[0, 0].get_legend_handles_labels() if n_vars > 1 else axes.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      frameon=True, facecolor='white', edgecolor='grey', fontsize=10)

    # Format tick labels
    for ax_row in axes if n_vars > 1 else [axes]:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                format_tick_labels(ax)
        else:
            format_tick_labels(ax_row)

    apply_titles(fig, title, subtitle, n=n)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if subtitle else 0.97))
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════

def pair_interactive(
    df,
    vars=None,
    hue=None,
    title=None,
    subtitle=None,
    style_mode="house",
):
    """Interactive pair plot using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    set_style(style_mode)

    if vars is None:
        vars = df.select_dtypes(include=[np.number]).columns.tolist()

    n_vars = len(vars)

    # Create subplots
    fig = make_subplots(
        rows=n_vars,
        cols=n_vars,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
    )

    # Generate colors
    if hue is None:
        color = matplotlib.colormaps["viridis"](0.6)
        if hasattr(color, '__iter__'):
            colors = [f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.6)"]
        else:
            colors = [color]
        groups = [None]
    else:
        groups = df[hue].unique()
        groups_dict = {"positive": list(groups)}
        palette = generate_semantic_palette(groups_dict, mode=style_mode)
        colors = []
        for g in groups:
            color = palette.get(g, matplotlib.colormaps["viridis"](0.6))
            if hasattr(color, '__iter__'):
                colors.append(f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.6)")
            else:
                colors.append(color)

    # Plot
    for i, var1 in enumerate(vars):
        for j, var2 in enumerate(vars):
            if i == j:
                # Diagonal: histogram
                for idx, (g, color_val) in enumerate(zip(groups, colors)):
                    if g is None:
                        data = df[var1]
                    else:
                        data = df[df[hue] == g][var1]

                    fig.add_trace(go.Histogram(
                        x=data,
                        name=str(g) if g is not None else var1,
                        marker_color=color_val,
                        showlegend=(i == 0 and j == 0 and g is not None),
                        legendgroup=str(g) if g is not None else "data",
                    ), row=i+1, col=j+1)
            else:
                # Off-diagonal: scatter
                for idx, (g, color_val) in enumerate(zip(groups, colors)):
                    if g is None:
                        x_data = df[var2]
                        y_data = df[var1]
                    else:
                        subset = df[df[hue] == g]
                        x_data = subset[var2]
                        y_data = subset[var1]

                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        marker=dict(color=color_val, size=4, line=dict(width=0.5, color='grey')),
                        name=str(g) if g is not None else "",
                        showlegend=False,
                        legendgroup=str(g) if g is not None else "data",
                    ), row=i+1, col=j+1)

            # Labels
            if i == n_vars - 1:
                fig.update_xaxes(title_text=var2.replace("_", " ").title(), row=i+1, col=j+1)
            if j == 0:
                fig.update_yaxes(title_text=var1.replace("_", " ").title(), row=i+1, col=j+1)

    # Layout
    if subtitle:
        title_text = f"<b>{title or 'Pair Plot'}</b><br><span style='color:grey;font-size:14px;'>{subtitle}</span>"
    else:
        title_text = f"<b>{title or 'Pair Plot'}</b>"

    fig.update_layout(
        template="plotly_white",
        height=200 * n_vars,
        margin=dict(t=90, b=50, l=60, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        plot_bgcolor="white",
        showlegend=(hue is not None),
    )

    return fig
