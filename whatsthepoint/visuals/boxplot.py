"""
boxplot.py: whatsthepoint Visualization Module
--------------------------------------------------
Box plots and violin plots for distribution comparison:
- traditional box plots
- violin plots (density visualization)
- subplots for comparing multiple distributions
- consistent whatsthepoint styling
- Plotly interactive counterpart
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils.style import (
    set_style,
    generate_semantic_palette,
    apply_titles,
    get_data_element_kwargs,
    format_tick_labels,
    is_house,
)
from ..utils.layout import finish, sentence_label


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════

def boxplot(
    df,
    x=None,
    y=None,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="house",
    violin=False,
    orientation="vertical",
    showmeans=True,
    show_points=False,
    point_alpha=0.3,
    figsize=(10, 6),
):
    """
    whatsthepoint box plot or violin plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str, optional
        Grouping variable (categorical)
    y : str
        Value variable (continuous)
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
    violin : bool
        If True, create violin plot instead of box plot
    orientation : str
        'vertical' or 'horizontal'
    showmeans : bool
        Show mean markers
    show_points : bool
        Overlay individual data points
    point_alpha : float
        Transparency of points (if show_points=True)
    figsize : tuple
        Figure size (width, height) in inches

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Prepare data
    if x is None:
        # Single distribution
        data = [df[y].dropna()]
        labels = [y]
        single_var = True
    else:
        # Multiple distributions by group - filter out groups with no meaningful data
        def has_meaningful_data(vals):
            if vals.empty:
                return False
            if (vals.abs() < 1e-10).all():
                return False
            if len(vals) > 1 and vals.std() < 1e-10:
                return vals.abs().mean() > 1e-10
            return True

        groups = [g for g in df[x].unique() if has_meaningful_data(df[df[x] == g][y].dropna())]
        data = [df[df[x] == g][y].dropna() for g in groups]
        labels = list(groups)
        single_var = False

    # Generate palette
    if palette is None:
        groups_dict = {"positive": labels}
        palette = generate_semantic_palette(groups_dict, mode=style_mode)

    colors = [palette.get(label, matplotlib.colormaps["viridis"](0.6)) for label in labels]
    house = is_house(style_mode)
    if house:
        # the groups are named on the axis, so every box is white with a black
        # outline: the median and mean stay visible inside it
        colors = ["#FFFFFF"] * len(labels)

    # Create plot
    if violin:
        parts = ax.violinplot(
            data,
            positions=range(len(data)),
            showmeans=showmeans,
            showmedians=True,
            orientation=orientation
        )

        # Color violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(1.0 if house else 0.7)
            pc.set_linewidth(1.5)

        # Style the other parts
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(1.5)
    else:
        # Box plot
        bp = ax.boxplot(
            data,
            positions=range(len(data)),
            tick_labels=labels,
            patch_artist=True,
            showmeans=showmeans,
            meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5),
            boxprops=dict(linewidth=1.5, edgecolor='black'),
            flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black',
                            markersize=5, markeredgewidth=1.0),
            orientation=orientation
        )

        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            if not house:
                patch.set_alpha(0.7)

    # Overlay points if requested
    if show_points:
        for i, (d, color) in enumerate(zip(data, colors)):
            if len(d) > 0:
                # Add jitter
                x_vals = np.random.normal(i, 0.04, size=len(d))
                if orientation == "vertical":
                    ax.scatter(x_vals, d, alpha=point_alpha, s=20, color=color,
                              edgecolors='black', linewidth=0.5, zorder=3)
                else:
                    ax.scatter(d, x_vals, alpha=point_alpha, s=20, color=color,
                              edgecolors='black', linewidth=0.5, zorder=3)

    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    if orientation == "vertical":
        # Check if ACTUAL DATA appears to be percentage (0-100) or proportion (0-1)
        data_max = df[y].max()
        data_min = df[y].min()
        data_range = data_max - data_min

        # Use small tolerance for floating point comparison
        is_percentage = (data_max <= 100.0 + 1e-9 and data_min >= -1e-9)
        is_proportion = (data_max <= 1.0 + 1e-9 and data_min >= -1e-9)

        if is_percentage:
            # For percentage data: use data range for padding, then clamp to [0, 100]
            padding = data_range * 0.20
            new_min = max(0, data_min - padding)
            new_max = min(100.0, data_max + padding)
            if data_range < 20:  # Less than 20 percentage points
                center = (data_min + data_max) / 2
                new_min = max(0, center - 15)
                new_max = min(100.0, center + 15)
        elif is_proportion:
            # For proportion data: use data range for padding, then clamp to [0, 1]
            padding = data_range * 0.20
            new_min = max(0, data_min - padding)
            new_max = min(1.0, data_max + padding)
            if data_range < 0.2:
                center = (data_min + data_max) / 2
                new_min = max(0, center - 0.15)
                new_max = min(1.0, center + 0.15)
        else:
            # For non-percentage data: use autoscaled limits for padding
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            padding = y_range * 0.20
            new_min = y_min - padding
            new_max = y_max + padding

        if house:
            # a distribution has no zero baseline: the axis follows the data,
            # so the boxes fill the plot instead of sitting in a corner of it
            pad_y = (data_range or 1.0) * 0.08
            new_min, new_max = data_min - pad_y, data_max + pad_y
        ax.set_ylim(new_min, new_max)

    # Labels and styling
    name = sentence_label if house else (lambda s: s.replace("_", " ").title())
    if orientation == "vertical":
        ax.set_ylabel(name(y), fontsize=12, weight="bold", color="black")
        if x:
            ax.set_xlabel(name(x), fontsize=12, weight="bold", color="black")
        # Ensure tick positions match violin/box positions
        many = len(labels) > (8 if house else 5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45 if many else 0, ha='right' if many else 'center')
    else:
        ax.set_xlabel(name(y), fontsize=12, weight="bold", color="black")
        if x:
            ax.set_ylabel(name(x), fontsize=12, weight="bold", color="black")
        # Ensure tick positions match violin/box positions
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_type = "Violin" if violin else "Box"
    apply_titles(fig, title, subtitle, n=None if house else n)

    # Format tick labels: bold and angled
    if house:
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_fontweight("bold")
        finish(fig, legend=False)
        return fig, ax
    format_tick_labels(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# SUBPLOTS VERSION
# ══════════════════════════════════════════════════════════════════════════════

def boxplot_subplots(
    df,
    x,
    y,
    facet_col=None,
    facet_row=None,
    title=None,
    subtitle=None,
    palette=None,
    style_mode="house",
    violin=False,
    orientation="vertical",
    figsize=None,
    n=None,
):
    """
    Create subplots of box/violin plots.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str
        Grouping variable
    y : str or list
        Value variable(s)
    facet_col : str, optional
        Column to facet by
    facet_row : str, optional
        Row to facet by
    title : str, optional
        Overall title
    subtitle : str, optional
        Overall subtitle
    palette : dict, optional
        Color mapping
    style_mode : str
        whatsthepoint style mode
    violin : bool
        Create violin plots instead of box plots
    orientation : str
        'vertical' or 'horizontal'
    figsize : tuple, optional
        Figure size
    n : int, optional
        Sample size annotation

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    set_style(style_mode)

    # Determine layout
    if isinstance(y, list):
        n_plots = len(y)
        n_cols = 2 if n_plots > 1 else 1
        n_rows = int(np.ceil(n_plots / n_cols))
        plot_type = "multi_y"
        y_vars = y
    elif facet_col is not None or facet_row is not None:
        if facet_col and facet_row:
            col_vals = sorted(df[facet_col].unique())
            row_vals = sorted(df[facet_row].unique())
            n_cols = len(col_vals)
            n_rows = len(row_vals)
        elif facet_col:
            col_vals = sorted(df[facet_col].unique())
            n_cols = len(col_vals)
            n_rows = 1
            row_vals = [None]
        else:
            row_vals = sorted(df[facet_row].unique())
            n_rows = len(row_vals)
            n_cols = 1
            col_vals = [None]
        plot_type = "facet"
        y_vars = [y]
    else:
        raise ValueError("Must specify either multiple y variables or facet_col/facet_row")

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=130)
    fig.set_facecolor("white")

    # Ensure axes is 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Generate palette
    if palette is None:
        groups_dict = {"positive": df[x].unique().tolist()}
        palette = generate_semantic_palette(groups_dict, mode=style_mode)

    # Create subplots
    if plot_type == "multi_y":
        for idx, y_var in enumerate(y_vars):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            ax.set_facecolor("white")

            # Prepare data
            groups = df[x].unique()
            data = [df[df[x] == g][y_var].dropna() for g in groups]
            labels = list(groups)
            colors = [palette.get(label, matplotlib.colormaps["viridis"](0.6)) for label in labels]

            # Plot
            if violin:
                parts = ax.violinplot(data, positions=range(len(data)), showmeans=True,
                                     showmedians=True, orientation=orientation)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
                    pc.set_linewidth(1.5)
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                    if partname in parts:
                        parts[partname].set_edgecolor('black')
                        parts[partname].set_linewidth(1.5)
            else:
                bp = ax.boxplot(data, positions=range(len(data)), tick_labels=labels, patch_artist=True,
                               showmeans=True, orientation=orientation)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            if orientation == "vertical":
                ax.set_ylabel(y_var.replace("_", " ").title(), fontsize=11, weight="bold")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 0)
            else:
                ax.set_xlabel(y_var.replace("_", " ").title(), fontsize=11, weight="bold")
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(y_var.replace("_", " ").title(), fontsize=12, weight="bold", pad=10)

    else:  # facet
        for row_idx, row_val in enumerate(row_vals):
            for col_idx, col_val in enumerate(col_vals):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("white")

                # Filter data
                if facet_col and facet_row:
                    plot_df = df[(df[facet_col] == col_val) & (df[facet_row] == row_val)]
                    subplot_title = f"{col_val} | {row_val}"
                elif facet_col:
                    plot_df = df[df[facet_col] == col_val]
                    subplot_title = str(col_val)
                else:
                    plot_df = df[df[facet_row] == row_val]
                    subplot_title = str(row_val)

                groups = plot_df[x].unique()
                data = [plot_df[plot_df[x] == g][y].dropna() for g in groups]
                labels = list(groups)
                colors = [palette.get(label, matplotlib.colormaps["viridis"](0.6)) for label in labels]

                # Plot
                if violin:
                    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True,
                                         showmedians=True, orientation=orientation)
                    for i, pc in enumerate(parts['bodies']):
                        pc.set_facecolor(colors[i])
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)
                        pc.set_linewidth(1.5)
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                        if partname in parts:
                            parts[partname].set_edgecolor('black')
                            parts[partname].set_linewidth(1.5)
                else:
                    bp = ax.boxplot(data, positions=range(len(data)), tick_labels=labels, patch_artist=True,
                                   showmeans=True, orientation=orientation)
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                if orientation == "vertical":
                    ax.set_xticks(range(len(labels)))
                    if row_idx == n_rows - 1:
                        ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 0)
                    if col_idx == 0:
                        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11, weight="bold")
                else:
                    ax.set_yticks(range(len(labels)))
                    if col_idx == 0:
                        ax.set_yticklabels(labels)
                    if row_idx == n_rows - 1:
                        ax.set_xlabel(y.replace("_", " ").title(), fontsize=11, weight="bold")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_title(subplot_title, fontsize=12, weight="bold", pad=10)

    apply_titles(fig, title, subtitle, n=n)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════

def boxplot_interactive(
    df,
    x=None,
    y=None,
    title=None,
    subtitle=None,
    style_mode="house",
    violin=False,
    show_points=False,
):
    """
    whatsthepoint interactive box/violin plot.

    Parameters
    ----------
    df : pd.DataFrame
        Data for plotting
    x : str, optional
        Grouping variable
    y : str
        Value variable
    title : str, optional
        Plot title
    subtitle : str, optional
        Plot subtitle
    style_mode : str
        whatsthepoint style mode
    violin : bool
        Create violin plot instead of box plot
    show_points : bool
        Show individual data points

    Returns
    -------
    fig : plotly figure
    """
    set_style(style_mode)

    # Prepare data and colors
    if x is None:
        groups = [y]
        x_col = None
    else:
        # Filter out groups with no meaningful data
        def has_meaningful_data(vals):
            if vals.empty:
                return False
            if (vals.abs() < 1e-10).all():
                return False
            if len(vals) > 1 and vals.std() < 1e-10:
                return vals.abs().mean() > 1e-10
            return True

        groups = sorted([g for g in df[x].unique() if has_meaningful_data(df[df[x] == g][y].dropna())])
        x_col = x

    groups_dict = {"positive": groups}
    palette = generate_semantic_palette(groups_dict, mode=style_mode)

    fig = go.Figure()

    if x is None:
        # Single distribution
        color = palette.get(y, matplotlib.colormaps["viridis"](0.6))
        color_str = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]:.2f})"

        if violin:
            fig.add_trace(go.Violin(
                y=df[y],
                name=y,
                marker_color=color_str,
                box_visible=True,
                meanline_visible=True,
                points='all' if show_points else False,
            ))
        else:
            fig.add_trace(go.Box(
                y=df[y],
                name=y,
                marker_color=color_str,
                boxmean=True,
                boxpoints='all' if show_points else False,
            ))
    else:
        # Multiple distributions
        for group in groups:
            group_data = df[df[x] == group][y]
            color = palette.get(group, matplotlib.colormaps["viridis"](0.6))
            color_str = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]:.2f})"

            if violin:
                fig.add_trace(go.Violin(
                    y=group_data,
                    name=str(group),
                    marker_color=color_str,
                    box_visible=True,
                    meanline_visible=True,
                    points='all' if show_points else False,
                ))
            else:
                fig.add_trace(go.Box(
                    y=group_data,
                    name=str(group),
                    marker_color=color_str,
                    boxmean=True,
                    boxpoints='all' if show_points else False,
                ))

    # Layout
    title_dict = {}
    plot_type = "Violin" if violin else "Box"
    default_title = f"{plot_type} Plot: {y}"

    if subtitle:
        title_dict = dict(
            text=f"<b>{title or default_title}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or default_title}</b>",
            x=0.5,
            xanchor="center",
            yanchor="top",
            y=0.96,
        )

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=50, l=60, r=30),
        title=title_dict,
        yaxis_title=dict(
            text=y.replace("_", " ").title(),
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        ),
        xaxis_title=dict(
            text=x.replace("_", " ").title() if x else "",
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        ),
        plot_bgcolor="white",
        showlegend=(x is not None),
    )

    fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))

    return fig
