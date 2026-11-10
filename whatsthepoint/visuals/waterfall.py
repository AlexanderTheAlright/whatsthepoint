"""
waterfall.py: whatsthepoint Visualization Module
---------------------------------------------------------
Waterfall charts with trend arrows and styling options.

Features:
- Smoothed trend line with arrow
- Percentage change labels
- Common y-offset padding for aligned annotations
- Color intensity mapping
- Interactive Plotly version with the same features
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from ..utils.style import set_style, apply_titles, format_tick_labels


def waterfall(
        df,
        x,
        y,
        title=None,
        subtitle=None,
        style_mode="house",
        draw_trend=True,
        smooth_trend=True,
        annotate_percent=True,
        show_caps=True,
        pos_color="#006400",
        neg_color="#00008B",
        label_pad_frac=0.04,
        buffer_frac=0.20,
        min_bar_height=1e-4,
        figsize=(10, 7),
        dpi=150,
):
    """
    Waterfall plot with trend arrow and percentage labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with categorical x and numeric y.
    x, y : str
        Column names for categories and values.
    title, subtitle : str, optional
        Plot titles.
    style_mode : str
        Style theme ('viridis', 'seaborn', etc.).
    draw_trend : bool, default=True
        Draw smoothed trend line through bar midpoints.
    smooth_trend : bool, default=True
        Use cubic spline smoothing (requires scipy).
    annotate_percent : bool, default=True
        Show percentage change labels above/below bars.
    show_caps : bool, default=True
        Draw horizontal caps at cumulative values.
    pos_color, neg_color : str
        Base colors for positive/negative changes.
    label_pad_frac : float, default=0.04
        Padding for labels as fraction of y-range.
    buffer_frac : float, default=0.20
        Y-axis buffer as fraction of data range.
    min_bar_height : float, default=1e-4
        Threshold to avoid division by zero in percent calc.
    figsize : tuple, default=(10, 7)
        Figure dimensions.
    dpi : int, default=150
        Resolution.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    set_style(style_mode)

    # Prepare data
    df = df.sort_values(x).reset_index(drop=True)
    values = df[y].values
    deltas = np.diff(values)
    start_val = values[0]
    labels = [f"{p} → {n}" for p, n in zip(df[x].iloc[:-1], df[x].iloc[1:])]

    # Color intensity scaling
    vmax = np.max(deltas[deltas > 0], initial=1) if np.any(deltas > 0) else 1
    vmin = np.min(deltas[deltas < 0], initial=-1) if np.any(deltas < 0) else -1

    # Calculate y-range for padding
    cumulative_vals = [start_val]
    cum = start_val
    for d in deltas:
        cum += d
        cumulative_vals.append(cum)

    y_min, y_max = min(cumulative_vals), max(cumulative_vals)
    y_range = y_max - y_min if y_max != y_min else 1
    buffer = buffer_frac * y_range
    pad_units = label_pad_frac * (y_range + buffer)

    # Setup figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Colormaps
    pos_cmap = matplotlib.colormaps["Greens"]
    neg_cmap = matplotlib.colormaps["Blues_r"]

    # Draw bars with percentage annotations
    cumulative = start_val
    cumulative_tops = [start_val]

    for i, d in enumerate(deltas):
        is_pos = d >= 0

        # Color intensity (0.3 to 0.9 range)
        if is_pos and vmax != 0:
            norm_val = 0.3 + 0.6 * (d / vmax)
        elif not is_pos and vmin != 0:
            norm_val = 0.3 + 0.6 * (abs(d) / abs(vmin))
        else:
            norm_val = 0.6

        bar_col = pos_cmap(norm_val) if is_pos else neg_cmap(norm_val)

        # Draw bar
        ax.bar(i, d, bottom=cumulative, color=bar_col,
               edgecolor='black', linewidth=1.0, width=0.7, zorder=10)

        # Percentage change annotation (before updating cumulative)
        if annotate_percent:
            pct = 0 if abs(cumulative) < min_bar_height else 100 * d / cumulative
            lbl_txt = f"(+{pct:.0f}%)" if d >= 0 else f"({pct:.0f}%)"
            lbl_y = cumulative + d + pad_units * (1 if d >= 0 else -1)

            ax.text(i, lbl_y, lbl_txt,
                    ha='center',
                    va='bottom' if d >= 0 else 'top',
                    fontsize=9,
                    color=pos_color if d >= 0 else neg_color,
                    fontweight='bold',
                    zorder=11)

        cumulative += d
        cumulative_tops.append(cumulative)

    # Horizontal caps at cumulative values
    if show_caps:
        for i in range(len(cumulative_tops) - 1):
            ax.plot([i - 0.35, i + 0.35], [cumulative_tops[i], cumulative_tops[i]],
                    color='grey', linestyle=':', linewidth=1, zorder=5)

    # Smoothed trend line with arrow
    if draw_trend:
        mid_x = np.arange(len(deltas))
        mid_y = (np.array(cumulative_tops[:-1]) + np.array(cumulative_tops[1:])) / 2

        if smooth_trend:
            try:
                from scipy.interpolate import make_interp_spline
                x_smooth = np.linspace(mid_x.min(), mid_x.max(), 200)
                spl = make_interp_spline(mid_x, mid_y, k=min(2, len(mid_x) - 1))
                y_smooth = spl(x_smooth)
            except (ModuleNotFoundError, ImportError):
                x_smooth, y_smooth = mid_x, mid_y
        else:
            x_smooth, y_smooth = mid_x, mid_y

        # Draw trend line
        ax.plot(x_smooth, y_smooth, color='black', linewidth=2, zorder=12)

        # Arrow at end
        ax.annotate("", xy=(x_smooth[-1], y_smooth[-1]),
                    xytext=(x_smooth[-2], y_smooth[-2]),
                    arrowprops=dict(arrowstyle='-|>', color='black',
                                    linewidth=2, shrinkA=0, shrinkB=0),
                    zorder=13)

    # Styling
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right',
                       fontsize=10, color='grey')
    ax.set_ylabel(y.replace("_", " ").title(),
                  fontsize=12, fontweight='bold', color='grey')

    # Spine styling
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('black')

    ax.tick_params(axis='y', length=0, labelsize=10, colors='grey')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Set y-limits with buffer
    y_center = (y_min + y_max) / 2
    ax.set_ylim(y_center - y_range / 2 - buffer,
                y_center + y_range / 2 + buffer)

    # Titles
    apply_titles(fig, title,
                 subtitle)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    fig.tight_layout(rect=(0, 0, 1, 0.90 if subtitle else 0.94))

    return fig, ax


def waterfall_interactive(
        df,
        x,
        y,
        title=None,
        subtitle=None,
        draw_trend=True,
        smooth_trend=True,
        annotate_percent=True,
        pos_color="#2E8B57",
        neg_color="#4682B4",
        height=600,
        width=1000,
):
    """
    Interactive Plotly waterfall with trend arrow and percentage labels.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    x, y : str
        Column names.
    title, subtitle : str, optional
        Plot titles.
    draw_trend : bool, default=True
        Show smoothed trend line with arrow.
    smooth_trend : bool, default=True
        Use cubic spline (requires scipy).
    annotate_percent : bool, default=True
        Show percentage change labels.
    pos_color, neg_color : str
        Colors for positive/negative bars.
    height, width : int
        Figure dimensions.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    df = df.sort_values(x).reset_index(drop=True)
    values = df[y].values
    deltas = np.diff(values)
    start_val = values[0]

    # Calculate percentages
    percentages = []
    cum = start_val
    for d in deltas:
        pct = 0 if abs(cum) < 1e-4 else 100 * d / cum
        percentages.append(pct)
        cum += d

    # Prepare labels
    categories = df[x].tolist()
    transition_labels = [f"{p} → {n}" for p, n in zip(categories[:-1], categories[1:])]

    # Create waterfall
    measures = ["absolute"] + ["relative"] * len(deltas)
    display_values = [start_val] + list(deltas)

    # Text annotations with percentages
    if annotate_percent:
        text_labels = [f"{start_val:.2f}"] + [
            f"+{d:.2f}<br>(+{pct:.0f}%)" if d >= 0 else f"−{abs(d):.2f}<br>({pct:.0f}%)"
            for d, pct in zip(deltas, percentages)
        ]
    else:
        text_labels = [f"{v:.2f}" for v in display_values]

    all_labels = [categories[0]] + transition_labels

    fig = go.Figure()

    # Add waterfall
    fig.add_trace(go.Waterfall(
        x=all_labels,
        y=display_values,
        measure=measures,
        textposition="outside",
        text=text_labels,
        connector={"line": {"color": "grey", "dash": "dot", "width": 1}},
        increasing={"marker": {"color": pos_color}},
        decreasing={"marker": {"color": neg_color}},
        totals={"marker": {"color": "#4169E1"}},
    ))

    # Add trend line with arrow
    if draw_trend and len(deltas) > 1:
        cumulative = [start_val]
        cum = start_val
        for d in deltas:
            cum += d
            cumulative.append(cum)

        # Midpoints
        mid_x = np.arange(len(deltas))
        mid_y = (np.array(cumulative[:-1]) + np.array(cumulative[1:])) / 2

        if smooth_trend and len(mid_x) > 2:
            try:
                from scipy.interpolate import make_interp_spline
                x_smooth = np.linspace(0, len(deltas) - 1, 100)
                spl = make_interp_spline(mid_x, mid_y, k=min(2, len(mid_x) - 1))
                y_smooth = spl(x_smooth)
                # Map to categorical x positions
                x_positions = [all_labels[int(round(i)) + 1] for i in x_smooth]
            except (ModuleNotFoundError, ImportError):
                x_positions = [all_labels[i + 1] for i in mid_x]
                y_smooth = mid_y
        else:
            x_positions = [all_labels[i + 1] for i in mid_x]
            y_smooth = mid_y

        # Trend line
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_smooth,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))

        # Arrow annotation at end
        if len(x_positions) > 1:
            fig.add_annotation(
                x=x_positions[-1],
                y=y_smooth[-1],
                ax=x_positions[-2],
                ay=y_smooth[-2],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='black',
            )

    # Layout
    title_text = f"<b>{title or 'Waterfall Chart'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey; font-size:14px'>{subtitle}</span>"

    fig.update_layout(
        title=title_text,
        xaxis_title="",
        yaxis_title=y.replace("_", " ").title(),
        template="plotly_white",
        height=height,
        width=width,
        showlegend=False,
        font=dict(family="Arial, sans-serif"),
    )

    fig.update_xaxes(tickangle=-45)

    return fig