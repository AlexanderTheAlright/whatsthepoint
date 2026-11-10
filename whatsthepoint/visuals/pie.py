"""
pie.py: whatsthepoint Visualization Module
------------------------------------------
Semantic pie chart for categorical distributions.

Features:
- Automatic category collapsing and labeling
- Semantic palettes from utils.style
- Annotated center-weighted text labels
- whatsthepoint consistent typography and title placement
- Interactive Plotly version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..utils.style import set_style, apply_titles, format_tick_labels


def pie(
    df,
    category_col,
    title=None,
    subtitle=None,
    style_mode="house",
    value_col=None,
    cmap=None,
    top_n=None,
    min_pct=0.03,
    label_map=None,
    color_map=None,
    annotate=True,
    figsize=(8, 8),
):
    """
    whatsthepoint flexible pie chart.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    category_col : str
        Column name with categorical values.
    title, subtitle : str
        Main and sub titles.
    style_mode : str
        {'house', 'viridis', 'reviewer3', 'sentiment', 'fiery', 'plainjane'}.
    value_col : str, optional
        Optional column for weights.
    cmap : str, optional
        Override colormap name.
    top_n : int, optional
        Show only top N categories; collapse others.
    min_pct : float
        Minimum share (0–1) before collapsing into "Other".
    label_map : dict, optional
        Mapping of raw → simplified category labels.
    color_map : dict, optional
        Custom category → color mapping.
    annotate : bool
        Whether to display labels on slices.
    figsize : tuple
        Figure size in inches.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Data preparation ---
    data = df.copy()
    if label_map:
        data[category_col] = data[category_col].map(label_map).fillna(data[category_col])

    if value_col and value_col in data.columns:
        counts = data.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    else:
        counts = data[category_col].value_counts(dropna=False)

    total_n = counts.sum()

    # Collapse small categories
    if top_n is not None:
        if len(counts) > top_n:
            top_items = counts.iloc[:top_n]
            other_sum = counts.iloc[top_n:].sum()
            counts = pd.concat([top_items, pd.Series({"Other": other_sum})])
    elif min_pct > 0:
        threshold = min_pct * total_n
        small_sum = counts[counts < threshold].sum()
        counts = counts[counts >= threshold]
        if small_sum > 0:
            counts["Other"] = counts.get("Other", 0) + small_sum

    percentages = (counts / total_n) * 100

    # --- Color palette ---
    if color_map:
        colors = [color_map.get(c, "#cccccc") for c in counts.index]
    else:
        style_colors = {
            "fiery": ["#C53E3E", "#E76F51", "#FFB703", "#8B1E3F", "#F4A261", "#BDBDBD"],
            "viridis": ["#440154", "#31688E", "#35B779", "#FDE725", "#BDBDBD"],
            "sentiment": ["#2E8B57", "#D62828", "#F77F00", "#70A37F", "#BDBDBD"],
            "plainjane": ["#4682B4", "#CD5C5C", "#9ACD32", "#9370DB", "#BDBDBD"],
            "reviewer3": ["#333333", "#777777", "#AAAAAA", "#DDDDDD", "#BDBDBD"],
            "house": ["#333333", "#777777", "#AAAAAA", "#DDDDDD", "#BDBDBD"],
        }
        palette = style_colors.get(style_mode, style_colors["viridis"])
        colors = [palette[i % len(palette)] for i in range(len(counts))]

    # --- Draw pie ---
    wedges, _ = ax.pie(
        counts,
        startangle=140,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )

    # Assign colors
    for w, c in zip(wedges, colors):
        w.set_facecolor(c)

    # --- Labels ---
    if annotate:
        for w, (cat, pct) in zip(wedges, percentages.items()):
            ang = (w.theta2 - w.theta1) / 2.0 + w.theta1
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))
            ax.text(
                x * 0.65,
                y * 0.65,
                f"{cat}\n{pct:.2f}%",
                ha="center",
                va="center",
                fontsize=11,
                weight="bold",
                color="black",
                bbox=dict(facecolor="white", edgecolor='#333333', linewidth=1.5, boxstyle="round,pad=0.35", alpha=0.95),
            )

    # --- Titles and footer ---
    apply_titles(fig, title, subtitle, n=int(total_n))
    fig.text(
        0.5, 0.04,
        f"(Overall N = {int(total_n):,})",
        ha="center", fontsize=12, color="grey",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, ax


def pie_interactive(
    df,
    category_col,
    title=None,
    subtitle=None,
    style_mode="house",
    value_col=None,
    top_n=None,
    min_pct=0.03,
    label_map=None,
):
    """Interactive Plotly pie chart."""
    set_style(style_mode)

    # --- Data preparation ---
    data = df.copy()
    if label_map:
        data[category_col] = data[category_col].map(label_map).fillna(data[category_col])

    if value_col and value_col in data.columns:
        counts = data.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    else:
        counts = data[category_col].value_counts(dropna=False)

    total_n = counts.sum()

    # Collapse small categories
    if top_n is not None:
        if len(counts) > top_n:
            top_items = counts.iloc[:top_n]
            other_sum = counts.iloc[top_n:].sum()
            counts = pd.concat([top_items, pd.Series({"Other": other_sum})])
    elif min_pct > 0:
        threshold = min_pct * total_n
        small_sum = counts[counts < threshold].sum()
        counts = counts[counts >= threshold]
        if small_sum > 0:
            counts["Other"] = counts.get("Other", 0) + small_sum

    percentages = (counts / total_n) * 100

    # --- Color palette ---
    style_colors = {
        "fiery": ["#C53E3E", "#E76F51", "#FFB703", "#8B1E3F", "#F4A261", "#BDBDBD"],
        "viridis": ["#440154", "#31688E", "#35B779", "#FDE725", "#BDBDBD"],
        "sentiment": ["#2E8B57", "#D62828", "#F77F00", "#70A37F", "#BDBDBD"],
        "plainjane": ["#4682B4", "#CD5C5C", "#9ACD32", "#9370DB", "#BDBDBD"],
        "reviewer3": ["#333333", "#777777", "#AAAAAA", "#DDDDDD", "#BDBDBD"],
        "house": ["#333333", "#777777", "#AAAAAA", "#DDDDDD", "#BDBDBD"],
    }
    palette = style_colors.get(style_mode, style_colors["viridis"])
    colors = [palette[i % len(palette)] for i in range(len(counts))]

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='inside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
    )])

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle or ''}</span><br><span style='font-size:12px;color:grey'>(Overall N = {int(total_n):,})</span>",
        template="plotly_white",
        height=600,
        width=700,
        showlegend=True,
    )

    return fig
