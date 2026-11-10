"""
hierarchical.py: whatsthepoint Hierarchical Visualizations
-----------------------------------------------------------
Treemap and sunburst charts for hierarchical data.

Features:
- Treemap (nested rectangles)
- Sunburst (radial sectors)
- Support for multi-level hierarchies
- Color by category or value
- Plotly interactive counterparts
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Union


try:
    from ..utils.style import (
        set_style,
        generate_semantic_palette,
        apply_titles,
        COLORS_DICT,
        format_tick_labels,
    )
except ImportError:
    def set_style(*args, **kwargs):
        pass
    def apply_titles(*args, **kwargs):
        pass
    COLORS_DICT = {'viridis': plt.cm.viridis}


# ==============================================================================
# TREEMAP
# ==============================================================================

def treemap(
    df,
    hierarchy: List[str],
    values: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
):
    """
    Create a treemap visualization for hierarchical data.

    Treemaps display hierarchical data as nested rectangles, where the size
    of each rectangle represents a quantitative value.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing hierarchy and values.
    hierarchy : list of str
        Column names representing hierarchy levels (outermost to innermost).
    values : str
        Column name for values to determine rectangle sizes.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (12, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'category': ['A', 'A', 'B', 'B'],
    ...     'subcategory': ['A1', 'A2', 'B1', 'B2'],
    ...     'value': [10, 20, 15, 25]
    ... })
    >>> treemap(df, hierarchy=['category', 'subcategory'], values='value',
    ...         title='Sales by Category')
    """
    set_style(style_mode)

    # Use plotly for treemap as matplotlib treemap is complex
    # This is a simplified static version
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Group data by first level of hierarchy
    grouped = df.groupby(hierarchy[0])[values].sum().sort_values(ascending=False)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(i / len(grouped)) for i in range(len(grouped))]
    else:
        colors = [color_map] * len(grouped)

    # Simple squarified treemap algorithm
    total = grouped.sum()

    def squarify(sizes, x, y, width, height):
        """Simple squarified treemap layout."""
        rects = []
        if len(sizes) == 0:
            return rects

        if width >= height:
            # Split horizontally
            cum = 0
            for i, size in enumerate(sizes):
                w = (size / sum(sizes)) * width
                rects.append((x + cum, y, w, height))
                cum += w
        else:
            # Split vertically
            cum = 0
            for i, size in enumerate(sizes):
                h = (size / sum(sizes)) * height
                rects.append((x, y + cum, width, h))
                cum += h
        return rects

    # Create rectangles
    rects = squarify(grouped.values, 0, 0, 1, 1)

    for (x, y, w, h), (label, value), color in zip(rects, grouped.items(), colors):
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor='white', facecolor=color, alpha=0.8)
        ax.add_patch(rect)

        # Add label with white-bordered background if rectangle is large enough
        if w > 0.1 and h > 0.1:
            ax.text(x + w/2, y + h/2, f"{label}\n{value:.2f}",
                   ha='center', va='center', fontsize=10, weight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # No tick labels needed for treemap (axis is off)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def treemap_interactive(
    df,
    hierarchy: List[str],
    values: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive treemap using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing hierarchy and values.
    hierarchy : list of str
        Column names representing hierarchy levels.
    values : str
        Column name for values.
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
    >>> treemap_interactive(df, hierarchy=['category', 'subcategory'],
    ...                      values='value', title='Interactive Treemap')
    """
    import plotly.express as px

    set_style(style_mode)

    # Create path column for plotly
    df_plot = df.copy()

    # Build path for treemap
    path_cols = hierarchy.copy()

    # Create treemap
    fig = px.treemap(
        df_plot,
        path=path_cols,
        values=values,
        title=f"<b>{title or 'Treemap'}</b>",
    )

    # Update layout
    title_text = f"<b>{title or 'Treemap'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
    )

    return fig


# ==============================================================================
# SUNBURST
# ==============================================================================

def sunburst(
    df,
    hierarchy: List[str],
    values: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 10),
    output_path: Optional[str] = None,
):
    """
    Create a sunburst chart for hierarchical data.

    Sunburst charts display hierarchical data as nested radial sectors,
    useful for showing part-to-whole relationships across multiple levels.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing hierarchy and values.
    hierarchy : list of str
        Column names representing hierarchy levels.
    values : str
        Column name for values.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 10)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> sunburst(df, hierarchy=['category', 'subcategory'], values='value',
    ...          title='Sales Hierarchy')
    """
    set_style(style_mode)

    # Matplotlib sunburst is complex - use simple radial visualization
    fig, ax = plt.subplots(figsize=figsize, dpi=130, subplot_kw=dict(projection='polar'))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Group by first level
    grouped = df.groupby(hierarchy[0])[values].sum().sort_values(ascending=False)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(i / len(grouped)) for i in range(len(grouped))]
    else:
        colors = [color_map] * len(grouped)

    # Calculate angles
    total = grouped.sum()
    angles = (grouped / total) * 2 * np.pi

    # Plot wedges
    theta_start = 0
    for (label, value), angle, color in zip(grouped.items(), angles, colors):
        theta = np.linspace(theta_start, theta_start + angle, 100)
        r = np.ones_like(theta)

        ax.fill_between(theta, 0, r, color=color, alpha=0.8, edgecolor='white', linewidth=2)

        # Add label with white-bordered background
        mid_angle = theta_start + angle/2
        if angle > 0.2:  # Only label if segment is large enough
            ax.text(mid_angle, 0.7, label, ha='center', va='center',
                   fontsize=10, weight='bold', rotation=np.degrees(mid_angle)-90, color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

        theta_start += angle

    # Remove radial labels
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # No tick labels needed for sunburst (polar plot with no ticks)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def sunburst_interactive(
    df,
    hierarchy: List[str],
    values: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive sunburst chart using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing hierarchy and values.
    hierarchy : list of str
        Column names representing hierarchy levels.
    values : str
        Column name for values.
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
    >>> sunburst_interactive(df, hierarchy=['category', 'subcategory'],
    ...                       values='value', title='Interactive Sunburst')
    """
    import plotly.express as px

    set_style(style_mode)

    # Create path for sunburst
    df_plot = df.copy()

    # Create sunburst
    fig = px.sunburst(
        df_plot,
        path=hierarchy,
        values=values,
        title=f"<b>{title or 'Sunburst Chart'}</b>",
    )

    # Update layout
    title_text = f"<b>{title or 'Sunburst Chart'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
    )

    return fig
