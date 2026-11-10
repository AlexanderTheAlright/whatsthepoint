"""
flow.py: whatsthepoint Flow Diagrams
--------------------------------------
Alluvial and Sankey diagrams for flow visualization.

Features:
- Alluvial diagrams (categorical flows over stages)
- Sankey diagrams (node-link flows)
- Flow quantities and proportions
- Interactive Plotly versions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Dict, Tuple


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
# SANKEY DIAGRAM
# ==============================================================================

def sankey(
    source: List,
    target: List,
    value: List,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
):
    """
    Create a Sankey diagram showing flows between nodes.

    Sankey diagrams show the flow of quantities between nodes,
    with link width proportional to flow magnitude.

    Parameters
    ----------
    source : list
        List of source node indices.
    target : list
        List of target node indices.
    value : list
        List of flow values.
    labels : list of str, optional
        Node labels.
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
    >>> source = [0, 0, 1, 1, 2]
    >>> target = [2, 3, 2, 3, 4]
    >>> value = [10, 15, 20, 5, 25]
    >>> labels = ['A', 'B', 'C', 'D', 'E']
    >>> sankey(source, target, value, labels, title='Flow Diagram')
    """
    # Matplotlib Sankey is limited - recommend using interactive version
    # This is a placeholder static version
    set_style(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(0.5, 0.5, 'Sankey diagrams work best in interactive mode.\n' +
           'Use sankey_interactive() for full functionality.',
           ha='center', va='center', fontsize=14, color='gray',
           transform=ax.transAxes)

    ax.axis('off')

    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def sankey_interactive(
    source: List,
    target: List,
    value: List,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive Sankey diagram using Plotly.

    Parameters
    ----------
    source : list
        List of source node indices.
    target : list
        List of target node indices.
    value : list
        List of flow values.
    labels : list of str, optional
        Node labels.
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
    >>> sankey_interactive(source, target, value, labels,
    ...                     title='Interactive Sankey Diagram')
    """
    set_style(style_mode)

    # Get color scheme
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        n_nodes = len(set(source + target))
        node_colors = [color_map(i / n_nodes) for i in range(n_nodes)]
        node_colors = [f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.8)"
                      for c in node_colors]
    else:
        node_colors = ['steelblue'] * len(set(source + target))

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels if labels else [f"Node {i}" for i in range(len(set(source + target)))],
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(180,180,180,0.4)',
        )
    )])

    # Layout
    title_text = f"<b>{title or 'Sankey Diagram'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        font=dict(size=12, color='black'),
    )

    return fig


# ==============================================================================
# ALLUVIAL DIAGRAM
# ==============================================================================

def alluvial(
    df,
    stages: List[str],
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
):
    """
    Create an alluvial diagram showing flows across stages.

    Alluvial diagrams show how categories flow and change across
    multiple stages or time periods.

    Parameters
    ----------
    df : pd.DataFrame
        Data with one row per observation and columns for each stage.
    stages : list of str
        Column names representing stages (in order).
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
    ...     'stage1': ['A', 'A', 'B', 'B'],
    ...     'stage2': ['X', 'Y', 'X', 'Y'],
    ...     'stage3': ['M', 'M', 'N', 'N']
    ... })
    >>> alluvial(df, stages=['stage1', 'stage2', 'stage3'],
    ...          title='Flow Across Stages')
    """
    # Static alluvial is complex - recommend interactive version
    set_style(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(0.5, 0.5, 'Alluvial diagrams work best in interactive mode.\n' +
           'Use alluvial_interactive() for full functionality.',
           ha='center', va='center', fontsize=14, color='gray',
           transform=ax.transAxes)

    ax.axis('off')

    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def alluvial_interactive(
    df,
    stages: List[str],
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive alluvial diagram using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns for each stage.
    stages : list of str
        Column names representing stages.
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
    >>> alluvial_interactive(df, stages=['stage1', 'stage2', 'stage3'],
    ...                       title='Interactive Flow Diagram')
    """
    set_style(style_mode)

    # Build flows between consecutive stages
    all_categories = []
    for stage in stages:
        all_categories.extend(df[stage].unique())
    unique_categories = list(set(all_categories))

    # Create label mapping
    label_map = {cat: idx for idx, cat in enumerate(unique_categories)}

    # Build source, target, value lists
    source = []
    target = []
    value = []

    for i in range(len(stages) - 1):
        stage1 = stages[i]
        stage2 = stages[i + 1]

        # Count flows
        flows = df.groupby([stage1, stage2]).size().reset_index(name='count')

        for _, row in flows.iterrows():
            source.append(label_map[row[stage1]])
            target.append(label_map[row[stage2]])
            value.append(row['count'])

    # Use sankey for alluvial
    return sankey_interactive(
        source=source,
        target=target,
        value=value,
        labels=unique_categories,
        title=title,
        subtitle=subtitle,
        style_mode=style_mode
    )
