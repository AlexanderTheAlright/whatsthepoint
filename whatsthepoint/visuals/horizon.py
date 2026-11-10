"""
horizon.py: whatsthepoint Horizon Charts
-----------------------------------------
Space-efficient time series visualization using horizon charts.

Features:
- Horizon charts (layered area charts)
- Panel view for multiple series
- Negative/positive layering
- Space-efficient display
- Plotly interactive counterpart
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List


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
# HORIZON CHART
# ==============================================================================

def horizon(
    df,
    x: str,
    y: str,
    n_bands: int = 4,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (12, 3),
    output_path: Optional[str] = None,
):
    """
    Create a horizon chart for time series data.

    Horizon charts layer positive and negative deviations using color bands,
    allowing efficient visualization of many time series in limited space.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing time series.
    x : str
        Column name for x-axis (typically time).
    y : str
        Column name for y-axis values.
    n_bands : int, default 4
        Number of color bands to use.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme (sentiment recommended for pos/neg).
    figsize : tuple, default (12, 3)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'value': np.random.randn(100).cumsum()
    ... })
    >>> horizon(df, x='date', y='value', title='Horizon Chart Example')
    """
    set_style(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Extract data
    x_data = df[x].values
    y_data = df[y].values

    # Normalize to 0-1 range for each direction
    y_max = np.abs(y_data).max()
    if y_max == 0:
        y_max = 1

    # Create bands
    band_height = y_max / n_bands

    # Colors for positive and negative
    if style_mode == "sentiment":
        pos_colors = ['#d4edda', '#a8ddb5', '#7bccc4', '#43a2ca'][:n_bands]
        neg_colors = ['#f8d7da', '#f4a6a3', '#e75d6f', '#c73a3a'][:n_bands]
    else:
        color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
        if callable(color_map):
            pos_colors = [color_map(0.3 + 0.7 * i / n_bands) for i in range(n_bands)]
            neg_colors = [color_map(0.7 - 0.7 * i / n_bands) for i in range(n_bands)]
        else:
            pos_colors = ['#7bccc4'] * n_bands
            neg_colors = ['#f4a6a3'] * n_bands

    # Plot positive bands (layered)
    for i in range(n_bands):
        band_min = i * band_height
        band_max = (i + 1) * band_height

        y_band = np.clip(y_data - band_min, 0, band_height)
        ax.fill_between(x_data, 0, y_band, color=pos_colors[i],
                       alpha=0.7, linewidth=0)

    # Plot negative bands (layered, mirrored)
    for i in range(n_bands):
        band_min = i * band_height
        band_max = (i + 1) * band_height

        y_band = np.clip(-y_data - band_min, 0, band_height)
        ax.fill_between(x_data, 0, y_band, color=neg_colors[i],
                       alpha=0.7, linewidth=0)

    # Zero line
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=11, weight='bold', color='black')
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=11, weight='bold', color='black')
    ax.set_ylim(-band_height, band_height)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Format tick labels
    format_tick_labels(ax)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def horizon_panel(
    df,
    x: str,
    y: str,
    group: str,
    n_bands: int = 4,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: Optional[tuple] = None,
    output_path: Optional[str] = None,
):
    """
    Create a panel of horizon charts for multiple time series.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing multiple time series.
    x : str
        Column name for x-axis (time).
    y : str
        Column name for y-axis values.
    group : str
        Column name to group by (creates one chart per group).
    n_bands : int, default 4
        Number of color bands.
    title : str, optional
        Overall title.
    subtitle : str, optional
        Overall subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, optional
        Figure size (auto-calculated if None).
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> horizon_panel(df, x='date', y='value', group='category',
    ...               title='Multiple Time Series')
    """
    set_style(style_mode)

    groups = df[group].unique()
    n_groups = len(groups)

    if figsize is None:
        figsize = (12, 2 * n_groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=figsize, dpi=130, sharex=True)
    fig.set_facecolor("white")

    if n_groups == 1:
        axes = [axes]

    # Colors
    if style_mode == "sentiment":
        pos_colors = ['#d4edda', '#a8ddb5', '#7bccc4', '#43a2ca'][:n_bands]
        neg_colors = ['#f8d7da', '#f4a6a3', '#e75d6f', '#c73a3a'][:n_bands]
    else:
        color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
        if callable(color_map):
            pos_colors = [color_map(0.3 + 0.7 * i / n_bands) for i in range(n_bands)]
            neg_colors = [color_map(0.7 - 0.7 * i / n_bands) for i in range(n_bands)]
        else:
            pos_colors = ['#7bccc4'] * n_bands
            neg_colors = ['#f4a6a3'] * n_bands

    for ax, grp in zip(axes, groups):
        ax.set_facecolor("white")
        df_grp = df[df[group] == grp].sort_values(x)

        x_data = df_grp[x].values
        y_data = df_grp[y].values

        # Normalize
        y_max = np.abs(y_data).max()
        if y_max == 0:
            y_max = 1
        band_height = y_max / n_bands

        # Plot positive bands
        for i in range(n_bands):
            band_min = i * band_height
            y_band = np.clip(y_data - band_min, 0, band_height)
            ax.fill_between(x_data, 0, y_band, color=pos_colors[i], alpha=0.7)

        # Plot negative bands
        for i in range(n_bands):
            band_min = i * band_height
            y_band = np.clip(-y_data - band_min, 0, band_height)
            ax.fill_between(x_data, 0, y_band, color=neg_colors[i], alpha=0.7)

        # Zero line
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)

        # Styling
        ax.set_ylabel(str(grp), fontsize=10, weight='bold', rotation=0, ha='right', va='center')
        ax.set_ylim(-band_height, band_height)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # X-label on bottom plot only
    axes[-1].set_xlabel(x.replace("_", " ").title(), fontsize=11, weight='bold', color='black')

    # Format tick labels on all panels
    for ax in axes:
        format_tick_labels(ax)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


# ==============================================================================
# INTERACTIVE VERSION
# ==============================================================================

def horizon_interactive(
    df,
    x: str,
    y: str,
    group: Optional[str] = None,
    n_bands: int = 4,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive horizon chart using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing time series.
    x : str
        Column name for x-axis (time).
    y : str
        Column name for y-axis values.
    group : str, optional
        Column to group by for panel view.
    n_bands : int, default 4
        Number of color bands.
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
    >>> horizon_interactive(df, x='date', y='value',
    ...                      title='Interactive Horizon Chart')
    """
    set_style(style_mode)

    # For simplicity, create a line chart with fill
    # True horizon charts are complex in Plotly
    if group is None:
        fig = go.Figure()

        x_data = df[x]
        y_data = df[y]

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            fill='tozeroy',
            line=dict(color='steelblue', width=2),
            fillcolor='rgba(70, 130, 180, 0.3)',
            hovertemplate=f'{x}: %{{x}}<br>{y}: %{{y:.2f}}<extra></extra>',
        ))

        # Add zero line
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))

    else:
        # Panel view
        groups = df[group].unique()
        fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True,
                           subplot_titles=[str(g) for g in groups],
                           vertical_spacing=0.05)

        for idx, grp in enumerate(groups, 1):
            df_grp = df[df[group] == grp].sort_values(x)

            fig.add_trace(go.Scatter(
                x=df_grp[x],
                y=df_grp[y],
                mode='lines',
                fill='tozeroy',
                line=dict(color='steelblue', width=1.5),
                fillcolor='rgba(70, 130, 180, 0.3)',
                showlegend=False,
                hovertemplate=f'{x}: %{{x}}<br>{y}: %{{y:.2f}}<extra></extra>',
            ), row=idx, col=1)

            # Zero line
            fig.add_hline(y=0, line=dict(color='black', width=0.5, dash='dash'),
                         row=idx, col=1)

    # Layout
    title_text = f"<b>{title or 'Horizon Chart'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=600 if group is None else 150 * len(groups),
        margin=dict(t=90, b=50, l=60, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        plot_bgcolor="white",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)")

    return fig
