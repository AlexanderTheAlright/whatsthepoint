"""
geographic.py: whatsthepoint Geographic Visualizations
-------------------------------------------------------
Spatial data visualization for choropleth maps and point maps.

Features:
- Choropleth maps (colored regions)
- Point maps (scatter on map)
- Hexbin density maps
- Interactive Plotly versions

Note: For advanced geographic features, install geopandas
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# plotly >= 5.24 renamed scatter_mapbox to scatter_map
_scatter_map = getattr(px, "scatter_map", None) or px.scatter_mapbox
_MAP_STYLE_KEY = "map_style" if hasattr(px, "scatter_map") else "mapbox_style"
from typing import Optional, Union


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
# POINT MAP
# ==============================================================================

def point_map(
    df,
    lat: str,
    lon: str,
    value: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
):
    """
    Create a point map with latitude/longitude data.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing coordinates.
    lat : str
        Column name for latitude.
    lon : str
        Column name for longitude.
    value : str, optional
        Column name for point values (colors points).
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
    ...     'lat': [40.7, 34.05, 41.88],
    ...     'lon': [-74.0, -118.24, -87.63],
    ...     'value': [10, 20, 15]
    ... })
    >>> point_map(df, lat='lat', lon='lon', value='value',
    ...           title='US Cities')
    """
    set_style(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Get colors
    if value is not None:
        color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
        values_norm = (df[value] - df[value].min()) / (df[value].max() - df[value].min())
        if callable(color_map):
            colors = [color_map(v) for v in values_norm]
        else:
            colors = [color_map] * len(df)
        sizes = 50 + 200 * values_norm
    else:
        color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
        if callable(color_map):
            colors = [color_map(0.6)] * len(df)
        else:
            colors = ['steelblue'] * len(df)
        sizes = [100] * len(df)

    # Plot points
    ax.scatter(df[lon], df[lat], c=colors, s=sizes, alpha=0.6,
              edgecolors='black', linewidth=1)

    # Styling
    ax.set_xlabel('Longitude', fontsize=12, weight='bold', color='black')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold', color='black')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def point_map_interactive(
    df,
    lat: str,
    lon: str,
    value: Optional[str] = None,
    hover_data: Optional[list] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    zoom: int = 3,
):
    """
    Interactive point map using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing coordinates.
    lat : str
        Column name for latitude.
    lon : str
        Column name for longitude.
    value : str, optional
        Column name for point values.
    hover_data : list, optional
        Additional columns to show on hover.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    zoom : int, default 3
        Initial zoom level.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> point_map_interactive(df, lat='lat', lon='lon', value='value',
    ...                        title='Interactive Map')
    """
    set_style(style_mode)

    # Create scatter mapbox
    if value is not None:
        fig = _scatter_map(
            df,
            lat=lat,
            lon=lon,
            color=value,
            size=value if value else None,
            hover_data=hover_data,
            zoom=zoom,
            height=600,
        )
    else:
        fig = _scatter_map(
            df,
            lat=lat,
            lon=lon,
            hover_data=hover_data,
            zoom=zoom,
            height=600,
        )

    # Update layout
    title_text = f"<b>{title or 'Point Map'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        **{_MAP_STYLE_KEY: "open-street-map"},
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
    )

    return fig


# ==============================================================================
# CHOROPLETH MAP
# ==============================================================================

def choropleth(
    locations,
    values,
    location_mode: str = "USA-states",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    colorbar_title: Optional[str] = None,
):
    """
    Create a choropleth map (colored regions).

    Note: This function requires geographic boundary data.
    For static choropleth, recommend using choropleth_interactive().

    Parameters
    ----------
    locations : list or array
        Location identifiers (state codes, country codes, etc.).
    values : list or array
        Values to map to colors.
    location_mode : str, default "USA-states"
        Type of locations ('USA-states', 'country names', etc.).
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    colorbar_title : str, optional
        Title for colorbar.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> states = ['CA', 'TX', 'FL', 'NY']
    >>> values = [100, 85, 90, 95]
    >>> choropleth(states, values, location_mode='USA-states',
    ...            title='Values by State')
    """
    # Choropleth works best with Plotly - redirect to interactive
    return choropleth_interactive(
        locations=locations,
        values=values,
        location_mode=location_mode,
        title=title,
        subtitle=subtitle,
        style_mode=style_mode,
        colorbar_title=colorbar_title
    )


def choropleth_interactive(
    locations,
    values,
    location_mode: str = "USA-states",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    colorbar_title: Optional[str] = None,
):
    """
    Interactive choropleth map using Plotly.

    Parameters
    ----------
    locations : list or array
        Location identifiers.
    values : list or array
        Values to map to colors.
    location_mode : str, default "USA-states"
        Type of locations.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    colorbar_title : str, optional
        Colorbar title.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> choropleth_interactive(['CA', 'TX', 'FL'], [100, 85, 90],
    ...                         title='Interactive Choropleth')
    """
    set_style(style_mode)

    # Create choropleth
    fig = go.Figure(data=go.Choropleth(
        locations=locations,
        z=values,
        locationmode=location_mode,
        colorscale='Viridis',
        colorbar_title=colorbar_title or "Value",
    ))

    # Layout
    title_text = f"<b>{title or 'Choropleth Map'}</b>"
    if subtitle:
        title_text += f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=30, l=30, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        geo=dict(scope='usa' if location_mode == 'USA-states' else 'world'),
    )

    return fig


# ==============================================================================
# HEXBIN MAP
# ==============================================================================

def hexbin_map(
    df,
    lat: str,
    lon: str,
    gridsize: int = 20,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (12, 8),
    output_path: Optional[str] = None,
):
    """
    Create a hexbin density map.

    Hexbin maps aggregate points into hexagonal bins, useful for
    visualizing point density across geographic regions.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing coordinates.
    lat : str
        Column name for latitude.
    lon : str
        Column name for longitude.
    gridsize : int, default 20
        Number of hexagons in x-direction.
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
    >>> hexbin_map(df, lat='lat', lon='lon', gridsize=25,
    ...            title='Density Map')
    """
    set_style(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Get colormap
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if not callable(color_map):
        color_map = plt.cm.viridis

    # Create hexbin
    hb = ax.hexbin(df[lon], df[lat], gridsize=gridsize, cmap=color_map,
                  edgecolors='white', linewidths=0.5, alpha=0.8)

    # Colorbar
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15, fontsize=11, weight='bold')

    # Styling
    ax.set_xlabel('Longitude', fontsize=12, weight='bold', color='black')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold', color='black')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig
