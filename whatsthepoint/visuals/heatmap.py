"""
heatmap.py: whatsthepoint Visualization Module
----------------------------------------------
Matrix or correlation heatmap with full theme styling.

Features:
- Static matplotlib/seaborn version
- Interactive Plotly version
- Distribution heatmap for any data matrix
- Correlation heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from ..utils.style import set_style, apply_titles, get_continuous_cmap, format_tick_labels


def heatmap(df, title=None, subtitle=None, cmap=None, annot=False, style_mode="house"):
    """
    whatsthepoint correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe for correlation.
    title, subtitle : str, optional
        Plot titles.
    cmap : str, optional
        Custom colormap. If None, uses style_mode's continuous colormap.
    annot : bool, default False
        Show correlation values in cells.
    style_mode : str, default "house"
        Style theme: fiery (dark red heat), viridis, sentiment (RdYlGn),
        plainjane (RdBu), reviewer3 (grayscale).
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Create heatmap with crisp borders and proper spacing
    corr_data = df.corr()
    hm = sns.heatmap(corr_data, cmap=cmap, annot=False, fmt=".2f",
                     cbar_kws={"shrink": 0.8, "label": "Correlation"},
                     center=0 if "Rd" in cmap else None,
                     vmin=-1 if "Rd" in cmap else None, vmax=1 if "Rd" in cmap else None,
                     linewidths=2.5, linecolor='white',  # Crisp cell borders with increased width
                     ax=ax)

    # Fix colorbar label orientation (rotate to face towards the bar, not away)
    cbar = hm.collections[0].colorbar
    cbar.set_label("Correlation", rotation=270, labelpad=15, fontsize=11, weight='bold')

    # Add custom white-bordered annotations if requested
    if annot:
        for i in range(len(corr_data.index)):
            for j in range(len(corr_data.columns)):
                value = corr_data.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}',
                       ha='center', va='center',
                       fontsize=10, weight='bold', color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#333333', linewidth=1.5, alpha=0.95))

    # Bold axis labels with larger fonts for documents
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold', color='black')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold', color='black', rotation=0)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    apply_titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


def heatmap_interactive(df, title=None, subtitle=None, cmap=None, style_mode="house"):
    """
    Interactive Plotly correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    title, subtitle : str, optional
        Plot titles.
    cmap : str, optional
        Colormap name. If None, uses style_mode's continuous colormap.
    style_mode : str, default "house"
        Style theme.
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    corr = df.corr()

    # Create heatmap with annotations that have white backgrounds
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=cmap,
        text=corr.values,
        texttemplate='<b>%{text:.2f}</b>',  # Bold text values
        textfont={"size": 11, "color": "black", "family": "Arial Black"},  # Bold font
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>',
        xgap=2,  # Add gap between cells for border effect
        ygap=2,
    ))

    # Add white-bordered annotations using scatter traces with text
    for i, row_label in enumerate(corr.index):
        for j, col_label in enumerate(corr.columns):
            value = corr.iloc[i, j]
            fig.add_annotation(
                x=j, y=i,
                text=f"<b>{value:.2f}</b>",
                showarrow=False,
                font=dict(size=11, color="black", family="Arial Black"),
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="#333333",
                borderwidth=1.5,
                borderpad=4,
            )

    fig.update_layout(
        title=f"<b>{title or 'Correlation Heatmap'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
        height=600,
        width=700,
        xaxis=dict(side="bottom", tickfont=dict(size=11, color="black", family="Arial Black")),  # Bold x-axis
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="black", family="Arial Black")),  # Bold y-axis
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION HEATMAP (FOR ANY DATA MATRIX)
# ══════════════════════════════════════════════════════════════════════════════

def heatmap_distribution(
    data,
    title=None,
    subtitle=None,
    cmap=None,
    annot=True,
    fmt=".2f",
    style_mode="house",
    vmin=None,
    vmax=None,
    center=None,
    cbar_label="Value",
    figsize=(10, 8),
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
):
    """
    whatsthepoint distribution heatmap for any data matrix.

    Parameters
    ----------
    data : pd.DataFrame or array-like
        Input data matrix to visualize. Can be counts, percentages, or any numeric values.
    title, subtitle : str, optional
        Plot titles.
    cmap : str, optional
        Custom colormap. If None, uses style_mode's continuous colormap.
    annot : bool, default True
        Show numeric values in cells.
    fmt : str, default ".2f"
        Format string for annotations (e.g., ".0f" for integers, ".1%" for percentages).
    style_mode : str, default "house"
        Style theme: fiery (dark red heat), viridis, sentiment (RdYlGn),
        plainjane (RdBu), reviewer3 (grayscale).
    vmin, vmax : float, optional
        Min and max values for color scale. Auto-detected if None.
    center : float, optional
        Value to center the colormap at (useful for diverging colormaps).
    cbar_label : str, default "Value"
        Label for the colorbar.
    figsize : tuple, default (10, 8)
        Figure size (width, height).
    xlabel, ylabel : str, optional
        Axis labels.
    xticklabels, yticklabels : list, optional
        Custom tick labels. If None, uses data index/columns if available.

    Returns
    -------
    fig, ax : matplotlib figure and axes
        The generated plot objects.

    Examples
    --------
    >>> # Heatmap of counts
    >>> heatmap_distribution(count_matrix, title="Event Counts by Category",
    ...                      cbar_label="Count", fmt=".0f")

    >>> # Heatmap of percentages
    >>> heatmap_distribution(pct_matrix, title="Distribution (%)",
    ...                      cbar_label="Percentage", fmt=".1f")
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data_array = data.values
        if xticklabels is None:
            xticklabels = data.columns.tolist() if hasattr(data, 'columns') else None
        if yticklabels is None:
            yticklabels = data.index.tolist() if hasattr(data, 'index') else None
    else:
        data_array = np.array(data)

    # Auto-detect vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(data_array)
    if vmax is None:
        vmax = np.nanmax(data_array)

    # Create heatmap with crisp borders and proper spacing
    hm = sns.heatmap(
        data_array,
        cmap=cmap,
        annot=False,  # We'll add custom annotations
        fmt=fmt,
        cbar_kws={"shrink": 0.8, "label": cbar_label},
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=2.5,  # Crisp cell borders
        linecolor='white',
        xticklabels=xticklabels if xticklabels else False,
        yticklabels=yticklabels if yticklabels else False,
        ax=ax
    )

    # Fix colorbar label orientation (rotate to face towards the bar, not away)
    cbar = hm.collections[0].colorbar
    cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=11, weight='bold')

    # Add custom white-bordered annotations if requested
    if annot:
        fontsize = min(10, 140 // max(data_array.shape[0], data_array.shape[1]))
        for i in range(data_array.shape[0]):
            for j in range(data_array.shape[1]):
                value = data_array[i, j]
                if not np.isnan(value):
                    # Format the value according to fmt
                    if fmt.endswith('%'):
                        text = f'{value:{fmt}}'
                    elif fmt == '.0f':
                        text = f'{int(value):,}'  # Add thousands separator for integers
                    else:
                        text = f'{value:{fmt}}'

                    ax.text(j + 0.5, i + 0.5, text,
                           ha='center', va='center',
                           fontsize=fontsize, weight='bold', color='black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                    edgecolor='#333333', linewidth=1.5, alpha=0.95))

    # Bold axis labels
    if xticklabels:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, fontweight='bold',
                          color='black', rotation=45, ha='right')
    if yticklabels:
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, fontweight='bold',
                          color='black', rotation=0)

    # Set axis labels if provided
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='black')

    apply_titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


def heatmap_distribution_interactive(
    data,
    title=None,
    subtitle=None,
    cmap=None,
    fmt=".2f",
    style_mode="house",
    vmin=None,
    vmax=None,
    cbar_label="Value",
    xticklabels=None,
    yticklabels=None,
):
    """
    Interactive Plotly distribution heatmap for any data matrix.

    Parameters
    ----------
    data : pd.DataFrame or array-like
        Input data matrix to visualize.
    title, subtitle : str, optional
        Plot titles.
    cmap : str, optional
        Colormap name. If None, uses style_mode's continuous colormap.
    fmt : str, default ".2f"
        Format string for annotations (e.g., ".0f" for integers, ".1%" for percentages).
    style_mode : str, default "house"
        Style theme.
    vmin, vmax : float, optional
        Min and max values for color scale.
    cbar_label : str, default "Value"
        Label for the colorbar.
    xticklabels, yticklabels : list, optional
        Custom tick labels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure.
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data_array = data.values
        if xticklabels is None:
            xticklabels = data.columns.tolist() if hasattr(data, 'columns') else None
        if yticklabels is None:
            yticklabels = data.index.tolist() if hasattr(data, 'index') else None
    else:
        data_array = np.array(data)

    # Auto-detect vmin/vmax if not provided
    if vmin is None:
        vmin = np.nanmin(data_array)
    if vmax is None:
        vmax = np.nanmax(data_array)

    # Format values for display
    text_array = np.empty_like(data_array, dtype=object)
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            value = data_array[i, j]
            if not np.isnan(value):
                if fmt.endswith('%'):
                    text_array[i, j] = f'{value:{fmt}}'
                elif fmt == '.0f':
                    text_array[i, j] = f'{int(value):,}'
                else:
                    text_array[i, j] = f'{value:{fmt}}'
            else:
                text_array[i, j] = ''

    # Create heatmap with crisp borders
    fig = go.Figure(data=go.Heatmap(
        z=data_array,
        x=xticklabels if xticklabels else list(range(data_array.shape[1])),
        y=yticklabels if yticklabels else list(range(data_array.shape[0])),
        colorscale=cmap,
        text=text_array,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 10, "color": "black", "family": "Arial Black"},
        colorbar=dict(title=cbar_label),
        hovertemplate='%{y} | %{x}<br>' + cbar_label + ': %{z' + (fmt if not fmt.endswith('%') else '.2f') + '}<extra></extra>',
        xgap=2,  # Add gap between cells for border effect
        ygap=2,
        zmin=vmin,
        zmax=vmax,
    ))

    # Add white-bordered annotations
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            value = data_array[i, j]
            if not np.isnan(value):
                fig.add_annotation(
                    x=xticklabels[j] if xticklabels else j,
                    y=yticklabels[i] if yticklabels else i,
                    text=f"<b>{text_array[i, j]}</b>",
                    showarrow=False,
                    font=dict(size=10, color="black", family="Arial Black"),
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="#333333",
                    borderwidth=1.5,
                    borderpad=4,
                )

    fig.update_layout(
        title=f"<b>{title or 'Distribution Heatmap'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
        height=max(400, data_array.shape[0] * 40),
        width=max(500, data_array.shape[1] * 50),
        xaxis=dict(side="bottom", tickfont=dict(size=11, color="black", family="Arial Black")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="black", family="Arial Black")),
    )

    return fig
