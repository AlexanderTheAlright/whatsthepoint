"""
residuals.py: whatsthepoint Residuals Diagnostic Plots
-------------------------------------------------------
Diagnostic plots for regression model assumptions.

Features:
- Residuals vs fitted values
- Q-Q plots for normality
- Scale-location plots
- Residuals vs leverage
- Combined diagnostic panel
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import warnings
from scipy import stats


try:
    from ..utils.style import apply_titles, COLORS_DICT, get_color, format_tick_labels, legend_below
except ImportError:
    def apply_titles(*args, **kwargs):
        pass


    def get_color(color_type='default', style_mode="house"):
        """Fallback get_color function with proper parameter handling."""
        color_map = {
            'reference': 'red',
            'line': 'blue',
            'warning': 'orange',
            'default': '#333333'
        }
        return color_map.get(color_type, color_map['default'])


    COLORS_DICT = {'viridis': plt.cm.viridis}


def residuals(
        model,
        plot_type: str = "all",
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        style_mode: str = "house",
        figsize: tuple = (12, 10),
        output_path: Optional[str] = None,
):
    """
    Create diagnostic residual plots for regression model.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    plot_type : str, default "all"
        Type of plot: "all", "fitted", "qq", "scale", "leverage".
        - "all": 2x2 panel with all diagnostic plots
        - "fitted": Residuals vs fitted values
        - "qq": Q-Q plot for normality
        - "scale": Scale-location plot (sqrt standardized residuals vs fitted)
        - "leverage": Residuals vs leverage
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (12, 10)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    All diagnostics:
    >>> from whatsthepoint.analyses.regress import ols
    >>> model = ols(df, 'income', ['education', 'age'])
    >>> residuals(model, title='Model Diagnostics')

    Just residuals vs fitted:
    >>> residuals(model, plot_type='fitted')

    Just Q-Q plot:
    >>> residuals(model, plot_type='qq')
    """
    # Extract model information
    if not hasattr(model, 'results'):
        raise TypeError("model must have results attribute (fitted regression model)")

    results = model.results
    fitted = results.fittedvalues
    resid = results.resid
    standardized_resid = resid / np.sqrt(results.scale)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    # Create plots based on type
    if plot_type == "all":
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # 1. Residuals vs Fitted
        _plot_residuals_fitted(axes[0], fitted, resid, main_color, style_mode)

        # 2. Q-Q Plot
        _plot_qq(axes[1], standardized_resid, main_color, style_mode)

        # 3. Scale-Location
        _plot_scale_location(axes[2], fitted, standardized_resid, main_color, style_mode)

        # 4. Residuals vs Leverage
        _plot_leverage(axes[3], results, standardized_resid, main_color, style_mode)

    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "fitted":
            _plot_residuals_fitted(ax, fitted, resid, main_color, style_mode)
        elif plot_type == "qq":
            _plot_qq(ax, standardized_resid, main_color, style_mode)
        elif plot_type == "scale":
            _plot_scale_location(ax, fitted, standardized_resid, main_color, style_mode)
        elif plot_type == "leverage":
            _plot_leverage(ax, results, standardized_resid, main_color, style_mode)
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}")

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def _plot_residuals_fitted(ax, fitted, resid, color, style_mode):
    """Plot residuals vs fitted values."""
    ax.scatter(fitted, resid, alpha=0.5, s=30, color=color, edgecolors='grey', linewidth=0.5)

    # Add horizontal line at 0
    ref_color = get_color('reference', style_mode)
    ax.axhline(0, color=ref_color, linestyle='--', linewidth=1.5, alpha=0.7)

    # Add lowess smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(resid, fitted, frac=0.3)
        line_color = get_color('line', style_mode)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=line_color, linewidth=2, alpha=0.8, label='Lowess')
    except Exception:
        pass

    ax.set_xlabel('Fitted Values', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('Residuals', fontsize=11, weight='bold', color='black')
    ax.set_title('Residuals vs Fitted', fontsize=12, weight='bold', pad=10)
    ax.set_axisbelow(True)


def _plot_qq(ax, standardized_resid, color, style_mode):
    """Q-Q plot for normality assessment."""
    # Use probplot properly
    (osm, osr), (slope, intercept, r) = stats.probplot(standardized_resid, dist="norm", plot=None)

    # Scatter plot
    ax.scatter(osm, osr, alpha=0.5, s=30,
               color=color, edgecolors='grey', linewidth=0.5)

    # Reference line
    ref_color = get_color('reference', style_mode)
    ax.plot(osm, slope * osm + intercept, '--', color=ref_color, linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Theoretical Quantiles', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('Standardized Residuals', fontsize=11, weight='bold', color='black')
    ax.set_title('Normal Q-Q Plot', fontsize=12, weight='bold', pad=10)
    ax.set_axisbelow(True)


def _plot_scale_location(ax, fitted, standardized_resid, color, style_mode):
    """Scale-location plot (spread-location plot)."""
    sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))

    ax.scatter(fitted, sqrt_abs_resid, alpha=0.5, s=30, color=color,
               edgecolors='grey', linewidth=0.5)

    # Add lowess smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(sqrt_abs_resid, fitted, frac=0.3)
        line_color = get_color('line', style_mode)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=line_color, linewidth=2, alpha=0.8)
    except Exception:
        pass

    ax.set_xlabel('Fitted Values', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('√|Standardized Residuals|', fontsize=11, weight='bold', color='black')
    ax.set_title('Scale-Location Plot', fontsize=12, weight='bold', pad=10)
    ax.set_axisbelow(True)


def _plot_leverage(ax, results, standardized_resid, color, style_mode):
    """Residuals vs leverage plot with Cook's distance."""
    try:
        from statsmodels.stats.outliers_influence import OLSInfluence

        influence = OLSInfluence(results)
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        ax.scatter(leverage, standardized_resid, alpha=0.5, s=30, color=color,
                   edgecolors='grey', linewidth=0.5)

        # Add Cook's distance contours
        x_range = np.linspace(0.001, max(leverage), 100)  # Start from 0.001 to avoid division by zero
        warning_color = get_color('warning', style_mode)
        for d in [0.5, 1.0]:
            y_pos = np.sqrt(d * len(standardized_resid) * (1 - x_range) / x_range)
            y_neg = -y_pos
            ax.plot(x_range, y_pos, '--', color=warning_color, alpha=0.5, linewidth=1)
            ax.plot(x_range, y_neg, '--', color=warning_color, alpha=0.5, linewidth=1)

        # Highlight high leverage points
        high_leverage = leverage > (2 * len(results.params) / len(leverage))
        if high_leverage.any():
            ax.scatter(leverage[high_leverage], standardized_resid[high_leverage],
                       s=80, facecolors='none', edgecolors=warning_color, linewidth=2)

    except Exception as e:
        warnings.warn(f"Could not compute leverage statistics: {e}")
        ax.text(0.5, 0.5, 'Leverage plot unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='grey')

    ax.set_xlabel('Leverage', fontsize=11, weight='bold', color='black')
    ax.set_ylabel('Standardized Residuals', fontsize=11, weight='bold', color='black')
    ax.set_title('Residuals vs Leverage', fontsize=12, weight='bold', pad=10)
    ax.set_axisbelow(True)


def residuals_histogram(
        model,
        bins: int = 30,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        style_mode: str = "house",
        figsize: tuple = (10, 6),
        output_path: Optional[str] = None,
):
    """
    Plot histogram of residuals with normal distribution overlay.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    bins : int, default 30
        Number of histogram bins.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 6)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> residuals_histogram(model, title='Distribution of Residuals')
    """
    results = model.results
    resid = results.resid
    standardized_resid = resid / np.sqrt(results.scale)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    n, bins_edges, patches = ax.hist(standardized_resid, bins=bins, density=True,
                                     alpha=0.7, color=main_color, edgecolor='grey')

    # Overlay normal distribution
    x_range = np.linspace(standardized_resid.min(), standardized_resid.max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range), 'r-', linewidth=2, alpha=0.8,
            label='Normal Distribution')

    # Labels
    ax.set_xlabel('Standardized Residuals', fontsize=12, weight='bold', color='black')
    ax.set_ylabel('Density', fontsize=12, weight='bold', color='black')

    # Grid
    ax.set_axisbelow(True)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # One legend centred below the axes
    legend_below(ax)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════

def residuals_interactive(
        model,
        plot_type="fitted",
        title=None,
        subtitle=None,
        style_mode="house",
):
    """Interactive residuals plot using Plotly."""
    import plotly.graph_objects as go

    # Extract model information
    if not hasattr(model, 'results'):
        raise TypeError("model must have results attribute")

    results = model.results
    fitted = results.fittedvalues
    resid = results.resid
    standardized_resid = resid / np.sqrt(results.scale)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    if hasattr(main_color, '__iter__') and len(main_color) >= 3:
        color_str = f"rgba({int(main_color[0] * 255)},{int(main_color[1] * 255)},{int(main_color[2] * 255)},0.6)"
    else:
        color_str = main_color

    fig = go.Figure()

    if plot_type == "fitted":
        # Residuals vs Fitted
        fig.add_trace(go.Scatter(
            x=fitted,
            y=resid,
            mode='markers',
            marker=dict(color=color_str, size=5, line=dict(width=0.5, color='grey')),
            hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

        # Add horizontal line at 0
        ref_color = get_color('reference', style_mode)
        # Handle color conversion for Plotly
        if isinstance(ref_color, str):
            ref_color_str = ref_color
        else:
            ref_color_str = 'red'
        fig.add_hline(y=0, line_dash="dash", line_color=ref_color_str, line_width=1.5)

        xlabel = "Fitted Values"
        ylabel = "Residuals"
        plot_title = title or "Residuals vs Fitted"

    elif plot_type == "qq":
        # Q-Q plot
        (osm, osr), (slope, intercept, r) = stats.probplot(standardized_resid, dist="norm", plot=None)

        fig.add_trace(go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            marker=dict(color=color_str, size=5, line=dict(width=0.5, color='grey')),
            hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

        # Reference line
        ref_color = get_color('reference', style_mode)
        if isinstance(ref_color, str):
            ref_color_str = ref_color
        else:
            ref_color_str = 'red'

        fig.add_trace(go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode='lines',
            line=dict(color=ref_color_str, dash='dash', width=1.5),
            showlegend=False,
        ))

        xlabel = "Theoretical Quantiles"
        ylabel = "Standardized Residuals"
        plot_title = title or "Normal Q-Q Plot"

    else:
        raise ValueError(f"plot_type must be 'fitted' or 'qq', got {plot_type}")

    # Layout
    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{plot_title}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{plot_title}</b>",
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
        xaxis_title=dict(text=xlabel, font=dict(size=12, color="black")),
        yaxis_title=dict(text=ylabel, font=dict(size=12, color="black")),
        plot_bgcolor="white",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))

    return fig