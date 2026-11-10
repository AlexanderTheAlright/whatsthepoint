"""
ice.py: whatsthepoint ICE Plots Module
---------------------------------------
Individual Conditional Expectation plots for model interpretability.

Features:
- ICE plots showing individual prediction curves
- Partial Dependence Plot (PDP) overlay
- Centered ICE plots
- Support for regression and classification models
- Plotly interactive counterpart
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Union, List


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
# STATIC VERSION
# ==============================================================================

def ice(
    model,
    X,
    feature,
    feature_name: Optional[str] = None,
    n_points: int = 50,
    sample_size: Optional[int] = None,
    pdp: bool = True,
    centered: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 6),
    output_path: Optional[str] = None,
):
    """
    Create Individual Conditional Expectation (ICE) plot.

    ICE plots show how predictions change for individual observations
    as a feature varies, revealing heterogeneous effects.

    Parameters
    ----------
    model : fitted model
        Model with .predict() method (sklearn-compatible).
    X : pd.DataFrame or np.ndarray
        Feature data used for predictions.
    feature : str or int
        Feature name (if X is DataFrame) or index (if array).
    feature_name : str, optional
        Display name for feature (if feature is an index).
    n_points : int, default 50
        Number of points to evaluate along feature range.
    sample_size : int, optional
        Random sample of observations to plot (default: all).
    pdp : bool, default True
        Overlay Partial Dependence Plot (average of ICE curves).
    centered : bool, default False
        Center ICE curves at first value (c-ICE plots).
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
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    >>> X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    >>> model = RandomForestRegressor(random_state=42)
    >>> model.fit(X_df, y)
    >>> ice(model, X_df, 'feature_0', title='ICE Plot for Feature 0')

    With centering:
    >>> ice(model, X_df, 'feature_0', centered=True, title='Centered ICE')
    """
    set_style(style_mode)

    # Convert to DataFrame if necessary
    if not isinstance(X, pd.DataFrame):
        if isinstance(feature, int):
            feature_name = feature_name or f"Feature {feature}"
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            feature = f"feature_{feature}"
        else:
            raise TypeError("If X is not a DataFrame, feature must be an integer index")
    else:
        feature_name = feature_name or feature

    # Sample observations if requested
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    # Get feature range
    feature_values = np.linspace(
        X[feature].min(),
        X[feature].max(),
        n_points
    )

    # Compute ICE curves
    ice_curves = []
    for idx in X_sample.index:
        X_temp = X_sample.loc[[idx]].copy()
        predictions = []

        for val in feature_values:
            X_temp[feature] = val
            pred = model.predict(X_temp)[0]
            predictions.append(pred)

        ice_curves.append(predictions)

    ice_curves = np.array(ice_curves)

    # Center curves if requested
    if centered:
        ice_curves = ice_curves - ice_curves[:, 0:1]

    # Compute PDP (average)
    pdp_values = ice_curves.mean(axis=0)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        ice_color = color_map(0.4)
        pdp_color = color_map(0.8)
    else:
        ice_color = 'steelblue'
        pdp_color = 'darkblue'

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot individual ICE curves
    for curve in ice_curves:
        ax.plot(feature_values, curve, color=ice_color, alpha=0.3, linewidth=0.8)

    # Plot PDP if requested
    if pdp:
        ax.plot(feature_values, pdp_values, color=pdp_color, linewidth=3,
               label='PDP (Average)', zorder=100)
        ax.legend(loc='best', frameon=True, facecolor='white',
                 edgecolor='grey', fontsize=10)

    # Labels
    ax.set_xlabel(feature_name.replace("_", " ").title(),
                 fontsize=12, weight='bold', color='black')
    ylabel = 'Centered Prediction' if centered else 'Prediction'
    ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='black')

    # Grid
    ax.set_axisbelow(True)

    # Spines
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


def ice_comparison(
    model,
    X,
    features: List[Union[str, int]],
    feature_names: Optional[List[str]] = None,
    n_points: int = 50,
    sample_size: Optional[int] = None,
    pdp: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: Optional[tuple] = None,
    output_path: Optional[str] = None,
):
    """
    Create multiple ICE plots in a grid for comparison.

    Parameters
    ----------
    model : fitted model
        Model with .predict() method.
    X : pd.DataFrame or np.ndarray
        Feature data.
    features : list
        List of feature names or indices to plot.
    feature_names : list, optional
        Display names for features.
    n_points : int, default 50
        Number of points per feature.
    sample_size : int, optional
        Random sample of observations.
    pdp : bool, default True
        Show PDP overlay.
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
    >>> ice_comparison(model, X_df, ['feature_0', 'feature_1', 'feature_2'],
    ...                title='ICE Comparison')
    """
    set_style(style_mode)

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = int(np.ceil(n_features / n_cols))

    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=130)
    fig.set_facecolor("white")

    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        ice_color = color_map(0.4)
        pdp_color = color_map(0.8)
    else:
        ice_color = 'steelblue'
        pdp_color = 'darkblue'

    # Convert to DataFrame if necessary
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    # Sample observations
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    for idx, (feature, ax) in enumerate(zip(features, axes)):
        # Get feature name
        if isinstance(feature, int):
            feat_col = X.columns[feature]
            feat_name = feature_names[idx] if feature_names else f"Feature {feature}"
        else:
            feat_col = feature
            feat_name = feature_names[idx] if feature_names else feature

        # Get feature range
        feature_values = np.linspace(
            X[feat_col].min(),
            X[feat_col].max(),
            n_points
        )

        # Compute ICE curves
        ice_curves = []
        for obs_idx in X_sample.index:
            X_temp = X_sample.loc[[obs_idx]].copy()
            predictions = []

            for val in feature_values:
                X_temp[feat_col] = val
                pred = model.predict(X_temp)[0]
                predictions.append(pred)

            ice_curves.append(predictions)

        ice_curves = np.array(ice_curves)

        # Plot ICE curves
        for curve in ice_curves:
            ax.plot(feature_values, curve, color=ice_color, alpha=0.3, linewidth=0.8)

        # Plot PDP
        if pdp:
            pdp_values = ice_curves.mean(axis=0)
            ax.plot(feature_values, pdp_values, color=pdp_color, linewidth=2.5,
                   label='PDP', zorder=100)

        # Formatting
        ax.set_facecolor("white")
        ax.set_xlabel(feat_name.replace("_", " ").title(),
                     fontsize=11, weight='bold', color='black')
        ax.set_ylabel('Prediction', fontsize=11, weight='bold', color='black')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide extra axes
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    # Add legend to first plot
    if pdp and n_features > 0:
        axes[0].legend(loc='best', frameon=True, facecolor='white',
                      edgecolor='grey', fontsize=9)

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

def ice_interactive(
    model,
    X,
    feature,
    feature_name: Optional[str] = None,
    n_points: int = 50,
    sample_size: Optional[int] = None,
    pdp: bool = True,
    centered: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive ICE plot using Plotly.

    Parameters
    ----------
    model : fitted model
        Model with .predict() method.
    X : pd.DataFrame or np.ndarray
        Feature data.
    feature : str or int
        Feature name or index.
    feature_name : str, optional
        Display name for feature.
    n_points : int, default 50
        Number of points to evaluate.
    sample_size : int, optional
        Random sample of observations.
    pdp : bool, default True
        Show PDP overlay.
    centered : bool, default False
        Center ICE curves.
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
    >>> ice_interactive(model, X_df, 'feature_0', title='Interactive ICE Plot')
    """
    import plotly.graph_objects as go

    # Convert to DataFrame if necessary
    if not isinstance(X, pd.DataFrame):
        if isinstance(feature, int):
            feature_name = feature_name or f"Feature {feature}"
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            feature = f"feature_{feature}"
        else:
            raise TypeError("If X is not a DataFrame, feature must be an integer index")
    else:
        feature_name = feature_name or feature

    # Sample observations if requested
    if sample_size is not None and sample_size < len(X):
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    # Get feature range
    feature_values = np.linspace(
        X[feature].min(),
        X[feature].max(),
        n_points
    )

    # Compute ICE curves
    ice_curves = []
    for idx in X_sample.index:
        X_temp = X_sample.loc[[idx]].copy()
        predictions = []

        for val in feature_values:
            X_temp[feature] = val
            pred = model.predict(X_temp)[0]
            predictions.append(pred)

        ice_curves.append(predictions)

    ice_curves = np.array(ice_curves)

    # Center curves if requested
    if centered:
        ice_curves = ice_curves - ice_curves[:, 0:1]

    # Compute PDP (average)
    pdp_values = ice_curves.mean(axis=0)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        ice_color = color_map(0.4)
        pdp_color = color_map(0.8)
    else:
        ice_color = 'steelblue'
        pdp_color = 'darkblue'

    # Convert colors to rgba strings
    if hasattr(ice_color, '__iter__'):
        ice_color_str = f"rgba({int(ice_color[0]*255)},{int(ice_color[1]*255)},{int(ice_color[2]*255)},0.3)"
        pdp_color_str = f"rgba({int(pdp_color[0]*255)},{int(pdp_color[1]*255)},{int(pdp_color[2]*255)},1.0)"
    else:
        ice_color_str = ice_color
        pdp_color_str = pdp_color

    # Create figure
    fig = go.Figure()

    # Add ICE curves
    for i, curve in enumerate(ice_curves):
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=curve,
            mode='lines',
            line=dict(color=ice_color_str, width=1),
            hovertemplate=f'Observation {i}<br>' +
                         f'{feature_name}: %{{x:.2f}}<br>' +
                         'Prediction: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))

    # Add PDP if requested
    if pdp:
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=pdp_values,
            mode='lines',
            line=dict(color=pdp_color_str, width=3),
            name='PDP (Average)',
            hovertemplate=f'{feature_name}: %{{x:.2f}}<br>' +
                         'Average Prediction: %{y:.2f}<extra></extra>',
        ))

    # Layout
    ylabel = 'Centered Prediction' if centered else 'Prediction'

    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{title or 'ICE Plot'}</b>" +
                 f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or 'ICE Plot'}</b>",
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
        xaxis_title=dict(text=feature_name.replace("_", " ").title(),
                        font=dict(size=12, color="black")),
        yaxis_title=dict(text=ylabel, font=dict(size=12, color="black")),
        plot_bgcolor="white",
        showlegend=pdp,
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="grey",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)",
                     tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)",
                     tickfont=dict(size=11, color="#333"))

    return fig
