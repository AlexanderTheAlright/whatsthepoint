"""
feature.py: whatsthepoint Feature Importance Module
---------------------------------------------------------------
SHAP-style waterfall charts and feature importance visualizations.

Features:
- Waterfall charts showing cumulative contributions
- Traditional bar charts for feature importance
- Support for positive/negative impacts
- Works with SHAP values, permutation importance, etc.
- Plotly interactive counterpart
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Union, List


try:
    from ..utils.style import (
        set_style,
        generate_semantic_palette,
        apply_titles,
        COLORS_DICT,
        get_color,
        format_tick_labels,
    )
except ImportError:
    def set_style(*args, **kwargs):
        pass
    def apply_titles(*args, **kwargs):
        pass
    def get_color(*args, **kwargs):
        return '#333333'
    COLORS_DICT = {'viridis': plt.cm.viridis}


# ==============================================================================
# STATIC VERSION
# ==============================================================================

def feature_waterfall(
    importance: Union[Dict[str, float], pd.Series, pd.DataFrame],
    base_value: float = 0.0,
    prediction_value: Optional[float] = None,
    feature_col: Optional[str] = None,
    importance_col: Optional[str] = None,
    max_features: int = 10,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Create waterfall chart for feature importance/contributions.

    Shows cumulative contribution of features from a base value to
    final prediction, similar to SHAP waterfall plots.

    Parameters
    ----------
    importance : dict, pd.Series, or pd.DataFrame
        Feature importance values. If dict/Series, keys/index are feature names.
        If DataFrame, specify feature_col and importance_col.
    base_value : float, default 0.0
        Base value (e.g., average prediction, intercept).
    prediction_value : float, optional
        Final prediction value. If None, computed as base_value + sum(importance).
    feature_col : str, optional
        Column name for features (if importance is DataFrame).
    importance_col : str, optional
        Column name for importance values (if importance is DataFrame).
    max_features : int, default 10
        Maximum number of features to display (sorted by absolute importance).
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme (sentiment mode uses red/green for negative/positive).
    figsize : tuple, default (10, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    From dictionary:
    >>> importance = {'age': 0.15, 'income': 0.25, 'education': -0.10}
    >>> feature_waterfall(importance, base_value=0.5, title='SHAP Waterfall')

    From SHAP values:
    >>> import shap
    >>> explainer = shap.Explainer(model, X_train)
    >>> shap_values = explainer(X_test)
    >>> importance_dict = dict(zip(X_test.columns, shap_values.values[0]))
    >>> feature_waterfall(importance_dict, base_value=explainer.expected_value[0])
    """
    set_style(style_mode)

    # Convert importance to Series
    if isinstance(importance, dict):
        importance_series = pd.Series(importance)
    elif isinstance(importance, pd.DataFrame):
        if feature_col is None or importance_col is None:
            raise ValueError("Must specify feature_col and importance_col for DataFrame")
        importance_series = importance.set_index(feature_col)[importance_col]
    else:
        importance_series = importance

    # Sort by absolute importance and limit
    importance_series = importance_series.reindex(
        importance_series.abs().sort_values(ascending=False).index
    )
    if len(importance_series) > max_features:
        other_value = importance_series.iloc[max_features:].sum()
        importance_series = importance_series.iloc[:max_features]
        if abs(other_value) > 1e-6:
            importance_series['Other'] = other_value

    # Calculate prediction value if not provided
    if prediction_value is None:
        prediction_value = base_value + importance_series.sum()

    # Create cumulative values
    cumulative = [base_value]
    for val in importance_series.values:
        cumulative.append(cumulative[-1] + val)

    # Get colors for positive/negative
    if style_mode == "sentiment":
        pos_color = '#2ecc71'  # Green
        neg_color = '#e74c3c'  # Red
    else:
        color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
        if callable(color_map):
            pos_color = color_map(0.7)
            neg_color = color_map(0.3)
        else:
            pos_color = get_color('increasing', style_mode)
            neg_color = get_color('decreasing', style_mode)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    n_features = len(importance_series)
    positions = np.arange(n_features + 2)  # +2 for base and final

    # Plot base value
    ax.barh(0, base_value, color='lightgrey', edgecolor='black', linewidth=1.5,
           label='Base Value', height=0.6)
    ax.text(base_value/2, 0, f'{base_value:.2f}',
           ha='center', va='center', fontsize=10, weight='bold', color='black',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

    # Plot feature contributions
    for i, (feature, value) in enumerate(importance_series.items()):
        start = cumulative[i]
        color = pos_color if value > 0 else neg_color

        ax.barh(i + 1, value, left=start, color=color, edgecolor='black',
               linewidth=1.5, height=0.6, alpha=0.8)

        # Add value label with white-bordered background
        label_x = start + value/2
        label_text = f'{value:+.2f}'
        ax.text(label_x, i + 1, label_text,
               ha='center', va='center', fontsize=9, weight='bold', color='black',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

        # Add feature name
        ax.text(-0.02, i + 1, feature.replace("_", " ").title(),
               ha='right', va='center', fontsize=10, transform=ax.get_yaxis_transform())

    # Plot final prediction
    ax.barh(n_features + 1, prediction_value, color='steelblue',
           edgecolor='black', linewidth=1.5, height=0.6, alpha=0.9)
    ax.text(prediction_value/2, n_features + 1, f'{prediction_value:.2f}',
           ha='center', va='center', fontsize=10, weight='bold', color='black',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

    # Set y-axis
    y_labels = ['Base Value'] + list(importance_series.index) + ['Prediction']
    ax.set_yticks(positions)
    ax.set_yticklabels([''] * len(y_labels))  # Hide default labels (we added custom ones)
    ax.set_ylim(-0.5, n_features + 1.5)

    # Add labels at left for base and prediction
    ax.text(-0.02, 0, 'Base Value', ha='right', va='center',
           fontsize=10, weight='bold', transform=ax.get_yaxis_transform())
    ax.text(-0.02, n_features + 1, 'Prediction', ha='right', va='center',
           fontsize=10, weight='bold', transform=ax.get_yaxis_transform())

    # X-axis
    ax.set_xlabel('Value', fontsize=12, weight='bold', color='black')

    # Grid
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0.15, 0, 1, 0.94 if subtitle or title else 0.98))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def feature_importance_bar(
    importance: Union[Dict[str, float], pd.Series, pd.DataFrame],
    feature_col: Optional[str] = None,
    importance_col: Optional[str] = None,
    max_features: int = 15,
    horizontal: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Create traditional bar chart for feature importance.

    Parameters
    ----------
    importance : dict, pd.Series, or pd.DataFrame
        Feature importance values.
    feature_col : str, optional
        Column name for features (if importance is DataFrame).
    importance_col : str, optional
        Column name for importance values (if importance is DataFrame).
    max_features : int, default 15
        Maximum number of features to display.
    horizontal : bool, default True
        Use horizontal bars.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 8)
        Figure size.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model.fit(X_train, y_train)
    >>> importance = dict(zip(X_train.columns, model.feature_importances_))
    >>> feature_importance_bar(importance, title='Random Forest Feature Importance')
    """
    set_style(style_mode)

    # Convert importance to Series
    if isinstance(importance, dict):
        importance_series = pd.Series(importance)
    elif isinstance(importance, pd.DataFrame):
        if feature_col is None or importance_col is None:
            raise ValueError("Must specify feature_col and importance_col for DataFrame")
        importance_series = importance.set_index(feature_col)[importance_col]
    else:
        importance_series = importance

    # Sort and limit
    importance_series = importance_series.sort_values(ascending=False).head(max_features)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(0.3 + 0.6 * i / len(importance_series))
                 for i in range(len(importance_series))]
    else:
        colors = ['steelblue'] * len(importance_series)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if horizontal:
        bars = ax.barh(range(len(importance_series)), importance_series.values,
                      color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(importance_series)))
        ax.set_yticklabels([name.replace("_", " ").title()
                           for name in importance_series.index],
                          fontsize=10)
        ax.set_xlabel('Importance', fontsize=12, weight='bold', color='black')
        ax.invert_yaxis()

        # Add value labels with white borders
        for i, (bar, value) in enumerate(zip(bars, importance_series.values)):
            ax.text(value + 0.01 * importance_series.max(), i, f'{value:.2f}',
                   va='center', fontsize=9, weight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))
    else:
        bars = ax.bar(range(len(importance_series)), importance_series.values,
                     color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(importance_series)))
        ax.set_xticklabels([name.replace("_", " ").title()
                           for name in importance_series.index],
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Importance', fontsize=12, weight='bold', color='black')

        # Add value labels with white borders
        for bar, value in zip(bars, importance_series.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * importance_series.max(),
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9, weight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#333333', linewidth=1.5, alpha=0.95))

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


# ==============================================================================
# INTERACTIVE VERSION
# ==============================================================================

def feature_waterfall_interactive(
    importance: Union[Dict[str, float], pd.Series, pd.DataFrame],
    base_value: float = 0.0,
    prediction_value: Optional[float] = None,
    feature_col: Optional[str] = None,
    importance_col: Optional[str] = None,
    max_features: int = 10,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive waterfall chart using Plotly.

    Parameters
    ----------
    importance : dict, pd.Series, or pd.DataFrame
        Feature importance values.
    base_value : float, default 0.0
        Base value.
    prediction_value : float, optional
        Final prediction value.
    feature_col : str, optional
        Column name for features (if DataFrame).
    importance_col : str, optional
        Column name for importance (if DataFrame).
    max_features : int, default 10
        Maximum features to display.
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
    >>> feature_waterfall_interactive(importance_dict, base_value=0.5,
    ...                               title='Interactive SHAP Waterfall')
    """
    import plotly.graph_objects as go

    # Convert importance to Series
    if isinstance(importance, dict):
        importance_series = pd.Series(importance)
    elif isinstance(importance, pd.DataFrame):
        if feature_col is None or importance_col is None:
            raise ValueError("Must specify feature_col and importance_col for DataFrame")
        importance_series = importance.set_index(feature_col)[importance_col]
    else:
        importance_series = importance

    # Sort by absolute importance and limit
    importance_series = importance_series.reindex(
        importance_series.abs().sort_values(ascending=False).index
    )
    if len(importance_series) > max_features:
        other_value = importance_series.iloc[max_features:].sum()
        importance_series = importance_series.iloc[:max_features]
        if abs(other_value) > 1e-6:
            importance_series['Other'] = other_value

    # Calculate prediction value if not provided
    if prediction_value is None:
        prediction_value = base_value + importance_series.sum()

    # Prepare data for waterfall
    measure = ['absolute'] + ['relative'] * len(importance_series) + ['total']
    x_labels = ['Base Value'] + list(importance_series.index) + ['Prediction']
    y_values = [base_value] + list(importance_series.values) + [prediction_value]

    # Colors
    if style_mode == "sentiment":
        increasing_color = '#2ecc71'
        decreasing_color = '#e74c3c'
        totals_color = '#3498db'
    else:
        increasing_color = get_color('increasing', style_mode)
        decreasing_color = get_color('decreasing', style_mode)
        totals_color = 'steelblue'

    # Create waterfall
    fig = go.Figure(go.Waterfall(
        orientation='v',
        measure=measure,
        x=x_labels,
        y=y_values,
        textposition='outside',
        text=[f'{v:+.2f}' if m == 'relative' else f'{v:.2f}'
              for v, m in zip(y_values, measure)],
        increasing={'marker': {'color': increasing_color}},
        decreasing={'marker': {'color': decreasing_color}},
        totals={'marker': {'color': totals_color}},
        connector={'line': {'color': 'rgba(100,100,100,0.5)', 'width': 2}},
    ))

    # Layout
    if subtitle:
        title_text = f"<b>{title or 'Feature Importance Waterfall'}</b>" + \
                    f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"
    else:
        title_text = f"<b>{title or 'Feature Importance Waterfall'}</b>"

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=120, l=60, r=30),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        xaxis_title=dict(text="Feature", font=dict(size=12, color="black")),
        yaxis_title=dict(text="Value", font=dict(size=12, color="black")),
        plot_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(
        tickangle=-45,
        showgrid=True,
        gridcolor="rgba(180,180,180,0.3)",
        tickfont=dict(size=10, color="#333")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(180,180,180,0.3)",
        tickfont=dict(size=11, color="#333")
    )

    return fig


def feature_importance_bar_interactive(
    importance: Union[Dict[str, float], pd.Series, pd.DataFrame],
    feature_col: Optional[str] = None,
    importance_col: Optional[str] = None,
    max_features: int = 15,
    horizontal: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    style_mode: str = "house",
):
    """
    Interactive feature importance bar chart using Plotly.

    Parameters
    ----------
    importance : dict, pd.Series, or pd.DataFrame
        Feature importance values.
    feature_col : str, optional
        Column name for features (if DataFrame).
    importance_col : str, optional
        Column name for importance (if DataFrame).
    max_features : int, default 15
        Maximum features to display.
    horizontal : bool, default True
        Use horizontal bars.
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
    >>> feature_importance_bar_interactive(importance_dict,
    ...                                    title='Interactive Feature Importance')
    """
    import plotly.graph_objects as go

    # Convert importance to Series
    if isinstance(importance, dict):
        importance_series = pd.Series(importance)
    elif isinstance(importance, pd.DataFrame):
        if feature_col is None or importance_col is None:
            raise ValueError("Must specify feature_col and importance_col for DataFrame")
        importance_series = importance.set_index(feature_col)[importance_col]
    else:
        importance_series = importance

    # Sort and limit
    importance_series = importance_series.sort_values(ascending=False).head(max_features)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [f"rgba({int(color_map(0.3 + 0.6*i/len(importance_series))[0]*255)},"
                 f"{int(color_map(0.3 + 0.6*i/len(importance_series))[1]*255)},"
                 f"{int(color_map(0.3 + 0.6*i/len(importance_series))[2]*255)},0.8)"
                 for i in range(len(importance_series))]
    else:
        colors = ['steelblue'] * len(importance_series)

    # Create figure
    fig = go.Figure()

    if horizontal:
        fig.add_trace(go.Bar(
            y=[name.replace("_", " ").title() for name in importance_series.index],
            x=importance_series.values,
            orientation='h',
            marker=dict(color=colors, line=dict(color='black', width=1.5)),
            text=[f'{v:.2f}' for v in importance_series.values],
            textposition='outside',
            hovertemplate='%{y}<br>Importance: %{x:.2f}<extra></extra>',
        ))

        fig.update_yaxes(autorange="reversed")
        xaxis_title = "Importance"
        yaxis_title = None
    else:
        fig.add_trace(go.Bar(
            x=[name.replace("_", " ").title() for name in importance_series.index],
            y=importance_series.values,
            marker=dict(color=colors, line=dict(color='black', width=1.5)),
            text=[f'{v:.2f}' for v in importance_series.values],
            textposition='outside',
            hovertemplate='%{x}<br>Importance: %{y:.2f}<extra></extra>',
        ))

        xaxis_title = None
        yaxis_title = "Importance"

    # Layout
    if subtitle:
        title_text = f"<b>{title or 'Feature Importance'}</b>" + \
                    f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>"
    else:
        title_text = f"<b>{title or 'Feature Importance'}</b>"

    fig.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(t=90, b=120 if not horizontal else 50, l=150 if horizontal else 60, r=50),
        title=dict(text=title_text, x=0.5, xanchor="center"),
        xaxis_title=dict(text=xaxis_title, font=dict(size=12, color="black")) if xaxis_title else None,
        yaxis_title=dict(text=yaxis_title, font=dict(size=12, color="black")) if yaxis_title else None,
        plot_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(
        tickangle=-45 if not horizontal else 0,
        showgrid=True,
        gridcolor="rgba(180,180,180,0.3)",
        tickfont=dict(size=10, color="#333")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(180,180,180,0.3)",
        tickfont=dict(size=10, color="#333")
    )

    return fig
