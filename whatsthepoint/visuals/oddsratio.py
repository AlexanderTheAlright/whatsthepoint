"""
oddsratio.py: whatsthepoint Odds Ratio Plots
---------------------------------------------
Forest plots for logistic regression odds ratios.

Features:
- Forest plots with confidence intervals
- Odds ratios and log odds
- Reference line at OR=1
- Multiple model comparison
- House-style formatting
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Union


try:
    from ..utils.style import apply_titles, COLORS_DICT, format_tick_labels, legend_below
except ImportError:
    def apply_titles(*args, **kwargs):
        pass
    COLORS_DICT = {'viridis': plt.cm.viridis}


def oddsratio(
    model=None,
    coef_df: Optional[pd.DataFrame] = None,
    exclude_intercept: bool = True,
    log_scale: bool = False,
    ci_level: float = 0.95,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    var_labels: Optional[dict] = None,
    style_mode: str = "house",
    figsize: tuple = (8, 6),
    output_path: Optional[str] = None,
):
    """
    Create forest plot of odds ratios from logistic regression.

    Parameters
    ----------
    model : RegressionModel, optional
        Fitted logistic regression model. Either model or coef_df required.
    coef_df : DataFrame, optional
        Coefficient dataframe with columns: term, estimate, conf.low, conf.high.
        Estimates should be log odds (will be exponentiated).
    exclude_intercept : bool, default True
        Exclude intercept from plot.
    log_scale : bool, default False
        Use log scale for x-axis (shows log odds instead of odds ratios).
    ci_level : float, default 0.95
        Confidence level (for display purposes only, intervals from model).
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label.
    var_labels : dict, optional
        Custom labels for variables. Example: {'age': 'Age (years)', 'educ': 'Education'}.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (8, 6)
        Figure size (width, height).
    output_path : str, optional
        Save plot to file.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    From fitted model:
    >>> from whatsthepoint.analyses.regress import logit
    >>> model = logit(df, 'hired', ['age', 'education', 'experience'])
    >>> oddsratio(model, title='Predictors of Being Hired')

    From coefficient dataframe:
    >>> coef_df = pd.DataFrame({
    ...     'term': ['age', 'education'],
    ...     'estimate': [0.05, 0.15],
    ...     'conf.low': [0.01, 0.05],
    ...     'conf.high': [0.09, 0.25]
    ... })
    >>> oddsratio(coef_df=coef_df)

    With custom labels:
    >>> oddsratio(model, var_labels={'age': 'Age (years)', 'educ': 'Years of Education'})
    """
    # Get coefficient data
    if model is not None:
        if not hasattr(model, 'get_tidy'):
            raise TypeError("model must have get_tidy() method")
        coef_df = model.get_tidy()
    elif coef_df is None:
        raise ValueError("Either model or coef_df must be provided")

    coef_df = coef_df.copy()

    # Exclude intercept if requested
    if exclude_intercept:
        coef_df = coef_df[~coef_df['term'].isin(['const', 'Intercept', 'intercept'])]

    if len(coef_df) == 0:
        raise ValueError("No coefficients to plot after filtering")

    # Convert log odds to odds ratios (unless log_scale=True)
    if not log_scale:
        coef_df['or'] = np.exp(coef_df['estimate'])
        if 'conf.low' in coef_df.columns:
            coef_df['or_ci_low'] = np.exp(coef_df['conf.low'])
            coef_df['or_ci_high'] = np.exp(coef_df['conf.high'])
        plot_col = 'or'
        ci_low_col = 'or_ci_low'
        ci_high_col = 'or_ci_high'
        ref_line = 1.0
    else:
        plot_col = 'estimate'
        ci_low_col = 'conf.low'
        ci_high_col = 'conf.high'
        ref_line = 0.0

    # Apply variable labels
    if var_labels:
        coef_df['label'] = coef_df['term'].map(lambda x: var_labels.get(x, x))
    else:
        coef_df['label'] = coef_df['term'].str.replace('_', ' ').str.title()

    # Sort by effect size (optional)
    coef_df = coef_df.sort_values(plot_col, ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    # Y positions
    y_pos = np.arange(len(coef_df))

    # Plot points
    ax.scatter(coef_df[plot_col], y_pos, s=100, color=main_color, zorder=3, alpha=0.8)

    # Plot confidence intervals
    if ci_low_col in coef_df.columns and ci_high_col in coef_df.columns:
        for i, (idx, row) in enumerate(coef_df.iterrows()):
            ci_low = row[ci_low_col]
            ci_high = row[ci_high_col]
            ax.plot([ci_low, ci_high], [i, i], color=main_color, linewidth=2, alpha=0.6, zorder=2)

    # Reference line
    ax.axvline(ref_line, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df['label'].values, fontsize=10)

    # X-axis label
    if xlabel is None:
        xlabel = 'Log Odds' if log_scale else 'Odds Ratio'
    ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')

    # Grid
    ax.set_axisbelow(True)

    # Log scale if not already in log
    if not log_scale:
        ax.set_xscale('log')
        # Set reasonable x limits
        all_vals = list(coef_df[plot_col].values)
        if ci_low_col in coef_df.columns:
            all_vals.extend(coef_df[ci_low_col].values)
            all_vals.extend(coef_df[ci_high_col].values)
        x_min = min(all_vals) * 0.8
        x_max = max(all_vals) * 1.2
        ax.set_xlim(x_min, x_max)

    # Title
    if title or subtitle:
        apply_titles(fig, title=title, subtitle=subtitle)

    # Layout
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))

    if output_path:
        from ..utils.style import save_figure
        save_figure(fig, output_path)

    return fig


def oddsratio_comparison(
    models: List,
    model_names: Optional[List[str]] = None,
    exclude_intercept: bool = True,
    log_scale: bool = False,
    ci_level: float = 0.95,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    var_labels: Optional[dict] = None,
    style_mode: str = "house",
    figsize: tuple = (10, 8),
    output_path: Optional[str] = None,
):
    """
    Compare odds ratios across multiple models.

    Parameters
    ----------
    models : list
        List of fitted logistic regression models.
    model_names : list of str, optional
        Names for each model. Defaults to 'Model 1', 'Model 2', etc.
    exclude_intercept : bool, default True
        Exclude intercept.
    log_scale : bool, default False
        Use log scale.
    ci_level : float, default 0.95
        Confidence level.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label.
    var_labels : dict, optional
        Variable labels.
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
    >>> model1 = logit(df, 'hired', ['age', 'education'])
    >>> model2 = logit(df, 'hired', ['age', 'education', 'experience'])
    >>> oddsratio_comparison([model1, model2],
    ...                      model_names=['Basic', 'Full'],
    ...                      title='Model Comparison')
    """
    # Get coefficient dataframes
    coef_dfs = []
    for model in models:
        if hasattr(model, 'get_tidy'):
            df = model.get_tidy()
        else:
            raise TypeError("All models must have get_tidy() method")

        if exclude_intercept:
            df = df[~df['term'].isin(['const', 'Intercept', 'intercept'])]

        coef_dfs.append(df)

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    # Get all unique terms
    all_terms = []
    for df in coef_dfs:
        for term in df['term'].unique():
            if term not in all_terms:
                all_terms.append(term)

    # Apply variable labels
    if var_labels:
        term_labels = {t: var_labels.get(t, t.replace('_', ' ').title()) for t in all_terms}
    else:
        term_labels = {t: t.replace('_', ' ').title() for t in all_terms}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(i / len(models)) for i in range(len(models))]
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    # Y positions
    n_terms = len(all_terms)
    spacing = 0.8 / len(models)
    y_base = np.arange(n_terms)

    # Plot each model
    for i, (coef_df, model_name) in enumerate(zip(coef_dfs, model_names)):
        y_offset = (i - len(models)/2 + 0.5) * spacing

        for j, term in enumerate(all_terms):
            term_data = coef_df[coef_df['term'] == term]

            if not term_data.empty:
                est = term_data['estimate'].iloc[0]
                val = np.exp(est) if not log_scale else est

                y_pos = y_base[j] + y_offset

                # Plot point
                ax.scatter(val, y_pos, s=80, color=colors[i], zorder=3, alpha=0.8, label=model_name if j == 0 else "")

                # Plot CI
                if 'conf.low' in term_data.columns and 'conf.high' in term_data.columns:
                    ci_low = term_data['conf.low'].iloc[0]
                    ci_high = term_data['conf.high'].iloc[0]

                    if not log_scale:
                        ci_low = np.exp(ci_low)
                        ci_high = np.exp(ci_high)

                    ax.plot([ci_low, ci_high], [y_pos, y_pos], color=colors[i], linewidth=1.5, alpha=0.6, zorder=2)

    # Reference line
    ref_line = 1.0 if not log_scale else 0.0
    ax.axvline(ref_line, color='grey', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Y-axis labels
    ax.set_yticks(y_base)
    ax.set_yticklabels([term_labels[t] for t in all_terms], fontsize=10)

    # X-axis label
    if xlabel is None:
        xlabel = 'Log Odds' if log_scale else 'Odds Ratio'
    ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')

    # Grid
    ax.set_axisbelow(True)

    # Log scale if not already
    if not log_scale:
        ax.set_xscale('log')

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

def oddsratio_interactive(
    model=None,
    coef_df=None,
    exclude_intercept=True,
    log_scale=False,
    ci_level=0.95,
    title=None,
    subtitle=None,
    xlabel=None,
    var_labels=None,
    style_mode="house",
):
    """Interactive odds ratio forest plot using Plotly."""
    import plotly.graph_objects as go

    # Get coefficient data
    if model is not None:
        if not hasattr(model, 'get_tidy'):
            raise TypeError("model must have get_tidy() method")
        coef_df = model.get_tidy()
    elif coef_df is None:
        raise ValueError("Either model or coef_df must be provided")

    coef_df = coef_df.copy()

    # Exclude intercept if requested
    if exclude_intercept:
        coef_df = coef_df[~coef_df['term'].isin(['const', 'Intercept', 'intercept'])]

    if len(coef_df) == 0:
        raise ValueError("No coefficients to plot after filtering")

    # Convert log odds to odds ratios
    if not log_scale:
        coef_df['or'] = np.exp(coef_df['estimate'])
        if 'conf.low' in coef_df.columns:
            coef_df['or_ci_low'] = np.exp(coef_df['conf.low'])
            coef_df['or_ci_high'] = np.exp(coef_df['conf.high'])
        plot_col = 'or'
        ci_low_col = 'or_ci_low'
        ci_high_col = 'or_ci_high'
        ref_line = 1.0
    else:
        plot_col = 'estimate'
        ci_low_col = 'conf.low'
        ci_high_col = 'conf.high'
        ref_line = 0.0

    # Apply variable labels
    if var_labels:
        coef_df['label'] = coef_df['term'].map(lambda x: var_labels.get(x, x))
    else:
        coef_df['label'] = coef_df['term'].str.replace('_', ' ').str.title()

    # Sort by effect size
    coef_df = coef_df.sort_values(plot_col, ascending=True)

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    if hasattr(main_color, '__iter__'):
        color_str = f"rgba({int(main_color[0]*255)},{int(main_color[1]*255)},{int(main_color[2]*255)},0.8)"
    else:
        color_str = main_color

    fig = go.Figure()

    # Add CI error bars
    if ci_low_col in coef_df.columns and ci_high_col in coef_df.columns:
        fig.add_trace(go.Scatter(
            x=coef_df[plot_col],
            y=coef_df['label'],
            error_x=dict(
                type='data',
                symmetric=False,
                array=coef_df[ci_high_col] - coef_df[plot_col],
                arrayminus=coef_df[plot_col] - coef_df[ci_low_col],
                color=color_str,
                thickness=2,
            ),
            mode='markers',
            marker=dict(color=color_str, size=10),
            hovertemplate='<b>%{y}</b><br>OR: %{x:.2f}<extra></extra>',
            showlegend=False,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=coef_df[plot_col],
            y=coef_df['label'],
            mode='markers',
            marker=dict(color=color_str, size=10),
            hovertemplate='<b>%{y}</b><br>OR: %{x:.2f}<extra></extra>',
            showlegend=False,
        ))

    # Reference line
    fig.add_vline(x=ref_line, line_dash="dash", line_color="grey", line_width=1.5)

    # Layout
    if xlabel is None:
        xlabel = 'Log Odds' if log_scale else 'Odds Ratio'

    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{title or 'Odds Ratios'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or 'Odds Ratios'}</b>",
            x=0.5,
            xanchor="center",
            yanchor="top",
            y=0.96,
        )

    fig.update_layout(
        template="plotly_white",
        height=max(400, len(coef_df) * 40),
        margin=dict(t=90, b=50, l=150, r=30),
        title=title_dict,
        xaxis_title=dict(text=xlabel, font=dict(size=12, color="black")),
        plot_bgcolor="white",
        xaxis_type="log" if not log_scale else "linear",
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(tickfont=dict(size=11, color="#333"))

    return fig
