"""
margins.py: whatsthepoint Marginal Effects Plots
-------------------------------------------------
Visualize predicted values and marginal effects from regression models.

Features:
- Margin plots showing predicted values across predictor ranges
- Confidence intervals
- Multiple predictor comparison
- Interaction effects visualization
- Support for both continuous and categorical predictors
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Union
import warnings


try:
    from ..utils.style import apply_titles, COLORS_DICT, format_tick_labels, legend_below
except ImportError:
    def apply_titles(*args, **kwargs):
        pass


    COLORS_DICT = {'viridis': plt.cm.viridis}


def margins(
        model,
        variable: str,
        at_values: Optional[Union[List, np.ndarray]] = None,
        other_vars: Optional[dict] = None,
        ci: bool = True,
        ci_level: float = 0.95,
        n_points: int = 50,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        style_mode: str = "house",
        figsize: tuple = (10, 6),
        output_path: Optional[str] = None,
):
    """
    Plot marginal effects (predicted values) from a regression model.

    Shows how the predicted outcome changes across values of a focal variable,
    holding other variables at specified values (typically means).

    Parameters
    ----------
    model : RegressionModel or MultilevelModel
        Fitted regression model from whatsthepoint.analyses.regress.
    variable : str
        Focal variable to vary (x-axis).
    at_values : list or array, optional
        Specific values to evaluate. If None, uses range of observed values.
    other_vars : dict, optional
        Values to hold other variables at. If None, uses means.
        Example: {'age': 30, 'gender': 'F'}
    ci : bool, default True
        Show confidence intervals.
    ci_level : float, default 0.95
        Confidence level for intervals.
    n_points : int, default 50
        Number of points to evaluate along variable range.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label. Defaults to variable name.
    ylabel : str, optional
        Y-axis label. Defaults to outcome name.
    style_mode : str, default "house"
        Color scheme.
    figsize : tuple, default (10, 6)
        Figure size.
    output_path : str, optional
        Save plot to file.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    Simple margin plot:
    >>> from whatsthepoint.analyses.regress import ols
    >>> model = ols(df, 'income', ['education', 'age', 'gender'])
    >>> margins(model, 'education', title='Income by Education')

    Margin plot at specific values:
    >>> margins(model, 'education', other_vars={'age': 30, 'gender': 'F'})

    Margin plot with specific evaluation points:
    >>> margins(model, 'age', at_values=[20, 30, 40, 50, 60])
    """
    # Extract model information - check for required attributes
    required_attrs = ['predict', 'df', 'outcome']
    missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr)]

    if missing_attrs:
        raise TypeError(
            f"model must have the following attributes: {', '.join(required_attrs)}. "
            f"Missing: {', '.join(missing_attrs)}"
        )

    # Get data and variable info
    df = model.df
    inputs = getattr(model, 'inputs', None) or getattr(model, 'fixed', None)

    if inputs is None:
        raise TypeError("model must have 'inputs' or 'fixed' attribute containing predictor variables")

    outcome = model.outcome

    if variable not in inputs:
        raise ValueError(f"{variable} not found in model inputs")

    # Determine values to evaluate
    if at_values is None:
        var_min = df[variable].min()
        var_max = df[variable].max()
        at_values = np.linspace(var_min, var_max, n_points)

    # Create prediction data
    pred_data = pd.DataFrame({variable: at_values})

    # Set other variables
    if other_vars is None:
        other_vars = {}

    for var in inputs:
        if var != variable and var not in other_vars:
            # Use mean for continuous, mode for categorical
            if df[var].dtype in [np.float64, np.int64, np.float32, np.int32]:
                other_vars[var] = df[var].mean()
            else:
                other_vars[var] = df[var].mode()[0]

    for var, val in other_vars.items():
        pred_data[var] = val

    # Generate predictions
    try:
        predictions = model.predict(pred_data)

        # Get confidence intervals if available
        if ci:
            try:
                # Get prediction standard errors (approximate)
                from scipy import stats
                residual_std = np.sqrt(model.results.scale) if hasattr(model.results, 'scale') else np.std(
                    model.results.resid)
                z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
                margin_error = z_crit * residual_std

                ci_lower = predictions - margin_error
                ci_upper = predictions + margin_error
            except:
                ci = False
                warnings.warn("Could not calculate confidence intervals")
    except Exception as e:
        raise RuntimeError(f"Error generating predictions: {e}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = color_map[0] if hasattr(color_map, '__getitem__') else 'steelblue'

    # Plot predictions
    ax.plot(at_values, predictions, color=main_color, linewidth=2.5, label='Predicted')

    # Plot confidence interval
    if ci:
        ax.fill_between(at_values, ci_lower, ci_upper, color=main_color, alpha=0.2, label=f'{int(ci_level * 100)}% CI')

    # Add actual data points (optional, with transparency)
    if len(df) < 1000:
        # Sample if too many points
        df_plot = df.sample(min(200, len(df)), random_state=42)
        ax.scatter(df_plot[variable], df_plot[outcome], alpha=0.1, color='grey', s=10, zorder=0)

    # Labels
    xlabel = xlabel or variable.replace('_', ' ').title()
    ylabel = ylabel or f"Predicted {outcome.replace('_', ' ').title()}"

    ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='black')

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


def margins_comparison(
        model,
        variable: str,
        by: str,
        by_values: Optional[List] = None,
        at_values: Optional[Union[List, np.ndarray]] = None,
        other_vars: Optional[dict] = None,
        ci: bool = True,
        ci_level: float = 0.95,
        n_points: int = 50,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        style_mode: str = "house",
        figsize: tuple = (10, 6),
        output_path: Optional[str] = None,
):
    """
    Plot marginal effects comparing across groups or levels of another variable.

    Useful for visualizing interaction effects.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    variable : str
        Focal variable to vary (x-axis).
    by : str
        Variable to compare across (different lines).
    by_values : list, optional
        Specific values of 'by' variable to plot. If None, uses quartiles or unique values.
    at_values : list or array, optional
        Values of focal variable to evaluate.
    other_vars : dict, optional
        Values for other variables.
    ci : bool, default True
        Show confidence intervals.
    ci_level : float, default 0.95
        Confidence level.
    n_points : int, default 50
        Number of evaluation points.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
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
    Interaction between education and gender:
    >>> margins_comparison(model, 'education', by='gender')

    Age effect at different income levels:
    >>> margins_comparison(model, 'age', by='income',
    ...                    by_values=[25000, 50000, 75000])
    """
    # Get model info
    df = model.df
    inputs = getattr(model, 'inputs', None) or getattr(model, 'fixed', None)
    outcome = model.outcome

    if variable not in inputs:
        raise ValueError(f"{variable} not found in model inputs")
    if by not in inputs:
        raise ValueError(f"{by} not found in model inputs")

    # Determine values to evaluate
    if at_values is None:
        var_min = df[variable].min()
        var_max = df[variable].max()
        at_values = np.linspace(var_min, var_max, n_points)

    # Determine by values
    if by_values is None:
        if df[by].dtype in [np.float64, np.int64, np.float32, np.int32]:
            # Use quartiles for continuous
            by_values = df[by].quantile([0.25, 0.50, 0.75]).values
        else:
            # Use unique values for categorical (max 5)
            by_values = df[by].value_counts().head(5).index.tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(i / len(by_values)) for i in range(len(by_values))]
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(by_values)))

    # Plot each group
    for i, by_val in enumerate(by_values):
        # Create prediction data
        pred_data = pd.DataFrame({variable: at_values})
        pred_data[by] = by_val

        # Set other variables
        temp_other_vars = other_vars.copy() if other_vars else {}

        for var in inputs:
            if var not in [variable, by] and var not in temp_other_vars:
                if df[var].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    temp_other_vars[var] = df[var].mean()
                else:
                    temp_other_vars[var] = df[var].mode()[0]

        for var, val in temp_other_vars.items():
            if var not in [variable, by]:
                pred_data[var] = val

        # Generate predictions
        predictions = model.predict(pred_data)

        # Plot line
        label = f"{by}={by_val:.2f}" if isinstance(by_val, (int, float)) else f"{by}={by_val}"
        ax.plot(at_values, predictions, color=colors[i], linewidth=2.5, label=label)

        # Confidence interval (optional, makes plot busy)
        if ci and len(by_values) <= 3:
            try:
                from scipy import stats
                residual_std = np.sqrt(model.results.scale)
                z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
                margin_error = z_crit * residual_std
                ci_lower = predictions - margin_error
                ci_upper = predictions + margin_error
                ax.fill_between(at_values, ci_lower, ci_upper, color=colors[i], alpha=0.15)
            except:
                pass

    # Labels
    xlabel = xlabel or variable.replace('_', ' ').title()
    ylabel = ylabel or f"Predicted {outcome.replace('_', ' ').title()}"

    ax.set_xlabel(xlabel, fontsize=12, weight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=12, weight='bold', color='black')

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
# INTERACTIVE VERSIONS
# ══════════════════════════════════════════════════════════════════════════════

def margins_interactive(
        model,
        variable,
        at_values=None,
        other_vars=None,
        ci=True,
        ci_level=0.95,
        n_points=50,
        title=None,
        subtitle=None,
        xlabel=None,
        ylabel=None,
        style_mode="house",
):
    """Interactive margin plot using Plotly."""
    import plotly.graph_objects as go

    # Get data and variable info
    df = model.df
    inputs = getattr(model, 'inputs', None) or getattr(model, 'fixed', None)
    outcome = model.outcome

    if variable not in inputs:
        raise ValueError(f"{variable} not found in model inputs")

    # Determine values to evaluate
    if at_values is None:
        var_min = df[variable].min()
        var_max = df[variable].max()
        at_values = np.linspace(var_min, var_max, n_points)

    # Create prediction data
    pred_data = pd.DataFrame({variable: at_values})

    # Set other variables
    if other_vars is None:
        other_vars = {}

    for var in inputs:
        if var != variable and var not in other_vars:
            if df[var].dtype in [np.float64, np.int64, np.float32, np.int32]:
                other_vars[var] = df[var].mean()
            else:
                other_vars[var] = df[var].mode()[0]

    for var, val in other_vars.items():
        pred_data[var] = val

    # Generate predictions
    predictions = model.predict(pred_data)

    # Get confidence intervals if available
    if ci:
        try:
            from scipy import stats
            residual_std = np.sqrt(model.results.scale) if hasattr(model.results, 'scale') else np.std(
                model.results.resid)
            z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
            margin_error = z_crit * residual_std

            ci_lower = predictions - margin_error
            ci_upper = predictions + margin_error
        except:
            ci = False

    # Get color
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        main_color = color_map(0.6)
    else:
        main_color = 'steelblue'

    if hasattr(main_color, '__iter__') and not isinstance(main_color, str):
        color_str = f"rgba({int(main_color[0] * 255)},{int(main_color[1] * 255)},{int(main_color[2] * 255)},1.0)"
    else:
        color_str = main_color

    # Create figure
    fig = go.Figure()

    # Add confidence interval
    if ci:
        if hasattr(main_color, '__iter__') and not isinstance(main_color, str):
            fill_color = f"rgba({int(main_color[0] * 255)},{int(main_color[1] * 255)},{int(main_color[2] * 255)},0.2)"
        else:
            fill_color = 'rgba(70,130,180,0.2)'

        fig.add_trace(go.Scatter(
            x=at_values,
            y=ci_upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=at_values,
            y=ci_lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=fill_color,
            name=f'{int(ci_level * 100)}% CI',
        ))

    # Add prediction line
    fig.add_trace(go.Scatter(
        x=at_values,
        y=predictions,
        mode='lines',
        line=dict(color=color_str, width=3),
        name='Predicted',
    ))

    # Labels
    xlabel = xlabel or variable.replace('_', ' ').title()
    ylabel = ylabel or f"Predicted {outcome.replace('_', ' ').title()}"

    # Layout
    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{title or 'Marginal Effects'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or 'Marginal Effects'}</b>",
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
        showlegend=True,
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top'),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))

    return fig


def margins_comparison_interactive(
        model,
        variable,
        by,
        by_values=None,
        at_values=None,
        other_vars=None,
        ci=True,
        ci_level=0.95,
        n_points=50,
        title=None,
        subtitle=None,
        xlabel=None,
        ylabel=None,
        style_mode="house",
):
    """Interactive margin comparison plot using Plotly."""
    import plotly.graph_objects as go

    # Get model info
    df = model.df
    inputs = getattr(model, 'inputs', None) or getattr(model, 'fixed', None)
    outcome = model.outcome

    if variable not in inputs:
        raise ValueError(f"{variable} not found in model inputs")
    if by not in inputs:
        raise ValueError(f"{by} not found in model inputs")

    # Determine values to evaluate
    if at_values is None:
        var_min = df[variable].min()
        var_max = df[variable].max()
        at_values = np.linspace(var_min, var_max, n_points)

    # Determine by values
    if by_values is None:
        if df[by].dtype in [np.float64, np.int64, np.float32, np.int32]:
            by_values = df[by].quantile([0.25, 0.50, 0.75]).values
        else:
            by_values = df[by].value_counts().head(5).index.tolist()

    # Get colors
    color_map = COLORS_DICT.get(style_mode, plt.cm.viridis)
    if callable(color_map):
        colors = [color_map(i / len(by_values)) for i in range(len(by_values))]
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(by_values)))

    # Create figure
    fig = go.Figure()

    # Plot each group
    for i, by_val in enumerate(by_values):
        # Create prediction data
        pred_data = pd.DataFrame({variable: at_values})
        pred_data[by] = by_val

        # Set other variables
        temp_other_vars = other_vars.copy() if other_vars else {}

        for var in inputs:
            if var not in [variable, by] and var not in temp_other_vars:
                if df[var].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    temp_other_vars[var] = df[var].mean()
                else:
                    temp_other_vars[var] = df[var].mode()[0]

        for var, val in temp_other_vars.items():
            if var not in [variable, by]:
                pred_data[var] = val

        # Generate predictions
        predictions = model.predict(pred_data)

        # Get color
        color = colors[i]
        if hasattr(color, '__iter__') and not isinstance(color, str):
            color_str = f"rgba({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)},1.0)"
        else:
            color_str = color

        # Plot line
        label = f"{by}={by_val:.2f}" if isinstance(by_val, (int, float)) else f"{by}={by_val}"

        # Add confidence interval if requested
        if ci and len(by_values) <= 3:
            try:
                from scipy import stats
                residual_std = np.sqrt(model.results.scale)
                z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
                margin_error = z_crit * residual_std
                ci_lower = predictions - margin_error
                ci_upper = predictions + margin_error

                if hasattr(color, '__iter__') and not isinstance(color, str):
                    fill_color = f"rgba({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)},0.15)"
                else:
                    fill_color = 'rgba(70,130,180,0.15)'

                fig.add_trace(go.Scatter(
                    x=at_values,
                    y=ci_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=label,
                ))
                fig.add_trace(go.Scatter(
                    x=at_values,
                    y=ci_lower,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=fill_color,
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup=label,
                ))
            except:
                pass

        fig.add_trace(go.Scatter(
            x=at_values,
            y=predictions,
            mode='lines',
            line=dict(color=color_str, width=3),
            name=label,
            legendgroup=label,
        ))

    # Labels
    xlabel = xlabel or variable.replace('_', ' ').title()
    ylabel = ylabel or f"Predicted {outcome.replace('_', ' ').title()}"

    # Layout
    title_dict = {}
    if subtitle:
        title_dict = dict(
            text=f"<b>{title or 'Margin Comparison'}</b>"
                 + f"<br><span style='color:grey;font-size:14px;'>{subtitle}</span>",
            x=0.02,
            xanchor="left",
            yanchor="top",
            y=0.96,
        )
    else:
        title_dict = dict(
            text=f"<b>{title or 'Margin Comparison'}</b>",
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
        showlegend=True,
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top',
                    title=dict(text=by.replace('_', ' ').title(), font=dict(weight='bold'))),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", tickfont=dict(size=11, color="#333"))

    return fig