"""
panel.py: whatsthepoint Panel Data Module
-----------------------------------------
Panel data (longitudinal) regression models.

Features:
- Fixed effects (FE) models
- Random effects (RE) models
- First-differences models
- Hausman test for FE vs RE
- Panel diagnostics

Functions:
- fixed_effects: Within (fixed effects) estimator
- random_effects: GLS random effects estimator
- first_differences: First-difference estimator
- hausman_test: Test FE vs RE specification
- panel_summary: Descriptive statistics for panel data
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict
import warnings

# statsmodels/linearmodels imports
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

try:
    from linearmodels.panel import PanelOLS, RandomEffects, FirstDifferenceOLS
    from linearmodels.panel import compare as panel_compare
    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False  # pip install whatsthepoint[panel]



# ==============================================================================
# FIXED EFFECTS MODELS
# ==============================================================================

def fixed_effects(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    entity: str,
    time: Optional[str] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    entity_effects: bool = True,
    time_effects: bool = False,
) -> Dict:
    """
    Fixed effects (within) panel regression.

    Parameters
    ----------
    df : DataFrame
        Panel data in long format.
    outcome : str
        Outcome (dependent) variable.
    inputs : str or list of str
        Input (independent) variable(s).
    entity : str
        Entity identifier (e.g., person ID, country).
    time : str, optional
        Time identifier (e.g., year). Required for time effects.
    weights : str, optional
        Weight variable.
    cluster : str, optional
        Variable to cluster standard errors by.
    entity_effects : bool, default True
        Include entity fixed effects.
    time_effects : bool, default False
        Include time fixed effects.

    Returns
    -------
    dict
        Dictionary with 'coefficients', 'model', 'fit_stats'.

    Examples
    --------
    >>> result = fixed_effects(df, 'wage', ['experience', 'hours'],
    ...                         entity='person_id', time='year')
    >>> print(result['coefficients'])
    """
    if not LINEARMODELS_AVAILABLE:
        # Fallback to manual FE estimation
        return _fixed_effects_manual(
            df, outcome, inputs, entity, time, weights,
            entity_effects, time_effects
        )

    # Prepare inputs
    if isinstance(inputs, str):
        inputs = [inputs]

    # Set panel index
    if time:
        df_panel = df.set_index([entity, time])
    else:
        # Create dummy time index if not provided
        df_temp = df.copy()
        df_temp['_time'] = df_temp.groupby(entity).cumcount()
        df_panel = df_temp.set_index([entity, '_time'])

    # Prepare variables
    y = df_panel[outcome]
    X = df_panel[inputs]

    # Fit model
    if weights:
        w = df_panel[weights]
        model = PanelOLS(
            y, X,
            entity_effects=entity_effects,
            time_effects=time_effects,
            weights=w
        )
    else:
        model = PanelOLS(
            y, X,
            entity_effects=entity_effects,
            time_effects=time_effects
        )

    # Fit with clustering if specified
    if cluster:
        if cluster == entity:
            results = model.fit(cov_type='clustered', cluster_entity=True)
        else:
            results = model.fit(cov_type='clustered', clusters=df_panel[cluster])
    else:
        results = model.fit(cov_type='robust')

    # Extract coefficients
    coefficients = pd.DataFrame({
        'variable': results.params.index,
        'estimate': results.params.values,
        'std.error': results.std_errors.values,
        't_stat': results.tstats.values,
        'p.value': results.pvalues.values,
    })

    # Fit statistics
    fit_stats = {
        'N': results.nobs,
        'N_entities': results.entity_info['total'],
        'R_squared': results.rsquared,
        'R_squared_within': results.rsquared_within,
        'R_squared_between': results.rsquared_between,
        'R_squared_overall': results.rsquared_overall,
        'F_statistic': results.f_statistic.stat,
        'F_pvalue': results.f_statistic.pval,
    }

    return {
        'coefficients': coefficients,
        'model': results,
        'fit_stats': fit_stats,
    }


def _fixed_effects_manual(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    entity: str,
    time: Optional[str],
    weights: Optional[str],
    entity_effects: bool,
    time_effects: bool,
) -> Dict:
    """
    Manual fixed effects estimation (fallback when linearmodels unavailable).

    Uses within-transformation (demeaning).
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    if isinstance(inputs, str):
        inputs = [inputs]

    # Prepare data
    df_work = df[[outcome] + inputs + [entity]].copy()
    if time:
        df_work['_time'] = df[time]
    if weights:
        df_work['_weights'] = df[weights]

    df_work = df_work.dropna()

    # Within-transformation: demean by entity
    if entity_effects:
        for var in [outcome] + inputs:
            entity_means = df_work.groupby(entity)[var].transform('mean')
            df_work[f'{var}_demeaned'] = df_work[var] - entity_means

        y = df_work[f'{outcome}_demeaned']
        X = df_work[[f'{inp}_demeaned' for inp in inputs]]
    else:
        y = df_work[outcome]
        X = df_work[inputs]

    # Add time fixed effects if requested
    if time_effects and time:
        time_dummies = pd.get_dummies(df_work['_time'], prefix='time', drop_first=True)
        X = pd.concat([X, time_dummies], axis=1)

    # Fit OLS on demeaned data
    if weights:
        w = df_work['_weights']
        model = sm.WLS(y, X, weights=w).fit(cov_type='HC1')
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    # Extract coefficients (rename back to original names)
    coef_names = [name.replace('_demeaned', '') for name in model.params.index]
    coefficients = pd.DataFrame({
        'variable': coef_names,
        'estimate': model.params.values,
        'std.error': model.bse.values,
        't_stat': model.tvalues.values,
        'p.value': model.pvalues.values,
    })

    # Fit statistics
    n_entities = df_work[entity].nunique()
    fit_stats = {
        'N': int(model.nobs),
        'N_entities': n_entities,
        'R_squared': model.rsquared,
    }

    return {
        'coefficients': coefficients,
        'model': model,
        'fit_stats': fit_stats,
    }


# ==============================================================================
# RANDOM EFFECTS MODELS
# ==============================================================================

def random_effects(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    entity: str,
    time: Optional[str] = None,
    weights: Optional[str] = None,
) -> Dict:
    """
    Random effects (GLS) panel regression.

    Parameters
    ----------
    df : DataFrame
        Panel data in long format.
    outcome : str
        Outcome variable.
    inputs : str or list of str
        Input variable(s).
    entity : str
        Entity identifier.
    time : str, optional
        Time identifier.
    weights : str, optional
        Weight variable.

    Returns
    -------
    dict
        Dictionary with 'coefficients', 'model', 'fit_stats'.

    Examples
    --------
    >>> result = random_effects(df, 'wage', ['experience', 'education'],
    ...                          entity='person_id', time='year')
    """
    if not LINEARMODELS_AVAILABLE:
        raise ImportError(
            "Random effects requires linearmodels. "
            "Install it with: pip install whatsthepoint[panel]"
        )

    # Prepare inputs
    if isinstance(inputs, str):
        inputs = [inputs]

    # Set panel index
    if time:
        df_panel = df.set_index([entity, time])
    else:
        df_temp = df.copy()
        df_temp['_time'] = df_temp.groupby(entity).cumcount()
        df_panel = df_temp.set_index([entity, '_time'])

    # Prepare variables
    y = df_panel[outcome]
    X = df_panel[inputs]

    # Fit model
    if weights:
        w = df_panel[weights]
        model = RandomEffects(y, X, weights=w)
    else:
        model = RandomEffects(y, X)

    results = model.fit()

    # Extract coefficients
    coefficients = pd.DataFrame({
        'variable': results.params.index,
        'estimate': results.params.values,
        'std.error': results.std_errors.values,
        't_stat': results.tstats.values,
        'p.value': results.pvalues.values,
    })

    # Fit statistics
    fit_stats = {
        'N': results.nobs,
        'N_entities': results.entity_info['total'],
        'R_squared': results.rsquared,
        'R_squared_within': results.rsquared_within,
        'R_squared_between': results.rsquared_between,
        'R_squared_overall': results.rsquared_overall,
    }

    return {
        'coefficients': coefficients,
        'model': results,
        'fit_stats': fit_stats,
    }


# ==============================================================================
# FIRST DIFFERENCES
# ==============================================================================

def first_differences(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    entity: str,
    time: str,
) -> Dict:
    """
    First-difference panel regression.

    Parameters
    ----------
    df : DataFrame
        Panel data in long format.
    outcome : str
        Outcome variable.
    inputs : str or list of str
        Input variable(s).
    entity : str
        Entity identifier.
    time : str
        Time identifier.

    Returns
    -------
    dict
        Dictionary with 'coefficients', 'model', 'fit_stats'.

    Examples
    --------
    >>> result = first_differences(df, 'wage', ['experience'],
    ...                             entity='person_id', time='year')
    """
    if not LINEARMODELS_AVAILABLE:
        # Manual first-differences
        return _first_differences_manual(df, outcome, inputs, entity, time)

    # Prepare inputs
    if isinstance(inputs, str):
        inputs = [inputs]

    # Set panel index
    df_panel = df.set_index([entity, time]).sort_index()

    # Prepare variables
    y = df_panel[outcome]
    X = df_panel[inputs]

    # Fit model
    model = FirstDifferenceOLS(y, X)
    results = model.fit(cov_type='robust')

    # Extract coefficients
    coefficients = pd.DataFrame({
        'variable': results.params.index,
        'estimate': results.params.values,
        'std.error': results.std_errors.values,
        't_stat': results.tstats.values,
        'p.value': results.pvalues.values,
    })

    # Fit statistics
    fit_stats = {
        'N': results.nobs,
        'N_entities': results.entity_info['total'],
        'R_squared': results.rsquared,
    }

    return {
        'coefficients': coefficients,
        'model': results,
        'fit_stats': fit_stats,
    }


def _first_differences_manual(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    entity: str,
    time: str,
) -> Dict:
    """Manual first-difference estimation."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    if isinstance(inputs, str):
        inputs = [inputs]

    # Sort by entity and time
    df_sorted = df.sort_values([entity, time]).copy()

    # Calculate first differences
    vars_to_diff = [outcome] + inputs
    df_sorted[f'{entity}_lag'] = df_sorted.groupby(entity)[entity].shift(1)

    # Only keep consecutive observations
    df_sorted = df_sorted[df_sorted[entity] == df_sorted[f'{entity}_lag']].copy()

    for var in vars_to_diff:
        df_sorted[f'{var}_diff'] = df_sorted.groupby(entity)[var].diff()

    df_diff = df_sorted.dropna(subset=[f'{var}_diff' for var in vars_to_diff])

    # Regression on differences
    y = df_diff[f'{outcome}_diff']
    X = df_diff[[f'{inp}_diff' for inp in inputs]]

    model = sm.OLS(y, X).fit(cov_type='HC1')

    # Extract coefficients (rename back)
    coef_names = [name.replace('_diff', '') for name in model.params.index]
    coefficients = pd.DataFrame({
        'variable': coef_names,
        'estimate': model.params.values,
        'std.error': model.bse.values,
        't_stat': model.tvalues.values,
        'p.value': model.pvalues.values,
    })

    n_entities = df_diff[entity].nunique()
    fit_stats = {
        'N': int(model.nobs),
        'N_entities': n_entities,
        'R_squared': model.rsquared,
    }

    return {
        'coefficients': coefficients,
        'model': model,
        'fit_stats': fit_stats,
    }


# ==============================================================================
# HAUSMAN TEST
# ==============================================================================

def hausman_test(
    fe_result: Dict,
    re_result: Dict,
) -> Dict:
    """
    Hausman specification test for fixed vs random effects.

    Tests whether random effects model is consistent (no correlation
    between effects and regressors).

    Parameters
    ----------
    fe_result : dict
        Results from fixed_effects().
    re_result : dict
        Results from random_effects().

    Returns
    -------
    dict
        Dictionary with 'statistic', 'p_value', 'dof', 'recommendation'.

    Examples
    --------
    >>> fe_res = fixed_effects(df, 'wage', 'experience', entity='person_id')
    >>> re_res = random_effects(df, 'wage', 'experience', entity='person_id')
    >>> hausman = hausman_test(fe_res, re_res)
    >>> print(f"Hausman test: chi2={hausman['statistic']:.2f}, p={hausman['p_value']:.3f}")
    """
    # Extract coefficients
    b_fe = fe_result['coefficients'].set_index('variable')['estimate']
    b_re = re_result['coefficients'].set_index('variable')['estimate']

    # Get common variables
    common_vars = b_fe.index.intersection(b_re.index)
    b_fe = b_fe[common_vars]
    b_re = b_re[common_vars]

    # Covariance matrices
    V_fe = np.diag(fe_result['coefficients'].set_index('variable').loc[common_vars, 'std.error']**2)
    V_re = np.diag(re_result['coefficients'].set_index('variable').loc[common_vars, 'std.error']**2)

    # Hausman statistic
    b_diff = b_fe.values - b_re.values
    V_diff = V_fe - V_re

    # Check if V_diff is positive definite
    try:
        V_diff_inv = np.linalg.inv(V_diff)
        statistic = b_diff.T @ V_diff_inv @ b_diff
        dof = len(common_vars)

        # Chi-square test
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(statistic, dof)

        if p_value < 0.05:
            recommendation = "Fixed effects (FE) preferred (reject RE)"
        else:
            recommendation = "Random effects (RE) preferred (fail to reject RE)"

        return {
            'statistic': statistic,
            'p_value': p_value,
            'dof': dof,
            'recommendation': recommendation,
        }

    except np.linalg.LinAlgError:
        warnings.warn("Hausman test failed (singular variance matrix)")
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'dof': len(common_vars),
            'recommendation': "Test failed - use theory or compare fit statistics",
        }


# ==============================================================================
# PANEL DESCRIPTIVES
# ==============================================================================

def panel_summary(
    df: pd.DataFrame,
    variables: Union[str, List[str]],
    entity: str,
    time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Descriptive statistics for panel data.

    Decomposes variance into within and between components.

    Parameters
    ----------
    df : DataFrame
        Panel data.
    variables : str or list of str
        Variables to summarize.
    entity : str
        Entity identifier.
    time : str, optional
        Time identifier.

    Returns
    -------
    DataFrame
        Summary statistics with overall, within, and between variance.

    Examples
    --------
    >>> summary = panel_summary(df, ['wage', 'hours'], entity='person_id')
    """
    if isinstance(variables, str):
        variables = [variables]

    results = []

    for var in variables:
        data = df[[var, entity]].dropna()

        # Overall statistics
        overall_mean = data[var].mean()
        overall_std = data[var].std()
        overall_min = data[var].min()
        overall_max = data[var].max()

        # Between variation (across entities)
        entity_means = data.groupby(entity)[var].mean()
        between_std = entity_means.std()
        between_min = entity_means.min()
        between_max = entity_means.max()

        # Within variation (within entities over time)
        entity_mean_map = data.groupby(entity)[var].transform('mean')
        within_var = data[var] - entity_mean_map
        within_std = within_var.std()
        within_min = within_var.min()
        within_max = within_var.max()

        # Number of entities and observations
        n_entities = data[entity].nunique()
        n_obs = len(data)

        results.append({
            'variable': var,
            'N': n_obs,
            'N_entities': n_entities,
            'mean': overall_mean,
            'std_overall': overall_std,
            'std_between': between_std,
            'std_within': within_std,
            'min': overall_min,
            'max': overall_max,
        })

    return pd.DataFrame(results)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def compare_panel_models(
    fe_result: Dict,
    re_result: Dict,
    fd_result: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compare panel model specifications.

    Parameters
    ----------
    fe_result : dict
        Fixed effects results.
    re_result : dict
        Random effects results.
    fd_result : dict, optional
        First-difference results.

    Returns
    -------
    DataFrame
        Comparison table.

    Examples
    --------
    >>> comparison = compare_panel_models(fe_res, re_res, fd_res)
    """
    models = {
        'Fixed Effects': fe_result,
        'Random Effects': re_result,
    }
    if fd_result:
        models['First Difference'] = fd_result

    comparison = []

    for name, result in models.items():
        stats = result['fit_stats'].copy()
        stats['Model'] = name
        comparison.append(stats)

    df_compare = pd.DataFrame(comparison)

    # Reorder columns
    cols = ['Model'] + [c for c in df_compare.columns if c != 'Model']
    return df_compare[cols]
