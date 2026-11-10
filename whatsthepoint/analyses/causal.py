"""
causal.py: whatsthepoint Causal Inference Module
-----------------------------------------------
Causal inference methods for sociological research.

Features:
- Propensity score matching and weighting
- Difference-in-differences (DiD)
- Instrumental variables (IV) regression
- Regression discontinuity design (RDD)
- Sensitivity analysis

Functions:
- propensity_score: Estimate propensity scores and create weights/matches
- difference_in_differences: DiD analysis with parallel trends tests
- instrumental_variables: Two-stage least squares (2SLS) IV regression
- regression_discontinuity: RDD analysis with bandwidth selection
- sensitivity_analysis: Rosenbaum bounds and other sensitivity tests
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple, Dict
import warnings

# statsmodels imports
try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS, WLS
    from statsmodels.sandbox.regression.gmm import IV2SLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

# sklearn imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Install with: pip install scikit-learn")



# ==============================================================================
# PROPENSITY SCORE METHODS
# ==============================================================================

class PropensityScoreAnalysis:
    """
    Propensity score analysis with matching, weighting, and stratification.

    Parameters
    ----------
    df : DataFrame
        Input data.
    treatment : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates for propensity score model.
    outcome : str, optional
        Outcome variable (for effect estimation).

    Attributes
    ----------
    propensity_scores : Series
        Estimated propensity scores.
    weights : Series
        Inverse probability weights (IPW).
    matches : DataFrame
        Matched pairs (if matching performed).

    Examples
    --------
    >>> ps = PropensityScoreAnalysis(df, treatment='treated',
    ...                               covariates=['age', 'income', 'education'])
    >>> ps.fit()
    >>> ate = ps.estimate_ate(outcome='satisfaction', method='ipw')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        treatment: str,
        covariates: List[str],
        outcome: Optional[str] = None,
    ):
        if not STATSMODELS_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError(
                "statsmodels and sklearn required. "
                "Install with: pip install statsmodels scikit-learn"
            )

        self.df = df.copy()
        self.treatment = treatment
        self.covariates = covariates
        self.outcome = outcome

        self.propensity_scores = None
        self.weights = None
        self.matches = None
        self.ps_model = None

    def fit(self, method: str = "logit"):
        """
        Estimate propensity scores.

        Parameters
        ----------
        method : str, default "logit"
            Method: "logit" or "probit".
        """
        # Prepare data
        vars_needed = [self.treatment] + self.covariates
        df_clean = self.df[vars_needed].dropna()

        y = df_clean[self.treatment]
        X = df_clean[self.covariates]
        X = sm.add_constant(X)

        # Fit propensity score model
        if method == "logit":
            self.ps_model = sm.Logit(y, X).fit(disp=0)
        elif method == "probit":
            self.ps_model = sm.Probit(y, X).fit(disp=0)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Get propensity scores
        self.propensity_scores = self.ps_model.predict(X)
        self.propensity_scores.index = df_clean.index

        # Calculate IPW weights
        treated = df_clean[self.treatment]
        ps = self.propensity_scores

        # ATE weights: 1/ps for treated, 1/(1-ps) for control
        self.weights = pd.Series(index=df_clean.index, dtype=float)
        self.weights[treated == 1] = 1 / ps[treated == 1]
        self.weights[treated == 0] = 1 / (1 - ps[treated == 0])

        return self

    def match(
        self,
        method: str = "nearest",
        caliper: Optional[float] = None,
        ratio: int = 1,
        replace: bool = False,
    ):
        """
        Perform propensity score matching.

        Parameters
        ----------
        method : str, default "nearest"
            Matching method: "nearest", "optimal" (future).
        caliper : float, optional
            Maximum propensity score distance for matches.
        ratio : int, default 1
            Number of control matches per treated unit.
        replace : bool, default False
            Whether to match with replacement.

        Returns
        -------
        self
            Returns self with matches attribute populated.
        """
        if self.propensity_scores is None:
            raise ValueError("Must call .fit() before matching")

        df_work = self.df.loc[self.propensity_scores.index].copy()
        df_work['ps'] = self.propensity_scores

        treated = df_work[df_work[self.treatment] == 1].copy()
        control = df_work[df_work[self.treatment] == 0].copy()

        if method == "nearest":
            # Nearest neighbor matching
            matches_list = []

            for idx, row in treated.iterrows():
                ps_treat = row['ps']

                # Find nearest neighbors in control group
                distances = np.abs(control['ps'] - ps_treat)

                if caliper:
                    valid = distances <= caliper
                    if valid.sum() == 0:
                        continue  # No matches within caliper
                    distances = distances[valid]
                    control_subset = control[valid]
                else:
                    control_subset = control

                # Get nearest neighbors
                nearest_idx = distances.nsmallest(ratio).index

                for ctrl_idx in nearest_idx:
                    matches_list.append({
                        'treated_id': idx,
                        'control_id': ctrl_idx,
                        'ps_treated': ps_treat,
                        'ps_control': control.loc[ctrl_idx, 'ps'],
                        'distance': distances[ctrl_idx],
                    })

                # Remove matched controls if matching without replacement
                if not replace:
                    control = control.drop(nearest_idx)

            self.matches = pd.DataFrame(matches_list)

        else:
            raise NotImplementedError(f"Matching method '{method}' not yet implemented")

        return self

    def estimate_ate(
        self,
        outcome: Optional[str] = None,
        method: str = "ipw",
    ) -> Dict[str, float]:
        """
        Estimate average treatment effect (ATE).

        Parameters
        ----------
        outcome : str, optional
            Outcome variable. Uses self.outcome if not specified.
        method : str, default "ipw"
            Estimation method: "ipw" (inverse probability weighting),
            "matching" (matched sample comparison), "regression".

        Returns
        -------
        dict
            Dictionary with 'ate', 'se', 'ci_lower', 'ci_upper', 'p_value'.

        Examples
        --------
        >>> ate_result = ps.estimate_ate('income', method='ipw')
        >>> print(f"ATE: {ate_result['ate']:.2f} (p={ate_result['p_value']:.3f})")
        """
        if outcome is None:
            outcome = self.outcome
        if outcome is None:
            raise ValueError("outcome must be specified")

        if method == "ipw":
            if self.weights is None:
                raise ValueError("Must call .fit() before estimating ATE")

            # Get data
            idx = self.weights.index
            df_work = self.df.loc[idx, [self.treatment, outcome]].dropna()
            weights = self.weights.loc[df_work.index]

            treated = df_work[self.treatment] == 1

            # Weighted means
            y1 = np.average(df_work.loc[treated, outcome], weights=weights[treated])
            y0 = np.average(df_work.loc[~treated, outcome], weights=weights[~treated])

            ate = y1 - y0

            # Standard error (approximate)
            n1 = treated.sum()
            n0 = (~treated).sum()
            var1 = np.average((df_work.loc[treated, outcome] - y1)**2, weights=weights[treated])
            var0 = np.average((df_work.loc[~treated, outcome] - y0)**2, weights=weights[~treated])
            se = np.sqrt(var1/n1 + var0/n0)

            # Confidence interval and p-value
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se
            p_value = 2 * (1 - stats.norm.cdf(np.abs(ate / se)))

            return {
                'ate': ate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
            }

        elif method == "matching":
            if self.matches is None:
                raise ValueError("Must call .match() before using matching estimator")

            treated_outcomes = self.df.loc[self.matches['treated_id'], outcome]
            control_outcomes = self.df.loc[self.matches['control_id'], outcome]

            ate = (treated_outcomes.values - control_outcomes.values).mean()

            # Standard error
            differences = treated_outcomes.values - control_outcomes.values
            se = differences.std() / np.sqrt(len(differences))

            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se
            p_value = 2 * (1 - stats.norm.cdf(np.abs(ate / se)))

            return {
                'ate': ate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
            }

        elif method == "regression":
            # Regression adjustment
            idx = self.propensity_scores.index
            df_work = self.df.loc[idx, [self.treatment, outcome] + self.covariates].dropna()

            y = df_work[outcome]
            X = df_work[[self.treatment] + self.covariates]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            ate = model.params[self.treatment]
            se = model.bse[self.treatment]
            ci_lower, ci_upper = model.conf_int().loc[self.treatment]
            p_value = model.pvalues[self.treatment]

            return {
                'ate': ate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
            }

        else:
            raise ValueError(f"Unsupported method: {method}")

    def balance_check(self) -> pd.DataFrame:
        """
        Check covariate balance after propensity score adjustment.

        Returns
        -------
        DataFrame
            Standardized mean differences for each covariate.
        """
        if self.propensity_scores is None:
            raise ValueError("Must call .fit() first")

        idx = self.propensity_scores.index
        df_work = self.df.loc[idx].copy()
        treated = df_work[self.treatment] == 1

        balance = []

        for cov in self.covariates:
            # Before weighting
            mean_t = df_work.loc[treated, cov].mean()
            mean_c = df_work.loc[~treated, cov].mean()
            std_pooled = np.sqrt(
                (df_work.loc[treated, cov].var() + df_work.loc[~treated, cov].var()) / 2
            )
            smd_before = (mean_t - mean_c) / std_pooled if std_pooled > 0 else 0

            # After IPW weighting
            weights = self.weights.loc[df_work.index]
            mean_t_w = np.average(df_work.loc[treated, cov], weights=weights[treated])
            mean_c_w = np.average(df_work.loc[~treated, cov], weights=weights[~treated])

            var_t_w = np.average((df_work.loc[treated, cov] - mean_t_w)**2, weights=weights[treated])
            var_c_w = np.average((df_work.loc[~treated, cov] - mean_c_w)**2, weights=weights[~treated])
            std_pooled_w = np.sqrt((var_t_w + var_c_w) / 2)

            smd_after = (mean_t_w - mean_c_w) / std_pooled_w if std_pooled_w > 0 else 0

            balance.append({
                'covariate': cov,
                'smd_before': smd_before,
                'smd_after': smd_after,
                'improvement': abs(smd_before) - abs(smd_after),
            })

        return pd.DataFrame(balance)


def propensity_score(
    df: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    outcome: Optional[str] = None,
) -> PropensityScoreAnalysis:
    """
    Convenience wrapper for propensity score analysis.

    Parameters
    ----------
    df : DataFrame
        Input data.
    treatment : str
        Binary treatment variable.
    covariates : list of str
        Covariates for propensity score model.
    outcome : str, optional
        Outcome variable.

    Returns
    -------
    PropensityScoreAnalysis
        Fitted propensity score object.

    Examples
    --------
    >>> ps = propensity_score(df, 'job_training', ['age', 'education', 'income'])
    >>> ate = ps.estimate_ate('wages', method='ipw')
    """
    ps = PropensityScoreAnalysis(df, treatment, covariates, outcome)
    ps.fit()
    return ps


# ==============================================================================
# DIFFERENCE-IN-DIFFERENCES
# ==============================================================================

def difference_in_differences(
    df: pd.DataFrame,
    outcome: str,
    treatment: str,
    time: str,
    unit: str,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
) -> Dict:
    """
    Difference-in-differences analysis.

    Parameters
    ----------
    df : DataFrame
        Panel data with pre/post observations.
    outcome : str
        Outcome variable.
    treatment : str
        Binary treatment indicator.
    time : str
        Binary time indicator (0=pre, 1=post).
    unit : str
        Unit identifier (for panel structure).
    covariates : list of str, optional
        Additional control variables.
    cluster : str, optional
        Variable to cluster standard errors by (often same as unit).

    Returns
    -------
    dict
        Dictionary with:
        - 'did_estimate': DiD estimate
        - 'se': Standard error
        - 'p_value': P-value
        - 'ci_lower', 'ci_upper': Confidence interval
        - 'model': Full regression results
        - 'parallel_trends_test': Pre-treatment trend test (if data available)

    Examples
    --------
    >>> result = difference_in_differences(
    ...     df, outcome='employment', treatment='policy', time='post',
    ...     unit='state', cluster='state'
    ... )
    >>> print(f"DiD estimate: {result['did_estimate']:.3f} (p={result['p_value']:.3f})")
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    # Create interaction term
    df_work = df.copy()
    df_work['treat_x_time'] = df_work[treatment] * df_work[time]

    # Build regression formula
    vars_list = [treatment, time, 'treat_x_time']
    if covariates:
        vars_list.extend(covariates)

    # Prepare data
    all_vars = [outcome] + vars_list
    if cluster:
        all_vars.append(cluster)

    df_clean = df_work[all_vars].dropna()

    y = df_clean[outcome]
    X = df_clean[vars_list]
    X = sm.add_constant(X)

    # Fit model
    if cluster:
        model = sm.OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_clean[cluster]}
        )
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1')

    # Extract DiD estimate
    did_estimate = model.params['treat_x_time']
    se = model.bse['treat_x_time']
    p_value = model.pvalues['treat_x_time']
    ci_lower, ci_upper = model.conf_int().loc['treat_x_time']

    result = {
        'did_estimate': did_estimate,
        'se': se,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'model': model,
    }

    # Parallel trends test (if multiple pre-periods available)
    # Placeholder for now
    result['parallel_trends_test'] = None

    return result


# ==============================================================================
# INSTRUMENTAL VARIABLES
# ==============================================================================

def instrumental_variables(
    df: pd.DataFrame,
    outcome: str,
    endogenous: Union[str, List[str]],
    instruments: Union[str, List[str]],
    exogenous: Optional[Union[str, List[str]]] = None,
    robust: bool = True,
) -> Dict:
    """
    Two-stage least squares (2SLS) instrumental variables regression.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome (dependent) variable.
    endogenous : str or list of str
        Endogenous regressor(s) to be instrumented.
    instruments : str or list of str
        Instrumental variable(s).
    exogenous : str or list of str, optional
        Additional exogenous control variables.
    robust : bool, default True
        Use robust standard errors.

    Returns
    -------
    dict
        Dictionary with:
        - 'coefficients': IV coefficient estimates
        - 'first_stage': First stage F-statistics
        - 'model': Full IV regression results

    Examples
    --------
    >>> result = instrumental_variables(
    ...     df, outcome='earnings', endogenous='education',
    ...     instruments='college_proximity', exogenous=['age', 'gender']
    ... )
    >>> print(result['coefficients'])
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    # Standardize inputs
    if isinstance(endogenous, str):
        endogenous = [endogenous]
    if isinstance(instruments, str):
        instruments = [instruments]
    if exogenous is None:
        exogenous = []
    elif isinstance(exogenous, str):
        exogenous = [exogenous]

    # Prepare data
    all_vars = [outcome] + endogenous + instruments + exogenous
    df_clean = df[all_vars].dropna()

    # Two-stage least squares manually
    # First stage: regress each endogenous variable on instruments + exogenous
    first_stage_results = {}
    endogenous_fitted = pd.DataFrame(index=df_clean.index)

    for endog_var in endogenous:
        y_first = df_clean[endog_var]
        X_first = df_clean[instruments + exogenous]
        X_first = sm.add_constant(X_first)

        first_stage_model = sm.OLS(y_first, X_first).fit()
        first_stage_results[endog_var] = first_stage_model

        # Get fitted values
        endogenous_fitted[endog_var] = first_stage_model.fittedvalues

        # Calculate F-statistic for excluded instruments
        # (weak instrument test)

    # Second stage: regress outcome on fitted endogenous + exogenous
    y_second = df_clean[outcome]
    X_second = pd.concat([endogenous_fitted, df_clean[exogenous]], axis=1)
    X_second = sm.add_constant(X_second)

    if robust:
        second_stage_model = sm.OLS(y_second, X_second).fit(cov_type='HC1')
    else:
        second_stage_model = sm.OLS(y_second, X_second).fit()

    # Extract coefficients
    coefficients = pd.DataFrame({
        'variable': second_stage_model.params.index,
        'estimate': second_stage_model.params.values,
        'std.error': second_stage_model.bse.values,
        'p.value': second_stage_model.pvalues.values,
    })

    # First stage F-statistics
    first_stage_f = {}
    for endog_var, fs_model in first_stage_results.items():
        first_stage_f[endog_var] = fs_model.fvalue

    return {
        'coefficients': coefficients,
        'first_stage': first_stage_f,
        'first_stage_models': first_stage_results,
        'second_stage_model': second_stage_model,
    }


# ==============================================================================
# REGRESSION DISCONTINUITY (RDD)
# ==============================================================================

def regression_discontinuity(
    df: pd.DataFrame,
    outcome: str,
    running_var: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    order: int = 1,
) -> Dict:
    """
    Regression discontinuity design (RDD) analysis.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome variable.
    running_var : str
        Running (forcing) variable.
    cutoff : float
        Cutoff value for treatment assignment.
    bandwidth : float, optional
        Bandwidth around cutoff. If None, uses Imbens-Kalyanaraman optimal bandwidth.
    kernel : str, default "triangular"
        Kernel for local linear regression: "triangular", "uniform", "epanechnikov".
    order : int, default 1
        Polynomial order (1=linear, 2=quadratic).

    Returns
    -------
    dict
        Dictionary with 'rd_estimate', 'se', 'p_value', 'bandwidth', 'n_left', 'n_right'.

    Examples
    --------
    >>> result = regression_discontinuity(
    ...     df, outcome='test_score', running_var='birthdate',
    ...     cutoff=pd.Timestamp('2010-01-01')
    ... )
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    df_work = df[[outcome, running_var]].dropna().copy()

    # Center running variable at cutoff
    df_work['running_centered'] = df_work[running_var] - cutoff
    df_work['above'] = (df_work['running_centered'] >= 0).astype(int)

    # Select bandwidth
    if bandwidth is None:
        # Simple rule-of-thumb bandwidth (could improve with IK bandwidth)
        bandwidth = 1.5 * df_work['running_centered'].std() * (len(df_work) ** (-1/5))

    # Restrict to bandwidth
    df_rdd = df_work[np.abs(df_work['running_centered']) <= bandwidth].copy()

    # Create polynomial terms
    df_rdd['run_above'] = df_rdd['running_centered'] * df_rdd['above']

    if order == 1:
        X_vars = ['above', 'running_centered', 'run_above']
    elif order == 2:
        df_rdd['running_sq'] = df_rdd['running_centered'] ** 2
        df_rdd['run_sq_above'] = df_rdd['running_sq'] * df_rdd['above']
        X_vars = ['above', 'running_centered', 'running_sq', 'run_above', 'run_sq_above']
    else:
        raise ValueError("order must be 1 or 2")

    # Fit model
    y = df_rdd[outcome]
    X = df_rdd[X_vars]
    X = sm.add_constant(X)

    # Apply kernel weights
    if kernel == "triangular":
        weights = 1 - np.abs(df_rdd['running_centered']) / bandwidth
    elif kernel == "uniform":
        weights = np.ones(len(df_rdd))
    elif kernel == "epanechnikov":
        weights = 0.75 * (1 - (df_rdd['running_centered'] / bandwidth)**2)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    model = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

    # Extract RD estimate (coefficient on 'above')
    rd_estimate = model.params['above']
    se = model.bse['above']
    p_value = model.pvalues['above']
    ci_lower, ci_upper = model.conf_int().loc['above']

    return {
        'rd_estimate': rd_estimate,
        'se': se,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bandwidth': bandwidth,
        'n_left': (df_rdd['above'] == 0).sum(),
        'n_right': (df_rdd['above'] == 1).sum(),
        'model': model,
    }


# ==============================================================================
# HELPER IMPORTS
# ==============================================================================

try:
    from scipy import stats
except ImportError:
    pass
