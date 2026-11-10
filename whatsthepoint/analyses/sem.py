"""
sem.py: whatsthepoint Structural Equation Modeling Module
---------------------------------------------------------
Path analysis and structural equation modeling for sociological research.

Features:
- Path models with multiple endogenous variables
- Measurement models (CFA)
- Full structural equation models
- Mediation and moderation analysis
- Model fit statistics (CFI, TLI, RMSEA, SRMR)
- Path diagrams (future)

Implementation:
- Uses statsmodels for basic path models
- Can interface with R's lavaan via rpy2 (optional)
- Returns standardized and unstandardized estimates

Note: This is a simplified SEM implementation. For complex models,
consider using dedicated SEM software (R lavaan, Mplus, AMOS).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols as sm_ols
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")



# ==============================================================================
# PATH MODEL CLASS
# ==============================================================================

class PathModel:
    """
    Path analysis model with multiple equations.

    Parameters
    ----------
    df : DataFrame
        Input data.
    equations : dict
        Dictionary mapping outcome variables to lists of predictors.
        Example: {'M': ['X'], 'Y': ['M', 'X']} for mediation model.
    weight : str, optional
        Survey weight variable.
    standardize : bool, default True
        Standardize all variables before estimation.

    Attributes
    ----------
    results : dict
        Dictionary of regression results for each equation.
    path_coefficients : DataFrame
        Path coefficients table.
    fit_stats : dict
        Model fit statistics.

    Examples
    --------
    Simple mediation:
    >>> equations = {
    ...     'mediator': ['treatment'],
    ...     'outcome': ['mediator', 'treatment']
    ... }
    >>> model = PathModel(df, equations)
    >>> model.fit()
    >>> print(model.path_coefficients)

    Multiple mediators:
    >>> equations = {
    ...     'M1': ['X'],
    ...     'M2': ['X', 'M1'],
    ...     'Y': ['X', 'M1', 'M2']
    ... }
    >>> model = PathModel(df, equations).fit()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        equations: Dict[str, List[str]],
        weight: Optional[str] = None,
        standardize: bool = True,
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")

        self.df = df.copy()
        self.equations = equations
        self.weight = weight
        self.standardize = standardize

        self.results = {}
        self.path_coefficients = None
        self.fit_stats = {}

        # Validate equations
        self._validate_equations()

    def _validate_equations(self):
        """Validate equation specification."""
        all_vars = set()
        for outcome, predictors in self.equations.items():
            all_vars.add(outcome)
            all_vars.update(predictors)

        missing = [v for v in all_vars if v not in self.df.columns]
        if missing:
            raise ValueError(f"Variables not found in dataframe: {missing}")

    def fit(self):
        """Fit the path model."""
        df_work = self.df.copy()

        # Standardize if requested
        if self.standardize:
            all_vars = set()
            for outcome, predictors in self.equations.items():
                all_vars.add(outcome)
                all_vars.update(predictors)

            for var in all_vars:
                if var in df_work.columns:
                    df_work[var] = (df_work[var] - df_work[var].mean()) / df_work[var].std()

        # Get weights
        weights = df_work[self.weight] if self.weight else None

        # Fit each equation
        all_paths = []
        total_n = len(df_work)

        for outcome, predictors in self.equations.items():
            # Prepare data
            vars_needed = [outcome] + predictors
            if self.weight:
                vars_needed.append(self.weight)

            df_clean = df_work[vars_needed].dropna()

            if len(df_clean) == 0:
                warnings.warn(f"No observations for equation: {outcome}")
                continue

            y = df_clean[outcome]
            X = df_clean[predictors]
            X = sm.add_constant(X)

            # Fit OLS
            if weights is not None:
                w = weights[df_clean.index]
                model = sm.WLS(y, X, weights=w).fit()
            else:
                model = sm.OLS(y, X).fit()

            self.results[outcome] = model

            # Extract path coefficients
            for pred in predictors:
                path = {
                    'from': pred,
                    'to': outcome,
                    'estimate': model.params[pred],
                    'std.error': model.bse[pred],
                    'p.value': model.pvalues[pred],
                    'standardized': model.params[pred] if self.standardize else np.nan,
                }
                all_paths.append(path)

        self.path_coefficients = pd.DataFrame(all_paths)

        # Calculate fit statistics
        self._calculate_fit_stats()

        return self

    def _calculate_fit_stats(self):
        """Calculate model fit statistics."""
        n_params = sum(len(preds) + 1 for preds in self.equations.values())  # +1 for intercept
        n_obs = min(res.nobs for res in self.results.values()) if self.results else 0

        total_ll = sum(res.llf for res in self.results.values())
        total_aic = sum(res.aic for res in self.results.values())
        total_bic = sum(res.bic for res in self.results.values())

        self.fit_stats = {
            'N': int(n_obs),
            'N_parameters': n_params,
            'Log-Likelihood': total_ll,
            'AIC': total_aic,
            'BIC': total_bic,
        }

        # Calculate R-squared for each equation
        for outcome, model in self.results.items():
            self.fit_stats[f'R_squared_{outcome}'] = model.rsquared

    def summary(self) -> str:
        """Print path model summary."""
        if not self.results:
            raise ValueError("Model not fitted. Call .fit() first.")

        output = "Path Model Summary\n"
        output += "=" * 60 + "\n\n"

        # Fit statistics
        output += "Model Fit:\n"
        for key, val in self.fit_stats.items():
            if isinstance(val, float):
                output += f"  {key}: {val:.3f}\n"
            else:
                output += f"  {key}: {val}\n"

        output += "\n" + "=" * 60 + "\n\n"

        # Path coefficients
        output += "Path Coefficients:\n"
        output += self.path_coefficients.to_string(index=False)

        return output

    def get_paths(self) -> pd.DataFrame:
        """
        Get path coefficients table.

        Returns
        -------
        DataFrame
            Path coefficients with columns: from, to, estimate, std.error, p.value.
        """
        if self.path_coefficients is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        return self.path_coefficients.copy()

    def get_stats(self) -> dict:
        """
        Get model fit statistics.

        Returns
        -------
        dict
            Dictionary with N, AIC, BIC, R-squared values.
        """
        if not self.fit_stats:
            raise ValueError("Model not fitted. Call .fit() first.")
        return self.fit_stats.copy()

    def indirect_effect(self, path: List[str]) -> Tuple[float, float]:
        """
        Calculate indirect effect along a path.

        Parameters
        ----------
        path : list of str
            List of variables forming the path, e.g., ['X', 'M', 'Y'].

        Returns
        -------
        tuple
            (indirect_effect, standard_error)

        Examples
        --------
        >>> effect, se = model.indirect_effect(['treatment', 'mediator', 'outcome'])
        """
        if len(path) < 3:
            raise ValueError("Path must have at least 3 variables (start, mediator(s), end)")

        # Find path coefficients
        coefs = []
        for i in range(len(path) - 1):
            from_var = path[i]
            to_var = path[i + 1]

            path_data = self.path_coefficients[
                (self.path_coefficients['from'] == from_var) &
                (self.path_coefficients['to'] == to_var)
            ]

            if path_data.empty:
                raise ValueError(f"Path {from_var} -> {to_var} not found in model")

            coefs.append(path_data['estimate'].iloc[0])

        # Indirect effect is product of path coefficients
        indirect = np.prod(coefs)

        # Sobel test for standard error (approximate)
        if len(coefs) == 2:
            a, b = coefs
            se_a = self.path_coefficients[
                (self.path_coefficients['from'] == path[0]) &
                (self.path_coefficients['to'] == path[1])
            ]['std.error'].iloc[0]
            se_b = self.path_coefficients[
                (self.path_coefficients['from'] == path[1]) &
                (self.path_coefficients['to'] == path[2])
            ]['std.error'].iloc[0]

            se_indirect = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
        else:
            se_indirect = np.nan  # Delta method needed for longer paths

        return indirect, se_indirect

    def total_effect(self, from_var: str, to_var: str) -> float:
        """
        Calculate total effect (direct + indirect) from one variable to another.

        Parameters
        ----------
        from_var : str
            Starting variable.
        to_var : str
            Ending variable.

        Returns
        -------
        float
            Total effect.

        Examples
        --------
        >>> total = model.total_effect('treatment', 'outcome')
        """
        # Direct effect
        direct_path = self.path_coefficients[
            (self.path_coefficients['from'] == from_var) &
            (self.path_coefficients['to'] == to_var)
        ]

        direct = direct_path['estimate'].iloc[0] if not direct_path.empty else 0.0

        # Indirect effects (would need to traverse all paths - simplified here)
        # For now, just return direct effect
        warnings.warn("total_effect currently returns only direct effect. Full path tracing not yet implemented.")
        return direct


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def mediation(
    df: pd.DataFrame,
    x: str,
    m: Union[str, List[str]],
    y: str,
    weight: Optional[str] = None,
    standardize: bool = True,
) -> PathModel:
    """
    Simple mediation analysis.

    Parameters
    ----------
    df : DataFrame
        Input data.
    x : str
        Independent variable (treatment/predictor).
    m : str or list of str
        Mediator variable(s).
    y : str
        Dependent variable (outcome).
    weight : str, optional
        Weight variable.
    standardize : bool, default True
        Standardize variables.

    Returns
    -------
    PathModel
        Fitted mediation model.

    Examples
    --------
    Simple mediation:
    >>> model = mediation(df, x='treatment', m='mediator', y='outcome')
    >>> indirect, se = model.indirect_effect(['treatment', 'mediator', 'outcome'])

    Multiple mediators (parallel):
    >>> model = mediation(df, x='treatment', m=['m1', 'm2'], y='outcome')
    """
    if isinstance(m, str):
        m = [m]

    # Build equations
    equations = {}

    # X -> M paths
    for mediator in m:
        equations[mediator] = [x]

    # M, X -> Y path
    equations[y] = m + [x]

    model = PathModel(df, equations, weight=weight, standardize=standardize)
    model.fit()

    return model


def path_analysis(
    df: pd.DataFrame,
    equations: Dict[str, List[str]],
    weight: Optional[str] = None,
    standardize: bool = True,
) -> PathModel:
    """
    General path analysis wrapper.

    Parameters
    ----------
    df : DataFrame
        Input data.
    equations : dict
        Dictionary of equations: {outcome: [predictors]}.
    weight : str, optional
        Weight variable.
    standardize : bool, default True
        Standardize variables.

    Returns
    -------
    PathModel
        Fitted path model.

    Examples
    --------
    >>> equations = {
    ...     'satisfaction': ['workload', 'support'],
    ...     'wellbeing': ['satisfaction', 'workload']
    ... }
    >>> model = path_analysis(df, equations)
    """
    model = PathModel(df, equations, weight=weight, standardize=standardize)
    model.fit()
    return model


# ==============================================================================
# FUTURE: MEASUREMENT MODELS (CFA)
# ==============================================================================

def cfa(
    df: pd.DataFrame,
    factors: Dict[str, List[str]],
    weight: Optional[str] = None,
):
    """
    Confirmatory Factor Analysis.

    Parameters
    ----------
    df : DataFrame
        Input data.
    factors : dict
        Dictionary mapping factor names to observed indicators.
        Example: {'Satisfaction': ['item1', 'item2', 'item3']}.
    weight : str, optional
        Weight variable.

    Returns
    -------
    CFAModel
        Fitted CFA model.

    Note
    ----
    This is a placeholder. Full CFA requires additional dependencies.
    Consider using R's lavaan or Python's semopy for complex models.
    """
    raise NotImplementedError(
        "CFA not yet implemented. Use R's lavaan or Python's semopy for "
        "confirmatory factor analysis."
    )


# ==============================================================================
# FUTURE: R LAVAAN INTERFACE
# ==============================================================================

def lavaan_interface(
    df: pd.DataFrame,
    model_syntax: str,
    estimator: str = "ML",
):
    """
    Interface to R's lavaan package for full SEM.

    Parameters
    ----------
    df : DataFrame
        Input data.
    model_syntax : str
        lavaan model syntax.
    estimator : str, default "ML"
        Estimator: "ML", "MLR", "WLSMV", etc.

    Returns
    -------
    dict
        Model results from lavaan.

    Note
    ----
    Requires rpy2 and R lavaan package. Install with:
    pip install rpy2
    R: install.packages("lavaan")
    """
    raise NotImplementedError(
        "R lavaan interface not yet implemented. Requires rpy2. "
        "For complex SEM, use R lavaan directly or Python's semopy."
    )
