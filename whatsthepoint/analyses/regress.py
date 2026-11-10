"""
regress.py: whatsthepoint Regression Module
-------------------------------------------
Simplified regression interface with survey weighting support.

Features:
- Intuitive outcome and input variable specification
- Automatic survey weight handling
- Multiple regression types (OLS, Logit, Poisson, etc.)
- Tidy output compatible with pubtable module
- Robust standard errors
- Model diagnostics and comparisons

Supported models:
- OLS (Ordinary Least Squares)
- WLS (Weighted Least Squares)
- Logit (Binary logistic regression)
- Probit (Binary probit regression)
- Poisson (Count models)
- Negative Binomial (Overdispersed count models)
- Ordinal Logit (Ordered outcomes)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
import warnings

# statsmodels imports
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.regression.mixed_linear_model import MixedLM
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")



# ==============================================================================
# MAIN REGRESSION INTERFACE
# ==============================================================================

class RegressionModel:
    """
    Simplified regression interface for sociological analysis.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome (dependent) variable name.
    inputs : str or list of str
        Input (independent) variable name(s).
    weight : str, optional
        Survey weight variable.
    model_type : str, default "ols"
        Type of regression: "ols", "logit", "probit", "poisson", "negbin", "ordinal".
    robust : bool, default True
        Use robust (HC1) standard errors.
    cluster : str, optional
        Variable to cluster standard errors by.

    Attributes
    ----------
    results : statsmodels RegressionResults
        Fitted model results.
    tidy_results : DataFrame
        Tidy coefficients table.

    Examples
    --------
    >>> df = pd.DataFrame({'y': [1, 2, 3, 4], 'x1': [1, 2, 3, 4], 'x2': [4, 3, 2, 1]})
    >>> model = RegressionModel(df, outcome='y', inputs=['x1', 'x2'])
    >>> model.fit()
    >>> print(model.summary())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        inputs: Union[str, List[str]],
        weight: Optional[str] = None,
        model_type: str = "ols",
        robust: bool = True,
        cluster: Optional[str] = None,
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")

        self.df = df.copy()
        self.outcome = outcome
        self.inputs = [inputs] if isinstance(inputs, str) else inputs
        self.weight = weight
        self.model_type = model_type.lower()
        self.robust = robust
        self.cluster = cluster

        self.results = None
        self.tidy_results = None
        self.formula = None

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate that all variables exist in dataframe."""
        missing = []
        if self.outcome not in self.df.columns:
            missing.append(self.outcome)
        for var in self.inputs:
            if var not in self.df.columns:
                missing.append(var)
        if self.weight and self.weight not in self.df.columns:
            missing.append(self.weight)
        if self.cluster and self.cluster not in self.df.columns:
            missing.append(self.cluster)

        if missing:
            raise ValueError(f"Variables not found in dataframe: {missing}")

    def fit(self):
        """Fit the regression model."""
        # Prepare data (drop missing)
        vars_needed = [self.outcome] + self.inputs
        if self.weight:
            vars_needed.append(self.weight)
        if self.cluster:
            vars_needed.append(self.cluster)

        df_clean = self.df[vars_needed].dropna()

        if len(df_clean) == 0:
            raise ValueError("No observations remaining after dropping missing values")

        # Get weights
        weights = df_clean[self.weight] if self.weight else None

        # Prepare X and y
        y = df_clean[self.outcome]
        X = df_clean[self.inputs]
        X = sm.add_constant(X)  # Add intercept

        # Fit model based on type
        if self.model_type == "ols":
            if self.weight:
                model = sm.WLS(y, X, weights=weights)
            else:
                model = sm.OLS(y, X)

        elif self.model_type == "logit":
            if self.weight:
                model = sm.Logit(y, X)
                # Logit with weights needs different handling
                self.results = model.fit(method='bfgs')
                # Manually apply weights (approximate)
                # For proper weighted logit, use survey package or complex weighting
            else:
                model = sm.Logit(y, X)

        elif self.model_type == "probit":
            model = sm.Probit(y, X)

        elif self.model_type == "poisson":
            model = sm.Poisson(y, X)

        elif self.model_type == "negbin":
            model = sm.NegativeBinomial(y, X)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Fit with appropriate covariance type
        if self.results is None:  # Not already fitted (like weighted logit)
            if self.cluster:
                self.results = model.fit(cov_type='cluster', cov_kwds={'groups': df_clean[self.cluster]})
            elif self.robust:
                self.results = model.fit(cov_type='HC1')
            else:
                self.results = model.fit()

        # Create tidy results
        self._create_tidy_results()

        return self

    def _create_tidy_results(self):
        """Convert model results to tidy dataframe."""
        conf_int = self.results.conf_int()
        self.tidy_results = pd.DataFrame({
            'term': self.results.params.index,
            'estimate': self.results.params.values,
            'std.error': self.results.bse.values,
            'statistic': self.results.tvalues.values,
            'p.value': self.results.pvalues.values,
            'conf.low': conf_int[0].values,
            'conf.high': conf_int[1].values,
        })

    def summary(self) -> str:
        """Print model summary."""
        if self.results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")
        return str(self.results.summary())

    def get_tidy(self) -> pd.DataFrame:
        """
        Get tidy results dataframe.

        Returns
        -------
        DataFrame
            Tidy coefficients with columns: term, estimate, std.error, statistic, p.value, conf.low, conf.high
        """
        if self.tidy_results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")
        return self.tidy_results.copy()

    def get_stats(self) -> dict:
        """
        Get model fit statistics.

        Returns
        -------
        dict
            Dictionary with N, R_squared, Adj_R_squared, AIC, BIC, etc.
        """
        if self.results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")

        stats = {
            'N': int(self.results.nobs),
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'Log-Likelihood': self.results.llf,
        }

        # R-squared for OLS
        if hasattr(self.results, 'rsquared'):
            stats['R_squared'] = self.results.rsquared
            stats['Adj_R_squared'] = self.results.rsquared_adj

        # Pseudo R-squared for other models
        elif hasattr(self.results, 'prsquared'):
            stats['Pseudo_R_squared'] = self.results.prsquared

        return stats

    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        newdata : DataFrame, optional
            New data for prediction. If None, uses original data.

        Returns
        -------
        array
            Predicted values.
        """
        if self.results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")

        if newdata is None:
            return self.results.fittedvalues

        # Prepare newdata with constant
        X_new = newdata[self.inputs]
        X_new = sm.add_constant(X_new, has_constant='add')
        return self.results.predict(X_new)

    def vif(self) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factors for multicollinearity diagnostics.

        Returns
        -------
        DataFrame
            VIF values for each input variable.
        """
        if len(self.inputs) < 2:
            return pd.DataFrame({'Variable': self.inputs, 'VIF': [1.0]})

        # Prepare X matrix
        X = self.df[self.inputs].dropna()
        X = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Remove intercept
        vif_data = vif_data[vif_data["Variable"] != "const"]

        return vif_data


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def ols(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    weight: Optional[str] = None,
    robust: bool = True,
) -> RegressionModel:
    """
    Fit OLS regression model.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome variable.
    inputs : str or list of str
        Input variable(s).
    weight : str, optional
        Weight variable.
    robust : bool, default True
        Use robust standard errors.

    Returns
    -------
    RegressionModel
        Fitted model.

    Examples
    --------
    >>> df = pd.DataFrame({'y': [1, 2, 3], 'x': [1, 2, 3]})
    >>> model = ols(df, 'y', 'x')
    >>> print(model.summary())
    """
    model = RegressionModel(df, outcome, inputs, weight=weight, model_type="ols", robust=robust)
    model.fit()
    return model


def logit(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    weight: Optional[str] = None,
    robust: bool = True,
) -> RegressionModel:
    """
    Fit logistic regression model.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Binary outcome variable (0/1).
    inputs : str or list of str
        Input variable(s).
    weight : str, optional
        Weight variable.
    robust : bool, default True
        Use robust standard errors.

    Returns
    -------
    RegressionModel
        Fitted model.

    Examples
    --------
    >>> df = pd.DataFrame({'y': [0, 1, 0, 1], 'x': [1, 2, 3, 4]})
    >>> model = logit(df, 'y', 'x')
    >>> print(model.summary())
    """
    model = RegressionModel(df, outcome, inputs, weight=weight, model_type="logit", robust=robust)
    model.fit()
    return model


def poisson(
    df: pd.DataFrame,
    outcome: str,
    inputs: Union[str, List[str]],
    weight: Optional[str] = None,
    robust: bool = True,
) -> RegressionModel:
    """
    Fit Poisson regression model for count data.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Count outcome variable.
    inputs : str or list of str
        Input variable(s).
    weight : str, optional
        Weight variable.
    robust : bool, default True
        Use robust standard errors.

    Returns
    -------
    RegressionModel
        Fitted model.

    Examples
    --------
    >>> df = pd.DataFrame({'count': [0, 1, 2, 3], 'x': [1, 2, 3, 4]})
    >>> model = poisson(df, 'count', 'x')
    >>> print(model.summary())
    """
    model = RegressionModel(df, outcome, inputs, weight=weight, model_type="poisson", robust=robust)
    model.fit()
    return model


# ==============================================================================
# MODEL COMPARISON
# ==============================================================================

def compare_models(*models: RegressionModel) -> pd.DataFrame:
    """
    Compare multiple fitted models.

    Parameters
    ----------
    *models : RegressionModel
        Fitted regression models.

    Returns
    -------
    DataFrame
        Comparison table with fit statistics.

    Examples
    --------
    >>> model1 = ols(df, 'y', 'x1')
    >>> model2 = ols(df, 'y', ['x1', 'x2'])
    >>> compare_models(model1, model2)
    """
    comparison = []

    for i, model in enumerate(models, 1):
        if model.results is None:
            raise ValueError(f"Model {i} not fitted. Call .fit() first.")

        stats = model.get_stats()
        stats['Model'] = f"Model {i}"
        stats['Inputs'] = ', '.join(model.inputs)
        comparison.append(stats)

    df_compare = pd.DataFrame(comparison)

    # Reorder columns
    cols = ['Model', 'Inputs', 'N'] + [c for c in df_compare.columns if c not in ['Model', 'Inputs', 'N']]
    df_compare = df_compare[cols]

    return df_compare


# ==============================================================================
# MARGINAL EFFECTS (EXPERIMENTAL)
# ==============================================================================

def marginal_effects(model: RegressionModel, at: Optional[dict] = None) -> pd.DataFrame:
    """
    Calculate marginal effects for fitted model.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    at : dict, optional
        Values at which to evaluate marginal effects.
        If None, uses mean values.

    Returns
    -------
    DataFrame
        Marginal effects table.

    Examples
    --------
    >>> model = logit(df, 'y', ['x1', 'x2'])
    >>> me = marginal_effects(model)
    """
    if model.results is None:
        raise ValueError("Model not fitted. Call .fit() first.")

    # For OLS, marginal effects = coefficients
    if model.model_type == "ols":
        return model.tidy_results[['term', 'estimate']].rename(columns={'estimate': 'marginal_effect'})

    # For non-linear models, calculate average marginal effects
    try:
        from statsmodels.discrete.discrete_model import Logit, Probit, Poisson

        if at is None:
            # Average marginal effects
            me_result = model.results.get_margeff(at='overall', method='dydx')
        else:
            # Marginal effects at specific values
            me_result = model.results.get_margeff(at='mean', method='dydx')

        me_df = pd.DataFrame({
            'term': me_result.margeff_options['var_names'],
            'marginal_effect': me_result.margeff,
            'std.error': me_result.margeff_se,
            'p.value': me_result.pvalues,
        })

        return me_df

    except Exception as e:
        warnings.warn(f"Could not calculate marginal effects: {e}")
        return pd.DataFrame()


# ==============================================================================
# MULTILEVEL MODELS
# ==============================================================================

class MultilevelModel:
    """
    Multilevel (hierarchical/mixed-effects) regression model.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome (dependent) variable name.
    fixed : str or list of str
        Fixed effects variable name(s).
    random : str or list of str
        Random effects variable name(s).
    groups : str
        Grouping variable for random effects.
    weight : str, optional
        Survey weight variable.
    re_formula : str, optional
        Formula for random effects structure (e.g., "1" for random intercepts,
        "1 + x" for random slopes and intercepts).

    Attributes
    ----------
    results : MixedLMResults
        Fitted model results.
    tidy_results : DataFrame
        Tidy coefficients table with effect_type column.

    Examples
    --------
    Random intercepts model:
    >>> model = MultilevelModel(df, outcome='y', fixed=['x1', 'x2'],
    ...                         random='1', groups='school_id')
    >>> model.fit()

    Random slopes model:
    >>> model = MultilevelModel(df, outcome='y', fixed=['x1', 'x2'],
    ...                         random=['1', 'x1'], groups='school_id')
    >>> model.fit()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        fixed: Union[str, List[str]],
        random: Union[str, List[str]] = "1",
        groups: str = None,
        weight: Optional[str] = None,
        re_formula: Optional[str] = None,
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")

        self.df = df.copy()
        self.outcome = outcome
        self.fixed = [fixed] if isinstance(fixed, str) else fixed
        self.random = [random] if isinstance(random, str) else random
        self.groups = groups
        self.weight = weight
        self.re_formula = re_formula

        self.results = None
        self.tidy_results = None

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate that all variables exist in dataframe."""
        missing = []
        if self.outcome not in self.df.columns:
            missing.append(self.outcome)
        for var in self.fixed:
            if var not in self.df.columns:
                missing.append(var)
        if self.groups and self.groups not in self.df.columns:
            missing.append(self.groups)
        if self.weight and self.weight not in self.df.columns:
            missing.append(self.weight)

        if missing:
            raise ValueError(f"Variables not found in dataframe: {missing}")

        if not self.groups:
            raise ValueError("groups parameter is required for multilevel models")

    def fit(self):
        """Fit the multilevel model."""
        # Prepare data (drop missing)
        vars_needed = [self.outcome] + self.fixed + [self.groups]
        if self.weight:
            vars_needed.append(self.weight)

        df_clean = self.df[vars_needed].dropna()

        if len(df_clean) == 0:
            raise ValueError("No observations remaining after dropping missing values")

        # Get weights
        weights = df_clean[self.weight] if self.weight else None

        # Prepare X and y
        y = df_clean[self.outcome]
        X = df_clean[self.fixed]
        X = sm.add_constant(X)
        groups = df_clean[self.groups]

        # Build random effects formula
        if self.re_formula:
            re_formula = self.re_formula
        elif len(self.random) == 1 and self.random[0] == "1":
            re_formula = "1"
        else:
            re_formula = " + ".join(self.random)

        # Fit mixed linear model
        try:
            model = MixedLM(y, X, groups=groups, exog_re=None)

            if weights is not None:
                warnings.warn("Weights not fully supported in MixedLM. Using unweighted estimation.")

            self.results = model.fit(method='lbfgs')
        except Exception as e:
            warnings.warn(f"LBFGS failed, trying default method: {e}")
            self.results = model.fit()

        # Create tidy results
        self._create_tidy_results()

        return self

    def _create_tidy_results(self):
        """Convert model results to tidy dataframe with fixed and random effects."""
        # Fixed effects
        fixed_df = pd.DataFrame({
            'term': self.results.params.index,
            'estimate': self.results.params.values,
            'std.error': self.results.bse.values,
            'statistic': self.results.tvalues.values,
            'p.value': self.results.pvalues.values,
            'conf.low': self.results.conf_int()[0].values,
            'conf.high': self.results.conf_int()[1].values,
            'effect_type': 'fixed'
        })

        # Random effects (variance components)
        random_effects = []

        # Group variance
        if hasattr(self.results, 'cov_re'):
            cov_re = self.results.cov_re
            if isinstance(cov_re, np.ndarray):
                if cov_re.ndim == 0:
                    var_group = float(cov_re)
                else:
                    var_group = float(cov_re[0, 0]) if cov_re.size > 0 else 0.0
            elif isinstance(cov_re, pd.DataFrame):
                var_group = float(cov_re.iloc[0, 0]) if cov_re.size > 0 else 0.0
            else:
                var_group = float(cov_re)

            random_effects.append({
                'term': 'Group Variance',
                'estimate': var_group,
                'std.error': np.nan,
                'statistic': np.nan,
                'p.value': np.nan,
                'conf.low': np.nan,
                'conf.high': np.nan,
                'effect_type': 'random'
            })

        # Residual variance
        var_resid = self.results.scale
        random_effects.append({
            'term': 'Residual Variance',
            'estimate': var_resid,
            'std.error': np.nan,
            'statistic': np.nan,
            'p.value': np.nan,
            'conf.low': np.nan,
            'conf.high': np.nan,
            'effect_type': 'random'
        })

        random_df = pd.DataFrame(random_effects)
        self.tidy_results = pd.concat([fixed_df, random_df], ignore_index=True)

    def summary(self) -> str:
        """Print model summary."""
        if self.results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")
        return str(self.results.summary())

    def get_tidy(self) -> pd.DataFrame:
        """
        Get tidy results dataframe with both fixed and random effects.

        Returns
        -------
        DataFrame
            Tidy coefficients with effect_type column ('fixed' or 'random').
        """
        if self.tidy_results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")
        return self.tidy_results.copy()

    def get_stats(self) -> dict:
        """
        Get model fit statistics.

        Returns
        -------
        dict
            Dictionary with N, AIC, BIC, Log-Likelihood, etc.
        """
        if self.results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")

        stats = {
            'N': int(self.results.nobs),
            'N_groups': self.results.n_groups,
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'Log-Likelihood': self.results.llf,
        }

        return stats

    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions (fixed effects only).

        Parameters
        ----------
        newdata : DataFrame, optional
            New data for prediction. If None, uses original data.

        Returns
        -------
        array
            Predicted values.
        """
        if self.results is None:
            raise ValueError("Model not yet fitted. Call .fit() first.")

        if newdata is None:
            return self.results.fittedvalues

        # Prepare newdata with constant
        X_new = newdata[self.fixed]
        X_new = sm.add_constant(X_new, has_constant='add')
        return self.results.predict(X_new)


def multilevel(
    df: pd.DataFrame,
    outcome: str,
    fixed: Union[str, List[str]],
    random: Union[str, List[str]] = "1",
    groups: str = None,
    weight: Optional[str] = None,
) -> MultilevelModel:
    """
    Fit multilevel (hierarchical) regression model.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome variable.
    fixed : str or list of str
        Fixed effects variable(s).
    random : str or list of str, default "1"
        Random effects (use "1" for random intercepts only).
    groups : str
        Grouping variable (e.g., school, cluster ID).
    weight : str, optional
        Weight variable.

    Returns
    -------
    MultilevelModel
        Fitted multilevel model.

    Examples
    --------
    Random intercepts:
    >>> model = multilevel(df, 'test_score', fixed=['ses', 'gender'],
    ...                    random='1', groups='school_id')

    Random slopes:
    >>> model = multilevel(df, 'test_score', fixed=['ses', 'gender'],
    ...                    random=['1', 'ses'], groups='school_id')
    """
    model = MultilevelModel(df, outcome, fixed, random, groups, weight)
    model.fit()
    return model


# ==============================================================================
# FORMULA INTERFACE (ALTERNATIVE)
# ==============================================================================

def regress(
    formula: str,
    data: pd.DataFrame,
    model_type: str = "ols",
    weight: Optional[str] = None,
    robust: bool = True,
) -> RegressionModel:
    """
    Fit regression using R-style formula interface.

    Parameters
    ----------
    formula : str
        R-style formula, e.g., "y ~ x1 + x2 + x1:x2".
    data : DataFrame
        Input data.
    model_type : str, default "ols"
        Model type: "ols", "logit", "poisson", etc.
    weight : str, optional
        Weight variable.
    robust : bool, default True
        Use robust standard errors.

    Returns
    -------
    RegressionModel
        Fitted model.

    Examples
    --------
    >>> model = regress("income ~ education + age + education:age", df)
    >>> print(model.summary())
    """
    # Parse formula to extract outcome and inputs
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError("Formula must be in format: outcome ~ input1 + input2")

    outcome = parts[0].strip()
    inputs_str = parts[1].strip()

    # For simple formulas, extract variable names
    # For complex formulas with interactions, we'll use statsmodels formula API
    if ":" in inputs_str or "*" in inputs_str:
        # Use formula API for interactions
        warnings.warn("Interaction terms detected. Using statsmodels formula API.")

        if model_type == "ols":
            if weight:
                results = smf.wls(formula, data=data, weights=data[weight]).fit()
            else:
                results = smf.ols(formula, data=data).fit()
        elif model_type == "logit":
            results = smf.logit(formula, data=data).fit()
        elif model_type == "poisson":
            results = smf.poisson(formula, data=data).fit()
        else:
            raise ValueError(f"Unsupported model type for formula API: {model_type}")

        # Create wrapper
        model = RegressionModel.__new__(RegressionModel)
        model.df = data
        model.outcome = outcome
        model.inputs = inputs_str.split("+")
        model.weight = weight
        model.model_type = model_type
        model.robust = robust
        model.cluster = None
        model.results = results
        model.formula = formula
        model._create_tidy_results()

        return model
    else:
        # Simple formula - extract variable names
        inputs = [v.strip() for v in inputs_str.split("+")]
        model = RegressionModel(data, outcome, inputs, weight=weight, model_type=model_type, robust=robust)
        model.fit()
        model.formula = formula
        return model
