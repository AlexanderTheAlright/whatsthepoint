"""
descriptive.py: whatsthepoint Descriptive Statistics Module
------------------------------------------------------------
Descriptive statistics and exploratory data analysis.

Features:
- Correlation matrices (Pearson, Spearman, Kendall)
- Crosstabs with chi-square tests
- Weighted and grouped summaries
- Distribution diagnostics
- Normality tests

Functions:
- correlation_matrix: Correlation analysis with significance tests
- crosstab: Cross-tabulation with chi-square and effect sizes
- group_summary: Grouped descriptive statistics
- distribution_test: Test for normality and other distributional assumptions
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Tuple
import warnings

# scipy imports
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, pearsonr, spearmanr, kendalltau
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Install with: pip install scipy")



# ==============================================================================
# CORRELATION ANALYSIS
# ==============================================================================

def correlation_matrix(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    method: str = "pearson",
    min_periods: int = 10,
    show_pvalues: bool = True,
    weight: Optional[str] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate correlation matrix with significance tests.

    Parameters
    ----------
    df : DataFrame
        Input data.
    variables : list of str, optional
        Variables to correlate. If None, uses all numeric variables.
    method : str, default "pearson"
        Correlation method: "pearson", "spearman", or "kendall".
    min_periods : int, default 10
        Minimum number of observations required per pair.
    show_pvalues : bool, default True
        Return p-values in addition to correlations.
    weight : str, optional
        Weight variable for weighted correlations (only for Pearson).

    Returns
    -------
    DataFrame or tuple of DataFrames
        If show_pvalues=False: correlation matrix
        If show_pvalues=True: (correlation matrix, p-value matrix)

    Examples
    --------
    >>> corr = correlation_matrix(df, ['age', 'income', 'education'])
    >>> corr_mat, pval_mat = correlation_matrix(df, method='spearman')
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required. Install with: pip install scipy")

    # Select variables
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
        if weight and weight in variables:
            variables.remove(weight)

    df_subset = df[variables].copy()

    # Calculate correlation matrix
    if method.lower() == "pearson" and weight is None:
        corr_mat = df_subset.corr(method='pearson', min_periods=min_periods)
    elif method.lower() == "spearman":
        corr_mat = df_subset.corr(method='spearman', min_periods=min_periods)
    elif method.lower() == "kendall":
        corr_mat = df_subset.corr(method='kendall', min_periods=min_periods)
    elif method.lower() == "pearson" and weight:
        # Weighted correlation
        corr_mat = pd.DataFrame(index=variables, columns=variables, dtype=float)
        weights = df[weight]

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i == j:
                    corr_mat.loc[var1, var2] = 1.0
                else:
                    valid = ~(df_subset[var1].isna() | df_subset[var2].isna() | weights.isna())
                    if valid.sum() >= min_periods:
                        x = df_subset.loc[valid, var1]
                        y = df_subset.loc[valid, var2]
                        w = weights[valid]

                        # Weighted correlation
                        cov_xy = np.cov(x, y, aweights=w)[0, 1]
                        std_x = np.sqrt(np.cov(x, aweights=w))
                        std_y = np.sqrt(np.cov(y, aweights=w))
                        corr_mat.loc[var1, var2] = cov_xy / (std_x * std_y)
                    else:
                        corr_mat.loc[var1, var2] = np.nan
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not show_pvalues:
        return corr_mat

    # Calculate p-values
    pval_mat = pd.DataFrame(index=variables, columns=variables, dtype=float)

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                pval_mat.loc[var1, var2] = 0.0
            elif i < j:
                valid = ~(df_subset[var1].isna() | df_subset[var2].isna())
                if valid.sum() >= min_periods:
                    x = df_subset.loc[valid, var1]
                    y = df_subset.loc[valid, var2]

                    if method.lower() == "pearson":
                        _, p = pearsonr(x, y)
                    elif method.lower() == "spearman":
                        _, p = spearmanr(x, y)
                    elif method.lower() == "kendall":
                        _, p = kendalltau(x, y)

                    pval_mat.loc[var1, var2] = p
                    pval_mat.loc[var2, var1] = p
                else:
                    pval_mat.loc[var1, var2] = np.nan
                    pval_mat.loc[var2, var1] = np.nan

    return corr_mat, pval_mat


# ==============================================================================
# CROSS-TABULATION
# ==============================================================================

def crosstab(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    weight: Optional[str] = None,
    normalize: Optional[str] = None,
    show_chi2: bool = True,
    show_effect_size: bool = True,
) -> dict:
    """
    Cross-tabulation with chi-square test and effect sizes.

    Parameters
    ----------
    df : DataFrame
        Input data.
    row_var : str
        Row variable name.
    col_var : str
        Column variable name.
    weight : str, optional
        Weight variable.
    normalize : str, optional
        Normalization: None, 'all', 'index' (rows), 'columns'.
    show_chi2 : bool, default True
        Include chi-square test results.
    show_effect_size : bool, default True
        Include effect size measures (Cramér's V, phi).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'table': contingency table
        - 'proportions': proportion table (if normalize specified)
        - 'chi2': chi-square statistic (if show_chi2=True)
        - 'p_value': p-value (if show_chi2=True)
        - 'dof': degrees of freedom (if show_chi2=True)
        - 'cramers_v': Cramér's V effect size (if show_effect_size=True)
        - 'phi': Phi coefficient for 2x2 tables (if show_effect_size=True)

    Examples
    --------
    Simple crosstab:
    >>> result = crosstab(df, 'gender', 'vote_choice')
    >>> print(result['table'])
    >>> print(f"Chi-square: {result['chi2']:.3f}, p={result['p_value']:.3f}")

    Weighted with proportions:
    >>> result = crosstab(df, 'education', 'income_category',
    ...                    weight='survey_weight', normalize='index')
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required. Install with: pip install scipy")

    # Build contingency table
    if weight:
        table = pd.crosstab(
            df[row_var],
            df[col_var],
            values=df[weight],
            aggfunc='sum',
            dropna=False
        )
    else:
        table = pd.crosstab(df[row_var], df[col_var], dropna=False)

    result = {'table': table}

    # Proportions
    if normalize:
        if normalize == 'all':
            props = table / table.sum().sum()
        elif normalize == 'index':
            props = table.div(table.sum(axis=1), axis=0)
        elif normalize == 'columns':
            props = table.div(table.sum(axis=0), axis=1)
        else:
            raise ValueError("normalize must be 'all', 'index', or 'columns'")
        result['proportions'] = props

    # Chi-square test
    if show_chi2:
        chi2, p, dof, expected = chi2_contingency(table)
        result['chi2'] = chi2
        result['p_value'] = p
        result['dof'] = dof
        result['expected'] = pd.DataFrame(
            expected,
            index=table.index,
            columns=table.columns
        )

    # Effect sizes
    if show_effect_size and show_chi2:
        n = table.sum().sum()
        k = min(table.shape[0], table.shape[1])

        # Cramér's V
        cramers = np.sqrt(chi2 / (n * (k - 1)))
        result['cramers_v'] = cramers

        # Phi coefficient (for 2x2 tables)
        if table.shape == (2, 2):
            phi = np.sqrt(chi2 / n)
            result['phi'] = phi

    return result


# ==============================================================================
# GROUPED SUMMARIES
# ==============================================================================

def group_summary(
    df: pd.DataFrame,
    variables: Union[str, List[str]],
    group_var: str,
    weight: Optional[str] = None,
    stats: List[str] = ['mean', 'std', 'min', 'max', 'count'],
) -> pd.DataFrame:
    """
    Calculate grouped summary statistics.

    Parameters
    ----------
    df : DataFrame
        Input data.
    variables : str or list of str
        Variable(s) to summarize.
    group_var : str
        Grouping variable.
    weight : str, optional
        Weight variable for weighted statistics.
    stats : list of str, default ['mean', 'std', 'min', 'max', 'count']
        Statistics to calculate: 'mean', 'std', 'median', 'min', 'max',
        'count', 'sum', 'var', 'sem'.

    Returns
    -------
    DataFrame
        Grouped summary statistics.

    Examples
    --------
    >>> summary = group_summary(df, ['income', 'education'], 'region')
    >>> summary = group_summary(df, 'satisfaction', 'gender',
    ...                          weight='weight', stats=['mean', 'sem'])
    """
    if isinstance(variables, str):
        variables = [variables]

    if weight:
        # Weighted statistics
        results = []

        for var in variables:
            for group_val in df[group_var].unique():
                mask = df[group_var] == group_val
                data = df.loc[mask, var]
                weights = df.loc[mask, weight]

                # Remove NaN
                valid = ~(data.isna() | weights.isna())
                data = data[valid]
                weights = weights[valid]

                row = {'variable': var, group_var: group_val}

                if 'count' in stats:
                    row['count'] = len(data)
                if 'sum' in stats:
                    row['sum'] = np.sum(data * weights) / np.sum(weights) * len(data)
                if 'mean' in stats:
                    row['mean'] = np.average(data, weights=weights)
                if 'std' in stats:
                    mean = np.average(data, weights=weights)
                    variance = np.average((data - mean)**2, weights=weights)
                    row['std'] = np.sqrt(variance)
                if 'var' in stats:
                    mean = np.average(data, weights=weights)
                    row['var'] = np.average((data - mean)**2, weights=weights)
                if 'median' in stats:
                    # Weighted median (approximate)
                    sorted_idx = np.argsort(data)
                    cumsum = np.cumsum(weights.iloc[sorted_idx])
                    cutoff = cumsum[-1] / 2
                    median_idx = sorted_idx[cumsum >= cutoff][0]
                    row['median'] = data.iloc[median_idx]
                if 'min' in stats:
                    row['min'] = data.min()
                if 'max' in stats:
                    row['max'] = data.max()
                if 'sem' in stats:
                    mean = np.average(data, weights=weights)
                    variance = np.average((data - mean)**2, weights=weights)
                    row['sem'] = np.sqrt(variance / len(data))

                results.append(row)

        return pd.DataFrame(results)

    else:
        # Unweighted statistics
        grouped = df.groupby(group_var)[variables]

        # Build aggregation list/dict
        agg_funcs = []
        for stat in stats:
            if stat in ['mean', 'std', 'median', 'min', 'max', 'count', 'sum', 'var']:
                agg_funcs.append(stat)
            elif stat == 'sem':
                agg_funcs.append(('sem', lambda x: x.std() / np.sqrt(len(x))))

        result = grouped.agg(agg_funcs)

        # Flatten column names if multi-variable
        if len(variables) > 1 and isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(str(c) for c in col).strip() for col in result.columns.values]

        return result.reset_index()


# ==============================================================================
# DISTRIBUTION TESTS
# ==============================================================================

def distribution_test(
    df: pd.DataFrame,
    variable: str,
    test: str = "shapiro",
    group_var: Optional[str] = None,
) -> pd.DataFrame:
    """
    Test distributional assumptions.

    Parameters
    ----------
    df : DataFrame
        Input data.
    variable : str
        Variable to test.
    test : str, default "shapiro"
        Test type:
        - "shapiro": Shapiro-Wilk test for normality
        - "ks": Kolmogorov-Smirnov test for normality
        - "anderson": Anderson-Darling test for normality
        - "jarque_bera": Jarque-Bera test for normality
        - "levene": Levene's test for equal variances (requires group_var)
    group_var : str, optional
        Grouping variable (required for Levene's test).

    Returns
    -------
    DataFrame
        Test results with statistic and p-value.

    Examples
    --------
    >>> result = distribution_test(df, 'income', test='shapiro')
    >>> print(f"Shapiro-Wilk W={result['statistic'][0]:.3f}, p={result['p_value'][0]:.3f}")

    Test equal variances:
    >>> result = distribution_test(df, 'income', test='levene', group_var='region')
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required. Install with: pip install scipy")

    data = df[variable].dropna()

    if test.lower() == "shapiro":
        stat, p = stats.shapiro(data)
        return pd.DataFrame({
            'test': ['Shapiro-Wilk'],
            'statistic': [stat],
            'p_value': [p]
        })

    elif test.lower() == "ks":
        # KS test against normal distribution
        mean, std = data.mean(), data.std()
        stat, p = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))
        return pd.DataFrame({
            'test': ['Kolmogorov-Smirnov'],
            'statistic': [stat],
            'p_value': [p]
        })

    elif test.lower() == "anderson":
        result = stats.anderson(data, dist='norm')
        return pd.DataFrame({
            'test': ['Anderson-Darling'],
            'statistic': [result.statistic],
            'critical_values': [result.critical_values],
            'significance_levels': [result.significance_level]
        })

    elif test.lower() == "jarque_bera":
        stat, p = stats.jarque_bera(data)
        return pd.DataFrame({
            'test': ['Jarque-Bera'],
            'statistic': [stat],
            'p_value': [p]
        })

    elif test.lower() == "levene":
        if not group_var:
            raise ValueError("group_var required for Levene's test")

        groups = [df.loc[df[group_var] == g, variable].dropna()
                  for g in df[group_var].unique()]
        stat, p = stats.levene(*groups)

        return pd.DataFrame({
            'test': ['Levene'],
            'statistic': [stat],
            'p_value': [p]
        })

    else:
        raise ValueError(f"Unsupported test: {test}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def describe_by_group(
    df: pd.DataFrame,
    variables: Union[str, List[str]],
    group_var: str,
    weight: Optional[str] = None,
) -> pd.DataFrame:
    """
    Grouped descriptive statistics.

    Wrapper around group_summary with common statistics.

    Parameters
    ----------
    df : DataFrame
        Input data.
    variables : str or list of str
        Variable(s) to describe.
    group_var : str
        Grouping variable.
    weight : str, optional
        Weight variable.

    Returns
    -------
    DataFrame
        Grouped descriptive statistics.

    Examples
    --------
    >>> desc = describe_by_group(df, 'income', 'education')
    """
    return group_summary(
        df, variables, group_var, weight=weight,
        stats=['count', 'mean', 'std', 'min', 'max']
    )


def compare_groups(
    df: pd.DataFrame,
    variable: str,
    group_var: str,
    test: str = "auto",
) -> dict:
    """
    Compare distributions across groups with appropriate test.

    Parameters
    ----------
    df : DataFrame
        Input data.
    variable : str
        Variable to compare.
    group_var : str
        Grouping variable.
    test : str, default "auto"
        Test type: "auto", "ttest", "anova", "kruskal", "mannwhitney".
        If "auto", selects based on number of groups and normality.

    Returns
    -------
    dict
        Test results with statistic, p-value, and test name.

    Examples
    --------
    >>> result = compare_groups(df, 'income', 'education')
    >>> print(f"{result['test']}: F={result['statistic']:.3f}, p={result['p_value']:.3f}")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required. Install with: pip install scipy")

    groups = [df.loc[df[group_var] == g, variable].dropna()
              for g in df[group_var].unique()]
    n_groups = len(groups)

    if test == "auto":
        # Auto-select test
        if n_groups == 2:
            # Check normality
            all_normal = all(
                stats.shapiro(g)[1] > 0.05 if len(g) >= 3 else True
                for g in groups
            )
            test = "ttest" if all_normal else "mannwhitney"
        else:
            # Check normality
            all_normal = all(
                stats.shapiro(g)[1] > 0.05 if len(g) >= 3 else True
                for g in groups
            )
            test = "anova" if all_normal else "kruskal"

    if test == "ttest":
        if n_groups != 2:
            raise ValueError("t-test requires exactly 2 groups")
        stat, p = stats.ttest_ind(*groups)
        return {'test': 't-test', 'statistic': stat, 'p_value': p}

    elif test == "mannwhitney":
        if n_groups != 2:
            raise ValueError("Mann-Whitney test requires exactly 2 groups")
        stat, p = stats.mannwhitneyu(*groups)
        return {'test': 'Mann-Whitney U', 'statistic': stat, 'p_value': p}

    elif test == "anova":
        stat, p = stats.f_oneway(*groups)
        return {'test': 'One-way ANOVA', 'statistic': stat, 'p_value': p}

    elif test == "kruskal":
        stat, p = stats.kruskal(*groups)
        return {'test': 'Kruskal-Wallis', 'statistic': stat, 'p_value': p}

    else:
        raise ValueError(f"Unsupported test: {test}")
