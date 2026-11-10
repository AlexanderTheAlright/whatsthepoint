"""
Harmonization & Pre-Analysis Preparation Utilities
===================================================

Tools for harmonizing variables, building combined datasets, and preparing for analysis.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import warnings

__all__ = [
    'harmonize_columns',
    'build_harmonized_dataset',
    'to_categorical_ordered',
    'numeric_codes',
    'prepare_for_analysis',
]


def harmonize_columns(df: pd.DataFrame,
                      var_mappings: Dict[str, Dict[str, str]],
                      handle_missing: str = 'drop') -> pd.DataFrame:
    """
    Standardize categorical orders and labels across variables.

    Parameters
    ----------
    df : DataFrame
        Input data.
    var_mappings : dict
        Nested dict: {var_name: {old_value: new_value}}.
    handle_missing : {'drop', 'na', 'keep'}, default 'drop'
        How to handle missing value indicators:
        - 'drop': Remove rows with missing indicators
        - 'na': Convert to NaN
        - 'keep': Keep as-is

    Returns
    -------
    DataFrame
        Data with harmonized variables.

    Examples
    --------
    >>> mappings = {
    ...     'satisfaction': {
    ...         'very satisfied': 'Very satisfied',
    ...         'satisfied': 'Satisfied',
    ...         'neutral': 'Neutral',
    ...         'dissatisfied': 'Dissatisfied',
    ...         'very dissatisfied': 'Very dissatisfied'
    ...     }
    ... }
    >>> df_harmonized = harmonize_columns(df, mappings, handle_missing='na')
    """
    df = df.copy()

    for var, mapping in var_mappings.items():
        if var not in df.columns:
            warnings.warn(f"Variable '{var}' not found in DataFrame")
            continue

        # Apply mapping
        df[var] = df[var].map(lambda x: mapping.get(x, x))

        # Handle missing values
        if handle_missing == 'na':
            missing_indicators = ['dk', "don't know", 'refused', 'n/a', 'not applicable']
            df[var] = df[var].replace(
                {val: np.nan for val in missing_indicators if val in df[var].values}
            )
        elif handle_missing == 'drop':
            missing_indicators = ['dk', "don't know", 'refused', 'n/a', 'not applicable']
            df = df[~df[var].isin(missing_indicators)]

    return df


def build_harmonized_dataset(data_dict: Dict[str, pd.DataFrame],
                              target_vars: List[str],
                              id_col: str,
                              add_identifiers: bool = True) -> pd.DataFrame:
    """
    Combine shared target variables across surveys into one long-form dataset.

    Parameters
    ----------
    data_dict : dict
        Dictionary of {survey_id: DataFrame}.
    target_vars : list of str
        Variables to include in harmonized dataset.
    id_col : str
        ID column name.
    add_identifiers : bool, default True
        Whether to add surveyid, year, and wave columns.

    Returns
    -------
    DataFrame
        Combined long-form dataset with harmonized variables.

    Examples
    --------
    >>> data = {'survey_2020': df1, 'survey_2021': df2}
    >>> harmonized = build_harmonized_dataset(
    ...     data,
    ...     target_vars=['age', 'income', 'satisfaction'],
    ...     id_col='pid'
    ... )
    """
    dfs = []

    for survey_id, df in data_dict.items():
        # Select available target variables
        available = [v for v in target_vars if v in df.columns]

        if not available:
            warnings.warn(f"{survey_id}: No target variables found. Skipping.")
            continue

        # Subset data
        keep_cols = [id_col] + available if id_col in df.columns else available
        subset = df[keep_cols].copy()

        # Add identifiers
        if add_identifiers:
            subset['surveyid'] = survey_id

            # Extract year from survey_id if possible
            import re
            year_match = re.search(r'(\d{4})', survey_id)
            if year_match:
                subset['year'] = int(year_match.group(1))
            else:
                subset['year'] = np.nan

        dfs.append(subset)

    # Concatenate
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    # Add wave labels
    if add_identifiers and 'surveyid' in combined.columns:
        unique_surveys = sorted(combined['surveyid'].unique())
        wave_map = {sid: f"wave_{i+1}" for i, sid in enumerate(unique_surveys)}
        combined['wave'] = combined['surveyid'].map(wave_map)

    return combined


def to_categorical_ordered(df: pd.DataFrame,
                           column: str,
                           order_list: List[str]) -> pd.DataFrame:
    """
    Convert a column to ordered categorical with specified order.

    Parameters
    ----------
    df : DataFrame
        Input data.
    column : str
        Column to convert.
    order_list : list of str
        Desired category order (lowest to highest).

    Returns
    -------
    DataFrame
        Data with converted column.

    Examples
    --------
    >>> order = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
    >>> df = to_categorical_ordered(df, 'satisfaction', order)
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Create ordered categorical
    cat_dtype = CategoricalDtype(categories=order_list, ordered=True)
    df[column] = df[column].astype(cat_dtype)

    return df


def numeric_codes(series: pd.Series,
                  start: int = 1) -> pd.Series:
    """
    Convert ordered categorical series to numeric codes.

    Parameters
    ----------
    series : Series
        Categorical series to convert.
    start : int, default 1
        Starting numeric code.

    Returns
    -------
    Series
        Numeric codes.

    Examples
    --------
    >>> satisfaction = pd.Categorical(
    ...     ['Agree', 'Disagree', 'Agree', 'Neutral'],
    ...     categories=['Disagree', 'Neutral', 'Agree'],
    ...     ordered=True
    ... )
    >>> codes = numeric_codes(satisfaction)
    >>> print(codes)
    0    3
    1    1
    2    3
    3    2
    """
    if not isinstance(series.dtype, CategoricalDtype):
        raise ValueError("Series must be categorical")

    if not series.dtype.ordered:
        warnings.warn("Series is not ordered. Results may not be meaningful.")

    # Create mapping
    categories = series.cat.categories
    code_map = {cat: i + start for i, cat in enumerate(categories)}

    return series.map(code_map).astype(float)


def prepare_for_analysis(df: pd.DataFrame,
                         satisfaction_vars: Optional[List[str]] = None,
                         weight_col: Optional[str] = None,
                         normalize_weights: bool = True,
                         apply_harmonization: bool = True,
                         min_valid_pct: float = 0.5) -> pd.DataFrame:
    """
    Apply harmonization rules and prepare dataset for analysis.

    Parameters
    ----------
    df : DataFrame
        Input data.
    satisfaction_vars : list of str, optional
        Variables to ensure have proper categorical ordering.
    weight_col : str, optional
        Name of weight column to normalize.
    normalize_weights : bool, default True
        Whether to normalize weights to sum to N.
    apply_harmonization : bool, default True
        Whether to apply standard harmonization rules.
    min_valid_pct : float, default 0.5
        Minimum proportion of non-missing values per row to keep.

    Returns
    -------
    DataFrame
        Analysis-ready dataset.

    Examples
    --------
    >>> df_ready = prepare_for_analysis(
    ...     df,
    ...     satisfaction_vars=['jobsat', 'satguess'],
    ...     weight_col='weight',
    ...     min_valid_pct=0.6
    ... )
    """
    df = df.copy()

    # Apply harmonization
    if apply_harmonization:
        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Handle common missing value codes
        missing_codes = [-99, -98, -97, 999, 998, 997]
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(missing_codes, np.nan)

    # Ensure categorical ordering for satisfaction variables
    if satisfaction_vars:
        for var in satisfaction_vars:
            if var in df.columns:
                # Check if already categorical
                if not isinstance(df[var].dtype, CategoricalDtype):
                    # Try to infer ordering
                    unique_vals = df[var].dropna().unique()
                    if len(unique_vals) <= 10:  # Reasonable for Likert
                        df[var] = pd.Categorical(df[var], ordered=True)

    # Normalize weights
    if weight_col and weight_col in df.columns:
        if normalize_weights:
            n = len(df)
            weight_sum = df[weight_col].sum()
            if weight_sum > 0:
                df[weight_col] = df[weight_col] * (n / weight_sum)

    # Filter rows with too much missing data
    if min_valid_pct > 0:
        valid_pct = df.notna().mean(axis=1)
        n_before = len(df)
        df = df[valid_pct >= min_valid_pct]
        n_after = len(df)

        if n_after < n_before:
            print(f"Removed {n_before - n_after} rows with >{(1-min_valid_pct)*100:.0f}% missing data")

    return df
