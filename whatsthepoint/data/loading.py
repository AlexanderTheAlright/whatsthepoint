"""
Data Loading & Preparation Utilities
=====================================

Flexible data loading with automatic ID detection and categorical handling.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import warnings

__all__ = [
    'load_stata',
    'load_all_surveys',
    'normalize_ids',
    'load_and_combine',
]


def load_stata(filepath: Union[str, Path],
               convert_categoricals: bool = True,
               auto_detect_id: bool = True,
               lowercase_cols: bool = True,
               id_candidates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load a Stata file, with error handling and preprocessing.

    Parameters
    ----------
    filepath : str or Path
        Path to .dta file.
    convert_categoricals : bool, default True
        Whether to convert labeled values to pandas Categorical.
    auto_detect_id : bool, default True
        Whether to automatically detect ID columns.
    lowercase_cols : bool, default True
        Whether to standardize column names to lowercase.
    id_candidates : list of str, optional
        List of potential ID column names to look for (e.g., ['indid', 'pid', 'respondent_id']).

    Returns
    -------
    DataFrame
        Loaded and preprocessed data with metadata attached as df.attrs.

    Examples
    --------
    >>> df = load_stata('survey_2020.dta', auto_detect_id=True)
    >>> print(df.attrs.get('id_column'))
    'indid'
    >>> print(df.attrs.get('variable_labels'))
    {'age': 'Respondent age', 'income': 'Annual income'}
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if id_candidates is None:
        id_candidates = ['indid', 'pid', 'respondent_id', 'id', 'caseid']

    # Try loading with categoricals first
    try:
        df = pd.read_stata(
            filepath,
            convert_categoricals=convert_categoricals,
            order_categoricals=True
        )
    except (ValueError, KeyError) as e:
        if "not unique" in str(e).lower():
            warnings.warn(
                f"Non-unique value labels in {filepath.name}. "
                f"Loading without categorical conversion."
            )
            df = pd.read_stata(filepath, convert_categoricals=False)
        else:
            raise

    # Extract variable labels
    try:
        with pd.read_stata(filepath, iterator=True) as reader:
            variable_labels = reader.variable_labels()
            df.attrs['variable_labels'] = variable_labels
    except Exception as e:
        warnings.warn(f"Could not extract variable labels: {e}")
        df.attrs['variable_labels'] = {}

    # Standardize column names
    if lowercase_cols:
        col_map = {col: col.strip().lower().replace(' ', '_') for col in df.columns}
        df = df.rename(columns=col_map)

        # Also update variable labels
        if 'variable_labels' in df.attrs:
            df.attrs['variable_labels'] = {
                col_map.get(k, k).lower(): v
                for k, v in df.attrs['variable_labels'].items()
            }

    # Auto-detect ID column
    if auto_detect_id:
        id_col = _detect_id_column(df, id_candidates)
        df.attrs['id_column'] = id_col

    # Store file metadata
    df.attrs['source_file'] = str(filepath)
    df.attrs['n_rows'] = len(df)
    df.attrs['n_cols'] = len(df.columns)

    return df


def load_all_surveys(data_dir: Union[str, Path],
                     file_extensions: Optional[List[str]] = None,
                     target_vars: Optional[List[str]] = None,
                     **load_kwargs) -> Dict[str, pd.DataFrame]:
    """
    Load all survey files from a directory with optional variable filtering.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing survey files.
    file_extensions : list of str, optional
        File types to load. Defaults to ['.dta', '.csv', '.xlsx'].
    target_vars : list of str, optional
        Specific variables to keep (subset). If None, keeps all variables.
    **load_kwargs
        Additional arguments passed to load_stata() or pd.read_csv()/pd.read_excel().

    Returns
    -------
    dict
        Dictionary keyed by filename (without extension) containing DataFrames.

    Examples
    --------
    >>> surveys = load_all_surveys('data/', target_vars=['age', 'income', 'satisfaction'])
    >>> print(surveys.keys())
    dict_keys(['survey_2020', 'survey_2021', 'survey_2022'])
    >>> print(surveys['survey_2020'].columns)
    Index(['age', 'income', 'satisfaction'], dtype='object')
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if file_extensions is None:
        file_extensions = ['.dta', '.csv', '.xlsx']

    surveys = {}
    errors = []

    for ext in file_extensions:
        for filepath in data_dir.glob(f"*{ext}"):
            survey_id = filepath.stem

            try:
                # Load based on file type
                if ext == '.dta':
                    df = load_stata(filepath, **load_kwargs)
                elif ext == '.csv':
                    df = pd.read_csv(filepath)
                    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                    df.attrs['source_file'] = str(filepath)
                elif ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(filepath)
                    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                    df.attrs['source_file'] = str(filepath)
                else:
                    continue

                # Filter to target variables if specified
                if target_vars is not None:
                    available_vars = [v for v in target_vars if v.lower() in df.columns]
                    if available_vars:
                        df = df[available_vars]
                    else:
                        warnings.warn(
                            f"{survey_id}: None of the target variables found. Skipping."
                        )
                        continue

                surveys[survey_id] = df

            except Exception as e:
                errors.append((filepath.name, str(e)))
                warnings.warn(f"Failed to load {filepath.name}: {e}")

    # Summary
    print(f"Loaded {len(surveys)} surveys")
    if errors:
        print(f"Failed to load {len(errors)} files")

    return surveys


def _detect_id_column(df: pd.DataFrame,
                      candidates: List[str]) -> Optional[str]:
    """
    Automatically identify the ID column.

    Parameters
    ----------
    df : DataFrame
        Input data.
    candidates : list of str
        List of potential ID column names.

    Returns
    -------
    str or None
        Name of detected ID column, or None if not found.
    """
    # Check explicit candidates first
    for candidate in candidates:
        if candidate in df.columns:
            col = df[candidate]
            # Check if it looks like an ID (unique values, numeric or string)
            if col.nunique() == len(df):
                return candidate

    # Check first column as fallback
    first_col = df.columns[0]
    if df[first_col].nunique() == len(df):
        return first_col

    # Check for columns with 'id' in the name
    for col in df.columns:
        if 'id' in col.lower():
            if df[col].nunique() == len(df):
                return col

    return None


def normalize_ids(df: pd.DataFrame,
                  id_cols: Optional[Union[str, List[str]]] = None,
                  remove_decimals: bool = True,
                  convert_to_int: bool = True,
                  handle_missing: str = 'keep') -> pd.DataFrame:
    """
    Normalize ID columns by removing .0 suffixes and standardizing format.

    Parameters
    ----------
    df : DataFrame
        Input data.
    id_cols : str or list of str, optional
        ID column name(s) to normalize. If None, attempts auto-detection.
    remove_decimals : bool, default True
        Remove .0 suffixes from numeric IDs (e.g., 123.0 -> 123).
    convert_to_int : bool, default True
        Convert to integer type if possible (after removing decimals).
    handle_missing : {'keep', 'drop', 'fill'}, default 'keep'
        How to handle missing IDs:
        - 'keep': Keep as NaN
        - 'drop': Remove rows with missing IDs
        - 'fill': Fill with sequential integers

    Returns
    -------
    DataFrame
        Data with normalized ID columns.

    Examples
    --------
    >>> df = pd.DataFrame({'id': [1.0, 2.0, 3.0], 'value': [10, 20, 30]})
    >>> df_norm = normalize_ids(df, id_cols='id')
    >>> print(df_norm['id'].dtype)
    int64
    """
    df = df.copy()

    # Auto-detect ID columns if not specified
    if id_cols is None:
        id_candidates = ['indid', 'pid', 'respondent_id', 'id', 'caseid']
        detected_id = _detect_id_column(df, id_candidates)
        if detected_id:
            id_cols = [detected_id]
        else:
            warnings.warn("Could not auto-detect ID column. Please specify id_cols.")
            return df

    # Ensure list
    if isinstance(id_cols, str):
        id_cols = [id_cols]

    for id_col in id_cols:
        if id_col not in df.columns:
            warnings.warn(f"ID column '{id_col}' not found in DataFrame")
            continue

        id_series = df[id_col].copy()

        # Handle missing values first
        if handle_missing == 'drop':
            df = df[id_series.notna()]
            id_series = df[id_col].copy()
        elif handle_missing == 'fill':
            missing_mask = id_series.isna()
            if missing_mask.any():
                # Fill with sequential IDs starting after max
                max_id = id_series.max() if id_series.notna().any() else 0
                fill_ids = range(int(max_id) + 1, int(max_id) + 1 + missing_mask.sum())
                id_series.loc[missing_mask] = list(fill_ids)

        # Remove decimals if requested
        if remove_decimals:
            if pd.api.types.is_numeric_dtype(id_series):
                # Check if all values are effectively integers (no fractional part)
                non_null = id_series.dropna()
                if len(non_null) > 0:
                    if (non_null % 1 == 0).all():
                        if convert_to_int:
                            id_series = id_series.astype('Int64')  # Nullable integer type
                        else:
                            id_series = id_series.astype(float).round(0)
            elif pd.api.types.is_string_dtype(id_series):
                # Remove .0 from string representations
                id_series = id_series.str.replace(r'\.0+$', '', regex=True)

                # Try converting to int if possible
                if convert_to_int:
                    try:
                        id_series = pd.to_numeric(id_series, errors='coerce').astype('Int64')
                    except:
                        pass

        df[id_col] = id_series

    return df


def load_and_combine(data_dir: Union[str, Path],
                     file_types: Optional[List[str]] = None,
                     include_patterns: Optional[List[str]] = None,
                     exclude_patterns: Optional[List[str]] = None,
                     target_vars: Optional[List[str]] = None,
                     combine_method: str = 'concat',
                     id_col: Optional[str] = None,
                     normalize_id: bool = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load multiple datasets and optionally combine them.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing data files.
    file_types : list of str, optional
        File extensions to load. Defaults to all supported types.
    include_patterns : list of str, optional
        Only load files matching these patterns.
    exclude_patterns : list of str, optional
        Exclude files matching these patterns.
    target_vars : list of str, optional
        Specific variables to keep. If None, keeps all.
    combine_method : {'concat', 'merge', 'separate'}, default 'concat'
        How to combine datasets:
        - 'concat': Stack all datasets (long form)
        - 'merge': Merge on ID column (wide form)
        - 'separate': Return dict of separate DataFrames
    id_col : str, optional
        ID column for merging. Auto-detected if None.
    normalize_id : bool, default True
        Whether to normalize ID columns (remove .0, etc.).

    Returns
    -------
    DataFrame or dict
        Combined data or dict of separate DataFrames.

    Examples
    --------
    >>> # Load and stack all datasets
    >>> df_long = load_and_combine('data/', combine_method='concat')

    >>> # Load only survey files
    >>> df = load_and_combine('data/', include_patterns=['survey_*'])

    >>> # Load and merge on ID
    >>> df_wide = load_and_combine('data/', combine_method='merge', id_col='pid')

    >>> # Load but keep separate
    >>> data_dict = load_and_combine('data/', combine_method='separate')
    """
    data_dir = Path(data_dir)

    # Use discover_data for flexible loading
    from .discovery import discover_data

    data_dict = discover_data(
        data_dir=data_dir,
        file_types=file_types,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        auto_detect_types=True
    )

    if not data_dict:
        warnings.warn("No datasets loaded")
        return {} if combine_method == 'separate' else pd.DataFrame()

    # Filter to target variables if specified
    if target_vars is not None:
        for survey_id in list(data_dict.keys()):
            df = data_dict[survey_id]
            available_vars = [v for v in target_vars if v.lower() in df.columns]

            if id_col and id_col.lower() in df.columns and id_col.lower() not in available_vars:
                available_vars.insert(0, id_col.lower())

            if available_vars:
                data_dict[survey_id] = df[available_vars]
            else:
                warnings.warn(f"{survey_id}: No target variables found. Excluding.")
                del data_dict[survey_id]

    # Normalize IDs if requested
    if normalize_id:
        for survey_id in data_dict:
            data_dict[survey_id] = normalize_ids(data_dict[survey_id], id_cols=id_col)

    # Return based on combine method
    if combine_method == 'separate':
        return data_dict

    elif combine_method == 'concat':
        # Stack all datasets
        dfs = []
        for survey_id, df in data_dict.items():
            df_copy = df.copy()
            df_copy['source'] = survey_id
            dfs.append(df_copy)

        combined = pd.concat(dfs, axis=0, ignore_index=True)

        # Extract wave info from source
        import re
        combined['wave'] = combined['source'].str.extract(r'(\d+)', expand=False)
        if combined['wave'].notna().any():
            combined['wave'] = pd.to_numeric(combined['wave'], errors='coerce')

        return combined

    elif combine_method == 'merge':
        # Merge all datasets on ID
        if id_col is None:
            # Try to detect ID column
            first_df = next(iter(data_dict.values()))
            id_col = _detect_id_column(first_df, ['indid', 'pid', 'id', 'caseid'])

        if id_col is None:
            raise ValueError("Could not detect ID column for merging. Please specify id_col.")

        # Start with first dataset
        combined = None
        for survey_id, df in data_dict.items():
            if id_col not in df.columns:
                warnings.warn(f"{survey_id}: ID column '{id_col}' not found. Skipping.")
                continue

            # Rename columns to avoid conflicts (except ID)
            rename_map = {col: f"{col}_{survey_id}" for col in df.columns if col != id_col}
            df_renamed = df.rename(columns=rename_map)

            if combined is None:
                combined = df_renamed
            else:
                combined = combined.merge(df_renamed, on=id_col, how='outer')

        return combined if combined is not None else pd.DataFrame()

    else:
        raise ValueError(f"Invalid combine_method: {combine_method}. Choose 'concat', 'merge', or 'separate'.")
