"""
Longitudinal Recognition & Ordering Utilities
==============================================

Tools for detecting, aligning, and managing panel/longitudinal survey data.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings

__all__ = [
    'detect_longitudinal',
    'align_longitudinal_data',
    'sort_by_wave',
    'build_longit_from_dir',
]


def detect_longitudinal(data_dict: Dict[str, pd.DataFrame],
                        id_col: Optional[str] = None,
                        auto_detect_id: bool = True) -> Dict[str, any]:
    """
    Detect whether datasets contain repeated measures (panel/longitudinal structure).

    Parameters
    ----------
    data_dict : dict
        Dictionary of {survey_id: DataFrame}.
    id_col : str, optional
        Name of ID column to check across datasets. If None, attempts auto-detection.
    auto_detect_id : bool, default True
        Whether to automatically detect ID columns if not specified.

    Returns
    -------
    dict
        Contains:
        - 'is_longitudinal': bool indicating if data has panel structure
        - 'shared_ids': set of IDs appearing in multiple waves
        - 'id_column': name of the ID column used
        - 'overlap_matrix': DataFrame showing ID overlap between surveys

    Examples
    --------
    >>> data = {'wave1': df1, 'wave2': df2, 'wave3': df3}
    >>> result = detect_longitudinal(data, id_col='pid')
    >>> print(result['is_longitudinal'])
    True
    >>> print(f"Shared IDs: {len(result['shared_ids'])}")
    Shared IDs: 450
    """
    if not data_dict:
        raise ValueError("data_dict is empty")

    # Auto-detect ID column if not provided
    if id_col is None and auto_detect_id:
        # Try to find common ID column across datasets
        for survey_id, df in data_dict.items():
            if 'id_column' in df.attrs:
                id_col = df.attrs['id_column']
                break

        # Fallback: check common candidates
        if id_col is None:
            candidates = ['indid', 'pid', 'respondent_id', 'id', 'caseid']
            first_df = next(iter(data_dict.values()))
            for candidate in candidates:
                if candidate in first_df.columns:
                    id_col = candidate
                    break

    if id_col is None:
        warnings.warn("Could not detect ID column. Please specify id_col parameter.")
        return {
            'is_longitudinal': False,
            'shared_ids': set(),
            'id_column': None,
            'overlap_matrix': None,
        }

    # Check if ID column exists in all datasets
    missing_id = [sid for sid, df in data_dict.items() if id_col not in df.columns]
    if missing_id:
        warnings.warn(f"ID column '{id_col}' not found in: {missing_id}")
        return {
            'is_longitudinal': False,
            'shared_ids': set(),
            'id_column': id_col,
            'overlap_matrix': None,
        }

    # Collect all IDs from each survey
    id_sets = {}
    for survey_id, df in data_dict.items():
        id_sets[survey_id] = set(df[id_col].dropna().unique())

    # Find shared IDs
    all_ids = set.union(*id_sets.values()) if id_sets else set()
    shared_ids = all_ids.copy()

    for id_set in id_sets.values():
        shared_ids &= id_set

    # Create overlap matrix
    survey_ids = list(data_dict.keys())
    overlap_matrix = pd.DataFrame(
        index=survey_ids,
        columns=survey_ids,
        dtype=int
    )

    for sid1 in survey_ids:
        for sid2 in survey_ids:
            if sid1 == sid2:
                overlap_matrix.loc[sid1, sid2] = len(id_sets[sid1])
            else:
                overlap = len(id_sets[sid1] & id_sets[sid2])
                overlap_matrix.loc[sid1, sid2] = overlap

    is_longitudinal = len(shared_ids) > 0

    return {
        'is_longitudinal': is_longitudinal,
        'shared_ids': shared_ids,
        'id_column': id_col,
        'overlap_matrix': overlap_matrix,
    }


def align_longitudinal_data(data_dict: Dict[str, pd.DataFrame],
                            id_col: str,
                            wave_labels: Optional[Dict[str, str]] = None,
                            align_vars: Optional[List[str]] = None,
                            merge_type: str = 'outer') -> pd.DataFrame:
    """
    Merge or align longitudinal data by ID across detected waves.

    Parameters
    ----------
    data_dict : dict
        Dictionary of {survey_id: DataFrame}.
    id_col : str
        Name of ID column for merging.
    wave_labels : dict, optional
        Mapping of survey_id to wave labels (e.g., {'survey_2020': 'wave_1'}).
    align_vars : list of str, optional
        Specific variables to include. If None, uses all common variables.
    merge_type : {'outer', 'inner', 'left'}, default 'outer'
        Type of merge to perform.

    Returns
    -------
    DataFrame
        Merged longitudinal dataset with columns:
        - ID column
        - wave indicator
        - source survey indicator
        - aligned variables

    Examples
    --------
    >>> data = {'survey_2020': df1, 'survey_2021': df2}
    >>> waves = {'survey_2020': 'wave_1', 'survey_2021': 'wave_2'}
    >>> long_df = align_longitudinal_data(data, id_col='pid', wave_labels=waves)
    >>> print(long_df.columns)
    Index(['pid', 'wave', 'source', 'age', 'income', 'satisfaction'], dtype='object')
    """
    if not data_dict:
        raise ValueError("data_dict is empty")

    # Generate wave labels if not provided
    if wave_labels is None:
        wave_labels = {
            survey_id: f"wave_{i+1}"
            for i, survey_id in enumerate(sorted(data_dict.keys()))
        }

    # Find common variables if not specified
    if align_vars is None:
        # Get intersection of column names (excluding ID)
        all_cols = [set(df.columns) - {id_col} for df in data_dict.values()]
        align_vars = list(set.intersection(*all_cols)) if all_cols else []

    # Build long-form dataset
    long_dfs = []

    for survey_id, df in data_dict.items():
        # Select ID + aligned variables
        keep_cols = [id_col] + [v for v in align_vars if v in df.columns]
        subset = df[keep_cols].copy()

        # Add wave and source indicators
        subset['wave'] = wave_labels.get(survey_id, survey_id)
        subset['source'] = survey_id

        long_dfs.append(subset)

    # Concatenate all waves
    long_df = pd.concat(long_dfs, axis=0, ignore_index=True)

    # Ensure consistent variable naming and type alignment
    long_df = _align_variable_types(long_df, align_vars)

    # Sort by ID and wave
    long_df = long_df.sort_values([id_col, 'wave']).reset_index(drop=True)

    return long_df


def sort_by_wave(df: pd.DataFrame,
                 id_col: str,
                 wave_col: str = 'wave',
                 verify: bool = True) -> pd.DataFrame:
    """
    Sort concatenated longitudinal data by ID and chronological wave.

    Parameters
    ----------
    df : DataFrame
        Long-form longitudinal data.
    id_col : str
        Name of ID column.
    wave_col : str, default 'wave'
        Name of wave indicator column.
    verify : bool, default True
        Whether to verify wave order integrity.

    Returns
    -------
    DataFrame
        Sorted longitudinal data.

    Examples
    --------
    >>> long_df = sort_by_wave(long_df, id_col='pid', wave_col='wave')
    >>> print(long_df.head())
       pid  wave  age  income
    0    1     1   25   50000
    1    1     2   26   52000
    2    1     3   27   54000
    """
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame")

    if wave_col not in df.columns:
        raise ValueError(f"Wave column '{wave_col}' not found in DataFrame")

    # Extract wave number for proper sorting
    df = df.copy()

    if df[wave_col].dtype == 'object':
        # Try to extract numeric wave number
        wave_nums = df[wave_col].str.extract(r'(\d+)', expand=False)
        if wave_nums.notna().all():
            df['_wave_num'] = wave_nums.astype(int)
            sort_col = '_wave_num'
        else:
            sort_col = wave_col
    else:
        sort_col = wave_col

    # Sort
    df = df.sort_values([id_col, sort_col]).reset_index(drop=True)

    # Clean up temporary column
    if '_wave_num' in df.columns:
        df = df.drop(columns=['_wave_num'])

    # Verify wave order
    if verify:
        issues = _verify_wave_order(df, id_col, wave_col)
        if issues:
            warnings.warn(f"Wave order issues detected: {issues}")

    return df


def _align_variable_types(df: pd.DataFrame,
                          align_vars: List[str]) -> pd.DataFrame:
    """
    Ensure consistent variable types across waves.

    Parameters
    ----------
    df : DataFrame
        Long-form data.
    align_vars : list of str
        Variables to align.

    Returns
    -------
    DataFrame
        Data with aligned types.
    """
    for var in align_vars:
        if var not in df.columns:
            continue

        # Check if variable has mixed types
        var_series = df[var]

        # Convert to most appropriate common type
        if var_series.dtype == 'object':
            # Try numeric first
            try:
                numeric = pd.to_numeric(var_series, errors='coerce')
                if numeric.notna().sum() > len(var_series) * 0.8:
                    df[var] = numeric
            except:
                pass

    return df


def _verify_wave_order(df: pd.DataFrame,
                       id_col: str,
                       wave_col: str) -> List[str]:
    """
    Verify that waves are in chronological order for each ID.

    Parameters
    ----------
    df : DataFrame
        Long-form data.
    id_col : str
        ID column name.
    wave_col : str
        Wave column name.

    Returns
    -------
    list of str
        List of issues found (empty if no issues).
    """
    issues = []

    # Check for duplicate ID-wave combinations
    duplicates = df.groupby([id_col, wave_col]).size()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) > 0:
        issues.append(f"Found {len(duplicates)} duplicate ID-wave combinations")

    # Check for gaps in wave sequence
    grouped = df.groupby(id_col)[wave_col]

    return issues


def build_longit_from_dir(data_dir: Union[str, 'Path'],
                           target_vars: Optional[List[str]] = None,
                           id_col: Optional[str] = None,
                           file_types: Optional[List[str]] = None,
                           include_patterns: Optional[List[str]] = None,
                           exclude_patterns: Optional[List[str]] = None,
                           normalize_ids: bool = True,
                           auto_detect_waves: bool = True) -> pd.DataFrame:
    """
    Build long-form longitudinal dataset from directory of files.

    This function:
    1. Discovers all data files in directory
    2. Auto-detects wave structure from filenames
    3. Normalizes IDs across datasets (handles .0, etc.)
    4. Combines into long-form dataset with wave indicators

    Parameters
    ----------
    data_dir : str or Path
        Directory containing longitudinal data files.
    target_vars : list of str, optional
        Variables to include. If None, uses common variables across all files.
    id_col : str, optional
        ID column name. Auto-detected if None.
    file_types : list of str, optional
        File types to load. Defaults to all supported types.
    include_patterns : list of str, optional
        Only load files matching these patterns.
    exclude_patterns : list of str, optional
        Exclude files matching these patterns.
    normalize_ids : bool, default True
        Whether to normalize IDs (remove .0, standardize format).
    auto_detect_waves : bool, default True
        Whether to auto-detect wave info from filenames.

    Returns
    -------
    DataFrame
        Long-form dataset with columns: id, wave, source, [target_vars]

    Examples
    --------
    >>> # Load all surveys from directory
    >>> df_long = build_longit_from_dir('data/surveys/')

    >>> # Load specific variables only
    >>> df_long = build_longit_from_dir(
    ...     'data/',
    ...     target_vars=['age', 'income', 'satisfaction'],
    ...     id_col='pid'
    ... )

    >>> # Load only files matching pattern
    >>> df_long = build_longit_from_dir(
    ...     'data/',
    ...     include_patterns=['survey_wave*']
    ... )
    """
    from pathlib import Path
    from .loading import load_and_combine, normalize_ids as normalize_id_func
    from .metadata import extract_wave_info

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Load all datasets
    print(f"Loading data from {data_dir}...")
    data_dict = load_and_combine(
        data_dir=data_dir,
        file_types=file_types,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        target_vars=target_vars,
        combine_method='separate',
        id_col=id_col,
        normalize_id=normalize_ids
    )

    if not data_dict:
        raise ValueError("No datasets loaded from directory")

    # Extract wave information
    if auto_detect_waves:
        print("Detecting wave structure...")
        wave_info = extract_wave_info(data_dict)
        wave_labels = dict(zip(wave_info['survey_id'], wave_info['wave_id']))
    else:
        wave_labels = {sid: f"wave_{i+1}" for i, sid in enumerate(sorted(data_dict.keys()))}

    # Detect ID column if not specified
    if id_col is None:
        first_df = next(iter(data_dict.values()))
        from .loading import _detect_id_column
        id_col = _detect_id_column(first_df, ['indid', 'pid', 'respondent_id', 'id', 'caseid'])

        if id_col is None:
            raise ValueError("Could not auto-detect ID column. Please specify id_col parameter.")

    print(f"Using ID column: {id_col}")

    # Normalize IDs across all datasets if requested
    if normalize_ids:
        print("Normalizing IDs...")
        for survey_id in data_dict:
            data_dict[survey_id] = normalize_id_func(data_dict[survey_id], id_cols=id_col)

    # Find common variables if target_vars not specified
    if target_vars is None:
        all_cols = [set(df.columns) - {id_col} for df in data_dict.values()]
        target_vars = list(set.intersection(*all_cols)) if all_cols else []
        print(f"Found {len(target_vars)} common variables across datasets")

    # Build long-form dataset
    print("Building long-form dataset...")
    long_df = align_longitudinal_data(
        data_dict=data_dict,
        id_col=id_col,
        wave_labels=wave_labels,
        align_vars=target_vars
    )

    # Sort by ID and wave
    long_df = sort_by_wave(long_df, id_col=id_col, wave_col='wave', verify=False)

    print(f"Created long-form dataset: {len(long_df)} observations")
    print(f"  - {long_df[id_col].nunique()} unique individuals")
    print(f"  - {long_df['wave'].nunique()} waves")
    print(f"  - {len(target_vars)} variables")

    return long_df
