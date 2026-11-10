"""
Metadata & Structure Utilities
===============================

Survey metadata extraction, summarization, and ID variable detection.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

__all__ = [
    'summarize_surveys',
    'extract_wave_info',
    'get_id_vars',
]


def summarize_surveys(data_dict: Dict[str, pd.DataFrame],
                      key_constructs: Optional[List[str]] = None,
                      return_df: bool = True) -> Optional[pd.DataFrame]:
    """
    Generate summary table of survey datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary of {survey_id: DataFrame}.
    key_constructs : list of str, optional
        Variable names to check for presence (e.g., ['jobsat', 'satguess', 'perceivedinequality']).
    return_df : bool, default True
        Whether to return DataFrame or just print.

    Returns
    -------
    DataFrame or None
        Summary table if return_df=True, otherwise None.

    Examples
    --------
    >>> data = load_all_surveys('surveys/')
    >>> summary = summarize_surveys(data, key_constructs=['age', 'income', 'satisfaction'])
    >>> print(summary)
         survey_id     N  n_vars  has_age  has_income  has_satisfaction
    0  survey_2020  1000      50     True        True              True
    1  survey_2021  1200      52     True        True              True
    """
    if key_constructs is None:
        key_constructs = []

    # Standardize construct names
    key_constructs = [k.lower() for k in key_constructs]

    rows = []

    for survey_id, df in data_dict.items():
        row = {
            'survey_id': survey_id,
            'N': len(df),
            'n_vars': len(df.columns),
        }

        # Detect ID column
        id_col = df.attrs.get('id_column', 'unknown')
        row['id_column'] = id_col

        # Check for key constructs
        df_cols_lower = [c.lower() for c in df.columns]
        for construct in key_constructs:
            row[f'has_{construct}'] = construct in df_cols_lower

        # Additional metadata from attrs
        if 'source_file' in df.attrs:
            row['source_file'] = Path(df.attrs['source_file']).name

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not return_df:
        print("\n" + "="*80)
        print("SURVEY SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80 + "\n")
        return None

    return summary_df


def extract_wave_info(data_source: Union[List[str], Dict[str, pd.DataFrame], str, 'Path'],
                      pattern: Optional[str] = None,
                      auto_detect: bool = True) -> pd.DataFrame:
    """
    Extract time information from survey filenames, IDs, or directory.

    Parameters
    ----------
    data_source : list, dict, str, or Path
        Can be:
        - List of survey IDs (filenames without extensions)
        - Dict of {survey_id: DataFrame}
        - Path to directory (will scan for files)
    pattern : str, optional
        Regex pattern to extract date components.
        Defaults to extracting 4-digit year: r'(\\d{4})'
    auto_detect : bool, default True
        Whether to auto-detect wave patterns from filenames.

    Returns
    -------
    DataFrame
        Contains columns: survey_id, year, month (if found), date (if found), wave_id

    Examples
    --------
    >>> # From list of IDs
    >>> ids = ['survey_jan_2020', 'survey_mar_2021', 'survey_dec_2022']
    >>> wave_info = extract_wave_info(ids)

    >>> # From dictionary
    >>> data = {'survey_2020': df1, 'survey_2021': df2}
    >>> wave_info = extract_wave_info(data)

    >>> # From directory
    >>> wave_info = extract_wave_info('data/surveys/')
    """
    from pathlib import Path

    # Convert data_source to list of survey IDs
    if isinstance(data_source, dict):
        survey_ids = list(data_source.keys())
    elif isinstance(data_source, (str, Path)):
        # Scan directory for files
        data_dir = Path(data_source)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        survey_ids = []
        for filepath in data_dir.glob('*'):
            if filepath.is_file() and filepath.suffix in ['.dta', '.csv', '.xlsx', '.xls', '.sav']:
                survey_ids.append(filepath.stem)

        if not survey_ids:
            raise ValueError(f"No data files found in {data_dir}")
    elif isinstance(data_source, list):
        survey_ids = data_source
    else:
        raise TypeError("data_source must be list, dict, str, or Path")

    if pattern is None:
        pattern = r'(\d{4})'  # Default: 4-digit year

    rows = []

    for survey_id in survey_ids:
        row = {'survey_id': survey_id}

        # Extract year
        year_match = re.search(r'(\d{4})', survey_id)
        if year_match:
            row['year'] = int(year_match.group(1))
        else:
            row['year'] = None

        # Extract month (if present)
        month_match = re.search(
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
            survey_id.lower()
        )
        if month_match:
            row['month'] = month_match.group(1)
        else:
            row['month'] = None

        # Extract date (if present in YYYYMMDD format)
        date_match = re.search(r'(\d{8})', survey_id)
        if date_match:
            date_str = date_match.group(1)
            try:
                row['date'] = pd.to_datetime(date_str, format='%Y%m%d')
            except:
                row['date'] = None
        else:
            row['date'] = None

        # Extract wave number (if explicitly present)
        wave_match = re.search(r'wave[_\s]?(\d+)', survey_id.lower())
        if wave_match:
            row['wave_number'] = int(wave_match.group(1))
        else:
            row['wave_number'] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by date/year/wave_number
    if 'date' in df.columns and df['date'].notna().any():
        df = df.sort_values('date')
    elif 'wave_number' in df.columns and df['wave_number'].notna().any():
        df = df.sort_values('wave_number')
    elif 'year' in df.columns and df['year'].notna().any():
        df = df.sort_values('year')
    else:
        # Sort alphabetically as fallback
        df = df.sort_values('survey_id')

    # Assign wave IDs
    df['wave_id'] = [f'wave_{i+1}' for i in range(len(df))]

    return df


def get_id_vars(df: pd.DataFrame,
                candidates: Optional[List[str]] = None,
                return_all: bool = False) -> Union[str, List[str], None]:
    """
    Automatically identify unique respondent or household ID variables.

    Parameters
    ----------
    df : DataFrame
        Input data.
    candidates : list of str, optional
        Potential ID column names to check first.
    return_all : bool, default False
        If True, returns all potential ID columns. If False, returns best match.

    Returns
    -------
    str, list of str, or None
        ID column name(s) or None if not found.

    Examples
    --------
    >>> df = pd.DataFrame({'pid': [1, 2, 3], 'age': [25, 30, 35], 'income': [50000, 60000, 70000]})
    >>> id_col = get_id_vars(df)
    >>> print(id_col)
    'pid'
    """
    if candidates is None:
        candidates = [
            'indid', 'pid', 'respondent_id', 'id', 'caseid', 'personid',
            'householdid', 'hhid', 'hh_id', 'household_id'
        ]

    # Standardize for case-insensitive matching
    candidates_lower = [c.lower() for c in candidates]
    df_cols = list(df.columns)
    df_cols_lower = [c.lower() for c in df_cols]

    potential_ids = []

    # Check explicit candidates
    for candidate in candidates_lower:
        if candidate in df_cols_lower:
            idx = df_cols_lower.index(candidate)
            actual_col = df_cols[idx]

            # Verify uniqueness
            if df[actual_col].nunique() == len(df):
                potential_ids.append(actual_col)

    # Check columns with 'id' in the name
    for col in df_cols:
        if 'id' in col.lower() and col not in potential_ids:
            if df[col].nunique() == len(df):
                potential_ids.append(col)

    # Check first column as fallback
    if not potential_ids and df.columns[0] not in potential_ids:
        first_col = df.columns[0]
        if df[first_col].nunique() == len(df):
            potential_ids.append(first_col)

    if not potential_ids:
        return None

    if return_all:
        return potential_ids

    # Return best match (prioritize candidates list)
    for candidate in candidates_lower:
        for pot_id in potential_ids:
            if candidate == pot_id.lower():
                return pot_id

    return potential_ids[0]


from pathlib import Path
