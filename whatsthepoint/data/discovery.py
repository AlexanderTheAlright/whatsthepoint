"""
Data Discovery & Path Handling Utilities
=========================================

File discovery, path resolution, and wave identification for longitudinal survey data.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import warnings

__all__ = [
    'resolve_data_root',
    'discover_data',
    'find_latest_wave',
]


def resolve_data_root(start_path: Optional[Union[str, Path]] = None,
                      root_patterns: Optional[List[str]] = None) -> Path:
    """
    Locate the data root directory by searching up the directory tree.

    Parameters
    ----------
    start_path : str or Path, optional
        Starting directory for search. Defaults to current working directory.
    root_patterns : list of str, optional
        Directory names to match (case-insensitive, partial match). When
        omitted, the name in the environment variable ``WHATSTHEPOINT_DATA``
        is used; when that is unset, ``['data']``.

    Returns
    -------
    Path
        The resolved root directory.

    Raises
    ------
    FileNotFoundError
        If no matching root directory is found.

    Examples
    --------
    >>> root = resolve_data_root(root_patterns=["survey_files"])
    >>> print(root)
    /path/to/survey_files
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    if root_patterns is None:
        named = os.environ.get("WHATSTHEPOINT_DATA", "").strip()
        root_patterns = [named] if named else ['data']

    # Search up the directory tree
    current = start_path.resolve()

    # Check if current directory matches
    for pattern in root_patterns:
        if pattern.lower() in current.name.lower():
            return current

    # Search parent directories
    for parent in current.parents:
        for pattern in root_patterns:
            if pattern.lower() in parent.name.lower():
                return parent

            # Also check for subdirectories matching pattern
            for child in parent.iterdir():
                if child.is_dir() and pattern.lower() in child.name.lower():
                    return child

    raise FileNotFoundError(
        f"Could not locate data root directory. "
        f"Searched from {start_path} for patterns: {root_patterns}"
    )


def discover_data(data_dir: Union[str, Path],
                  file_types: Optional[List[str]] = None,
                  include_patterns: Optional[List[str]] = None,
                  exclude_patterns: Optional[List[str]] = None,
                  recursive: bool = True,
                  auto_detect_types: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Scan directory for data files and load them dynamically.

    Parameters
    ----------
    data_dir : str or Path
        Directory to search for data files.
    file_types : list of str, optional
        File extensions to include (e.g., ['dta', 'csv', 'xlsx']).
        Defaults to all supported types if auto_detect_types=True.
    include_patterns : list of str, optional
        Only include files matching these patterns (e.g., ['survey_*', '*_2020']).
    exclude_patterns : list of str, optional
        Exclude files matching these patterns (e.g., ['*_test*', 'backup_*']).
    recursive : bool, default True
        Whether to search subdirectories.
    auto_detect_types : bool, default True
        Whether to auto-detect and load all supported file types.

    Returns
    -------
    dict
        Dictionary mapping {survey_id: DataFrame} with lowercase column names.

    Examples
    --------
    >>> # Load only CSV and Stata files
    >>> data = discover_data('path/to/data', file_types=['csv', 'dta'])

    >>> # Load only files matching pattern
    >>> data = discover_data('path/to/data', include_patterns=['survey_*'])

    >>> # Load all except test files
    >>> data = discover_data('path/to/data', exclude_patterns=['*_test*', 'backup_*'])

    >>> # Auto-detect all supported file types
    >>> data = discover_data('path/to/data')  # Loads dta, csv, xlsx, xls, sav, parquet
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # Define supported file types
    supported_types = {
        '.dta': 'stata',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.sav': 'spss',
        '.parquet': 'parquet',
        '.feather': 'feather',
    }

    # Determine which file types to load
    if auto_detect_types and file_types is None:
        file_types = list(supported_types.keys())
    elif file_types is not None:
        # Normalize extensions (add dot if missing)
        file_types = ['.' + ft.lstrip('.') for ft in file_types]
    else:
        file_types = ['.dta', '.csv']  # Default

    if include_patterns is None:
        include_patterns = ['*']
    if exclude_patterns is None:
        exclude_patterns = []

    # Compile patterns
    include_regexes = [re.compile(p.replace('*', '.*').replace('?', '.')) for p in include_patterns]
    exclude_regexes = [re.compile(p.replace('*', '.*').replace('?', '.')) for p in exclude_patterns]

    loaded_data = {}
    skipped_files = []
    type_counts = {}

    # Find all matching files
    search_method = data_dir.rglob if recursive else data_dir.glob

    for file_ext in file_types:
        if file_ext not in supported_types:
            warnings.warn(f"Unsupported file type: {file_ext}")
            continue

        for filepath in search_method(f"*{file_ext}"):
            fname = filepath.name

            # Check include patterns
            if include_patterns != ['*']:
                matches_include = any(regex.search(fname) for regex in include_regexes)
                if not matches_include:
                    continue

            # Check exclude patterns
            matches_exclude = any(regex.search(fname) for regex in exclude_regexes)
            if matches_exclude:
                skipped_files.append((fname, "matched exclusion pattern"))
                continue

            # Load the file
            survey_id = filepath.stem

            try:
                file_type = supported_types[file_ext]

                if file_type == 'stata':
                    df = pd.read_stata(filepath, convert_categoricals=False)
                elif file_type == 'csv':
                    df = pd.read_csv(filepath)
                elif file_type == 'excel':
                    df = pd.read_excel(filepath)
                elif file_type == 'spss':
                    df = pd.read_spss(filepath)
                elif file_type == 'parquet':
                    df = pd.read_parquet(filepath)
                elif file_type == 'feather':
                    df = pd.read_feather(filepath)
                else:
                    skipped_files.append((fname, f"unsupported type: {file_ext}"))
                    continue

                # Standardize column names: lowercase, no spaces
                df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

                # Store metadata
                df.attrs['source_file'] = str(filepath)
                df.attrs['file_type'] = file_type

                loaded_data[survey_id] = df

                # Track types
                type_counts[file_type] = type_counts.get(file_type, 0) + 1

            except Exception as e:
                skipped_files.append((fname, str(e)))
                warnings.warn(f"Failed to load {fname}: {e}")

    # Log summary
    print(f"Loaded {len(loaded_data)} datasets")
    if type_counts:
        print(f"  File types: {', '.join(f'{k}={v}' for k, v in type_counts.items())}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files")
        if len(skipped_files) <= 5:
            for fname, reason in skipped_files:
                print(f"  - {fname}: {reason}")
        else:
            for fname, reason in skipped_files[:3]:
                print(f"  - {fname}: {reason}")
            print(f"  ... and {len(skipped_files) - 3} more")

    return loaded_data


def find_latest_wave(data_dict: Dict[str, pd.DataFrame],
                     name_pattern: Optional[str] = None,
                     date_col: Optional[str] = None) -> Dict[str, any]:
    """
    Identify latest or earliest wave from a collection of datasets.

    Parameters
    ----------
    data_dict : dict
        Dictionary of {survey_id: DataFrame} from discover_data().
    name_pattern : str, optional
        Regex pattern to extract date from filename (e.g., r'(\\d{4})').
    date_col : str, optional
        Column name containing date information in the data.

    Returns
    -------
    dict
        Contains:
        - 'latest': (survey_id, DataFrame) for most recent wave
        - 'earliest': (survey_id, DataFrame) for oldest wave
        - 'sorted_waves': List of (survey_id, year/date) sorted by time
        - 'wave_labels': Dict mapping survey_id to wave_1, wave_2, etc.

    Examples
    --------
    >>> data = discover_data('surveys/')
    >>> waves = find_latest_wave(data, name_pattern=r'_(\\d{4})')
    >>> print(waves['latest'][0])
    'survey_2023'
    >>> print(waves['wave_labels'])
    {'survey_2020': 'wave_1', 'survey_2021': 'wave_2', 'survey_2023': 'wave_3'}
    """
    if not data_dict:
        raise ValueError("data_dict is empty")

    wave_info = []

    for survey_id, df in data_dict.items():
        date_value = None

        # Try extracting from filename
        if name_pattern:
            match = re.search(name_pattern, survey_id)
            if match:
                try:
                    date_value = int(match.group(1))
                except (ValueError, IndexError):
                    pass

        # Try extracting from data column
        if date_value is None and date_col and date_col in df.columns:
            try:
                date_series = pd.to_datetime(df[date_col], errors='coerce')
                if not date_series.isna().all():
                    date_value = date_series.min()
            except:
                pass

        # Fallback: use survey_id alphabetically
        if date_value is None:
            date_value = survey_id

        wave_info.append((survey_id, date_value, df))

    # Sort by date
    sorted_waves = sorted(wave_info, key=lambda x: x[1])

    # Create wave labels
    wave_labels = {
        survey_id: f"wave_{i+1}"
        for i, (survey_id, _, _) in enumerate(sorted_waves)
    }

    return {
        'latest': (sorted_waves[-1][0], sorted_waves[-1][2]),
        'earliest': (sorted_waves[0][0], sorted_waves[0][2]),
        'sorted_waves': [(sid, date) for sid, date, _ in sorted_waves],
        'wave_labels': wave_labels,
    }
