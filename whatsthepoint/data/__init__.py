"""
whatsthepoint Data Module
========================

Data management and loading utilities for survey/panel research.

Modules
-------
discovery
    File discovery and path handling utilities.
loading
    Data loading with automatic preprocessing.
metadata
    Survey metadata extraction and summarization.
longitudinal
    Panel/longitudinal data detection and alignment.
preparation
    Harmonization and pre-analysis preparation.
"""

from . import discovery
from . import loading
from . import metadata
from . import longitudinal
from . import preparation

# Import key functions for convenience
from .discovery import (
    resolve_data_root,
    discover_data,
    find_latest_wave,
)

from .loading import (
    load_stata,
    load_all_surveys,
    normalize_ids,
    load_and_combine,
)

from .metadata import (
    summarize_surveys,
    extract_wave_info,
    get_id_vars,
)

from .longitudinal import (
    detect_longitudinal,
    align_longitudinal_data,
    sort_by_wave,
    build_longit_from_dir,
)

from .preparation import (
    harmonize_columns,
    build_harmonized_dataset,
    to_categorical_ordered,
    numeric_codes,
    prepare_for_analysis,
)

__all__ = [
    # Modules
    'discovery',
    'loading',
    'metadata',
    'longitudinal',
    'preparation',
    # Discovery functions
    'resolve_data_root',
    'discover_data',
    'find_latest_wave',
    # Loading functions
    'load_stata',
    'load_all_surveys',
    'normalize_ids',
    'load_and_combine',
    # Metadata functions
    'summarize_surveys',
    'extract_wave_info',
    'get_id_vars',
    # Longitudinal functions
    'detect_longitudinal',
    'align_longitudinal_data',
    'sort_by_wave',
    'build_longit_from_dir',
    # Preparation functions
    'harmonize_columns',
    'build_harmonized_dataset',
    'to_categorical_ordered',
    'numeric_codes',
    'prepare_for_analysis',
]
