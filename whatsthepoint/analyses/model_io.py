"""
model_io.py: whatsthepoint Model Save/Load Module
---------------------------------------------------
Save and load regression model results for reproducibility and cross-platform compatibility.

Features:
- Save model estimates, standard errors, and fit statistics
- JSON and CSV export formats
- R compatibility (export to CSV with R-style column names)
- Load saved results back into regression tables
- Preserve model metadata

Supported formats:
- JSON: Full model information including metadata
- CSV: Coefficients and statistics for Excel/R compatibility
- RDS (via conversion): Export to R-compatible format
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings



# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================

def save_model(
    model,
    filepath: str,
    format: str = "json",
    include_data: bool = False,
) -> None:
    """
    Save regression model results to file.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model from regress.py.
    filepath : str
        Output file path.
    format : str, default "json"
        Output format: "json", "csv", or "both".
    include_data : bool, default False
        Include original data in saved file (JSON only).

    Examples
    --------
    >>> from whatsthepoint.analyses.regress import ols
    >>> model = ols(df, 'y', ['x1', 'x2'])
    >>> save_model(model, 'my_model.json')
    """
    filepath = Path(filepath)

    # Extract model information
    model_dict = {
        'model_type': model.model_type,
        'outcome': model.outcome,
        'inputs': model.inputs,
        'weight': model.weight,
        'formula': getattr(model, 'formula', None),
        'coefficients': model.get_tidy().to_dict('records'),
        'statistics': model.get_stats(),
        'n_obs': int(model.results.nobs),
    }

    if include_data:
        model_dict['data'] = model.df.to_dict('records')

    # Save based on format
    if format in ["json", "both"]:
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(model_dict, f, indent=2, default=_json_serializer)
        print(f"Model saved to: {json_path}")

    if format in ["csv", "both"]:
        csv_path = filepath.with_suffix('.csv')
        coef_df = model.get_tidy()

        # Add statistics as additional rows
        stats = model.get_stats()
        stat_rows = []
        for key, val in stats.items():
            stat_rows.append({
                'term': f'STAT_{key}',
                'estimate': val,
                'std.error': np.nan,
                'statistic': np.nan,
                'p.value': np.nan,
                'conf.low': np.nan,
                'conf.high': np.nan,
            })

        full_df = pd.concat([coef_df, pd.DataFrame(stat_rows)], ignore_index=True)
        full_df.to_csv(csv_path, index=False)
        print(f"Model saved to: {csv_path}")


def save_models(
    models: Union[List, Dict],
    filepath: str,
    model_names: Optional[List[str]] = None,
    format: str = "json",
) -> None:
    """
    Save multiple regression models to file.

    Parameters
    ----------
    models : list or dict
        List of RegressionModel objects, or dict for grouped models.
    filepath : str
        Output file path.
    model_names : list of str, optional
        Names for each model.
    format : str, default "json"
        Output format: "json" or "csv".

    Examples
    --------
    >>> models = [model1, model2, model3]
    >>> save_models(models, 'comparison.json', model_names=['M1', 'M2', 'M3'])

    Grouped models:
    >>> grouped = {'Satisfaction': [m1, m2], 'Well-being': [m3, m4]}
    >>> save_models(grouped, 'grouped_models.json')
    """
    filepath = Path(filepath)

    if isinstance(models, dict):
        # Grouped models
        output = {}
        for group, group_models in models.items():
            output[group] = []
            for i, model in enumerate(group_models):
                model_data = {
                    'model_type': model.model_type,
                    'outcome': model.outcome,
                    'inputs': model.inputs,
                    'coefficients': model.get_tidy().to_dict('records'),
                    'statistics': model.get_stats(),
                }
                output[group].append(model_data)
    else:
        # List of models
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(models))]

        output = {}
        for name, model in zip(model_names, models):
            output[name] = {
                'model_type': model.model_type,
                'outcome': model.outcome,
                'inputs': model.inputs,
                'coefficients': model.get_tidy().to_dict('records'),
                'statistics': model.get_stats(),
            }

    if format == "json":
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(output, f, indent=2, default=_json_serializer)
        print(f"Models saved to: {filepath.with_suffix('.json')}")
    elif format == "csv":
        # For CSV, save each model as separate file
        for name, data in output.items():
            safe_name = name.replace(' ', '_').replace('/', '_')
            csv_path = filepath.parent / f"{filepath.stem}_{safe_name}.csv"
            if 'coefficients' in data:
                coef_df = pd.DataFrame(data['coefficients'])
                coef_df.to_csv(csv_path, index=False)
                print(f"Model '{name}' saved to: {csv_path}")


# ==============================================================================
# LOAD FUNCTIONS
# ==============================================================================

def load_model(filepath: str) -> Dict:
    """
    Load regression model results from file.

    Parameters
    ----------
    filepath : str
        Input file path (.json or .csv).

    Returns
    -------
    dict
        Model information including coefficients and statistics.

    Examples
    --------
    >>> model_data = load_model('my_model.json')
    >>> coef_df = pd.DataFrame(model_data['coefficients'])
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath)

        # Separate coefficients and statistics
        coef_df = df[~df['term'].str.startswith('STAT_', na=False)]
        stat_df = df[df['term'].str.startswith('STAT_', na=False)]

        stats = {}
        for _, row in stat_df.iterrows():
            key = row['term'].replace('STAT_', '')
            stats[key] = row['estimate']

        return {
            'coefficients': coef_df.to_dict('records'),
            'statistics': stats,
        }
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_models(filepath: str) -> Dict:
    """
    Load multiple regression models from file.

    Parameters
    ----------
    filepath : str
        Input file path (.json).

    Returns
    -------
    dict
        Dictionary of model data.

    Examples
    --------
    >>> models_data = load_models('comparison.json')
    >>> for name, data in models_data.items():
    ...     print(name, data['statistics'])
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("load_models only supports JSON format")


# ==============================================================================
# R COMPATIBILITY
# ==============================================================================

def export_for_r(
    model,
    filepath: str,
    include_vcov: bool = False,
) -> None:
    """
    Export model results in R-compatible format.

    Creates CSV files that can be easily loaded into R.

    Parameters
    ----------
    model : RegressionModel
        Fitted regression model.
    filepath : str
        Output file path (will create _coef.csv and optionally _vcov.csv).
    include_vcov : bool, default False
        Include variance-covariance matrix.

    Examples
    --------
    >>> export_for_r(model, 'model_for_r')
    # Creates: model_for_r_coef.csv (and model_for_r_vcov.csv if include_vcov=True)
    """
    filepath = Path(filepath)

    # Save coefficients with R-style names
    coef_df = model.get_tidy().rename(columns={
        'term': 'term',
        'estimate': 'estimate',
        'std.error': 'std_error',
        'statistic': 't_value',
        'p.value': 'p_value',
        'conf.low': 'conf_low',
        'conf.high': 'conf_high',
    })

    coef_path = filepath.parent / f"{filepath.stem}_coef.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Coefficients exported for R: {coef_path}")

    # Save statistics
    stats_df = pd.DataFrame([model.get_stats()])
    stats_path = filepath.parent / f"{filepath.stem}_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics exported for R: {stats_path}")

    # Variance-covariance matrix (if requested)
    if include_vcov:
        try:
            vcov = pd.DataFrame(
                model.results.cov_params(),
                index=model.results.params.index,
                columns=model.results.params.index
            )
            vcov_path = filepath.parent / f"{filepath.stem}_vcov.csv"
            vcov.to_csv(vcov_path)
            print(f"Variance-covariance matrix exported: {vcov_path}")
        except Exception as e:
            warnings.warn(f"Could not export vcov matrix: {e}")


def import_from_r(coef_filepath: str, stats_filepath: Optional[str] = None) -> Dict:
    """
    Import model results from R-style CSV files.

    Parameters
    ----------
    coef_filepath : str
        Path to coefficients CSV file.
    stats_filepath : str, optional
        Path to statistics CSV file.

    Returns
    -------
    dict
        Model data compatible with regression_table.

    Examples
    --------
    >>> model_data = import_from_r('r_model_coef.csv', 'r_model_stats.csv')
    >>> html = regression_table(pd.DataFrame(model_data['coefficients']))
    """
    coef_df = pd.read_csv(coef_filepath)

    # Convert R-style names to Python style
    name_map = {
        'std_error': 'std.error',
        't_value': 'statistic',
        'p_value': 'p.value',
        'conf_low': 'conf.low',
        'conf_high': 'conf.high',
    }
    coef_df = coef_df.rename(columns=name_map)

    result = {'coefficients': coef_df.to_dict('records')}

    if stats_filepath:
        stats_df = pd.read_csv(stats_filepath)
        result['statistics'] = stats_df.iloc[0].to_dict()

    return result


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _json_serializer(obj):
    """Handle non-serializable objects for JSON."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return str(obj)


def models_to_dataframes(models_data: Union[Dict, List[Dict]]) -> Union[List[pd.DataFrame], Dict[str, List[pd.DataFrame]]]:
    """
    Convert loaded model data to DataFrames for use with regression_table.

    Parameters
    ----------
    models_data : dict or list of dict
        Loaded model data from load_model or load_models.

    Returns
    -------
    list of DataFrame or dict of str: list of DataFrame
        Ready for regression_table function.

    Examples
    --------
    >>> models_data = load_models('comparison.json')
    >>> model_dfs = models_to_dataframes(models_data)
    >>> html = regression_table(model_dfs)
    """
    if isinstance(models_data, list):
        return [pd.DataFrame(m['coefficients']) for m in models_data]
    elif isinstance(models_data, dict):
        # Check if grouped structure
        first_val = next(iter(models_data.values()))
        if isinstance(first_val, list):
            # Grouped models
            return {
                group: [pd.DataFrame(m['coefficients']) for m in group_models]
                for group, group_models in models_data.items()
            }
        else:
            # Simple dict of models
            return [pd.DataFrame(m['coefficients']) for m in models_data.values()]
    else:
        raise ValueError("Invalid models_data structure")
