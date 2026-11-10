"""
ml.py: whatsthepoint Machine Learning Module
-------------------------------------------
Machine learning pipelines for prediction and feature analysis.

Features:
- Automated feature preprocessing pipelines
- Model training and cross-validation
- Feature importance analysis
- SHAP values for interpretability
- Model comparison and selection

Functions:
- build_pipeline: Create preprocessing + model pipeline
- train_model: Train ML model with cross-validation
- feature_importance: Extract and visualize feature importance
- shap_analysis: SHAP values for model interpretability
- predict_proba: Probability predictions with confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union, Dict, Tuple, Any
import warnings

# sklearn imports
try:
    from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Install with: pip install scikit-learn")

# Optional: SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # Don't warn - SHAP is optional



# ==============================================================================
# PIPELINE BUILDING
# ==============================================================================

def build_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    model_type: str = "random_forest",
    task: str = "classification",
    **model_params
) -> Pipeline:
    """
    Build preprocessing + model pipeline.

    Parameters
    ----------
    numeric_features : list of str, optional
        Numeric feature names.
    categorical_features : list of str, optional
        Categorical feature names.
    model_type : str, default "random_forest"
        Model: "random_forest", "gradient_boosting", "logistic", "ridge", "lasso".
    task : str, default "classification"
        Task type: "classification" or "regression".
    **model_params
        Additional parameters for the model.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline.

    Examples
    --------
    >>> pipeline = build_pipeline(
    ...     numeric_features=['age', 'income'],
    ...     categorical_features=['education', 'region'],
    ...     model_type='random_forest',
    ...     task='classification',
    ...     n_estimators=100
    ... )
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required. Install with: pip install scikit-learn")

    # Build preprocessor
    transformers = []

    if numeric_features:
        numeric_transformer = StandardScaler()
        transformers.append(('num', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(('cat', categorical_transformer, categorical_features))

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    else:
        preprocessor = None

    # Select model
    if task == "classification":
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_params)
        elif model_type == "logistic":
            model = LogisticRegression(**model_params)
        else:
            raise ValueError(f"Unsupported classification model: {model_type}")

    elif task == "regression":
        if model_type == "random_forest":
            model = RandomForestRegressor(**model_params)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(**model_params)
        elif model_type == "ridge":
            model = Ridge(**model_params)
        elif model_type == "lasso":
            model = Lasso(**model_params)
        else:
            raise ValueError(f"Unsupported regression model: {model_type}")
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Build pipeline
    if preprocessor:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('model', model)
        ])

    return pipeline


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

class MLModel:
    """
    Machine learning model with training, evaluation, and interpretation.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome (target) variable.
    features : list of str
        Feature variables.
    task : str, default "classification"
        Task: "classification" or "regression".
    test_size : float, default 0.2
        Proportion of data for test set.
    random_state : int, default 42
        Random seed.

    Attributes
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline.
    train_scores : dict
        Training set performance metrics.
    test_scores : dict
        Test set performance metrics.
    feature_importance : DataFrame
        Feature importance scores.

    Examples
    --------
    >>> ml = MLModel(df, outcome='employed', features=['age', 'education', 'income'])
    >>> ml.train(model_type='random_forest', n_estimators=100)
    >>> print(ml.test_scores)
    >>> ml.plot_feature_importance()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str,
        features: List[str],
        task: str = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required. Install with: pip install scikit-learn")

        self.df = df.copy()
        self.outcome = outcome
        self.features = features
        self.task = task
        self.test_size = test_size
        self.random_state = random_state

        self.pipeline = None
        self.train_scores = {}
        self.test_scores = {}
        self.feature_importance = None

        # Identify numeric and categorical features
        self.numeric_features = []
        self.categorical_features = []

        for feat in features:
            if pd.api.types.is_numeric_dtype(df[feat]):
                self.numeric_features.append(feat)
            else:
                self.categorical_features.append(feat)

        # Train-test split
        data = df[[outcome] + features].dropna()
        self.X = data[features]
        self.y = data[outcome]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train(
        self,
        model_type: str = "random_forest",
        cv: int = 5,
        **model_params
    ):
        """
        Train model with cross-validation.

        Parameters
        ----------
        model_type : str, default "random_forest"
            Model type.
        cv : int, default 5
            Number of cross-validation folds.
        **model_params
            Model hyperparameters.
        """
        # Build pipeline
        self.pipeline = build_pipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            model_type=model_type,
            task=self.task,
            **model_params
        )

        # Train
        self.pipeline.fit(self.X_train, self.y_train)

        # Cross-validation
        if self.task == "classification":
            cv_scores = cross_val_score(self.pipeline, self.X_train, self.y_train,
                                       cv=cv, scoring='accuracy')
        else:
            cv_scores = cross_val_score(self.pipeline, self.X_train, self.y_train,
                                       cv=cv, scoring='r2')

        # Evaluate
        self._evaluate()

        self.cv_scores = cv_scores
        return self

    def _evaluate(self):
        """Calculate performance metrics."""
        y_train_pred = self.pipeline.predict(self.X_train)
        y_test_pred = self.pipeline.predict(self.X_test)

        if self.task == "classification":
            self.train_scores = {
                'accuracy': accuracy_score(self.y_train, y_train_pred),
                'precision': precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0),
                'f1': f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0),
            }

            self.test_scores = {
                'accuracy': accuracy_score(self.y_test, y_test_pred),
                'precision': precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0),
                'f1': f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0),
            }

            # AUC if binary
            if len(np.unique(self.y)) == 2:
                try:
                    y_train_proba = self.pipeline.predict_proba(self.X_train)[:, 1]
                    y_test_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
                    self.train_scores['auc'] = roc_auc_score(self.y_train, y_train_proba)
                    self.test_scores['auc'] = roc_auc_score(self.y_test, y_test_proba)
                except:
                    pass

        else:  # regression
            self.train_scores = {
                'r2': r2_score(self.y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                'mae': mean_absolute_error(self.y_train, y_train_pred),
            }

            self.test_scores = {
                'r2': r2_score(self.y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                'mae': mean_absolute_error(self.y_test, y_test_pred),
            }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importance.

        Returns
        -------
        DataFrame
            Feature importance scores.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call .train() first.")

        model = self.pipeline.named_steps['model']

        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Get feature names after preprocessing
            if 'preprocessor' in self.pipeline.named_steps:
                preprocessor = self.pipeline.named_steps['preprocessor']
                feature_names = []

                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        feature_names.extend(features)
                    elif name == 'cat':
                        # Get one-hot encoded names
                        if hasattr(transformer, 'get_feature_names_out'):
                            cat_names = transformer.get_feature_names_out(features)
                            feature_names.extend(cat_names)
                        else:
                            feature_names.extend(features)
            else:
                feature_names = self.features

            # Handle length mismatch
            if len(feature_names) != len(importances):
                feature_names = [f'feature_{i}' for i in range(len(importances))]

            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

        # Coefficient-based importance (linear models)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # Take first class for multiclass

            if 'preprocessor' in self.pipeline.named_steps:
                preprocessor = self.pipeline.named_steps['preprocessor']
                feature_names = []
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'num':
                        feature_names.extend(features)
                    elif name == 'cat':
                        if hasattr(transformer, 'get_feature_names_out'):
                            cat_names = transformer.get_feature_names_out(features)
                            feature_names.extend(cat_names)
            else:
                feature_names = self.features

            if len(feature_names) != len(coef):
                feature_names = [f'feature_{i}' for i in range(len(coef))]

            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)

        else:
            warnings.warn("Model does not support feature importance extraction")
            self.feature_importance = pd.DataFrame()

        return self.feature_importance

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for new data.

        Parameters
        ----------
        X_new : DataFrame
            New data with same features.

        Returns
        -------
        array
            Predictions.
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call .train() first.")

        return self.pipeline.predict(X_new)

    def summary(self) -> str:
        """Print model summary."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call .train() first.")

        output = "ML Model Summary\n"
        output += "=" * 60 + "\n\n"

        output += f"Task: {self.task}\n"
        output += f"Outcome: {self.outcome}\n"
        output += f"Features: {len(self.features)}\n"
        output += f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}\n\n"

        output += "Training Performance:\n"
        for metric, value in self.train_scores.items():
            output += f"  {metric}: {value:.4f}\n"

        output += "\nTest Performance:\n"
        for metric, value in self.test_scores.items():
            output += f"  {metric}: {value:.4f}\n"

        if self.cv_scores is not None:
            output += f"\nCross-validation: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})\n"

        return output


# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================

def feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Parameters
    ----------
    model : trained model or Pipeline
        Trained sklearn model or pipeline.
    feature_names : list of str, optional
        Feature names. Inferred if not provided.
    top_n : int, default 20
        Number of top features to return.

    Returns
    -------
    DataFrame
        Feature importance scores.

    Examples
    --------
    >>> importance = feature_importance(fitted_model, feature_names=['age', 'income'])
    """
    # Accept a fitted MLModel (from train_model) as well as a Pipeline
    if getattr(model, 'pipeline', None) is not None:
        model = model.pipeline
    # Extract model from pipeline if needed
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps['model']
    else:
        actual_model = model

    # Get importances
    if hasattr(actual_model, 'feature_importances_'):
        importances = actual_model.feature_importances_
    elif hasattr(actual_model, 'coef_'):
        coef = actual_model.coef_
        if len(coef.shape) > 1:
            coef = coef[0]
        importances = np.abs(coef)
    else:
        raise ValueError("Model does not support feature importance")

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    return df_importance


# ==============================================================================
# SHAP ANALYSIS (OPTIONAL)
# ==============================================================================

def shap_analysis(
    model: Any,
    X: pd.DataFrame,
    background_size: int = 100,
) -> Any:
    """
    Calculate SHAP values for model interpretability.

    Parameters
    ----------
    model : trained model
        Trained sklearn model or pipeline.
    X : DataFrame
        Input data for SHAP calculation.
    background_size : int, default 100
        Background dataset size for TreeExplainer.

    Returns
    -------
    shap.Explanation
        SHAP values object.

    Examples
    --------
    >>> shap_values = shap_analysis(fitted_model, X_test)
    >>> shap.summary_plot(shap_values)

    Note
    ----
    Requires shap package: pip install shap
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is required. Install it with: pip install whatsthepoint[ml]")

    # Unwrap an MLModel (as returned by train_model) to its sklearn pipeline
    if hasattr(model, 'pipeline'):
        model = model.pipeline

    # Extract model from pipeline if needed
    if hasattr(model, 'named_steps'):
        # Need to transform data first
        preprocessor = model.named_steps.get('preprocessor')
        if preprocessor:
            X_transformed = preprocessor.transform(X)
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
        else:
            X_transformed = X.values

        actual_model = model.named_steps['model']
    else:
        actual_model = model
        X_transformed = X.values

    # Select appropriate explainer
    if hasattr(actual_model, 'tree_') or hasattr(actual_model, 'estimators_'):
        # Tree-based model
        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Use KernelExplainer for other models
        background = X_transformed[:background_size]
        explainer = shap.KernelExplainer(actual_model.predict, background)
        shap_values = explainer.shap_values(X_transformed)

    return shap_values


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def train_model(
    df: pd.DataFrame,
    outcome: str,
    features: List[str],
    model_type: str = "random_forest",
    task: str = "classification",
    test_size: float = 0.2,
    **model_params
) -> MLModel:
    """
    Convenience wrapper to train ML model.

    Parameters
    ----------
    df : DataFrame
        Input data.
    outcome : str
        Outcome variable.
    features : list of str
        Feature variables.
    model_type : str, default "random_forest"
        Model type.
    task : str, default "classification"
        Task type.
    test_size : float, default 0.2
        Test set proportion.
    **model_params
        Model hyperparameters.

    Returns
    -------
    MLModel
        Trained model object.

    Examples
    --------
    >>> model = train_model(df, 'employed', ['age', 'education'],
    ...                     model_type='random_forest', n_estimators=100)
    >>> print(model.summary())
    """
    ml = MLModel(df, outcome, features, task=task, test_size=test_size)
    ml.train(model_type=model_type, **model_params)
    return ml
