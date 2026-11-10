"""
whatsthepoint Analyses Module
============================

Statistical modeling and regression analysis tools.

Modules:
- model_io: Model input/output utilities
- pubtable: Formatted HTML tables (proportions, descriptives, regressions)
- regress: Regression models (OLS, logit, poisson, multilevel)
- sem: Structural equation modeling and path analysis
- descriptive: Descriptive statistics and exploratory analysis
- causal: Causal inference methods (propensity scores, DiD, IV, RDD)
- panel: Panel data models (fixed effects, random effects)
- ml: Machine learning pipelines and feature importance
- text_analysis: Text analysis, topic modeling, similarity, complexity scores
- network: Network data preparation and creation
"""

from . import model_io
from . import pubtable
from . import regress
from . import sem
from . import descriptive
from . import causal
from . import panel
from . import ml
from . import text_analysis
from . import network

__all__ = [
    'model_io',
    'pubtable',
    'regress',
    'sem',
    'descriptive',
    'causal',
    'panel',
    'ml',
    'text_analysis',
    'network',
]
