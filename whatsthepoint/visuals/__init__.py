"""
whatsthepoint Visuals Module
===========================

This module provides visualization functions for sociological data analysis.
"""

# Import all visualization modules
from . import apc
from . import bar
from . import boxplot
from . import cluster
from . import coef
from . import cooccur
from . import dag
from . import density
from . import factormap
from . import feature
from . import flow
from . import geographic
from . import heatmap
from . import hierarchical
from . import hist
from . import horizon
from . import ice
from . import margins
from . import oddsratio
from . import pair
from . import pie
from . import residuals
from . import scatter
from . import trend
from . import waterfall
from . import wordcloud

# Import ryder functions directly for convenience
from .density import ryder, ryder_interactive

__all__ = [
    'apc',
    'bar',
    'boxplot',
    'cluster',
    'coef',
    'cooccur',
    'dag',
    'density',
    'factormap',
    'feature',
    'flow',
    'geographic',
    'heatmap',
    'hierarchical',
    'hist',
    'horizon',
    'ice',
    'margins',
    'oddsratio',
    'pair',
    'pie',
    'residuals',
    'ryder',
    'ryder_interactive',
    'scatter',
    'trend',
    'waterfall',
    'wordcloud',
]
