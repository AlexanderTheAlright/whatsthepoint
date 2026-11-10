"""
whatsthepoint: A Python Package for Sociological Data Analysis
=============================================================

whatsthepoint handles survey data, runs common analyses and draws figures
for social science research.

Modules:
    - analyses: Statistical modeling and regression analysis
    - cleansing: Data harmonization and cleaning
    - visuals: Data visualization functions
    - utils: Utility functions and styling
    - data: Data management
"""

__version__ = "0.3.0"

# Import submodules
from . import analyses
from . import cleansing
from . import visuals
from . import utils
from . import data

__all__ = [
    'analyses',
    'cleansing',
    'visuals',
    'utils',
    'data',
]
