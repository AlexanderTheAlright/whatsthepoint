"""
whatsthepoint Utils Module
=========================

Utility functions and styling tools.

Modules
-------
style
    Color palettes, plot themes, and style utilities.
optional
    Lazy imports for optional libraries, with install hints.
"""

from . import style
from . import optional

__all__ = [
    'style',
    'optional',
]
