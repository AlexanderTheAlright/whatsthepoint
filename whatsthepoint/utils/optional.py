"""
optional.py: import optional libraries only when a function needs them.

`import whatsthepoint` works with the core dependencies alone. A function that
needs an optional library calls `require()`, which raises an ImportError that
names the pip extra to install.
"""

import importlib


def require(module: str, extra: str):
    """Import `module` or raise an ImportError naming the whatsthepoint extra."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"'{module}' is needed for this function. "
            f"Install it with: pip install whatsthepoint[{extra}]"
        ) from err


def available(module: str) -> bool:
    """True if `module` can be imported."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False
