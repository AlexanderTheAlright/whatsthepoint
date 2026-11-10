"""
Import tests: every submodule imports, and the package imports with only
the core dependencies (optional libraries blocked).
"""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import whatsthepoint

SUBMODULES = sorted(
    m.name for m in pkgutil.walk_packages(whatsthepoint.__path__, prefix="whatsthepoint.")
)

OPTIONAL = [
    "adjustText", "wordcloud", "networkx", "openpyxl", "linearmodels", "shap",
    "nltk", "transformers", "torch", "pingouin", "bertopic", "umap", "requests",
]


def test_version():
    assert whatsthepoint.__version__ == "0.3.0"


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name):
    importlib.import_module(name)


def test_imports_without_optional_libraries():
    code = f"""
import sys
BLOCKED = set({OPTIONAL!r})

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked: {{name}}")
        return None

for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        sys.modules.pop(name)
sys.meta_path.insert(0, Block())
import whatsthepoint, whatsthepoint.visuals, whatsthepoint.analyses, whatsthepoint.data, whatsthepoint.cleansing
import pandas as pd
from whatsthepoint.visuals import cooccur
try:
    cooccur.cooccur(pd.DataFrame({{"t": [["a", "b"]]}}), "t")
except ImportError as err:
    assert "whatsthepoint[network]" in str(err), err
else:
    raise AssertionError("expected ImportError")
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
