"""RadiomicsViz: a stage-wise, manifest-driven MRI radiomics pipeline.

The distribution is named ``radiomicsviz``; the import package and the command-line
program are named ``radiomicviz``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("radiomicsviz")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
