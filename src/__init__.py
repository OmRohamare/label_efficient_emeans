"""label_efficient_emeans – top-level package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("label-efficient-emeans")
except PackageNotFoundError:
    __version__ = "0.1.0.dev"

__all__ = ["__version__"]
