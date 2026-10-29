"""Fiberpose: distance-transform-based segmentation and morphometry of fibrous microstructures."""

__version__ = "0.1.0"

__all__ = ["AnalysisResult", "AppConfig", "FiberposeError", "analyze", "__version__"]


def __getattr__(name):
    # Imported lazily so `import fiberpose` and `fiberpose --help` stay fast
    # and don't load ONNX Runtime or compile numba kernels.
    if name in ("AnalysisResult", "FiberposeError", "analyze"):
        from . import api

        return getattr(api, name)
    if name == "AppConfig":
        from .domain import AppConfig

        return AppConfig
    raise AttributeError(f"module 'fiberpose' has no attribute {name!r}")
