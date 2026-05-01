"""Internal package metadata for Fiberpose source modules."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "__version__",
    "AppConfig",
    "SampleProperties",
    "ImageBatch",
    "BatchResult",
    "Frame",
    "FiberFragment",
    "OnnxFiberposeModel",
    "onnx_cuda_available",
    "PipelineController",
    "PipelineOutcome",
    "BatchService",
    "BatchCallbacks",
    "ExportService",
    "BatchResultExporter",
    "MainWindow",
]

__version__ = "0.1.0"

_EXPORTS = {
    "AppConfig": ("sources.domain", "AppConfig"),
    "SampleProperties": ("sources.domain", "SampleProperties"),
    "ImageBatch": ("sources.domain", "ImageBatch"),
    "BatchResult": ("sources.domain", "BatchResult"),
    "Frame": ("sources.domain", "Frame"),
    "FiberFragment": ("sources.domain", "FiberFragment"),
    "OnnxFiberposeModel": ("sources.ml", "OnnxFiberposeModel"),
    "onnx_cuda_available": ("sources.ml", "onnx_cuda_available"),
    "PipelineController": ("sources.pipeline", "PipelineController"),
    "PipelineOutcome": ("sources.pipeline", "PipelineOutcome"),
    "BatchService": ("sources.pipeline", "BatchService"),
    "BatchCallbacks": ("sources.pipeline", "BatchCallbacks"),
    "ExportService": ("sources.io_export", "ExportService"),
    "BatchResultExporter": ("sources.io_export", "BatchResultExporter"),
    "MainWindow": ("sources.ui_main", "MainWindow"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'sources' has no attribute {name!r}")
    module_name, symbol = target
    module = import_module(module_name)
    value = getattr(module, symbol)
    globals()[name] = value
    return value
