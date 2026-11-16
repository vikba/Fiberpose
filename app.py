"""Entry point of the Hugging Face Gradio Space (configured in deploy/hf_header.md).

The Space installs requirements.txt and runs this file from the repository root,
so the package is imported from src/ instead of being installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import imageio.v3 as iio  # noqa: E402

import fiberpose  # noqa: E402
from fiberpose.domain import AppConfig  # noqa: E402
from fiberpose.paths import EXAMPLE_IMAGE_PATH, default_config_path  # noqa: E402
from fiberpose.web import launch  # noqa: E402

# ZeroGPU hardware refuses to start an app without a @spaces.GPU function. Fiberpose runs on
# the CPU, so this one is never called and uses no GPU quota; elsewhere it does nothing.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder() -> None:
        pass
except ImportError:  # not running on a Space
    pass

# Analyse the example once so the first visitor does not wait for numba to compile.
fiberpose.analyze(iio.imread(EXAMPLE_IMAGE_PATH))

launch(AppConfig.load_from_json(default_config_path()))
