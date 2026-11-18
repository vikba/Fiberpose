"""Gradio web interface: ``fiberpose --web``.

``build_demo(config)`` only describes the page; the caller decides how to serve it.
Analysis settings come from the same config file as the GUI and CLI.
Host, port and proxy path come from Gradio's own environment variables
(``GRADIO_SERVER_NAME``, ``GRADIO_SERVER_PORT``, ``GRADIO_ROOT_PATH``);
``FIBERPOSE_MAX_UPLOAD`` caps the upload size (default ``50mb``) and
``FIBERPOSE_MAX_SIDE`` the image width and height (default ``1920`` px);
``FIBERPOSE_USAGE_REPO`` turns on the usage log (see ``usage.py``).
"""

from .demo import build_demo, launch

__all__ = ["build_demo", "launch"]
