"""Filesystem locations used by Fiberpose.

Bundled resources live inside the package and are read-only once installed.
Anything Fiberpose writes (the user config) goes to the per-user config folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from platformdirs import user_config_dir

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = PACKAGE_DIR / "resources"

DEFAULT_UI_PATH = RESOURCES_DIR / "fiber_tool.ui"
DARK_THEME_PATH = RESOURCES_DIR / "dark_theme.qss"
APP_ICON_PATH = RESOURCES_DIR / "fiberpose.png"
EXAMPLE_IMAGES_DIR = RESOURCES_DIR / "images"
EXAMPLE_IMAGE_PATH = EXAMPLE_IMAGES_DIR / "test.png"
# Examples offered on the web page: the default one first, then the rest of the folder.
EXAMPLE_IMAGE_PATHS = [EXAMPLE_IMAGE_PATH] + sorted(
    path
    for path in EXAMPLE_IMAGES_DIR.iterdir()
    if path.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"} and path != EXAMPLE_IMAGE_PATH
)
BUNDLED_CONFIG_PATH = RESOURCES_DIR / "config.json"

USER_CONFIG_PATH = Path(user_config_dir("fiberpose", appauthor=False)) / "config.json"


def default_config_path() -> str:
    """Return the user config path, creating it from the bundled default on first use."""
    if not USER_CONFIG_PATH.exists():
        USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BUNDLED_CONFIG_PATH, USER_CONFIG_PATH)
    return str(USER_CONFIG_PATH)
