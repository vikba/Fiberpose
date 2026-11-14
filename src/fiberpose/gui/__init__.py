"""PyQt desktop interface: ``fiberpose --gui``.

Importing this package needs the ``gui`` extra (PyQt6, pyqtgraph).
"""

from __future__ import annotations

import sys

from PyQt6.QtGui import QColor, QIcon, QPalette
from PyQt6.QtWidgets import QApplication

from ..paths import APP_ICON_PATH

__all__ = ["launch"]


def launch(config_path: str) -> int:
    """Open the main window and run the Qt event loop; returns its exit code."""
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    app.setStyle("Fusion")

    dark, text, white = QColor(45, 45, 45), QColor(220, 220, 220), QColor(255, 255, 255)
    palette = QPalette()
    for role, color in {
        "Window": dark,
        "WindowText": text,
        "Base": QColor(30, 30, 30),
        "AlternateBase": dark.darker(),
        "ToolTipBase": white,
        "ToolTipText": text,
        "Text": text,
        "Button": dark,
        "ButtonText": text,
        "BrightText": QColor(255, 0, 0),
        "Highlight": QColor(42, 130, 218),
        "HighlightedText": white,
    }.items():
        palette.setColor(QPalette.ColorRole[role], color)
    app.setPalette(palette)

    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    win = MainWindow(config_path=config_path)
    win.show()
    return int(app.exec())
