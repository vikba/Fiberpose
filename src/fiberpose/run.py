"""Application entry point for launching Fiberpose GUI or CLI.

Installed as the ``fiberpose`` command and run by ``python -m fiberpose``.
"""

from __future__ import annotations

import os

# Must be set before numpy/onnxruntime/torch load their OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys

from . import __version__
from .paths import APP_ICON_PATH, USER_CONFIG_PATH, default_config_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fiberpose", description="Fiberpose launcher (GUI + CLI).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", action="store_true", help="Run command-line batch processing mode.")
    group.add_argument("--gui", action="store_true", help="Force GUI mode (default when --cli is not set).")

    parser.add_argument("--input", dest="input_path", help="Input path for CLI mode (folder, stack, or image).")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help=f"Config JSON path (default: {USER_CONFIG_PATH}).",
    )
    parser.add_argument("--output", dest="output_dir", help="Output directory for CLI exports.")
    parser.add_argument(
        "--measurements",
        action="store_true",
        help="Compute full measurements in CLI mode (default: segmentation only).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        default=None,
        help="Invert input intensities before inference (CLI mode).",
    )
    parser.add_argument(
        "--save-kind",
        choices=("results", "csv"),
        default="results",
        help="CLI export preset kind (default: results).",
    )
    return parser


def _launch_gui(config_path: str) -> int:
    try:
        from PyQt6.QtGui import QColor, QIcon, QPalette
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print('Error: the GUI requires PyQt6. In the Fiberpose folder, run:  pip install -e ".[gui]"')
        return 2

    from .ui_main import MainWindow

    def apply_dark_palette(app: QApplication) -> None:
        app.setStyle("Fusion")
        dark_palette = QPalette()

        dark_color = QColor(45, 45, 45)
        highlight_color = QColor(42, 130, 218)

        dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color.darker())
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        app.setPalette(dark_palette)

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    apply_dark_palette(app)
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    win = MainWindow(config_path=config_path)
    win.show()
    return int(app.exec())


def _launch_cli(args: argparse.Namespace, config_path: str) -> int:
    if not args.input_path:
        print("Error: --input is required with --cli.")
        return 2
    if not os.path.exists(os.path.abspath(args.input_path)):
        print(f"Error: input path does not exist: {os.path.abspath(args.input_path)}")
        return 2

    from .cli_runner import CliRunner

    runner = CliRunner(
        input_path=args.input_path,
        config_path=config_path,
        output_dir=args.output_dir,
        compute_measurements=bool(args.measurements),
        invert_input=args.invert,
        save_kind=args.save_kind,
    )
    return int(runner.run())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.config_path:
        config_path = os.path.abspath(args.config_path)
        if not os.path.exists(config_path):
            print(f"Error: config path does not exist: {config_path}")
            return 2
    else:
        config_path = default_config_path()

    if args.cli:
        return _launch_cli(args, config_path)
    return _launch_gui(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
