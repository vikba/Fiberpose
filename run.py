"""Application entry point for launching Fiberpose GUI or CLI."""

from __future__ import annotations

import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "resources", "config.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fiberpose launcher (GUI + CLI).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", action="store_true", help="Run command-line batch processing mode.")
    group.add_argument("--gui", action="store_true", help="Force GUI mode (default when --cli is not set).")

    parser.add_argument("--input", dest="input_path", help="Input path for CLI mode (folder, stack, or image).")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Config JSON path (default: {DEFAULT_CONFIG_PATH}).",
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


def _launch_gui() -> int:
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtWidgets import QApplication

    from sources.ui_main import MainWindow

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
    apply_dark_palette(app)
    win = MainWindow()
    win.show()
    return int(app.exec())


def _launch_cli(args: argparse.Namespace) -> int:
    if not args.input_path:
        print("Error: --input is required with --cli.")
        return 2
    if not os.path.exists(os.path.abspath(args.input_path)):
        print(f"Error: input path does not exist: {os.path.abspath(args.input_path)}")
        return 2
    if args.config_path and not os.path.exists(os.path.abspath(args.config_path)):
        print(f"Error: config path does not exist: {os.path.abspath(args.config_path)}")
        return 2

    from sources.cli_runner import CliRunner

    runner = CliRunner(
        input_path=args.input_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        compute_measurements=bool(args.measurements),
        invert_input=args.invert,
        save_kind=args.save_kind,
    )
    return int(runner.run())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cli:
        return _launch_cli(args)
    return _launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
