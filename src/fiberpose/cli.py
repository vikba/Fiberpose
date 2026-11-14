"""Application entry point for launching the Fiberpose GUI, CLI or web interface.

Installed as the ``fiberpose`` command and run by ``python -m fiberpose``.
"""

from __future__ import annotations

import os

# Must be set before numpy/onnxruntime/torch load their OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

from . import __version__
from .paths import USER_CONFIG_PATH, default_config_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fiberpose", description="Fiberpose launcher (GUI, CLI, web).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cli", action="store_true", help="Run command-line batch processing mode.")
    group.add_argument("--gui", action="store_true", help="Force GUI mode (default when --cli is not set).")
    group.add_argument("--web", action="store_true", help="Serve the Gradio web interface.")

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
        from .gui import launch
    except ImportError:
        print('Error: the GUI requires PyQt6. In the Fiberpose folder, run:  pip install -e ".[gui]"')
        return 2
    return launch(config_path)


def _launch_web(config_path: str) -> int:
    try:
        from .web import launch
    except ImportError:
        print('Error: the web interface requires Gradio. Run:  pip install "fiberpose[web]"')
        return 2
    from .domain import AppConfig

    launch(AppConfig.load_from_json(config_path))
    return 0


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
    if args.web:
        return _launch_web(config_path)
    return _launch_gui(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
