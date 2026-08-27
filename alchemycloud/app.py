"""Entry point for the LiDAR PCD editor + occupancy-grid exporter."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from . import __version__
from .main_window import MainWindow

log = logging.getLogger("alchemycloud")

LOG_FILENAME = "alchemycloud.log"


def log_directory() -> Path:
    """Platform-appropriate directory for the rotating log file."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "alchemycloud"


def configure_logging(verbose: bool = False) -> Path | None:
    """Send INFO+ to a rotating file and WARNING+ to stderr.

    There was previously no logging at all and no log file, so a user hitting a
    problem had nothing to attach to a bug report. Failing to open the file is
    not fatal — the app still runs, it just logs to stderr only.
    """
    root = logging.getLogger("alchemycloud")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG if verbose else logging.WARNING)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    try:
        directory = log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOG_FILENAME
        rotating = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        rotating.setLevel(logging.DEBUG)
        rotating.setFormatter(fmt)
        root.addHandler(rotating)
        return path
    except OSError as exc:  # read-only home, missing permissions, …
        root.warning("Could not open a log file: %s", exc)
        return None


def install_excepthook(parent: QtWidgets.QWidget | None = None) -> None:
    """Route unhandled exceptions to the log and a dialog.

    PyQt5 does not abort on an exception that escapes a slot: it prints a
    traceback to a stderr the user probably cannot see and *carries on* with
    whatever half-updated state caused it. Surfacing it makes the failure
    visible and reportable instead of silent.
    """

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("Unhandled exception:\n%s", text)
        try:
            QtWidgets.QMessageBox.critical(
                parent,
                "Unexpected error",
                f"{exc_type.__name__}: {exc}\n\n"
                "The app may be in an inconsistent state — save your work to a new "
                "file and restart. Details are in the log file.",
            )
        except Exception:  # noqa: BLE001 - never let the hook itself explode
            pass

    sys.excepthook = hook


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="alchemycloud",
        description="Edit LiDAR point clouds and export ROS occupancy grids.",
    )
    parser.add_argument("path", nargs="?", help="point cloud to open on startup (.pcd/.ply/.xyz)")
    parser.add_argument("--version", action="version", version=f"alchemycloud {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    log_path = configure_logging(args.verbose)
    log.info("AlchemyCloud %s starting", __version__)

    # Opt into HiDPI *before* the QApplication exists — Qt5 ignores these
    # afterwards. Without them the UI is rendered at 1x and upscaled on a
    # scaled display, so text and the point cloud both look soft.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv[:1])
    # Identity: without it the window manager shows "python3" and QSettings
    # would write under an empty organisation.
    app.setApplicationName("AlchemyCloud")
    app.setApplicationDisplayName("AlchemyCloud")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("AlchemyCloud")

    w = MainWindow()
    install_excepthook(w)
    w.show()
    if log_path is not None:
        log.info("Logging to %s", log_path)

    if args.path:
        # Check it here rather than handing a bad path to the loader, so a typo
        # produces one clear message instead of an Open-failed dialog.
        if Path(args.path).is_file():
            w.load_path(args.path)
        else:
            QtWidgets.QMessageBox.warning(w, "File not found", f"No such file: {args.path}")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
