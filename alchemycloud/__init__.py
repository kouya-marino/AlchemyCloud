"""AlchemyCloud — LiDAR point-cloud editor and ROS occupancy-grid exporter."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version_from_source_tree() -> str | None:
    """Read `version` out of a sibling pyproject.toml, for uninstalled checkouts.

    `python main.py` from a clone has no distribution metadata, and reporting
    "0.0.0+unknown" in the window title and `--version` would be a worse answer
    than the truth sitting two directories up.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        import tomllib  # Python 3.11+

        with pyproject.open("rb") as f:
            return str(tomllib.load(f)["project"]["version"])
    except (OSError, KeyError, ModuleNotFoundError):
        pass
    # Python 3.10 has no tomllib; a targeted scan beats adding a dependency.
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
        elif in_project and stripped.startswith("version"):
            _, _, raw = stripped.partition("=")
            return raw.strip().strip('"').strip("'") or None
    return None


# Single source of truth: the installed distribution's metadata, which comes
# from pyproject.toml. The version used to be written independently in
# pyproject.toml, CHANGELOG.md, version_data.md and README.md, with nothing
# keeping the four in step.
try:
    __version__ = version("alchemycloud")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = _version_from_source_tree() or "0.0.0+unknown"

__all__ = ["__version__"]
