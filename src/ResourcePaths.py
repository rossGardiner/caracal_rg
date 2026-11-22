"""Locate read-only assets in a checkout or a PyInstaller bundle."""

import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Return an absolute path for an asset shipped with the application."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return (root / relative_path).resolve()
