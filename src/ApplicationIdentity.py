"""Single source of truth for the application's public-facing name.

Change *only* APP_NAME to rename the product. Executable names, archives,
Qt settings names, documentation variable names and per-user folders are
all derived automatically.

Project format and extension are intentionally stable protocol identifiers:
renaming a product must never make existing project files unreadable.
Caracal references below are only for backwards compatibility.
"""

import re
import unicodedata
from pathlib import Path

# The only product-name setting you need to edit.
APP_NAME = "Audio Analysis"


def _name_tokens(name: str) -> list[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[A-Za-z0-9]+", ascii_name)
    if not words:
        raise ValueError("APP_NAME must contain at least one letter or number")
    return words


_TOKENS = _name_tokens(APP_NAME)
BINARY_NAME = "".join(word[:1].upper() + word[1:] for word in _TOKENS)
STORAGE_DIRECTORY = "-".join(word.lower() for word in _TOKENS)
ENV_PREFIX = "_".join(word.upper() for word in _TOKENS)
ORGANIZATION_NAME = BINARY_NAME
SETTINGS_NAME = BINARY_NAME
CI_SMOKE_TEST_ENV = f"{ENV_PREFIX}_CI_SMOKE_TEST"
WIKI_URL_ENV = f"{ENV_PREFIX}_WIKI_URL"
DEVELOPER_DOCS_URL_ENV = f"{ENV_PREFIX}_DEVELOPER_DOCS_URL"

# Project identifiers are a file-format contract; do NOT rename with APP_NAME.
PROJECT_EXTENSION = ".audioproj"
PROJECT_FORMAT = "audio-analysis-project"

# Old formats/settings remain readable after changes to the product name.
LEGACY_PROJECT_EXTENSION = ".caracal"
LEGACY_PROJECT_FORMAT = "caracal-project"
LEGACY_STORAGE_DIRECTORY = "caracal"
LEGACY_STORAGE_DIRECTORIES = ("audio-analysis", "caracal")
LEGACY_SETTINGS_IDENTITIES = (("AudioAnalysis", "AudioAnalysis"), ("Caracal", "Caracal"))


def existing_or_default_path(root: Path, filename: str) -> Path:
    """Find data from a previous app name without silently discarding it."""
    target = root / STORAGE_DIRECTORY / filename
    if target.exists():
        return target
    for directory in LEGACY_STORAGE_DIRECTORIES:
        old = root / directory / filename
        if old.exists():
            return old
    return target
