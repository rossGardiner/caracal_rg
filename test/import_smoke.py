"""Explicit full-runtime import smoke check, independent of the caller's cwd.

Run as a separate process: python test/import_smoke.py
Importing this helper itself does not import application modules.
"""

import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    modules = sorted(
        path.stem
        for path in (PROJECT_ROOT / "src").glob("*.py")
        if path.name not in {"main.py", "__init__.py"}
    )
    if not modules:
        raise RuntimeError(f"No source modules found in {PROJECT_ROOT / 'src'}")

    # Script execution otherwise puts test/, rather than the project, on sys.path.
    sys.path.insert(0, str(PROJECT_ROOT))
    for name in modules:
        print(f"Importing src.{name}", flush=True)
        importlib.import_module(f"src.{name}")

    print(f"All {len(modules)} modules imported successfully")


if __name__ == "__main__":
    main()
