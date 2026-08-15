# A simple test which imports all python files in src/, except main.py. Useful for integration runners.

import importlib.util
from pathlib import Path

for path in Path("src").glob("*.py"):
    if path.name == "main.py":
        continue

    print(f"Importing {path}")

    spec = importlib.util.spec_from_file_location(
        path.stem,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

print("All modules imported successfully")
