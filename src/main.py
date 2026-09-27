"""Compatibility entry point for the interactive explorer."""

import sys

from src.explorer import main


if __name__ == "__main__":
    sys.exit(
        main()
    )
