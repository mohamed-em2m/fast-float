"""Package entry point allowing `python -m fast_float ...`."""

import sys

from fast_float.cli import main

if __name__ == "__main__":
    sys.exit(main())
