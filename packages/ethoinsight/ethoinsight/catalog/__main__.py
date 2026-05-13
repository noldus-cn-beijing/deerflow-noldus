"""Make `python -m ethoinsight.catalog.resolve` work via package __main__."""

from ethoinsight.catalog.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
