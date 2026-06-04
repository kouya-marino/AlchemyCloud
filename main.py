"""Dev entry point: launch AlchemyCloud without installing.

For the installed console command use `alchemycloud` (see pyproject.toml).
"""
from alchemycloud.app import main

if __name__ == "__main__":
    raise SystemExit(main())
