"""Backward-compatible shim.

The project renamed `save_tables.py` to `save_csvs.py`.
Import from `code.results.save_csvs` going forward.
"""

from .save_csvs import *  # noqa: F401,F403
