"""Vercel Python function for /api/classify (logic lives in _core.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core import ApiHandler as handler  # noqa: E402,F401
