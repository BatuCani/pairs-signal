"""Vercel Python function for /api/feed (logic lives in _core.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core import ApiHandler  # noqa: E402


class handler(ApiHandler):
    pass
