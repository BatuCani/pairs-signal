# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "langfuse>=3,<4"]
# ///
"""News Gate local dev server: serves index.html plus the same /api routes as the Vercel functions.

Run: uv run ui/server.py  ->  http://localhost:8000
"""

from __future__ import annotations

import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "api"))
from _core import *  # noqa: E402,F403  (re-exported for ui/terminal_demo.py)
from _core import DATA_DIR, LANGFUSE_ON, MOCK, UI_DIR, VARIANTS, ApiHandler  # noqa: E402


STATIC = {"/polish.css": "text/css; charset=utf-8", "/polish.js": "text/javascript; charset=utf-8"}


class Handler(ApiHandler):
    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/", "/index.html"):
            self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path.split("?", 1)[0] in STATIC:
            name = self.path.split("?", 1)[0].lstrip("/")
            self._send(200, (UI_DIR / name).read_bytes(), STATIC["/" + name])
        else:
            super().do_GET()


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    mode = "MOCK" if MOCK else ", ".join(f"{k}={v['model']}" for k, v in VARIANTS.items())
    print(f"News Gate UI on http://localhost:{port}  models: {mode}  langfuse: {'on' if LANGFUSE_ON else 'off'}  data: {DATA_DIR}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
