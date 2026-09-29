"""Serve the mock support app over HTTP so it's reachable at a real URL.

The app is plain static files, so Python's built-in http.server is enough.

Usage:
    python -m automation.app_server            # serve at http://localhost:8000
    python -m automation.app_server --port 9000

From code (used by the Playwright workflow):
    with serve_mock_app() as url:
        page.goto(url)
"""

import argparse
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

MOCK_APP_DIR = Path(__file__).resolve().parent.parent / "mock_support_app"
DEFAULT_PORT = 8000


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass  # keep agent output readable; http.server logs every request by default


def _make_server(port: int, quiet: bool) -> ThreadingHTTPServer:
    handler = _QuietHandler if quiet else SimpleHTTPRequestHandler
    return ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(MOCK_APP_DIR)))


@contextmanager
def serve_mock_app(port: int = 0) -> Iterator[str]:
    """Serve the app on a background thread and yield its URL. port=0 picks any free port."""
    server = _make_server(port, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the mock support app.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    server = _make_server(args.port, quiet=False)
    print(f"Mock support app running at http://localhost:{args.port}/index.html  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
