#!/usr/bin/env python3
"""Serve the PolicyShift site locally (with optional FastAPI playback)."""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def serve_static(host: str, port: int, open_browser: bool) -> None:
    if not WEB.exists():
        raise SystemExit(f"Missing web root: {WEB}")
    handler = partial(SimpleHTTPRequestHandler, directory=str(WEB))
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving static site from {WEB}")
    print(f"Open: {url}")
    print("Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def serve_fastapi(host: str, port: int, open_browser: bool) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "FastAPI extras missing. Either:\n"
            "  pip install -e '.[api]'\n"
            "or run static mode:\n"
            "  python scripts/serve_playback.py --static\n"
        ) from exc

    # Ensure portfolio card exists for /api/portfolio
    card = ROOT / "portfolio_export" / "website_card.json"
    if not card.exists():
        from policyshift.portfolio import write_portfolio_export

        write_portfolio_export(ROOT / "portfolio_export")
        print("Generated portfolio_export/ (was missing).")

    url = f"http://127.0.0.1:{port}/"
    print(f"Serving FastAPI playback + site")
    print(f"Open: {url}")
    print("API docs: http://127.0.0.1:{}/docs".format(port))
    if open_browser:
        webbrowser.open(url)
    uvicorn.run("policyshift.api.app:app", host=host, port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Serve apps/web only (no API). Works without fastapi.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open a browser tab.",
    )
    args = parser.parse_args()

    if not _port_free(args.host, args.port):
        raise SystemExit(
            f"Port {args.port} is already in use. Try: "
            f"python scripts/serve_playback.py --port 8010"
        )

    # Prefer package import path
    sys.path.insert(0, str(ROOT / "src"))

    open_browser = not args.no_browser
    if args.static:
        serve_static(args.host, args.port, open_browser)
    else:
        # Fall back to static if api deps missing
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            print("API extras not installed — serving static site instead.")
            print("Tip: pip install -e '.[api]' for trajectory playback.")
            serve_static(args.host, args.port, open_browser)
            return
        serve_fastapi(args.host, args.port, open_browser)


if __name__ == "__main__":
    main()
