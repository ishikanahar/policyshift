#!/usr/bin/env python3
"""Serve Phase 8 artifact playback UI (FastAPI + static web)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Install API extras: pip install 'policyshift[api]'\n" + str(exc)
        ) from exc
    uvicorn.run("policyshift.api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
