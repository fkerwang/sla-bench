"""Bundled audited v7 Locust OpenAI SSE benchmark client."""

from pathlib import Path

CLIENT_LOCUSTFILE = Path(__file__).resolve().parent / "client.py"

__all__ = ["CLIENT_LOCUSTFILE"]
