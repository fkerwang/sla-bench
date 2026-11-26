"""Helpers for per-run acc answer logs (timestamp + full reply text)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ_CN = timezone(timedelta(hours=8))

ANSWER_LOG_KEYS = (
    "sample_index",
    "timestamp",
    "timestamp_iso",
    "request_id",
    "source_dataset",
    "tier",
    "n_tokens",
    "prompt_len",
    "output_len",
    "gold_ans",
    "answers",
    "extracted_answer",
    "score",
    "keyword_recall",
    "gated",
    "reasoning_effort",
    "generated_text",
    "error",
    "ttft",
    "e2e_lat",
)


def timestamp_iso(ts: float | None = None) -> str:
    t = datetime.fromtimestamp(ts if ts is not None else datetime.now(TZ_CN).timestamp(), tz=TZ_CN)
    return t.isoformat(timespec="milliseconds")


def enrich_row(row: dict) -> dict:
    out = dict(row)
    if out.get("timestamp") is not None and not out.get("timestamp_iso"):
        try:
            out["timestamp_iso"] = timestamp_iso(float(out["timestamp"]))
        except (TypeError, ValueError):
            pass
    return out


def answer_log_row(row: dict) -> dict:
    row = enrich_row(row)
    return {k: row[k] for k in ANSWER_LOG_KEYS if k in row}


def write_answer_log(detail_path: Path, out_path: Path | None = None) -> Path:
    """Write sorted answer_log.jsonl next to detailed_results.jsonl."""
    detail_path = Path(detail_path)
    out_path = out_path or (detail_path.parent / "answer_log.jsonl")
    rows: list[dict] = []
    with detail_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(answer_log_row(json.loads(line)))

    def sort_key(r: dict) -> tuple:
        si = r.get("sample_index")
        if si is not None:
            return (0, int(si))
        ts = r.get("timestamp")
        return (1, float(ts) if ts is not None else 0.0)

    rows.sort(key=sort_key)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
