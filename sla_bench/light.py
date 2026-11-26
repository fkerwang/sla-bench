"""Ultra-light result files: one BurstGPT-style JSONL per model, one row per request rate.

The row keeps the BurstGPT summary keys (`request_throughput`, `mean_ttft_ms`, `mean_tpot_ms`, `completed`,
`total_input_tokens`, `total_output_tokens`, `stable`, ...) that the downstream importer reads, and adds
`acc`, input-length and real-output-length (`eff_*`, tokens before the pad delimiter) summaries. Unknown keys
are ignored by that importer. `rounds` holds the full per-point records (one per replica/round), so the HTML
report can also be rebuilt from these files alone.

File name (parsed by the importer):
  - FP8 (default): `<served_model>_<N>card_TP<tp>_FP8_<backend>_<date>_burst.jsonl` -> `<out_dir>/<card>/`
  - BF16: `<served_model>_<N>card_TP<tp>_BF16_<backend>_<date>_bf16_burst.jsonl` -> `<out_dir>/<card>-bf16/`
written under `<out_dir>/`.
"""

from __future__ import annotations

import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .stats import by_rate, is_point_valid, pct

META_KEYS = (
    "card", "model", "served_model_name", "tp", "cards", "card_num", "replicas", "precision", "arrival", "methodology",
)
STAT_KEYS = (
    "n", "acc", "n_scored", "in_min", "in_avg", "in_p50", "in_max", "out_min", "out_avg", "out_max",
    "eff_min", "eff_avg", "eff_max", "eff_miss", "eff_unknown",
)
METHODOLOGY = "v7 open-loop Poisson, padded fixed output, len_tol gate"
LIGHT_SUFFIX = "_burst.jsonl"

_DTYPE_ALIASES = {
    "BFLOAT16": "BF16",
    "BF16": "BF16",
    "FLOAT16": "FP16",
    "HALF": "FP16",
    "FP16": "FP16",
    "FLOAT32": "FP32",
    "FP32": "FP32",
    "FLOAT8_E4M3FN": "FP8",
    "FLOAT8_E5M2": "FP8",
    "FP8_E4M3": "FP8",
    "E4M3": "FP8",
    "FP8": "FP8",
}
_MODEL_PREC_RE = re.compile(
    r"(?:^|[-_./])(nvfp4|mxfp4|fp8|fp4|w8a8|w4a16|int8|int4|awq|gptq|fp16|bf16)(?:$|[-_./])",
    re.IGNORECASE,
)
_FILE_PREC_RE = re.compile(r"_TP\d+_([A-Za-z0-9]+)_")


def infer_precision(
    model: str = "",
    served_model_name: str = "",
    precision: str = "BF16",
    filename: str = "",
) -> str:
    """Infer canonical precision tag (`FP8`, `BF16`, `W8A8`, `INT8`, ...) from spec/model/dtype/filename."""
    raw_prec = (precision or "").strip().upper()
    norm_prec = _DTYPE_ALIASES.get(raw_prec, raw_prec)
    if norm_prec and norm_prec not in ("AUTO", "BF16"):
        return norm_prec

    for candidate in (model, served_model_name):
        if not candidate:
            continue
        matches = _MODEL_PREC_RE.findall(candidate)
        if matches:
            tag = matches[-1].upper()
            return _DTYPE_ALIASES.get(tag, tag)

    if filename:
        m = _FILE_PREC_RE.search(filename)
        if m:
            tag = m.group(1).upper()
            return _DTYPE_ALIASES.get(tag, tag)

    return norm_prec or "BF16"


def burst_card_folder(card: str, precision: str) -> str:
    """Card folder name (`<card>` for default FP8; `<card>-bf16` for BF16)."""
    base = (card or "").strip()
    prec = (precision or "").strip().upper()
    if prec == "BF16" and base and not base.lower().endswith("-bf16"):
        return f"{base}-bf16"
    return base


def fmt_cards(cards: float) -> str:
    return str(int(round(cards))) if abs(cards - round(cards)) < 1e-9 else f"{cards:g}"


def iter_light_files(root: Path) -> List[Path]:
    """Burst-style summary files anywhere under `root` (depth <= 3: `<root>`, `<root>/light`,
    and `<root>/light/<card>` as produced by `export`)."""
    root = Path(root)
    if not root.is_dir():
        return []
    found: set[Path] = set()
    for pat in (f"*{LIGHT_SUFFIX}", f"*/*{LIGHT_SUFFIX}", f"*/*/*{LIGHT_SUFFIX}"):
        found |= set(root.glob(pat))
    return sorted(found)


def _first_row(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    return {}


def find_light(root: Path, model: str) -> Optional[Path]:
    """Light file under `root` (depth <= 2) whose rows belong to the model dir name `model`."""
    for p in iter_light_files(root):
        try:
            if _first_row(p).get("model") == model:
                return p
        except (OSError, ValueError):
            continue
    return None


def read_light(path: Path) -> Tuple[dict, List[dict]]:
    """Return `(meta, points)`; each point carries its summary under `_raw` (report-internal)."""
    path = Path(path)
    meta: dict = {}
    pts: List[dict] = []
    seen_idx: Dict[tuple, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not meta:
                meta = {k: row[k] for k in META_KEYS if k in row}
            for r in row.get("rounds", []):
                m = dict(r)
                m["_raw"] = m.pop("stats", None)
                dedup_key = (
                    round(float(m.get("rate") or 0.0), 6),
                    m.get("port"),
                    m.get("ttft"),
                    m.get("tpot"),
                    m.get("n"),
                )
                if dedup_key in seen_idx:
                    prev_i = seen_idx[dedup_key]
                    if pts[prev_i].get("src") in (None, "", ".") and m.get("src") not in (None, "", "."):
                        pts[prev_i] = m
                    continue
                seen_idx[dedup_key] = len(pts)
                pts.append(m)
    if meta:
        meta["precision"] = infer_precision(
            str(meta.get("model") or ""),
            str(meta.get("served_model_name") or ""),
            str(meta.get("precision") or ""),
            path.name,
        )
    return meta, pts


def _num_stats(xs: Sequence[float]) -> Tuple[float, float, float]:
    return (min(xs), st.mean(xs), max(xs)) if xs else (0.0, 0.0, 0.0)


def _build_rows(results_dir: Path, sp, cfg, pts: List[dict], replicas: int, date_str: str, precision: str = "") -> List[dict]:
    from .report import _raw_point_stats

    prec = precision or infer_precision(sp.model, sp.served_model_name, sp.precision)
    cards = sp.card_num
    rows: List[dict] = []
    for rate, ms in by_rate(pts).items():
        rounds, keeps = [], []
        for m in sorted(ms, key=lambda x: (str(x.get("src", "")), int(x.get("port") or 0))):
            a = _raw_point_stats(results_dir, m, sp.model, sp.v7_mode, sp.tokenizer_url or "", sp.served_model_name)
            keeps.extend(a["recs"] if a else [])
            r = {k: v for k, v in m.items() if not k.startswith("_")}
            r["stats"] = {k: a[k] for k in STAT_KEYS} if a else None
            rounds.append(r)
        if not keeps:
            raise ValueError(f"No raw per-request records for {sp.model} @ rate {rate}; run `export` where the raw results live.")
        valid = all(is_point_valid(m, v7_mode=sp.v7_mode) for m in ms)
        ttfts = [float(r["ttft"]) for r in keeps]
        tpots = [float(r["tpot"]) for r in keeps if (r.get("output_len") or 0) > 5]
        e2es = [float(r["e2e_lat"]) for r in keeps]
        in_lens = [int(r.get("prompt_len") or 0) for r in keeps]
        out_lens = [int(r.get("output_len") or 0) for r in keeps]
        effs = [int(r["eff_len"]) for r in keeps if r.get("eff_len") is not None]
        unknown = sum(1 for r in keeps if r.get("_eff_unknown"))
        scores = [float(r["score"]) for r in keeps if r.get("score") is not None]
        completed = len(keeps)
        dur = max(
            max(float(r["timestamp"]) for r in keeps) - min(float(r["timestamp"]) - float(r["e2e_lat"]) / 1000.0 for r in keeps),
            1e-6,
        )
        meas = [float(m["rate_sent"]) for m in ms if m.get("rate_sent")]
        req_tp = float(rate) if valid else float(min(rate, st.mean(meas) if meas else rate))
        avg_in, avg_out = st.mean(in_lens), st.mean(out_lens)
        tpot_mean = st.mean(tpots) if tpots else 0.0
        e_min, e_avg, e_max = _num_stats(effs)
        i_min, i_avg, i_max = _num_stats(in_lens)
        rows.append({
            "date": date_str, "endpoint_type": "openai", "backend": sp.backend,
            "label": f"TP{sp.tp}_{fmt_cards(cards)}card", "model_id": sp.served_model_name,
            "tokenizer_id": sp.served_model_name,
            "num_prompts": int(sum(int(m.get("n_planned") or m.get("n") or 0) for m in ms)),
            "request_rate": round(float(rate), 6), "burstiness": 1.0, "max_concurrency": None,
            "duration": round(dur, 3), "completed": completed,
            "failed": int(sum(int(m.get("errors") or 0) + max(0, int(m.get("missing") or 0)) for m in ms)),
            "total_input_tokens": int(sum(in_lens)), "total_output_tokens": int(sum(out_lens)),
            "request_throughput": round(req_tp, 6), "request_goodput": None,
            "output_throughput": round(req_tp * avg_out, 4),
            "total_token_throughput": round(req_tp * (avg_in + avg_out), 4),
            "mean_ttft_ms": round(st.mean(ttfts), 3), "median_ttft_ms": round(st.median(ttfts), 3),
            "std_ttft_ms": round(st.pstdev(ttfts) if len(ttfts) > 1 else 0.0, 3), "p99_ttft_ms": round(pct(ttfts, 0.99), 3),
            "mean_tpot_ms": round(tpot_mean, 3), "median_tpot_ms": round(st.median(tpots) if tpots else 0.0, 3),
            "std_tpot_ms": round(st.pstdev(tpots) if len(tpots) > 1 else 0.0, 3),
            "p99_tpot_ms": round(pct(tpots, 0.99) if tpots else 0.0, 3),
            "mean_itl_ms": round(tpot_mean, 3), "median_itl_ms": round(st.median(tpots) if tpots else 0.0, 3),
            "std_itl_ms": round(st.pstdev(tpots) if len(tpots) > 1 else 0.0, 3),
            "p99_itl_ms": round(pct(tpots, 0.99) if tpots else 0.0, 3),
            "mean_e2e_latency_ms": round(st.mean(e2es), 3),
            "tp": sp.tp, "card_num": cards, "cards": cards, "card": sp.card or cfg.card,
            "model": sp.model, "served_model_name": sp.served_model_name, "precision": prec, "replicas": replicas,
            "arrival": cfg.sweep.arrival, "methodology": METHODOLOGY,
            "successful_requests_min": 50, "meets_min_completed": bool(completed >= 50),
            "stable": bool(valid),
            "max_round_ttft_ms": round(max(float(m["ttft"]) for m in ms), 3),
            "max_round_tpot_ms": round(max(float(m["tpot"]) for m in ms), 3),
            "rounds_count": len(ms),
            "acc": round(st.mean(scores) * 100.0, 3) if scores else None, "n_scored": len(scores),
            "in_min": i_min, "in_avg": round(i_avg, 2), "in_max": i_max,
            "real_out_min": e_min if effs else None, "real_out_avg": round(e_avg, 2) if effs else None,
            "real_out_max": e_max if effs else None,
            "real_out_n": len(effs), "real_out_missing": completed - len(effs) - unknown,
            "rounds": rounds,
        })
    return rows


def export_light(cfg, out_dir: Optional[str | Path] = None, specs: Optional[Sequence] = None) -> List[Path]:
    """Write `<out_dir>/<card>/<served>_<N>card_TP<tp>_<prec>_<backend>_<date>_burst.jsonl` per spec."""
    from .report import auto_discover_specs_from_dir, load_clean_spec_points
    from .sweep import discover_round_roots

    results_dir = Path(cfg.results_dir)
    out = Path(out_dir) if out_dir else results_dir / "light"
    specs = list(specs or cfg.specs or auto_discover_specs_from_dir(results_dir, card=cfg.card))
    written: List[Path] = []
    for sp in specs:
        pts = load_clean_spec_points(results_dir, sp, stab_z=cfg.sweep.stab_z, restabilize=True)
        if not pts:
            continue
        prec = infer_precision(sp.model, sp.served_model_name, sp.precision)
        replicas, latest = 1, 0.0
        for rp in discover_round_roots(results_dir, sp.model, sp.roots or None):
            cj = rp / sp.model / "curve.json"
            if cj.is_file():
                replicas = int(json.loads(cj.read_text(encoding="utf-8")).get("replicas") or replicas)
                latest = max(latest, cj.stat().st_mtime)
        date_str = datetime.fromtimestamp(latest or datetime.now().timestamp()).strftime("%Y%m%d-%H%M%S")
        rows = _build_rows(results_dir, sp, cfg, pts, replicas, date_str, precision=prec)
        card_dir = out / burst_card_folder(sp.card or cfg.card, prec)
        card_dir.mkdir(parents=True, exist_ok=True)
        prec_tail = "_bf16" if prec == "BF16" else ""
        dest = card_dir / (
            f"{sp.served_model_name}_{fmt_cards(sp.card_num)}card_TP{sp.tp}_{prec}_{sp.backend}_{date_str}{prec_tail}{LIGHT_SUFFIX}"
        )
        with open(dest, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.append(dest)
    return written
