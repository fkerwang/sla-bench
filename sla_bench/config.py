"""Configuration models and YAML/CLI loaders for sla_bench."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

DEFAULT_PALETTE = (
    "#2563EB",
    "#F59E0B",
    "#7C3AED",
    "#D1495B",
    "#0D9488",
    "#16A34A",
    "#0EA5E9",
    "#EA580C",
)


@dataclass
class SLALevel:
    """One SLA latency tier (e.g. SLA① TTFT<=2005ms & TPOT<=51ms)."""

    label: str
    ttft_ms: float
    tpot_ms: float
    color: str = "#DC2626"


def default_slas() -> List[SLALevel]:
    return [
        SLALevel(label="①", ttft_ms=2005.0, tpot_ms=51.0, color="#DC2626"),
        SLALevel(label="②", ttft_ms=1000.0, tpot_ms=20.0, color="#EA580C"),
    ]


def parse_sla_arg(spec: str, label: str, color: str) -> SLALevel:
    """Parse 'TTFT:TPOT' string (e.g. '2005:51') into an SLALevel."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid SLA format {spec!r}; expected TTFT_MS:TPOT_MS (e.g. 2005:51)")
    return SLALevel(label=label, ttft_ms=float(parts[0]), tpot_ms=float(parts[1]), color=color)


def normalize_urls(ports: Optional[Sequence[int]] = None, urls: Optional[Sequence[str]] = None, host: str = "127.0.0.1") -> List[str]:
    """Convert port numbers and/or endpoint URLs into canonical /v1/chat/completions URLs."""
    out: List[str] = []
    if urls:
        for u in urls:
            u = u.strip()
            if not u:
                continue
            if "://" not in u:
                u = f"http://{u}"
            if u.endswith("/"):
                u = u.rstrip("/")
            if not u.endswith("/v1/chat/completions"):
                u = f"{u}/v1/chat/completions"
            out.append(u)
    if ports:
        for p in ports:
            out.append(f"http://{host}:{int(p)}/v1/chat/completions")
    return out


def extract_port_from_url(url: str, fallback_idx: int = 0) -> int:
    """Extract TCP port from an HTTP URL for per-replica directory naming."""
    m = re.search(r":(\d+)(?:/|$)", url)
    if m:
        return int(m.group(1))
    return 8000 + fallback_idx


def health_url_from_chat_url(chat_url: str, health_path: str = "/health_generate") -> str:
    """Derive server health check URL from a /v1/chat/completions URL."""
    base = re.sub(r"/v1/chat/completions/?$", "", chat_url.rstrip("/"))
    if not health_path.startswith("/"):
        health_path = "/" + health_path
    return base + health_path


def strip_tp_suffix(name: str) -> str:
    return re.sub(r"-tp\d+$", "", name)


@dataclass
class ModelTPSpec:
    """Specification for one (card, model, TP) configuration to sweep/report/export."""

    key: str
    card: str
    model: str  # Directory / config identifier, e.g. Qwen3.8-27B-tp2
    served_model_name: str  # OpenAI model field, e.g. Qwen3.8-27B
    model_short: str  # Display short name, e.g. 27B dense or 35B-A3B MoE
    tp: int
    cards: float  # physical cards occupied per replica (fractions like 0.25/0.33/0.5 = part of one card)
    tokenizer: str = ""
    tokenizer_url: str = ""
    ds_in: Optional[float] = None
    urls: List[str] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)
    roots: List[str] = field(default_factory=list)
    color: str = "#2563EB"
    precision: str = "BF16"
    backend: str = "sglang"
    v7_mode: bool = True
    port_labels: Dict[int, str] = field(default_factory=dict)

    @property
    def card_num(self) -> float:
        return float(self.cards)

    @property
    def display_label(self) -> str:
        c_str = f"{self.card_num:g}"
        return f"{self.model_short} · TP={self.tp} · {c_str} card"


@dataclass
class ExternalBaselineSpec:
    """Optional external card baseline row for cross-platform comparison in the HTML report."""

    key: str
    card: str
    model: str  # Subdirectory name inside roots, e.g. Qwen3.5-35B-A3B or Qwen3.8-27B-tp2
    canonical_model: str  # Canonical model name for grouping, e.g. Qwen3.8-27B
    tp: int
    cards: float  # physical cards occupied per replica
    roots: List[str]
    how_note: str = ""
    v7_mode: bool = True
    color: str = "#059669"
    precision: str = "BF16"
    backend: str = "sglang"

    @property
    def card_num(self) -> float:
        return float(self.cards)


@dataclass
class SweepSettings:
    """Runtime & statistical parameters for v7 sweeps and adaptive gap<10% refinement."""

    rounds: int = 2
    warmup_s: float = 60.0
    min_s: float = 180.0
    max_s: float = 2400.0
    max_tokens: int = 1000
    len_tol: float = 0.03
    min_completed: int = 0
    min_per_round: int = 32
    stab_z: float = 3.0
    gap_threshold: float = 0.10
    min_bracket_rel: float = 0.025
    max_refine_steps: int = 6
    arrival: str = "poisson"
    cap_users: int = 1024
    spawn_rate: float = 50.0
    inter_round_cooldown_s: float = 15.0
    health_check: bool = True
    health_path: str = "/health_generate"
    health_timeout_s: float = 10.0
    locust_bin: str = ""
    dataset_path: str = ""
    cache_dir: str = ""
    # Fixed accuracy pass: one open-loop pass over a fixed slice of the dataset (`acc_seed=0`,
    # `acc_warm_prompts=128` warm-up items followed by `acc_prompts=300` scored items), matching
    # the H20 baseline question set byte-for-byte across cards and re-runs.
    acc_prompts: int = 300
    acc_warmup_s: float = 60.0
    acc_warm_prompts: int = 128
    acc_cap_users: int = 256
    acc_seed: int = 0


@dataclass
class BenchConfig:
    """Top-level configuration combining card metadata, sweep settings, SLAs, and model specs."""

    title: str = ""
    card: str = "GPU"
    results_dir: str = "results/run"
    report_out: str = "results/run/sla_report.html"
    methodology_note: str = ""
    slas: List[SLALevel] = field(default_factory=default_slas)
    sweep: SweepSettings = field(default_factory=SweepSettings)
    specs: List[ModelTPSpec] = field(default_factory=list)
    external_baselines: List[ExternalBaselineSpec] = field(default_factory=list)


def load_bench_config(yaml_path: str | Path) -> BenchConfig:
    """Load a BenchConfig from a YAML file."""
    p = Path(yaml_path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    if raw.get("card") in (None, ""):
        raise ValueError(
            f"{p}: missing required top-level field `card`: write the real card model of the machine under test."
        )
    legacy = {"die_per_card", "dies"}
    if legacy & set(raw) or any(legacy & set(x) for x in (raw.get("models", []) or []) + (raw.get("external_baselines", []) or [])):
        raise ValueError(
            f"{p}: `die_per_card` / `dies` are no longer supported. Use `cards` (physical cards occupied per "
            "replica); for a replica that uses only part of a card write 0.25 / 0.33 / 0.5."
        )
    for _sec in ("models", "external_baselines"):
        for _i, _item in enumerate(raw.get(_sec, []) or []):
            if _item.get("cards") in (None, ""):
                raise ValueError(
                    f"{p}: {_sec}[{_i}] ({_item.get('model')}): `cards` (physical cards occupied per replica) is required."
                )
    card = str(raw["card"])
    results_dir = str(raw.get("results_dir", "results/run"))
    report_out = str(raw.get("report_out", f"{results_dir}/sla_report.html"))
    title = str(raw.get("title", f"{card} · 不同 TTFT / TPOT 门限下的 QPM/card"))
    methodology_note = str(raw.get("methodology_note", ""))

    slas_raw = raw.get("slas")
    if slas_raw:
        default_cols = ["#DC2626", "#EA580C", "#2563EB", "#059669"]
        slas = [
            SLALevel(
                label=str(s.get("label", f"#{i + 1}")),
                ttft_ms=float(s["ttft_ms"]),
                tpot_ms=float(s["tpot_ms"]),
                color=str(s.get("color", default_cols[i % len(default_cols)])),
            )
            for i, s in enumerate(slas_raw)
        ]
    else:
        slas = default_slas()

    sw_raw = raw.get("sweep", {}) or {}
    sweep = SweepSettings(
        rounds=int(sw_raw.get("rounds", 2)),
        warmup_s=float(sw_raw.get("warmup_s", 30.0)),
        min_s=float(sw_raw.get("min_s", 180.0)),
        max_s=float(sw_raw.get("max_s", 2400.0)),
        max_tokens=int(sw_raw.get("max_tokens", 1000)),
        len_tol=float(sw_raw.get("len_tol", 0.03)),
        min_completed=int(sw_raw.get("min_completed", 0)),
        min_per_round=int(sw_raw.get("min_per_round", 32)),
        stab_z=float(sw_raw.get("stab_z", 3.0)),
        gap_threshold=float(sw_raw.get("gap_threshold", 0.10)),
        min_bracket_rel=float(sw_raw.get("min_bracket_rel", 0.025)),
        max_refine_steps=int(sw_raw.get("max_refine_steps", 6)),
        arrival=str(sw_raw.get("arrival", "poisson")),
        cap_users=int(sw_raw.get("cap_users", 1024)),
        spawn_rate=float(sw_raw.get("spawn_rate", 50.0)),
        inter_round_cooldown_s=float(sw_raw.get("inter_round_cooldown_s", 15.0)),
        health_check=bool(sw_raw.get("health_check", True)),
        health_path=str(sw_raw.get("health_path", "/health_generate")),
        health_timeout_s=float(sw_raw.get("health_timeout_s", 10.0)),
        locust_bin=str(sw_raw.get("locust_bin", "")),
        dataset_path=str(sw_raw.get("dataset_path", "")),
        cache_dir=str(sw_raw.get("cache_dir", "")),
        acc_prompts=int(sw_raw.get("acc_prompts", 200)),
        acc_warmup_s=float(sw_raw.get("acc_warmup_s", 30.0)),
        acc_cap_users=int(sw_raw.get("acc_cap_users", 256)),
        acc_seed=int(sw_raw.get("acc_seed", 0)),
    )

    specs: List[ModelTPSpec] = []
    for idx, item in enumerate(raw.get("models", []) or []):
        m_card = str(item.get("card", card))
        tp = int(item.get("tp", 1))
        cards = float(item["cards"])
        model_dir = str(item["model"])
        served = str(item.get("served_model_name") or strip_tp_suffix(model_dir))
        short = str(item.get("model_short") or served)
        key = str(item.get("key") or f"{served}_tp{tp}_{idx}")
        ports = [int(x) for x in (item.get("ports") or [])]
        urls = normalize_urls(ports=ports, urls=item.get("urls"), host=str(item.get("host", "127.0.0.1")))
        rates = [float(x) for x in (item.get("rates") or [])]
        roots = [str(x) for x in (item.get("roots") or [])]
        port_labels = {int(pk): str(pv) for pk, pv in (item.get("port_labels") or {}).items()}
        specs.append(
            ModelTPSpec(
                key=key,
                card=m_card,
                model=model_dir,
                served_model_name=served,
                model_short=short,
                tp=tp,
                cards=cards,
                tokenizer=str(item.get("tokenizer", "")),
                tokenizer_url=str(item.get("tokenizer_url", "")),
                ds_in=float(item["ds_in"]) if item.get("ds_in") is not None else None,
                urls=urls,
                rates=rates,
                roots=roots,
                color=str(item.get("color") or DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]),
                precision=str(item.get("precision", "BF16")),
                backend=str(item.get("backend", "sglang")),
                v7_mode=bool(item.get("v7_mode", True)),
                port_labels=port_labels,
            )
        )

    ext_specs: List[ExternalBaselineSpec] = []
    for idx, item in enumerate(raw.get("external_baselines", []) or []):
        tp = int(item.get("tp", 2))
        cards = float(item["cards"])
        model_dir = str(item["model"])
        canon = str(item.get("canonical_model") or strip_tp_suffix(model_dir))
        ext_specs.append(
            ExternalBaselineSpec(
                key=str(item.get("key") or f"ext_{idx}"),
                card=str(item["card"]),
                model=model_dir,
                canonical_model=canon,
                tp=tp,
                cards=cards,
                roots=[str(x) for x in (item.get("roots") or [])],
                how_note=str(item.get("how_note", "")),
                v7_mode=bool(item.get("v7_mode", True)),
                color=str(item.get("color", "#059669")),
                precision=str(item.get("precision", "BF16")),
                backend=str(item.get("backend", "sglang")),
            )
        )

    return BenchConfig(
        title=title,
        card=card,
        results_dir=results_dir,
        report_out=report_out,
        methodology_note=methodology_note,
        slas=slas,
        sweep=sweep,
        specs=specs,
        external_baselines=ext_specs,
    )


def validate_card_identity(cfg: "BenchConfig") -> None:
    """Fail fast on a missing/placeholder card name or a non-positive card count.

    The card name and per-replica card count end up in curve.json, the HTML report and the
    data handed back to the requester, so they must be written explicitly and correctly.
    """
    names = {cfg.card} | {sp.card for sp in cfg.specs}
    for n in names:
        if not str(n).strip() or str(n).strip().upper() == "GPU":
            raise ValueError(
                f"card name {n!r} is a placeholder: set `card:` in the config (or --card) to the real card model."
            )
    for sp in cfg.specs:
        if sp.tp < 1 or not (sp.cards > 0):
            raise ValueError(f"{sp.model}: tp must be >= 1 and cards > 0 (tp={sp.tp} cards={sp.cards})")
