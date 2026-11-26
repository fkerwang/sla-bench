"""Unified CLI for `sla-bench` (`python3 -m sla_bench <subcommand>`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import (
    BenchConfig,
    ModelTPSpec,
    SLALevel,
    SweepSettings,
    default_slas,
    load_bench_config,
    normalize_urls,
    parse_sla_arg,
    strip_tp_suffix,
    validate_card_identity,
)
from .dataset import compute_or_load_tokenizer_lens, ensure_dataset
from .light import export_light
from .report import generate_html_report
from .sweep import run_fixed_acc_pass, run_refine_for_spec, run_sweep_for_spec

TOP_EPILOG = """Examples:
  # 1. One-command complete benchmark (sweep + gap<10% refine + HTML report) from YAML:
  python3 -m sla_bench run -c configs/example_card.yaml

  # 2. Direct CLI sweep + auto gap<10% refinement on any new GPU card (no YAML required):
  python3 -m sla_bench sweep \\
    --card H20 --tp 2 --cards 2 \\
    --ports 30100,30101 \\
    --model Qwen3.5-35B-A3B \\
    --rates 1.8,2.4,2.8,3.05,3.2 \\
    --results-dir results/h20_v7 --report-out results/h20_v7/report.html

  # 3. Inspect or execute adaptive SLA①/② gap<10% refinement on existing round* folders:
  python3 -m sla_bench refine --results-dir results/h20_v7 --card H20 --dry-run

  # 4. Generate self-contained inline-SVG HTML report from completed results:
  python3 -m sla_bench report --results-dir results/h20_v7 --card H20 -o report.html
"""


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=str, default="", help="Optional YAML config file (see configs/example_card.yaml)")
    p.add_argument("--card", type=str, default="", help="GPU card model of the machine under test (e.g. your card; H20 is the benchmark card)")
    p.add_argument("--results-dir", type=str, default="", help="Directory holding round0/, round1/, round*_ref*/")
    p.add_argument("--sla1", type=str, default="", help="SLA① TTFT_MS:TPOT_MS (default 2005:51)")
    p.add_argument("--sla2", type=str, default="", help="SLA② TTFT_MS:TPOT_MS (default 1000:20)")
    p.add_argument("--stab-z", type=float, default=None, help="v7 queue stability SE multiplier (default 3.0)")
    p.add_argument("--gap-threshold", type=float, default=None, help="Adaptive SLA boundary max gap ratio (default 0.10 = 10%%)")
    p.add_argument("--min-bracket-rel", type=float, default=None, help="Stop refining if (fail-best)/best <= this step cliff (default 0.025 = 2.5%%)")
    p.add_argument("--min-completed", type=int, default=None, help="Minimum total judged-window completed requests across rounds (default 0 = use --min-per-round)")


def _add_sweep_spec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=str, default="", help="Model config directory tag (e.g. Qwen3.5-35B-A3B or Qwen3.8-27B-tp2)")
    p.add_argument("--served-model-name", type=str, default="", help="OpenAI served model name (defaults to --model without -tpN)")
    p.add_argument("--tp", type=int, default=None, help="Tensor parallelism degree per replica")
    p.add_argument("--cards", type=float, default=None, help="Physical cards occupied per replica (e.g. 2, 1, or 0.5/0.33/0.25 when only part of a card is used)")
    p.add_argument("--ports", type=str, default="", help="Comma-separated replica ports on --host (e.g. 19030,19031)")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Host for --ports (default 127.0.0.1)")
    p.add_argument("--urls", type=str, default="", help="Comma-separated full /v1/chat/completions replica URLs")
    p.add_argument("--tokenizer", type=str, default="", help="DEPRECATED and ignored: tokenization is server-side (/tokenize); no local tokenizer is loaded")
    p.add_argument("--tokenizer-url", type=str, default="", help="Optional /tokenize URL override (default: derived from each replica URL)")
    p.add_argument("--ds-in", type=float, default=None, help="Optional explicit client-unit prompt length target override")
    p.add_argument("--rates", type=str, default="", help="Comma-separated initial arrival rates per replica in req/s (e.g. 0.2,0.4,0.6,0.8)")
    p.add_argument("--rounds", type=int, default=None, help="Number of replica-rotated rounds per rate (default 2)")
    p.add_argument("--warmup-s", type=float, default=None, help="Per-point warm-up seconds excluded before block window (default 30)")
    p.add_argument("--min-s", type=float, default=None, help="Minimum post-warmup steady sending seconds per point (default 180)")
    p.add_argument("--max-s", type=float, default=None, help="Safety wall-clock cap seconds per point (default 2400)")
    p.add_argument("--max-tokens", type=int, default=None, help="Forced padded output token length (default 1000)")
    p.add_argument("--len-tol", type=float, default=None, help="Prompt & output mean length tolerance (default 0.03 = +-3%%)")
    p.add_argument("--min-per-round", type=int, default=None, help="Minimum judged-window requests per round (default 32)")
    p.add_argument("--max-refine-steps", type=int, default=None, help="Maximum adaptive refinement iterations (default 6)")
    p.add_argument("--dataset", type=str, default="", help="Custom NDJSON(.gz) dataset path (defaults to bundled nocache_sla_in3669_v3.ndjson.gz)")
    p.add_argument("--cache-dir", type=str, default="", help="Working cache directory for decompressed dataset & tokenizer lens")
    p.add_argument("--locust-bin", type=str, default="", help="Explicit path to locust binary")
    p.add_argument("--no-health-check", action="store_true", help="Skip pre-batch /health_generate check")
    p.add_argument("--health-path", type=str, default="", help="Replica health endpoint path (default /health_generate)")
    p.add_argument("--dry-run", action="store_true", help="Print planned batches / refinement rates without launching Locust")


def _build_effective_config(args: argparse.Namespace) -> BenchConfig:
    cfg = load_bench_config(args.config) if getattr(args, "config", "") else BenchConfig()

    if getattr(args, "card", ""):
        cfg.card = args.card
        for sp in cfg.specs:
            if sp.card == "GPU":
                sp.card = args.card
    if getattr(args, "results_dir", ""):
        cfg.results_dir = args.results_dir
    if getattr(args, "sla1", "") or getattr(args, "sla2", ""):
        base_slas = cfg.slas or default_slas()
        s1 = parse_sla_arg(args.sla1, "①", "#DC2626") if getattr(args, "sla1", "") else base_slas[0]
        s2 = parse_sla_arg(args.sla2, "②", "#EA580C") if getattr(args, "sla2", "") else base_slas[1]
        cfg.slas = [s1, s2]

    sw = cfg.sweep
    if getattr(args, "stab_z", None) is not None:
        sw.stab_z = float(args.stab_z)
    if getattr(args, "gap_threshold", None) is not None:
        sw.gap_threshold = float(args.gap_threshold)
    if getattr(args, "min_bracket_rel", None) is not None:
        sw.min_bracket_rel = float(args.min_bracket_rel)
    if getattr(args, "min_completed", None) is not None:
        sw.min_completed = int(args.min_completed)
    if getattr(args, "rounds", None) is not None:
        sw.rounds = int(args.rounds)
    if getattr(args, "warmup_s", None) is not None:
        sw.warmup_s = float(args.warmup_s)
    if getattr(args, "min_s", None) is not None:
        sw.min_s = float(args.min_s)
    if getattr(args, "max_s", None) is not None:
        sw.max_s = float(args.max_s)
    if getattr(args, "max_tokens", None) is not None:
        sw.max_tokens = int(args.max_tokens)
    if getattr(args, "len_tol", None) is not None:
        sw.len_tol = float(args.len_tol)
    if getattr(args, "min_per_round", None) is not None:
        sw.min_per_round = int(args.min_per_round)
    if getattr(args, "max_refine_steps", None) is not None:
        sw.max_refine_steps = int(args.max_refine_steps)
    if getattr(args, "dataset", ""):
        sw.dataset_path = args.dataset
    if getattr(args, "cache_dir", ""):
        sw.cache_dir = args.cache_dir
    if getattr(args, "locust_bin", ""):
        sw.locust_bin = args.locust_bin
    if getattr(args, "no_health_check", False):
        sw.health_check = False
    if getattr(args, "health_path", ""):
        sw.health_path = args.health_path
    if getattr(args, "prompts", None) is not None:
        sw.acc_prompts = int(args.prompts)
    if getattr(args, "acc_seed", None) is not None:
        sw.acc_seed = int(args.acc_seed)

    # Build or filter CLI-specified single ModelTPSpec when --model is provided
    cli_model = getattr(args, "model", "")
    ports = [int(x) for x in args.ports.split(",") if x.strip()] if getattr(args, "ports", "") else []
    urls_raw = [x.strip() for x in args.urls.split(",") if x.strip()] if getattr(args, "urls", "") else []
    cli_urls = normalize_urls(ports=ports, urls=urls_raw, host=getattr(args, "host", "127.0.0.1"))
    cli_rates = [float(x) for x in args.rates.split(",") if x.strip()] if getattr(args, "rates", "") else []

    created_new = False
    if cli_model:
        matched = [sp for sp in cfg.specs if sp.model == cli_model or sp.key == cli_model]
        if matched:
            cfg.specs = matched
            for sp in cfg.specs:
                if cli_urls:
                    sp.urls = cli_urls
                if cli_rates:
                    sp.rates = cli_rates
                if getattr(args, "tokenizer_url", ""):
                    sp.tokenizer_url = args.tokenizer_url
                if getattr(args, "served_model_name", ""):
                    sp.served_model_name = args.served_model_name
                if getattr(args, "tp", None) is not None:
                    sp.tp = int(args.tp)
                if getattr(args, "cards", None) is not None:
                    sp.cards = float(args.cards)
                if getattr(args, "ds_in", None) is not None:
                    sp.ds_in = float(args.ds_in)
        else:
            created_new = True
            tp = int(args.tp) if getattr(args, "tp", None) is not None else 1
            cards = float(args.cards) if getattr(args, "cards", None) is not None else 0.0
            served = getattr(args, "served_model_name", "") or strip_tp_suffix(cli_model)
            cfg.specs = [
                ModelTPSpec(
                    key=f"{cli_model}_tp{tp}",
                    card=cfg.card,
                    model=cli_model,
                    served_model_name=served,
                    model_short=served,
                    tp=tp,
                    cards=cards,
                    tokenizer_url=getattr(args, "tokenizer_url", ""),
                    ds_in=getattr(args, "ds_in", None),
                    urls=cli_urls,
                    rates=cli_rates,
                )
            ]
    elif cli_urls:
        for sp in cfg.specs:
            sp.urls = cli_urls

    if not getattr(args, "config", "") and not getattr(args, "card", ""):
        raise SystemExit("Error: without -c config.yaml you must pass --card <real card name>.")
    if created_new and getattr(args, "cards", None) is None:
        raise SystemExit("Error: a new --model spec needs an explicit --cards (physical cards occupied per replica).")
    try:
        validate_card_identity(cfg)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}")
    for sp in cfg.specs:
        print(
            f"[config] card={sp.card} | {sp.model}: tp={sp.tp} cards={sp.card_num:g} per replica",
            flush=True,
        )

    if not cfg.title:
        cfg.title = f"{cfg.card} · 不同 TTFT / TPOT 门限下的 QPM/card"
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sla-bench",
        description=(
            "Audited v7 LLM SLA benchmark suite: open-loop absolute-timetable Poisson Locust client, "
            "serpentine octile length-stratified prompt sampler (+-3% ds_in & output 1000 gate), "
            "2-level SLA gap<10% adaptive sweep/refiner, and inline-SVG HTML auto-plotter."
        ),
        epilog=TOP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. sweep
    p_sweep = sub.add_parser(
        "sweep",
        help="Run coarse arrival-rate sweep across rotated replica rounds + adaptive gap<10%% SLA refinement",
        description=(
            "Run a v7 multi-replica parallel arrival-rate sweep across rotated rounds (default 2 rounds), "
            "then automatically refine SLA① & SLA② brackets until gap < 10% (unless --no-auto-refine is passed)."
        ),
    )
    _add_common_args(p_sweep)
    _add_sweep_spec_args(p_sweep)
    p_sweep.add_argument("--no-auto-refine", action="store_true", help="Stop after coarse grid without running adaptive gap<10%% refinement")
    p_sweep.add_argument("--report-out", type=str, default="", help="Automatically generate HTML report to this path after sweep")

    # 2. refine
    p_refine = sub.add_parser(
        "refine",
        help="Inspect or run adaptive 2-level SLA gap<10%% boundary refinement on existing round* folders",
        description=(
            "Load existing round* curve data on disk, evaluate SLA① & SLA② [best_pass_rate, first_fail_rate] "
            "brackets (and step upward if top rate still passes SLA), and run rotated round{r}_ref{step} "
            "points until bottleneck gap < 10% (or bracket width <= 2.5% step cliff)."
        ),
    )
    _add_common_args(p_refine)
    _add_sweep_spec_args(p_refine)
    p_refine.add_argument("--report-out", type=str, default="", help="Regenerate HTML report after refinement")

    # 3. report
    p_report = sub.add_parser(
        "report",
        help="Generate self-contained inline-SVG HTML SLA report from completed round* results",
        description=(
            "Generate a standalone inline-SVG HTML report featuring: (1) 2-level SLA ceiling table by (Card, Model) "
            "with bottleneck gap%, (2) cross-platform table with bold optimal TP/card/@req/s & overall accuracy, "
            "(3) per-model broken-axis (TTFT, TPOT) operating-point scatter plots above curves, (4) 2 latency-vs-QPM/card "
            "curves per model, (5) v7 methodology + per-board TPOT offset table, and (6) merged per-point detail sheet."
        ),
    )
    _add_common_args(p_report)
    p_report.add_argument("-o", "--output", type=str, default="", help="Output HTML file path (overrides report_out in YAML)")
    p_report.add_argument("--title", type=str, default="", help="Custom HTML report title")

    # 4. run (all-in-one pipeline)
    p_run = sub.add_parser(
        "run",
        help="Run complete end-to-end benchmark (sweep + gap<10%% refine + HTML report) in one command",
        description="Execute coarse sweep, adaptive gap<10% SLA①/② refinement, and inline-SVG HTML report generation in one command.",
    )
    _add_common_args(p_run)
    _add_sweep_spec_args(p_run)
    p_run.add_argument("-o", "--report-out", type=str, default="", help="Output HTML report path")

    # 4b. export (light per-point jsonl)
    p_exp = sub.add_parser(
        "export",
        help="Write the ultra-light BurstGPT-style *_burst.jsonl (deliverable / baseline format) from raw results",
        description="Summarize raw results into <out-dir>/<card>/<model>_..._burst.jsonl: one row per rate with TTFT/TPOT, acc, input / real-output length stats.",
    )
    _add_common_args(p_exp)
    p_exp.add_argument("--out-dir", type=str, default="", help="Output directory (default <results_dir>/light)")

    # 4c. acc (fixed 200-item closed-loop accuracy pass)
    p_acc = sub.add_parser(
        "acc",
        help="Fixed-slice accuracy pass: same N prompts on every card, closed-loop saturated",
        description=(
            "Send a fixed slice of the dataset (same items and order for every card) once each, "
            "closed-loop with no pacing so the server runs at its own maximum throughput, and "
            "record per-request accuracy into <results_dir>/acc/<model>/. Unlike the per-point "
            "accuracy inside a sweep, this number is comparable across cards and re-runs."
        ),
    )
    _add_common_args(p_acc)
    _add_sweep_spec_args(p_acc)
    p_acc.add_argument("--prompts", type=int, default=None, help="Prompts in the fixed slice (default sweep.acc_prompts=200)")
    p_acc.add_argument("--seed", type=int, default=None, help="Slice seed; identical seed = identical questions (default sweep.acc_seed=0)")

    # 5. dataset-info
    p_ds = sub.add_parser(
        "dataset-info",
        help="Decompress bundled nocache_sla_in3669_v3 dataset and compute/cache client-unit ds_in via a server /tokenize",
        description="Verify bundled dataset decompression and compute/cache per-prompt lengths + client-unit ds_in (with PAD_INSTRUCTION).",
    )
    p_ds.add_argument("--dataset", type=str, default="", help="Optional dataset path (defaults to bundled nocache_sla_in3669_v3.ndjson.gz)")
    p_ds.add_argument("--tokenizer-url", type=str, default="", help="Server URL (/tokenize or any chat URL on it) used to count tokens & cache lens + ds_in")
    p_ds.add_argument("--model", type=str, default="", help="Served model name passed to /tokenize")
    p_ds.add_argument("--pad-to-tokens", type=int, default=1000, help="Output pad target (default 1000; 0 disables PAD_INSTRUCTION)")
    p_ds.add_argument("--cache-dir", type=str, default="", help="Optional cache directory")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset-info":
        ds_file = ensure_dataset(args.dataset or None, cache_dir=args.cache_dir or None)
        lens, ds_in = compute_or_load_tokenizer_lens(
            dataset_path=ds_file,
            tokenizer_url=args.tokenizer_url or None,
            model=args.model,
            pad_to_tokens=args.pad_to_tokens,
            cache_dir=args.cache_dir or None,
        )
        s_lens = sorted(lens)
        print(
            f"dataset={ds_file}\ncount={len(lens)}  ds_in={ds_in:.2f}  "
            f"min={s_lens[0]}  p50={s_lens[len(s_lens) // 2]}  max={s_lens[-1]}"
        )
        return 0

    cfg = _build_effective_config(args)

    if args.command == "sweep":
        if not cfg.specs:
            raise SystemExit("Error: provide -c config.yaml or --model/--ports/--rates to run `sweep`.")
        for sp in cfg.specs:
            run_sweep_for_spec(cfg, sp, auto_refine=not args.no_auto_refine, dry_run=args.dry_run)
        if args.report_out and not args.dry_run:
            out_p = generate_html_report(cfg, out_path=args.report_out)
            print(f"[report] wrote {out_p}", flush=True)
        return 0

    if args.command == "refine":
        if not cfg.specs:
            from .report import auto_discover_specs_from_dir

            cfg.specs = auto_discover_specs_from_dir(cfg.results_dir, card=cfg.card)
        if not cfg.specs:
            raise SystemExit(f"Error: no specs configured or discovered in {cfg.results_dir}")
        for sp in cfg.specs:
            run_refine_for_spec(cfg, sp, dry_run=args.dry_run)
        if args.report_out and not args.dry_run:
            out_p = generate_html_report(cfg, out_path=args.report_out)
            print(f"[report] wrote {out_p}", flush=True)
        return 0

    if args.command == "acc":
        if not cfg.specs:
            raise SystemExit("Error: provide -c config.yaml or --model/--ports to run `acc`.")
        rc = 0
        for sp in cfg.specs:
            if getattr(sp, "urls", None) is None or not sp.urls:
                spec_urls = []
            else:
                spec_urls = sp.urls
            if not spec_urls:
                raise SystemExit(f"Error: {sp.model} has no replica URL/ports; pass --ports or --urls.")
            run_fixed_acc_pass(cfg, sp, dry_run=args.dry_run)
        return rc

    if args.command == "export":
        paths = export_light(cfg, out_dir=args.out_dir or None)
        if not paths:
            raise SystemExit(f"Error: no measured points found under {cfg.results_dir}")
        for pth in paths:
            print(f"[export] wrote {pth} ({pth.stat().st_size} bytes)", flush=True)
        return 0

    if args.command == "report":
        if args.title:
            cfg.title = args.title
        out_p = generate_html_report(cfg, out_path=args.output or None)
        print(f"[report] wrote {out_p} ({out_p.stat().st_size} bytes)", flush=True)
        return 0

    if args.command == "run":
        if not cfg.specs:
            raise SystemExit("Error: provide -c config.yaml or --model/--ports/--rates to run `run`.")
        for sp in cfg.specs:
            run_sweep_for_spec(cfg, sp, auto_refine=True, dry_run=args.dry_run)
            run_fixed_acc_pass(cfg, sp, dry_run=args.dry_run)
        if not args.dry_run:
            out_p = generate_html_report(cfg, out_path=args.report_out or None)
            print(f"[report] wrote {out_p} ({out_p.stat().st_size} bytes)", flush=True)
            for pth in export_light(cfg):
                print(f"[export] wrote {pth} ({pth.stat().st_size} bytes)", flush=True)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
