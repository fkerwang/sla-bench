# sla-bench

**English** | [简体中文](README_CN.md)

LLM serving SLA throughput and accuracy benchmark suite (reference card: **H20**). Bundled with pre-measured H20 FP8 baselines (`results/h20_fp8/`) so you do not need to re-run H20.

## 1. Installation & Server Requirements

```bash
pip install -e .            # Python >= 3.9
python3 -m sla_bench --help
```

Target server requirements:
- Expose vLLM / SGLang-compatible `/v1/chat/completions`, `/tokenize`, and `/detokenize` endpoints (the client does not depend on local `transformers`; all token counting is performed server-side).
- **Prefix caching must be disabled** (SGLang: `--disable-radix-cache`; vLLM: `--no-enable-prefix-caching`).
- For hybrid linear-attention models (e.g., `Qwen3.5-35B-A3B`) on SGLang, set `--max-running-requests` above peak concurrency (recommended: `256`) to avoid Mamba state-slot queueing.
- Default health check endpoint is `/health_generate` (for vLLM, pass `--health-path /health` or `--no-health-check`).

## 2. Configuration & One-Command Run

Copy the vendor template `configs/vendor_card.yaml` pre-wired with the H20 FP8 baseline (for benchmarking H20 itself, see `configs/example_card.yaml`):

```bash
cp configs/vendor_card.yaml my_card.yaml
```

Update three fields for your hardware:
- `card`: Exact model name of the card under test (required; never leave `"GPU"`).
- `models[].tp` & `models[].cards`: **Number of physical cards occupied by a single replica** (`QPM/card = achieved_arrival_rate × 60 ÷ cards`; never split by die — e.g., 1 TP=2 replica on a dual-die single physical card uses `cards: 1`, 1 TP=2 replica across 2 single-die cards uses `cards: 2`, half a card uses `0.5`).
- `models[].ports` (or `urls`): Port list of independent replicas (defaults to **FP8**, with `h20_fp8` pre-mounted under `external_baselines`).

Run the end-to-end pipeline in one command (automatically executes: 2-round replica-rotated coarse sweep → adaptive SLA①/② boundary refinement → fixed 300-item `acc` pass → standalone HTML comparison report → `light` JSONL export):

```bash
python3 -m sla_bench run -c my_card.yaml
```

Step-by-step subcommands are also available: `dataset-info`, `sweep`, `refine [--dry-run]`, `acc`, `export`, and `report` (see `--help` for options).

## 3. Fixed Evaluation Protocol (Do Not Modify — Required for H20 Parity)

| Item | Value | Description |
| --- | --- | --- |
| Dataset | `dataset/nocache_sla_in3669_v3.ndjson.gz` | Bundled 1,000 long-context QA prompts (~3,755 input tokens on average) |
| Output Length | `max_tokens: 1000` | `ignore_eos` + `===ANSWER_END===` pads every response to 1,000 decoded tokens; `acc` and `real-out` evaluate only the answer before the delimiter |
| Window & Gates | `warmup_s: 60`, `min_s: 180` | First 60s excluded; open-loop Poisson steady window ≥ 180s; requires queue stability (drift ≤ 3 SE), zero errors, and input/output mean length within ±3% (`len_ok`) |
| Accuracy Slice | `acc_prompts: 300`, `acc_seed: 0` | Fixed 128 warmup prompts + identical 300 scored prompts across all cards |
| SLA Tiers | ① `2005 / 51 ms`; ② `1000 / 20 ms` | ① mean TTFT ≤ 2005 ms & TPOT ≤ 51 ms; ② mean TTFT ≤ 1000 ms & TPOT ≤ 20 ms (refined until bottleneck `gap < 10%`) |

## 4. Bundled H20 Reference Baseline (4×H20, 2 TP=2 Replicas, 2 Cards per Replica)

| Baseline Dir | Precision | Model | SLA① QPM/card | SLA② QPM/card | Fixed-300 `acc` |
| --- | --- | --- | --- | --- | --- |
| `results/h20_fp8` | FP8 | `Qwen3.5-35B-A3B` | **102.9** (102.9–104.4) | **69.2** (66.7–73.7) | **58.3%** |
| `results/h20_fp8` | FP8 | `Qwen3.8-27B` | **28.6** (28.1–31.8) | **12.9** (12.9–15.9) | **65.7%** |

## 5. Deliverables to Return

Return the **entire `<results_dir>` directory** in full (raw prompt and generated response text are omitted automatically by the client and do not need to be returned):
1. Standalone HTML report (`report_out` file, e.g. `<results_dir>/sla_report.html`).
2. All `<results_dir>/round*` per-point & per-request metric files (`curve.json` and text-free `pre_detail_stat_*.jsonl`).
3. `<results_dir>/light/<card>/*_burst.jsonl` and `<results_dir>/acc/<model>/` (`acc_summary.json`, `item_scores.json`).

## 6. Self-Check Unit Tests

```bash
python3 -m unittest discover -s tests
```
