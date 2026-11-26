"""Comprehensive unit & integration tests for `sla-bench`."""

from __future__ import annotations

import os
import statistics as st
import subprocess
import sys
import tempfile
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sla_bench.cli import build_parser, main
from sla_bench.config import BenchConfig, load_bench_config
from sla_bench.dataset import (
    balanced_file,
    compute_or_load_tokenizer_lens,
    ensure_dataset,
)
from sla_bench.report import generate_html_report
from sla_bench.stats import (
    by_rate,
    ceiling_rates,
    len_ok,
    point_size,
    stability,
)
from sla_bench.sweep import (
    build_rotated_batches,
    plan_refinement_step,
)


class _StubTokenizeServer:
    """Minimal vLLM-like /tokenize + /detokenize stub: one token per 4 characters."""

    def __enter__(self):
        import http.server
        import json as _json
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    text = body.get("prompt")
                    if text is None:
                        text = " ".join(str(m.get("content") or "") for m in body["messages"])
                    toks = list(range(max(1, len(text) // 4)))
                    out = {"count": len(toks), "tokens": toks}
                else:
                    out = {"prompt": " ".join("w%d" % t for t in body["tokens"])}
                raw = _json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return "http://127.0.0.1:%d" % self.srv.server_address[1]

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


class TestSlaBench(unittest.TestCase):

    def test_dataset_decompression_and_serpentine_octile_balance(self):
        with tempfile.TemporaryDirectory(prefix="sla_test_ds_") as tmp:
            ds_path = ensure_dataset(cache_dir=tmp)
            self.assertTrue(ds_path.is_file())
            with open(ds_path, encoding="utf-8-sig") as fh:
                n_lines = sum(1 for l in fh if l.strip())
            self.assertEqual(n_lines, 1000)

            # Server-side counting against a stub /tokenize (whitespace "tokens"); no local tokenizer.
            with _StubTokenizeServer() as base:
                lens, ds_in = compute_or_load_tokenizer_lens(
                    ds_path, tokenizer_url=base + "/v1/chat/completions", model="stub", pad_to_tokens=1000, cache_dir=tmp
                )
                self.assertEqual(len(lens), 1000)
                self.assertGreater(ds_in, 100)
            # Cache hit works even after the server is gone
            lens_cached, ds_in_cached = compute_or_load_tokenizer_lens(
                ds_path, tokenizer_url=base + "/tokenize", model="stub", pad_to_tokens=1000, cache_dir=tmp
            )
            self.assertEqual(lens, lens_cached)
            self.assertAlmostEqual(ds_in, ds_in_cached, places=6)
            # No server (dry-run): model-agnostic estimate
            lens, ds_in = compute_or_load_tokenizer_lens(ds_path, tokenizer_url=None, pad_to_tokens=1000, cache_dir=tmp)
            self.assertEqual(len(lens), 1000)

            for seed in (100, 117, 205):
                for n_req in (40, 64, 128, 320):
                    bal_file, ordered_lens = balanced_file(
                        dataset_path=ds_path,
                        lens=lens,
                        seed=seed,
                        tag="unit",
                        n=n_req,
                        out_dir=tmp,
                    )
                    self.assertTrue(bal_file.is_file())
                    self.assertEqual(len(ordered_lens), n_req)
                    # Full sample mean within +-3%
                    self.assertLessEqual(abs(st.mean(ordered_lens) / ds_in - 1.0), 0.03)
                    # Judged window after dropping warmup blocks (e.g. first 8 or 16 prompts) within +-3%
                    win_lens = ordered_lens[8:]
                    self.assertLessEqual(abs(st.mean(win_lens) / ds_in - 1.0), 0.03)

    def test_point_size_and_len_ok(self):
        for rate in (0.01, 0.05, 0.25, 0.80, 2.50):
            n2 = point_size(rate, warmup_s=90, min_s=180, min_completed=50, rounds=2, min_per_round=32)
            self.assertEqual(n2 % 8, 0)
            # Across 2 rounds, post-warmup window has >= 2 * 32 = 64 >= 50 requests
            est_win_per_round = n2 - rate * 90
            self.assertGreaterEqual(est_win_per_round, 32.0 - 1e-9)
            self.assertGreaterEqual(2 * est_win_per_round, 50.0)

            n1 = point_size(rate, warmup_s=90, min_s=180, min_completed=50, rounds=1, min_per_round=32)
            self.assertEqual(n1 % 8, 0)
            self.assertGreaterEqual(n1 - (rate * 90 + 8), 50.0 - 1e-9)

        good_m = {"in_len": 3740.0, "in_len_all": 3710.0, "out_len": 1000.0, "ds_in": 3727.0}
        self.assertTrue(len_ok(good_m, ds_in=3727.0, max_tokens=1000, len_tol=0.03))
        bad_all = {"in_len": 3727.0, "in_len_all": 3950.0, "out_len": 1000.0, "ds_in": 3727.0}
        self.assertFalse(len_ok(bad_all, ds_in=3727.0, max_tokens=1000, len_tol=0.03))
        bad_out = {"in_len": 3727.0, "in_len_all": 3727.0, "out_len": 850.0, "ds_in": 3727.0}
        self.assertFalse(len_ok(bad_out, ds_in=3727.0, max_tokens=1000, len_tol=0.03))

    def test_v7_stability_detector(self):
        # 1. Stationary point where 2nd half happens to draw longer prompts -> must stay STABLE after length regression!
        keep_len_shift = []
        for idx in range(64):
            plen = 2500 if idx < 32 else 5000
            ttft = 300.0 + 0.35 * plen + (12.0 if idx % 2 == 0 else -12.0)
            keep_len_shift.append({"sample_index": idx, "prompt_len": plen, "ttft": ttft, "tpot": 28.0, "output_len": 1000})
        res_shift = stability(keep_len_shift, i0=0, stab_z=3.0)
        self.assertTrue(res_shift["stable"])

        # 2. Isolated 3-request Poisson burst raising 2nd-half mean, but median stays flat -> STABLE!
        keep_burst = []
        for idx in range(64):
            ttft = 700.0 + (8.0 if idx % 2 == 0 else -8.0)
            if idx in (50, 51, 52):
                ttft = 9000.0
            keep_burst.append({"sample_index": idx, "prompt_len": 3700, "ttft": ttft, "tpot": 30.0, "output_len": 1000})
        res_burst = stability(keep_burst, i0=0, stab_z=3.0)
        self.assertTrue(res_burst["stable"])

        # 3. Genuine growing queue across whole 2nd half -> UNSTABLE!
        keep_overload = []
        for idx in range(64):
            ttft = 800.0 + (idx * 60.0 if idx >= 32 else (idx * 2.0))
            keep_overload.append({"sample_index": idx, "prompt_len": 3700, "ttft": ttft, "tpot": 35.0, "output_len": 1000})
        res_over = stability(keep_overload, i0=0, stab_z=3.0)
        self.assertFalse(res_over["stable"])

    def test_adaptive_gap_refinement_planner_and_rotation(self):
        cfg = BenchConfig()
        slas = cfg.slas

        # Verify replica rotation assigns distinct replicas across round 0 and round 1
        urls = [f"http://127.0.0.1:{19030 + i}/v1/chat/completions" for i in range(6)]
        rates = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        b0 = dict((r, u) for u, r in build_rotated_batches(urls, rates, round_idx=0)[0])
        b1 = dict((r, u) for u, r in build_rotated_batches(urls, rates, round_idx=1)[0])
        for r in rates:
            self.assertNotEqual(b0[r], b1[r])

        def synth_point(rate: float) -> list[dict]:
            # Synthetic monotonic model: SLA② hits TPOT=20ms at rate=0.40; SLA① hits TPOT=51ms at rate=0.82
            ttft = 550.0 + 450.0 * rate
            tpot = 10.0 + 25.0 * rate if rate <= 0.40 else (20.0 + (rate - 0.40) * (31.0 / 0.42))
            stable = rate <= 0.95
            return [
                {"rate": rate, "ttft": ttft, "tpot": tpot, "n_kept": 40, "n": 64, "stable": stable, "errors": 0, "missing": 0, "len_ok": True},
                {"rate": rate, "ttft": ttft + 10, "tpot": tpot + 0.2, "n_kept": 40, "n": 64, "stable": stable, "errors": 0, "missing": 0, "len_ok": True},
            ]

        # A. Unbounded top grid -> must step upward first
        pts_low = [m for r in (0.15, 0.30, 0.50) for m in synth_point(r)]
        plan_up = plan_refinement_step(by_rate(pts_low), card_num=1.0, slas=slas, num_replicas=4)
        self.assertFalse(plan_up.converged)
        self.assertTrue(all(r > 0.50 for r in plan_up.proposed_rates))

        # B. Coarse bracketed grid with large gaps -> iteratively converges to gap < 10% on BOTH SLA① and SLA②!
        pts = [m for r in (0.20, 0.50, 0.60, 0.95) for m in synth_point(r)]
        for _step in range(5):
            plan = plan_refinement_step(by_rate(pts), card_num=1.0, slas=slas, num_replicas=6, gap_threshold=0.10)
            if plan.converged:
                break
            self.assertTrue(len(plan.proposed_rates) > 0)
            for nr in plan.proposed_rates:
                pts.extend(synth_point(nr))
        final_plan = plan_refinement_step(by_rate(pts), card_num=1.0, slas=slas, num_replicas=6, gap_threshold=0.10)
        self.assertTrue(final_plan.converged)
        for st_item in final_plan.statuses:
            self.assertIsNotNone(st_item.gap)
            self.assertLess(st_item.gap, 0.10)

        # C. Step cliff (bracket relative width <= 2.5% even though gap >= 10%) -> converges as cliff!
        cliff_pts = [
            {"rate": 0.500, "ttft": 700.0, "tpot": 32.0, "n_kept": 64, "n": 96, "stable": True, "errors": 0, "missing": 0, "len_ok": True},
            {"rate": 0.500, "ttft": 710.0, "tpot": 32.2, "n_kept": 64, "n": 96, "stable": True, "errors": 0, "missing": 0, "len_ok": True},
            {"rate": 0.510, "ttft": 2500.0, "tpot": 85.0, "n_kept": 64, "n": 96, "stable": False, "errors": 0, "missing": 0, "len_ok": True},
            {"rate": 0.510, "ttft": 2510.0, "tpot": 85.2, "n_kept": 64, "n": 96, "stable": False, "errors": 0, "missing": 0, "len_ok": True},
        ]
        cliff_plan = plan_refinement_step(by_rate(cliff_pts), card_num=1.0, slas=[slas[0]], num_replicas=4, gap_threshold=0.10, min_bracket_rel=0.025)
        self.assertTrue(cliff_plan.converged)
        self.assertEqual(cliff_plan.statuses[0].state, "converged_cliff")

    def test_locust_client_accepts_arbitrary_url_and_model_and_cli_help(self):
        import contextlib
        import io

        # Verify Locust client accepts arbitrary --url / --model_name / --file in a clean Locust process
        check_code = (
            "import sys; sys.path.insert(0, %r); "
            "from locust.argument_parser import LocustArgumentParser; "
            "from sla_bench.locust_client import client as lc; "
            "lp = LocustArgumentParser(); lc.add_custom_arguments(lp); "
            "o = lp.parse_args(['--url', 'http://arbitrary-gpu-host:45678/v1/chat/completions', "
            "'--model_name', 'Custom-NextGen-70B', '--file', '/tmp/any_path_dataset.ndjson']); "
            "assert o.url == 'http://arbitrary-gpu-host:45678/v1/chat/completions'; "
            "assert o.model_name == 'Custom-NextGen-70B'; "
            "assert o.file == '/tmp/any_path_dataset.ndjson'; "
            "print('LOCUST_CLIENT_ARGS_OK')"
        ) % str(REPO_ROOT)
        proc = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("LOCUST_CLIENT_ARGS_OK", proc.stdout)

        # Verify YAML example config loads cleanly
        ex_cfg = load_bench_config(REPO_ROOT / "configs" / "example_card.yaml")
        self.assertEqual(ex_cfg.card, "H20")
        self.assertEqual(len(ex_cfg.specs), 2)
        # TP=2 on single-die GPUs occupies 2 physical cards
        self.assertEqual([sp.card_num for sp in ex_cfg.specs], [2.0, 2.0])
        self.assertEqual(ex_cfg.external_baselines, [])  # benchmark config measures H20 itself

        # Verify CLI --help on top-level and every subcommand exits 0
        parser = build_parser()
        self.assertIsNotNone(parser)
        for subcmd in ([], ["sweep"], ["refine"], ["report"], ["run"], ["dataset-info"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(SystemExit) as cm:
                    main(subcmd + ["--help"])
            self.assertEqual(cm.exception.code, 0)
            self.assertTrue(len(buf.getvalue()) > 50)

    def test_ceiling_skips_single_unstable_repeat_not_saturation(self):
        """One noisy (unstable) repeat must not hide higher rates that pass cleanly."""

        def pt(rate, ttft, tpot, stable=True, errors=0, missing=0, len_ok_v=True, n=64):
            return {
                "rate": rate,
                "ttft": ttft,
                "tpot": tpot,
                "stable": stable,
                "errors": errors,
                "missing": missing,
                "len_ok": len_ok_v,
                "n_kept": n,
                "n": n,
            }

        groups = by_rate([
            pt(0.60, 1200, 28),
            pt(0.60, 1150, 26),
            pt(0.65, 1390, 34, stable=False),   # noisy round only
            pt(0.65, 1450, 36),
            pt(0.70, 1336, 31),
            pt(0.70, 1750, 37),
            pt(0.72, 1411, 51.3),               # TPOT over SLA① -> first fail
            pt(0.72, 1811, 77.8),
        ])
        best, fail = ceiling_rates(groups, 2005.0, 51.0)
        self.assertAlmostEqual(best, 0.70)
        self.assertAlmostEqual(fail, 0.72)

        # A rate with real errors/missing is genuine failure, not noise: it ends the scan.
        groups_err = by_rate([
            pt(0.60, 1200, 28),
            pt(0.65, 1300, 30, errors=2),
            pt(0.70, 1336, 31),
        ])
        best_e, fail_e = ceiling_rates(groups_err, 2005.0, 51.0)
        self.assertAlmostEqual(best_e, 0.60)
        self.assertAlmostEqual(fail_e, 0.65)

    def test_readonly_report_on_h20_baselines(self):
        import contextlib
        import io

        # H20 FP8 baseline data committed in results/h20_fp8
        for sub_name, exp_acc in (("h20_fp8", "<b>58.3%</b>"),):
            h20_dir = REPO_ROOT / "results" / sub_name
            self.assertTrue(h20_dir.is_dir(), f"Missing bundled baseline {h20_dir}")

            out_html = Path(f"/tmp/test_sla_bench_report_{sub_name}.html")
            if out_html.exists():
                out_html.unlink()

            with contextlib.redirect_stdout(io.StringIO()):
                rc = main([
                    "report",
                    "--results-dir",
                    str(h20_dir),
                    "--card",
                    "H20",
                    "-o",
                    str(out_html),
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_html.is_file())
            content = out_html.read_text(encoding="utf-8")
            self.assertGreater(len(content), 20000)

            # Verify all required HTML report sections & ordering
            self.assertIn("两档 SLA 下的上限", content)
            self.assertIn("最短板余量 gap%", content)
            self.assertIn("跨平台对比", content)
            self.assertIn("<b>最优数据点：", content)
            self.assertIn("req/s)", content)
            self.assertIn("总体 acc", content)
            self.assertIn(exp_acc, content)
            idx_op = content.index("工作点图：横轴 TTFT、纵轴 TPOT，点上数字为 QPM/card")
            idx_curve = content.index("曲线（分模型展示，同图对比各 TP）")
            self.assertLess(idx_op, idx_curve, "Operating-point plots must appear ABOVE curves!")
            # 2 models -> 2 operating-point charts + 4 latency curves = 6 SVGs total (NO budget charts)
            self.assertEqual(content.count("<svg "), 6)
            self.assertNotIn("最大 QPM/card vs TPOT 门限", content)
            self.assertIn("逐点明细（含输入与真实有效输出长度）", content)
            self.assertIn("real-out (min/avg/max)", content)
            self.assertIn("无分隔符", content)
            self.assertIn("TP=2 · 2 card", content)

            # Also test refine --dry-run on bundled H20 baseline data (strictly read-only!)
            buf_ref = io.StringIO()
            with contextlib.redirect_stdout(buf_ref):
                rc_ref = main([
                    "refine",
                    "--results-dir",
                    str(h20_dir),
                    "--card",
                    "H20",
                    "--dry-run",
                ])
            self.assertEqual(rc_ref, 0)
            self.assertIn("SLA①", buf_ref.getvalue())

    def test_baseline_missing_root_fails_loudly(self):
        """A configured baseline that cannot be found must error, not silently vanish."""
        import contextlib
        import io
        import shutil

        src = REPO_ROOT / "results" / "h20_fp8"
        with tempfile.TemporaryDirectory(prefix="sla_test_miss_") as tmp:
            res = Path(tmp) / "results"
            res.mkdir(parents=True, exist_ok=True)
            from sla_bench.light import find_light

            light_src = find_light(src, "Qwen3.8-27B-fp8")
            self.assertIsNotNone(light_src)
            shutil.copy(light_src, res / "Qwen3.8-27B_2card_TP2_FP8_sglang_x_burst.jsonl")
            cfg_yaml = Path(tmp) / "cfg.yaml"
            cfg_yaml.write_text(
                "card: L20\nresults_dir: %s\nreport_out: %s\n"
                "models:\n  - {key: l20_d27, model: Qwen3.8-27B-fp8, served_model_name: Qwen3.8-27B, tp: 2, cards: 2, precision: FP8}\n"
                "external_baselines:\n  - {key: h20_d27, card: H20, model: Qwen3.8-27B-fp8,"
                " canonical_model: Qwen3.8-27B, tp: 2, cards: 2, precision: FP8, roots: ['no_such_baseline']}\n"
                % (res, Path(tmp) / "r.html"),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError) as cm:
                    main(["report", "-c", str(cfg_yaml)])
            self.assertIn("no_such_baseline", str(cm.exception))

    def test_h20_baseline_in_l20_report(self):
        import contextlib
        import io
        import shutil

        src = REPO_ROOT / "results" / "h20_fp8"
        with tempfile.TemporaryDirectory(prefix="sla_test_l20_") as tmp:
            res = Path(tmp) / "results"
            # Pretend one H20 model dir is the L20 run under test
            res.mkdir(parents=True, exist_ok=True)
            from sla_bench.light import find_light

            light_src = find_light(src, "Qwen3.8-27B-fp8")
            self.assertIsNotNone(light_src)
            shutil.copy(light_src, res / "Qwen3.8-27B_2card_TP2_FP8_sglang_x_burst.jsonl")
            cfg_yaml = Path(tmp) / "cfg.yaml"
            cfg_yaml.write_text(
                "card: L20\nresults_dir: %s\nreport_out: %s\n"
                "models:\n  - {key: l20_d27, model: Qwen3.8-27B-fp8, served_model_name: Qwen3.8-27B, tp: 2, cards: 2, precision: FP8}\n"
                "external_baselines:\n  - {key: h20_d27, card: H20, model: Qwen3.8-27B-fp8,"
                " canonical_model: Qwen3.8-27B, tp: 2, cards: 2, precision: FP8, roots: ['h20_fp8']}\n"
                % (res, Path(tmp) / "r.html"),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["report", "-c", str(cfg_yaml)])
            self.assertEqual(rc, 0)
            html_txt = (Path(tmp) / "r.html").read_text(encoding="utf-8")
            self.assertIn("L20", html_txt)
            self.assertIn("H20", html_txt)

    def test_export_light_roundtrip(self):
        """raw results -> `export` -> light jsonl carries acc / real-out / lengths and rebuilds the report."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory(prefix="sla_test_export_") as tmp:
            res = Path(tmp) / "res"
            mdir = res / "Qwen3.8-27B-tp2"
            points = []
            for rate, port in ((0.2, 30100), (0.3, 30101)):
                rep = mdir / ("r%07.4f" % rate) / f"p{port}" / "report"
                rep.mkdir(parents=True)
                with open(rep / "pre_detail_stat_1.jsonl", "w", encoding="utf-8") as fh:
                    for i in range(16):
                        fh.write(json.dumps({
                            "timestamp": 1000.0 + i, "e2e_lat": 9000.0, "prompt_len": 3700 + i, "output_len": 1000,
                            "ttft": 500.0, "tpot": 20.0, "error": None, "finished": True, "sample_index": i,
                            "score": 0.5, "eff_len": 300 + i if i % 4 else None,
                        }) + "\n")
                points.append({
                    "rate": rate, "n": 16, "n_kept": 8, "n_planned": 16, "missing": 0, "errors": 0, "window_from": 8,
                    "ttft": 500.0, "ttft_p90": 520.0, "tpot": 20.0, "tpot_p90": 21.0, "e2e": 9000.0,
                    "out_len": 1000.0, "in_len": 3707.0, "in_len_all": 3707.0, "drift": 0.0, "stable": True,
                    "len_ok": True, "port": port, "elapsed": 100.0, "ds_in": 3707.0,
                })
            (mdir / "curve.json").write_text(json.dumps({
                "card": "X1", "model": "Qwen3.8-27B-tp2", "tp": 2, "cards": 2.0, "replicas": 2, "points": points,
            }), encoding="utf-8")
            out = Path(tmp) / "light"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["export", "--results-dir", str(res), "--card", "X1", "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            files = list((out / "X1-bf16").glob("*_2card_TP2_*_bf16_burst.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.startswith("Qwen3.8-27B_"), files[0].name)
            lines = [json.loads(x) for x in files[0].read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 2)  # one BurstGPT row per request rate
            # keys the downstream importer requires, plus the added acc / real-out summaries
            for k in ("request_throughput", "mean_ttft_ms", "mean_tpot_ms", "completed", "total_input_tokens", "stable"):
                self.assertIn(k, lines[0])
            self.assertAlmostEqual(lines[0]["acc"], 50.0)
            self.assertEqual(lines[0]["real_out_missing"], 2)  # i in {8,12} inside the window carry eff_len=None
            self.assertEqual((lines[0]["real_out_min"], lines[0]["real_out_max"]), (309, 315))
            self.assertEqual((lines[0]["card"], lines[0]["cards"]), ("X1", 2.0))
            html_out = Path(tmp) / "r.html"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["report", "--results-dir", str(out), "--card", "X1", "-o", str(html_out)])
            self.assertEqual(rc, 0)
            self.assertIn("real-out (min/avg/max)", html_out.read_text(encoding="utf-8"))

    def test_baseline_root_resolves_from_repo_results(self):
        """`roots: ["h20_fp8"]` must resolve to <repo>/results/* from any cwd/results_dir."""
        from sla_bench.light import find_light
        from sla_bench.sweep import discover_round_roots

        for sub_name, mname in (("h20_fp8", "Qwen3.8-27B-fp8"),):
            roots = discover_round_roots(Path("/tmp/definitely-not-here"), mname, [sub_name])
            self.assertTrue(roots)
            base = REPO_ROOT / "results" / sub_name
            self.assertTrue(all(base in r.parents or r == base for r in roots))
            self.assertTrue(find_light(base, mname).is_file())
        self.assertEqual(discover_round_roots(Path("/tmp/definitely-not-here"), "Qwen3.8-27B-fp8", ["nope"]), [])

    def test_fp8_precision_inference_and_no_round_light_double_loading(self):
        """(a) `-fp8` models export `_FP8_` burst filenames; (b) coexisting `round*` + `light/` never double-load."""
        import contextlib
        import io
        from sla_bench.light import infer_precision, read_light
        from sla_bench.report import auto_discover_specs_from_dir, load_clean_spec_points
        from sla_bench.sweep import load_spec_points

        self.assertEqual(infer_precision("Qwen3.5-35B-A3B-fp8", "Qwen3.5-35B-A3B", "BF16"), "FP8")
        self.assertEqual(infer_precision("Qwen3.8-27B-FP8-tp2", "Qwen3.8-27B", "BF16"), "FP8")
        self.assertEqual(infer_precision("qwen3_32b_w8a8", "Qwen3-32B", "BF16"), "W8A8")
        self.assertEqual(infer_precision("Qwen3.8-27B-tp2", "Qwen3.8-27B", "FP8"), "FP8")
        self.assertEqual(infer_precision("Qwen3.8-27B-tp2", "Qwen3.8-27B", "float8_e4m3fn"), "FP8")
        self.assertEqual(
            infer_precision("Qwen3.8-27B", "Qwen3.8-27B", "", "Qwen3.8-27B_2card_TP2_FP8_sglang_20261124_burst.jsonl"),
            "FP8",
        )
        self.assertEqual(infer_precision("Qwen3.8-27B-tp2", "Qwen3.8-27B", "BF16"), "BF16")

        with tempfile.TemporaryDirectory(prefix="sla_test_fp8_dedup_") as tmp:
            res = Path(tmp) / "fp8_run"
            for r_idx, port in ((0, 30100), (1, 30101)):
                mdir = res / f"round{r_idx}" / "Qwen3.8-27B-fp8"
                rep = mdir / "r00.4000" / f"p{port}" / "report"
                rep.mkdir(parents=True)
                with open(rep / "pre_detail_stat_1.jsonl", "w", encoding="utf-8") as fh:
                    for i in range(16):
                        fh.write(json.dumps({
                            "timestamp": 2000.0 + i, "e2e_lat": 8000.0, "prompt_len": 3720 + (i % 4),
                            "output_len": 1000, "ttft": 480.0 + r_idx * 10.0, "tpot": 18.0 + r_idx * 0.5,
                            "error": None, "finished": True, "sample_index": i, "score": 0.75, "eff_len": 320,
                        }) + "\n")
                (mdir / "curve.json").write_text(json.dumps({
                    "card": "H20",
                    "model": "Qwen3.8-27B-fp8",
                    "served_model_name": "Qwen3.8-27B",
                    "tp": 2,
                    "cards": 2.0,
                    "replicas": 2,
                    "points": [{
                        "rate": 0.4, "rate_sent": 0.4, "n": 16, "n_kept": 8, "n_planned": 16,
                        "missing": 0, "errors": 0, "window_from": 8,
                        "ttft": 480.0 + r_idx * 10.0, "ttft_p90": 500.0,
                        "tpot": 18.0 + r_idx * 0.5, "tpot_p90": 19.0, "e2e": 8000.0,
                        "out_len": 1000.0, "in_len": 3721.5, "in_len_all": 3721.5,
                        "drift": 0.0, "stable": True, "len_ok": True, "port": port,
                        "elapsed": 120.0, "ds_in": 3721.5,
                    }],
                }), encoding="utf-8")

            # 1st export into default `<res>/light` -> must infer `_FP8_` (not `_BF16_`)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["export", "--results-dir", str(res), "--card", "H20"])
            self.assertEqual(rc, 0)
            fp8_files = list((res / "light" / "H20").glob("Qwen3.8-27B_2card_TP2_FP8_sglang_*_burst.jsonl"))
            bf16_files = list((res / "light" / "H20").glob("*_BF16_*_burst.jsonl"))
            self.assertEqual(len(fp8_files), 1)
            self.assertEqual(bf16_files, [])

            # With both `round*` and `light/` now coexisting in `res`, discovery & point loading must NOT double-load!
            specs = auto_discover_specs_from_dir(res, card="H20")
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].precision, "FP8")
            self.assertEqual(len(load_clean_spec_points(res, specs[0], restabilize=False)), 2)
            self.assertEqual(len(load_spec_points(res, specs[0], restabilize=False)), 2)

            # 2nd export on the same `res` must stay idempotent (rounds_count == 2, not 4)
            with contextlib.redirect_stdout(io.StringIO()):
                rc2 = main(["export", "--results-dir", str(res), "--card", "H20"])
            self.assertEqual(rc2, 0)
            meta, lpts = read_light(fp8_files[0])
            self.assertEqual(meta["precision"], "FP8")
            self.assertEqual(len(lpts), 2)
            row0 = json.loads(fp8_files[0].read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row0["rounds_count"], 2)

            # HTML report on `res` with coexisting `round*` + `light/` must report n_sum = 8 + 8 = 16 (not 32)
            out_html = Path(tmp) / "fp8_report.html"
            with contextlib.redirect_stdout(io.StringIO()):
                rc_rep = main(["report", "--results-dir", str(res), "--card", "H20", "-o", str(out_html)])
            self.assertEqual(rc_rep, 0)
            html_txt = out_html.read_text(encoding="utf-8")
            self.assertIn("<td>16</td></tr>", html_txt)
            self.assertNotIn("<td>32</td></tr>", html_txt)

            # Light-only discovery also preserves `FP8` precision
            light_specs = auto_discover_specs_from_dir(res / "light", card="H20")
            self.assertEqual(len(light_specs), 1)
            self.assertEqual(light_specs[0].precision, "FP8")

    def test_multi_host_multi_precision_report_and_achieved_rate_gap(self):
        """Multi-host (H20 + CardX) + multi-precision (FP8 + BF16) keeps rows distinct and populates gap% when rate_sent != rate."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory(prefix="sla_test_xplat_fp8_") as tmp:
            root = Path(tmp)
            for host_dir, card, mname, prec, rate, r_sent, tpot, acc_v in (
                ("fp8_h20", "H20", "Qwen3.8-27B-fp8", "FP8", 0.90, 0.935, 41.0, 0.75),
                ("fp8_cardx", "CardX", "Qwen3.8-27B-fp8", "FP8", 0.30, 0.285, 38.0, 0.50),
                ("bf16_h20", "H20", "Qwen3.8-27B-tp2", "BF16", 0.70, 0.680, 44.0, 0.70),
            ):
                mdir = root / host_dir / "round0" / mname
                mdir.mkdir(parents=True)
                (mdir / "curve.json").write_text(json.dumps({
                    "card": card, "model": mname, "served_model_name": "Qwen3.8-27B",
                    "tp": 2, "cards": 2.0 if card == "H20" else 1.0, "replicas": 2,
                    "points": [{
                        "rate": rate, "rate_sent": r_sent, "n": 40, "n_kept": 32, "n_planned": 40,
                        "missing": 0, "errors": 0, "window_from": 8,
                        "ttft": 600.0, "ttft_p90": 720.0, "tpot": tpot, "tpot_p90": tpot + 2.0,
                        "e2e": 42000.0, "out_len": 1000.0, "in_len": 3755.0, "in_len_all": 3755.0,
                        "drift": 0.0, "stable": True, "len_ok": True, "port": 30100, "elapsed": 240.0, "ds_in": 3755.0,
                    }],
                }), encoding="utf-8")
                acc_rep = root / host_dir / "acc" / mname / "report"
                acc_rep.mkdir(parents=True)
                with open(acc_rep / "pre_detail_stat_1.jsonl", "w", encoding="utf-8") as fh:
                    for q_i in range(4):
                        fh.write(json.dumps({
                            "finished": True, "error": None, "score": acc_v if q_i < 3 else acc_v,
                            "gold_ans": f"ans_{q_i}", "prompt_len": 3500 + q_i,
                        }) + "\n")

            cfg_yaml = root / "xplat.yaml"
            out_html = root / "xplat.html"
            cfg_yaml.write_text(
                f"card: H20\nresults_dir: {root / 'fp8_h20'}\nreport_out: {out_html}\n"
                "models:\n"
                f"  - {{key: h20_fp8, card: H20, model: Qwen3.8-27B-fp8, served_model_name: Qwen3.8-27B, tp: 2, cards: 2, precision: FP8, roots: ['{root / 'fp8_h20'}']}}\n"
                f"  - {{key: cardx_fp8, card: CardX, model: Qwen3.8-27B-fp8, served_model_name: Qwen3.8-27B, tp: 2, cards: 1, precision: FP8, roots: ['{root / 'fp8_cardx'}']}}\n"
                "external_baselines:\n"
                f"  - {{key: h20_bf16, card: H20, model: Qwen3.8-27B-tp2, canonical_model: Qwen3.8-27B, tp: 2, cards: 2, precision: BF16, roots: ['{root / 'bf16_h20'}']}}\n",
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(["report", "-c", str(cfg_yaml)])
            self.assertEqual(rc, 0)
            html_txt = out_html.read_text(encoding="utf-8")
            self.assertIn("Qwen3.8-27B (FP8)", html_txt)
            self.assertIn("Qwen3.8-27B (BF16)", html_txt)
            self.assertNotIn("<td>-</td><td>-</td><td>-</td><td>0</td>", html_txt)
            self.assertIn("75.0%", html_txt)
            self.assertIn("50.0%", html_txt)
            self.assertIn("70.0%", html_txt)


if __name__ == "__main__":
    unittest.main()
