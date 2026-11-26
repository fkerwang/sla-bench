"""Audited v7 multi-replica parallel sweep & 2-level SLA `gap < 10%` adaptive refinement engine."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import glob
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .config import (
    BenchConfig,
    ModelTPSpec,
    SLALevel,
    SweepSettings,
    extract_port_from_url,
    health_url_from_chat_url,
)
from .dataset import balanced_file, compute_or_load_tokenizer_lens, ensure_dataset
from .locust_client import CLIENT_LOCUSTFILE
from .locust_client.remote_tok import derive_tokenize_url
from .stats import (
    bottleneck_metrics,
    by_rate,
    effective_rate,
    ceiling_rates,
    is_point_valid,
    len_ok,
    measure_point,
    point_size,
    stability,
)


@dataclass
class SLABracketStatus:
    """Evaluation of one SLA level on the currently swept rate grid."""

    sla: SLALevel
    best_pass_rate: float
    first_fail_rate: Optional[float]
    best_pass_qpm_card: float
    first_fail_qpm_card: Optional[float]
    mean_ttft: Optional[float] = None
    mean_tpot: Optional[float] = None
    bottleneck_ratio: Optional[float] = None
    bottleneck_dim: Optional[str] = None
    gap: Optional[float] = None
    bracket_rel: Optional[float] = None
    completed: int = 0
    state: str = "empty"  # converged_gap | converged_cliff | needs_bracket_refinement | unbounded_top | below_grid | empty


@dataclass
class RefinementPlan:
    """Result of evaluating all SLA brackets and planning the next adaptive refinement step."""

    converged: bool
    reason: str
    statuses: List[SLABracketStatus] = field(default_factory=list)
    proposed_rates: List[float] = field(default_factory=list)


def spec_tokenizer_url(spec: ModelTPSpec) -> Optional[str]:
    """Server `/tokenize` endpoint used for this spec (None when no server is known, e.g. dry-run)."""
    if spec.tokenizer_url:
        return spec.tokenizer_url
    return derive_tokenize_url(spec.urls[0]) if spec.urls else None


_HELD_SWEEP_LOCKS: Dict[Tuple[int, str], int] = {}


@contextmanager
def acquire_results_lock(results_dir: Path, dry_run: bool = False) -> Iterator[None]:
    """Acquire a re-entrant (per-PID), non-blocking OS flock on `<results_dir>/.sla_bench_sweep.lock`.

    Prevents two concurrent `sla_bench` drivers from writing overlapping Locust streams and
    racing `curve.json` snapshots in the same results directory.
    """
    if dry_run:
        yield
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    key = (os.getpid(), str(results_dir.resolve()))
    if _HELD_SWEEP_LOCKS.get(key, 0) > 0:
        _HELD_SWEEP_LOCKS[key] += 1
        try:
            yield
        finally:
            _HELD_SWEEP_LOCKS[key] -= 1
        return

    lock_path = results_dir / ".sla_bench_sweep.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip() or "<unknown holder>"
        fh.close()
        raise RuntimeError(
            f"Another sla_bench sweep/refine/acc driver is already holding {lock_path}: {holder}. "
            f"Refusing to start a duplicate arrival stream against {results_dir}."
        )
    fh.seek(0)
    fh.truncate()
    fh.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "run_uuid": uuid.uuid4().hex[:12],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "argv": sys.argv,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    fh.flush()
    _HELD_SWEEP_LOCKS[key] = 1
    try:
        yield
    finally:
        _HELD_SWEEP_LOCKS.pop(key, None)
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def resolve_locust_cmd(custom_bin: str = "") -> List[str]:
    """Locate the Locust executable matching the active Python environment."""
    if custom_bin:
        return [custom_bin]
    env_bin = os.environ.get("LOCUST_BIN", "").strip()
    if env_bin:
        return [env_bin]
    sibling = Path(sys.executable).resolve().parent / "locust"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return [str(sibling)]
    which_loc = shutil.which("locust")
    if which_loc:
        return [which_loc]
    return [sys.executable, "-m", "locust"]


def build_rotated_batches(
    urls: Sequence[str],
    rates: Sequence[float],
    round_idx: int,
    step_idx: int = 0,
) -> List[List[Tuple[str, float]]]:
    """Assign `rates` across `urls` in parallel batches, rotating replica assignments per round.

    When `len(urls) >= 2`, `round 0` and `round 1` are guaranteed to assign each rate to a
    different replica/board so board-to-board variance averages out across rounds.
    """
    if not urls:
        raise ValueError("At least one replica URL or port is required to run a sweep.")
    if not rates:
        return []
    k = len(urls)
    if k == 1:
        shift = 0
    else:
        # Round r and the next round must land on DIFFERENT replicas for a rate to be measured
        # on two boards. The old formula `(round_idx // 2 + step_idx) % k` plus an extra
        # correction produced the same shift for round 0 and round 1 on odd refinement steps,
        # so both repeats of a rate ran on the same board (observed in h20_full ref1: 0.19 and
        # 0.21 twice on p30101). `round_idx + step_idx` keeps consecutive rounds distinct for
        # any k >= 2, and the step term still rotates the pairing between refinement steps.
        shift = (round_idx + step_idx) % k

    rotated_urls = [urls[(i + shift) % k] for i in range(k)]
    batches: List[List[Tuple[str, float]]] = []
    for start in range(0, len(rates), k):
        chunk = rates[start : start + k]
        batches.append([(rotated_urls[i], float(r)) for i, r in enumerate(chunk)])
    return batches


REPO_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _resolve_baseline_root(r: str, base: Path) -> List[Path]:
    """Map one configured root to existing directories holding `<model>/curve.json`.

    Lookup order for a relative root: `<results_dir>/<root>`, then `<results_dir>/../<root>`
    (so `h20_v7` works whether results live beside the baseline or inside it), then
    `<repo>/results/<root>` for the baseline shipped with a source checkout.
    """
    cands: List[Path] = []
    p = Path(r)
    if p.is_absolute():
        cands.append(p)
    else:
        # The checkout's own results/ wins: it is the shipped baseline, and a same-named
        # directory next to an unrelated results_dir must not shadow it.
        cands.extend([REPO_RESULTS_DIR / r, base / r, base.parent / r])
    out: List[Path] = []
    for c in cands:
        if not c.is_dir():
            continue
        out.append(c)
        out.extend(sorted(d for d in c.glob("round*") if d.is_dir()))
        break
    return out


def _has_model_data(root: Path, model: str) -> bool:
    from .light import find_light

    return (root / model / "curve.json").exists() or find_light(root, model) is not None


def discover_round_roots(results_dir: str | Path, model_dir_name: str, explicit_roots: Optional[Sequence[str]] = None) -> List[Path]:
    """Discover all round directories (`round0`, `round1`, `round0_ref1`, ...) for a model."""
    base = Path(results_dir)
    if explicit_roots:
        out: List[Path] = []
        for r in explicit_roots:
            # A root may hold <model>/curve.json directly (merged baseline) and/or round* dirs.
            out.extend(c for c in _resolve_baseline_root(r, base) if _has_model_data(c, model_dir_name))
        return out

    if not base.exists():
        return []
    # Direct curve.json inside results_dir/<model>/curve.json
    candidates: List[Path] = []
    if _has_model_data(base, model_dir_name):
        candidates.append(base)
    for child in sorted(base.iterdir()):
        if child.is_dir() and child.name.startswith("round"):
            if _has_model_data(child, model_dir_name):
                candidates.append(child)
    return candidates


def load_spec_points(
    results_dir: str | Path,
    spec: ModelTPSpec,
    stab_z: float = 3.0,
    restabilize: bool = True,
) -> List[dict]:
    """Load all measured points across rounds for `spec`, optionally recomputing v7 stability."""
    from .light import find_light, read_light

    roots = discover_round_roots(results_dir, spec.model, spec.roots or None)
    pts: List[dict] = []
    for rpath in roots:
        cj = rpath / spec.model / "curve.json"
        if cj.is_file():
            try:
                payload = json.loads(cj.read_text(encoding="utf-8"))
            except Exception:
                continue
            raw_points = payload.get("points", [])
        elif find_light(rpath, spec.model) is not None:
            # Light file: points already carry their (re-stabilized) flags and summary stats.
            raw_points = read_light(find_light(rpath, spec.model))[1]
        else:
            continue
        for raw_m in raw_points:
            m = dict(raw_m)
            if not spec.v7_mode and m.get("errors", 0):
                continue
            try:
                rel_src = str(rpath.relative_to(Path(results_dir)))
            except ValueError:
                rel_src = str(rpath)
            m["src"] = rel_src
            if restabilize and spec.v7_mode and m.get("port") and "window_from" in m:
                rep_dir = rpath / spec.model / ("r%07.4f" % float(m["rate"])) / f"p{m['port']}" / "report"
                keep = []
                for fpath in sorted(glob.glob(str(rep_dir / "pre_detail_stat_*.jsonl"))):
                    with open(fpath, encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except ValueError:
                                continue
                            if (
                                rec.get("finished")
                                and not rec.get("error")
                                and (rec.get("sample_index") or 0) >= m["window_from"]
                            ):
                                keep.append(rec)
                if len(keep) >= 8:
                    # Mirror `measure_point()`'s steady-state rule: exclude requests still in
                    # flight when the last prompt was sent, otherwise the unloading drain tail
                    # (low TTFT/TPOT) is folded back into the second half and flips overloaded
                    # points to stable=True. That made refine/report disagree with the sweep that
                    # produced the numbers (H20 27B 0.72/s: drift +370ms unstable during the run,
                    # but +86ms stable after a reload, moving the reported ceiling).
                    start_of = lambda r: float(r["timestamp"]) - float(r["e2e_lat"]) / 1000.0
                    t_last = max(start_of(r) for r in keep)
                    steady = [r for r in keep if start_of(r) + float(r["e2e_lat"]) / 1000.0 <= t_last]
                    if len(steady) >= 8:
                        keep = steady
                    m.update(stability(keep, int(m["window_from"]), stab_z=stab_z))
            pts.append(m)
    return pts


def _round_rate(r: float, ref_scale: float) -> float:
    if ref_scale < 0.05:
        return round(r, 5)
    return round(r, 4)


def _interp_target_rate(
    r_lo: float,
    ms_lo: Sequence[dict],
    r_hi: float,
    ms_hi: Sequence[dict],
    sla: SLALevel,
    target_ratio: float = 0.94,
) -> Optional[float]:
    """Estimate interior rate where bottleneck latency reaches `target_ratio` (e.g. 94%) of SLA."""
    ttft_lo = st.mean(float(m["ttft"]) for m in ms_lo)
    tpot_lo = st.mean(float(m["tpot"]) for m in ms_lo)
    ttft_hi = st.mean(float(m["ttft"]) for m in ms_hi)
    tpot_hi = st.mean(float(m["tpot"]) for m in ms_hi)

    cands: List[float] = []
    ttft_goal = sla.ttft_ms * target_ratio
    if ttft_lo < ttft_goal < ttft_hi:
        frac = (ttft_goal - ttft_lo) / (ttft_hi - ttft_lo)
        cands.append(r_lo + frac * (r_hi - r_lo))

    tpot_goal = sla.tpot_ms * target_ratio
    if tpot_lo < tpot_goal < tpot_hi:
        frac = (tpot_goal - tpot_lo) / (tpot_hi - tpot_lo)
        cands.append(r_lo + frac * (r_hi - r_lo))

    if not cands:
        return None
    est = min(cands)
    span = r_hi - r_lo
    return max(r_lo + 0.12 * span, min(r_hi - 0.12 * span, est))


def plan_refinement_step(
    groups: Dict[float, List[dict]],
    card_num: float,
    slas: Sequence[SLALevel],
    num_replicas: int = 2,
    gap_threshold: float = 0.10,
    min_bracket_rel: float = 0.025,
    min_completed: int = 0,
    v7_mode: bool = True,
    rounds_hint: int = 2,
) -> RefinementPlan:
    """Evaluate all SLAs on `groups` and propose interior (or upward) rates until `gap < 10%`.

    `rounds_hint` is the configured number of board measurements per rate. A boundary point with
    fewer repeats than that is re-measured before being accepted, which is what makes a
    single-board coarse pass safe.
    """
    if not groups:
        return RefinementPlan(converged=False, reason="No measured rate points found.", statuses=[], proposed_rates=[])

    scale_qpm = 60.0 / float(card_num)
    statuses: List[SLABracketStatus] = []
    needs_upward = False
    brackets_to_refine: List[Tuple[float, float, SLALevel]] = []
    under_sampled_rates: List[float] = []

    for sla in slas:
        best_r, fail_r = ceiling_rates(groups, sla.ttft_ms, sla.tpot_ms, v7_mode=v7_mode)
        st_obj = SLABracketStatus(
            sla=sla,
            best_pass_rate=best_r,
            first_fail_rate=fail_r,
            # Achieved rate, not requested: at low rates Poisson sampling shifts it by
            # +-20-25% and overload pushes it below target, so scaling the requested value
            # would report throughput the server never delivered.
            best_pass_qpm_card=(
                st.mean(effective_rate(m) for m in groups[best_r]) * scale_qpm
                if best_r > 0 and groups.get(best_r)
                else 0.0
            ),
            first_fail_qpm_card=(
                st.mean(effective_rate(m) for m in groups[fail_r]) * scale_qpm
                if fail_r is not None and groups.get(fail_r)
                else None
            ),
        )
        if best_r <= 0.0:
            st_obj.state = "below_grid"
            statuses.append(st_obj)
            continue

        ms_best = groups[best_r]
        bm = bottleneck_metrics(ms_best, sla.ttft_ms, sla.tpot_ms)
        st_obj.mean_ttft = bm["mean_ttft"]
        st_obj.mean_tpot = bm["mean_tpot"]
        st_obj.bottleneck_ratio = bm["bottleneck_ratio"]
        st_obj.bottleneck_dim = bm["bottleneck_dim"]
        st_obj.gap = bm["gap"]
        st_obj.completed = bm["completed"]

        if bm["completed"] < min_completed and best_r not in under_sampled_rates:
            under_sampled_rates.append(best_r)

        # A boundary that only has ONE board measurement cannot be trusted: board spread flips
        # the verdict at exactly these points (H20 27B: 0.78 passes on the coarse board but the
        # two-board answer is 0.72). The single-board coarse pass is what makes this routine, so
        # any boundary point measured once is scheduled for a confirming second board.
        if len(ms_best) < max(2, int(rounds_hint)) and best_r not in under_sampled_rates:
            under_sampled_rates.append(best_r)

        if fail_r is None:
            st_obj.state = "unbounded_top"
            needs_upward = True
        else:
            b_rel = (fail_r - best_r) / best_r
            st_obj.bracket_rel = b_rel
            if bm["gap"] < gap_threshold:
                st_obj.state = "converged_gap"
            elif b_rel <= min_bracket_rel:
                st_obj.state = "converged_cliff"
            else:
                st_obj.state = "needs_bracket_refinement"
                brackets_to_refine.append((best_r, fail_r, sla))
        statuses.append(st_obj)

    existing = sorted(groups.keys())

    # 1. If the highest swept rate still passes an SLA (`first_fail_rate is None`), step upward first!
    if needs_upward:
        top_r = max(existing)
        n_up = max(2, min(max(num_replicas, 2), 4))
        mults = [1.25, 1.50, 1.85, 2.25][:n_up]
        up_rates: List[float] = []
        for mult in mults:
            cand = _round_rate(top_r * mult, top_r)
            if all(abs(cand - e) / max(e, 1e-9) > 0.01 for e in existing + up_rates):
                up_rates.append(cand)
        return RefinementPlan(
            converged=False,
            reason=f"Top swept rate {top_r:g} req/s still passes SLA; stepping upward to bracket ceiling.",
            statuses=statuses,
            proposed_rates=sorted(up_rates),
        )

    # 2. Propose interior rates for brackets where gap >= gap_threshold and bracket_rel > min_bracket_rel
    if brackets_to_refine:
        # Deduplicate identical (r_lo, r_hi) brackets while keeping the stricter SLA if shared
        uniq_brackets: Dict[Tuple[float, float], SLALevel] = {}
        for r_lo, r_hi, sla in brackets_to_refine:
            key = (r_lo, r_hi)
            if key not in uniq_brackets or sla.tpot_ms < uniq_brackets[key].tpot_ms:
                uniq_brackets[key] = sla

        n_brackets = len(uniq_brackets)
        # Fill available replicas cleanly (at least 2-3 interior points per bracket)
        pts_per_bracket = max(2, min(5, max(num_replicas, 3) // n_brackets))
        if num_replicas >= 4 and n_brackets == 1:
            pts_per_bracket = min(num_replicas, 5)

        proposed: List[float] = []
        for (r_lo, r_hi), sla in sorted(uniq_brackets.items()):
            span = r_hi - r_lo
            even = [r_lo + span * (j / float(pts_per_bracket + 1)) for j in range(1, pts_per_bracket + 1)]
            est = _interp_target_rate(r_lo, groups[r_lo], r_hi, groups[r_hi], sla, target_ratio=0.94)
            if est is not None and even:
                # Replace the closest evenly spaced point with the interpolation-guided anchor
                nearest_idx = min(range(len(even)), key=lambda idx: abs(even[idx] - est))
                even[nearest_idx] = est

            min_sep = max(1e-5, min(0.008 * r_lo, 0.20 * span))
            for raw_c in sorted(even):
                cand = _round_rate(raw_c, r_lo)
                if cand <= r_lo or cand >= r_hi:
                    continue
                if all(abs(cand - e) >= min_sep for e in existing + proposed):
                    proposed.append(cand)

            # Fallback midpoint if rounding/filtering dropped everything in a narrow but >min_bracket_rel interval
            if not any(r_lo < p < r_hi for p in proposed):
                mid = _round_rate(0.5 * (r_lo + r_hi), r_lo)
                if r_lo < mid < r_hi and mid not in existing and mid not in proposed:
                    proposed.append(mid)

        if proposed:
            parts = [
                f"SLA{s.sla.label} [{s.best_pass_rate:g}, {s.first_fail_rate:g}] gap={s.gap * 100:.1f}%>={gap_threshold * 100:.0f}%"
                for s in statuses
                if s.state == "needs_bracket_refinement" and s.gap is not None
            ]
            return RefinementPlan(
                converged=False,
                reason="Refining SLA bracket(s): " + "; ".join(parts),
                statuses=statuses,
                proposed_rates=sorted(proposed),
            )

    # 3. If gap/cliff converged but a boundary point has < min_completed requests (e.g. single-round legacy data), re-run it
    if under_sampled_rates:
        return RefinementPlan(
            converged=False,
            reason=(
                f"Boundary rate(s) {sorted(under_sampled_rates)} need a confirming measurement "
                f"(single-board only, or fewer than {int(min_completed) or 'the configured'} "
                f"requests); adding a repeat pass on another board."
            ),
            statuses=statuses,
            proposed_rates=sorted(under_sampled_rates),
        )

    return RefinementPlan(
        converged=True,
        reason=f"All SLA boundaries converged (gap < {gap_threshold * 100:.1f}% or cliff <= {min_bracket_rel * 100:.1f}%).",
        statuses=statuses,
        proposed_rates=[],
    )


def _log_line(log_path: Path, prefix: str, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(f"[{prefix}] {line}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def check_replicas_healthy(urls: Sequence[str], health_path: str = "/health_generate", timeout_s: float = 10.0) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for u in urls:
        hurl = health_url_from_chat_url(u, health_path=health_path)
        try:
            urllib.request.urlopen(hurl, timeout=timeout_s).read()
        except Exception as exc:
            errors.append(f"{hurl}: {exc}")
    return len(errors) == 0, errors


def build_dual_board_batches(
    urls: Sequence[str],
    rates: Sequence[float],
    boards: int = 2,
) -> List[List[Tuple[str, float]]]:
    """Place every rate on `boards` DIFFERENT replicas inside one batch.

    Refinement needs each candidate measured on two boards (board spread flips verdicts:
    H20 27B at 0.70/s gives TPOT 52.2 on one board and 34.8 on the other). Collecting the two
    measurements in one pass instead of two serial passes halves the refinement cost whenever
    there are at least `2*len(rates)` replicas -- e.g. a 6-replica host with 2-3 candidates
    goes from 2 batches to 1.
    """
    if not urls:
        raise ValueError("At least one replica URL or port is required to run a sweep.")
    if not rates:
        return []
    k = len(urls)
    boards = max(1, min(int(boards), k))
    per_batch = max(1, k // boards)
    batches: List[List[Tuple[str, float]]] = []
    bi = 0
    for start in range(0, len(rates), per_batch):
        chunk = rates[start:start + per_batch]
        pairs: List[Tuple[str, float]] = []
        off = (bi * boards) % k
        for i, r in enumerate(chunk):
            for b in range(boards):
                pairs.append((urls[(off + i * boards + b) % k], float(r)))
        seen: set[str] = set()
        uniq: List[Tuple[str, float]] = []
        for u, r in pairs:
            if u in seen:
                continue
            seen.add(u)
            uniq.append((u, r))
        if uniq:
            batches.append(uniq)
        bi += 1
    return batches


def run_parallel_batches_for_round(
    round_dir: Path,
    spec: ModelTPSpec,
    batches: Sequence[Sequence[Tuple[str, float]]],
    settings: SweepSettings,
    round_idx: int,
    lens: List[int],
    ds_in: float,
    ds_file: Path,
) -> List[dict]:
    """Execute one round of parallel `(url, rate)` batches and write `<round_dir>/<spec.model>/curve.json`."""
    model_out_dir = round_dir / spec.model
    model_out_dir.mkdir(parents=True, exist_ok=True)
    log_file = round_dir / f"{spec.model}.log"
    locust_cmd_prefix = resolve_locust_cmd(settings.locust_bin)

    # Load any already-completed points in this round_dir so interrupted runs resume cleanly
    cj_path = model_out_dir / "curve.json"
    existing_pts: List[dict] = []
    done_rates = set()
    if cj_path.is_file():
        try:
            existing_pts = list(json.loads(cj_path.read_text(encoding="utf-8")).get("points", []))
            done_rates = {round(float(p["rate"]), 5) for p in existing_pts}
        except Exception:
            existing_pts = []

    pts = list(existing_pts)
    _log_line(
        log_file,
        spec.model,
        f"=== Round {round_idx} ({round_dir.name}): {len(batches)} batch(es), warmup={settings.warmup_s:.0f}s "
        f"min_s={settings.min_s:.0f}s min_completed={settings.min_completed} ds_in={ds_in:.1f} max_tokens={settings.max_tokens} ===",
    )

    for bi, batch in enumerate(batches):
        todo_batch = [(u, r) for u, r in batch if round(float(r), 5) not in done_rates]
        if not todo_batch:
            continue

        if settings.health_check:
            ok, errs = check_replicas_healthy([u for u, _ in todo_batch], settings.health_path, settings.health_timeout_s)
            if not ok:
                raise RuntimeError(f"Replica health check failed before {spec.model} batch {bi}: {'; '.join(errs)}")

        running: Dict[Tuple[str, float], Tuple[subprocess.Popen, object, Path, float, int, float, Path, int]] = {}
        for i, (url, rate) in enumerate(todo_batch):
            port = extract_port_from_url(url, fallback_idx=i)
            rate_dir = model_out_dir / ("r%07.4f" % rate)
            sub_dir = rate_dir / f"p{port}"
            rep_dir = sub_dir / "report"
            if rep_dir.is_dir():
                for stale in rep_dir.glob("pre_detail_stat_*.jsonl"):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
            rep_dir.mkdir(parents=True, exist_ok=True)

            seed = 100 + 7 * bi + i + 17 * round_idx
            n_planned = point_size(
                rate=rate,
                warmup_s=settings.warmup_s,
                min_s=settings.min_s,
                min_completed=settings.min_completed,
                rounds=settings.rounds,
                min_per_round=settings.min_per_round,
            )
            bal_path, _ = balanced_file(
                dataset_path=ds_file,
                lens=lens,
                seed=seed,
                tag=f"{spec.model}_r{rate:g}_p{port}_rd{round_idx}",
                n=n_planned,
                out_dir=sub_dir,
            )
            # Size the user pool to the concurrency this point actually needs (Little's law:
            # users = rate * e2e). Two opposite failure modes to avoid:
            #   * too many users (the old fixed 1024): users hold future arrival slots and their
            #     sleeping/waking churn delays due sends -- measured realised rate 0.84 at 0.1/s;
            #   * too few users: the pool caps in-flight requests, so the arrival schedule cannot
            #     be kept -- with `rate*30+8` a 0.52/s point got 23 users while 48 were needed
            #     (e2e 92 s under load) and only realised 60% of the target rate.
            # e2e grows with load, so estimate with a generous per-request duration and clamp.
            e2e_per_request_s = 120.0
            users = int(min(settings.cap_users, max(32, rate * e2e_per_request_s + 16)))
            cmd = locust_cmd_prefix + [
                "-f",
                str(CLIENT_LOCUSTFILE),
                "--headless",
                "-u",
                str(users),
                "-r",
                f"{settings.spawn_rate:g}",
                "-t",
                "300m",
                "--url",
                url,
                "--file",
                str(bal_path),
                "--model_name",
                spec.served_model_name,
                "--send_mode",
                "rate_limit",
                "--burstiness",
                "1.0",
                "--max_tokens",
                str(settings.max_tokens),
                "--num_prompts",
                "0",
                "--sample_type",
                "0",
                # Shared single-pass queue. Multi-replica runs were observed replaying prompts
                # (848 records for 316 prompts, each up to 3x) under this mode, so the scored
                # set is deduplicated by content downstream; the mode itself still gives each
                # request a distinct prompt while the queue lasts.
                "--share_data",
                "2",
                "--report_dir",
                str(sub_dir / "report"),
                "--csv",
                str(sub_dir / "report" / "report"),
            ]
            env = dict(
                os.environ,
                CAT_LOCUST_RATE=f"{rate:.6f}",
                CAT_LOCUST_ARRIVAL=settings.arrival,
                CAT_LOCUST_PAD_TO_TOKENS=str(settings.max_tokens),
            )
            # All tokenization is server-side: count against the very replica under test
            # (explicit `tokenizer_url` overrides; otherwise derived from this batch URL).
            env["CAT_LOCUST_TOKENIZER_URL"] = spec.tokenizer_url or derive_tokenize_url(url)
            env["CAT_LOCUST_TOKENIZER_MODEL"] = spec.served_model_name

            lf = open(sub_dir / "locust.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            cap_s = max(settings.max_s, n_planned / max(rate, 1e-6) * 1.3 + 600.0)
            running[(url, rate)] = (proc, lf, sub_dir, time.time(), n_planned, cap_s, bal_path, port)

        while running:
            time.sleep(5)
            for key in list(running.keys()):
                proc, lf, sub_dir, t0, n_planned, cap_s, bal_path, port = running[key]
                el = time.time() - t0
                done = sum(
                    1
                    for fpath in glob.glob(str(sub_dir / "report" / "pre_detail_stat_*.jsonl"))
                    for _ in open(fpath, encoding="utf-8", errors="ignore")
                )
                if proc.poll() is None and el < cap_s and done < n_planned:
                    continue
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                lf.close()  # type: ignore[attr-defined]
                del running[key]
                try:
                    os.remove(bal_path)
                except OSError:
                    pass

                rate = key[1]
                m = measure_point(
                    sub_dir=sub_dir,
                    rate=rate,
                    n_planned=n_planned,
                    warmup_s=settings.warmup_s,
                    stab_z=settings.stab_z,
                )
                if m is None:
                    _log_line(log_file, spec.model, f"port {port} rate {rate:.4f} -> insufficient samples")
                    continue
                m.update(
                    port=port,
                    url=key[0],
                    elapsed=el,
                    ds_in=ds_in,
                    tp=spec.tp,
                    cards=spec.card_num,
                    card_num=spec.card_num,
                    # QPM/card from the rate actually achieved, not the requested one: at low
                    # rates Poisson sampling moves it +-20-50%, and under overload the achieved
                    # rate is necessarily below target. Either way the honest figure is what the
                    # server was really asked to serve.
                    qpm_per_card=effective_rate(m) * 60.0 / float(spec.card_num),
                )
                m["len_ok"] = len_ok(m, ds_in=ds_in, max_tokens=settings.max_tokens, len_tol=settings.len_tol) and (
                    m["missing"] == 0
                )
                pts.append(m)
                pts.sort(key=lambda x: float(x["rate"]))
                cj_payload = {
                    "card": spec.card,
                    "model": spec.model,
                    "served_model_name": spec.served_model_name,
                    "tp": spec.tp,
                    "cards": spec.card_num,
                    "card_num": spec.card_num,
                    "replicas": len(spec.urls),
                    "arrival": settings.arrival,
                    "mode": "parallel",
                    "points": pts,
                }
                _atomic_write_json(cj_path, cj_payload)
                _log_line(
                    log_file,
                    spec.model,
                    f"port {port} rate {rate:.4f} = {m['qpm_per_card']:.2f} QPM/card  "
                    f"n={m['n_kept']}/{m['n_planned']} err={m['errors']}  {m['elapsed']:.0f}s  "
                    f"in {m['in_len']:.0f} (all {m['in_len_all']:.0f}) out {m['out_len']:.0f} "
                    f"{'LEN_OK' if m['len_ok'] else 'LEN_BAD'}  "
                    f"TTFT {m['ttft']:.0f} (p90 {m['ttft_p90']:.0f})  TPOT {m['tpot']:.1f} (p90 {m['tpot_p90']:.1f})  "
                    f"drift {m['drift']:+.0f}+-{m['drift_se']:.0f}  {'STABLE' if m['stable'] else 'UNSTABLE'}  "
                    f"sent {m['rate_sent']:.3f}/s",
                )
    return pts


def _max_pass_rate(results_dir: Path, spec: ModelTPSpec, cfg: BenchConfig) -> float:
    """Highest rate that still satisfies SLA① in the sweep, else the top measured rate.

    Accuracy is reported at the card's own maximum sustainable throughput, so the run is
    executed at this rate instead of an arbitrary number.
    """
    pts = load_spec_points(results_dir, spec, stab_z=cfg.sweep.stab_z, restabilize=True)
    if not pts:
        return 0.0
    groups = by_rate(pts)
    sla1 = cfg.slas[0]
    best = 0.0
    for r, ms in groups.items():
        if all(is_point_valid(m, v7_mode=spec.v7_mode) for m in ms):
            m_ttft = st.mean(float(m["ttft"]) for m in ms)
            m_tpot = st.mean(float(m["tpot"]) for m in ms)
            if m_ttft <= sla1.ttft_ms and m_tpot <= sla1.tpot_ms:
                best = max(best, float(r))
    return best or min(groups.keys())


def run_fixed_acc_pass(
    cfg: BenchConfig,
    spec: ModelTPSpec,
    dry_run: bool = False,
) -> Optional[dict]:
    """Accuracy on a fixed item slice at the card's own maximum sustainable throughput.

    Every per-point accuracy inside a sweep is computed over whatever prompts landed in that
    point's judged window, so two cards are scored on different questions. This pass fixes
    both knobs that make accuracy comparable:

    * **items** -- one fixed slice (`acc_seed`), identical for every card;
    * **load** -- the highest rate that still met SLA① in this run, i.e. the card at its own
      ceiling, so neither a light nor an overloaded point is scored.

    Same open-loop Poisson arrival as the sweep (`warmup_s` discarded), results in
    `<results_dir>/acc/<model>/`, which the report prefers over the per-point number.
    """
    settings = cfg.sweep
    n = int(settings.acc_prompts or 0)
    if n <= 0 or not spec.urls:
        return None

    results_dir = Path(cfg.results_dir)
    acc_dir = results_dir / "acc" / spec.model
    report_dir = acc_dir / "report"
    rate = _max_pass_rate(results_dir, spec, cfg)
    if dry_run and rate <= 0:
        est_rate = min(spec.rates) if spec.rates else 0.0
        print(
            f"[dry-run] {spec.model}: acc pass {n} fixed prompts (seed={settings.acc_seed}, "
            f"warm={settings.acc_warm_prompts}) @ SLA① ceiling after sweep (fallback {est_rate:g}/s)",
            flush=True,
        )
        return None
    if rate <= 0:
        raise SystemExit(
            f"[acc] {spec.model}: no measured sweep points under {results_dir}, so the SLA① "
            f"ceiling rate is unknown. Run `sweep` first, or pass --results-dir pointing at a "
            f"completed run. (A zero rate previously produced a ZeroDivisionError storm in the "
            f"locust client instead of a clear error here.)"
        )
    if dry_run:
        print(f"[dry-run] {spec.model}: acc pass {n} fixed prompts @ {rate:g}/s (SLA① ceiling)", flush=True)
        return None

    with acquire_results_lock(results_dir, dry_run=False):
        return _run_fixed_acc_pass_locked(cfg, spec, results_dir, acc_dir, report_dir, rate, n)


def _run_fixed_acc_pass_locked(
    cfg: BenchConfig,
    spec: ModelTPSpec,
    results_dir: Path,
    acc_dir: Path,
    report_dir: Path,
    rate: float,
    n: int,
) -> Optional[dict]:
    settings = cfg.sweep
    ds_file = ensure_dataset(settings.dataset_path or None, cache_dir=settings.cache_dir or None)
    lens, ds_in_val = compute_or_load_tokenizer_lens(
        dataset_path=ds_file,
        tokenizer_url=spec_tokenizer_url(spec),
        model=spec.served_model_name,
        pad_to_tokens=settings.max_tokens,
        cache_dir=settings.cache_dir or None,
        fallback_ds_in=spec.ds_in,
    )
    # Start from a clean report dir: `share_data=2` replays the same file, so any leftover
    # jsonl from an earlier attempt would be scored a second time (one run reported 297/329
    # "judged" against an expected 200).
    if report_dir.is_dir():
        for stale in report_dir.glob("pre_detail_stat_*.jsonl"):
            try:
                stale.unlink()
            except OSError:
                pass
    report_dir.mkdir(parents=True, exist_ok=True)
    # Comparability contract: the SAME `n` questions scored on every card.
    #
    # The client serves a file front-to-back, so the file is built as
    #     [ warm-up block ][ the fixed n questions ]
    # and only the tail is scored. BOTH blocks are fixed across cards: the warm-up size must
    # not depend on the rate, because the file becomes the `sample_index` axis -- a
    # rate-dependent warm-up (99 items on H20 at 3.32/s vs 43 on a slower card) made identical indices
    # refer to different questions, so the two accuracy numbers were not comparable at all.
    #
    # The block is sized for the LOWEST rate we score at (e.g. 27B at ~0.29/s -> 30 s is ~9
    # requests), so it must stay small: a 128-item block takes 137 s at 0.93/s and would push
    # the scored slice past the time-based warm-up cut. Scoring is by TIME (`warmup_s`), and
    # this block only exists to absorb the cold-start requests inside that window.
    # The prompt file must be BYTE-IDENTICAL on every card, and `balanced_file` cannot give
    # that: it stratifies by `lens`, which comes from this host's own `/tokenize`, so two hosts
    # produce different orderings and the same `sample_index` ends up meaning different
    # questions (observed: 6 of 301 indices differed between two hosts on 27B). For an
    # accuracy slice that ordering buys nothing -- the point is the same questions, not the
    # same length profile -- so take a plain contiguous slice of the dataset in file order.
    warmup_burst = max(8, int(settings.acc_warm_prompts))
    fixed_path = acc_dir / f"_acc_fixed_{spec.model}.ndjson"
    warm_path = acc_dir / f"_acc_warm_{spec.model}.ndjson"
    with open(ds_file, encoding="utf-8-sig") as fh:
        all_lines = [ln for ln in fh if ln.strip()]
    start = settings.acc_seed % max(1, len(all_lines) - n - warmup_burst)
    with open(warm_path, "w", encoding="utf-8") as out:
        for ln in all_lines[start:start + warmup_burst]:
            out.write(ln if ln.endswith("\n") else ln + "\n")
    fixed_lines = all_lines[start + warmup_burst:start + warmup_burst + n]
    with open(fixed_path, "w", encoding="utf-8") as out:
        for ln in fixed_lines:
            out.write(ln if ln.endswith("\n") else ln + "\n")
    fixed_lens = lens[start + warmup_burst:start + warmup_burst + n]
    bal_path = acc_dir / f"_bal_acc_{spec.model}_s{settings.acc_seed}.ndjson"
    with open(bal_path, "w", encoding="utf-8") as out:
        for src in (warm_path, fixed_path):
            with open(src, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    n_planned = warmup_burst + n
    url = spec.urls[0]
    # At least as many users as the queue needs, but never fewer than the in-flight
    # concurrency (`L = rate * e2e_s`). Too few users would cap throughput below the target
    # rate; too many leaves users waiting on an empty queue after it drains.
    acc_users = min(int(settings.cap_users), max(32, int(rate * 30) + 16))
    cmd = resolve_locust_cmd(settings.locust_bin) + [
        "-f", str(CLIENT_LOCUSTFILE), "--headless",
        "-u", str(acc_users), "-r", f"{settings.spawn_rate:g}", "-t", "300m",
        "--url", url, "--file", str(bal_path),
        "--model_name", spec.served_model_name,
        "--send_mode", "rate_limit",     # same open-loop arrival as the sweep
        "--burstiness", "1.0",
        "--max_tokens", str(settings.max_tokens),
        "--num_prompts", "0",
        "--sample_type", "0",
        "--share_data", "2",
        "--report_dir", str(report_dir),
        "--csv", str(report_dir / "report"),
    ]
    env = dict(
        os.environ,
        CAT_LOCUST_RATE=f"{rate:.6f}",
        CAT_LOCUST_ARRIVAL=settings.arrival,
        CAT_LOCUST_PAD_TO_TOKENS=str(settings.max_tokens),
    )
    env["CAT_LOCUST_TOKENIZER_URL"] = spec.tokenizer_url or derive_tokenize_url(url)
    env["CAT_LOCUST_TOKENIZER_MODEL"] = spec.served_model_name

    t0 = time.time()
    # Stop as soon as the fixed slice has been served. The count is the number of DISTINCT
    # questions answered, not raw records: with several replicas sharing one queue the file can
    # be replayed past its length (observed 848 records for 316 prompts), so counting lines
    # would stop late and let duplicates in.
    target_recs = n_planned

    def _distinct_answered() -> int:
        keys: set[tuple] = set()
        for fpath in glob.glob(str(report_dir / "pre_detail_stat_*.jsonl")):
            with open(fpath, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if not r.get("finished") or r.get("error"):
                        continue
                    keys.add((str(r.get("gold_ans") or ""), int(r.get("prompt_len") or 0)))
        return len(keys)

    with open(acc_dir / "locust.log", "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        cap_s = max(settings.max_s, n_planned / max(rate, 1e-6) * 1.3 + 600.0)
        while proc.poll() is None:
            if _distinct_answered() >= target_recs or time.time() - t0 > cap_s:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(5)

    # Accuracy over the scored slice only: drop the warm-up block by request index (guaranteed
    # correct even if the clock or the server's pacing shifts), which keeps every card's score
    # over exactly `n` items.
    t_warm = t0 + settings.warmup_s
    recs_total = recs_win = recs_warm = 0
    scores: list[float] = []
    for fpath in sorted(glob.glob(str(report_dir / "pre_detail_stat_*.jsonl"))):
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                recs_total += 1
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not r.get("finished") or r.get("error") or r.get("score") is None:
                    continue
                si = r.get("sample_index")
                # Scored = exactly the fixed slice, identified by file position. This alone is
                # both the warm-up cut and the comparability guarantee: the warm-up block is
                # the only part served before steady state, and the same rows map to the same
                # questions on every card. Adding a separate time cut would drop extra items at
                # low rates (at 0.93/s a 30 s cut removes 12 more) and break the "same n items"
                # contract; at high rates the index cut already covers warm-up.
                if isinstance(si, int):
                    if si < warmup_burst or si >= n_planned:
                        recs_warm += 1
                        continue
                elif float(r.get("timestamp") or 0) < t_warm:
                    recs_warm += 1
                    continue
                recs_win += 1
                scores.append(float(r["score"]))
    try:
        for tmp in (bal_path, fixed_path, warm_path):
            os.remove(tmp)
    except OSError:
        pass

    summary = {
        "model": spec.model,
        "card": spec.card,
        "rate": rate,
        "prompts_planned": n_planned,
        "prompts_scored": n,
        "records": recs_total,
        "records_judged": recs_win,
        "warmup_s": settings.warmup_s,
        "ds_in": ds_in_val,
        "in_len_mean": st.mean(fixed_lens) if fixed_lens else None,
        "elapsed_s": time.time() - t0,
        "mode": "open_loop_poisson_at_sla1_ceiling",
        "seed": settings.acc_seed,
        "acc": (st.mean(scores) * 100.0) if scores else None,
    }
    (acc_dir / "acc_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    acc_txt = f"{summary['acc']:.1f}%" if summary["acc"] is not None else "n/a"
    print(
        f"[acc] {spec.model}: {recs_win}/{recs_total} judged @ {rate:g}/s, acc {acc_txt}, "
        f"{summary['elapsed_s']:.0f}s",
        flush=True,
    )
    return summary


def format_plan_summary(spec: ModelTPSpec, plan: RefinementPlan) -> str:
    lines = [f"[{spec.card} · {spec.model} (TP={spec.tp}, {spec.card_num:g} card)] {plan.reason}"]
    for s in plan.statuses:
        if s.best_pass_rate <= 0:
            lines.append(f"  SLA{s.sla.label} (TTFT<={s.sla.ttft_ms:g}, TPOT<={s.sla.tpot_ms:g}): no passing rate ({s.state})")
        else:
            f_str = f"{s.first_fail_qpm_card:.2f}" if s.first_fail_qpm_card is not None else "unbounded"
            gap_str = f"{s.gap * 100:.1f}% ({s.bottleneck_dim})" if s.gap is not None else "-"
            rel_str = f"{s.bracket_rel * 100:.1f}%" if s.bracket_rel is not None else "-"
            lines.append(
                f"  SLA{s.sla.label}: best={s.best_pass_qpm_card:.2f} QPM/card (@{s.best_pass_rate:g}/s) "
                f"fail={f_str} | TTFT={s.mean_ttft:.0f}ms TPOT={s.mean_tpot:.1f}ms | "
                f"gap={gap_str} bracket_rel={rel_str} n={s.completed} [{s.state}]"
            )
    if plan.proposed_rates:
        qpm_str = ", ".join(f"{r:g}/s ({r * 60.0 / spec.card_num:.2f} QPM/card)" for r in plan.proposed_rates)
        lines.append(f"  -> Proposed rates: {qpm_str}")
    return "\n".join(lines)


def run_refine_for_spec(
    cfg: BenchConfig,
    spec: ModelTPSpec,
    dry_run: bool = False,
) -> RefinementPlan:
    """Iteratively refine SLA① & SLA② brackets on `spec` until `gap < gap_threshold` (default 10%)."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with acquire_results_lock(results_dir, dry_run=dry_run):
        return _run_refine_for_spec_locked(cfg, spec, results_dir, dry_run=dry_run)


def _run_refine_for_spec_locked(
    cfg: BenchConfig,
    spec: ModelTPSpec,
    results_dir: Path,
    dry_run: bool = False,
) -> RefinementPlan:
    settings = cfg.sweep

    # Determine starting refinement step index from existing round0_ref<k> folders
    existing_steps = []
    for p in results_dir.glob("round0_ref*"):
        suffix = p.name[len("round0_ref") :]
        if suffix.isdigit():
            existing_steps.append(int(suffix))
        elif suffix == "":
            existing_steps.append(0)
    next_step = (max(existing_steps) + 1) if existing_steps else 1

    ds_file: Optional[Path] = None
    lens: Optional[List[int]] = None
    ds_in_val: Optional[float] = None

    for step_offset in range(settings.max_refine_steps):
        pts = load_spec_points(results_dir, spec, stab_z=settings.stab_z, restabilize=True)
        groups = by_rate(pts)
        plan = plan_refinement_step(
            groups=groups,
            card_num=spec.card_num,
            slas=cfg.slas,
            num_replicas=max(len(spec.urls), 1),
            gap_threshold=settings.gap_threshold,
            min_bracket_rel=settings.min_bracket_rel,
            min_completed=settings.min_completed,
            v7_mode=spec.v7_mode,
            rounds_hint=settings.rounds,
        )
        print(format_plan_summary(spec, plan), flush=True)
        if plan.converged or not plan.proposed_rates or dry_run:
            return plan

        if not spec.urls:
            raise ValueError(f"Spec {spec.model} needs refinement {plan.proposed_rates}, but no --ports/--urls were given.")

        if ds_file is None or lens is None or ds_in_val is None:
            ds_file = ensure_dataset(settings.dataset_path or None, cache_dir=settings.cache_dir or None)
            lens, ds_in_val = compute_or_load_tokenizer_lens(
                dataset_path=ds_file,
                tokenizer_url=spec_tokenizer_url(spec),
                model=spec.served_model_name,
                pad_to_tokens=settings.max_tokens,
                cache_dir=settings.cache_dir or None,
                fallback_ds_in=spec.ds_in,
            )
            if spec.ds_in is not None:
                ds_in_val = float(spec.ds_in)

        step_num = next_step + step_offset
        # Refinement measures each candidate on two boards (board spread flips verdicts). When
        # the replicas allow it, both measurements happen in ONE batch; otherwise fall back to
        # the two serial rotating passes.
        reps = max(2, int(settings.rounds))
        if len(spec.urls) >= 2 * max(1, len(plan.proposed_rates)):
            round_dir = results_dir / f"round0_ref{step_num}"
            batches = build_dual_board_batches(spec.urls, plan.proposed_rates, boards=reps)
            run_parallel_batches_for_round(
                round_dir=round_dir,
                spec=spec,
                batches=batches,
                settings=settings,
                round_idx=step_num,
                lens=lens,
                ds_in=ds_in_val,
                ds_file=ds_file,
            )
        else:
            for r_idx in range(settings.rounds):
                round_dir = results_dir / f"round{r_idx}_ref{step_num}"
                batches = build_rotated_batches(spec.urls, plan.proposed_rates, round_idx=r_idx, step_idx=step_num)
                run_parallel_batches_for_round(
                    round_dir=round_dir,
                    spec=spec,
                    batches=batches,
                    settings=settings,
                    round_idx=r_idx + 2 * step_num,
                    lens=lens,
                    ds_in=ds_in_val,
                    ds_file=ds_file,
                )
                if r_idx + 1 < settings.rounds and settings.inter_round_cooldown_s > 0:
                    time.sleep(settings.inter_round_cooldown_s)

    pts = load_spec_points(results_dir, spec, stab_z=settings.stab_z, restabilize=True)
    final_plan = plan_refinement_step(
        groups=by_rate(pts),
        card_num=spec.card_num,
        slas=cfg.slas,
        num_replicas=max(len(spec.urls), 1),
        gap_threshold=settings.gap_threshold,
        min_bracket_rel=settings.min_bracket_rel,
        min_completed=settings.min_completed,
        v7_mode=spec.v7_mode,
        rounds_hint=settings.rounds,
    )
    print(format_plan_summary(spec, final_plan), flush=True)
    return final_plan


def run_sweep_for_spec(
    cfg: BenchConfig,
    spec: ModelTPSpec,
    auto_refine: bool = True,
    dry_run: bool = False,
) -> RefinementPlan:
    """Run initial coarse rate grid across rotated rounds, then optionally run `gap < 10%` refinement."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with acquire_results_lock(results_dir, dry_run=dry_run):
        return _run_sweep_for_spec_locked(cfg, spec, results_dir, auto_refine=auto_refine, dry_run=dry_run)


def _run_sweep_for_spec_locked(
    cfg: BenchConfig,
    spec: ModelTPSpec,
    results_dir: Path,
    auto_refine: bool = True,
    dry_run: bool = False,
) -> RefinementPlan:
    settings = cfg.sweep

    if spec.rates:
        if dry_run:
            for r_idx in range(settings.rounds):
                batches = build_rotated_batches(spec.urls or ["http://127.0.0.1:8000/v1/chat/completions"], spec.rates, r_idx)
                print(f"[dry-run] {spec.model} round{r_idx} batches: {batches}", flush=True)
        else:
            ds_file = ensure_dataset(settings.dataset_path or None, cache_dir=settings.cache_dir or None)
            lens, ds_in_val = compute_or_load_tokenizer_lens(
                dataset_path=ds_file,
                tokenizer_url=spec_tokenizer_url(spec),
                model=spec.served_model_name,
                pad_to_tokens=settings.max_tokens,
                cache_dir=settings.cache_dir or None,
                fallback_ds_in=spec.ds_in,
            )
            if spec.ds_in is not None:
                ds_in_val = float(spec.ds_in)

            # Coarse pass: ONE board per rate.
            #
            # The coarse grid only has to bracket the boundary roughly; the refinement then
            # places candidates inside the bracket and measures each of them on BOTH boards.
            # Measuring every coarse rate twice doubles the most expensive stage (each point
            # costs warmup+min_s ~= 200 s) to buy precision that refinement discards.
            #
            # This is safe only because refinement re-measures the boundary on two boards: a
            # single-board coarse pass can over-estimate the bracket edge (H20 27B SLA① looked
            # like 0.78 on one board where the two-board answer is 0.72), and the two-board
            # refinement is what corrects it -- so the bracket from the coarse pass must stay
            # open-ended until refinement closes it.
            round_dir = results_dir / "round0"
            batches = build_rotated_batches(spec.urls, spec.rates, round_idx=0, step_idx=0)
            run_parallel_batches_for_round(
                round_dir=round_dir,
                spec=spec,
                batches=batches,
                settings=settings,
                round_idx=0,
                lens=lens,
                ds_in=ds_in_val,
                ds_file=ds_file,
            )

    if auto_refine:
        return run_refine_for_spec(cfg, spec, dry_run=dry_run)
    pts = load_spec_points(results_dir, spec, stab_z=settings.stab_z, restabilize=True)
    plan = plan_refinement_step(
        groups=by_rate(pts),
        card_num=spec.card_num,
        slas=cfg.slas,
        num_replicas=max(len(spec.urls), 1),
        gap_threshold=settings.gap_threshold,
        min_bracket_rel=settings.min_bracket_rel,
        min_completed=settings.min_completed,
        v7_mode=spec.v7_mode,
        rounds_hint=settings.rounds,
    )
    print(format_plan_summary(spec, plan), flush=True)
    return plan
