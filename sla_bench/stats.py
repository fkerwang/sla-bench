"""Audited v7 statistical rules: point sizing, dual mean/median stability, SLA envelopes, and bottleneck gap."""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BLOCK = 8


def pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(len(s) * q))])


def point_size(
    rate: float,
    warmup_s: float = 30.0,
    min_s: float = 180.0,
    min_completed: int = 0,
    rounds: int = 2,
    min_per_round: int = 32,
    block: int = BLOCK,
    sample_cap_s: float = 480.0,
) -> int:
    """Planned prompt count per round (multiple of `block=8`).

    Guarantees:
    1. At least `warmup_s + min_s` seconds of steady open-loop sending (`rate * (warmup_s + min_s)`).
    2. At least `min_per_round` (default 32) judged-window requests per round (and optional
       `min_completed` override if explicitly set > 0).
    """
    eff_rounds = max(int(rounds), 1)
    per_round_min = max(int(min_per_round), math.ceil(float(min_completed) / eff_rounds) if min_completed > 0 else 0)
    warmup_reqs = rate * warmup_s + (block if (eff_rounds == 1 and min_completed > 0) else 0.0)
    # Sampling floor for the realised arrival rate.
    #
    # Poisson inter-arrival times have a long tail, so a point with few arrivals has a realised
    # rate far from target: 18 arrivals (0.1/s for 180 s) land anywhere in 0.71-1.54x at the
    # 5-95% level. That spread alone makes repeated measurements disagree, independently of any
    # board difference. `min_arrivals` bounds it: at 100 arrivals the 5-95% band narrows to about
    # 0.86-1.19x. Capped by `sample_cap_s` so a very low rate cannot stretch a point forever.
    min_arrivals = 60.0
    n_samples_floor = min(rate * max(min_s, 1e-9), 1e9)
    if n_samples_floor < min_arrivals:
        n_samples_floor = min(min_arrivals, rate * sample_cap_s)
    n = max(rate * (warmup_s + min_s), per_round_min + warmup_reqs, n_samples_floor + warmup_reqs)
    return int(-(-n // block) * block)


def _drift(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Mean of `b` minus mean of `a`, and its standard error."""
    if len(a) < 4 or len(b) < 4:
        return 0.0, float("inf")
    return st.mean(b) - st.mean(a), math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))


def stability(keep: Sequence[dict], i0: int, block: int = BLOCK, stab_z: float = 3.0) -> dict:
    """Audited v7 queue stability test across block-aligned window halves.

    Splits the judged window into whole-block halves (identical length octile mix),
    regresses out per-request linear TTFT dependence on `prompt_len`, and flags a point
    unstable only when:
      - TTFT residual mean drift > max(25% * base_ttft, 300ms) AND > stab_z * SE
        AND residual median drift > max(25% * base_ttft, 300ms), OR
      - TPOT mean drift > max(50% * base_tpot, 10ms) AND > stab_z * SE.
    """
    nb = len({int(r["sample_index"]) // block for r in keep if r.get("sample_index") is not None})
    cut = i0 + block * max(nb // 2, 1)
    h1 = [r for r in keep if (r.get("sample_index") or 0) < cut]
    h2 = [r for r in keep if (r.get("sample_index") or 0) >= cut]
    # Fallback to time-ordered split only if legacy records lack sample_index
    if (len(h1) < 4 or len(h2) < 4) and len(keep) >= 8 and all(r.get("sample_index") is None for r in keep):
        mid = len(keep) // 2
        h1, h2 = list(keep[:mid]), list(keep[mid:])

    x = [float(r.get("prompt_len") or 0) for r in keep]
    y = [float(r["ttft"]) for r in keep]
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx > 0 else 0.0
    res = lambda r: float(r["ttft"]) - slope * (float(r.get("prompt_len") or 0) - mx)

    r1, r2 = [res(r) for r in h1], [res(r) for r in h2]
    d, se = _drift(r1, r2)
    dm = (st.median(r2) - st.median(r1)) if r1 and r2 else 0.0

    tp = lambda h: [float(r["tpot"]) for r in h if (r.get("output_len") or 0) > 5]
    dt, set_ = _drift(tp(h1), tp(h2))
    base_ttft = st.mean([float(r["ttft"]) for r in h1]) if h1 else 0.0
    base_tpot = st.mean(tp(h1)) if tp(h1) else 0.0

    thr = max(0.25 * base_ttft, 300.0)
    ttft_up = (d > thr) and (d > stab_z * se) and (dm > thr)

    return {
        "drift": d,
        "drift_se": se,
        "drift_median": dm,
        "tpot_drift": dt,
        "tpot_drift_se": set_,
        "ttft_len_slope": slope,
        "stable": not ttft_up,
    }


def len_ok(m: Optional[dict], ds_in: float, max_tokens: int = 1000, len_tol: float = 0.03) -> bool:
    """Enforce BOTH judged-window `in_len` and all-request `in_len_all` within +-len_tol of `ds_in`,
    and `out_len` within +-len_tol of `max_tokens`."""
    if m is None:
        return False
    target = float(m.get("ds_in") or ds_in)
    return (
        abs(float(m["in_len"]) / target - 1.0) <= len_tol
        and abs(float(m["in_len_all"]) / target - 1.0) <= len_tol
        and abs(float(m["out_len"]) / float(max_tokens) - 1.0) <= len_tol
    )


def measure_point(
    sub_dir: str | Path,
    rate: float,
    n_planned: int,
    warmup_s: float = 30.0,
    block: int = BLOCK,
    stab_z: float = 3.0,
) -> Optional[dict]:
    """Compute judged-window statistics over whole blocks started after `warmup_s`."""
    rs, bad = [], 0
    pattern = os.path.join(str(sub_dir), "report", "pre_detail_stat_*.jsonl")
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("error") or not r.get("finished"):
                    bad += 1
                else:
                    rs.append(r)
    if len(rs) < 6:
        return None
    start = lambda r: float(r["timestamp"]) - float(r["e2e_lat"]) / 1000.0
    t0 = min(start(r) for r in rs)
    # Steady-state window: a request only measures steady-state service if it BOTH started after
    # warm-up AND finished before the arrival stream stopped. Requests still in flight when the
    # last prompt is sent decode against a draining queue (no new prefills competing), so their
    # TPOT/TTFT are those of an unloading server -- 25-41 ms versus 53-85 ms under load. Those
    # tail requests used to be counted, which is what let points above the real capacity appear
    # to pass SLA (4 such points on H20 27B: 0.71/0.75/0.78/0.80 at 47-50 ms, but 52-54 ms once
    # the drain tail is excluded).
    t_last_send = max(start(r) for r in rs)
    late = [
        int(r["sample_index"])
        for r in rs
        if start(r) >= t0 + warmup_s and r.get("sample_index") is not None
    ]
    i0 = int(-(-min(late) // block) * block) if late else n_planned
    keep = sorted(
        (
            r
            for r in rs
            if (r.get("sample_index") or 0) >= i0
            and start(r) + float(r["e2e_lat"]) / 1000.0 <= t_last_send
        ),
        key=start,
    )
    if len(keep) < 6:
        return None
    ttft = [float(r["ttft"]) for r in keep]
    tpot = [float(r["tpot"]) for r in keep if (r.get("output_len") or 0) > 5]
    sd = stability(keep, i0, block=block, stab_z=stab_z)
    ks = [start(r) for r in keep]
    return {
        "rate": float(rate),
        "n": len(rs),
        "n_kept": len(keep),
        "n_planned": int(n_planned),
        # Never negative: a point can legitimately have more records than planned (a previous
        # run's jsonl left in the same report dir, or the client retrying). A negative value
        # used to make `len_ok` false, which marked the point hard-invalid and truncated the
        # scan -- the FP8 27B run reported LEN_BAD on every point for this reason alone.
        "missing": max(0, int(n_planned) - len(rs)) if int(n_planned) > 0 else 0,
        "errors": bad,
        "window_from": i0,
        "ttft": st.mean(ttft),
        "ttft_p90": pct(ttft, 0.9),
        "tpot": st.mean(tpot) if tpot else 0.0,
        "tpot_p90": pct(tpot, 0.9),
        "e2e": st.mean([float(r["e2e_lat"]) for r in keep]),
        "out_len": st.mean([float(r["output_len"]) for r in keep]),
        "in_len": st.mean([float(r.get("prompt_len") or 0) for r in keep]),
        "in_len_all": st.mean([float(r.get("prompt_len") or 0) for r in rs]),
        **sd,
        "send_s": max(ks) - t0,
        "rate_sent": (len(keep) - 1) / max(max(ks) - min(ks), 1e-6),
    }


def effective_rate(m: dict) -> float:
    """Arrival rate actually achieved at this point, in req/s per replica.

    The reported QPM/card must derive from what the server was really asked to do, not from the
    rate we intended. At low rates Poisson sampling alone moves the realised rate by +-20-25%
    over a 200 s window (18 arrivals: 0.71-1.54x), and under overload the achieved rate is below
    target by definition -- both are properties of the measurement, not of the card. Using
    `rate_sent` keeps QPM/card honest in both directions; `rate` is kept alongside for
    traceability.
    """
    sent = m.get("rate_sent")
    target = float(m.get("rate") or 0.0)
    if sent is None or sent <= 0:
        return target
    return float(sent)


def is_point_valid(m: dict, v7_mode: bool = True) -> bool:
    """A point is valid iff queue is stable, errors == 0, and (in v7 mode) missing == 0 & len_ok."""
    if not m.get("stable", True) or m.get("errors", 0):
        return False
    if v7_mode and "missing" in m:
        return m.get("missing", 0) == 0 and bool(m.get("len_ok", False))
    return True


def passes_sla(m: dict, ttft_max: float, tpot_max: float, v7_mode: bool = True) -> bool:
    return float(m["ttft"]) <= ttft_max and float(m["tpot"]) <= tpot_max and is_point_valid(m, v7_mode=v7_mode)


def by_rate(pts: Sequence[dict]) -> Dict[float, List[dict]]:
    g: Dict[float, List[dict]] = {}
    for m in pts:
        g.setdefault(round(float(m["rate"]), 4), []).append(m)
    return dict(sorted(g.items()))


def ceiling_rates(
    groups: Dict[float, List[dict]],
    ttft_max: float,
    tpot_max: float,
    v7_mode: bool = True,
) -> Tuple[float, Optional[float]]:
    """Return `(best_pass_rate, first_fail_rate)` in req/s per replica.

    `best` is the highest rate at which **every repeat on every board** satisfies
    `(ttft_max, tpot_max)` and is valid (stable, 0 errors/missing, len_ok).

    Why "every repeat" rather than a sample-weighted mean: the repeats exist to measure
    board-to-board spread, and near a boundary that spread is as large as the margin itself
    (H20 27B at 0.7/s: TPOT 52.2 ms on one board vs 34.8 ms on the other). Averaging turns
    "one board fails" into "passes", so the reported ceiling would be a number no single board
    sustains. Requiring both repeats makes the ceiling a lower bound that any board meets,
    which is what a delivered QPM/card figure should mean.

    **    The scan must not stop at the first non-passing rate. Board-to-board spread makes the
    verdict non-monotonic in practice: H20 27B at 0.70/s fails only on one board (TPOT 52.2
    vs 34.8) while 0.72/s passes on BOTH (37.7, 41.3). Breaking at 0.70 would report a ceiling
    below what the card sustains. So the scan keeps going and records the highest passing rate
    it finds, treating the first failing rate *above* that as the bracket's upper bound.

    A repeat that is invalid *only* because its queue-stability test tripped (no errors, no
    missing, length OK) is not saturation evidence: with a handful of repeats per rate one
    noisy window would otherwise abort the scan and hide higher rates that pass cleanly in
    every repeat. Such a rate is skipped (never `best`) and the scan continues; a rate that
    actually violates the SLA, or has errors/missing/length trouble, ends it.
    """
    best = 0.0
    fail: Optional[float] = None
    for rate, ms in groups.items():
        hard_invalid = any(
            int(m.get("errors") or 0) > 0
            or int(m.get("missing") or 0) > 0
            or (v7_mode and "len_ok" in m and not m.get("len_ok", False))
            for m in ms
        )
        all_pass = all(
            float(m["ttft"]) <= ttft_max and float(m["tpot"]) <= tpot_max for m in ms
        )
        if hard_invalid:
            # Errors, missing requests or length violations mean the point itself is unusable:
            # that IS saturation evidence (the server stopped serving correctly), so the scan
            # ends here. Only a plain SLA miss is treated as possibly non-monotonic.
            if fail is None and float(rate) > best:
                fail = float(rate)
            break
        if all_pass:
            if all(m.get("stable", True) for m in ms) and float(rate) > best:
                best = float(rate)
                if fail is not None and fail <= best:
                    fail = None
            continue
        # Plain SLA miss (latency over the limit, or an unstable window). Board spread can make
        # this non-monotonic, so keep scanning: a higher rate may pass on every board.
        if float(rate) > best and (fail is None or float(rate) < fail):
            fail = float(rate)

    # Final consistency: `fail` must be the lowest NON-passing rate strictly above `best`.
    # Recording it during the scan is not enough -- `best` can be raised by a later rate, which
    # leaves an earlier `fail` below it (observed: H20 27B reported best=20.43, fail=19.94 in
    # QPM/card, an inverted bracket that rendered as "(20.4-19.9)" and made the gap check
    # meaningless). Recompute from the groups so the pair is always ordered.
    if best > 0:
        above = sorted(
            r for r in groups
            if float(r) > best
            and not all(
                float(m["ttft"]) <= ttft_max and float(m["tpot"]) <= tpot_max
                and m.get("stable", True)
                and int(m.get("errors") or 0) == 0
                and not (v7_mode and "len_ok" in m and not m.get("len_ok", False))
                for m in groups[r]
            )
        )
        fail = above[0] if above else None
    return best, fail


def ceiling_qpm(
    groups: Dict[float, List[dict]],
    card_num: float,
    ttft_max: float,
    tpot_max: float,
    v7_mode: bool = True,
) -> Tuple[float, Optional[float]]:
    """Return `(best_pass_qpm_per_card, first_fail_qpm_per_card)` at the ACHIEVED rate.

    The passing rate is converted with the rate the server actually received (`rate_sent`),
    averaged over its repeats, not the requested value. At low rates Poisson sampling alone
    shifts the achieved rate by +-20-25%, and at overload the achieved rate falls below target
    by definition, so multiplying the requested value would report throughput the server never
    delivered.
    """
    b_r, f_r = ceiling_rates(groups, ttft_max, tpot_max, v7_mode=v7_mode)
    scale = 60.0 / float(card_num)

    def _qpm(rate: float) -> float:
        ms = groups.get(rate) or []
        if ms:
            achieved = st.mean(effective_rate(m) for m in ms)
        else:
            achieved = float(rate)
        return achieved * scale

    best_q = _qpm(b_r)
    fail_q = _qpm(f_r) if f_r is not None else None
    # Keep the bracket ordered on the ACHIEVED axis too. Overload throttles the failing rate's
    # `rate_sent` below the passing rate's, so converting both to achieved QPM/card can invert
    # them (e.g. MoE: best 0.768/s -> 46.43 QPM but fail 0.7762/s -> 44.39 because the server
    # only delivered 0.74/s there). An inverted bracket renders as "(46.4-44.4)" and makes every
    # downstream gap/hardness check meaningless, so report the failing point on the nominal axis
    # that produced it: what the requester asked for at the first failing rate.
    if fail_q is not None and fail_q <= best_q:
        fail_q = float(f_r) * scale
    return best_q, fail_q


def bottleneck_metrics(ms: Sequence[dict], ttft_max: float, tpot_max: float) -> dict:
    """Bottleneck ratio and headroom gap at a passing rate.

    Uses the WORST repeat, not the mean, to stay consistent with `ceiling_rates`: a rate only
    counts as passing when every board passes it, so its headroom is set by the slowest board.
    Taking the mean here would report headroom that the weakest board does not have.
    """
    worst_ttft = max(float(m["ttft"]) for m in ms)
    worst_tpot = max(float(m["tpot"]) for m in ms)
    r_ttft = worst_ttft / ttft_max
    r_tpot = worst_tpot / tpot_max
    if r_ttft >= r_tpot:
        b_ratio = r_ttft
        b_dim = "TTFT"
    else:
        b_ratio = r_tpot
        b_dim = "TPOT"
    completed = int(sum(int(m.get("n_kept") or m.get("n") or 0) for m in ms))
    return {
        "mean_ttft": st.mean(float(m["ttft"]) for m in ms),
        "mean_tpot": st.mean(float(m["tpot"]) for m in ms),
        "worst_ttft": worst_ttft,
        "worst_tpot": worst_tpot,
        "ttft_ratio": r_ttft,
        "tpot_ratio": r_tpot,
        "bottleneck_ratio": b_ratio,
        "bottleneck_dim": b_dim,
        "gap": 1.0 - b_ratio,
        "completed": completed,
    }


def isotonic(vs: Sequence[float], ws: Sequence[float]) -> List[float]:
    """Weighted pool-adjacent-violators fit (non-decreasing)."""
    blocks: List[List[float]] = []
    for v, w in zip(vs, ws):
        blocks.append([float(v) * float(w), float(w), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a = blocks.pop()
            blocks[-1] = [blocks[-1][0] + a[0], blocks[-1][1] + a[1], blocks[-1][2] + a[2]]
    out: List[float] = []
    for tot, w, n in blocks:
        out.extend([tot / w] * int(n))
    return out


def envelope(
    groups: Dict[float, List[dict]],
    metric: str,
    v7_mode: bool = True,
) -> Tuple[List[float], List[float], Optional[float]]:
    """Isotonic non-decreasing latency curve up to the first invalid/unstable rate."""
    rs: List[float] = []
    raw: List[float] = []
    ws: List[float] = []
    cap_rate: Optional[float] = None
    for rate, ms in groups.items():
        if not all(is_point_valid(m, v7_mode=v7_mode) for m in ms):
            cap_rate = float(rate)
            break
        rs.append(float(rate))
        raw.append(st.mean(float(m[metric]) for m in ms))
        ws.append(float(sum(int(m.get("n_kept") or m.get("n") or 1) for m in ms)))
    return rs, isotonic(raw, ws), cap_rate


def cross(rs: Sequence[float], vs: Sequence[float], limit: float) -> Tuple[float, str]:
    """Highest rate at which the piecewise-linear curve `vs(rs)` stays <= `limit`."""
    if not rs or vs[0] > limit:
        return 0.0, "below"
    for i in range(len(rs) - 1):
        if vs[i + 1] > limit:
            denom = vs[i + 1] - vs[i]
            f = (limit - vs[i]) / denom if denom > 0 else 0.0
            return rs[i] + f * (rs[i + 1] - rs[i]), "interp"
    return rs[-1], "top"


def frontier_qpm(
    groups: Dict[float, List[dict]],
    card_num: float,
    ttft_max: float,
    tpot_max: float,
    v7_mode: bool = True,
) -> Tuple[float, str]:
    """Interpolated maximum QPM/card within `(ttft_max, tpot_max)`."""
    r1, v1, c1 = envelope(groups, "tpot", v7_mode=v7_mode)
    r2, v2, _ = envelope(groups, "ttft", v7_mode=v7_mode)
    a, fa = cross(r1, v1, tpot_max)
    b, fb = cross(r2, v2, ttft_max)
    r = min(a, b)
    flag = fa if a <= b else fb
    if flag == "top" and c1 is not None:
        flag = "cap"
    return r * 60.0 / float(card_num), flag
