"""Self-contained inline-SVG HTML SLA report generator (v7 methodology)."""

from __future__ import annotations

import glob
import html
import json
import math
import os
import re
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .config import (
    DEFAULT_PALETTE,
    BenchConfig,
    ExternalBaselineSpec,
    ModelTPSpec,
    SLALevel,
    strip_tp_suffix,
)
from .locust_client.model_config import PAD_DELIM
from .stats import (
    bottleneck_metrics,
    by_rate,
    ceiling_qpm,
    ceiling_rates,
    envelope,
    frontier_qpm,
    is_point_valid,
    passes_sla,
)
from .sweep import _resolve_baseline_root, discover_round_roots, load_spec_points

W, H = 560, 330
PAD_L, PAD_R, PAD_T, PAD_B = 62, 18, 34, 46
OP_KINDS = [
    ("满足 ① 与 ②", "diamond"),
    ("仅满足 ①", "circle"),
    ("①② 都不满足", "triangle"),
    ("排队不收敛", "cross"),
]

def _compute_eff_lengths(recs: Sequence[dict], tok_url: str = "", model: str = "") -> List[Optional[int]]:
    """Real answer tokens preceding `PAD_DELIM` for each record, counted server-side.

    Prefers the `eff_len` the client recorded at run time (already counted by the server's
    `/tokenize`). Older runs without it are recounted via `tok_url` when given; otherwise
    they stay `None` (shown as 'missing'). No local tokenizer and no length heuristics.
    """
    out: List[Optional[int]] = [None] * len(recs)
    todo: List[int] = []
    prefixes: List[str] = []
    for idx, r in enumerate(recs):
        if "eff_len" in r:
            out[idx] = None if r["eff_len"] is None else int(r["eff_len"])
            continue
        t = r.get("generated_text") or ""
        cut = t.find(PAD_DELIM)
        if cut < 0:
            continue
        todo.append(idx)
        prefixes.append(t[:cut])
    if todo and tok_url:
        from .locust_client.remote_tok import RemoteTokenizer, derive_tokenize_url

        url = tok_url if tok_url.rstrip("/").endswith("/tokenize") else derive_tokenize_url(tok_url)
        try:
            for i_pos, n in zip(todo, RemoteTokenizer(url, model=model).count_texts(prefixes)):
                out[i_pos] = int(n)
        except Exception as exc:  # server unreachable at report time: leave as missing
            print(f"[report] server /tokenize unavailable ({exc}); {len(todo)} eff_len left missing", flush=True)
    return out


def load_clean_spec_points(
    results_dir: str | Path,
    spec: ModelTPSpec,
    stab_z: float = 3.0,
    restabilize: bool = True,
) -> List[dict]:
    """Load points for `spec`, preferring raw `round*/<model>/curve.json` over exported `light/*_burst.jsonl`."""
    from dataclasses import replace
    from .light import find_light, read_light

    base = Path(results_dir)
    eff_spec = spec
    if not spec.roots and base.is_dir():
        round_roots = [
            str(child.resolve())
            for child in sorted(base.iterdir())
            if child.is_dir() and child.name.startswith("round") and (child / spec.model / "curve.json").is_file()
        ]
        if round_roots:
            eff_spec = replace(spec, roots=round_roots)

    pts = load_spec_points(base, eff_spec, stab_z=stab_z, restabilize=restabilize)
    light_raw_map: Dict[Tuple[float, Optional[int]], dict] = {}
    if any("_raw" not in m for m in pts) and any("_raw" in m for m in pts):
        for m in pts:
            if "_raw" in m and m["_raw"]:
                light_raw_map[(round(float(m["rate"]), 4), m.get("port"))] = m["_raw"]
        pts = [m for m in pts if "_raw" not in m]

    # When `round*/curve.json` was pulled without raw `pre_detail_stat_*.jsonl`, backfill `_raw`
    # summaries (acc, prompt_len, eff_len) from any co-located `light/*_burst.jsonl` without duplicating points.
    if pts and any("_raw" not in m for m in pts):
        if not light_raw_map:
            cand_dirs: List[Path] = []
            if spec.roots:
                for r in spec.roots:
                    for c in _resolve_baseline_root(r, base):
                        if not c.name.startswith("round") and c not in cand_dirs:
                            cand_dirs.append(c)
            if base.is_dir() and base not in cand_dirs:
                cand_dirs.append(base)
            for cd in cand_dirs:
                lf = find_light(cd, spec.model)
                if lf is not None:
                    for lm in read_light(lf)[1]:
                        if lm.get("_raw"):
                            light_raw_map[(round(float(lm["rate"]), 4), lm.get("port"))] = lm["_raw"]
        if light_raw_map:
            for m in pts:
                if "_raw" in m:
                    continue
                src = m.get("src", "")
                rpath = (Path(src) if Path(src).is_absolute() else (base / src)) if src else base
                d_pt = rpath / spec.model / ("r%07.4f" % float(m["rate"]))
                sub = f"p{m['port']}" if (spec.v7_mode and m.get("port")) else "*"
                if not glob.glob(str(d_pt / sub / "report" / "pre_detail_stat_*.jsonl")):
                    lraw = light_raw_map.get((round(float(m["rate"]), 4), m.get("port")))
                    if lraw:
                        m["_raw"] = lraw
    return pts


def auto_discover_specs_from_dir(
    results_dir: str | Path,
    card: str = "GPU",
) -> List[ModelTPSpec]:
    """Auto-discover `(model, tp, cards)` specs from `<results_dir>/round*/<model>/curve.json` (preferred) or `*_burst.jsonl`."""
    base = Path(results_dir)
    if not base.exists():
        return []

    round_dirs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("round"))

    from .light import infer_precision, iter_light_files, read_light

    model_rounds: Dict[str, List[str]] = {}
    sources: List[Tuple[str, Path, bool]] = []
    for rd in round_dirs:
        for cj in sorted(rd.glob("*/curve.json")):
            mdir_name = cj.parent.name
            model_rounds.setdefault(mdir_name, []).append(str(rd.resolve()))
            sources.append((mdir_name, cj, False))
    for cj in sorted(base.glob("*/curve.json")):
        sources.append((cj.parent.name, cj, False))
    for lf in iter_light_files(base):
        sources.append(("", lf, True))

    found: Dict[str, dict] = {}
    for mdir, cj, is_light in sources:
        try:
            if is_light:
                meta, lpts = read_light(cj)
                meta = dict(meta, points=lpts)
                mdir = str(meta.get("model") or cj.stem)
            else:
                meta = json.loads(cj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if mdir in found:
            continue
        if not meta.get("points"):
            continue
        # `cards` (current) / `card_num` (older runs) / `dies / die_per_card` (oldest runs)
        cards = float(
            meta.get("cards")
            or meta.get("card_num")
            or (float(meta.get("dies") or 1) / float(meta.get("die_per_card") or 1))
        )
        m_tp = re.search(r"-tp(\d+)$", mdir)
        tp = int(meta.get("tp") or (int(m_tp.group(1)) if m_tp else max(1, round(cards))))
        canon = str(meta.get("served_model_name") or strip_tp_suffix(mdir))
        prec = infer_precision(
            mdir,
            canon,
            str(meta.get("precision") or meta.get("dtype") or ""),
            cj.name if is_light else "",
        )
        found[mdir] = {
            "model": mdir,
            "canon": canon,
            "tp": tp,
            "cards": cards,
            "card": str(meta.get("card") or card),
            "precision": prec,
            "roots": model_rounds.get(mdir, []),
        }

    sorted_items = sorted(found.values(), key=lambda x: (x["canon"], x["cards"], x["model"]))
    specs: List[ModelTPSpec] = []
    for idx, it in enumerate(sorted_items):
        canon = it["canon"]
        specs.append(
            ModelTPSpec(
                key=f"auto_{idx}_{it['model']}",
                card=it["card"],
                model=it["model"],
                served_model_name=canon,
                model_short=canon,
                tp=it["tp"],
                cards=it["cards"],
                roots=list(it["roots"]),
                color=DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)],
                precision=it["precision"],
                v7_mode=True,
            )
        )
    return specs


def nice_ticks(lo: float, hi: float, n: int = 5) -> Tuple[List[float], float]:
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t = math.floor(lo / step) * step
    out: List[float] = []
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 10))
        t += step
    return out, step


def glyph(shape: str, x: float, y: float, fill: str, stroke: str, tip: str, r: float = 2.6) -> str:
    t = f"<title>{html.escape(tip)}</title>" if tip else ""
    a = f'fill="{fill}" stroke="{stroke}" stroke-width="1"'
    if shape == "triangle":
        return f'<path d="M{x:.1f} {y - r * 1.25:.1f}L{x + r * 1.15:.1f} {y + r * 0.8:.1f}L{x - r * 1.15:.1f} {y + r * 0.8:.1f}Z" {a}>{t}</path>'
    if shape == "square":
        return f'<rect x="{x - r * 0.9:.1f}" y="{y - r * 0.9:.1f}" width="{r * 1.8:.1f}" height="{r * 1.8:.1f}" {a}>{t}</rect>'
    if shape == "diamond":
        return (
            f'<path d="M{x:.1f} {y - r * 1.3:.1f}L{x + r * 1.1:.1f} {y:.1f}'
            f'L{x:.1f} {y + r * 1.3:.1f}L{x - r * 1.1:.1f} {y:.1f}Z" {a}>{t}</path>'
        )
    if shape == "cross":
        return (
            f'<path d="M{x - r:.1f} {y - r:.1f}L{x + r:.1f} {y + r:.1f}'
            f'M{x - r:.1f} {y + r:.1f}L{x + r:.1f} {y - r:.1f}" stroke="{stroke}" stroke-width="1.8">{t}</path>'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" {a}>{t}</circle>'


def zoom_lo(vs: Sequence[float]) -> float:
    lo, hi = min(vs), max(vs)
    _, step = nice_ticks(lo, hi)
    return max(0.0, math.floor((lo - 0.05 * (hi - lo)) / step) * step)


def broken_axis(lo: float, brk: float, vmax: float, p0: float, p1: float, frac: float = 0.8):
    """Linear scale from `lo` to `brk` over `frac` of axis, `log10` above `brk`."""
    lin, _ = nice_ticks(lo, brk)
    lo = min(lo, lin[0])
    top = vmax * 1.12
    lb, lt = math.log10(brk), math.log10(top)

    def f(v: float) -> float:
        if v <= brk:
            return p0 + (v - lo) / max(brk - lo, 1e-9) * (p1 - p0) * frac
        return p0 + (p1 - p0) * (frac + (1.0 - frac) * (math.log10(min(v, top)) - lb) / max(lt - lb, 1e-9))

    ticks = [t for t in lin if t <= brk] + [
        m * (10**e)
        for e in range(int(math.floor(lb)), int(math.ceil(lt)) + 1)
        for m in (1, 2, 5)
        if brk * 1.3 < m * (10**e) <= top
    ]
    if ticks[-1] <= brk:
        ticks.append(float(f"{vmax:.2g}"))
    return f, ticks


def _as_lines(v) -> list:
    return [] if not v else (list(v) if isinstance(v, list) else [v])


def render_svg_chart(
    title: str,
    series: list,
    ylabel: str,
    xlabel: str,
    logy: bool = False,
    hline=None,
    vline=None,
    step: bool = False,
    dots: bool = True,
    ymax: Optional[float] = None,
    labels: bool = True,
    marks: Sequence = (),
    ordered: bool = False,
    zoom: bool = False,
    mark_legend: Optional[Sequence] = None,
    xbrk: Optional[float] = None,
    ybrk: Optional[float] = None,
) -> str:
    xs = [p[0] for it in series for p in it[2]] + [p[0] for m in marks for p in m[1]]
    ys = [p[1] for it in series for p in it[2]] + [p[1] for m in marks for p in m[1]]
    hlines, vlines = _as_lines(hline), _as_lines(vline)
    ys = ys + [l[0] for l in hlines]
    xs = xs + [l[0] for l in vlines]
    if not xs or not ys:
        return ""

    n_leg = (
        len([it for it in series if not it[0].startswith("_")])
        + (1 if any(len(it) > 3 and it[3] for it in series) else 0)
        + (len(mark_legend) if mark_legend else (1 if marks else 0))
    )
    top = PAD_T + 13 * ((n_leg + 1) // 2) + 6
    clip = ymax if ymax is not None else None
    if clip is not None:
        ys = [min(y, clip) for y in ys]

    if logy:
        ymin = max(min(y for y in ys if y > 0), 1e-9)
        lo, hi = math.log10(ymin) - 0.08, math.log10(max(ys)) + 0.08
        fy = lambda v: H - PAD_B - (math.log10(max(v, ymin)) - lo) / max(hi - lo, 1e-9) * (H - top - PAD_B)
        ticks = [
            m * (10**e)
            for e in range(math.floor(lo), math.ceil(hi) + 1)
            for m in (1, 2, 5)
            if ymin * 0.9 <= m * (10**e) <= max(ys) * 1.15
        ]
    elif ybrk and max(ys) > ybrk * 1.05:
        under_y = [y for y in ys if y <= ybrk] or [0.0]
        fy, ticks = broken_axis(zoom_lo(under_y) if zoom else 0.0, ybrk, max(ys), H - PAD_B, top)
    else:
        ybrk = None
        lo = zoom_lo(ys) if zoom else 0.0
        ticks, ystep = nice_ticks(lo, max(ys) + (max(ys) - lo) * 0.08)
        lo = min(lo, ticks[0])
        hi = ticks[-1] + (ystep if ticks[-1] < max(ys) * 1.05 else 0)
        fy = lambda v: H - PAD_B - (min(v, hi) - lo) / max(hi - lo, 1e-9) * (H - top - PAD_B)

    if xbrk and max(xs) > xbrk * 1.05:
        under_x = [x for x in xs if x <= xbrk] or [0.0]
        fx, xticks = broken_axis(zoom_lo(under_x) if zoom else 0.0, xbrk, max(xs), PAD_L, W - PAD_R)
    else:
        xbrk = None
        xlo = zoom_lo(xs) if zoom else 0.0
        xticks, xstep = nice_ticks(xlo, max(xs) + (max(xs) - xlo) * 0.04)
        xlo = min(xlo, xticks[0])
        xhi = xticks[-1] + (xstep if xticks[-1] < max(xs) * 1.02 else 0)
        fx = lambda v: PAD_L + (v - xlo) / max(xhi - xlo, 1e-9) * (W - PAD_L - PAD_R)

    s = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif">'
    ]
    s.append(
        f'<text x="{W // 2}" y="19" text-anchor="middle" font-size="13" font-weight="700" '
        f'fill="#1F2937">{html.escape(title)}</text>'
    )
    for t in ticks:
        y = fy(t)
        if not (top - 2 <= y <= H - PAD_B + 2):
            continue
        s.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="#E5E7EB"/>')
        s.append(f'<text x="{PAD_L - 6}" y="{y + 3.5:.1f}" text-anchor="end" font-size="9.5" fill="#6B7280">{t:g}</text>')
    for x in xticks:
        s.append(f'<text x="{fx(x):.1f}" y="{H - PAD_B + 15}" text-anchor="middle" font-size="9.5" fill="#6B7280">{x:g}</text>')
    s.append(f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}" stroke="#9CA3AF" stroke-width="1.2"/>')
    s.append(f'<line x1="{PAD_L}" y1="{top}" x2="{PAD_L}" y2="{H - PAD_B}" stroke="#9CA3AF" stroke-width="1.2"/>')
    s.append(f'<text x="{(W + PAD_L) / 2:.1f}" y="{H - 8}" text-anchor="middle" font-size="10.5" fill="#374151">{html.escape(xlabel)}</text>')
    ymid = (H - PAD_B + top) // 2
    s.append(
        f'<text x="13" y="{ymid}" font-size="10.5" fill="#374151" transform="rotate(-90 13 {ymid})" '
        f'text-anchor="middle">{html.escape(ylabel)}</text>'
    )
    if ybrk:
        c = fy(ybrk)
        s.append(f'<rect x="{PAD_L}" y="{top}" width="{W - PAD_L - PAD_R}" height="{c - top:.1f}" fill="#F3F4F6" opacity="0.7"/>')
        s.append(f'<text x="{PAD_L + 4}" y="{c - 4:.1f}" font-size="8.5" fill="#6B7280">≥ {ybrk:g} 非等比（对数）</text>')
    if xbrk:
        c = fx(xbrk)
        s.append(f'<rect x="{c:.1f}" y="{top}" width="{W - PAD_R - c:.1f}" height="{H - PAD_B - top}" fill="#F3F4F6" opacity="0.7"/>')
        s.append(f'<text x="{c + 3:.1f}" y="{H - PAD_B - 5}" font-size="8.5" fill="#6B7280">≥ {xbrk:g} 非等比</text>')

    for line, horiz in [(l, True) for l in hlines] + [(l, False) for l in vlines]:
        col = line[2] if len(line) > 2 else "#DC2626"
        if horiz:
            c = fy(line[0])
            s.append(
                f'<line x1="{PAD_L}" y1="{c:.1f}" x2="{W - PAD_R}" y2="{c:.1f}" stroke="{col}" '
                f'stroke-width="1.6" stroke-dasharray="7 4"/>'
            )
            s.append(
                f'<text x="{W - PAD_R - 4}" y="{c - 5:.1f}" text-anchor="end" font-size="9.5" '
                f'font-weight="700" fill="{col}">{html.escape(line[1])}</text>'
            )
        else:
            c = fx(line[0])
            s.append(
                f'<line x1="{c:.1f}" y1="{top}" x2="{c:.1f}" y2="{H - PAD_B}" stroke="{col}" '
                f'stroke-width="1.6" stroke-dasharray="7 4"/>'
            )
            s.append(
                f'<text x="{c + 4:.1f}" y="{top + 10}" font-size="9.5" font-weight="700" '
                f'fill="{col}">{html.escape(line[1])}</text>'
            )

    for item in series:
        label, color, pts = item[:3]
        dash = len(item) > 3 and item[3]
        pts = list(pts) if ordered else sorted(pts)
        if not pts:
            continue
        if step:
            d_parts, prev = [], None
            for i, (x, y) in enumerate(pts):
                if i == 0:
                    d_parts.append(f"M{fx(x):.1f} {fy(y):.1f}")
                else:
                    d_parts.append(f"L{fx(x):.1f} {fy(prev):.1f} L{fx(x):.1f} {fy(y):.1f}")
                prev = y
            d_str = " ".join(d_parts)
        else:
            d_str = " ".join(("M" if i == 0 else "L") + f"{fx(x):.1f} {fy(y):.1f}" for i, (x, y) in enumerate(pts))
        dash_attr = ' stroke-dasharray="4 3" opacity="0.7"' if dash else ""
        s.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')
        last = None
        for x, y in (pts if dots else []):
            clipped = clip is not None and y > clip
            px, py = fx(x), fy(y)
            r_dot = "4" if clipped else "3"
            s.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r_dot}" fill="{color}">'
                f"<title>{html.escape(label)} x={round(x, 2):g} y={round(y, 1):g}</title></circle>"
            )
            if (labels or clipped) and (last is None or math.hypot(px - last[0], py - last[1]) > 18):
                ly = py - 7 if not clipped else py + 13
                pref = "↑" if clipped else ""
                s.append(
                    f'<text x="{px:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="8.5" '
                    f'fill="{color}">{pref}{round(y, 1):g}</text>'
                )
                last = (px, py)

    placed: List[Tuple[float, float]] = []
    for mk in marks:
        color, pts, hollow = mk[:3]
        names = mk[3] if len(mk) > 3 else [None] * len(pts)
        shape = mk[4] if len(mk) > 4 else "circle"
        for (x, y), name in sorted(zip(pts, names)):
            px, py = fx(x), fy(y)
            s.append(glyph(shape, px, py, "#FFFFFF" if hollow else color, color, f"x={round(x, 1):g} y={round(y, 1):g}"))
            for dx, dy in ((0, -7), (0, 14), (17, 3), (-17, 3), (14, -9), (-14, -9), (14, 13), (-14, 13)):
                lx, ly = px + dx, py + dy
                if all(abs(lx - a) > 19 or abs(ly - b) > 10 for a, b in placed):
                    break
            else:
                lx = ly = None
                if name is not None:
                    lx, ly = px, py - 7
            if lx is not None and ly is not None:
                txt = name if name is not None else f"{round(y, 1):g}"
                s.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="8.5" fill="{color}">{html.escape(str(txt))}</text>')
                placed.append((lx, ly))
            placed.append((px, py - 3))

    shown = [it for it in series if not it[0].startswith("_")]
    if any(len(it) > 3 and it[3] for it in series):
        shown = shown + [("虚线 = 饱和平台（再高一档排队不收敛）", "#9CA3AF", [], True)]
    if mark_legend:
        shown = shown + [(lbl, "#374151", [], False, shp) for lbl, shp in mark_legend]
    elif marks:
        shown = shown + [("● 实测点（稳定）  ○ 首个不稳定点", "#6B7280", [], False)]
    for i, item in enumerate(shown):
        label, color = item[0], item[1]
        ly = PAD_T + 2 + (i // 2) * 13
        lx = PAD_L + (i % 2) * ((W - PAD_L - PAD_R) // 2)
        if len(item) > 4:
            s.append(glyph(item[4], lx + 18, ly, color, color, "", r=3.4))
        else:
            d_attr = ' stroke-dasharray="4 3"' if len(item) > 3 and item[3] else ""
            s.append(f'<line x1="{lx + 8}" y1="{ly}" x2="{lx + 28}" y2="{ly}" stroke="{color}" stroke-width="2"{d_attr}/>')
        s.append(f'<text x="{lx + 32}" y="{ly + 3.5:.1f}" font-size="9" fill="#374151">{html.escape(label)}</text>')

    s.append("</svg>")
    return "".join(s)


def _raw_point_stats(base_dir: Path, m: dict, model_dir: str, v7_mode: bool, tok_url: str = "", model: str = "") -> Optional[dict]:
    """Read raw `pre_detail_stat_*.jsonl` for one point to compute accuracy, prompt lengths, and real-out lengths.

    Points loaded from a light `*_burst.jsonl` already carry these summaries (`_raw`); no raw records needed.
    """
    if "_raw" in m:
        return dict(m["_raw"], recs=[]) if m["_raw"] else None
    src = m.get("src", "")
    rpath = (Path(src) if Path(src).is_absolute() else (base_dir / src)) if src else base_dir
    d = rpath / model_dir / ("r%07.4f" % float(m["rate"]))
    sub = f"p{m['port']}" if (v7_mode and m.get("port")) else "*"
    rs = []
    for fpath in sorted(glob.glob(str(d / sub / "report" / "pre_detail_stat_*.jsonl"))):
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("finished") and not r.get("error"):
                    if v7_mode and "window_from" in m and (r.get("sample_index") or 0) < m.get("window_from", 0):
                        continue
                    rs.append(r)
    if not rs:
        return None
    sc = [float(r["score"]) for r in rs if r.get("score") is not None]
    pl = sorted(int(r.get("prompt_len") or 0) for r in rs)
    ol = sorted(int(r.get("output_len") or 0) for r in rs)
    # Records with neither a recorded `eff_len` nor `generated_text` (e.g. the results/h20_v7 baseline) carry
    # no information about the delimiter: they are "unknown", not "delimiter missing".
    unknown = [("eff_len" not in r and not r.get("generated_text")) for r in rs]
    effs = _compute_eff_lengths(rs, tok_url, model)
    for r, el_v, unk in zip(rs, effs, unknown):
        r["eff_len"] = el_v
        r["_eff_unknown"] = unk
    el = sorted(x for x in effs if x is not None)
    return {
        "eff_avg": st.mean(el) if el else 0.0,
        "eff_min": el[0] if el else 0,
        "eff_max": el[-1] if el else 0,
        "eff_miss": sum(1 for e, u in zip(effs, unknown) if e is None and not u),
        "eff_unknown": sum(unknown),
        "n": len(rs),
        "in_min": pl[0],
        "in_avg": st.mean(pl),
        "in_p50": pl[len(pl) // 2],
        "in_max": pl[-1],
        "out_avg": st.mean(ol),
        "out_min": ol[0],
        "out_max": ol[-1],
        "acc": st.mean(sc) * 100.0 if sc else None,
        "n_scored": len(sc),
        "recs": rs,
    }


def _short_tp_label(entry: dict, show_card: bool = False, show_prec: bool = False) -> str:
    c_str = f"{entry['cards']:g}"
    parts: List[str] = []
    if show_card:
        parts.append(str(entry["card"]))
    if show_prec:
        parts.append(str(entry["precision"]))
    parts.append(f"TP={entry['tp']} · {c_str} card")
    return " · ".join(parts)


def _resolve_spec_acc_stats(
    results_dir: Path,
    roots: Sequence[str],
    model_dir: str,
    tok_url: str = "",
    model: str = "",
) -> Optional[dict]:
    """Locate fixed `acc/<model_dir>` inside `roots` (if configured) or `results_dir`."""
    seen: set[Path] = set()
    cands: List[Path] = []
    if roots:
        for r in roots:
            for c in _resolve_baseline_root(r, results_dir):
                base_c = c.parent if c.name.startswith("round") else c
                rc = base_c.resolve()
                if rc not in seen:
                    seen.add(rc)
                    cands.append(base_c)
    else:
        cands.append(results_dir)
    for cand in cands:
        fx = _fixed_acc_stats(cand, model_dir, tok_url, model)
        if fx is not None:
            return fx
    return None


def _fixed_acc_stats(results_dir: Path, model_dir: str, tok_url: str = "", model: str = "") -> Optional[dict]:
    """Accuracy from the fixed `acc/` pass, if one was run: same prompts for every card.

    Excludes the leading warm-up block (`sample_index < prompts_planned - prompts_scored`) when
    reading raw `pre_detail_stat_*.jsonl`, and falls back to `item_scores.json` or
    `acc_summary.json` when lightweight baseline directories omit raw per-request JSONLs.
    """
    acc_dir = Path(results_dir) / "acc" / model_dir
    if not acc_dir.is_dir():
        return None
    summary_path = acc_dir / "acc_summary.json"
    meta: dict = {}
    if summary_path.is_file():
        try:
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}

    n_planned_meta = int(meta.get("prompts_planned") or 0)
    n_scored_meta = int(meta.get("prompts_scored") or 0)
    warmup_burst = max(0, n_planned_meta - n_scored_meta) if (n_planned_meta > 0 and n_scored_meta > 0) else 0

    by_q: dict[tuple, float] = {}
    n_recs = 0
    for fpath in sorted(glob.glob(str(acc_dir / "report" / "pre_detail_stat_*.jsonl"))):
        with open(fpath, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not r.get("finished") or r.get("error") or r.get("score") is None:
                    continue
                si = r.get("sample_index")
                if warmup_burst > 0 and isinstance(si, int):
                    if si < warmup_burst or (n_planned_meta > 0 and si >= n_planned_meta):
                        continue
                n_recs += 1
                key = (si,) if (warmup_burst > 0 and isinstance(si, int)) else (str(r.get("gold_ans") or ""), int(r.get("prompt_len") or 0))
                by_q.setdefault(key, float(r["score"]))

    if by_q:
        scored = list(by_q.values())
        acc_val = st.mean(scored) * 100.0
        n_scored = len(scored)
        return {
            "acc": float(acc_val),
            "n_scored": n_scored,
            "n": n_recs,
            "prompts": n_scored_meta or n_planned_meta or n_scored,
            "rate": meta.get("rate"),
            "mode": meta.get("mode", "open_loop_poisson_at_sla1_ceiling"),
            "seed": meta.get("seed"),
            "in_len_mean": meta.get("in_len_mean"),
        }

    item_scores_path = acc_dir / "item_scores.json"
    if item_scores_path.is_file():
        try:
            items = json.loads(item_scores_path.read_text(encoding="utf-8"))
            vals = [float(x["score"] if isinstance(x, dict) else x) for x in items if (x.get("score") is not None if isinstance(x, dict) else x is not None)]
            if vals:
                return {
                    "acc": float(st.mean(vals) * 100.0),
                    "n_scored": len(vals),
                    "n": int(meta.get("records_judged") or len(vals)),
                    "prompts": n_scored_meta or len(vals),
                    "rate": meta.get("rate"),
                    "mode": meta.get("mode", "open_loop_poisson_at_sla1_ceiling"),
                    "seed": meta.get("seed"),
                    "in_len_mean": meta.get("in_len_mean"),
                }
        except (ValueError, TypeError, KeyError):
            pass

    if meta.get("acc") is not None:
        n_judged = int(meta.get("records_judged") or n_scored_meta or 0)
        return {
            "acc": float(meta["acc"]),
            "n_scored": n_judged,
            "n": int(meta.get("records") or n_judged),
            "prompts": n_scored_meta or n_planned_meta or n_judged,
            "rate": meta.get("rate"),
            "mode": meta.get("mode", "open_loop_poisson_at_sla1_ceiling"),
            "seed": meta.get("seed"),
            "in_len_mean": meta.get("in_len_mean"),
        }

    return None


def _best_for_keys(data: Dict[str, dict], ks: Sequence[str], s_ttft: float, s_tpot: float):
    best_pick = None
    for k in ks:
        d = data[k]
        best, fail = ceiling_qpm(d["groups"], d["cards"], s_ttft, s_tpot, v7_mode=d["v7_mode"])
        est = frontier_qpm(d["groups"], d["cards"], s_ttft, s_tpot, v7_mode=d["v7_mode"])[0] if best > 0 else 0.0
        est = best if (fail is not None and est >= fail) else max(best, est)
        cand = (est, best, fail, k)
        if best_pick is None or (cand[0], cand[1]) > (best_pick[0], best_pick[1]):
            best_pick = cand
    return best_pick


def generate_html_report(cfg: BenchConfig, out_path: Optional[str | Path] = None) -> Path:
    """Build self-contained inline-SVG HTML SLA report and write to `out_path` (or `cfg.report_out`)."""
    results_dir = Path(cfg.results_dir)
    specs = list(cfg.specs)
    if not specs:
        specs = auto_discover_specs_from_dir(results_dir, card=cfg.card)
    if not specs:
        raise ValueError(f"No completed results (curve.json or *_burst.jsonl) found under {results_dir}")

    slas = cfg.slas
    sla1 = slas[0]
    sla2 = slas[1] if len(slas) > 1 else SLALevel("②", 1000.0, 20.0, "#EA580C")

    from .light import infer_precision

    data: Dict[str, dict] = {}
    primary_keys: List[str] = []

    for sp in specs:
        pts = load_clean_spec_points(results_dir, sp, stab_z=cfg.sweep.stab_z, restabilize=True)
        if len(pts) < 1:
            continue
        prec = infer_precision(sp.model, sp.served_model_name, sp.precision)
        data[sp.key] = {
            "lbl": sp.display_label,
            "col": sp.color,
            "tp": sp.tp,
            "cards": sp.card_num,
            "card": sp.card,
            "precision": prec,
            "roots": list(sp.roots),
            "name": sp.model,
            "canon_model": sp.served_model_name,
            "model_short": sp.model_short,
            "pts": pts,
            "groups": by_rate(pts),
            "v7_mode": sp.v7_mode,
            "tokenizer_url": sp.tokenizer_url,
            "base_dir": results_dir,
            "port_labels": sp.port_labels,
            "how_note": (
                f"每副本占 {sp.card_num:g} card，"
                f"{'v7' if sp.v7_mode else 'legacy'} 口径，多档并行两轮轮换"
            ),
        }
        primary_keys.append(sp.key)

    if not primary_keys:
        raise ValueError(
            f"No measured points found for the configured models under {results_dir} "
            f"(expected <results_dir>/[round*/]<model>/curve.json or *_burst.jsonl). Run `sweep` first, or check `results_dir` / model names."
        )

    # Load optional external baseline cards
    for ext in cfg.external_baselines:
        ext_prec = infer_precision(ext.model, ext.canonical_model, ext.precision)
        temp_sp = ModelTPSpec(
            key=ext.key,
            card=ext.card,
            model=ext.model,
            served_model_name=ext.canonical_model,
            model_short=ext.canonical_model,
            tp=ext.tp,
            cards=ext.cards,
            roots=ext.roots,
            color=ext.color,
            precision=ext_prec,
            v7_mode=ext.v7_mode,
        )
        # Roots resolve against results_dir, its parent, or the packaged baseline copy.
        ext_base = results_dir
        pts = load_clean_spec_points(ext_base, temp_sp, stab_z=cfg.sweep.stab_z, restabilize=ext.v7_mode)
        if len(pts) < 1:
            raise ValueError(
                f"External baseline '{ext.key}' ({ext.card} · {ext.model}) has no measured points: "
                f"roots={ext.roots} were not found under {ext_base}, its parent, or the checkout's "
                f"results/ dir. Fix `external_baselines[].roots` or remove the entry."
            )
        data[ext.key] = {
            "lbl": f"{ext.card} · {ext.canonical_model} · TP={ext.tp}",
            "col": ext.color,
            "tp": ext.tp,
            "cards": ext.card_num,
            "card": ext.card,
            "precision": ext_prec,
            "roots": list(ext.roots),
            "name": ext.model,
            "canon_model": ext.canonical_model,
            "model_short": ext.canonical_model,
            "pts": pts,
            "groups": by_rate(pts),
            "v7_mode": ext.v7_mode,
            "tokenizer_url": "",
            "base_dir": ext_base,
            "port_labels": {},
            "how_note": f"每副本占 {ext.card_num:g} card，" + (ext.how_note or ("v7 口径" if ext.v7_mode else "旧口径")),
        }

    multi_prec_all = len({d["precision"] for d in data.values()}) > 1
    multi_card_pri = len({data[k]["card"] for k in primary_keys}) > 1
    multi_prec_pri = len({data[k]["precision"] for k in primary_keys}) > 1
    if multi_card_pri or multi_prec_pri:
        for k in primary_keys:
            d = data[k]
            pfx: List[str] = []
            if multi_card_pri and d["card"] not in d["lbl"]:
                pfx.append(str(d["card"]))
            if multi_prec_pri and d["precision"].upper() not in d["lbl"].upper():
                pfx.append(str(d["precision"]))
            if pfx:
                d["lbl"] = " · ".join(pfx + [d["lbl"]])

    # Build (card, canonical_model) grouping across primary + external cards (strictly NOT split by TP/card_num!)
    cm_order: List[Tuple[str, str]] = []
    cm_keys: Dict[Tuple[str, str], List[str]] = {}
    cm_col: Dict[Tuple[str, str], str] = {}
    for k in list(data.keys()):
        d = data[k]
        model_tbl = (
            f"{d['canon_model']} ({d['precision']})"
            if multi_prec_all and d["precision"].upper() not in d["canon_model"].upper()
            else d["canon_model"]
        )
        pair = (d["card"], model_tbl)
        if pair not in cm_keys:
            cm_order.append(pair)
            cm_keys[pair] = []
            cm_col[pair] = d["col"]
        cm_keys[pair].append(k)

    # 1. Headline 2-level SLA summary table grouped strictly by (Card, Model) + bottleneck gap%
    head_rows: List[str] = []
    for si, sla in enumerate(slas):
        for ki, (card_lbl, model_lbl) in enumerate(cm_order):
            ks = cm_keys[(card_lbl, model_lbl)]
            col = cm_col[(card_lbl, model_lbl)]
            est, best, fail, win_k = _best_for_keys(data, ks, sla.ttft_ms, sla.tpot_ms)
            d = data[win_k]
            sla_cell = (
                f'<td class="m" rowspan="{len(cm_order)}" style="color:{sla.color}">'
                f"SLA{html.escape(sla.label)}<br>TTFT ≤ {sla.ttft_ms:g}<br>TPOT ≤ {sla.tpot_ms:g}</td>"
                if ki == 0
                else ""
            )
            tr_cls = ' class="grp"' if (ki == 0 and si > 0) else ""
            if best <= 0:
                k0 = min(ks, key=lambda x: min(data[x]["groups"]) * 60.0 / data[x]["cards"])
                d0 = data[k0]
                r0 = min(d0["groups"])
                tt_s = " / ".join(f"{m['ttft']:.0f}" for m in d0["groups"][r0])
                tp_s = " / ".join(f"{m['tpot']:.1f}" for m in d0["groups"][r0])
                head_rows.append(
                    f"<tr{tr_cls}>{sla_cell}<td class=\"m\" style=\"color:{col}\">{html.escape(card_lbl)}</td>"
                    f'<td class="m">{html.escape(model_lbl)}</td>'
                    f'<td colspan="7" class="m">最低负载档（{r0 * 60.0 / d0["cards"]:.2f} QPM/card）也不满足：'
                    f"TTFT {tt_s}、TPOT {tp_s}</td></tr>"
                )
                continue

            b_r, _ = ceiling_rates(d["groups"], sla.ttft_ms, sla.tpot_ms, v7_mode=d["v7_mode"])
            at = list(d["groups"].get(b_r, []))
            bm = bottleneck_metrics(at, sla.ttft_ms, sla.tpot_ms) if at else None
            if bm is not None:
                gap_pct = bm["gap"] * 100.0
                cliff_ok = fail is not None and ((fail - best) / max(best, 1e-9)) <= cfg.sweep.min_bracket_rel
                gap_cls = "ok" if (bm["gap"] < cfg.sweep.gap_threshold or cliff_ok) else ""
                gap_style = "" if gap_cls else ' style="color:#D97706;font-weight:600"'
                gap_cell = f'<td class="{gap_cls}"{gap_style}>{gap_pct:.1f}% ({bm["bottleneck_dim"]})</td>'
                tt_cell = f"{bm['mean_ttft']:.0f}"
                tp_cell = f"{bm['mean_tpot']:.1f}"
            else:
                gap_cell, tt_cell, tp_cell = "<td>-</td>", "-", "-"

            fail_str = f"{fail:.1f}" if fail is not None else "未触顶"
            n_sum = sum(int(m.get("n_kept") or m["n"]) for m in at)
            head_rows.append(
                f"<tr{tr_cls}>{sla_cell}<td class=\"m\" style=\"color:{col}\">{html.escape(card_lbl)}</td>"
                f'<td class="m">{html.escape(model_lbl)}</td>'
                f"<td><b>{est:.1f}</b></td><td>{best:.1f}</td><td>{fail_str}</td>"
                f"<td>{tt_cell}</td><td>{tp_cell}</td>{gap_cell}<td>{n_sum}</td></tr>"
            )

    # Precompute raw_stats & accuracy across primary & external configs
    raw_cache: Dict[Tuple[str, int], Optional[dict]] = {}
    acc_by_key: Dict[str, List[Tuple[float, int]]] = {}
    pool_by_key: Dict[str, List[dict]] = {}
    fixed_acc: Dict[str, Optional[dict]] = {}
    for k, d in data.items():
        fixed_acc[k] = _resolve_spec_acc_stats(
            results_dir, d["roots"], d["name"], d["tokenizer_url"], d["canon_model"]
        )
        idx_counter = 0
        for _rate, ms in d["groups"].items():
            for m in ms:
                a = _raw_point_stats(d["base_dir"], m, d["name"], d["v7_mode"], d["tokenizer_url"], d["canon_model"])
                raw_cache[(k, idx_counter)] = a
                idx_counter += 1
                if a and a["acc"] is not None:
                    acc_by_key.setdefault(k, []).append((a["acc"], a["n_scored"]))
                if a and k in primary_keys:
                    pool_by_key.setdefault(k, []).extend(a["recs"])

    # 2. Cross-platform comparison table (`跨平台对比`)
    def _opt_tp_tag(k: str, best_qpm: float) -> str:
        d = data[k]
        r_opt = best_qpm * d["cards"] / 60.0
        return f"TP={d['tp']} · {d['cards']:g} card (@{r_opt:.2f} req/s)"

    xplat_rows: List[str] = []
    for card_lbl, model_lbl in cm_order:
        ks = cm_keys[(card_lbl, model_lbl)]
        cells: List[str] = []
        picks: List[Tuple[float, str]] = []
        for sla in slas[:2]:
            est, best, fail, win_k = _best_for_keys(data, ks, sla.ttft_ms, sla.tpot_ms)
            picks.append((best, win_k))
            if best > 0:
                f_txt = f"{fail:.1f}" if fail is not None else "未触顶"
                cells.append(f'<td><b>{est:.1f}</b> <span style="color:#9CA3AF">({best:.1f}–{f_txt})</span></td>')
            else:
                cells.append("<td><b>不满足</b></td>")

        # Accuracy: prefer the fixed `acc/` pass (identical prompts for every card), else the
        # sample-weighted mean over the per-point judged windows of older runs.
        if any(fixed_acc.get(k) for k in ks):
            fx = next(fixed_acc[k] for k in ks if fixed_acc.get(k))
            fx_rate = f"@{fx['rate']:g}/s" if fx.get("rate") else ""
            acc_cell = (
                f'<td title="固定 {fx["prompts"]} 条题目 {fx_rate}（SLA① 上限速率，开环泊松，预热已剔除），'
                f'有效评分 {fx["n_scored"]} 条"><b>{fx["acc"]:.1f}%</b> '
                f'<span style="color:#9CA3AF">(固定{fx["prompts"]})</span></td>'
            )
        else:
            tot_sc_w = 0.0
            tot_sc_n = 0
            for k in ks:
                for acc_v, n_sc in acc_by_key.get(k, []):
                    tot_sc_w += acc_v * n_sc
                    tot_sc_n += n_sc
            if tot_sc_n > 0:
                acc_cell = (
                    f'<td title="有效评分样本 {tot_sc_n} 条"><b>{tot_sc_w / tot_sc_n:.1f}%</b> '
                    f'<span style="color:#9CA3AF">({tot_sc_n})</span></td>'
                )
            else:
                acc_cell = "<td>–</td>"

        b1, k1 = picks[0]
        b2, k2 = picks[1] if len(picks) > 1 else (0.0, k1)
        if b1 > 0 and b2 > 0:
            if k1 == k2:
                d1 = data[k1]
                opt_str = (
                    f"最优数据点：TP={d1['tp']} · {d1['cards']:g} card "
                    f"(SLA① @{b1 * d1['cards'] / 60.0:.2f} / SLA② @{b2 * d1['cards'] / 60.0:.2f} req/s)"
                )
            else:
                opt_str = f"最优数据点：SLA① {_opt_tp_tag(k1, b1)}，SLA② {_opt_tp_tag(k2, b2)}"
        elif b1 > 0:
            opt_str = f"最优数据点：SLA① {_opt_tp_tag(k1, b1)}（SLA② 无满足点）"
        else:
            opt_str = "最优数据点：无满足点"

        tps_tested = "/".join(str(x) for x in sorted({data[k]["tp"] for k in ks}))
        cards_tested = "/".join(f"{x:g}" for x in sorted({data[k]["cards"] for k in ks}))
        d_rep = data[ks[0]]
        if ks[0] in primary_keys:
            base_how = (
                f"每副本占 {d_rep['cards']:g} card，"
                f"{'v7' if d_rep['v7_mode'] else 'legacy'} 口径，已测 TP={tps_tested}（{cards_tested} card）多档并行轮换"
            )
        else:
            base_how = d_rep["how_note"]
        how_full = f"<b>{html.escape(opt_str)}</b>；{html.escape(base_how)}"
        xplat_rows.append(
            f'<tr><td class="m">{html.escape(card_lbl)}</td><td class="m">{html.escape(model_lbl)}</td>'
            f'{"".join(cells)}{acc_cell}<td class="m">{how_full}</td></tr>'
        )

    xplat_html = (
        '<table><tr><th style="text-align:left">Card</th><th style="text-align:left">Model</th>'
        "<th>SLA① 上限 QPM/card（插值，实测区间）</th><th>SLA② 上限 QPM/card</th>"
        '<th>总体 acc</th><th style="text-align:left">口径（含最优数据点并行方式与卡数）</th></tr>'
        f'{"".join(xplat_rows)}</table>'
    )

    # Group primary card configs by canonical model (smallest TP first)
    by_model: Dict[str, List[str]] = {}
    for k in sorted(primary_keys, key=lambda x: (data[x]["canon_model"], data[x]["tp"], data[x]["cards"])):
        by_model.setdefault(data[k]["canon_model"], []).append(k)

    # 3. Operating-point broken-axis scatter plots (ONE per model, ABOVE curves!)
    shape_map = dict(OP_KINDS)
    op_charts: List[str] = []
    for mdl_name, ks in by_model.items():
        marks, used = [], set()
        for k in ks:
            d = data[k]
            grp_kinds = {kind: [] for kind, _ in OP_KINDS}
            for rate, ms in d["groups"].items():
                x = st.mean(float(m["ttft"]) for m in ms)
                y = st.mean(float(m["tpot"]) for m in ms)
                if not all(is_point_valid(m, v7_mode=d["v7_mode"]) for m in ms):
                    kind = "排队不收敛"
                else:
                    kind = (
                        "满足 ① 与 ②"
                        if (x <= sla2.ttft_ms and y <= sla2.tpot_ms)
                        else ("仅满足 ①" if (x <= sla1.ttft_ms and y <= sla1.tpot_ms) else "①② 都不满足")
                    )
                grp_kinds[kind].append((x, y, rate * 60.0 / d["cards"]))
                if kind == "排队不收敛":
                    break
            for kind, v in grp_kinds.items():
                if v:
                    marks.append(
                        (
                            d["col"],
                            [(px, py) for px, py, _ in v],
                            False,
                            [f"{q:.1f}" for _, _, q in v],
                            shape_map[kind],
                        )
                    )
                    used.add(kind)
        mdl_short = data[ks[0]]["model_short"]
        tps_leg = [(_short_tp_label(data[k], multi_card_pri, multi_prec_pri), data[k]["col"], []) for k in ks]
        op_charts.append(
            render_svg_chart(
                f"{mdl_short} · 每个负载点的 (TTFT, TPOT)，点上数字 = QPM/card",
                tps_leg,
                "mean TPOT (ms)",
                "mean TTFT (ms)",
                dots=False,
                zoom=True,
                hline=[(s.tpot_ms, f"TPOT SLA{s.label} {s.tpot_ms:g}", s.color) for s in slas],
                vline=[(s.ttft_ms, f"TTFT SLA{s.label} {s.ttft_ms:g}", s.color) for s in slas],
                marks=marks,
                mark_legend=[(kind, shape_map[kind]) for kind, _ in OP_KINDS if kind in used],
                xbrk=2.0 * max(s.ttft_ms for s in slas),
                ybrk=2.0 * max(s.tpot_ms for s in slas),
            )
        )

    # 4. Curves split by model — ONLY 2 charts per model (mean TPOT vs QPM/card & mean TTFT vs QPM/card; NO budget charts!)
    curve_sections: List[str] = []
    for mdl_name, ks in by_model.items():
        mdl_short = data[ks[0]]["model_short"]
        tpot_series = [
            (
                _short_tp_label(data[k], multi_card_pri, multi_prec_pri),
                data[k]["col"],
                [(rate * 60.0 / data[k]["cards"], st.mean(float(m["tpot"]) for m in ms)) for rate, ms in data[k]["groups"].items()],
            )
            for k in ks
        ]
        ttft_series = [
            (
                _short_tp_label(data[k], multi_card_pri, multi_prec_pri),
                data[k]["col"],
                [(rate * 60.0 / data[k]["cards"], st.mean(float(m["ttft"]) for m in ms)) for rate, ms in data[k]["groups"].items()],
            )
            for k in ks
        ]
        c_tpot = render_svg_chart(
            f"{mdl_short} · mean TPOT vs 每卡吞吐（QPM/card）",
            tpot_series,
            "mean TPOT (ms)",
            "QPM / card",
            hline=[(s.tpot_ms, f"TPOT SLA{s.label} {s.tpot_ms:g}", s.color) for s in slas],
            ymax=sla1.tpot_ms * 2.2,
        )
        c_ttft = render_svg_chart(
            f"{mdl_short} · mean TTFT vs 每卡吞吐（QPM/card）",
            ttft_series,
            "mean TTFT (ms)",
            "QPM / card",
            logy=True,
            hline=[(s.ttft_ms, f"TTFT SLA{s.label} {s.ttft_ms:g}", s.color) for s in slas],
        )
        hdr_txt = html.escape(mdl_name) if mdl_name == mdl_short else f"{html.escape(mdl_name)}（{html.escape(mdl_short)}）"
        curve_sections.append(f'<h3>{hdr_txt}</h3><div class="grid"><div class="card">{c_tpot}</div><div class="card">{c_ttft}</div></div>')

    # 5. Per-board / per-port TPOT offset table + v7 methodology notes
    board_rows: List[str] = []
    for k in primary_keys:
        d = data[k]
        dev: Dict[int, List[float]] = {}
        for _rate, ms in d["groups"].items():
            ok = [m for m in ms if is_point_valid(m, v7_mode=d["v7_mode"]) and m.get("port")]
            if len(ok) < 2:
                continue
            mu = st.mean(float(m["tpot"]) for m in ok)
            for m in ok:
                dev.setdefault(int(m["port"]), []).append(float(m["tpot"]) - mu)
        if not dev:
            continue
        cells = "".join(
            '<td style="color:%s">%s<br><b>%+.1f</b> <span style="color:#9CA3AF">(%d)</span></td>'
            % (
                "#DC2626" if st.mean(v) > 3 else "#059669" if st.mean(v) < -3 else "#374151",
                html.escape(d["port_labels"].get(p, f"port {p}")),
                st.mean(v),
                len(v),
            )
            for p, v in sorted(dev.items(), key=lambda kv: -st.mean(kv[1]))
        )
        board_rows.append(f'<tr><td class="m" style="color:{d["col"]}">{html.escape(d["lbl"])}</td>{cells}</tr>')
    board_html = f"<table>{''.join(board_rows)}</table>" if board_rows else '<p class="sub">（单副本或同速率无多板对照点，暂无板间偏差项）</p>'

    all_primary_pts = [m for k in primary_keys for m in data[k]["pts"]]
    ns_kept = [int(m.get("n_kept") or m["n"]) for m in all_primary_pts]
    n_good = sum(
        1
        for m in all_primary_pts
        if m.get("len_ok", True) and not m.get("errors", 0) and not m.get("missing", 0)
    )
    rr = sorted(float(m["rate_sent"]) / float(m["rate"]) for m in all_primary_pts if m.get("rate_sent") and m.get("rate"))
    rr_med = rr[len(rr) // 2] if rr else 1.0
    rr_lo = rr[0] if rr else 1.0
    rr_hi = rr[-1] if rr else 1.0
    extra_note_li = f"<li><b>补充说明。</b>{html.escape(cfg.methodology_note)}</li>" if cfg.methodology_note else ""

    notes_html = f"""<h2>本版口径（v7）与审计方法</h2>
<div class="note"><ol>
<li><b>每点固定样本、长度八分位蛇形配对、全部回收。</b>每点条数按预热 {cfg.sweep.warmup_s:.0f} s + 稳态 ≥{cfg.sweep.min_s:.0f} s 且单轮判定窗口 ≥{cfg.sweep.min_per_round} 条取 8 的整数倍；prompt 按八分位蛇形配对使任意整 block 均值落在 ±1.3% 内；发完后等待全部在途请求返回，杜绝截断慢请求。</li>
<li><b>长度达标按被测服务端 /tokenize（即服务端自身 tokenizer）分别核算。</b>以含 <code>PAD_INSTRUCTION</code> 的客户端实际模板长度为 <code>ds_in</code> 基准，要求判定窗口均值与全量均值同时在 ±{cfg.sweep.len_tol * 100:g}% 内、输出均值在 {cfg.sweep.max_tokens} ±{cfg.sweep.len_tol * 100:g}% 内。本报告 {len(all_primary_pts)} 个测点中 {n_good} 个长度达标且零错误、零缺失；每点判定窗口 {min(ns_kept)}–{max(ns_kept)} 条。</li>
<li><b>绝对时刻表锁外泊松到达。</b>请求到达时刻在全局单调时刻表上生成，协程在锁外睡眠，速率由 <code>CAT_LOCUST_RATE</code> 独立驱动，彻底消除协调遗漏（Coordinated Omission）；prompt 分词统计移至 SSE 流结束后，不污染 TTFT。窗口实发速率 / 目标速率中位数 {rr_med:.3f}（{rr_lo:.2f}–{rr_hi:.2f}）。</li>
<li><b>稳定性判据（STAB_Z={cfg.sweep.stab_z:g}）与两档 SLA gap &lt; {cfg.sweep.gap_threshold * 100:.0f}% 自适应加密。</b>窗口按整 block 对半分，TTFT 先线性回归扣除 prompt 长度影响，仅当前后半段均值漂移与中位数漂移同时超阈值（TTFT max(25%, 300 ms)、TPOT max(50%, 10 ms)）且超 {cfg.sweep.stab_z:g} 倍标准误时判排队不收敛。对两档 SLA 边界自动内插补点轮换复测，直至实测最优通过点最短板余量 <code>gap &lt; {cfg.sweep.gap_threshold * 100:.0f}%</code>（或相对跨度 ≤ {cfg.sweep.min_bracket_rel * 100:.1f}% 证实真实台阶悬崖）。</li>
<li><b>多档多副本轮换与板间 TPOT 偏差（ms，稳定点相对同速率均值）。</b></li>
</ol>
{board_html}
{f'<ol start="6">{extra_note_li}</ol>' if extra_note_li else ''}
</div>"""

    # 6. Single merged per-point detail sheet (`逐点明细（含输入与真实有效输出长度）`)
    detail_rows: List[str] = []
    for k in primary_keys:
        d = data[k]
        first = True
        idx_counter = 0
        for rate, ms in d["groups"].items():
            for j, m in enumerate(ms):
                a = raw_cache.get((k, idx_counter))
                idx_counter += 1
                ok = passes_sla(m, sla1.ttft_ms, sla1.tpot_ms, v7_mode=d["v7_mode"])
                why_parts = [
                    tag
                    for tag, cond in (
                        ("TTFT", float(m["ttft"]) > sla1.ttft_ms),
                        ("TPOT", float(m["tpot"]) > sla1.tpot_ms),
                        ("漂移", not m.get("stable", True)),
                        (f"错误{m.get('errors', 0)}", d["v7_mode"] and m.get("errors", 0)),
                        (f"缺{m.get('missing', 0)}", d["v7_mode"] and m.get("missing", 0)),
                        ("长度", d["v7_mode"] and "len_ok" in m and not m.get("len_ok", False)),
                    )
                    if cond
                ]
                why = "+".join(why_parts) or "通过"
                if a:
                    in_str = f"{a['in_min']} / {a['in_avg']:.0f} / {a['in_max']}"
                    ro_str = (
                        f"{a['eff_min']} / {a['eff_avg']:.0f} / {a['eff_max']}"
                        if a["n"] > a["eff_miss"] + a["eff_unknown"]
                        else "–"
                    )
                    n_done_str = f"{a['n']}"
                    miss_str = "–" if a["eff_unknown"] == a["n"] else f"{a['eff_miss']}"
                else:
                    in_avg = m.get("in_len")
                    in_str = f"– / {in_avg:.0f} / –" if in_avg else "–"
                    ro_str = "–"
                    n_done_str = f"{int(m.get('n_kept') or m['n'])}"
                    miss_str = "–"

                el_s = m.get("elapsed")
                dur_str = ("%d:%02d" % divmod(int(round(float(el_s))), 60)) if el_s else "–"
                tr_cls = ' class="grp"' if first else ""
                cfg_cell = html.escape(d["lbl"]) if first else ""
                rep_cell = "重复" if j else ""
                ok_cls = "ok" if ok else "bad"
                detail_rows.append(
                    f'<tr{tr_cls}><td class="m" style="color:{d["col"]}">{cfg_cell}</td>'
                    f"<td>{rate:.4f}</td><td>{rate * 60.0 / d['cards']:.2f}</td><td>{rep_cell}</td>"
                    f"<td>{m['ttft']:.0f}</td><td>{m.get('ttft_p90', m['ttft']):.0f}</td>"
                    f"<td>{m['tpot']:.1f}</td><td>{m.get('tpot_p90', m['tpot']):.1f}</td>"
                    f"<td>{m.get('drift', 0.0):+.0f}</td><td>{dur_str}</td>"
                    f"<td>{in_str}</td><td>{ro_str}</td><td>{n_done_str}</td><td>{miss_str}</td>"
                    f'<td class="{ok_cls}">{html.escape(why)}</td></tr>'
                )
                first = False

        rs_pool = pool_by_key.get(k, [])
        if rs_pool:
            pl = sorted(int(r.get("prompt_len") or 0) for r in rs_pool)
            el = sorted(int(r["eff_len"]) for r in rs_pool if r.get("eff_len") is not None)
            ro_tot = f"{el[0]} / {st.mean(el):.0f} / {el[-1]}" if el else "–"
            n_unk = sum(1 for r in rs_pool if r.get("_eff_unknown"))
            miss_tot = "–" if n_unk == len(rs_pool) else str(len(rs_pool) - len(el) - n_unk)
            detail_rows.append(
                f'<tr style="background:#FEF3C7;font-weight:700"><td class="m" style="color:{d["col"]}">合计（该配置窗口请求）</td>'
                f"<td>–</td><td>–</td><td></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td>"
                f"<td>{pl[0]} / {st.mean(pl):.0f} / {pl[-1]}</td><td>{ro_tot}</td>"
                f"<td>{len(rs_pool)}</td><td>{miss_tot}</td><td></td></tr>"
            )

    # Dataset-wide reference row: counts come from the requests actually sent in this run, so
    # the numbers match this model's tokenizer instead of a table built with another one.
    all_in = [int(r.get("prompt_len") or 0) for k in data for r in data[k].get("recs", []) if r.get("prompt_len")]
    if all_in:
        all_in.sort()
        detail_rows.append(
            '<tr class="grp" style="background:#ECFDF5;font-weight:700"><td class="m">本次全部窗口请求（参照）</td>'
            "<td>–</td><td>–</td><td></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td>"
            f"<td>{all_in[0]} / {st.mean(all_in):.0f} / {all_in[-1]}</td><td>–</td><td>{len(all_in)}</td><td>–</td><td></td></tr>"
        )

    # Headline accuracy line: uses the same source as the table (fixed acc pass when present).
    def _acc_text(k: str) -> str:
        fx = fixed_acc.get(k)
        if fx:
            rate_txt = f" @{fx['rate']:g}/s" if fx.get("rate") else ""
            return f"{fx['acc']:.1f}%（固定{fx['prompts']}题{rate_txt}）"
        if k in acc_by_key:
            tot_w = sum(v * n for v, n in acc_by_key[k])
            tot_n = sum(n for _, n in acc_by_key[k])
            if tot_n:
                return f"{tot_w / tot_n:.1f}%（窗口加权）"
        return ""

    acc_summary = "；".join(
        f"{data[k]['lbl']} {_acc_text(k)}" for k in primary_keys if _acc_text(k)
    ) or "无评分数据"

    title_str = cfg.title or f"{cfg.card} · 不同 TTFT / TPOT 门限下的 QPM/card"
    dpc_desc = "QPM/card = 到达率 × 60 / 每副本占用的 card 数（只用到一张卡的一部分时按 0.25/0.33/0.5 等折算）"

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(title_str)}</title>
<style>
 body{{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;margin:24px auto;max-width:1260px;color:#111827;padding:0 16px}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px;border-left:4px solid #2563EB;padding-left:8px}}
 h3{{font-size:14px;margin:16px 0 6px}}
 .sub{{color:#6B7280;font-size:13px;margin-bottom:12px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 .card{{border:1px solid #E5E7EB;border-radius:8px;padding:8px;background:#fff}}
 table{{border-collapse:collapse;font-size:12px;width:100%}}
 th,td{{border:1px solid #E5E7EB;padding:4px 7px;text-align:right}}
 th{{background:#F3F4F6}} td.m{{text-align:left;font-weight:600}}
 tr.grp td{{border-top:2px solid #9CA3AF}}
 td.ok{{color:#059669;font-weight:700}} td.bad{{color:#DC2626;font-weight:700}}
 .note{{background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:10px 16px;font-size:13px;line-height:1.75}}
 .note li{{margin:4px 0}}
</style></head><body>
<h1>{html.escape(title_str)}</h1>
<div class="sub">开环泊松到达（绝对时刻表）· 输出强制 pad 到 {cfg.sweep.max_tokens} token · radix cache 关闭 · 输入 nocache_sla_in3669_v3（1000 条，八分位蛇形分层）·
判定窗口 = 预热 {cfg.sweep.warmup_s:.0f} s 后 ≥{cfg.sweep.min_s:.0f} s 的整 block（单轮 ≥{cfg.sweep.min_per_round} 条，输入/输出均值 ±{cfg.sweep.len_tol * 100:g}%）·
QPM/card 折算：{html.escape(dpc_desc)}</div>

<h2>两档 SLA 下的上限：① mean TTFT ≤ {sla1.ttft_ms:g} ms 且 mean TPOT ≤ {sla1.tpot_ms:g} ms；② mean TTFT ≤ {sla2.ttft_ms:g} ms 且 mean TPOT ≤ {sla2.tpot_ms:g} ms</h2>
<table><tr><th style="text-align:left">SLA</th><th style="text-align:left">Card</th><th style="text-align:left">Model</th><th>插值上限 QPM/card</th><th>实测通过点</th><th>首个失败点</th>
<th>该点 TTFT</th><th>该点 TPOT</th><th>最短板余量 gap%</th><th>样本</th></tr>{"".join(head_rows)}</table>
<p class="sub">仅按 <code>(Card, Model)</code> 汇总（不按占卡数 / TP 拆分，自动取已测各 TP 中最高可用 <code>QPM/card</code>）；每副本占用 card 数由配置的 <code>cards</code> 给出；<code>最短板余量 gap% = (1 - max(TTFT/TTFT_max, TPOT/TPOT_max)) × 100%</code>（自适应加密目标 <code>gap &lt; {cfg.sweep.gap_threshold * 100:.0f}%</code>）。</p>

<h2>跨平台对比（同一数据集 nocache_sla_in3669_v3、输出 pad 到 {cfg.sweep.max_tokens}、radix cache 关闭）</h2>
<p class="sub">统一按物理卡折算（<code>cards</code> 为每副本占用的卡数），各行取该 <code>(Card, Model)</code> 所有已测 TP 中的最高可用 <code>QPM/card</code>；总体 acc 为判定窗口内全部完成请求的样本加权得分均值（括号内为评分条数）。</p>
{xplat_html}

<h2>工作点图：横轴 TTFT、纵轴 TPOT，点上数字为 QPM/card</h2>
<p class="sub">同一模型的各 TP 配置画在同一张图里，颜色区分 TP。每个点是一个实测负载档（重复测取均值），形状表示该点满足哪档 SLA：◆ 同时满足 ① 与 ②、● 仅满足 ①、▲ 两档都不满足、✕ 排队不收敛（窗口均值偏乐观，不可用）。
红色虚线为 SLA①（TTFT {sla1.ttft_ms:g}、TPOT {sla1.tpot_ms:g}），橙色虚线为 SLA②（TTFT {sla2.ttft_ms:g}、TPOT {sla2.tpot_ms:g}），框内最大的数字即为该 SLA 下的上限。
坐标超过最宽松 SLA 的 2 倍之后改为对数刻度（灰底区域），避免个别饱和点把其余点挤到角落。</p>
<div class="grid">{"".join(f'<div class="card">{c}</div>' for c in op_charts)}</div>

<h2>曲线（分模型展示，同图对比各 TP）</h2>
{"".join(curve_sections)}

{notes_html}

<h2>逐点明细（含输入与真实有效输出长度）</h2>
<p class="sub">总耗时 = 该点从启动客户端到最后一条请求返回（分:秒，含预热与排空）；每条回复均 pad 到 {cfg.sweep.max_tokens} token 以固定解码量。右侧按判定窗口统计：<code>in (min/avg/max)</code> 为输入 prompt token 数，<code>real-out (min/avg/max)</code> 为答案结束分隔符 <code>{html.escape(PAD_DELIM)}</code> 之前的真实有效输出 token 数，「无分隔符」为未输出分隔符的条数。各配置总体 acc：{html.escape(acc_summary)}。</p>
<table><tr><th style="text-align:left">配置</th><th>req/s</th><th>QPM/card</th><th></th>
<th>TTFT</th><th>TTFT p90</th><th>TPOT</th><th>TPOT p90</th><th>漂移 ms</th><th>总耗时</th>
<th>in (min/avg/max)</th><th>real-out (min/avg/max)</th><th>完成请求</th><th>无分隔符</th><th>判定</th></tr>{"".join(detail_rows)}</table>
</body></html>
"""

    target_out = Path(out_path) if out_path else Path(cfg.report_out)
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(doc, encoding="utf-8")
    return target_out
