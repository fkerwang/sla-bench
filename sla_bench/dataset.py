"""Bundled dataset decompression, per-server length caching, and serpentine octile sampler."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import shutil
import statistics as st
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from .locust_client.model_config import PAD_DELIM, PAD_INSTRUCTION

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "dataset"
BUNDLED_DATASET_GZ = DATA_DIR / "nocache_sla_in3669_v3.ndjson.gz"


def default_cache_dir() -> Path:
    env_dir = os.environ.get("SLA_BENCH_CACHE_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
    else:
        p = Path(tempfile.gettempdir()) / "sla_bench_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dataset(dataset_path: Optional[str | Path] = None, cache_dir: Optional[str | Path] = None) -> Path:
    """Return an uncompressed working NDJSON path, auto-decompressing .gz atomically when needed."""
    cdir = Path(cache_dir) if cache_dir else default_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    src = Path(dataset_path) if dataset_path else BUNDLED_DATASET_GZ
    if not src.exists():
        raise FileNotFoundError(f"Dataset file not found: {src}")

    if src.suffix != ".gz":
        return src.resolve()
    # A plain <name>.ndjson sitting next to the .gz must never shadow the cache copy: the
    # compressed file is the single source of truth.

    stem_name = src.name[:-3]  # strip .gz
    src_stat = src.stat()
    tag = f"{src_stat.st_size}_{int(src_stat.st_mtime)}"
    target = cdir / f"{tag}_{stem_name}"
    if target.exists() and target.stat().st_size > 0:
        return target

    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_ds_", suffix=".ndjson", dir=str(cdir))
    os.close(fd)
    try:
        with gzip.open(src, "rb") as f_in, open(tmp_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise
    return target


def _dataset_fingerprint(ds_path: Path) -> str:
    st_info = ds_path.stat()
    raw = f"{ds_path.resolve()}:{st_info.st_size}:{int(st_info.st_mtime_ns)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_or_load_tokenizer_lens(
    dataset_path: str | Path,
    tokenizer_url: Optional[str] = None,
    model: str = "",
    pad_to_tokens: int = 1000,
    cache_dir: Optional[str | Path] = None,
    fallback_ds_in: Optional[float] = None,
    workers: int = 16,
) -> Tuple[List[int], float]:
    """Return `(lens, ds_in)` in client units, counted by the SERVER's `/tokenize`.

    Client units = prompt token count of the chat messages (with `PAD_INSTRUCTION`
    appended to the last message) after the server applies its own chat template, i.e.
    the same number the server later reports as `usage.prompt_tokens`.
    `tokenizer_url` is the server's `/tokenize` endpoint (or any chat URL on that server;
    it is mapped to `/tokenize`). Results are cached on disk keyed by
    server + model + dataset + padding settings. Without a URL (e.g. dry-run) a
    model-agnostic estimate is returned instead.
    """
    ds_file = ensure_dataset(dataset_path, cache_dir=cache_dir)
    cdir = Path(cache_dir) if cache_dir else default_cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)

    pad_suffix = PAD_INSTRUCTION.format(delim=PAD_DELIM) if pad_to_tokens > 0 else ""

    if tokenizer_url:
        from .locust_client.remote_tok import RemoteTokenizer, default_chat_template_kwargs, derive_tokenize_url

        ctk = default_chat_template_kwargs(model)
        tok_url = tokenizer_url if tokenizer_url.rstrip("/").endswith("/tokenize") else derive_tokenize_url(tokenizer_url)
        key_src = (
            f"{_dataset_fingerprint(ds_file)}_"
            f"{hashlib.sha256(f'{tok_url}|{model}|{json.dumps(ctk, sort_keys=True)}'.encode('utf-8')).hexdigest()[:16]}_"
            f"pad{pad_to_tokens}_{hashlib.md5(pad_suffix.encode('utf-8')).hexdigest()[:8]}"
        )
        cache_file = cdir / f"tok_lens_remote_{key_src}.json"
        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, dict) and cached.get("lens") and cached.get("ds_in"):
                    return [int(x) for x in cached["lens"]], float(cached["ds_in"])
            except Exception:
                pass

        all_messages: List[list] = []
        with open(ds_file, encoding="utf-8-sig") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                messages = rec["messages"] if isinstance(rec, dict) and "messages" in rec else rec
                messages = [
                    dict(m)
                    for m in messages
                    if not (
                        isinstance(m, dict)
                        and m.get("role") == "assistant"
                        and not (m.get("content") or "").strip()
                        and not m.get("tool_calls")
                    )
                ]
                if pad_suffix and messages:
                    messages[-1]["content"] = (messages[-1].get("content") or "") + pad_suffix
                all_messages.append(messages)

        remote = RemoteTokenizer(tok_url, model=model)
        lens = [int(n) for n in remote.count_messages_many(all_messages, chat_template_kwargs=ctk, workers=workers)]
        ds_in = float(st.mean(lens))
        payload = {
            "tokenizer_url": tok_url,
            "model": model,
            "dataset": str(ds_file),
            "pad_to_tokens": pad_to_tokens,
            "count": len(lens),
            "ds_in": ds_in,
            "lens": lens,
        }
        fd, tmp_json = tempfile.mkstemp(prefix=".tmp_lens_", suffix=".json", dir=str(cdir))
        os.close(fd)
        try:
            Path(tmp_json).write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp_json, cache_file)
        except Exception:
            try:
                os.remove(tmp_json)
            except OSError:
                pass
        return lens, ds_in

    # No server (dry-run): lengths are only used to stratify the sampling order and to print a
    # rough ds_in, so estimate them from the prompt text itself. Deliberately no bundled
    # per-model length table: token counts differ per tokenizer, and a stale table would be
    # silently wrong for every model but the one it was built with.
    approx_lens: List[int] = []
    pad_tok_est = 51 if pad_to_tokens > 0 else 0
    with open(ds_file, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msgs = json.loads(line)
                text = "\n".join(str(m.get("content") or "") for m in msgs) if isinstance(msgs, list) else line
            except ValueError:
                text = line
            # ~1 token per 2 UTF-8 bytes is a deliberately conservative model-agnostic proxy.
            approx_lens.append(max(1, len(text) // 2) + pad_tok_est)
    ds_in = float(fallback_ds_in) if fallback_ds_in is not None else float(st.mean(approx_lens))
    return approx_lens, ds_in


def balanced_file(
    dataset_path: str | Path,
    lens: List[int],
    seed: int,
    tag: str,
    n: int,
    out_dir: str | Path,
    blocks: int = 8,
) -> Tuple[Path, List[int]]:
    """Write `n` prompts (multiple of `blocks=8`) as length-balanced blocks.

    Each block takes one prompt from each length octile, chosen so the block's mean lands as
    close as possible to the dataset mean: octiles are visited in random order and each pick
    targets `remaining_needed / remaining_octiles`. Blocks are then shuffled, so any
    whole-block window (including the judged window after warm-up) has a mean within a
    fraction of a percent of the dataset mean, whatever the underlying length distribution.

    Guaranteeing this per block rather than relying on a fixed rank pairing matters when the
    lengths are only an estimate (no server available): a skewed proxy distribution otherwise
    lets individual blocks drift well past the +-3% gate.
    """
    if n % blocks != 0:
        n = int(-(-n // blocks) * blocks)

    with open(dataset_path, encoding="utf-8-sig") as fh:
        lines = [l for l in fh if l.strip()]
    if len(lines) != len(lens):
        raise ValueError(f"Dataset line count ({len(lines)}) != lens count ({len(lens)})")

    rnd = random.Random(seed)
    order = sorted(range(len(lens)), key=lambda i: lens[i])
    size = len(order) // blocks
    if size < 1:
        raise ValueError(f"Dataset must contain at least {blocks} prompts, got {len(lines)}")
    rem = len(order) % blocks
    octs, a = [], 0
    for j in range(blocks):
        b = a + size + (1 if j < rem else 0)
        octs.append(order[a:b])
        a = b
    pools = [sorted(o, key=lambda i: lens[i]) for o in octs]

    used = [0] * blocks
    n_blocks = n // blocks
    target = float(sum(lens)) / float(len(lens))
    out: list[int] = []
    # A point can legitimately need more prompts than the dataset holds: at 3.2 req/s with
    # warmup 60s + min_s 300s that is 1152 prompts against 1000 rows. The old loop stopped when
    # the octile pools ran out, silently writing a shorter file, so the run reported
    # `missing = n_planned - len(rs) > 0` and the point was discarded as `len_ok=False` -- which
    # truncated the reported ceiling (e.g. MoE: 2.8 and 3.2 req/s both invalid, hiding the real
    # boundary). Rebuild the pools when exhausted so the requested count is always honoured.
    while len(out) < n:
        left = [j for j in range(blocks) if used[j] < len(pools[j])]
        if not left:
            # Pools exhausted: refill with fresh per-octile shuffles, which keeps the octile
            # structure intact while allowing repeats across passes.
            pools = [sorted(rnd.sample(o, size), key=lambda i: lens[i]) for o in octs]
            used = [0] * blocks
            left = list(range(blocks))
        rnd.shuffle(left)
        need = target * len(left)
        blk: list[int] = []
        for pos, j in enumerate(left):
            pool = pools[j][used[j]:]
            if not pool:
                continue
            want = need / float(len(left) - pos)
            pick = min(pool, key=lambda i: abs(lens[i] - want))
            blk.append(pick)
            used[j] += 1
            need -= lens[pick]
        rnd.shuffle(blk)
        out.extend(blk)
    out = out[:n]

    # Octile-stratified picks average the octile centres, which for a right-skewed length
    # distribution sits below the global mean. Repair each block against the global mean by
    # swapping its worst offender for an unused prompt that reduces the deviation.
    chosen = set(out)
    spare = [i for i in range(len(lens)) if i not in chosen]
    rnd.shuffle(spare)
    for _pass in range(6):
        worst = 0.0
        for b in range(len(out) // blocks):
            lo, hi = b * blocks, b * blocks + blocks
            for _step in range(blocks * 2):
                if not spare:
                    break
                cur = sum(lens[i] for i in out[lo:hi])
                dev = abs(cur - target * (hi - lo))
                worst = max(worst, dev / (target * (hi - lo)))
                if dev / (target * (hi - lo)) <= 0.005:
                    break
                best = None
                for pos in range(lo, hi):
                    for si_pos, si in enumerate(spare[:64]):
                        new = cur - lens[out[pos]] + lens[si]
                        ndev = abs(new - target * (hi - lo))
                        if ndev < dev - 1e-9 and (best is None or ndev < best[0]):
                            best = (ndev, pos, si_pos)
                if best is None:
                    break
                _, pos, si_pos = best
                spare[si_pos], out[pos] = out[pos], spare[si_pos]
        if worst <= 0.005:
            break

    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    out_path = odir / f"_bal_{safe_tag}_s{seed}.ndjson"
    with open(out_path, "w", encoding="utf-8") as f:
        for i in out:
            f.write(lines[i] if lines[i].endswith("\n") else lines[i] + "\n")
    return out_path, [lens[i] for i in out]
