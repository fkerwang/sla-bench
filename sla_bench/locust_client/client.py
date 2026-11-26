"""Bundled v7 Locust benchmark client without allowlist restrictions.

Accepts ANY `--url http://host:port/v1/chat/completions`, ANY `--model_name`,
and ANY `--file /path/to/dataset.ndjson` out of the box on new GPU hosts.
"""

from __future__ import annotations

import concurrent.futures as cf
import fcntl
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from locust import events, tag, task
from locust.argument_parser import LocustArgumentParser
from locust.exception import StopUser

# Allow running both via `locust -f .../client.py` and `import sla_bench.locust_client.client`
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from . import model_config
    from .chat import ChatService, count_effective_tokens
    from .constant_rate_user import ConstantRateUser
    from .judge_acc import score_one
    from .remote_tok import default_chat_template_kwargs
    from .server_warm_up import ServerWarmUp
except ImportError:
    import model_config  # type: ignore
    from chat import ChatService, count_effective_tokens  # type: ignore
    from remote_tok import default_chat_template_kwargs  # type: ignore
    from constant_rate_user import ConstantRateUser  # type: ignore
    from judge_acc import score_one  # type: ignore
    from server_warm_up import ServerWarmUp  # type: ignore

_TZ_CN = timezone(timedelta(hours=8))
DATA_LOCK = threading.Lock()
HTTP_CLIENT_POOL: List["ChatHttpClient"] = []


def _now_log_ts() -> tuple[float, str]:
    ts = time.time()
    iso = datetime.fromtimestamp(ts, tz=_TZ_CN).isoformat(timespec="milliseconds")
    return ts, iso


class ChatHttpClient(object):
    """Proxies ChatService calls, fires Locust events, and appends pre_detail_stat_*.jsonl."""

    def __init__(
        self,
        request_event,
        url: str,
        model_name: str,
        api_key: str = "<API_KEY>",
        config=None,
        environment=None,
        report_dir: str = "report",
    ):
        self.service = ChatService(url=url, model=model_name, api_key=api_key, wsid="", config=config)
        self._request_event = request_event
        self.environment = environment
        self.report_dir = report_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.jsonl_filename = f"{report_dir}/pre_detail_stat_{timestamp}.jsonl"

    def __getattr__(self, name):
        func = getattr(self.service, name)

        def wrapper(*args, **kwargs):
            record_stats = kwargs.pop("record_stats", True)
            sample_index_kw = kwargs.pop("_sample_index", None)
            request_meta_first = {
                "request_type": "trpc",
                "name": "first",
                "start_time": time.time(),
                "response_length": 0,
                "response": None,
                "context": {},
                "exception": None,
                "response_time": None,
            }
            request_meta_all = {
                "request_type": "trpc",
                "name": "whole",
                "start_time": time.time(),
                "response_length": 0,
                "response": None,
                "context": {},
                "exception": None,
                "response_time": None,
            }
            request_meta_token_per_second = {
                "request_type": "trpc",
                "name": "token_per_second",
                "start_time": time.time(),
                "response_length": 0,
                "response": None,
                "context": {},
                "exception": None,
                "response_time": None,
            }
            request_meta_tpot = {
                "request_type": "trpc",
                "name": "tpot",
                "start_time": time.time(),
                "response_length": 0,
                "response": None,
                "context": {},
                "exception": None,
                "response_time": None,
            }
            request_meta_error = {
                "request_type": "trpc",
                "name": "error",
                "start_time": time.time(),
                "response_length": 0,
                "response": None,
                "context": {},
                "exception": None,
                "response_time": None,
            }
            t = time.perf_counter()
            judge_record = kwargs.pop("judge_record", None)
            reasoning_effort = kwargs.pop("reasoning_effort", None)
            try:
                (
                    content,
                    first_time,
                    all_time,
                    prompt_tokens,
                    completion_tokens,
                    query_id,
                    token_per_second,
                    itl_intervals,
                ) = func(*args, **kwargs)
                if not record_stats:
                    return content
                logging.info(
                    "## REQUEST_STAT success %s %s %s %s %s %s",
                    first_time,
                    all_time,
                    prompt_tokens,
                    completion_tokens,
                    query_id,
                    token_per_second,
                )

                tpot = 0.0
                if completion_tokens > 1:
                    tpot = (all_time - first_time) / (completion_tokens - 1)

                ts, ts_iso = _now_log_ts()
                detailed_request_data = {
                    "timestamp": ts,
                    "timestamp_iso": ts_iso,
                    "request_id": query_id,
                    "prompt_len": prompt_tokens,
                    "output_len": completion_tokens,
                    "ttft": first_time,
                    "itl": itl_intervals,
                    "tpot": tpot,
                    "e2e_lat": all_time,
                    "token_per_second": token_per_second,
                    "error": None,
                    "finished": True,
                }
                if os.environ.get("SLA_BENCH_SAVE_TEXT") == "1":
                    detailed_request_data["generated_text"] = content
                if sample_index_kw is not None:
                    detailed_request_data["sample_index"] = sample_index_kw
                if reasoning_effort:
                    detailed_request_data["reasoning_effort"] = reasoning_effort

                if judge_record and (judge_record.get("answers") or judge_record.get("gold_ans")):
                    try:
                        judged = score_one(
                            content,
                            judge_record.get("answers"),
                            judge_record.get("gold_ans"),
                            recall_threshold=float(os.environ.get("ACC_RECALL_THRESHOLD", "0.2")),
                        )
                        detailed_request_data.update({
                            "sample_index": judge_record.get("_sample_index", sample_index_kw),
                            "n_tokens": judge_record.get("n_tokens"),
                            "extracted_answer": judged["extracted_answer"],
                            "score": judged["score"],
                            "keyword_recall": judged["keyword_recall"],
                            "gated": judged["gated"],
                            "gold_ans": judge_record.get("gold_ans"),
                            "answers": judge_record.get("answers"),
                            "source_dataset": judge_record.get("source_dataset"),
                            "tier": judge_record.get("tier"),
                        })
                    except Exception:
                        logging.exception("acc judge failed")

                # Server-side count of real answer tokens before the pad delimiter. Runs after
                # `all_time` was captured, so it never enters TTFT/E2E of this request.
                try:
                    detailed_request_data["eff_len"] = count_effective_tokens(content)
                except Exception:
                    logging.warning("eff_len via server /tokenize failed", exc_info=True)

                self._write_to_jsonl(detailed_request_data)

                request_meta_first["response"] = content
                request_meta_first["response_length"] = prompt_tokens
                request_meta_first["response_time"] = first_time

                request_meta_all["response"] = content
                request_meta_all["response_length"] = completion_tokens
                request_meta_all["response_time"] = all_time

                request_meta_token_per_second["response_time"] = token_per_second

                request_meta_tpot["response"] = content
                request_meta_tpot["response_length"] = completion_tokens
                request_meta_tpot["response_time"] = tpot

                for itl_value in itl_intervals:
                    self._request_event.fire(
                        request_type="trpc",
                        name="itl",
                        start_time=time.time(),
                        response_length=1,
                        response=content,
                        context={},
                        exception=None,
                        response_time=itl_value,
                    )

                self._request_event.fire(**request_meta_first)
                self._request_event.fire(**request_meta_all)
                self._request_event.fire(**request_meta_token_per_second)
                self._request_event.fire(**request_meta_tpot)

            except Exception as e:
                if not record_stats:
                    raise
                logging.exception("request error!")
                response_time = (time.perf_counter() - t) * 1000
                ts, ts_iso = _now_log_ts()
                detailed_request_data = {
                    "timestamp": ts,
                    "timestamp_iso": ts_iso,
                    "request_id": f"error_{int(time.time() * 1000)}",
                    "prompt_len": 0,
                    "output_len": 0,
                    "ttft": 0,
                    "itl": [],
                    "tpot": 0,
                    "e2e_lat": response_time,
                    "token_per_second": 0,
                    "error": str(e),
                    "finished": False,
                }
                if os.environ.get("SLA_BENCH_SAVE_TEXT") == "1":
                    detailed_request_data["prompt"] = args[0] if args else []
                    detailed_request_data["generated_text"] = ""
                if sample_index_kw is not None:
                    detailed_request_data["sample_index"] = sample_index_kw
                self._write_to_jsonl(detailed_request_data)
                request_meta_error["response_time"] = response_time
                request_meta_error["exception"] = e
                self._request_event.fire(**request_meta_error)
                return None
            return content

        return wrapper

    def _write_to_jsonl(self, data: dict):
        os.makedirs(self.report_dir, exist_ok=True)
        with DATA_LOCK:
            try:
                with open(self.jsonl_filename, "a", encoding="utf-8") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(data, f, ensure_ascii=False)
                    f.write("\n")
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logging.error("Error writing to JSONL file: %s", e)


def warmup_clients(count: int, url: str, model_name: str, api_key: str, config: dict, report_dir: str = "report"):
    def _make_one(_) -> ChatHttpClient:
        return ChatHttpClient(
            request_event=None,
            url=url,
            model_name=model_name,
            api_key=api_key,
            config=config,
            environment=None,
            report_dir=report_dir,
        )

    global HTTP_CLIENT_POOL
    with cf.ThreadPoolExecutor(max_workers=count) as ex:
        HTTP_CLIENT_POOL = list(ex.map(_make_one, range(count)))


def on_locust_init(environment, **_):
    connector_counter = (
        int(os.environ["CAT_LOCUST_WARMUP_BS"])
        if os.environ.get("CAT_LOCUST_WARMUP_BS") is not None
        else environment.parsed_options.num_users
    )
    num_requests = (
        int(os.environ["CAT_LOCUST_WARMUP_NUM_PROMPTS"])
        if os.environ.get("CAT_LOCUST_WARMUP_NUM_PROMPTS") is not None
        else connector_counter * 2
    )
    url = environment.parsed_options.url
    model_name = environment.parsed_options.model_name
    api_key = environment.parsed_options.api_key
    config = {
        "max_tokens": environment.parsed_options.max_tokens,
        "use_uuid": environment.parsed_options.use_uuid,
    }
    warmup_clients(connector_counter, url, model_name, api_key, config, report_dir=environment.parsed_options.report_dir)
    request_rate = float(environment.parsed_options.spawn_rate or 5.0)
    try:
        ServerWarmUp(
            url=url,
            num_requests=num_requests,
            request_rate=request_rate,
            prefix_len=0,
            range_ratio=0.0,
            input_len=int(os.environ.get("CAT_LOCUST_WARMUP_INPUT", "3500")),
            output_len=int(os.environ.get("CAT_LOCUST_WARMUP_OUTPUT", "1200")),
            max_concurrency=connector_counter,
        )()
    except Exception:
        logging.error("Failed to run warmup with exception: %s", traceback.format_exc())
        environment.runner.quit()


if os.environ.get("CAT_LOCUST_WARMUP_ENABLE") == "1":
    events.init.add_listener(on_locust_init)


@events.init_command_line_parser.add_listener
def add_custom_arguments(parser: LocustArgumentParser):
    # No hardcoded choices= allowlists: any --url, --file, and --model_name is accepted!
    parser.add_argument("--url", type=str, default=model_config.url, help="OpenAI chat/completions endpoint URL")
    parser.add_argument("--file", type=str, default=model_config.data_file, help="NDJSON prompt dataset file path")
    parser.add_argument("--model_name", type=str, default=model_config.model_name, help="Served model name")
    parser.add_argument(
        "--share_data",
        type=int,
        choices=model_config.share_data_list,
        default=model_config.share_data,
        help="0=per-user, 1=shared ring, 2=shared single-pass queue (v7 default)",
    )
    parser.add_argument(
        "--sample_type",
        type=int,
        choices=model_config.sample_type_list,
        default=model_config.sample_type,
        help="0=sequential (preserve serpentine order), 1=random shuffle",
    )
    parser.add_argument(
        "--use_uuid",
        type=int,
        choices=model_config.use_uuid_list,
        default=model_config.use_uuid,
        help="1=prepend random UUID to bypass prefix cache (do not use for accuracy scoring)",
    )
    parser.add_argument("--max_tokens", type=int, default=0, help="Max output tokens (0=unlimited)")
    parser.add_argument("--api_key", type=str, default="", help="Bearer API key (optional)")
    parser.add_argument("--ramp_up_strategy", type=str, default="none", help="none/linear/exponential")
    parser.add_argument("--ramp_up_start_rps", type=float, default=1.0, help="Ramp-up start RPS")
    parser.add_argument("--ramp_up_end_rps", type=float, default=1.0, help="Ramp-up end RPS")
    parser.add_argument("--burstiness", type=float, default=1.0, help="Gamma arrival shape (1.0 = Poisson)")
    parser.add_argument("--report_dir", type=str, default="report", help="Output directory for pre_detail_stat_*.jsonl")
    parser.add_argument(
        "--send_mode",
        type=str,
        default="rate_limit",
        choices=["rate_limit", "after_prev"],
        help="rate_limit=open-loop arrival timetable; after_prev=closed-loop concurrency",
    )
    parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Dataset fraction (0-1]")
    parser.add_argument("--dataset_seed", type=int, default=0, help="Dataset sampling seed")
    parser.add_argument("--num_prompts", type=int, default=0, help="Number of prompts to sample (0=all)")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="",
        choices=["", "no_think", "low", "medium", "high"],
        help="Reasoning effort (empty or no_think disables thinking)",
    )
    parser.add_argument(
        "--bench_mode",
        type=str,
        default="e2e",
        choices=["e2e", "prefill", "decode"],
        help="e2e / prefill / decode",
    )
    parser.add_argument("--prefill_warmup", type=int, default=1, choices=[0, 1], help="Warm up prefix cache in decode mode")


class LLmuser(ConstantRateUser):

    def __init__(self, environment):
        super().__init__(environment)
        self.env = environment
        self.url = self.env.parsed_options.url
        self.file = self.env.parsed_options.file
        self.model_name = self.env.parsed_options.model_name
        self.share_data = self.env.parsed_options.share_data
        self.sample_type = self.env.parsed_options.sample_type
        self.api_key = self.env.parsed_options.api_key

        bench_mode = getattr(self.env.parsed_options, "bench_mode", "e2e")
        use_uuid = self.env.parsed_options.use_uuid
        max_tokens = self.env.parsed_options.max_tokens
        if bench_mode == "prefill":
            use_uuid = 1
            max_tokens = 1
        elif bench_mode == "decode":
            use_uuid = 0
        self.config = {
            "use_uuid": use_uuid,
            "max_tokens": max_tokens,
        }
        self.bench_mode = bench_mode
        if os.environ.get("CAT_LOCUST_WARMUP_ENABLE") == "1":
            self.client = HTTP_CLIENT_POOL.pop()
            self.client._request_event = environment.events.request
            self.client.environment = environment
        else:
            self.client = ChatHttpClient(
                request_event=environment.events.request,
                url=self.url,
                model_name=self.model_name,
                api_key=self.api_key,
                config=self.config,
                environment=environment,
                report_dir=self.env.parsed_options.report_dir,
            )

    def on_start(self):
        logging.info("## REQUEST_STAT user_start %s", time.time())

    def on_stop(self):
        logging.info("## REQUEST_STAT user_end %s", time.time())
        # Do NOT reset `read_file_flag` here. It used to be cleared whenever the concurrency
        # counter reached zero, so any later user to start re-read the file and rebuilt
        # `data_list` from scratch -- which is exactly how a 316-prompt file got served 848
        # times with each prompt repeated up to 3x. The dataset is loaded once per Locust
        # process; users that outlive the queue must exit, not reload it.
        _ = DATA_LOCK

    @tag("mix")
    @task
    def grpc_client_task(self):
        index = 0
        data_len = len(self.env.shared_data["data_list"])
        random_index = list(range(data_len))
        random.shuffle(random_index)
        while True:
            self.Wait()
            sample_index = None
            if self.share_data == 1:
                with DATA_LOCK:
                    line = self.env.shared_data["data_list"].pop(0)
                    self.env.shared_data["data_list"].append(line)
            elif self.share_data == 2:
                with DATA_LOCK:
                    line = self.env.shared_data["data_list"].pop(0) if self.env.shared_data["data_list"] else None
                    if line is not None:
                        total = int(self.env.shared_data.get("acc_dataset_total") or 0)
                        remaining = len(self.env.shared_data["data_list"])
                        sample_index = (total - remaining - 1) if total else None
                if line is None:
                    with DATA_LOCK:
                        self.env.shared_data["concurrency_num"] -= 1
                        last = self.env.shared_data["concurrency_num"] == 0
                    if last:
                        time.sleep(1)
                        self.env.runner.quit()
                    # Queue drained: this user is done. The old code slept 600 s here, which
                    # kept the process alive long after the last prompt was served.
                    raise StopUser("dataset queue drained")
            elif self.sample_type == 1:
                sample_index = random_index[index % data_len]
                line = self.env.shared_data["data_list"][sample_index]
                index += 1
            else:
                line = self.env.shared_data["data_list"][index % data_len]
                sample_index = index % data_len
                index += 1

            record = json.loads(line)
            if isinstance(record, dict) and record.get("_sample_index") is None:
                if self.share_data == 2 and sample_index is not None:
                    record["_sample_index"] = sample_index
                elif self.share_data != 2:
                    record["_sample_index"] = sample_index

            effective_sample_index = (
                record.get("_sample_index", sample_index) if isinstance(record, dict) else sample_index
            )
            messages = record["messages"] if isinstance(record, dict) and "messages" in record else record
            extra = {"_sample_index": effective_sample_index}
            effort = getattr(self.env.parsed_options, "reasoning_effort", "") or ""
            effort = effort or "no_think"
            model_name = getattr(self.env.parsed_options, "model_name", "") or ""
            extra["chat_template_kwargs"] = default_chat_template_kwargs(model_name, effort)
            extra["reasoning_effort"] = effort
            if isinstance(messages, list):
                messages = [
                    m
                    for m in messages
                    if not (
                        isinstance(m, dict)
                        and m.get("role") == "assistant"
                        and not (m.get("content") or "").strip()
                        and not m.get("tool_calls")
                    )
                ]
            if self.bench_mode == "decode":
                self.client.send_stream(messages=messages, max_tokens=1, record_stats=False, **extra)
            if isinstance(record, dict) and record.get("max_tokens") is not None:
                try:
                    extra["max_tokens"] = int(record["max_tokens"])
                except (TypeError, ValueError):
                    pass
            if isinstance(record, dict) and (record.get("answers") or record.get("gold_ans")):
                extra["judge_record"] = record
            self.client.send_stream(messages=messages, **extra)
