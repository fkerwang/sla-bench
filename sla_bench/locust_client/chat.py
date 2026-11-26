"""Audited v7 OpenAI SSE streaming client.

Key v7 guarantees preserved:
1. Prompt tokenization runs ONLY AFTER the SSE stream completes (`all_time`
   captured first), so client CPU tokenization never enters TTFT.
2. `CAT_LOCUST_PAD_TO_TOKENS` appends `PAD_INSTRUCTION` (`===ANSWER_END===`)
   and sets `ignore_eos=True`, `min_tokens=N`, `max_tokens=N`, `skip_special_tokens=False`
   to enforce constant decode work in a single pass.
3. Quiet per-request summary logging by default so gevent formatting of 1000 ITL
   floats does not perturb concurrent request timing (`CAT_LOCUST_LOG_FULL=1` opt-in).
4. Token counts come from the server (`usage`, falling back to `/tokenize`); no local
   tokenizer or `transformers` is ever loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import traceback
import uuid

import requests
import sseclient

try:
    from . import model_config
    from .remote_tok import RemoteTokenizer, derive_tokenize_url
except ImportError:
    import model_config  # type: ignore
    from remote_tok import RemoteTokenizer, derive_tokenize_url  # type: ignore

_TOK_LOCK = threading.Lock()
_REMOTE_TOK = None
_REMOTE_TOK_KEY = None
_DEFAULT_CHAT_URL = ""

PAD_DELIM = model_config.PAD_DELIM
PAD_INSTRUCTION = model_config.PAD_INSTRUCTION


def get_tokenizer_url() -> str:
    """`/tokenize` endpoint: explicit env override, else derived from the chat URL."""
    url = os.environ.get("CAT_LOCUST_TOKENIZER_URL", "")
    if url:
        return url
    return derive_tokenize_url(_DEFAULT_CHAT_URL) if _DEFAULT_CHAT_URL else ""


def get_tokenizer_model() -> str:
    return os.environ.get("CAT_LOCUST_TOKENIZER_MODEL", "")


def get_remote_tokenizer(default_chat_url: str = "") -> RemoteTokenizer:
    """Server-side tokenizer client (no local tokenizer/transformers is ever loaded)."""
    global _REMOTE_TOK, _REMOTE_TOK_KEY, _DEFAULT_CHAT_URL
    if default_chat_url and not _DEFAULT_CHAT_URL:
        _DEFAULT_CHAT_URL = default_chat_url
    url = get_tokenizer_url()
    if not url:
        raise RuntimeError(
            "No tokenizer endpoint: set CAT_LOCUST_TOKENIZER_URL or construct a ChatService with a chat URL."
        )
    key = (url, get_tokenizer_model())
    if _REMOTE_TOK is not None and _REMOTE_TOK_KEY == key:
        return _REMOTE_TOK
    with _TOK_LOCK:
        if _REMOTE_TOK is None or _REMOTE_TOK_KEY != key:
            _REMOTE_TOK = RemoteTokenizer(url, model=key[1])
            _REMOTE_TOK_KEY = key
        return _REMOTE_TOK


def get_tokens(text: str):
    return get_remote_tokenizer().encode_text(text)


def count_effective_tokens(content: str):
    """Server-counted tokens of the answer before `PAD_DELIM`; None if the delimiter is absent."""
    cut = (content or "").find(PAD_DELIM)
    if cut < 0:
        return None
    return len(get_tokens(content[:cut]))


def get_tokens_from_messages(messages, thinking_mode=None, chat_template_kwargs=None):
    if chat_template_kwargs is None and thinking_mode is not None:
        chat_template_kwargs = {"thinking": bool(thinking_mode)}
    return get_remote_tokenizer().encode_messages(messages, chat_template_kwargs)


_WARNED_UNLIMITED_DEFAULT_MODEL_LEN = False


def resolve_max_tokens(messages, requested: int, chat_template_kwargs=None) -> int:
    """max_tokens<=0 means use remaining context window (server-side natural stop)."""
    if requested > 0:
        return requested
    global _WARNED_UNLIMITED_DEFAULT_MODEL_LEN
    if "CAT_LOCUST_MAX_MODEL_LEN" not in os.environ and not _WARNED_UNLIMITED_DEFAULT_MODEL_LEN:
        _WARNED_UNLIMITED_DEFAULT_MODEL_LEN = True
        logging.warning(
            "## max_tokens=0 (unlimited) but CAT_LOCUST_MAX_MODEL_LEN is not set — falling back to 168000. "
            "Export CAT_LOCUST_MAX_MODEL_LEN=<target's real max_model_len> before running."
        )
    max_model_len = int(os.environ.get("CAT_LOCUST_MAX_MODEL_LEN", "168000"))
    if chat_template_kwargs and chat_template_kwargs.get("thinking") is not None:
        prompt_tokens = len(get_tokens_from_messages(messages, thinking_mode=chat_template_kwargs["thinking"]))
    else:
        prompt_tokens = len(get_tokens_from_messages(messages))
    return max(256, max_model_len - prompt_tokens - 16)


def modify_list_with_uuid(original_list):
    new_list = []
    for index, item in enumerate(original_list):
        new_item = item.copy()
        if index == 0 or index == 1:
            random_uuid = str(uuid.uuid4())
            new_item["content"] = f"{random_uuid} {new_item['content']}"
        new_list.append(new_item)
    return new_list


_LOCAL_KWARGS = frozenset({
    "output_callback",
    "record_stats",
    "judge_record",
    "reasoning_effort",
    "abort_on_repeat",
})
_MAPPED_KWARGS = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "logit_bias",
    "max_tokens",
    "chat_template_kwargs",
    "extra_body",
    "extra",
})


def _is_degenerate_repeat(text: str) -> bool:
    """Detect death-spiral repeats using zlib compressibility of the recent window."""
    import zlib

    if len(text) < 800:
        return False
    window = text[-1600:]
    raw = window.encode("utf-8", errors="ignore")
    if len(raw) < 600:
        return False
    ratio = len(zlib.compress(raw, 9)) / len(raw)
    if ratio < 0.12:
        return True
    gram = window[-24:]
    if len(gram) == 24 and window.count(gram) >= 12:
        return True
    return False


def _load_extra_body(kwargs) -> dict:
    extra = {}
    env_raw = os.environ.get("CAT_LOCUST_EXTRA_BODY")
    if env_raw:
        try:
            parsed = json.loads(env_raw)
            if isinstance(parsed, dict):
                extra.update(parsed)
            else:
                logging.warning("CAT_LOCUST_EXTRA_BODY must be a JSON object, ignoring")
        except json.JSONDecodeError:
            logging.warning("CAT_LOCUST_EXTRA_BODY is not valid JSON, ignoring")
    for key in ("extra_body", "extra"):
        body = kwargs.get(key)
        if isinstance(body, dict):
            extra.update(body)
    return extra


def _passthrough_request_kwargs(kwargs) -> dict:
    skip = _LOCAL_KWARGS | _MAPPED_KWARGS
    return {k: v for k, v in kwargs.items() if k not in skip and v is not None}


def _merge_into(dst: dict, overlay: dict) -> None:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merged = dict(dst[k])
            merged.update(v)
            dst[k] = merged
        else:
            dst[k] = v


def _apply_request_extras(json_data: dict, kwargs) -> dict:
    extras = _load_extra_body(kwargs)
    passthrough = _passthrough_request_kwargs(kwargs)
    if extras:
        _merge_into(json_data, extras)
    if passthrough:
        _merge_into(json_data, passthrough)
        logging.info("## request passthrough keys: %s", sorted(passthrough))
    return json_data


class ChatService(object):

    def __init__(
        self,
        url,
        model,
        api_key="<API_KEY>",
        wsid="11294",
        config=None,
    ) -> None:
        global _DEFAULT_CHAT_URL
        self.url = url
        self.model = model
        if not _DEFAULT_CHAT_URL:
            _DEFAULT_CHAT_URL = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Wsid": wsid,
        }
        self.wsid = wsid
        self.config = config or {}

    def send_stream(self, messages, output_callback=None, **kwargs):
        if self.wsid == "":
            return self.send_stream_lkeap(messages, output_callback=output_callback, **kwargs)
        return self.send_stream_taiji(messages, output_callback=output_callback)

    def send_stream_taiji(self, messages, output_callback=None):
        enable_stream = True
        start_perf_counter = time.perf_counter()
        query_id = "test_" + str(uuid.uuid4()).replace("-", "", -1)
        json_data = {
            "model": self.model,
            "query_id": query_id,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1,
            "output_seq_len": 4096,
            "max_input_seq_len": 128000,
            "stream": enable_stream,
            "random_seed": 1234,
        }
        response = requests.post(self.url, headers=self.headers, json=json_data, stream=enable_stream)
        content = ""
        index = 0
        token_info = {}
        first_time = 0
        client = sseclient.SSEClient(response)
        for event in client.events():
            if index == 0:
                first_time = (time.perf_counter() - start_perf_counter) * 1000
            index += 1
            if "[DONE]" in event.data:
                break
            chunk = json.loads(event.data)
            try:
                if chunk["object"] == "stream_server.event":
                    chunk_event = chunk.get("event")
                    if chunk_event.get("name", "") == "thinking" and chunk_event.get("state", -1) == 0:
                        delta = "<think>"
                        content += delta
                        if output_callback:
                            output_callback(delta)
                    elif chunk_event.get("name", "") == "thinking" and chunk_event.get("state", -1) == 2:
                        delta = "</think>"
                        content += delta
                        if output_callback:
                            output_callback(delta)
                    continue
                elif chunk["object"] == "chat.completion.chunk":
                    if chunk["choices"][0]["finish_reason"] != "stop":
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        content += delta
                        if output_callback:
                            output_callback(delta)
            except Exception:
                print(f"{traceback.format_exc()}")
        all_time = (time.perf_counter() - start_perf_counter) * 1000
        token_info["prompt_tokens"] = len(get_tokens_from_messages(messages))
        token_info["completion_tokens"] = len(get_tokens(content))
        token_time = all_time - first_time
        token_info["token_per_second"] = token_info["completion_tokens"] / token_time if token_time > 0 else None
        if token_info["completion_tokens"] == 0:
            raise Exception("completion_tokens is 0!")
        return content, first_time, all_time, token_info["prompt_tokens"], token_info["completion_tokens"], query_id, token_info["token_per_second"], []

    def send_stream_lkeap(self, messages, output_callback=None, **kwargs):
        if int(self.config.get("use_uuid", 0)) == 1:
            messages = modify_list_with_uuid(messages)

        _pad_to = int(os.environ.get("CAT_LOCUST_PAD_TO_TOKENS", 0) or 0)
        if _pad_to and messages:
            messages = [dict(m) for m in messages]
            messages[-1]["content"] = (messages[-1].get("content") or "") + PAD_INSTRUCTION.format(delim=PAD_DELIM)

        enable_stream = True
        start_perf_counter = time.perf_counter()
        query_id = "test_" + str(uuid.uuid4()).replace("-", "", -1)

        temperature = (
            float(os.environ["CAT_LOCUST_REQUEST_TEMPERATURE"])
            if os.environ.get("CAT_LOCUST_REQUEST_TEMPERATURE") is not None
            else kwargs.get("temperature", 0.5)
        )
        top_p = (
            float(os.environ["CAT_LOCUST_REQUEST_TOP_P"])
            if os.environ.get("CAT_LOCUST_REQUEST_TOP_P") is not None
            else kwargs.get("top_p", 0.9)
        )
        top_k = (
            int(os.environ["CAT_LOCUST_REQUEST_TOP_K"])
            if os.environ.get("CAT_LOCUST_REQUEST_TOP_K") is not None
            else kwargs.get("top_k", 40)
        )
        repetition_penalty = (
            float(os.environ["CAT_LOCUST_REQUEST_REPETITION_PENALTY"])
            if os.environ.get("CAT_LOCUST_REQUEST_REPETITION_PENALTY") is not None
            else kwargs.get("repetition_penalty", 1.0)
        )
        frequency_penalty = kwargs.get("frequency_penalty")
        presence_penalty = kwargs.get("presence_penalty")

        chat_template_kwargs = kwargs.get("chat_template_kwargs")
        requested_max = int(kwargs.get("max_tokens", self.config.get("max_tokens", 4096)))
        effective_max = resolve_max_tokens(messages, requested_max, chat_template_kwargs)

        json_data = {
            "model": self.model,
            "query_id": query_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_tokens": effective_max,
            "max_input_seq_len": 128000,
            "stream": enable_stream,
            "stream_options": {"include_usage": True},
            "random_seed": int(os.environ.get("CAT_LOCUST_REQUEST_SEED", 1234)),
        }
        if frequency_penalty is not None:
            json_data["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            json_data["presence_penalty"] = float(presence_penalty)
        if kwargs.get("stop") is not None:
            json_data["stop"] = kwargs["stop"]
        if kwargs.get("logit_bias") is not None:
            json_data["logit_bias"] = kwargs["logit_bias"]
        if chat_template_kwargs is not None:
            json_data["chat_template_kwargs"] = chat_template_kwargs

        if _pad_to:
            json_data["ignore_eos"] = True
            json_data["min_tokens"] = _pad_to
            json_data["max_tokens"] = _pad_to
            json_data["skip_special_tokens"] = False
        _apply_request_extras(json_data, kwargs)

        if "volceapi" in self.url:
            keys_to_keep = ["role", "content"]
            messages = [{key: item[key] for key in keys_to_keep if key in item} for item in messages]
            json_data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "max_tokens": int(kwargs.get("max_tokens", self.config.get("max_tokens", 4096))),
                "stream": enable_stream,
            }
            _apply_request_extras(json_data, kwargs)

        if "console.tenstorrent.com" in self.url:
            for k in ("query_id", "max_input_seq_len", "random_seed"):
                json_data.pop(k, None)

        response = requests.post(self.url, headers=self.headers, json=json_data, stream=enable_stream)
        if response.status_code != 200:
            raise Exception(f"status_code: {response.status_code}")

        content = ""
        index = 0
        token_info = {}
        first_time = 0.0
        status = "init"

        token_count = 0
        last_token_time = None
        itl_intervals = []
        itl_timestamps = []

        abort_on_repeat = bool(kwargs.get("abort_on_repeat")) or (
            os.environ.get("CAT_LOCUST_ABORT_ON_REPEAT", "").strip() in ("1", "true", "True")
        )
        aborted_repeat = False
        client = sseclient.SSEClient(response)
        for event in client.events():
            current_time = time.perf_counter()
            if index == 0:
                first_time = (time.perf_counter() - start_perf_counter) * 1000
            index += 1
            if "[DONE]" in event.data:
                break
            try:
                chunk = json.loads(event.data)
                if "object" in chunk and chunk["object"] == "stream_server.event":
                    chunk_event = chunk.get("event")
                    if chunk_event.get("name", "") == "thinking" and chunk_event.get("state", -1) == 0:
                        delta = "<think>"
                        content += delta
                        if output_callback:
                            output_callback(delta)
                    elif chunk_event.get("name", "") == "thinking" and chunk_event.get("state", -1) == 2:
                        delta = "</think>"
                        content += delta
                        if output_callback:
                            output_callback(delta)
                    continue
                elif "object" in chunk and chunk["object"] == "chat.completion.chunk" and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    reasoning_content = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    output_content = delta.get("content") or ""

                    has_new_token = False
                    if reasoning_content != "" and status == "init":
                        content += "<think>"
                        status = "thinking"
                        has_new_token = True
                    if reasoning_content != "":
                        content += reasoning_content
                        has_new_token = True
                    if output_content != "" and status == "thinking":
                        content += "</think>"
                        status = "output"
                        has_new_token = True
                    if output_content != "":
                        content += output_content
                        has_new_token = True

                    if has_new_token and token_count > 0:
                        interval = (current_time - last_token_time) * 1000
                        itl_intervals.append(interval)
                        itl_timestamps.append(current_time * 1000)

                    if has_new_token:
                        token_count += 1
                        last_token_time = current_time

                    if output_callback:
                        output_callback(content)
                    if chunk.get("usage"):
                        token_info["completion_tokens"] = chunk["usage"]["completion_tokens"]
                        if chunk["usage"].get("prompt_tokens"):
                            token_info["prompt_tokens"] = chunk["usage"]["prompt_tokens"]
                    if abort_on_repeat and has_new_token and token_count >= 256 and token_count % 32 == 0:
                        if _is_degenerate_repeat(content):
                            aborted_repeat = True
                            logging.warning(
                                "## abort_on_repeat: query_id=%s chars=%s tokens~%s",
                                query_id,
                                len(content),
                                token_count,
                            )
                            break
                elif "usage" in chunk and isinstance(chunk["usage"], dict):
                    token_info["server_token_stats"] = chunk["usage"]
                    token_info["completion_tokens"] = chunk["usage"]["completion_tokens"]
                    if chunk["usage"].get("prompt_tokens"):
                        token_info["prompt_tokens"] = chunk["usage"]["prompt_tokens"]
            except Exception as e:
                print(f"error: {e}, {traceback.format_exc()}")

        if aborted_repeat:
            try:
                response.close()
            except Exception:
                pass
            token_info["completion_tokens"] = len(get_tokens(content))
            token_info["aborted_repeat"] = True

        # Capture end-to-end latency BEFORE any client-side tokenization!
        all_time = (time.perf_counter() - start_perf_counter) * 1000

        if "completion_tokens" not in token_info:
            logging.info("Failed to find server usage stats, get completion_tokens by local tokenizer")
            token_info["completion_tokens"] = len(get_tokens(content))

        if _pad_to and token_info["completion_tokens"] < _pad_to:
            logging.warning(
                "## short reply %s/%s tokens: ignore_eos not honoured and the "
                "model stopped despite the padding instruction",
                token_info["completion_tokens"],
                _pad_to,
            )

        # v7 fix: prompt tokenization happens strictly AFTER stream completion so
        # client CPU time never contaminates TTFT.
        if "prompt_tokens" not in token_info:
            token_info["prompt_tokens"] = len(
                get_tokens_from_messages(messages, chat_template_kwargs=json_data.get("chat_template_kwargs"))
            )
        token_time = all_time - first_time
        token_info["token_per_second"] = (
            token_info["completion_tokens"] / token_time * 1000 if token_time > 0 else None
        )
        token_info["itl_intervals"] = itl_intervals
        token_info["itl_timestamps"] = itl_timestamps

        if os.environ.get("CAT_LOCUST_LOG_FULL", "") in ("1", "true", "True"):
            prompt_text = json.dumps(messages, ensure_ascii=False)
            message_unq_id = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()
            logging.info(
                "query_id: %s, message_unq_id: %s, message: %s, content: %s, token_info:%s, first_time: %s, all_time: %s, token_per_second: %s, itl_count: %s",
                query_id,
                message_unq_id,
                messages,
                content,
                token_info,
                first_time,
                all_time,
                token_info["token_per_second"],
                len(itl_intervals),
            )
        else:
            logging.info(
                "query_id: %s, prompt_tokens: %s, completion_tokens: %s, first_time: %.1f, all_time: %.1f",
                query_id,
                token_info.get("prompt_tokens"),
                token_info.get("completion_tokens"),
                first_time,
                all_time,
            )
        if token_info["completion_tokens"] == 0:
            raise Exception("completion_tokens is 0!")
        return (
            content,
            first_time,
            all_time,
            token_info["prompt_tokens"],
            token_info["completion_tokens"],
            query_id,
            token_info["token_per_second"],
            token_info["itl_intervals"],
        )
