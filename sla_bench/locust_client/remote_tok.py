"""Server-side tokenization helpers (vLLM-compatible `/tokenize` and `/detokenize`).

All token counting/splitting is delegated to the serving endpoint, so counts always
match the model actually under test and no local `transformers` install is needed.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

import requests


def _root(chat_url: str) -> str:
    p = urlsplit(chat_url)
    if not p.scheme or not p.netloc:
        raise ValueError(f"Cannot derive server root from URL: {chat_url!r}")
    return f"{p.scheme}://{p.netloc}"


def derive_tokenize_url(chat_url: str) -> str:
    """`http://h:p/v1/chat/completions` -> `http://h:p/tokenize`."""
    return _root(chat_url) + "/tokenize"


def derive_detokenize_url(tokenize_url: str) -> str:
    return tokenize_url.rsplit("/", 1)[0] + "/detokenize"


def default_chat_template_kwargs(model_name: str = "", effort: str = "") -> dict:
    """`chat_template_kwargs` sent with every request; shared by the client and `dataset-info`
    so server-side token counts of the dataset match what the client really sends."""
    effort = effort or "no_think"
    name = (model_name or "").lower()
    if "hy3" in name or "hunyuan" in name:
        if effort == "medium":
            effort = "high"
        elif effort not in ("low", "high", "no_think"):
            effort = "no_think"
        return {"reasoning_effort": effort}
    if effort != "no_think":
        return {"enable_thinking": True, "reasoning_effort": effort}
    return {"enable_thinking": False, "reasoning_effort": "no_think"}


class RemoteTokenizer:
    """Thin client over the server's `/tokenize` + `/detokenize` endpoints."""

    def __init__(self, url: str, model: str = "", timeout: float = 120.0):
        if not url:
            raise ValueError("RemoteTokenizer needs a /tokenize URL")
        self.url = url.rstrip("/") if url.rstrip("/").endswith("/tokenize") else derive_tokenize_url(url)
        self.model = model
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            self._local.s = s
        return s

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.model:
            payload = {"model": self.model, **payload}
        r = self._session().post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def encode_text(self, text: str) -> List[int]:
        return self._post(self.url, {"prompt": text, "add_special_tokens": False})["tokens"]

    def encode_messages(
        self,
        messages: Sequence[dict],
        chat_template_kwargs: Optional[dict] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        payload: Dict[str, Any] = {"messages": list(messages), "add_generation_prompt": add_generation_prompt}
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        return self._post(self.url, payload)["tokens"]

    def decode(self, tokens: Sequence[int]) -> str:
        out = self._post(derive_detokenize_url(self.url), {"tokens": list(tokens)})
        # vLLM replies {"prompt": ...}; SGLang replies {"text": ...}
        return out["prompt"] if "prompt" in out else out["text"]

    def count_texts(self, texts: Sequence[str], workers: int = 16) -> List[int]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            return list(ex.map(lambda t: len(self.encode_text(t)), texts))

    def count_messages_many(
        self,
        batch: Sequence[Sequence[dict]],
        chat_template_kwargs: Optional[dict] = None,
        workers: int = 16,
    ) -> List[int]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            return list(ex.map(lambda m: len(self.encode_messages(m, chat_template_kwargs)), batch))
