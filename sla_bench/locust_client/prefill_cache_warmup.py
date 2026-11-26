"""Warm up server prefix cache by sending max_tokens=1 for every dataset prompt."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

try:
    from .chat import ChatService
except ImportError:
    from chat import ChatService  # type: ignore


def _load_messages(data_file: str, dataset_fraction: float = 1.0) -> List[list]:
    path = Path(data_file)
    messages_list = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj["messages"] if "messages" in obj else obj
            messages_list.append(messages)
    if dataset_fraction < 1.0:
        keep = max(1, int(len(messages_list) * dataset_fraction))
        messages_list = messages_list[:keep]
    return messages_list


class DatasetPrefillWarmup:
    """Send each dataset prompt once with max_tokens=1 and use_uuid=0 to populate prefix cache."""

    def __init__(
        self,
        url: str,
        data_file: str,
        model_name: str,
        api_key: str = "",
        dataset_fraction: float = 1.0,
        max_concurrency: int = 8,
        request_rate: float = 2.0,
        reasoning_effort: str = "",
    ):
        self.url = url
        self.data_file = data_file
        self.model_name = model_name
        self.api_key = api_key
        self.dataset_fraction = dataset_fraction
        self.max_concurrency = max_concurrency
        self.request_rate = request_rate
        self.reasoning_effort = reasoning_effort

    def _send_one(self, service: ChatService, messages: list):
        extra = {}
        effort = self.reasoning_effort or ""
        if effort and effort != "no_think":
            extra["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": effort,
            }
        elif effort == "no_think":
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        service.send_stream(messages=messages, **extra)

    def __call__(self):
        messages_list = _load_messages(self.data_file, self.dataset_fraction)
        if not messages_list:
            raise RuntimeError(f"no prompts loaded from {self.data_file}")

        service = ChatService(
            url=self.url,
            model=self.model_name,
            api_key=self.api_key,
            wsid="",
            config={"use_uuid": 0, "max_tokens": 1},
        )

        logging.info(
            "dataset prefill warmup: %d prompts, concurrency=%d, rate=%.1f/s",
            len(messages_list),
            self.max_concurrency,
            self.request_rate,
        )
        tic = time.time()
        period = 1.0 / self.request_rate if self.request_rate > 0 else 0.0
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = []
            for messages in messages_list:
                futures.append(executor.submit(self._send_one, service, messages))
                if period > 0:
                    time.sleep(period)
            for f in as_completed(futures):
                f.result()

        logging.info("dataset prefill warmup done in %.1fs", time.time() - tic)
