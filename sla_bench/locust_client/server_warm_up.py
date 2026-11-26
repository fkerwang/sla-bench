"""Optional synthetic server warm-up helper."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import requests

try:
    from . import chat
except ImportError:
    import chat  # type: ignore


@dataclass
class SampleRequest:
    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int


class RandomDataset:
    DEFAULT_SEED = 2025
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(self, random_seed: int = DEFAULT_SEED, **_kwargs) -> None:
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    # Token ids drawn from a range that exists in every modern vocab (>=32k);
    # real splitting/decoding is done server-side via /tokenize + /detokenize.
    ID_LOW = 1000
    ID_HIGH = 30000

    def sample(
        self,
        tokenizer,  # sla_bench.locust_client.remote_tok.RemoteTokenizer
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **_kwargs,
    ) -> list[SampleRequest]:
        assert range_ratio < 1.0, "random_range_ratio must be < 1.0"
        span = self.ID_HIGH - self.ID_LOW
        real_input_len = input_len  # /tokenize is called with add_special_tokens=False
        prefix_token_ids = (
            (np.random.randint(0, span, size=prefix_len) + self.ID_LOW).tolist() if prefix_len > 0 else []
        )
        input_low = int(real_input_len * (1 - range_ratio))
        input_high = int(real_input_len * (1 + range_ratio))
        output_low = int(output_len * (1 - range_ratio))
        output_high = int(output_len * (1 + range_ratio))

        input_lens = np.random.randint(input_low, input_high + 1, size=num_requests)
        output_lens = np.random.randint(output_low, output_high + 1, size=num_requests)
        offsets = np.random.randint(0, span, size=num_requests)

        out: list[SampleRequest] = []
        for i in range(num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) % span + self.ID_LOW).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            total_input_len = prefix_len + int(input_lens[i])
            re_encoded_sequence = tokenizer.encode_text(prompt)[:total_input_len]
            prompt = tokenizer.decode(re_encoded_sequence)
            out.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=len(re_encoded_sequence),
                    expected_output_len=int(output_lens[i]),
                )
            )
        return out


class ServerWarmUp:
    def __init__(
        self,
        url: str,
        num_requests: int = 256,
        request_rate: float = 5.0,
        prefix_len: int = 0,
        range_ratio: float = 0.0,
        input_len: int = 3500,
        output_len: int = 1200,
        max_concurrency: int = 5,
    ):
        self.url = url
        self.num_requests = num_requests
        self.request_rate = request_rate
        self.prefix_len = prefix_len
        self.range_ratio = range_ratio
        self.input_len = input_len
        self.output_len = output_len
        self.max_concurrency = max_concurrency

    def call_generate(self, sample_request: SampleRequest):
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [{"role": "user", "content": sample_request.prompt}],
            "max_tokens": sample_request.expected_output_len,
        }
        if chat.get_tokenizer_model():
            data["model"] = chat.get_tokenizer_model()
        response = requests.post(self.url, headers=headers, data=json.dumps(data))
        out = response.json()
        try:
            return out["choices"][0]["message"]["content"]
        except Exception:
            return "X"

    def do_eval_parallel(self, sample_requests: List[SampleRequest]):
        period = 1.0 / self.request_rate
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = []
            for i in range(len(sample_requests)):
                futures.append(executor.submit(self.call_generate, sample_requests[i]))
                time.sleep(period)
            for _ in as_completed(futures):
                pass

    def __call__(self):
        tokenizer = chat.get_remote_tokenizer(self.url)
        dataset = RandomDataset()
        sample_requests = dataset.sample(
            tokenizer,
            num_requests=self.num_requests,
            prefix_len=self.prefix_len,
            range_ratio=self.range_ratio,
            input_len=self.input_len,
            output_len=self.output_len,
        )
        self.do_eval_parallel(sample_requests)
        time.sleep(1)
