"""Audited v7 ConstantRateUser with lock-free absolute Poisson timetable.

Key v7 guarantees preserved:
1. `CAT_LOCUST_RATE` sets the target arrival rate independently of Locust's
   `-r` user spawn rate, so `-r` can spawn all users fast at startup.
2. When `CAT_LOCUST_ARRIVAL=poisson` and `ramp_up_strategy=none`, each user
   claims its target send timestamp on an absolute monotonic timetable under
   `DATA_LOCK` and sleeps OUTSIDE the lock, eliminating coordinated omission.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Literal, Optional

import numpy as np
from locust import User, task

DATA_LOCK = threading.Lock()


class ConstantRateUser(User):
    abstract = True
    loop_index = 0
    data_len = 0
    next_send = None

    def __init__(self, environment):
        super().__init__(environment)
        self.env = environment
        self.user_count = self.env.parsed_options.num_users if self.env.parsed_options.num_users else 1
        self.grps = self.env.parsed_options.spawn_rate if self.env.parsed_options.spawn_rate else self.user_count
        if os.environ.get("CAT_LOCUST_RATE"):
            self.grps = float(os.environ["CAT_LOCUST_RATE"])
        self.file = self.env.parsed_options.file
        self.share_data = self.env.parsed_options.share_data
        self.burstiness = self.env.parsed_options.burstiness if self.env.parsed_options.burstiness else 1.0
        self.sample_type = self.env.parsed_options.sample_type
        self.ramp_up_strategy = (
            self.env.parsed_options.ramp_up_strategy if self.env.parsed_options.ramp_up_strategy else "none"
        )
        self.ramp_up_start_rps = (
            self.env.parsed_options.ramp_up_start_rps if self.env.parsed_options.ramp_up_start_rps else 1.0
        )
        self.ramp_up_end_rps = (
            self.env.parsed_options.ramp_up_end_rps if self.env.parsed_options.ramp_up_end_rps else 1.0
        )

        with DATA_LOCK:
            if not hasattr(self.env, "shared_data"):
                self.env.shared_data = {
                    "read_file_flag": False,
                    "data_list": [],
                    "concurrency_num": 0,
                }
            if self.env.shared_data["read_file_flag"] is False:
                self.env.shared_data["read_file_flag"] = True
                data_list = []
                with open(self.file, "r", encoding="utf-8-sig") as f_mix:
                    for line in f_mix:
                        if line.strip():
                            data_list.append(line)
                data_list = data_list * int(os.environ.get("CAT_REPEAT_LOCUST_DATASET_NUM", "1"))
                dataset_seed = getattr(self.env.parsed_options, "dataset_seed", None)
                if dataset_seed is None:
                    dataset_seed = int(os.environ.get("CAT_LOCUST_DATASET_SEED", "0"))
                num_prompts = getattr(self.env.parsed_options, "num_prompts", None)
                if num_prompts is None:
                    num_prompts = int(os.environ.get("CAT_LOCUST_NUM_PROMPTS", "0") or "0")
                num_prompts = int(num_prompts or 0)
                rng = np.random.RandomState(int(dataset_seed))
                if num_prompts > 0:
                    n = len(data_list)
                    idx = rng.choice(n, size=num_prompts, replace=(num_prompts > n))
                    data_list = [data_list[int(i)] for i in idx]
                    logging.info(
                        "vLLM-style sample: num_prompts=%d seed=%s replace=%s -> %d items",
                        num_prompts,
                        dataset_seed,
                        num_prompts > n,
                        len(data_list),
                    )
                elif self.sample_type == 1:
                    idx = rng.permutation(len(data_list))
                    data_list = [data_list[int(i)] for i in idx]
                    logging.info("data shuffle seed=%s (numpy RandomState)", dataset_seed)
                self.env.shared_data["data_list"] = data_list
                dataset_fraction = getattr(self.env.parsed_options, "dataset_fraction", 1.0)
                if dataset_fraction < 1.0:
                    keep = max(1, int(len(self.env.shared_data["data_list"]) * dataset_fraction))
                    self.env.shared_data["data_list"] = self.env.shared_data["data_list"][:keep]
                    logging.info("dataset fraction %.0f%% -> %d items", dataset_fraction * 100, keep)
                ConstantRateUser.data_len = len(self.env.shared_data["data_list"])
                logging.info("=== dataset len: %d ===", ConstantRateUser.data_len)
                if getattr(self.env.parsed_options, "share_data", 0) == 2:
                    self.env.shared_data["acc_dataset_total"] = ConstantRateUser.data_len
            if self.share_data == 2:
                self.env.shared_data["concurrency_num"] += 1

    def _get_current_rate(
        self,
        ramp_up_strategy: Optional[Literal["linear", "exponential", "none"]],
        ramp_up_start_rps: Optional[float],
        ramp_up_end_rps: Optional[float],
        times: int,
        total_times: int,
    ) -> float:
        if ramp_up_strategy == "linear":
            return ramp_up_start_rps + (ramp_up_end_rps - ramp_up_start_rps) * (float(times) / max(float(total_times), 1.0))
        if ramp_up_strategy == "exponential":
            return ramp_up_start_rps * ((ramp_up_end_rps / ramp_up_start_rps) ** (float(times) / max(float(total_times), 1.0)))
        return float(self.grps)

    def should_rate_limit(self) -> bool:
        send_mode = getattr(self.env.parsed_options, "send_mode", "rate_limit")
        return send_mode != "after_prev"

    def Wait(self):
        if not self.should_rate_limit():
            return
        if os.environ.get("CAT_LOCUST_ARRIVAL", "periodic") == "poisson" and self.ramp_up_strategy == "none":
            # Absolute Poisson timetable: each user claims the next arrival time under the
            # lock and sleeps outside it.
            #
            # A claimed slot can be overtaken while a user sleeps: with N users, some wake late
            # (GIL contention, request bookkeeping) and then find their slot long past, so they
            # fire immediately and bunch up with whoever else is late. Measured at 0.1/s this
            # produced 2 simultaneous sends plus a 50 s hole. Re-checking after the sleep and
            # claiming a fresh slot when this one is stale (more than one mean interval late)
            # keeps the realised rate on target instead of bursting to catch up.
            mean_gap = 1.0 / max(self.grps * self.burstiness, 1e-9)
            with DATA_LOCK:
                now = time.monotonic()
                if ConstantRateUser.next_send is None:
                    ConstantRateUser.next_send = now
                ConstantRateUser.next_send += np.random.gamma(
                    shape=self.burstiness, scale=mean_gap / self.burstiness
                )
                due = ConstantRateUser.next_send
                ConstantRateUser.loop_index += 1
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            # If the slot is already in the past, send NOW. Do NOT claim a new slot: the
            # timetable is an absolute monotonic schedule, and re-claiming (as an earlier
            # version of this code did) both drops the original arrival AND resets the
            # timeline forward to `now`, which makes later slots pile up. That produced bursts
            # of 42 arrivals in one minute followed by 5 in the next at a nominal 0.68/s.
            return
        with DATA_LOCK:
            current_rate = self._get_current_rate(
                ramp_up_strategy=self.ramp_up_strategy,
                ramp_up_start_rps=self.ramp_up_start_rps,
                ramp_up_end_rps=self.ramp_up_end_rps,
                times=ConstantRateUser.loop_index,
                total_times=ConstantRateUser.data_len,
            )
            if self.ramp_up_strategy == "exponential" or os.environ.get("CAT_LOCUST_ARRIVAL", "periodic") == "poisson":
                theta = 1.0 / (current_rate * self.burstiness)
                interval = np.random.gamma(shape=self.burstiness, scale=theta)
                time.sleep(interval)
            else:
                time.sleep(1.0 / current_rate)
            if ConstantRateUser.loop_index >= self.data_len:
                ConstantRateUser.loop_index = 0
            else:
                ConstantRateUser.loop_index += 1

    @task
    def t(self):
        pass
