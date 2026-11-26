"""Default configuration values for the bundled v7 Locust client.

Unlike legacy model_config.py, NO hardcoded url_list, model_name_list, or
data_file_list allowlist restrictions are imposed: any --url, --model_name,
and --file path works out of the box on any GPU host or SSH tunnel.
"""

from __future__ import annotations

import os

model_name = os.environ.get("CAT_LOCUST_MODEL_NAME", "Qwen3.5-35B-A3B")
url = os.environ.get("CAT_LOCUST_URL", "http://127.0.0.1:8000/v1/chat/completions")
data_file = os.environ.get("CAT_LOCUST_DATA_FILE", "dataset/nocache_sla_in3669_v3.ndjson")

share_data_list = [0, 1, 2]
share_data = 2

sample_type_list = [0, 1]
sample_type = 0

api_key = os.environ.get("API_KEY", "")

use_uuid_list = [0, 1]
use_uuid = 0

PAD_DELIM = os.environ.get("CAT_LOCUST_PAD_DELIM", "===ANSWER_END===")
PAD_INSTRUCTION = os.environ.get(
    "CAT_LOCUST_PAD_INSTRUCTION",
    "\n\n【输出格式要求】答案写完后，另起一行输出 {delim}，"
    "然后继续补充相关的背景说明与延伸细节，直到被系统截断为止；"
    "分隔符之后的内容不计入答案，但请勿提前结束、勿重复已写内容。",
)

