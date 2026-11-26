# sla-bench

[English](README.md) | **简体中文**

LLM 推理服务 SLA 吞吐与准确率压测工具（基准卡：**H20**）。仓库内置 H20 FP8 参考基线（`results/h20_fp8/`），无需重跑 H20。

## 1. 安装与服务端要求

```bash
pip install -e .            # Python >= 3.9
python3 -m sla_bench --help
```

被测服务端要求：
- 提供 vLLM / SGLang 兼容的 `/v1/chat/completions`、`/tokenize`、`/detokenize`（客户端不依赖本地 `transformers`，所有 token 计数均在服务端完成）。
- **必须关闭前缀缓存**（SGLang `--disable-radix-cache`，vLLM `--no-enable-prefix-caching`）。
- 混合线性注意力模型（如 `Qwen3.5-35B-A3B`）在 SGLang 上需将 `--max-running-requests` 设在峰值并发以上（推荐 `256`），避免 Mamba 状态槽排队。
- 默认健康检查路径为 `/health_generate`（vLLM 传 `--health-path /health` 或 `--no-health-check`）。

## 2. 配置与一键运行

复制预置 H20 基线的厂商模板 `configs/vendor_card.yaml`（测 H20 本身参考 `configs/example_card.yaml`）：

```bash
cp configs/vendor_card.yaml my_card.yaml
```

按实际硬件修改三处：
- `card`：被测卡真实型号（必填，禁止写 `GPU`）。
- `models[].tp` 与 `models[].cards`：**单个副本占用的物理卡数**（`QPM/card = 实际到达率 × 60 ÷ cards`；不按 die 拆分，如单张双 die 物理卡跑 1 个 TP=2 副本填 `cards: 1`，2 张单卡跑 TP=2 填 `cards: 2`，半卡填 `0.5`）。
- `models[].ports`（或 `urls`）：各独立副本端口（默认测 **FP8**，`external_baselines` 已挂载 `h20_fp8`）。

执行一键压测（自动完成：2 轮跨副本轮换粗扫 → SLA①/② 边界加密 → 固定 300 题 `acc` → HTML 对比报告 → `light` 导出）：

```bash
python3 -m sla_bench run -c my_card.yaml
```

支持分步子命令：`dataset-info`、`sweep`、`refine [--dry-run]`、`acc`、`export`、`report`（参数见 `--help`）。

## 3. 固定评测口径（请勿改动，否则与 H20 不可比）

| 项 | 取值 | 说明 |
| --- | --- | --- |
| 数据集 | `dataset/nocache_sla_in3669_v3.ndjson.gz` | 内置 1000 条长文本 QA（均值约 3755 token） |
| 输出长度 | `max_tokens: 1000` | `ignore_eos` + `===ANSWER_END===` 强制凑满 1000 token 固定解码量；`acc` 与 `real-out` 仅统计分隔符前真实回答 |
| 窗口与门禁 | `warmup_s: 60`, `min_s: 180` | 前 60s 丢弃，稳态开环泊松 ≥180s；需同时满足队列稳定（漂移 ≤ 3 SE）、零错误、输入/输出均值 ±3%（`len_ok`） |
| 准确率切片 | `acc_prompts: 300`, `acc_seed: 0` | 固定 128 题预热 + 同 300 题计分，跨卡严格同题 |
| 两档 SLA | ① `2005 / 51 ms`；② `1000 / 20 ms` | ① mean TTFT ≤ 2005 ms 且 TPOT ≤ 51 ms；② mean TTFT ≤ 1000 ms 且 TPOT ≤ 20 ms（加密至 `gap < 10%`） |

## 4. 内置 H20 参考基线（4×H20，2 个 TP=2 副本，每副本 2 card）

| 基线目录 | 精度 | 模型 | SLA① QPM/card | SLA② QPM/card | 固定 300 题 `acc` |
| --- | --- | --- | --- | --- | --- |
| `results/h20_fp8` | FP8 | `Qwen3.5-35B-A3B` | **102.9** (102.9–104.4) | **69.2** (66.7–73.7) | **58.3%** |
| `results/h20_fp8` | FP8 | `Qwen3.8-27B` | **28.6** (28.1–31.8) | **12.9** (12.9–15.9) | **65.7%** |

## 5. 回传交付物

请将整个 **`<results_dir>` 结果目录全量打包回传**（客户端已自动省略原始请求与生成文本，无需回传原始文本）：
1. HTML 报告（`report_out` 输出文件，如 `<results_dir>/sla_report.html`）。
2. `<results_dir>/round*` 全量轮次与逐请求指标文件（`curve.json` 及不含原始文本的 `pre_detail_stat_*.jsonl`）。
3. `<results_dir>/light/<卡名>/*_burst.jsonl` 与 `<results_dir>/acc/<model>/`（`acc_summary.json`、`item_scores.json`）。

## 6. 自检测试

```bash
python3 -m unittest discover -s tests
```
