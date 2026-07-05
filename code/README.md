# SafeAnywhere Code

这是 SafeAnywhere 数据构造的精简版代码。

核心训练格式沿用 Think-Anywhere：`<safety_think>...</safety_think>` 直接嵌入 assistant `response`，最终 SFT 文件仍是普通 `prompt/response` 数据。

## 安装

```powershell
cd D:\paper\SafeAnywhere\code
uv sync
```

真实调用 DeepSeek 前，在 `.env` 中写入：

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

不要把 `.env` 提交到仓库。

## 运行

检查环境：

```powershell
uv run python scripts/00_check_env.py --config configs/safechain_smoke_10.yaml --require-api
```

mock 构建：

```powershell
uv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --mock
```

真实 DeepSeek 构建：`n`n```powershell`nuv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --workers 2`n```

脚本默认显示 tqdm 进度条，并在失败时打印 failed/replacement 信息。`--workers` 控制并发 API 调用数；建议先用 2，稳定后再试 4。并发太高可能触发服务限流或更多空 content。`n`n如果不想看进度条，加：

```powershell
uv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --workers 2 --quiet
```

## 失败与 Replacement

API 空返回、JSON 解析失败、安全块校验失败都不会进入训练集。脚本会：

1. 将失败样本写入 `failed.jsonl`，保留 `id/label/instruction/error`。
2. 从同一个 label 的 SafeChain 池中补采 replacement。
3. 继续生成，直到每个 label 达到 config 中的目标数量，或达到 `sampling.max_replacements`。

这样最终训练集不会因为单条失败变成 999 条，也不会破坏四类标签平衡。

## 输出

脚本会边调用 API 边写入 `manifest.jsonl`、`annotations.jsonl` 和 `failed.jsonl`；正常结束后会再按 manifest 顺序重写一次，保证并发完成顺序不会打乱最终文件。`n`n每个 config 的输出目录只保留少量文件：

```text
build/<dataset_name>/
  manifest.jsonl       # 初始样本和 replacement 清单
  annotations.jsonl    # 通过校验的教师结果
  failed.jsonl         # 仅当存在失败样本时生成
  sft_train.jsonl      # 最终训练集
  sft_val.jsonl        # 最终验证集
  report.json          # 统计、失败 id、replacement 数、安全块位置分布、write_mode
```

`report.json` 中的 `safety_think_position` 会统计安全块在 `beginning` 还是 `middle`。

## 代码结构

```text
configs/
  safechain_smoke_10.yaml
  safechain_pilot_1k.yaml
scripts/
  00_check_env.py
  01_build_dataset.py
src/safeanywhere/
  sampling.py    # SafeChain 读取、去重、确定性采样和 same-label replacement
  prompts.py     # 四类 teacher prompt
  teacher.py     # DeepSeek/OpenAI-compatible 调用与 mock teacher
  filters.py     # response 与 <safety_think> 校验、位置统计
  export.py      # SFT train/val 导出
```
