# SafeAnywhere

SafeAnywhere 是一个用于构造 safety-think SFT pilot 数据的轻量数据工程项目。最终训练文件仍然是普通的 `prompt` / `response` 格式；在需要局部安全判断的 assistant response 中，可以插入简短的 `<safety_think>...</safety_think>` 块。

## 项目结构

```text
.
  configs/             # smoke / pilot 数据构造配置
  scripts/             # 环境检查与数据构造入口脚本
  src/safeanywhere/    # 采样、teacher prompt、校验、导出逻辑
  docs/dataset/        # 数据集设计说明与 prompt 规范
  docs/method/         # 方法方案与简版说明
  data/                # 本地数据源目录，Git 忽略
  build/               # 生成结果目录，Git 忽略
  .env.example         # 本地环境变量模板
  pyproject.toml       # Python 项目配置
```

仓库根目录就是 Python 项目根目录，不再维护嵌套的 `code/` 项目层。

## 安装

Linux 环境要求：

```bash
python3 --version  # 需要 Python >= 3.10
uv --version
```

在仓库根目录安装依赖，并严格使用 `uv.lock` 复现依赖版本：

```bash
uv sync --frozen
```

真实调用 DeepSeek 或 OpenAI-compatible teacher 前，复制 `.env.example` 为本地 `.env`，并填写：

```bash
cp .env.example .env
```

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

`.env` 已被 Git 忽略，不要提交。

## 数据源

默认配置会从下面的相对路径读取 SafeChain 训练集：

```text
data/UWNSL__SafeChain__train.jsonl
```

可以把数据文件放到这个位置，也可以按本机环境修改 `configs/*.yaml` 中的 `paths.safechain_jsonl`。`data/` 已被 Git 忽略，适合放本地数据集。

配置文件中的 `paths.*` 统一按仓库根目录解析；绝对路径也可以直接使用。项目代码按 UTF-8 读取配置、`.env` 和 JSONL，输出 JSON/JSONL 也使用 UTF-8 与 LF 换行。

## 运行

检查环境和配置，不要求 API key：

```bash
uv run python scripts/00_check_env.py --config configs/safechain_smoke_10.yaml
```

检查真实 teacher 运行所需的 API key：

```bash
uv run python scripts/00_check_env.py --config configs/safechain_smoke_10.yaml --require-api
```

不调用 API，构造 mock smoke 数据：

```bash
uv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --mock
```

调用真实 teacher，构造 10 条 smoke 数据：

```bash
uv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --workers 2
```

构造 1k pilot 数据：

```bash
uv run python scripts/01_build_dataset.py --config configs/safechain_pilot_1k.yaml --workers 1
```

provider 稳定后可以尝试 `--workers 2`。在 empty content 失败率较低之前，不建议开太高并发。

脚本默认显示 tqdm 进度条；使用 `--quiet` 可以关闭进度条。

## 失败与补采

API 空返回、JSON 解析失败、安全块校验失败都不会进入训练集。构造脚本会：

1. 将失败样本写入 `failed.jsonl`，保留 `id`、`label`、`instruction` 和 `error`。
2. 从同 label 的 SafeChain 池中补采 replacement。
3. 继续生成，直到每个 label 达到配置目标数量，或达到 `sampling.max_replacements`。

这样可以保持标签数量平衡，避免单条失败导致最终数据集静默变短。

中断运行可能留下部分 `manifest.jsonl`、`annotations.jsonl` 或 `failed.jsonl`。当前脚本在重新运行时会清理这些文件、SFT 导出文件和旧的 `report.json`，然后从头重建数据集。只有当输出目录中存在 `report.json`，且其中 `counts.accepted == counts.target_total` 时，才应视为完整构建；否则直接重跑同一条构建命令。

## Teacher Prompt Routing

为了降低简单 benign 样本上的 JSON 空返回概率，teacher prompt 按样本类型分流：

```text
vanilla_benign + requires_safety_think=false
  -> 短 JSON prompt，只要求 {"response": "..."}

adversarial_benign + requires_safety_think=false
  -> 短 JSON prompt，强调按安全 benign 解释回答，不插入 <safety_think>

requires_safety_think=true 或 harmful/adversarial_harmful
  -> 完整 safety block schema prompt
```

这样普通 benign 样本不会被较长的 safety schema 干扰，同时高风险样本仍保留严格格式约束。

## Safety Block Schema

`<safety_think>` 使用半受控 schema：

```text
risk={short_free_text}; trigger={short_free_text}; intent={intent}; decision={decision}; plan={short_free_text}
```

小词表字段如下：

```text
intent: benign, ambiguous, dual_use, harmful, jailbreak
decision: answer, clarify, limit, refuse, recover
```

`risk`、`trigger`、`plan` 是非空、简短的自由文本短语。校验器会把常见同义词归一化，例如 `harmless -> benign`、`allow -> answer`、`decline -> refuse`，最终写入的是归一化后的 response。

## 输出

构造脚本在 API 生成过程中会把完成样本流式写入磁盘。成功结束后，脚本会按 manifest 顺序重写 JSONL 文件，避免并发完成顺序打乱最终文件顺序。

每个配置默认写入：

```text
build/<dataset_name>/
  manifest.jsonl       # 初始样本与 replacement 清单
  annotations.jsonl    # 通过校验的 teacher 输出
  failed.jsonl         # 失败样本，仅存在失败时生成
  sft_train.jsonl      # 最终 SFT 训练集
  sft_val.jsonl        # 最终 SFT 验证集
  report.json          # 数量、失败、replacement、位置、写入模式与 teacher 元数据
```

`report.json` 中的 `safety_think_position` 会统计安全块出现在 response 开头还是中间。

## 文档

- 数据集设计：`docs/dataset/`
- 方法说明：`docs/method/`
