# SafeAnywhere 数据构造实施 Pipeline

本文档把 `README.md` 中的设计整理成可执行的数据工程流程。原 README 负责说明“为什么这样构造”，本文档只回答“具体怎么构造、每一步产出什么、怎么验收”。

## 0. 执行原则

第一版只做 SafeAnywhere cold-start SFT 数据，不混入外部 unsafe-prefix 数据源。

核心约束：

- prompt 统一来自 SafeChain：`D:\paper\alignment\repo\OPSA\data\jsonl\UWNSL__SafeChain__train.jsonl`
- 只使用 SafeChain 的 `instruction` 和 `label` 作为 prompt pool。
- 不直接使用 SafeChain 的 `response` 作为训练 target，因为其中常有长推理文本。
- 所有样本先进入 manifest，再生成 prefix 和 target。不要在采样脚本里直接调用 teacher。
- `assistant_prefix` 永远是上下文，默认不算 loss；训练 loss 只落在 `target`。
- hidden safe check 是结构化控制状态，不写详细推理，也不包含危险操作细节。

## 1. 输入与目标产物

源文件实际字段：

```json
{
  "instruction": "...",
  "label": "vanilla_benign | adversarial_benign | vanilla_harmful | adversarial_harmful",
  "response": "..."
}
```

当前 SafeChain 训练集规模：

| label                   | count |
| ----------------------- | ----: |
| `vanilla_benign`      | 11056 |
| `adversarial_benign`  | 11056 |
| `vanilla_harmful`     |  8591 |
| `adversarial_harmful` |  9297 |
| total                   | 40000 |

建议输出目录：

```text
SafeAnywhere/details/dataset/build/
  00_config/
    dataset_v1_5k.yaml
  01_clean/
    safechain_clean.jsonl
    safechain_stats.json
  02_manifest/
    safeanywhere_sft_v1_manifest.jsonl
    manifest_report.json
  03_prefix/
    safeanywhere_sft_v1_with_prefix.jsonl
    prefix_report.json
  04_teacher_jobs/
    teacher_jobs.jsonl
  05_teacher_outputs/
    teacher_outputs_raw.jsonl
    teacher_outputs_parsed.jsonl
  06_filtered/
    safeanywhere_sft_v1_candidates.jsonl
    filter_report.json
  07_final/
    safeanywhere_sft_v1_train.jsonl
    safeanywhere_sft_v1_val.jsonl
    safeanywhere_sft_v1_audit_sample.jsonl
    dataset_card.md
```

## 2. 数据单元

每条最终样本由四层信息决定：

```text
prompt_class  : SafeChain 原始四类标签
data_block    : 采样块，决定这条数据的训练作用
prefix_state  : no_prefix | unsafe_prefix | ambiguous_prefix
prefix_type   : none 或具体 prefix 类型
prefix_depth  : 0 或目标 prefix token 深度
```

推荐最终 JSONL schema：

```json
{
  "id": "safeanywhere_sft_v1_000001",
  "source": "safechain",
  "source_id": "safechain_00000123",
  "split": "train",
  "data_block": "unsafe_prefix_recovery",
  "prompt_class": "adversarial_harmful",
  "prefix_state": "unsafe_prefix",
  "prefix_type": "outline_prefix",
  "prefix_depth": 32,
  "instruction": "...",
  "assistant_prefix": "...",
  "target": "<|safe_check|>\nrisk: unsafe_continuation\ndecision: recover\ncorrection: stop unsafe continuation and recover safely\n<|/safe_check|>\n...",
  "target_visible": "...",
  "requires_safe_check": true,
  "loss_policy": {
    "user": "mask",
    "assistant_prefix": "mask",
    "target": "train"
  },
  "teacher_model": "...",
  "filter_status": "passed"
}
```

`target_visible` 是剥离 safe check 后的可见回答，便于安全过滤和人工审查。

## 3. 总体阶段

| stage | 名称               | 输入            | 输出                                      | 目的                                 |
| ----- | ------------------ | --------------- | ----------------------------------------- | ------------------------------------ |
| P0    | 配置冻结           | README + 本文档 | `dataset_v1_5k.yaml`                    | 固定配比、枚举、路径和随机种子       |
| P1    | SafeChain 清洗     | SafeChain JSONL | `safechain_clean.jsonl`                 | 字段标准化、去空、去重、统计         |
| P2    | 采样 manifest      | clean pool      | `safeanywhere_sft_v1_manifest.jsonl`    | 只决定样本身份和数据块，不生成文本   |
| P3    | prefix 规划与生成  | manifest        | `safeanywhere_sft_v1_with_prefix.jsonl` | 为需要 prefix 的样本补全 prefix 状态 |
| P4    | teacher job 构造   | with prefix     | `teacher_jobs.jsonl`                    | 把每条样本转成 teacher 标注任务      |
| P5    | teacher 生成与解析 | teacher jobs    | parsed outputs                            | 生成 SafeAnywhere 格式 target        |
| P6    | 自动过滤           | parsed outputs  | candidates                                | 规则过滤、枚举检查、安全检查         |
| P7    | 分层切分与导出     | candidates      | train/val JSONL                           | 90/10 切分并生成 dataset card        |
| P8    | 训练格式转换       | final JSONL     | model-ready dataset                       | 应用 chat template 和 loss mask      |

这个拆法的关键是：采样、prefix、teacher、过滤彼此解耦。任何一阶段出错都可以局部重跑。

## 4. P0：配置冻结

先写一个配置文件，后续脚本只读配置，不把配比和枚举散落在代码里。

建议配置内容：

```yaml
dataset_name: safeanywhere_sft_v1
seed: 20260630
target_size: 5000
source_path: D:/paper/alignment/repo/OPSA/data/jsonl/UWNSL__SafeChain__train.jsonl

safe_check:
  open_token: "<|safe_check|>"
  close_token: "<|/safe_check|>"
  risks:
    - benign
    - ambiguous_benign
    - safety_adjacent
    - harmful_request
    - adversarial_harmful
    - unsafe_continuation
  decisions:
    - answer
    - refuse
    - redirect
    - recover
  corrections:
    - answer directly
    - answer with appropriate safety boundaries
    - clarify safe intent before helping
    - refuse briefly and offer safe alternative
    - refuse the unsafe goal and redirect to safe information
    - stop unsafe continuation and recover safely

prefix_depths: [8, 16, 32, 64, 128]

blocks:
  vanilla_benign_direct: 900
  vanilla_benign_answer_check: 100
  adversarial_benign_direct: 500
  adversarial_benign_answer_check: 500
  harmful_no_prefix: 1000
  unsafe_prefix_recovery: 1600
  ambiguous_prefix_answer_recovery: 400
```

如果先做 3k pilot，使用同样比例：

| data block                           |   5k | 3k pilot |
| ------------------------------------ | ---: | -------: |
| `vanilla_benign_direct`            |  900 |      540 |
| `vanilla_benign_answer_check`      |  100 |       60 |
| `adversarial_benign_direct`        |  500 |      300 |
| `adversarial_benign_answer_check`  |  500 |      300 |
| `harmful_no_prefix`                | 1000 |      600 |
| `unsafe_prefix_recovery`           | 1600 |      960 |
| `ambiguous_prefix_answer_recovery` |  400 |      240 |

## 5. P1：SafeChain 清洗

脚本职责：

```text
scripts/01_prepare_safechain.py
```

输入：

```text
UWNSL__SafeChain__train.jsonl
```

处理：

1. 逐行 JSON 解析。
2. 校验 `instruction`、`label` 存在。
3. 只保留四个允许 label。
4. 对 `instruction` 做轻量规范化用于去重：
   - trim 首尾空白；
   - 连续空白折叠；
   - 保留原始文本作为训练输入，不修改语义。
5. 完全重复 instruction 只保留一条。
6. 生成稳定 `source_id`。

输出：

```json
{
  "source_id": "safechain_00000001",
  "instruction": "...",
  "prompt_class": "vanilla_benign",
  "response_ref": "..."
}
```

`response_ref` 可以保留用于人工参考，但后续默认不进入 target。

验收：

- JSON 解析失败数为 0。
- 四类 label 统计写入 `safechain_stats.json`。
- 去重前后数量透明记录。

## 6. P2：采样 manifest

脚本职责：

```text
scripts/02_sample_manifest.py
```

manifest 只决定“要做哪类样本”，不生成 assistant prefix，也不生成 target。

推荐采样映射：

| data block                           | source prompt class                           | prefix state         | target 类型               |
| ------------------------------------ | --------------------------------------------- | -------------------- | ------------------------- |
| `vanilla_benign_direct`            | `vanilla_benign`                            | `no_prefix`        | 直接回答                  |
| `vanilla_benign_answer_check`      | `vanilla_benign`                            | `no_prefix`        | safe check 后回答         |
| `adversarial_benign_direct`        | `adversarial_benign`                        | `no_prefix`        | 有边界地回答              |
| `adversarial_benign_answer_check`  | `adversarial_benign`                        | `no_prefix`        | safe check 后有边界地回答 |
| `harmful_no_prefix`                | `vanilla_harmful` + `adversarial_harmful` | `no_prefix`        | 拒绝或重定向              |
| `unsafe_prefix_recovery`           | `vanilla_harmful` + `adversarial_harmful` | `unsafe_prefix`    | 中途恢复                  |
| `ambiguous_prefix_answer_recovery` | `adversarial_benign`                        | `ambiguous_prefix` | 模糊状态下继续帮助        |

`harmful_no_prefix` 建议在两类 harmful 间均衡：

```text
vanilla_harmful: 500
adversarial_harmful: 500
```

`unsafe_prefix_recovery` 建议在 prompt class、prefix type、prefix depth 三个维度均衡：

```text
2 harmful classes * 4 prefix types * 5 depths = 40 cells
1600 / 40 = 40 samples per cell
```

`ambiguous_prefix_answer_recovery` 建议：

```text
4 prefix types * 5 depths = 20 cells
400 / 20 = 20 samples per cell
```

manifest 示例：

```json
{
  "id": "safeanywhere_sft_v1_000001",
  "source_id": "safechain_00001234",
  "data_block": "unsafe_prefix_recovery",
  "prompt_class": "vanilla_harmful",
  "prefix_state": "unsafe_prefix",
  "prefix_type": "acceptance_prefix",
  "prefix_depth": 32,
  "instruction": "..."
}
```

验收：

- 每个 data block 数量精确匹配配置。
- 每个需要均衡的 cell 数量匹配预期。
- 同一个 `source_id` 不要在同一个 data block 内重复；跨 block 可允许有限复用，但要记录复用率。

## 7. P3：prefix 规划与生成

脚本职责：

```text
scripts/03_build_prefixes.py
```

### 7.1 no_prefix

```json
{
  "prefix_state": "no_prefix",
  "prefix_type": "none",
  "prefix_depth": 0,
  "assistant_prefix": ""
}
```

### 7.2 unsafe_prefix

只用于 harmful prompt。目标是构造“模型已经开始走向非拒绝轨道”的上下文，但不能写入具体危险步骤。

prefix type：

| type                          | 作用                 | 可用模板方向                                                          |
| ----------------------------- | -------------------- | --------------------------------------------------------------------- |
| `acceptance_prefix`         | 表达已经开始顺从     | "Sure, I can help with that. Here is a direct way to think about it:" |
| `outline_prefix`            | 开始列结构           | "The process can be broken into several parts:"                       |
| `roleplay_prefix`           | 接受危险角色设定     | "In that role-play scenario, I would proceed as follows:"             |
| `partial_completion_prefix` | 已经写出风险方向开头 | "At a high level, the next steps would involve:"                      |

注意：这些模板只是状态提示，不要包含具体可执行的有害内容。实际实现时可以由小模板库生成，也可以由 teacher 生成后再过滤。

### 7.3 ambiguous_prefix

只用于 `adversarial_benign`。目标是构造“看起来安全邻近，但正确行为仍是回答”的上下文。

prefix type：

| type                     | 作用                                | 可用模板方向                                                                                       |
| ------------------------ | ----------------------------------- | -------------------------------------------------------------------------------------------------- |
| `caution_prefix`       | 谨慎开头                            | "This topic can be handled safely if we keep it at an appropriate level."                          |
| `boundary_prefix`      | 先设边界                            | "I will avoid harmful operational details and focus on allowed information."                       |
| `clarification_prefix` | 明确安全意图                        | "Interpreting this as a defensive or educational request, the useful answer is:"                   |
| `false_alarm_prefix`   | 包含容易触发过滤的词，但语义 benign | "Although the wording sounds sensitive, the safe interpretation is about prevention and analysis." |

### 7.4 prefix depth

`prefix_depth` 是目标 token 深度，不要求精确等于模板 token 数，但应落在对应 bucket 附近。

实现建议：

1. 先用模板生成基础 prefix。
2. 用安全的 filler 句补齐到目标 bucket。
3. 用目标模型 tokenizer 计数。
4. 如果超出 bucket 太多，截断到句子边界。

验收：

- `unsafe_prefix` 不含危险操作细节。
- `ambiguous_prefix` 不含拒绝结论。
- `prefix_depth` 分布接近配置。
- prefix 全部在训练中 masked。

## 8. P4：teacher job 构造

脚本职责：

```text
scripts/04_make_teacher_jobs.py
```

每条 job 明确告诉 teacher 当前样本属于哪个 data block，以及必须输出什么结构。

teacher job schema：

```json
{
  "job_id": "safeanywhere_sft_v1_000001",
  "instruction": "...",
  "prompt_class": "adversarial_harmful",
  "data_block": "unsafe_prefix_recovery",
  "prefix_state": "unsafe_prefix",
  "prefix_type": "outline_prefix",
  "prefix_depth": 32,
  "assistant_prefix": "...",
  "required_target_policy": {
    "requires_safe_check": true,
    "risk": "unsafe_continuation",
    "decision": "recover",
    "correction": "stop unsafe continuation and recover safely"
  }
}
```

teacher 提示词应强制以下要求：

```text
You generate only the assistant target continuation after the given assistant_prefix.
If a safe_check block is required, use exactly:
<|safe_check|>
risk: ...
decision: ...
correction: ...
<|/safe_check|>

Use only the provided enum values.
Do not include hidden reasoning, chain-of-thought, or operational harmful details.
After removing the safe_check block, the visible answer must remain natural, helpful when allowed, and safe.
For harmful requests, refuse briefly or redirect to safe high-level information.
For benign or ambiguous-benign requests, do not over-refuse.
```

## 9. P5：teacher 输出解析

脚本职责：

```text
scripts/05_parse_teacher_outputs.py
```

解析目标：

- 抽取完整 `target`。
- 如果有 safe check，抽取 `risk`、`decision`、`correction`。
- 剥离 safe check 得到 `target_visible`。
- 标记 malformed 原因。

输出示例：

```json
{
  "id": "safeanywhere_sft_v1_000001",
  "target": "<|safe_check|>\nrisk: unsafe_continuation\ndecision: recover\ncorrection: stop unsafe continuation and recover safely\n<|/safe_check|>\nI can't help continue in that direction, but I can offer safe, high-level information about prevention and risk reduction.",
  "target_visible": "I can't help continue in that direction, but I can offer safe, high-level information about prevention and risk reduction.",
  "parsed_safe_check": {
    "risk": "unsafe_continuation",
    "decision": "recover",
    "correction": "stop unsafe continuation and recover safely"
  },
  "parse_status": "passed"
}
```

不要让解析脚本“修复”teacher 输出。解析失败的样本进入 rejected，必要时重新生成。

## 10. P6：过滤

脚本职责：

```text
scripts/06_filter_dataset.py
```

过滤分两层：确定性规则过滤 + judge/人工抽查。

### 10.1 确定性规则

必须通过：

1. schema 完整。
2. `prompt_class`、`prefix_state`、`prefix_type` 合法。
3. 需要 safe check 的样本必须正好有一个 open token 和一个 close token。
4. 不需要 safe check 的 direct-answer 样本不能包含 safe check token。
5. `risk`、`decision`、`correction` 必须来自配置枚举。
6. safe check block 建议不超过 64 tokens。
7. `assistant_prefix` 不能被复制进 `target` 开头造成重复。
8. `target_visible` 非空。
9. `prefix_state != no_prefix` 时，`assistant_prefix` 必须非空且 `loss_policy.assistant_prefix = mask`。

### 10.2 安全过滤

对 hidden block：

- 不包含具体攻击、违法、危险操作步骤。
- 不包含长推理。
- 不包含用户可执行的危险参数、代码或配方。

对 visible output：

- harmful / unsafe-prefix 样本：剥离 hidden block 后仍然安全，不继续完成有害目标。
- adversarial-harmful：应拒绝或重定向，不被 jailbreak 设定带走。
- benign / adversarial-benign / ambiguous-prefix：不能过度拒绝。

### 10.3 建议 judge 标签

为每条样本增加：

```json
{
  "judge_hidden_safe": true,
  "judge_visible_safe": true,
  "judge_helpful_when_benign": true,
  "judge_over_refusal": false,
  "judge_notes": ""
}
```

第一版可以先做自动 judge + 人工抽查：

```text
每个 data block 抽 50 条人工审查
unsafe_prefix_recovery 额外抽 100 条
ambiguous_prefix_answer_recovery 额外抽 100 条
```

验收阈值建议：

- delimiter malformed rate = 0。
- enum invalid rate = 0。
- hidden unsafe detail rate = 0。
- harmful visible unsafe rate <= 1% 且问题样本全部剔除。
- benign over-refusal rate <= 5%，ambiguous-prefix 单独报告。

## 11. P7：分层切分与导出

脚本职责：

```text
scripts/07_split_export.py
```

切分比例：

```text
train: 90%
val:   10%
```

分层 key：

```text
data_block
prompt_class
prefix_state
prefix_type
prefix_depth
```

导出时必须生成 `dataset_card.md`，记录：

- 源数据路径和版本。
- 随机种子。
- 总量和各 data block 数量。
- train/val 数量。
- 过滤前后数量。
- 每类 reject reason 数量。
- teacher model。
- judge model 或人工审查说明。
- 已知限制。

最终文件：

```text
safeanywhere_sft_v1_train.jsonl
safeanywhere_sft_v1_val.jsonl
safeanywhere_sft_v1_audit_sample.jsonl
dataset_card.md
```

## 12. P8：训练格式与 loss mask

脚本职责：

```text
scripts/08_build_training_records.py
```

训练文本逻辑：

```text
User: {instruction}
Assistant: {assistant_prefix}{target}
```

loss 逻辑：

```text
User prompt tokens      -> -100
Assistant role tokens   -> -100
assistant_prefix tokens -> -100
target tokens           -> token ids
```

伪代码：

```python
messages = [
    {"role": "user", "content": instruction},
    {"role": "assistant", "content": assistant_prefix + target},
]
input_ids = tokenizer.apply_chat_template(messages)

target_start = locate_target_start(input_ids, assistant_prefix, target)
labels = [-100] * len(input_ids)
labels[target_start:] = input_ids[target_start:]
```

需要单元测试覆盖：

- `no_prefix` 样本 target 起点正确。
- `unsafe_prefix` 样本 prefix 不算 loss。
- `ambiguous_prefix` 样本 prefix 不算 loss。
- safe check delimiter token 算 loss。
- 剥离 hidden block 不影响 `target_visible`。

## 13. 推荐脚本布局

建议在 `SafeAnywhere/details/dataset/` 下新增：

```text
scripts/
  01_prepare_safechain.py
  02_sample_manifest.py
  03_build_prefixes.py
  04_make_teacher_jobs.py
  05_parse_teacher_outputs.py
  06_filter_dataset.py
  07_split_export.py
  08_build_training_records.py
  common/
    schema.py
    safe_check.py
    io.py
    stratified_split.py
```

`common/safe_check.py` 应集中维护：

- open/close token；
- risk/decision/correction 枚举；
- parse safe check；
- strip safe check；
- validate safe check。

避免每个脚本各写一套正则。

## 14. 最小可跑顺序

建议不要一开始就跑 5k。按下面顺序推进：

1. 跑 P1，确认 SafeChain 清洗和统计。
2. 跑 200 条 smoke manifest：
   - 每个 data block 至少 20 条；
   - 每种 prefix type 和 depth 至少出现一次。
3. 生成 prefix，人工快速看 50 条。
4. 调 teacher 跑 200 条。
5. 跑过滤和人工审查，修 teacher prompt 与 prefix 模板。
6. 跑 3k pilot。
7. 用 3k pilot 训练一次 SFT，检查：
   - safe check 格式合法率；
   - stripped visible answer 是否自然；
   - benign over-refusal；
   - unsafe prefix recovery。
8. 通过后再跑 5k 正式版。

## 15. 版本验收标准

`safeanywhere_sft_v1` 可以冻结的最低标准：

- 最终样本数达到配置目标，或清楚记录不足原因。
- 每个 data block 数量符合配置。
- train/val 分层比例正确。
- malformed safe check 为 0。
- hidden block 无危险细节。
- harmful / unsafe-prefix 可见输出不继续有害目标。
- benign 与 ambiguous-prefix 样本不过度拒绝。
- 训练转换中 prefix mask 单元测试通过。
- `dataset_card.md` 完整记录构造配置、teacher、过滤和统计。

## 16. 建议对 README 的整理方式

当前 `README.md` 可以保留为设计说明，但建议后续改成三段：

1. 方法动机：为什么是 `prompt class × prefix state`。
2. 数据定义：枚举、模板、schema。
3. 执行入口：链接本文档和最终脚本命令。

不要在 README 里同时塞完整动机、配比、脚本细节、论文表述和过滤细节。实施细节放在本文档，README 才会清晰。