# SafeAnywhere Remote Codex Handoff

本文档记录 SafeAnywhere 当前项目进度、远端操作 tips 和下一步 TODO，用于后续接手时快速判断数据生成、合并与 SFT 训练该从哪里继续。

## 0. 当前默认数据入口

当前默认数据构造入口已经收敛为一条命令：

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/build_sft_dataset.py \
  --config configs/safeanywhere_sft_v1.yaml \
  --workers 1 \
  --quiet
```

默认产物：

```text
build/safeanywhere_sft_v1/
  safechain/
  harmful_prefix/
  sft_train.jsonl
  sft_val.jsonl
  train_lf_v1_spanmasked.jsonl
  val_lf_v1_spanmasked.jsonl
  report.json
  dataset_card.md
```

当前默认数据类型只暴露两类：

```text
safechain       # cold-start safety-think
harmful_prefix  # masked assistant-prefix recovery, internally from HEx-PHI + SafeChain harmful prompts
```

下面关于 1500 pilot / prefix500 的内容保留为历史上下文；新复现优先按 README 和 `configs/safeanywhere_sft_v1.yaml` 执行。

## 1. 当前目标

SafeAnywhere 的核心目标是训练模型在生成轨迹中途也能做局部安全判断与恢复：

```text
用户请求 / 多轮上下文 / 已生成 assistant 前缀
-> 局部风险触发点
-> 插入短 <safety_think>
-> 安全恢复到合规可见回复
```

当前数据阶段目标是得到 1500 条 SFT pilot 数据：

```text
1000 SafeChain cold-start safety-think 数据
+ 500 HEx-PHI masked dangerous-prefix recovery 数据
= 1500 SafeAnywhere SFT pilot 数据
```

这 1500 条用于第一轮 SFT。`safechain_pilot_1k` 主要学习 `<safety_think>` 格式和基础安全决策；`hex_phi_prefix_500` 用 masked `assistant_prefill` 训练模型在完整 assistant 轨迹中于 `after_assistant_prefill` 位置恢复。

## 2. 仓库与环境

仓库路径：

```text
/root/workspace/SafeAnywhere
```

GitHub remote：

```text
https://github.com/zhb0119/SafeAnywhere.git
```

远端当前主要使用项目虚拟环境：

```bash
cd /root/workspace/SafeAnywhere
.venv/bin/python --version
.venv/bin/python -m compileall scripts src
```

不要提交：

```text
.env
data/
build/
任何 API key、服务器密码、私钥
```

提交前检查：

```bash
cd /root/workspace/SafeAnywhere
grep -R "sk-" -n --exclude-dir=.git --exclude=.env --exclude-dir=.venv --exclude-dir=build --exclude-dir=data . || true
git diff --check
git status --short --branch
```

## 3. 本地数据源

远端本地数据源：

```text
data/UWNSL__SafeChain__train.jsonl
data/Harmful-HEx-PHI.jsonl
```

`Harmful-HEx-PHI.jsonl` 每行是一个 user/assistant pair：

```json
[
  {"role": "user", "content": "... harmful request ..."},
  {"role": "assistant", "content": "... unsafe assistant response ..."}
]
```

当前 `hex_phi_prefix_500` 不直接复制 HEx-PHI unsafe assistant response 作为训练上下文，而是用 masked template 生成 `assistant_prefill`，例如：

```text
The process can be broken into concrete stages:
Step 1: [REDACTED unsafe operational detail].
Step 2: [REDACTED unsafe operational detail].
```

这样做的目的是避免把具体危险步骤、材料、链接、代码、剂量或规避方法写入训练 target。`assistant_prefill` 只作为 context，必须 `loss_mask=0`。

## 4. 当前数据进度

### SafeChain 1k

已完成：

```text
build/safechain_pilot_1k/
  manifest.jsonl
  annotations.jsonl
  failed.jsonl
  sft_train.jsonl
  sft_val.jsonl
  report.json
```

已知报告摘要：

```text
accepted: 1000
target_total: 1000
sft_train: 900
sft_val: 100
failed: 12
replacements_used: 12
labels: vanilla_benign/adversarial_benign/vanilla_harmful/adversarial_harmful 各 250
```

局限：SafeChain 1k 是 cold-start 格式，不能单独支撑 anywhere claim。

### HEx-PHI Dangerous Prefix 500

已完成并通过数量验收：

```text
build/hex_phi_prefix_500/
```

完成快照：

```text
time: 2026-07-06 14:23:21 CST +0800
accepted: 500 / 500
failed: 0
sft_train: 450
sft_val: 50
decision_counts.recover: 500
safety_think_position_in_full_trajectory.after_assistant_prefill: 500
report: build/hex_phi_prefix_500/report.json
```

输出文件：

```text
manifest:    build/hex_phi_prefix_500/manifest.jsonl
annotations: build/hex_phi_prefix_500/annotations.jsonl
sft_train:   build/hex_phi_prefix_500/sft_train.jsonl
sft_val:     build/hex_phi_prefix_500/sft_val.jsonl
report:      build/hex_phi_prefix_500/report.json
```

复查命令：

```bash
cd /root/workspace/SafeAnywhere
cat build/hex_phi_prefix_500/report.json
wc -l build/hex_phi_prefix_500/annotations.jsonl build/hex_phi_prefix_500/sft_train.jsonl build/hex_phi_prefix_500/sft_val.jsonl
```

## 5. Dangerous-Prefix 数据结构

每条样本核心结构：

```json
{
  "messages": [
    {"role": "user", "content": "...", "loss_mask": 0},
    {"role": "assistant", "content": "masked assistant_prefill", "loss_mask": 0},
    {"role": "assistant", "content": "<safety_think>...</safety_think>\n安全恢复回复", "loss_mask": 1}
  ],
  "attack_type": "dangerous_prefix",
  "risk_level": "disallowed",
  "label": "adversarial_harmful",
  "prefix_depth": 30,
  "prefix_type": "outline_prefix"
}
```

注意：

```text
target 片段内部：<safety_think> 在 beginning
完整 assistant 轨迹：assistant_prefill -> <safety_think> -> recovery
```

报告和论文表述应使用：

```text
safety_think_position_in_full_trajectory = after_assistant_prefill
```

## 6. 近期 TODO

1. 抽查 `build/hex_phi_prefix_500/annotations.jsonl` 的真实 teacher 输出质量：

```bash
cd /root/workspace/SafeAnywhere
.venv/bin/python - <<'PY'
import json, re
from pathlib import Path

path = Path("build/hex_phi_prefix_500/annotations.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
block = re.compile(r"<safety_think>.*?</safety_think>", re.S)

print("rows", len(rows))
for row in rows[:3]:
    print("---", row["id"], row.get("prefix_depth"), row.get("prefix_type"))
    print([(m["role"], m.get("loss_mask")) for m in row["messages"]])
    print(row["assistant_prefill"][:240].replace("\n", " | "))
    print(block.search(row["response"]).group(0)[:240].replace("\n", " "))
    print(block.sub("", row["response"]).strip()[:240].replace("\n", " "))
PY
```

2. 新增 merge/export 脚本，把 1k + 500 合并成 1500：

```text
build/mixed_safechain1k_prefix500/
  sft_train.jsonl
  sft_val.jsonl
  report.json
```

3. 对 mixed 1500 做最终校验。
4. 准备 SFT 数据转换与训练脚本。
5. 跑第一轮 SFT。
6. 做 safety / usefulness / dangerous-prefix recovery 评测。

## 7. 1500 数据合并方案

合并输入：

```text
build/safechain_pilot_1k/sft_train.jsonl
build/safechain_pilot_1k/sft_val.jsonl
build/hex_phi_prefix_500/sft_train.jsonl
build/hex_phi_prefix_500/sft_val.jsonl
```

合并输出建议：

```text
build/mixed_safechain1k_prefix500/
  sft_train.jsonl   # 1350 条：900 SafeChain + 450 Prefix
  sft_val.jsonl     # 150 条：100 SafeChain + 50 Prefix
  report.json
```

合并时必须保留 provenance：

```text
id
source
source_id
label
attack_type
requires_safety_think
prefix_depth
prefix_type
assistant_prefill
messages/loss_mask
```

推荐统一成 messages 格式：

```json
{
  "id": "...",
  "messages": [
    {"role": "user", "content": "...", "loss_mask": 0},
    {"role": "assistant", "content": "...", "loss_mask": 1}
  ],
  "source": "...",
  "attack_type": "...",
  "label": "...",
  "requires_safety_think": true
}
```

对于 `hex_phi_prefix_500`，保持三段 messages：

```json
[
  {"role": "user", "content": "...", "loss_mask": 0},
  {"role": "assistant", "content": "assistant_prefill", "loss_mask": 0},
  {"role": "assistant", "content": "target recovery", "loss_mask": 1}
]
```

如果旧 SafeChain 1k 只有 `prompt` / `response`，则转换为：

```json
[
  {"role": "user", "content": "prompt", "loss_mask": 0},
  {"role": "assistant", "content": "response", "loss_mask": 1}
]
```

最终校验：

```text
train_total == 1350
val_total == 150
total == 1500
prefix_train == 450
prefix_val == 50
safechain_train == 900
safechain_val == 100
所有 <safety_think> block schema 合法
所有 dangerous_prefix 样本 loss mask 为 0/0/1
```

## 8. 1500 数据后的 SFT 规划

### 8.1 训练目标

第一轮 SFT 只做格式和局部恢复能力验证，不直接追求最终最强模型：

```text
目标 1：模型能稳定生成简短 <safety_think> block。
目标 2：harmful/adversarial 输入能拒绝、限制或恢复。
目标 3：benign 输入不过度拒绝。
目标 4：dangerous-prefix context 后能停止 unsafe trajectory 并恢复。
```

### 8.2 训练格式要求

训练框架必须支持 per-message 或 per-token loss mask。

关键要求：

```text
user tokens: loss_mask=0
assistant_prefill tokens: loss_mask=0
assistant target tokens: loss_mask=1
```

如果训练框架不能保证 `assistant_prefill` 不计 loss，不要训练 dangerous-prefix 数据。否则模型可能学习 masked prefix 或 real unsafe prefix，而不是学习 recovery。

### 8.3 推荐训练输入

把 `build/mixed_safechain1k_prefix500/sft_train.jsonl` 转成训练框架需要的 conversation 格式，同时保留 mask。

建议先产出一个中间文件：

```text
build/mixed_safechain1k_prefix500/train_messages_masked.jsonl
build/mixed_safechain1k_prefix500/val_messages_masked.jsonl
```

每行包含：

```json
{
  "id": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "loss_mask": [0, 1],
  "metadata": {
    "source": "...",
    "attack_type": "...",
    "label": "...",
    "prefix_depth": null
  }
}
```

对于 dangerous-prefix 样本，`loss_mask` 对应三段：

```json
"loss_mask": [0, 0, 1]
```

### 8.4 第一轮 SFT 建议

先小规模跑通训练链路：

```text
run: safeanywhere_sft_debug
data: mixed 1500
epochs: 1
lr: 保守小学习率
batch: 以显存为准，优先保证无 OOM
save: 每个 epoch 保存 checkpoint
```

debug run 检查：

```text
loss 正常下降，无 NaN
训练样本 mask 生效
val loss 可计算
checkpoint 可加载推理
```

第二次再跑正式小模型 SFT：

```text
run: safeanywhere_sft_1500_v1
data: mixed 1500
epochs: 2-3
early stop: 根据 val loss 与人工抽样质量决定
```

### 8.5 SFT 后评测

至少做四组评测：

```text
benign utility:
  正常问题是否能回答，是否过度插入 <safety_think> 或过度拒绝

harmful refusal:
  有害请求是否拒绝/限制，visible response 是否安全

adversarial/jailbreak:
  越狱诱导下是否仍遵守安全策略

dangerous-prefix recovery:
  人工构造 assistant_prefill 后，模型是否能在后续插入 <safety_think> 并恢复
```

dangerous-prefix recovery 是 SafeAnywhere 的核心评测，应单独报告：

```text
recovery_rate
unsafe_continuation_rate
safety_think_insert_rate
visible_response_safe_rate
over_refusal_on_benign_prefix
```

### 8.6 后续 real-prefix 消融

当前 `hex_phi_prefix_500` 是 masked-prefix 版本。后续建议增加一个独立 real-prefix 消融，不覆盖当前数据：

```text
build/hex_phi_prefix_500_realprefix/
```

核心差异：

```text
masked-prefix:
  assistant_prefill = synthetic masked template

real-prefix:
  assistant_prefill = HEx-PHI unsafe assistant response 的 token-ish 前缀截断
```

real-prefix 只允许作为 `loss_mask=0` 的 context。只有在训练框架确认 mask 正确后，才可以跑 real-prefix SFT。

建议 ablation：

```text
SafeChain 1k
SafeChain 1k + masked dangerous-prefix 500
SafeChain 1k + real dangerous-prefix 500
```

## 9. 常用命令

```bash
cd /root/workspace/SafeAnywhere

# 状态
git status --short --branch
git log --oneline -5 --decorate

# 编译检查
.venv/bin/python -m compileall scripts src

# 当前 dangerous-prefix 500 进度
wc -l build/hex_phi_prefix_500/annotations.jsonl build/hex_phi_prefix_500/failed.jsonl 2>/dev/null || true
ps -fp $(cat build/hex_phi_prefix_500/run.pid 2>/dev/null) 2>/dev/null || true

# dangerous-prefix 500 resume
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --workers 1 --resume

# 完成后看报告
cat build/hex_phi_prefix_500/report.json

# 敏感信息检查
grep -R "sk-" -n --exclude-dir=.git --exclude=.env --exclude-dir=.venv --exclude-dir=build --exclude-dir=data . || true

# 推送代码
git push origin main
```
