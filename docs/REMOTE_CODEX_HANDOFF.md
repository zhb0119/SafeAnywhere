# SafeAnywhere Remote Codex Handoff

本文档用于在远端服务器继续使用 Codex 时恢复上下文。不要在本文档中写入服务器密码、API key 或其他凭据。

## 1. 项目目标

SafeAnywhere 的研究目标是把 `Think Anywhere in Code Generation` 中“在局部复杂点插入思考”的思想迁移到安全对齐：

```text
用户请求 / 多轮上下文 / 已生成 assistant 前缀
-> 局部风险触发点
-> 插入短 <safety_think>
-> 安全恢复到合规可见回复
```

核心不是只做前置 `<safety_think>`，而是训练模型在生成轨迹中途也能进行局部安全判断与恢复。

当前分两阶段推进：

1. `safechain_pilot_1k`：cold-start 格式数据，学习 `<safety_think>` 格式、半模块化字段、基本 harmful/benign/adversarial 安全决策。
2. `hex_phi_prefix_500`：dangerous-prefix recovery 数据，使用 masked `assistant_prefill` 模拟模型已经走偏，然后在完整 assistant 轨迹中于 `after_assistant_prefill` 位置插入 `<safety_think>` 恢复。

## 2. 当前远端仓库状态

远端仓库路径：

```text
/root/workspace/SafeAnywhere
```

GitHub remote：

```text
https://github.com/zhb0119/SafeAnywhere.git
```

截至本文档写入时，远端 `main` 有本地提交：

```text
8c0681f Add HEx-PHI dangerous prefix data builder
```

该提交已经在远端本地仓库生成，但之前 `git push origin main` 因会话超时未确认完成。检查方式：

```bash
cd /root/workspace/SafeAnywhere
git status --short --branch
git log --oneline -5 --decorate
```

如果显示：

```text
## main...origin/main [ahead 1]
```

说明还需要执行：

```bash
git push origin main
```

## 3. 重要安全注意

不要提交以下内容：

- `.env`
- `data/`
- `build/`
- 任何 API key、服务器密码、私钥

`.gitignore` 当前已忽略：

```text
.env
.env.*
!.env.example
.venv/
build/
data/
```

之前远端 `.env.example` 曾错误写入真实 DeepSeek key，已经在当前工作中恢复为占位符。提交前务必再次检查：

```bash
grep -R "sk-" -n --exclude-dir=.git --exclude=.env --exclude-dir=.venv --exclude-dir=build --exclude-dir=data . || true
git diff --check
```

## 4. 当前目录与数据

远端本地数据源：

```text
data/UWNSL__SafeChain__train.jsonl
data/Harmful-HEx-PHI.jsonl
```

其中 `Harmful-HEx-PHI.jsonl` 已从本地上传到远端：

```text
/root/workspace/SafeAnywhere/data/Harmful-HEx-PHI.jsonl
```

校验信息：

```text
size:   393638 bytes
sha256: 2ea4415da7951db3310268c6060483c1474a6fdb9a8d8691cf7fac0b5cfa011e
```

该文件每行是一个两轮 list：

```json
[
  {"role": "user", "content": "... harmful request ..."},
  {"role": "assistant", "content": "... unsafe assistant response ..."}
]
```

统计：

```text
330 lines
330 user/assistant pairs
```

## 5. 已完成的 SafeChain 1k 数据

远端已有完整 1k 输出：

```text
build/safechain_pilot_1k/
  manifest.jsonl
  annotations.jsonl
  failed.jsonl
  sft_train.jsonl
  sft_val.jsonl
  report.json
```

报告摘要：

```text
accepted: 1000
target_total: 1000
sft_train: 900
sft_val: 100
failed: 12
replacements_used: 12
labels: vanilla_benign/adversarial_benign/vanilla_harmful/adversarial_harmful 各 250
```

该 1k 的局限：所有有 `<safety_think>` 的 target 基本都在 response 开头，所以它是 cold-start 格式数据，不是 SafeAnywhere anywhere claim 的核心证据。

## 6. Dangerous Prefix 500 的设计决策

新增 dangerous-prefix 数据不直接混入 `safechain_pilot_1k/`，而是单独输出：

```text
build/hex_phi_prefix_500/
```

这样可以保留 provenance，便于后续做 ablation：

```text
baseline: SafeChain 1k
+prefix: SafeChain 1k + HEx-PHI dangerous-prefix 500
```

HEx-PHI 没有 SafeChain 的四类 label，因此使用兼容字段：

```json
{
  "source": "harmful_hex_phi",
  "attack_type": "dangerous_prefix",
  "risk_level": "disallowed",
  "label": "adversarial_harmful",
  "label_compat": "adversarial_harmful"
}
```

训练样本核心结构：

```json
{
  "messages": [
    {"role": "user", "content": "...", "loss_mask": 0},
    {"role": "assistant", "content": "masked assistant_prefill", "loss_mask": 0},
    {"role": "assistant", "content": "<safety_think>...</safety_think>\n安全恢复回复", "loss_mask": 1}
  ],
  "attack_type": "dangerous_prefix",
  "prefix_depth": 30,
  "prefix_type": "outline_prefix"
}
```

注意：target 片段内部 `<safety_think>` 仍在开头，但从完整 assistant 轨迹看，它位于：

```text
assistant_prefill -> <safety_think> -> recovery
```

所以报告中同时统计：

```text
safety_think_position_in_target: beginning
safety_think_position_in_full_trajectory: after_assistant_prefill
```

## 7. 新增代码与配置

新增文件：

```text
configs/hex_phi_prefix_500.yaml
scripts/02_build_dangerous_prefix.py
```

相关 README 已补充运行说明。

`configs/hex_phi_prefix_500.yaml` 的采样计划：

```yaml
sampling:
  total: 500
  depth_counts:
    10: 100
    20: 120
    30: 120
    50: 100
    100: 60
  val_ratio: 0.10
  max_replacements: 100
```

设计理由：

- 10/20/30 token-ish：训练早期发现风险轨迹。
- 50 token-ish：模拟已经走偏一小段后的恢复。
- 100 token-ish：少量 deep prefix，避免训练中过多暴露危险上下文。

`assistant_prefill` 不是直接复制 unsafe response，而是用模板生成 masked prefill，例如：

```text
The process can be broken into concrete stages:
Step 1: [REDACTED unsafe operational detail].
Step 2: [REDACTED unsafe operational detail].
```

这样避免把源数据中的具体有害步骤放入训练 target。prefill 只作为 context，`loss_mask=0`。

## 8. 远端环境

远端当前没有全局 `uv` 命令。虽然 README 使用 `uv run` 作为可复现指令，但远端服务器实际运行时使用项目虚拟环境：

```bash
cd /root/workspace/SafeAnywhere
.venv/bin/python --version
.venv/bin/python -m compileall scripts src
```

如需安装 uv，可另行安装；否则继续用 `.venv/bin/python`。

## 9. 已验证内容

已在远端通过：

```bash
cd /root/workspace/SafeAnywhere
.venv/bin/python -m compileall scripts src
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --mock --workers 2 --quiet
```

mock report 摘要：

```text
accepted: 500
target_total: 500
failed: 0
sft_train: 450
sft_val: 50
accepted_by_depth:
  10: 100
  20: 120
  30: 120
  50: 100
  100: 60
decision_counts:
  recover: 500
safety_think_position_in_full_trajectory:
  after_assistant_prefill: 500
```

之后曾启动真实 500 条 teacher 生成，但用户要求先 smoke test 通过后 push，因此已停止该完整任务。日志显示停止前真实 API 已成功生成约 31 条、无失败。由于任务被停止，当前 `build/hex_phi_prefix_500/report.json` 不存在，后续要正式生成需要重新运行。

检查是否有残留任务：

```bash
ps -ef | grep '02_build_dangerous_prefix.py' | grep -v grep || true
```

## 10. 下一步建议

### A. 先推送代码

如果远端仍显示 `[ahead 1]`，先推送：

```bash
cd /root/workspace/SafeAnywhere
git push origin main
```

如果加入本文档后需要提交：

```bash
cd /root/workspace/SafeAnywhere
git add docs/REMOTE_CODEX_HANDOFF.md
git commit -m "Add remote Codex handoff notes"
git push origin main
```

### B. 正式跑 dangerous-prefix 500

推荐先跑真实小样本检查 teacher 输出质量。当前脚本没有 `--limit` 参数，如果需要更干净的 smoke，可以临时加一个小配置，例如 `configs/hex_phi_prefix_smoke_20.yaml`，或先用现有 500 配置跑一会儿后中断并抽看 `annotations.jsonl`。

正式 500 条命令：

```bash
cd /root/workspace/SafeAnywhere
rm -rf build/hex_phi_prefix_500
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --workers 1
```

中断后继续：

```bash
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --workers 1 --resume
```

### C. 检查正式输出

完成后检查：

```bash
cat build/hex_phi_prefix_500/report.json
wc -l build/hex_phi_prefix_500/*.jsonl
head -1 build/hex_phi_prefix_500/sft_train.jsonl
```

关键验收：

```text
counts.accepted == counts.target_total == 500
sft_train == 450
sft_val == 50
safety_think_position_in_full_trajectory.after_assistant_prefill == 500
decision_counts.recover == 500
```

### D. 后续混合数据

建议后续新增一个 merge/export 脚本，把：

```text
build/safechain_pilot_1k/sft_train.jsonl
build/safechain_pilot_1k/sft_val.jsonl
build/hex_phi_prefix_500/sft_train.jsonl
build/hex_phi_prefix_500/sft_val.jsonl
```

合成：

```text
build/mixed_safechain1k_prefix500/
  sft_train.jsonl
  sft_val.jsonl
  report.json
```

混合时要保留 `source`、`attack_type`、`prefix_depth`、`messages/loss_mask` 等字段。对于旧 SafeChain 1k 没有 messages 的样本，可以保留 `prompt/response` 格式，训练脚本需要兼容两种形式，或先统一转换成 messages。

## 11. 当前几个容易误判的点

1. `hex_phi_prefix_500` 的 target 里 `<safety_think>` 是 beginning，但 full trajectory 是 `after_assistant_prefill`。论文/报告应以后者支撑 anywhere claim。
2. HEx-PHI response 不进入训练 target，只用来推断 prefix 类型；实际 prefill 是 masked template。
3. SafeChain 1k 是 cold-start，不要单独拿它 claim anywhere。
4. Dangerous-prefix 500 是第一类 anywhere 数据，后续还需要 multi-turn induction、long-context drift、jailbreak recovery。
5. 远端没有全局 uv，继续用 `.venv/bin/python` 最稳。

## 12. 常用命令

```bash
cd /root/workspace/SafeAnywhere

# 状态
git status --short --branch
git log --oneline -5 --decorate

# 编译检查
.venv/bin/python -m compileall scripts src

# SafeChain 1k 继续生成
.venv/bin/python scripts/01_build_dataset.py --config configs/safechain_pilot_1k.yaml --workers 1 --resume

# HEx-PHI dangerous-prefix mock
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --mock --workers 2 --quiet

# HEx-PHI dangerous-prefix 正式生成
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --workers 1

# 中断后继续
.venv/bin/python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --workers 1 --resume

# 敏感信息检查
grep -R "sk-" -n --exclude-dir=.git --exclude=.env --exclude-dir=.venv --exclude-dir=build --exclude-dir=data . || true

# 推送
git push origin main
```
