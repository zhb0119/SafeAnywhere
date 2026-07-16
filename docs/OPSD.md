# SafeAnywhere OPSD 实现

OPSD 是 SafeAnywhere 的 on-policy self-distillation 阶段。它不重新做 SFT，而是从 cold-start SFT merge 后的 checkpoint 出发，让 student 先自己生成 continuation，再用带安全系统提示的 teacher 分布去蒸馏这段 continuation。

## 分支

当前有两条 OPSD 分支：

```text
configs/opsd/safechain_qwen3_0_6b.yaml
configs/opsd/safechain_qwen3_0_6b_special_tokens.yaml
```

- 普通分支：从 `runs/merged/qwen3_0_6b_sft_v1` 启动。
- special-token 分支：从 `runs/merged/qwen3_0_6b_sft_special_v1` 启动，并保留 `embed_tokens` / `lm_head` 的 special-token 状态。

special-token OPSD 仍会启用：

- `modules_to_save: [embed_tokens, lm_head]`
- `ensure_weight_tying: true`
- `sparse_token_grad`

这样新加入的 `<safety_think>` 和 `</safety_think>` 行不会在 LoRA 训练和 merge 后丢失。

## 训练流程

单个 micro-batch 的主流程是：

```text
student prompt
  -> student on-policy generate continuation
  -> 同一 continuation 送入 student / teacher
  -> teacher prompt 额外加 safety system prompt
  -> 计算 token-level KL
  -> 叠加 structure 权重
  -> 必要时补 prefix CE
```

这里的关键点是：

- teacher 不重新采样文本，只对 student 已采样出的 continuation 给 logits
- student / teacher 的 continuation token 完全一致
- teacher prompt 由 `configs/opsd/teacher_prompts_v1.yaml` 提供，按 label 区分安全提示

## KL 设计

当前 loss 入口是：

```yaml
loss:
  type: adaptive_kl
  temperature: 1.0
  adaptive_kl:
    harmful_entropy_low: 0.2
    harmful_entropy_high: 0.5
    harmful_forward_floor: 0.2
    harmful_reverse_floor: 0.2
```

实现逻辑：

- `benign` 样本：只走 `forward KL`
- `harmful` 样本：按 teacher 的归一化熵，在 `forward KL` 和 `reverse KL` 之间线性插值
- 低熵 token 更偏 `reverse KL`
- 高熵 token 更偏 `forward KL`

这一步只作用在 continuation token 上，和 prefix CE、结构权重是分开的。

## 结构权重

`src/safeanywhere/opsd/structure.py` 会在 continuation 中查找 `<safety_think>` 和 `</safety_think>`，然后为不同位置分配 token 权重。

有害样本：

- 找不到 `<safety_think>` 时，只强化前若干 token，并标记 `needs_prefix_ce=True`
- 找到安全块时，会强化开标签、块内 token、闭标签和块后回复的不同区间

良性样本：

- 正常良性输出保留前缀权重和较低 body 权重
- 如果出现 safety tag 或拒答开头，会降权后续 body，防止过度拒答被放大

这些 token weights 会在 KL 上做加权平均，和 adaptive KL 叠加使用。

## Prefix CE

`prefix_ce` 只在 harmful 样本没有形成有效 safety block，或 block 未闭合时触发：

```yaml
loss:
  prefix_ce:
    enabled: true
    weight: 0.5
    max_tokens: 48
```

它训练的是一个短前缀目标：

```text
<safety_think>
risk=...
decision=...
...
</safety_think>
```

作用不是替代 KL，而是把“先开安全块”这个起始动作补稳。

## 产物

普通 OPSD：

```text
runs/opsd/qwen3_0_6b_opsd_v1
```

special-token OPSD：

```text
runs/opsd/qwen3_0_6b_opsd_special_v1
```

OPSD 之后再做 `scripts/opsd/merge_lora.py`，得到最终 merged checkpoint，供评测和后续分析使用。

