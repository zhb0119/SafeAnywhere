# SafeAnywhere SFT 实现

SafeAnywhere 的 SFT 只做一件事：把 cold-start 模型学到能在危险前缀或越狱式 prefill 后，稳定产生带 `<safety_think>` 的安全续写。当前有两条分支：普通 SFT 和 special-token SFT。

## 数据

训练数据来自 `build/data_build/safeanywhere_sft_v1/`，核心导出是：

```text
train_lf_v1_spanmasked.jsonl
val_lf_v1_spanmasked.jsonl
```

数据由两类样本组成：

- `safechain`：标准 prompt-level 安全对话。
- `harmful_prefix`：危险前缀 / assistant prefill 恢复样本。

LLaMA-Factory 导出时使用 `messages + loss_weight` 结构：

- `user` 和非监督内容：`loss_weight=0`
- `assistant` 的目标回复：`loss_weight=1`
- `harmful_prefix` 里的 `assistant_prefill`：`loss_weight=0`
- `harmful_prefix` 里的恢复目标：`loss_weight=1`

当前模板是 `template: safeanywhere_qwen3_nothink`，它把 message span 展开成 `input_ids / labels / loss_weights`，因此 SFT 实际优化的是带 span mask 的加权 CE，而不是额外的自定义 RL 目标。

## 普通 SFT

普通分支直接在原始 tokenizer 上训练文本形式的 `<safety_think>...</safety_think>` 块。

代表配置：

```text
configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml
configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v2.yaml
```

实现要点：

- 基座模型：`../models/Qwen3-0.6B`
- LoRA：`target_modules: all`
- 模板：`safeanywhere_qwen3_nothink`
- 损失：LLaMA-Factory 标准 weighted CE
- 训练产物：`runs/sft/qwen3_0_6b_v1` / `runs/sft/qwen3_0_6b_v2`

普通 SFT 不改 tokenizer，不保存特殊 token 行，也不加边界 token 特权。它的作用是先让模型学会“安全块的文本形状”和“危险输入后的基本恢复行为”。

## Special-token SFT

special-token 分支先把 `<safety_think>` 和 `</safety_think>` 加入词表，再做 SFT。

第一步是创建 special-token base：

```bash
python scripts/sft/add_safety_think_special_tokens.py \
  --base-model ../models/Qwen3-0.6B \
  --output runs/sft_special/qwen3_0_6b_safety_think_base
```

这个脚本会：

- 将两个 token 注册为 `additional_special_tokens`
- resize embedding / lm_head
- 用语义源 + functional source 初始化新 token 行
- 保存 `safeanywhere_special_tokens.json`

第二步是 special-token SFT：

```text
configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1_special_tokens.yaml
configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v2_special_tokens.yaml
```

运行入口不是原始 LLaMA-Factory 命令，而是：

```bash
python scripts/sft/run_special_sft.py <config>
```

这个入口做了两层 patch：

- LoRA 侧：`modules_to_save = [embed_tokens, lm_head]`，并启用 `sparse_token_grad`
- loss 侧：对 `SFTTrainer.compute_loss` 追加 boundary-token 加权 CE

special-token SFT 的关键约束：

- `embed_tokens` 和 `lm_head` 必须保留，否则新增 token 行无法随 adapter 一起导出
- 实际梯度只允许流到 `<safety_think>` / `</safety_think>` 两个新增 token 行
- `ensure_weight_tying: true` 用于维持输入 embedding 和输出 head 的一致性

v2 相比 v1 的区别：

- 训练更久：`num_train_epochs: 6`
- 加了 boundary loss：
  - `<safety_think>` 权重更高
  - `</safety_think>` 次之
  - block 内 token 也有中等权重

当前推荐使用 v2：

```text
runs/sft_special/qwen3_0_6b_safety_think_base
runs/sft_special/qwen3_0_6b_v2
```

## 产物链路

SFT 的标准链路是：

```text
data build
  -> span-masked LLaMA-Factory JSONL
  -> SFT LoRA
  -> merged checkpoint
  -> OPSD
```

普通分支的 merged checkpoint 进入 `runs/merged/qwen3_0_6b_sft_v1`，special-token 分支进入 `runs/merged/qwen3_0_6b_sft_special_v1`。后续 OPSD 必须和上游分支一致，不能混用原始 base 和 special-token base。

