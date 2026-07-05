# SafeAnywhere 数据构造 Pipeline

## 1. 本阶段目标

本阶段构造 SafeAnywhere 的 cold-start SFT 数据。目标不是再做一个普通的 harmful/benign 拒绝数据集，而是训练模型学会：

```text
在合适的生成状态下，调用一个紧凑、隐藏、可剥离的安全检查动作；
然后继续生成安全且对用户可见的回答。
```

所有 prompt 都统一来自 SafeChain：

```text
D:\paper\alignment\repo\OPSA\data\jsonl\UWNSL__SafeChain__train.jsonl
```

只使用 SafeChain 自带的四类标签：

```text
vanilla_benign
adversarial_benign
vanilla_harmful
adversarial_harmful
```

这一阶段暂时不混入 `Harmful-HEx-PHI.jsonl` 等外部 unsafe-prefix 数据源。这样做的好处是：prompt 来源、类别体系、论文叙述都更统一。我们只改变模型所处的 assistant prefix 状态，而不是混合多个异构数据集。

## 2. 核心思想：prompt class × prefix state

每条训练样本由两个轴决定：

```text
prompt class：用户请求本身属于哪类风险
prefix state：assistant 已经生成到什么状态
```

也就是说，我们不是只问“这个 prompt 是不是 harmful”，而是问：

```text
在这个 prompt 下，如果 assistant 已经生成了一段内容，
模型现在应该继续回答、拒绝、重定向，还是从风险轨迹中恢复？
```

## 3. Prompt Class 统一分类

SafeChain 的四类标签作为统一 taxonomy。

| 类别                    | 含义                                              | 模型应该学到                                          |
| ----------------------- | ------------------------------------------------- | ----------------------------------------------------- |
| `vanilla_benign`      | 普通无害请求                                      | 直接帮助，不要乱拒绝，也不要到处插安全检查块。        |
| `adversarial_benign`  | 表面像攻击、越狱或敏感请求，但实质是无害/允许回答 | 避免 over-refusal；必要时做短安全检查，然后继续帮助。 |
| `vanilla_harmful`     | 直接有害请求                                      | 识别风险，拒绝或安全重定向。                          |
| `adversarial_harmful` | 伪装、诱导、角色扮演、jailbreak 式有害请求        | 识别包装后的有害意图，并安全恢复。                    |

不使用 SafeChain 的 `response` 字段。SafeChain 的 response 经常包含很长的 `<think>` 式推理，和 SafeAnywhere 想要的“紧凑 hidden safety check”不一致。更稳妥的做法是：

```text
使用 SafeChain 的 instruction 和 label 作为 prompt pool；
由我们自己的 teacher 或重写脚本生成 SafeAnywhere 格式的 target。
```

## 4. Prefix State 统一分类

本阶段使用三种 prefix state。

| prefix state         | assistant prefix                                       | 训练目的                                             |
| -------------------- | ------------------------------------------------------ | ---------------------------------------------------- |
| `no_prefix`        | assistant 还没有开始回答                               | 学会初始决策：直接答、拒绝、重定向，或插入安全检查。 |
| `unsafe_prefix`    | assistant 已经进入非拒绝或风险轨迹                     | 学会从中途有害轨迹恢复，而不是继续写下去。           |
| `ambiguous_prefix` | assistant 当前上下文看起来敏感或风险化，但实际仍应帮助 | 学会不要过度拒绝；安全检查也可以得出`answer`。     |

训练输入统一为：

```text
User: {instruction}
Assistant: {assistant_prefix}{target}
```

训练 loss 只作用在 `target` 上：

```text
User prompt:      不算 loss
assistant_prefix: 不算 loss
target:           算 loss
```

尤其是 `unsafe_prefix`：它只是上下文，不是要模型模仿生成的目标。

## 5. 什么是 ambiguous_prefix？

`ambiguous_prefix` 指的是：assistant 当前生成状态看起来和安全风险有关，但实际上还没有越过安全边界，正确行为应该是继续帮助，而不是拒绝。

它和 `unsafe_prefix` 的区别非常关键：

```text
unsafe_prefix:
  assistant 已经朝有害回答方向走了。
  正确动作是 stop / recover / refuse / redirect。

ambiguous_prefix:
  assistant 只是处在一个敏感、双用途、或看起来像攻击的上下文里。
  正确动作是 answer with boundaries，而不是拒绝。
```

引入 `ambiguous_prefix` 是为了防止模型学到错误捷径：

```text
出现敏感词 / 攻击语气 / 安全相关上下文 -> 拒绝
```

我们希望模型学到的是：

```text
如果真实意图是 benign，即使表面敏感，也应该回答；
安全检查不是拒绝按钮，而是一个状态判断动作。
```

典型 `ambiguous_prefix` 场景包括：

- 防御性网络安全、隐私保护、风险教育类问题。
- 医疗、法律、金融等需要边界但可以给一般性信息的问题。
- prompt 含有高风险关键词，但实际是在问安全、历史、检测、预防或合规内容。
- adversarial-benign prompt：形式上像 jailbreak，但真实请求是允许的。
- assistant 已经用谨慎语气开头，但应该继续给有帮助的回答，而不是转成拒绝。

`ambiguous_prefix` 的标准 target 通常是：

```text
<|safe_check|>
risk: ambiguous_benign
decision: answer
correction: answer with appropriate safety boundaries
<|/safe_check|>
{visible_helpful_answer}
```

这类样本是控制 over-refusal 的核心。

## 6. 推荐数据规模与配比

第一版建议构造 5,000 条 cold-start SFT 数据。

| 数据块                           |  数量 | 来源类别                                      | 作用                                      |
| -------------------------------- | ----: | --------------------------------------------- | ----------------------------------------- |
| 普通 benign 直接回答             |   900 | `vanilla_benign`                            | 保持普通帮助能力。                        |
| 普通 benign answer-check         |   100 | `vanilla_benign`                            | 让模型知道 safe check 也可以导向 answer。 |
| adversarial benign 直接回答      |   500 | `adversarial_benign`                        | 抑制 over-refusal。                       |
| adversarial benign answer-check  |   500 | `adversarial_benign`                        | 学会在安全邻近场景中继续帮助。            |
| harmful no-prefix 拒绝/重定向    | 1,000 | `vanilla_harmful` + `adversarial_harmful` | 学会初始 harmful prompt 处理。            |
| unsafe-prefix recovery           | 1,600 | `vanilla_harmful` + `adversarial_harmful` | 学会中途风险恢复。                        |
| ambiguous-prefix answer recovery |   400 | `adversarial_benign`                        | 学会从模糊但 benign 的状态继续回答。      |

总数：

```text
900 + 100 + 500 + 500 + 1000 + 1600 + 400 = 5000
```

如果 teacher 标注成本太高，可以先做 3,000 条，保持同样比例。

## 7. safe_check 固定模板

隐藏块使用固定三行模板：

```text
<|safe_check|>
risk: {risk}
decision: {decision}
correction: {correction}
<|/safe_check|>
```

这个块必须短，不能写成长 chain-of-thought。它只表达策略级状态，不表达详细推理。

### 7.1 risk 枚举

```text
benign
ambiguous_benign
safety_adjacent
harmful_request
adversarial_harmful
unsafe_continuation
```

### 7.2 decision 枚举

```text
answer
refuse
redirect
recover
```

### 7.3 correction 枚举

```text
answer directly
answer with appropriate safety boundaries
clarify safe intent before helping
refuse briefly and offer safe alternative
refuse the unsafe goal and redirect to safe information
stop unsafe continuation and recover safely
```

后续如果需要更细分类，可以扩展枚举，但第一版越短越好。

## 8. 各类别 target 模板

### 8.1 vanilla_benign

大多数样本不插安全检查：

```text
{visible_helpful_answer}
```

少量样本插入 answer-check：

```text
<|safe_check|>
risk: benign
decision: answer
correction: answer directly
<|/safe_check|>
{visible_helpful_answer}
```

建议比例：

```text
90% direct answer
10% answer-check
```

### 8.2 adversarial_benign

直接回答子集：

```text
{visible_helpful_answer_with_boundaries_if_needed}
```

answer-check 子集：

```text
<|safe_check|>
risk: ambiguous_benign
decision: answer
correction: answer with appropriate safety boundaries
<|/safe_check|>
{visible_helpful_answer_with_boundaries_if_needed}
```

建议比例：

```text
50% direct answer
50% answer-check
```

### 8.3 vanilla_harmful

```text
<|safe_check|>
risk: harmful_request
decision: refuse
correction: refuse briefly and offer safe alternative
<|/safe_check|>
{visible_safe_refusal_or_redirection}
```

### 8.4 adversarial_harmful

```text
<|safe_check|>
risk: adversarial_harmful
decision: redirect
correction: refuse the unsafe goal and redirect to safe information
<|/safe_check|>
{visible_safe_refusal_or_redirection}
```

### 8.5 unsafe_prefix

```text
{masked_assistant_prefix}
<|safe_check|>
risk: unsafe_continuation
decision: recover
correction: stop unsafe continuation and recover safely
<|/safe_check|>
{visible_safe_recovery_response}
```

这里 `{masked_assistant_prefix}` 不算 loss。

### 8.6 ambiguous_prefix

```text
{masked_assistant_prefix}
<|safe_check|>
risk: ambiguous_benign
decision: answer
correction: answer with appropriate safety boundaries
<|/safe_check|>
{visible_helpful_answer}
```

这里 `{masked_assistant_prefix}` 也不算 loss。

## 9. unsafe_prefix 构造方式

`unsafe_prefix` 只从 harmful 类构造：

```text
vanilla_harmful
adversarial_harmful
```

不要直接复制长篇有害回答，也不要把具体危险操作细节写入训练目标。推荐构造“非拒绝轨道”式 prefix，让模型知道自己已经开始走错方向，但不训练具体有害步骤。

推荐 prefix type：

| prefix type                   | 含义                                                           |
| ----------------------------- | -------------------------------------------------------------- |
| `acceptance_prefix`         | assistant 已经开始表达顺从。                                   |
| `outline_prefix`            | assistant 已经开始列回答结构，但该结构会导向有害帮助。         |
| `roleplay_prefix`           | assistant 接受了危险角色或 jailbreak 设定。                    |
| `partial_completion_prefix` | assistant 已经写出一小段风险方向的回答，但不包含详细危险步骤。 |

推荐 prefix 深度：

```text
8, 16, 32, 64, 128 tokens
```

这些 prefix token 全部作为 masked context：

```text
labels[prefix_tokens] = -100
labels[target_tokens] = token ids
```

## 10. ambiguous_prefix 构造方式

`ambiguous_prefix` 只从 benign 但安全邻近的类构造：

```text
adversarial_benign
```

推荐 prefix type：

| prefix type              | 含义                                                     |
| ------------------------ | -------------------------------------------------------- |
| `caution_prefix`       | assistant 以谨慎语气开头，但不应该拒绝。                 |
| `boundary_prefix`      | assistant 先说明安全边界，然后继续给允许的帮助。         |
| `clarification_prefix` | assistant 将请求理解为教育、防御、分析、合规或高层信息。 |
| `false_alarm_prefix`   | prefix 含有容易触发安全过滤的词，但上下文实际 benign。   |

目标不是让模型“忽略风险”，而是让模型学会：

```text
检查风险 -> 判断为允许回答 -> 给有边界的帮助
```

## 11. JSONL Schema

建议导出的训练数据使用统一 schema：

```json
{
  "id": "safeanywhere_sft_000001",
  "source": "safechain",
  "prompt_class": "adversarial_harmful",
  "prefix_state": "unsafe_prefix",
  "prefix_type": "outline_prefix",
  "prefix_depth": 32,
  "instruction": "...",
  "assistant_prefix": "...",
  "target": "<|safe_check|>\nrisk: unsafe_continuation\ndecision: recover\ncorrection: stop unsafe continuation and recover safely\n<|/safe_check|>\n...",
  "loss_on_prefix": false,
  "teacher_model": "...",
  "filter_status": "passed"
}
```

对于 `no_prefix` 样本：

```json
{
  "prefix_state": "no_prefix",
  "prefix_type": "none",
  "prefix_depth": 0,
  "assistant_prefix": ""
}
```

## 12. 构造流程

### Step 1：读取 SafeChain

读取：

```text
D:\paper\alignment\repo\OPSA\data\jsonl\UWNSL__SafeChain__train.jsonl
```

保留：

```text
instruction
label
```

可选保留：

```text
response
```

但 `response` 只作为弱参考，不直接作为最终训练 target。

### Step 2：类别均衡采样

按照第 6 节的 5k 配比采样。

需要去重：

```text
完全重复 instruction 去掉
语义近重复可用 embedding / MinHash 去掉
```

### Step 3：分配 prefix state

给每条样本分配：

```text
no_prefix
unsafe_prefix
ambiguous_prefix
```

规则：

```text
vanilla_benign       -> no_prefix
adversarial_benign   -> no_prefix 或 ambiguous_prefix
vanilla_harmful      -> no_prefix 或 unsafe_prefix
adversarial_harmful  -> no_prefix 或 unsafe_prefix
```

### Step 4：构造 assistant_prefix

`no_prefix`：

```text
assistant_prefix = ""
```

`unsafe_prefix`：

```text
构造 acceptance / outline / roleplay / partial completion 风格 prefix
```

`ambiguous_prefix`：

```text
构造 caution / boundary / clarification / false alarm 风格 prefix
```

### Step 5：teacher 生成 target

teacher 生成：

```text
safe_check block
visible continuation
```

要求：

```text
hidden block 使用固定模板
risk / decision / correction 必须来自枚举
visible answer 在去掉 hidden block 后仍然自然、可读、安全
```

### Step 6：过滤

过滤规则：

1. delimiter 合法：

```text
需要 safe_check 的样本必须正好有一个 <|safe_check|> 和一个 <|/safe_check|>
开始 token 必须在结束 token 之前
```

2. 枚举合法：

```text
risk / decision / correction 必须在允许列表里
```

3. hidden block 安全：

```text
不能包含详细危险操作、攻击步骤、违法细节
```

4. visible output 安全：

```text
harmful / unsafe_prefix 样本剥离 hidden block 后仍然安全
```

5. benign helpful：

```text
vanilla_benign / adversarial_benign / ambiguous_prefix 样本不能过度拒绝
```

6. 长度预算：

```text
safe_check block 建议 <= 64 tokens
整体样本不超过模型上下文长度
```

7. prefix loss mask：

```text
prefix_state != no_prefix 时，assistant_prefix 必须被 mask，不算 loss
```

### Step 7：导出 train/validation

建议：

```text
90% train
10% validation
```

分层保持：

```text
prompt_class
prefix_state
prefix_type
prefix_depth
```

## 13. 论文中可以这样表述

中文表述：

```text
我们基于 SafeChain 的四类标签构造统一的 prompt-risk taxonomy：vanilla benign、adversarial benign、vanilla harmful 和 adversarial harmful。与混合多个异构安全数据集不同，我们保持 prompt 来源和标签体系一致，只改变模型所处的生成状态。对于 harmful 类请求，我们合成多深度的 masked non-refusal prefix，让模型在已经进入风险轨迹后生成紧凑的隐藏安全检查块并恢复到安全可见回答。对于 adversarial-benign 请求，我们构造 ambiguous-prefix 状态，其正确决策是 answer 而不是 refuse，从而抑制过度拒绝。该数据设计使模型学到：安全检查不是拒绝模板，而是一个可在生成中途调用的控制动作。
```

英文表述：

```text
We construct SafeAnywhere-SFT from a single unified source, SafeChain, using its four-way prompt-risk taxonomy: vanilla benign, adversarial benign, vanilla harmful, and adversarial harmful. Rather than mixing heterogeneous safety datasets, we transform a subset of SafeChain prompts into controlled prefix states. For harmful prompts, we synthesize masked non-refusal prefix states at multiple depths and train the model to emit a compact hidden safety-check action followed by a safe visible recovery. For adversarial-benign prompts, we include ambiguous-prefix states whose correct decision is to answer with appropriate boundaries. This design teaches the model that a safety check is not synonymous with refusal; it is a mid-generation control action that can answer, refuse, redirect, or recover depending on the current state.
```
