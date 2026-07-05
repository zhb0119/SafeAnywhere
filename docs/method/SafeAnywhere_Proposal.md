# Proposal: SafeAnywhere - 面向深层安全对齐的分布式安全思考机制

## 0. 一句话方法主张

SafeAnywhere 训练语言模型在自身生成轨迹中的任意风险位置调用一种隐藏、可剥离的安全检查动作，并通过来自特权安全教师的 on-policy 稠密蒸馏，将中途纠偏能力内化到模型参数中，从而避免安全行为只集中在开头几个拒绝 token 上。

## 1. 三篇论文带来的关键启发

### Think Anywhere in Code Generation

这篇论文最值得复用的是一种新的动作空间：推理不必只发生在生成开头。模型可以被训练为在任意 token 位置插入结构化推理块；这些推理块可以在最终输出前被剥离，剥离后的可见输出仍能自然衔接。该论文表明，cold-start SFT 可以教会模型基本格式，而后续在线目标对于学会有用的插入位置非常关键。它也给了我们一组重要诊断：插入位置应当与不确定性或局部困难决策相关；消融实验应当区分“块的位置”和“块的内容”各自带来的作用。

### Safety Alignment Should Be Made More Than Just a Few Tokens Deep

这篇论文给出的核心问题诊断是 shallow safety alignment。许多已对齐模型的大部分安全分布偏移集中在最开始几个输出 token 上，因此只要攻击能够迫使模型产生非拒绝开头，模型就可能沿着有害轨迹继续生成。论文中的 safety recovery examples 表明，训练模型在已经出现一段 unsafe prefix 后恢复到拒绝或安全回应，可以让安全对齐更深；但这种方法仍然是静态数据增强，而不是一个可学习的动态干预策略。

### Reducing the Safety Tax With On-Policy Self-Distillation

这篇论文最关键的训练原则是 on-policy 稠密监督。与其让学生模型模仿固定的安全示范，不如让学生从自身策略采样轨迹，然后在这些轨迹上接受带特权安全上下文的冻结教师模型逐 token KL 监督。这样可以缓解 off-policy mismatch，并改善 safety-reasoning tradeoff。不过，OPSA 的分析仍然强调早期 compliance-decision token；它并没有引入一个可以在序列中后段主动触发的内部安全动作。

## 2. Problem Anchor

- 底层问题：当前安全对齐往往是浅层的，主要通过早期拒绝 token 的分布偏移阻止有害行为，而对后续条件续写的控制较弱。
- 必须解决的瓶颈：模型需要能在任意深度的 unsafe 或 risky partial generation 后恢复安全，包括 prefilling、对抗后缀、解码扰动和下游微调后的安全退化场景。
- 非目标：本工作不是运行时 prompt defense，不是外部安全分类器 wrapper，不承诺绝对 jailbreak immunity，也不主张向用户暴露 chain-of-thought。
- 约束：方法应当能在 0.6B 到 8B 级别的开源 instruction/reasoning model 上用 SFT 加 on-policy distillation 实现；同时保持 helpfulness，避免简单地提高拒绝率。
- 成功条件：相比 SFT、deep-recovery augmentation 和 OPSA 等 baseline，模型应在 prefix/adaptive attack 下有更低 ASR，在 benign 或 adversarial fine-tuning 后有更小安全退化，并且 token-depth 证据应显示安全纠偏不只发生在前 10 到 30 个 token。

## 3. 核心技术缺口

现有方法分别解决了问题的一部分，但都不完整：

- Deep-alignment augmentation 可以训练模型从合成 unsafe prefix 中恢复，但恢复深度和纠偏行为由固定数据决定。
- OPSA 提供了 on-policy token-level supervision，但如果没有显式的中途安全动作，最强监督仍可能集中在早期拒绝决策位置。
- Think Anywhere 提供了任意位置推理机制，但其奖励来自代码正确性；安全任务没有同样干净的 verifier，并且还需要防止隐藏块中生成危险内容。

缺失的核心机制是：一个可以在生成过程中任意位置被调用的、可学习的安全干预动作，以及一个能教会模型在自身风险轨迹上何时调用该动作的 on-policy 目标。

## 4. 方法设计

### 4.1 名称与接口

使用隐藏安全检查块：

```text
<|safe_check|>
risk: ...
decision: refuse | redirect | answer
correction: ...
<|/safe_check|>
```

该块在训练和推理时属于模型内部 token stream，但在最终展示给用户前被剥离。为了避免 unsafe chain-of-thought 泄露，这个块不被训练成自由形式的详细推理，而是一个紧凑的结构化控制状态：风险类别、决策类型和纠偏意图。

剥离所有 `<|safe_check|>...</|safe_check|>` span 后，可见回答必须保持连贯；对 benign prompt 要有帮助，对 harmful prompt 要安全。

### 4.2 复杂度预算

- 复用组件：
  - 基础 instruction/reasoning model。
  - 来自同一模型族的冻结特权上下文教师，用于 KL 蒸馏。
  - 可选的更强外部安全教师，仅用于初始标注和过滤。
  - 现有安全 judge，例如 Llama-Guard 或 GPT-based judge，用于评估和过滤。
- 新增可训练组件：
  - 主方法不需要额外分类器。
  - 只训练基础模型参数或 adapter，以及两个专用安全检查 delimiter token。
- 明确排除：
  - 不使用独立运行时 monitor。
  - 不使用推理时 verifier loop。
  - 不使用 multi-agent debate。
  - 不依赖向用户暴露安全推理。

## 5. 训练流水线

### Stage A: 安全检查 token 与格式初始化

加入两个特殊 delimiter token：`<|safe_check|>` 和 `<|/safe_check|>`。初始化时混合已有 delimiter token embedding 与安全相关 token embedding。训练分两步：

1. Embedding alignment：冻结模型主体，只训练新 token 的 input embedding 和 LM head。
2. Joint SFT warmup：在格式化样本上训练 LoRA 或全参数，使模型学会打开、关闭并条件化使用安全检查块。

这借鉴了 Think Anywhere 中 dedicated trigger token 的经验：单个控制 token 比很长的多 token delimiter 更容易稳定学习。

### Stage B: 构造带分布式安全检查块的教师数据

从四类 prompt 构建 cold-start 数据集：

- Harmful prompts。
- 不应被拒绝的 benign prompts。
- 需要安全重定向的 borderline 或 dual-use prompts。
- 被攻击过的 prompts，包括 adversarial suffix、unsafe prefill、role-play jailbreak 和 decoding-induced non-refusal prefix。

对每个 harmful 或 risky 样本，在多个深度构造带安全检查块的轨迹：

1. 无 unsafe prefix：教师可以立即拒绝或安全回答。
2. 早期 unsafe prefix：从 1 到 30 个 token 采样 k，并要求教师恢复。
3. 中后段 unsafe prefix：从更宽范围采样 k，例如 30 到 200 个 token，或按句子/段落边界采样。
4. Student-like partial trajectory：使用早期学生 rollout 作为 prefix，并要求教师在第一个风险位置插入安全检查块。

教师标注指令如下：

- 只有当当前轨迹即将越过安全边界或需要重定向时，才插入 `<|safe_check|>`。
- 块内容保持紧凑，使用 policy-level 表达。
- 剥离安全检查块后，可见回答必须安全且流畅。
- 对 benign prompt，避免不必要的安全检查和拒绝。

过滤标准：

- delimiter 结构合法；
- 剥离后的可见输出安全；
- 隐藏块内不包含危险操作细节；
- benign prompt 上回答仍然有帮助；
- 块频率和长度落在目标预算内。

### Stage C: Cold-Start SFT

在教师标注轨迹上训练学生模型：

```text
L_SFT = CE(full sequence)
      + lambda_open * CE(open/close delimiter positions)
      + lambda_visible * CE(visible safe continuation)
```

delimiter 打开/关闭决策的权重应高于长块内容本身，因为关键能力不是写出漂亮的隐藏推理，而是在正确的 prefix state 调用纠偏动作。

该阶段让模型具备基本动作空间。和 Think Anywhere 的 "Only Cold Start" 消融类似，预计仅靠该阶段还不足以形成鲁棒的插入策略。

### Stage D: On-Policy 分布式安全蒸馏

从 SFT 模型初始化学生模型。冻结一个 SFT 模型副本作为教师，并给教师条件化特权上下文：

- Harmful/risky context："This trajectory may contain unsafe compliance. Insert a compact safety check if needed and recover to a safe answer."
- Benign context："This is benign. Answer helpfully and do not add unnecessary refusal or safety disclaimers."

学生模型在没有特权上下文的条件下采样 rollout。对每个 prompt q 和采样 prefix y_<t，比较学生的 next-token distribution 与特权教师的 next-token distribution：

```text
L_OPSD = E_{q, y ~ pi_theta(.|q)} sum_t w_t *
         D_sym(pi_teacher(. | c_type, q, y_<t) || pi_student(. | q, y_<t))
```

其中 `D_sym` 是 forward/reverse KL 的混合形式，可参考 OPSA。权重 w_t 可以设计为：

- 默认设为 1，依赖 KL 自然集中在教师和学生存在差异的位置；
- 当教师对 `<|safe_check|>` 分配高概率时增大；
- 当 prefix 包含已知 compliance marker、unsafe-prefill fragment 或 policy-risk category 时增大；
- 在很长的隐藏块内部降低，以避免过度优化冗长推理文本。

加入一个小的预算惩罚：

```text
L_budget = gamma_freq * max(0, block_count - B)^2
         + gamma_len  * max(0, block_tokens - L)^2
```

完整目标：

```text
L = L_OPSD + beta_ref * KL(pi_student || pi_SFT_ref on benign prompts)
    + L_budget
```

该阶段是主贡献。它让模型在自身风险状态上学习纠偏，而不是只模仿固定教师示范。

### Stage E: 可选的微调耐久性正则

对于 downstream fine-tuning 实验，可以加入一个轻量 token-wise regularizer，用来保持安全检查调用和安全恢复概率：

```text
L_FT = L_task + eta * sum_{t in R(q,y)} KL(pi_ft(.|q,y_<t) || pi_safeanywhere(.|q,y_<t))
```

其中 R(q,y) 是从 SafeAnywhere 训练/评估池中收集的风险 prefix。该部分应定位为部署扩展，而不是论文主贡献。

## 6. 为什么这不只是“Prompted Safety CoT”

该方法并不依赖推理时可见或隐藏的 prompt。特权上下文只在训练中用于构造教师分布；部署时模型只接收普通用户 prompt，并通过参数内化安全检查动作。

该方法也不只是拒绝模板模仿。主要监督对象是 prefix state 和 action timing：即使模型已经进入风险部分轨迹，也要学会打开安全检查块并恢复。这直接针对 shallow alignment 未充分控制的条件分布：

```text
pi(h_{>k} | x, h_{<=k})
```

其中 k 取多个深度，而不仅是 k = 0。

## 7. 主要研究主张

### Claim 1: 分布式安全检查动作能将安全对齐推进到早期拒绝 token 之后。

需要的证据：

- SafeAnywhere 与 base/aligned model 的 per-token KL 或 safety divergence 在后续风险 prefix 上仍然升高，而不只集中在 1 到 30 个 token。
- 当 unsafe prefix 被注入到多个深度时，模型仍保持较高 recovery success。
- 安全检查块的插入位置与风险/不确定性相关，而不是只出现在开头。

### Claim 2: On-policy distillation 对鲁棒的中途恢复是必要的。

需要的证据：

- SFT-only SafeAnywhere 能学会格式，但 skip rate 更高，并且在 on-policy attack 下 ASR 更差。
- Off-policy recovery augmentation 能提升 prefilling robustness，但在 student-sampled risky trajectory 上更弱。
- On-policy KL 能降低模型在风险 prefix 后直接继续生成有害文本的概率。

### Claim 3: 该方法提升鲁棒性，而不是简单提高拒绝率。

需要的证据：

- harmful 和 attacked prompts 上 ASR 更低。
- benign prompts 上 helpfulness 持平或提升。
- 相比强拒绝风格 baseline，在 benign safety-adjacent prompts 上 over-refusal 更低。

## 8. 实验计划

### 8.1 模型

先从小中型开源模型开始：

- Qwen3-0.6B 和 Qwen3-1.7B，用于快速迭代。
- Qwen3-8B 或 DeepSeek-R1-Distill-8B，用于规模验证。

早期开发可用 LoRA，主实验如计算资源允许则使用全参数微调。

### 8.2 Baselines

- 初始 instruction/reasoning model。
- 标准安全 SFT，使用 refusal/helpfulness 数据。
- 来自 "Safety Alignment Should Be Made More Than Just a Few Tokens Deep" 的 deep-recovery augmentation。
- ThinkSafe-style self-distilled safety SFT。
- OPSA-style on-policy privileged-context KL，但不引入 safety-check action。
- SafeAnywhere-SFT only。
- 只允许 front-only safety check 的 SafeAnywhere 变体。
- 完整 SafeAnywhere。

### 8.3 安全评估

使用已有 harmfulness benchmark 和攻击：

- HEx-PHI / HarmBench / AdvBench-style harmful prompts。
- Prefilling attacks，注入深度为 5、10、20、40、80、160 个可见 token。
- GCG 或 AutoDAN adversarial suffix attacks。
- Decoding-parameter attacks，使用 best-of-N sampling 并由 safety classifier 判断。
- Role-play 与 prompt-injection jailbreak sets。
- Adaptive attacks，尝试抑制、伪造或滥用 `<|safe_check|>` token。

指标：

- ASR / harmfulness rate。
- 多样本下的 pass@N ASR。
- unsafe prefix 后的 recovery success。
- safety-check skip rate：风险 prefix 后没有安全检查块却继续有害生成的比例。
- stripped-output safety：移除隐藏块后用户可见回答的安全性。

### 8.4 Helpfulness 与 Safety Tax

测量：

- AlpacaEval 或 MT-Bench-style helpfulness。
- GSM8K/MATH 或代码/推理 benchmark，取决于基础模型。
- benign 与 safety-adjacent prompts 上的 over-refusal。
- 剥离隐藏块后的可见回答流畅度。
- 隐藏块带来的 token overhead。

### 8.5 Fine-Tuning Robustness

复现 shallow-vs-deep paper 中的 fine-tuning attack 设置：

- Harmful-example fine-tuning。
- Identity-shift fine-tuning。
- Backdoor poisoning。
- 在 summarization、SQL、math 或 instruction-following 任务上的 benign fine-tuning。

报告：

- fine-tuning 前后的 ASR。
- 下游任务 utility。
- risk-prefix probes 上 safety-check invocation 的保持情况。

### 8.6 机制分析

- Token-depth KL curves：比较不同方法相对于 base model 在哪些位置改变了分布。
- Block position distribution：早期/中期/后期位置，并与 risk score 和 entropy 对齐。
- Boundary-vs-content ablation：将隐藏块内容替换为 padding，但保留 delimiter，测试 timing 与 content 各自作用。
- Trigger-token ablation：专用单 token 与普通多 token delimiter 对比。
- Risk-gating ablation：uniform KL、teacher-probability-weighted KL、risk-weighted KL 对比。
- Block budget ablation：展示安全收益与 token overhead 的 tradeoff。

## 9. 预期结果模式

预期优势不是 SafeAnywhere 拒绝更多，而是：

- 在 clean harmful prompts 上，达到强安全 baseline 水平；
- 在 prefilling 和 suffix attack 下，更常从非拒绝 prefix 中恢复；
- 在 downstream benign fine-tuning 后，因为模型拥有多个后续恢复点，而不是只有脆弱的初始拒绝模式，所以安全保持更好；
- 在 benign prompts 上，由于 benign privileged context 和 reference KL 抑制 over-refusal，不会不必要地插入安全块或拒绝；
- hidden block 的位置出现在风险转折处，而不是机械地集中在最初 token。

## 10. 最高风险假设

1. 隐藏块可能成为生成危险细节的位置，即使最终被剥离。
   - 缓解：训练紧凑结构化块；过滤隐藏内容；分别评估 hidden stream 和 visible stream。

2. 模型可能过度使用安全检查，损害 helpfulness。
   - 缓解：benign privileged context、block budget penalty、over-refusal evaluation 和 block frequency target。

3. 教师可能不知道何时在较深位置插入检查。
   - 缓解：从真实学生 rollout 和 unsafe-prefill attack 中构造 prefix；用 prefix-state 上的 teacher flip/recovery rate 选择上下文。

4. 攻击者可能显式操纵 safety-check token。
   - 缓解：加入提及、抑制或伪造 tag 的 adaptive attack；部署时可将这些 token 保留为普通用户不可访问的控制 token。

5. On-policy KL 仍可能集中在开头。
   - 缓解：刻意在多个深度采样 risk prefix，并在教师对安全检查块赋高概率的位置上调权重。

## 11. 论文叙事

### 建议标题

SafeAnywhere: Deep Safety Alignment via On-Policy Distributed Safety Deliberation

### 核心贡献

一种训练机制：将 deep safety alignment 从静态 recovery examples 推进为一个可学习的内部动作。模型可以在任意生成位置决定执行一个紧凑的隐藏安全检查，并在可见有害续写发生前恢复。

### 与最近工作的区别

- 相比 Think Anywhere：将任意位置结构化推理应用到安全对齐；由于安全任务没有代码执行式 correctness reward，因此用 privileged-teacher on-policy distillation 替代 RLVR。
- 相比 shallow-vs-deep alignment：将固定 safety recovery examples 泛化为学生自身 prefix 上的动态 on-policy recovery。
- 相比 OPSA：保留 on-policy dense KL，但加入显式 mid-sequence safety-check action，并评估安全深度是否真的扩展到早期 compliance-decision token 之后。

### 最小但强的论文 claim

当模型不仅被训练为选择安全的第一个回应 token，还被训练为在自身风险生成 prefix 上学习一个内部、任意位置的恢复动作时，安全对齐会更鲁棒。

## 12. 最小实现路线图

1. 在 Qwen3-0.6B 或 Qwen3-1.7B 上复现一个小规模 OPSA-style baseline。
2. 构造 3k 到 5k 条带教师插入紧凑安全检查块的 SafeAnywhere cold-start examples。
3. 训练 SafeAnywhere-SFT，并验证格式合法性、剥离后流畅度和 helpfulness。
4. 在 SafeChain/HarmBench-style prompt mixture 上运行 1 到 3 个 epoch 的 on-policy KL distillation。
5. 评估 prefilling ASR、over-refusal 和 token-depth KL。
6. 只有在机制成立后，再扩展到 adaptive suffix attack 和 downstream fine-tuning durability。

## 13. Go / No-Go 标准

Go 条件：

- SFT 学会合法 hidden safety-check 格式，malformed output 低于 2%。
- 完整 SafeAnywhere 在 20 到 80 token 的 prefilling ASR 上，相比 OPSA 或 deep-recovery augmentation 至少下降 30%。
- Benign over-refusal 增长低于 3 个百分点。
- 在攻击 rollout 中，安全检查位置至少有相当一部分出现在中后段风险 prefix。

No-go 或 pivot 条件：

- 即使做了深度采样，on-policy KL 仍然塌缩为 front-only refusal。
- 隐藏块中出现无法通过过滤控制的危险细节。
- 鲁棒性收益只来自更高拒绝率。
- Token overhead 过高，不适合实际部署。

