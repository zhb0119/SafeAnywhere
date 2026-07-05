---
title: "SafeAnywhere：面向深层安全对齐的分布式安全思考机制"
date: ""
fontsize: 11pt
geometry: margin=0.9in
colorlinks: true
header-includes:
  - \setlength{\parindent}{0pt}
  - \setlength{\parskip}{0.45em}
  - \setlength{\fboxsep}{8pt}
---

## 1. 研究动机

现有 LLM 安全对齐往往是“浅层”的：模型主要通过开头几个拒绝 token 来实现安全，例如生成 “I cannot...” 后进入拒绝轨迹。一旦攻击者通过 prefilling、对抗后缀、解码扰动或少量微调，让模型绕过这些开头拒绝 token，模型后续条件分布仍可能回到有害生成轨迹。这一 shallow safety alignment 现象及其与 prefilling、adversarial suffix、decoding parameter attack 和 fine-tuning attack 的关系，主要来自 Qi et al. 的分析 [1]。

已有工作进一步表明，安全对齐应该不止“前几个 token 深”，并提出通过 unsafe prefix 后的 safety recovery examples 将安全影响推进到更深 token 位置 [1]。但这类 deep alignment 方法多依赖静态 recovery 数据；OPSA 虽然使用 on-policy token-level KL 蒸馏，在学生模型自己的 rollout 上接受 privileged-context teacher 的稠密监督，但其监督和分析仍主要集中在早期 compliance-decision token [2]。因此，希望进一步让模型在生成全过程中具备动态安全校验和中途纠偏能力。

## 2. 核心想法

借鉴 Think Anywhere 中“可在任意 token 位置插入结构化推理块”的范式 [3]，提出 SafeAnywhere：训练模型在生成序列中的任意风险位置插入隐藏的安全检查块：

\begin{center}
\fbox{%
\begin{minipage}{0.76\linewidth}
\ttfamily\small
\textless|safe\_check|\textgreater\\
\quad risk: ...\\
\quad decision: refuse | redirect | answer\\
\quad correction: ...\\
\textless|/safe\_check|\textgreater
\end{minipage}}
\end{center}

该安全检查块只作为模型内部控制动作使用，最终展示给用户前会被剥离。剥离后，用户看到的回答仍需保持安全、流畅、自然。这一“插入后可剥离”的接口形式借鉴了 Think Anywhere 中 `<thinkanywhere>` 块剥离后代码仍可执行的设计 [3]，但这里将其改造为安全对齐中的隐藏控制动作。

核心目标不是让模型背诵开头拒绝模板，而是让模型在已经进入 risky partial generation 后，仍能主动触发安全检查并恢复到安全轨迹。这直接对应 [1] 中提出的深层安全对齐目标：不仅降低 $\pi(h\mid x)$，还要控制 unsafe prefix 后的条件续写 $\pi(h_{>k}\mid x,h_{\le k})$。

## 3. 方法路线

### Stage 1: 构造分布式安全检查数据

使用强安全教师模型，为 harmful、benign、borderline 和 attacked prompts 构造带 `<|safe_check|>` 的训练样本。安全检查块不仅出现在回答开头，也出现在早期、中段、后段风险位置。多深度 unsafe prefix recovery 的思路来自 [1]，而“任意位置插入结构化块”的动作空间来自 [3]。

数据构造包括：

- 正常 harmful prompt 的安全拒绝；
- unsafe prefix 后的中途恢复 [1]；
- adversarial suffix / prefilling / role-play jailbreak 下的风险纠偏 [1]；
- benign prompt 上避免不必要拒绝，以缓解 over-refusal，这一点与 OPSA 中 harmful/benign type-conditional privileged context 的思路一致 [2]。

过滤要求：

- 剥离 hidden block 后输出安全且流畅；
- hidden block 内不能包含危险操作细节；
- benign prompt 上不能过度拒绝；
- block 频率和长度受控。

### Stage 2: Cold-start SFT

冷启动的第一步是让目标模型真正掌握两个新增 special tokens：`<|safe_check|>` 和 `<|/safe_check|>`。具体做法是先把它们加入 tokenizer，扩展模型 embedding，然后用带安全检查块的样本做 SFT，使模型学会在合适位置打开和关闭 hidden block。

这一步的核心不是让模型立刻学会最优安全判断，而是先建立“可调用的安全检查动作空间”：模型需要知道何时生成 `<|safe_check|>`、何时生成 `<|/safe_check|>`，以及剥离 hidden block 后如何继续生成安全自然的可见回答。Think Anywhere 也表明，模型通常不会自然学会在任意位置调用结构化块，因此需要 cold-start training 先建立这种动作空间 [3]。

$$
\mathcal{L}_{\mathrm{cold}}
= \mathcal{L}_{\mathrm{CE}}^{\mathrm{all}}
+ \lambda_{\mathrm{delim}}\mathcal{L}_{\mathrm{CE}}^{\mathrm{open/close}}
+ \lambda_{\mathrm{visible}}\mathcal{L}_{\mathrm{CE}}^{\mathrm{visible}}
$$

其中 $\mathcal{L}_{\mathrm{CE}}^{\mathrm{open/close}}$ 只计算两个新增 delimiter token 的交叉熵：
`<|safe_check|>` 对应安全检查块的触发位置，`<|/safe_check|>` 对应安全检查块的结束位置。对这两个位置加权，是为了优先学会“何时触发/退出安全检查”；block 内部内容和最终可见回答则由全序列 CE 与 visible continuation CE 共同约束。

### Stage 3: On-policy 安全蒸馏

仅靠 SFT 容易学成静态模板，因此进一步让学生模型从自身策略采样 rollout。冻结一个带 privileged safety context 的教师模型，在学生自己的生成 prefix 上提供逐 token KL 监督。这一 on-policy self-distillation / privileged-context teacher 框架主要来自 OPSA [2]。

$$
\mathcal{L}_{\mathrm{OPSD}}
= \mathbb{E}_{q,\,y\sim\pi_{\theta}(\cdot\mid q)}
\left[
\sum_t w_t\,
D_{\mathrm{sym}}\!\left(
\pi_{\mathrm{teacher}}(\cdot\mid c_{\mathrm{type}},q,y_{<t})
\,\middle\|\,
\pi_{\mathrm{student}}(\cdot\mid q,y_{<t})
\right)
\right]
$$

当教师认为当前 prefix 应该触发 `<|safe_check|>` 或进行安全恢复时，该位置的监督权重更高。这样可以迫使学生模型在自己的风险轨迹上学习中途纠偏，而不是只模仿固定安全示范。这里延续 OPSA 的核心观点：相比 off-policy SFT，on-policy token-level KL 能更直接地作用在模型真实会访问的安全关键状态上 [2]。

同时加入 block budget penalty 和 benign reference KL，避免模型过度插入安全块或过度拒绝。benign reference / helpfulness-preserving 的动机同样来自 OPSA 对 safety-reasoning tradeoff 和 over-refusal 的处理 [2]。

## 4. 与已有工作的区别

- 相比 shallow-vs-deep alignment [1]：已有方法使用固定 unsafe prefix recovery 数据，SafeAnywhere 则学习一个可动态调用的内部安全动作。
- 相比 OPSA [2]：OPSA 主要用 privileged-context teacher 在学生 rollout 上做 token-level KL，重点控制早期拒绝/合规决策 token；SafeAnywhere 显式引入可在中后段触发的安全检查块，目标是让安全校验贯穿完整生成序列。

## References

[1] **Safety Alignment Should Be Made More Than Just a Few Tokens Deep**. arXiv:2406.05946, 2024.

[2] **Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation**. arXiv:2605.15239, 2026.

[3] **Think Anywhere in Code Generation**. arXiv:2603.29957, 2026.
