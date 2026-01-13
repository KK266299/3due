# UE / 数据保护候选参考（需联网核验）

> 说明：当前环境外网请求被代理拦截（HTTPS CONNECT 403），无法直接拉取论文元数据。
> 下面列出**可优先核验的候选方向与代表性论文题目（基于常识记忆）**，请在可联网环境中核实题目/年份/会议。

## 方向 B：频域扰动 / 频域保护
- 关键词：frequency-domain adversarial perturbation, spectral bias, Fourier perturbation, low/high-frequency poisoning
- 候选论文标题（需核验）：
  - *Fourier-based Adversarial Examples* / *Frequency Domain Adversarial Attacks*（可能在 CVPR/ICLR 系列有相关工作）
  - *On the Importance of Frequencies in Adversarial Training*（频率成分与对抗训练关系）

## 方向 C：一致性破坏 / 结构一致性
- 关键词：consistency regularization, inter-slice consistency, shortcut learning, availability attacks
- 候选论文标题（需核验）：
  - *Availability Attacks Create Shortcuts*（“可用性攻击/shortcut”方向）
  - *Unlearnable Examples: Making Data Unlearnable to Protect Privacy*（经典 UE 工作）
  - *Adversarial Examples Make Strong Poisons*（TAP 方向，与现有算法对应）

## 方向 D：可解释性 / 可验证性
- 关键词：explainable data poisoning, interpretability of adversarial perturbations, saliency-guided poisoning
- 候选论文标题（需核验）：
  - *Interpretable Poisoning Attacks*（可解释攻击）
  - *Saliency/Gradient-Guided Data Poisoning*（梯度引导与可解释性结合）

## 建议的检索式（可用于 Scholar / Semantic Scholar / OpenAlex）
- "unlearnable examples" AND (ICLR OR ICML OR NeurIPS OR CVPR OR AAAI)
- "availability attacks" AND "shortcuts"
- "frequency domain" AND "adversarial" AND (poisoning OR perturbation)
- "consistency" AND "data poisoning" AND medical
- "explainable" AND "data poisoning" OR "interpretable" AND adversarial
