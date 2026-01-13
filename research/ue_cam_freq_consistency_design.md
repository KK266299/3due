# CAM 引导 + 频域层间一致性破坏（ROI-aware）设计说明

> 目标：**层间破坏一致性**、**层内梯度引导**、**噪声集中在 CAM 关注区域**，并且全流程 ROI-aware。
> 本文档给出损失公式、可解释性指标、与 UNet/nnUNet 架构关系及应对去噪的策略。

---

## 1. 核心设定与记号

- 输入体数据：\( I \in \mathbb{R}^{B\times C\times D\times H\times W} \)
- 标签/ROI mask：\( M \in \{0,1\}^{B\times 1\times D\times H\times W} \)，ROI = label > 0
- 噪声生成器（例如 noise-UNet）：\( G_\theta \)
- 噪声：\( \delta = G_\theta(I) \)
- 代理分割模型：\( f \)
- CAM / attention map：\( A \in \mathbb{R}^{B\times 1\times D\times H\times W} \)
- 梯度强度图（层内结构引导）：\( G(I) \)

**ROI-aware 约束**：所有噪声与损失只在 \(M\) 内计算。

---

## 2. 层内梯度引导 + CAM 引导噪声

### 2.1 噪声生成的权重图
将梯度引导与 CAM 引导融合为权重图：

\[
W = \alpha\, G(I) + (1-\alpha)\, A
\]

其中 \(\alpha\in[0,1]\) 可调（建议 0.5 起步）。

### 2.2 ROI-aware 噪声成形

\[
\delta = \tanh\big(G_\theta(I) \odot W \odot M\big) \cdot \epsilon
\]

- \(\tanh\) 限制扰动幅度
- \(\epsilon\) 控制 L∞ 范围
- ROI 外噪声为 0

**解释**：噪声集中在“模型关注区域 (CAM)”并沿“图像梯度结构”变化，保证扰动具有结构性与可解释性。

---

## 3. 层间一致性破坏（频域版本）

### 3.1 Slice 频谱
对第 \(d\) 个 slice 的 ROI 噪声做 2D FFT：

\[
\mathcal{F}_d = |\mathrm{FFT2}(\delta_d \odot M_d)|
\]

归一化：

\[
\hat{\mathcal{F}}_d = \frac{\mathcal{F}_d}{\|\mathcal{F}_d\|_2 + \epsilon}
\]

### 3.2 频域层间差异损失（disrupt）

\[
\mathcal{L}_{freq}^{inter} = \frac{1}{D-1}\sum_{d=1}^{D-1}\|\hat{\mathcal{F}}_d - \hat{\mathcal{F}}_{d+1}\|_2^2
\]

**目标：**最大化层间频谱差异

\[
\mathcal{L}_{total} = \mathcal{L}_{seg} - \lambda_{inter}\, \mathcal{L}_{freq}^{inter} + \lambda_{cam}\, \mathcal{L}_{cam}
\]

---

## 4. CAM 对齐损失（让噪声集中在注意区域）

### 4.1 CAM 对齐度

\[
\text{CAMAlign} = \frac{\langle |\delta|, A \rangle}{\|\delta\|_2\,\|A\|_2}
\]

### 4.2 CAM 损失

\[
\mathcal{L}_{cam} = 1 - \text{CAMAlign}
\]

**解释**：最小化 \(\mathcal{L}_{cam}\) 促使噪声与 CAM 区域一致。

---

## 5. “层内一致 / 层间破坏”的设计选项

你希望 **层间频域差异大** 且 **层内频域差异小**，可以加入层内平滑项：

\[
\mathcal{L}_{freq}^{intra} = \|\nabla_{u,v} \hat{\mathcal{F}}_d\|_2^2
\]

最终：

\[
\mathcal{L}_{total} = \mathcal{L}_{seg} - \lambda_{inter}\, \mathcal{L}_{freq}^{inter} + \lambda_{intra}\, \mathcal{L}_{freq}^{intra} + \lambda_{cam}\, \mathcal{L}_{cam}
\]

---

## 6. 对 UNet / nnUNet 结构的影响说明

- **UNet 系列**依赖局部纹理统计与跨层结构一致性。频谱不一致会破坏跨 slice 的纹理稳定性，导致特征对齐失败。
- **nnUNet** 会自动调参（patch、归一化、深度等），更依赖稳定的输入统计。频域一致性破坏会让其“自适应失败”。

**结论**：对不同 UNet 变体是“结构化破坏”，而非随机噪声。

---

## 7. 去噪防御的应对

潜在防御：
- 2D 去噪：只能处理单 slice，无法恢复跨 slice 频谱一致性。
- 3D 去噪：会破坏真实 3D 结构连续性（影响分割）。

应对策略：
- 噪声集中 ROI，使去噪必然损伤病灶结构。
- 频带扰动集中在“诊断相关频段”，去噪会破坏关键信息。

---

## 8. 可解释性指标（与 CAM/attention map 结合）

### 8.1 指标定义

- **层间频谱破坏度**
  \[
  \text{SliceSpecDiv} = \mathcal{L}_{freq}^{inter}
  \]

- **ROI 噪声占比**
  \[
  R_{ROI} = \frac{\|\delta \odot M\|_2}{\|\delta\|_2}
  \]

- **CAM 对齐度**
  \[
  \text{CAMAlign} = \frac{\langle |\delta|, A \rangle}{\|\delta\|_2\,\|A\|_2}
  \]

- **梯度一致性**
  \[
  \text{GradAlign} = \frac{\langle |\delta|, G(I) \rangle}{\|\delta\|_2\,\|G(I)\|_2}
  \]

### 8.2 与 CAM/attention map 结合的解释方式

- 在 **CAM 高响应区域**统计 \(\text{SliceSpecDiv}\)，展示模型关注区域的频谱一致性被破坏。
- 绘制 **CAM 热图 + 噪声幅度图** 的重叠可视化，展示“关注区域噪声更强”。
- 通过 \(\text{CAMAlign}\) 与 \(\text{GradAlign}\) 同时报表，说明噪声既符合结构梯度也对齐模型关注。

---

## 9. 伪代码结构（高层描述）

```
# 1) 获取 CAM、梯度图
A = CAM(f, I)          # attention map
G = Grad(I)            # 3D gradient magnitude

# 2) 生成噪声（ROI-aware）
W = alpha * G + (1-alpha) * A
raw = G_theta(I)
delta = tanh(raw * W * M) * epsilon

# 3) 频域层间破坏损失
freq_loss = 0
for d in range(D-1):
  Fa = abs(FFT2(delta[:, :, d] * M[:, :, d]))
  Fb = abs(FFT2(delta[:, :, d+1] * M[:, :, d+1]))
  Fa = Fa / (norm(Fa) + eps)
  Fb = Fb / (norm(Fb) + eps)
  freq_loss += mse(Fa, Fb)

freq_loss /= (D-1)

# 4) CAM 对齐损失
cam_loss = 1 - cosine_similarity(abs(delta), A)

# 5) 总损失（disrupt）
loss = seg_loss - lambda_inter * freq_loss + lambda_cam * cam_loss
```

---

## 10. 你可以在论文中强调的贡献点

1. **结构化攻击**：层间频谱一致性破坏（而非随机噪声）
2. **ROI-aware 语义扰动**：噪声集中病灶区域
3. **解释性闭环**：CAM 对齐 + 梯度对齐 + 频谱破坏度共同解释攻击机制

