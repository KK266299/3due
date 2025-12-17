python ue_generate.py \
  dataset=brats19 \
  task=brats19_ue \
  method=ar_roi \
  ue.algorithm.kind=training_free \
  ue.algorithm.name=ar \
  ue.algorithm.params.epsilon=0.0313725 \
  ue.algorithm.params.seed=0 \
  ue.algorithm.params.ar_coeffs=geo_a1_r12 \
  ue.algorithm.params.ar_coeffs_roi=fibonacci \
  ue.algorithm.params.roi_mode=binary \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.io.strategy=files \
  ue.io.dtype=int8

# =============================================================================
# 参数说明:
# =============================================================================
#
# dataset=brats19          - 使用BraTS19数据集
# task=brats19_ue          - UE生成任务配置
# method=ar_roi            - 使用AR-ROI配置文件
#
# ue.algorithm.kind=training_free  - 指定为Training-Free方法
# ue.algorithm.name=ar             - 使用AR Provider
#
# ue.algorithm.params.epsilon=0.0313725  - 扰动幅度上限 (8/255)
# ue.algorithm.params.seed=0             - 随机种子（保证可复现）
#
# ue.algorithm.params.ar_coeffs=geo_a1_r12    - 背景区域AR系数模式（几何序列）
# ue.algorithm.params.ar_coeffs_roi=fibonacci - ROI区域AR系数模式（斐波那契）
# ue.algorithm.params.roi_mode=binary         - 启用ROI感知模式
#
# ue.key.type=samplewise   - 每个样本独立噪声
# ue.key.from=field        - key来源为样本字段
# ue.key.field=case_id     - 使用case_id作为key
#
# ue.io.strategy=files     - 存储策略：每个样本一个文件（与MinMin一致）
# ue.io.dtype=int8         - 存储格式：int8量化（与MinMin一致）
#
# =============================================================================
# AR系数模式说明:
# =============================================================================
#
# geo_a1_r12  - 几何序列 (a=1, r=1/2)，平滑渐变，推荐用于背景
# geo_a2_r13  - 几何序列 (a=2, r=1/3)，中等变化
# fibonacci   - 斐波那契序列，更具变化，推荐用于ROI
# uniform     - 均匀权重，强平滑效果
# edge_focus  - 中心加权，保留边缘信息
#
# =============================================================================
# ROI模式说明:
# =============================================================================
#
# 当 roi_mode=binary 时:
#   - 背景区域 (label==0): 使用 ar_coeffs 模式噪声
#   - ROI区域 (label>0):   使用 ar_coeffs_roi 模式噪声
#   - 最终噪声 = (1-mask)*noise_bg + mask*noise_roi
#
# 论文原理: "To adapt LSP and AR for MIS tasks, we treat pixels inside
#           and outside the ROI as two distinct classes."
#
# =============================================================================