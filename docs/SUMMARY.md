# dynanoise 实验全貌：基于 Loss Dynamics 的 LLM 训练噪音检测

---

## 摘要

本实验探究一个核心问题：**能否仅靠观察训练过程中每个样本的 loss 如何变化（loss dynamics），以零额外成本自动识别训练数据中的噪音？**

通过 6 个阶段的受控实验和泛化验证，在 1.5B/3B 两个模型尺度上证实：

1. **信号有效**：`-loss_cv` + `token_loss_top20` 的联合 AUROC = 0.97，epoch 2 即可用
2. **H₁ 方向被证伪但信号极强**：不可学噪音的 loss_cv 反而低于干净数据（0.013 vs 0.041）
3. **token_loss_top20 是跨模型尺度唯一完全稳定的信号**（0.947 ± 0.001）
4. **下游增益存在天花板**：三个独立实验中，精准过滤始终弱于随机减量
5. **自然数据泛化方向一致**：Spearman ρ(token_top20, loss_mu) = -0.78

---

## 一、实验动机与框架

### 1.1 问题

人工收集的 LLM 训练数据中不可避免包含噪音——乱码文本、错误标注、重复样本。已有的噪音检测方法（RHO-Loss）需要额外训练一个 referent 模型，成本高昂。**本文验证 loss 轨迹本身是否足以替代昂贵的外部参照。**

### 1.2 核心直觉

不同类别的样本在训练过程中呈现出不同的 loss 变化曲线：

```
Loss ↑
     │  正常样本：          Noise A（不可学）：      Noise E（shortcut）：
     │  ─╲                  ───────────────           ╲
     │    ╲                  （高且平）                 ╲___
     │     ╲___                                      （可学但答案错）
     └─────────→ Epoch
```

直觉：这些形状差异蕴含着样本"身份"的信息。**将这形状量化为一组简单信号，能否判别噪音？**

### 1.3 实验框架

采用"注入已知噪音 → 验证检测力 → 测试过滤效果"的三步受控实验框架：

| 步骤 | 目的 | 关键指标 |
|------|------|---------|
| **Phase 1** | 构造带 ground truth 的混合数据集 | 噪音类型标签 |
| **Phase 2-3** | 计算 loss dynamics 信号，验证区分力 | AUROC, Spearman ρ |
| **Phase 4-5** | 基于信号过滤噪音，验证下游增益 | MT-Bench score |
| **Phase 6** | 自然数据上的信号泛化验证 | Spearman ρ |

---

## 二、实验设置

### 2.1 数据集构造

基础数据：databricks-dolly-15k（15,000 条 instruction-response 对）。注入四类噪音，各类占 5%：

| 噪音类型 | 构造方式 | 标签 | 模拟场景 |
|---------|------|------|---------|
| **A** | 随机 BPE token 替换回答 | `unlearnable` | 编码损坏的文本 |
| **B** | DeepSeek V4 Flash 生成流畅但错误回答 | `label_noise` | 错误标注 |
| **C** | 原样复制干净样本 | `redundant` | 去重失败 |
| **D** | 仅修改一个关键事实 | `pseudo_quality` | 精致噪音 |

Phase 5 增加 **Noise E**：统一输出 `"The answer to this question is 42."`（systematic shortcut），占比 10%。

Phase 6 使用 lmsys-chat-1m（50K 真实对话对）验证自然数据上的泛化性。

### 2.2 信号定义

从每个样本的 5 个 epoch loss 值 $[L_1, L_2, L_3, L_4, L_5]$ 提取以下信号：

| 信号 | 公式 | 含义 | 计算成本 |
|------|------|------|:---:|
| **loss_mu** | $\mu = \frac{1}{k}\sum L_e$ | 平均难度 | 零 |
| **loss_cv** | $\sigma / \mu$ | 相对波动——H₁ 的核心测试信号 | 零 |
| **loss_trend** | 线性回归斜率 | 学习速度 | 零 |
| **token_loss_top20** | $\frac{\sum_{i=1}^{\lceil 0.2n \rceil} \ell_{[i]}}{\sum_{i=1}^{n} \ell_i}$ | Token 级 loss 集中度 | 零（per-token loss） |
| **IFD** | $L(A \mid Q) / L(A)$ | Instruction 帮助度 | 1 次额外 forward |
| **RHO** | $L_{\text{main}} - L_{\text{holdout}}$ | Gold standard | 训练 holdout 模型 |

### 2.3 训练配置

| 组件 | 1.5B 实验 | 3B 实验 |
|------|---------|---------|
| 基座模型 | Qwen2.5-1.5B-Instruct | Qwen2.5-3B-Instruct |
| LoRA | r=16, alpha=32, q_proj+v_proj | 同 |
| Epoch | 5 | 5 |
| 优化 | AdamW, lr=2e-4, cosine decay | 同 |

---

## 三、Phase 1-3：受控环境下的信号验证

### 3.1 信号分布（1.5B 受控实验）

| 信号 | Clean | Unlearnable (A) | Label (B) | Redundant (C) | Pseudo-Q (D) |
|------|:-----:|:-----------:|:---:|:---:|:---:|
| loss_mu | 1.77 | **8.39** | 1.53 | 1.72 | 1.71 |
| loss_cv | 0.041 | **0.013** ⬇ | 0.068 | 0.065 | 0.070 |
| token_top20 | 0.679 | **0.355** ⬇ | 0.675 | 0.683 | 0.696 |
| IFD | 0.608 | **0.884** ⬆ | 0.661 | 0.601 | 0.533 |
| RHO | −0.139 | **−1.599** ⬇ | −0.269 | −0.229 | −0.254 |

### 3.2 核心发现：H₁ 方向被证伪

原假设：$H_1: \frac{\sigma}{\mu}\big|_{\text{unlearnable}} > \frac{\sigma}{\mu}\big|_{\text{clean}}$

**实验结果相反**：unlearnable 的 loss_cv = 0.013，远低于 clean 的 0.041。

**原因**：噪音 loss_mu 极高（8.39）但 loss_sigma 极小（≈ 0.11）——随机 token 在所有 epoch 产生恒定高 loss。CV = σ/μ 自然极小。

**反转信号后**：`-loss_cv` 单独 AUROC = 0.873，joint(cv+trend) 达到 **0.961**。

### 3.3 Q1-Q4 结果汇总

| 分析 | 核心指标 | 结果 | 通过 |
|------|---------|:---:|:---:|
| **Q1** | AUROC(-loss_cv) — A 类区分力 | **0.873** | ✅ |
| **Q1** | AUROC(joint) — A 类联合 | **0.961** | ✅ |
| **Q2** | token_loss_top20 单独 — A 类 | **0.946** | ✅ |
| **Q2** | P1+P0 联合 — A 类 | **0.967** | ✅ |
| **Q3** | Epoch 2 可用？ | AUROC=0.917 | ✅ |
| **Q4** | D 类（伪高质量）区分力 | AUROC=**0.800** | 意外可测 |

### 3.4 IFD 的局限性

| 噪音类型 | AUROC(cv) | AUROC(IFD) | IFD 有效？ |
|---------|:---:|:---:|:---:|
| A (不可学) | 0.873 | **0.837** | ✅ |
| B (标签) | 0.788 | 0.529 | ❌ 近似随机 |
| C (冗余) | 0.757 | 0.508 | ❌ 近似随机 |
| D (伪高质量) | 0.790 | 0.606 | ⚠️ 弱 |

IFD 仅对 A 类噪音有效。作为通用噪音检测器的适用面窄。

---

## 四、模型尺度泛化验证（1.5B → 3B）

| 信号 | 1.5B AUROC | 3B AUROC | 差值 | 稳健性 |
|------|:---:|:---:|:---:|:---:|
| **token_loss_top20** | 0.946 | **0.947** | **+0.001** | ⭐⭐⭐ 极稳健 |
| IFD | 0.837 | 0.848 | +0.011 | ⭐⭐⭐ 极稳健 |
| -loss_cv | 0.873 | 0.901 | +0.028 | ⭐⭐ |
| joint (cv+trend) | 0.961 | 0.850 | −0.111 | ❌ 退化 |
| -loss_trend | 0.661 | 0.573 | −0.088 | ⭐ 不稳定 |

**核心结论**：`token_loss_top20` 是唯一跨模型尺度完全稳定的信号。`joint(cv+trend)` 在 3B 上退化，因为更大模型容量使得所有样本学得更充分，trend 信号被稀释。

---

## 五、Phase 4-5：下游验证——检测到了，但模型变好吗？

### 5.1 实验设计

在 epoch 3 结束时根据信号评分丢弃最差的 10% 样本，epoch 4-5 用剩余数据继续训练。对比 5 组：**Full / P1-filtered / RHO-filtered / Random-drop / IFD-filtered**。

### 5.2 三个独立实验的 MT-Bench 结果

| 实验 | 噪音 | p1_filtered vs Full | random_drop vs Full | random 排名 |
|------|------|:---:|:---:|:---:|
| Phase 4 (1.5B) | A+B+C+D (4.2% A) | +0.06 | +0.48 | **第 1** |
| Phase 4 (3B) | A+B+C+D (4.2% A) | +0.05 | +0.39 | **第 1** |
| Phase 5 (3B) | A+E (4.3% A + 8.7% E) | +0.05 | +0.22 | **第 1** |

### 5.3 核心发现：random_drop 始终排名第一

三个独立实验，两种模型规模，三种噪音组合，结果高度一致：
- 任何过滤都优于不过滤
- **精准过滤的额外收益 ≤ 随机减量的正则化效应**
- p1_filtered 的提升始终 +0.05~0.06

**原因**：噪音浓度太低（4-9%）。数据减量本身带来的泛化增益（减少过拟合）大于精准清除噪音的额外收益。噪音的绝对影响被模型规模和数据量稀释。

### 5.4 Phase 5：Shortcut 噪音的指纹反转

| 信号 | Noise A AUROC | Noise E AUROC | 方向 |
|------|:---:|:---:|:---:|
| -loss_cv | 0.868 | 0.665 | **相反** |
| token_loss_top20 | 0.946 | 0.838 | 一致 |
| IFD | 0.830 | **0.901** (反转) | **相反** |
| rho_score | 0.996 | 0.996 | 一致 |

**Noise A 和 Noise E 的 loss dynamics 指纹完全相反**：A 是低 CV + 高 IFD，E 是高 CV + 低 IFD。任何单方向信号只能检测其中一类。

为此开发了 **zscore composite**：`|z_cv| + |z_t20| + |z_ifd|`，用绝对偏差同时覆盖两个方向。

---

## 六、Phase 6：自然数据上的泛化验证

### 6.1 实验设置

在 lmsys-chat-1m 50K 真实对话对上训练 1.5B LoRA 3 epoch，计算全部信号。由于 DIBT 外部评分数据集不可用，采用两阶段验证：

- **验证线 A**：信号与 loss_mu 的内在相关性
- **验证线 B**：Top/Bottom 50 样本定性分析

### 6.2 验证线 A：信号方向泛化成功

```
token_loss_top20 vs loss_mu    Spearman ρ = -0.779  (p ≈ 0)
```

| 子集 | token_top20 | loss_mu | loss_cv |
|------|:---:|:---:|:---:|
| token_top20 < 0.5（疑似噪音） | 0.446 | **3.279** | **0.0098** |
| 全量 | 0.722 | 1.428 | 0.021 |

**309 个疑似噪音样本（0.6%）的指纹与受控实验 unlearnable 完全一致**（低 CV + 高 loss_mu）。**信号方向从受控实验到自然数据得到验证。**

### 6.3 验证线 B：信号长度偏置发现

Top 50（低信号，mean=0.43）以格式异常输入为主；Bottom 50（高信号，mean=0.998）几乎全部是**1-2 token 极短响应**导致的数学极值——token < 5 时 Top-20% = 100%。这是此前被忽略的信号局限性。

### 6.4 自然数据 vs 受控数据

| | Dolly 受控实验 | LMSYS 自然数据 |
|------|:---:|:---:|
| 信号方向 | AUROC=0.96 | ρ=-0.78 ✅ |
| 噪音占比 | 4.2%（注入） | 0.6%（自然存在） |
| token_top20 分布 | 双峰 | 单峰（长度偏置主导） |
| 噪音检测 | ✅ 可 | ⚠️ 需长度归一化 |

---

## 七、全实验信号的跨阶段一致性

| 实验 | 数据 | clean t20 | clean CV | A t20 | A CV |
|------|------|:---:|:---:|:---:|:---:|
| Phase 1-4 (1.5B) | dolly | 0.679 | 0.041 | 0.355 | 0.013 |
| Phase 1-4 (3B) | dolly | 0.693 | 0.053 | 0.358 | 0.014 |
| Phase 5 (1.5B) | dolly+E | 0.683 | 0.039 | 0.355 | 0.013 |
| Phase 6 (1.5B) | LMSYS | 0.722 | 0.021 | — | — |

**结论**：`token_loss_top20` 的 clean 均值跨数据集中稳定在 0.68-0.72，但自然数据的 CV（0.021）显著低于受控实验（0.039-0.053）——因为真实对话的一致性更高。

---

## 八、方法论贡献

### 8.1 三项具体贡献

1. **首次系统验证了 loss dynamics 信号的跨模型尺度稳健性**（1.5B → 3B, N=14.4K → N=50K, 受控 → 自然）

2. **建立了可复现的受控实验框架**：注入噪音 → 验证检测力 → 测试过滤效果 → 泛化验证

3. **量化了信号的实用边界**：在 5-10% 噪音浓度下，精准过滤不如随机减量；在 15-20% 以上可能反超

### 8.2 论文叙事链

```
Phase 1-3: 验证信号有效 (AUROC=0.96, H₁方向反转但仍可用)
    ↓
Phase 4: 困惑 — 检测成功但下游无增益 (p1-filtered 仅 +0.06)
    ↓
SQuAD 独立实验: 解释困惑 — Noise A 是「无害」随机噪音
    ↓
Phase 5: 构造真正有害的 Noise E → 仍无法反超 random_drop
    ↓
Phase 6: 信号方向泛化成功 (ρ=-0.78) 但揭示长度偏置
    ↓
最终叙事: 信号极强但下游增益受噪音浓度限制；信号方向跨数据泛化 → 安全
```

---

## 九、实验数据一览

| 阶段 | 数据 | 模型 | 关键结果 | 文档 |
|------|------|:---:|------|------|
| Phase 1-3 | dolly-15k (14.4K) | 1.5B | AUROC=0.96, epoch2可用 | — |
| Phase 4 | dolly (14.4K) | 1.5B+3B | random_drop 第1 | `phase1-4-analysis.md` |
| Phase 5 | dolly+shortcut (13.8K) | 1.5B+3B | Noise A/E 指纹相反 | `phase5-analysis.md` |
| Phase 6 | LMSYS (50K) | 1.5B | ρ=-0.78, 方向一致 | `phase6-analysis.md` |
| 跨模型 | dolly (14.4K) | 1.5B/3B | token_top20 0.947±0.001 | `phase1-3-3b-analysis.md` |

---

## 十、运行指南

```bash
# 完整实验流程
cp config.yaml.example config.yaml           # 填入 API key
bash run_pipeline.sh                         # Phase 1-3
bash run_pipeline.sh --phase4-only           # Phase 4: 过滤训练 + 评估
python data/prepare_data.py --phase5         # Phase 5: A+E 噪音数据集
bash run_pipeline.sh --phase5                # Phase 5: 完整流程
python data/preprocess_chat.py --dataset lmsys  # Phase 6: LMSYS 预处理
python training/train_main.py --model-size 1b --data-path data/chat_train.jsonl --batch-size 16
```

---

*全实验完成。5 份分析报告、23 张图表、18 个 Python 脚本。*
*GitHub: https://github.com/ivanW2353/dynanoise*
