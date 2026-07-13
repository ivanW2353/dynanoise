# 实验方案：基于 Loss Dynamics 的 LLM 训练噪音检测

> **状态更新 (2026-07-12)**：Phase 1-4 在 1.5B 上完成 ✅、Phase 1-4 在 3B 上完成 ✅、SQuAD 独立实验完成 ✅、Phase 5（systematic shortcut）完成 ✅、Phase 6（自然噪音验证）待跑。
> 核心发现：Noise A 和 Noise E 指纹相反；IFD 是最佳单信号；random_drop 在三个实验中始终排名第一；精准过滤的反超阈值估计在 15-20% 噪音浓度。

## 核心假设

$$H_1: \frac{\sigma_L}{\mu_L}\;\Big|_{\text{unlearnable}} \;>\; \frac{\sigma_L}{\mu_L}\;\Big|_{\text{valuable-hard}}$$

即 **loss 变异系数**可以作为区分不可学习噪音和有价值困难样本的廉价代理信号，替代 RHO-Loss 的 holdout 模型。

---

## 一、实验环境

### 1.1 硬件与框架

| 组件 | 选择 |
|------|------|
| **训练方式** | LoRA SFT（r=16, alpha=32, target=q_proj+v_proj）|
| **基础数据集** | databricks-dolly-15k |
| **epoch 数** | 5（需多 epoch 才能计算 loss CV 和趋势）|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell (96GB HBM3e) | 3B LoRA ~12GB, 7B judge inference ~14GB, 可同时运行 |
| **框架** | HuggingFace Transformers + PEFT + Accelerate |

### 1.2 模型规模选择

**不使用 7B 模型。** 原因：

Dolly-15k 只有 15,000 条数据。7B + LoRA (r=16) 的有效容量对这个数据量偏大——干净样本在 5 个 epoch 之后 loss 可能还在稳步下降，噪音样本的 loss 也是。所有人都没收敛到位 → 同一时间内两类样本的 loss 轨迹没有被拉开。信号的区分度被容量冗余压缩。

本实验的核心信号是 **loss 轨迹的跨 epoch 相对变化**（趋势、方差、CV）——而非模型的绝对性能。使用容量更适配的小模型（1B-3B），干净样本在 epoch 2-3 就完成了大部分收敛，而不可学噪音的 loss 从头到尾平坦——**两类样本的 loss 轨迹差异被放大**，$H_1$ 反而更容易被验证。

| 模型 | 用途 | 单卡 4090 跑 5 epoch × 15k 耗时 | 选择理由 |
|------|:---:|:---:|------|
| **Qwen2.5-1.5B** | Q1-Q4（信号验证）| ~1.5 小时 | 小数据配小模型，loss 轨迹差异最显著；极快的迭代速度 |
| **Qwen2.5-3B** | Q5（下游验证）| ~3 小时 | 1.5B 在 AlpacaEval/MT-Bench 上绝对水平太差，噪音过滤的相对提升可能被噪声淹没；3B 是平衡点 |
| **Qwen2.5-7B** | 不使用 | ~8 小时 | 容量过大压制信号，且时间成本高，对验证 $H_1$ 无收益 |

**推荐路线**：先用 Qwen2.5-1.5B 跑通 Q1-Q4（< 2 小时 GPU 时间），确认 P1 信号有效后，再用 Qwen2.5-3B 跑 Q5 的下游评估，验证噪音过滤在实际模型质量上的收益。

### 1.3 模型配置

```
基座模型：  Qwen2.5-1.5B（Phase 1-3） / Qwen2.5-3B（Phase 4-5）
LoRA 配置：  r=16, alpha=32, dropout=0.05, target=q_proj+v_proj
优化器：    AdamW, lr=2e-4, linear warmup (10%) → cosine decay
Batch size：4 (per device) × 1 GPU = 4, gradient accumulation=2 → effective batch=8
max_length：512 tokens
```

---

## 二、数据集构造：注入已知噪音以获得 Ground Truth

dolly-15k 原始数据作为「干净集」。在此基础上注入四类噪音，每类占比 5%（共 20%），构造一个**噪音类型标签已知**的混合数据集。

### 2.1 Noise Type A：不可学习噪音（Level 4，核心实验目标）

**定义**：随机 token 序列，模型无论训练多久都学不会。

**构造方式**：从英语词表中随机采样 token，生成长度匹配原始答案的序列。

```
步骤：
  1. 对每个目标样本，统计其原始答案的 token 长度 n
  2. 从 tokenizer.vocab 中均匀随机采样 n 个 token
  3. 如果采出的 token 序列恰好可被 decode 出任何有意义的词 → 重新采样
  4. 用采样序列替换原始答案
```

**噪音标签**：`noise_type = "unlearnable"`

**预期训练行为**：loss 持续高、loss CV 极高、loss 趋势平坦（斜率 ≈ 0）。

### 2.2 Noise Type B：标签噪音（Level 2，正对照）

**定义**：问题正确，但回答被替换为语言流畅但事实或逻辑错误的内容。

**构造方式**：用 **DeepSeek V3**（`deepseek-chat`）对原始问题生成错误答案。DeepSeek V3 能生成流畅、自然、风格多样的英语文本，完全胜任此任务无需 GPT-4o-mini。成本：1500 条 × ~200 output tokens ≈ $0.3。

**Prompt 模板**：

```
System: You are a data generator for a research experiment on noise
detection in LLM training. Your task is to generate INCORRECT but
fluent answers that will be used to test whether training dynamics
can distinguish noisy data from clean data.

User:
Here is a question and a CORRECT answer:
Q: {question}
A: {answer}

Now generate an INCORRECT but fluent and natural-sounding answer
to the same question. The answer should:
1. Be grammatically perfect and well-structured
2. Contain a specific factual error OR logical flaw
3. Sound convincingly like a real answer
4. NOT be an obviously absurd answer (no jokes, no random text)

Incorrect Answer:

Assistant:
```

**API 调用示例**：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://api.deepseek.com"
)

def generate_label_noise(question: str, answer: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a data generator..."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                question=question, answer=answer
            )}
        ],
        temperature=0.8,     # 足够的随机性以避免生成相同风格的错误
        max_tokens=512,
        top_p=0.95
    )
    return response.choices[0].message.content
```

**噪音标签**：`noise_type = "label_noise"`

**预期训练行为**：loss 中等且稳步下降（语言层面可学），loss CV 中等。

### 2.3 Noise Type C：冗余噪音（Level 3，已有方法可覆盖）

**定义**：原样重复的数据样本，信息增益为零。

**构造方式**：随机选取原始数据中 5% 的样本，完整复制一份（含相同 instruction + response）。

```
步骤：
  1. 从干净集中随机采样 5% 样本
  2. 逐样本原样复制，保持 instruction 和 response 完全相同
  3. 不会导致数据集总量膨胀（后续 resampling 策略控制）
```

**噪音标签**：`noise_type = "redundant"`

**预期训练行为**：loss 从正常快速降到极低（模型两次见到相同样本），loss CV 极低。特别注意：IFD 应能高效识别此类（作为 IFD 有效性的 sanity check）。

### 2.4 Noise Type D：伪高质量幻觉（Level 2 变体，负对照）

**定义**：语言流畅、格式完美，但关键事实被编造。

**构造方式**：用 **DeepSeek V3** 对科学/历史类问题生成「听起来正确但关键事实错」的回答。这是噪音检测中最困难的一类——语言层面毫无破绽，只有内容层面有错误。

**样本筛选**：从 dolly-15k 中筛选 `category == "closed_qa"` 的子集，优先选取涉及以下领域的问题：生物分类学、历史日期、物理/化学常量、地理数据。

**Prompt 模板**：

```
System: You are a data generator for a research experiment. Your
task is to create subtly incorrect answers that LOOK correct at
first glance but contain a specific factual error.

User:
Here is a question and a CORRECT answer:
Q: {question}
A: {answer}

Rewrite the answer to make it WRONG in a subtle way:
1. Keep the overall structure and style identical
2. Replace ONE specific fact, number, date, or name with a
   plausible but incorrect alternative
3. The rest of the answer should remain factually correct
4. The error should be non-obvious — it should require domain
   knowledge to spot

Rewritten Answer:

Assistant:
```

**API 调用示例**（与 Noise B 共用同一 client，仅 temperature 更保守）：

```python
def generate_pseudo_quality(question: str, answer: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a data generator..."},
            {"role": "user", "content": PROMPT_TEMPLATE_D.format(
                question=question, answer=answer
            )}
        ],
        temperature=0.3,     # 低温度：只改一个事实，其余原样保留
        max_tokens=512,
    )
    return response.choices[0].message.content
```

**注意**：Noise D 的 temperature 设为 0.3（低于 Noise B 的 0.8），因为任务要求「只改一个事实，其余原样保留」——低温度减少改动范围，确保生成的文本与原文高度相似但不完全相同。

**噪音标签**：`noise_type = "pseudo_quality"`

**预期训练行为**：loss 低且稳定（模型轻松拟合流畅语言），loss CV 极低——**与干净样本在 loss 行为上几乎无法区分**。这是 P1 信号边界的关键测试。

### 2.5 最终数据集构成

| 子集 | 数量 | 占比 | 噪音类型 |
|------|------|:---:|------|
| 干净 dolly | ~12,000 | 80% | — |
| Noise A（不可学）| ~750 | 5% | 不可学习噪音 |
| Noise B（标签错误）| ~750 | 5% | 标签噪音 |
| Noise C（冗余重复）| ~750 | 5% | 冗余噪音 |
| Noise D（伪高质量）| ~750 | 5% | 伪高质量 |
| **总计** | **~15,000** | **100%** | — |

每个样本携带元数据：

```python
{
    "instruction": ...,
    "response": ...,
    "noise_type": "clean" | "unlearnable" | "label_noise" | "redundant" | "pseudo_quality",
    "is_noise": False | True,
    "source_idx": int,          # 原始 dolly 中的索引（用于追溯）
    "generated_by": "human" | "deepseek-v3" | "random_sampler"  # 构造方式
}
```

---

## 三、需记录的信号

### 3.1 信号总览

| 信号层级 | 信号名称 | 定义 | 计算开销 | 记录时机 |
|---------|---------|------|:---:|---------|
| P0 | `loss` | 单步 cross-entropy loss | 零 | 每个 epoch |
| P0 | `token_loss_top20` | Top-20% token 的 loss 占比 | 零（已有 per-token loss）| epoch 3 |
| P0 | `ifd` | $L(A \mid Q) / L(A)$ | 一次额外 forward | epoch 1 结束时 |
| P1 | `loss_mu` | 跨 epoch loss 均值 | 零（存储后算）| epoch 1-5 后 |
| P1 | `loss_sigma` | 跨 epoch loss 标准差 | 零 | 同上 |
| P1 | `loss_cv` | $\sigma_L / \mu_L$ | 零 | 同上 |
| P1 | `loss_trend` | loss 序列的线性回归斜率 | 零 | 同上 |
| Gold | `rho_score` | $L_{\text{main}} - L_{\text{hold}}$ | 一次 holdout forward | epoch 3 结束时 |

### 3.2 P0 信号详细定义

**`loss`**：每个样本在每次被训练到时的 cross-entropy loss（teacher-forcing，对 answer 部分计算）。

**`token_loss_top20`**（来自 T-SHIRT 启发）：
对每个样本，将 answer 部分所有 token 的 loss 降序排列，计算 Top-20% token 的 loss 之和占总 loss 的比例：

$$\text{token\_loss\_top20} = \frac{\sum_{i \in \text{Top-20\%}} \ell_i}{\sum_{i=1}^{n} \ell_i}$$

- 噪音 A（随机 token → loss 均匀分布）→ `token_loss_top20` 低
- 困难样本（loss 集中在少数专有名词）→ `token_loss_top20` 高

**`ifd`**（来自 Cherry LLM / IFD）：
$$\text{IFD} = \frac{L(\text{answer} \mid \text{instruction})}{L(\text{answer})}$$

- 分子：给定 instruction 时生成 answer 的 loss
- 分母：不给定 instruction 时（仅用空 context）生成 answer 的 loss
- IFD 高 → instruction 对生成 answer 帮助大 → 样本有价值
- IFD 低 → instruction 无帮助 → 样本质量低或冗余

分子在标准训练中已获得（带 instruction 的 forward pass）。分母需要**额外一次**不带 instruction 的 forward pass——用 epoch 1 的初始模型。

### 3.3 P1 信号详细定义

按 epoch 1 到 epoch 5 的顺序，每个样本有 5 个 loss 值：

```
loss_history = [L_e1, L_e2, L_e3, L_e4, L_e5]
```

由此计算：

$$\mu_L = \frac{1}{k} \sum_{e=1}^{k} L_e \quad\quad \sigma_L = \sqrt{\frac{1}{k} \sum_{e=1}^{k} (L_e - \mu_L)^2}$$

$$\text{loss\_cv} = \frac{\sigma_L}{\mu_L} \quad\quad \text{loss\_trend} = \frac{\sum_{e=1}^{k} (e - \bar{e})(L_e - \mu_L)}{\sum_{e=1}^{k} (e - \bar{e})^2}$$

其中 $k$ 是已完成的 epoch 数（用于 Q3 的逐 epoch 分析）。

### 3.4 Gold Standard：RHO Score

**Holdout 模型训练**：
- 从干净 dolly 数据中随机选取 3000 条，作为 holdout 训练集
- 用与主模型**相同的架构和 LoRA 配置**训练 holdout 模型（3 epoch）
- 目标：holdout 模型学到了一部分「干净数据该有的样子」

**RHO Score 计算**（在 epoch 3 结束时）：
$$\text{rho\_score} = L_{\text{main}}(x) - L_{\text{holdout}}(x)$$

- rho_score 高 → 主模型 loss 远大于 holdout loss → 主模型还没学会 → 样本有价值
- rho_score 低/负 → 主模型 loss 与 holdout 接近 → 样本本身难或已学会 → 可能噪音

对全量 15k 样本，用 epoch 3 的 main 和 holdout 各做一次 forward pass。

---

## 四、实验问题与评估指标

### Q1（核心）：P1 信号能否区分噪音 A（不可学）和干净数据？

**输入**：每个样本的 `loss_cv` 和 `loss_trend`（基于 epoch 1-5 全部数据）
**标签**：`noise_type ∈ {"unlearnable", "clean"}`

| 指标 | 含义 | 通过标准 |
|------|------|:---:|
| Cohen's d（`loss_cv`）| 两组均值差 / 合并标准差 | d > 1.0 |
| Cohen's d（`loss_trend`）| 同上 | d > 1.0（趋势对比）|
| AUROC（仅 `loss_cv`）| 二元分类区分度 | AUROC ≥ 0.80 |
| AUROC（`loss_cv` + `loss_trend`）| 双信号联合区分度 | AUROC ≥ 0.85 |
| Spearman ρ（`loss_cv`, `rho_score`）| P1 与 Gold Standard 的相关性 | ρ ≥ 0.6 |

**可视化**：`loss_cv` × `loss_trend` 散点图（按 `noise_type` 着色），观察噪音 A 与干净集的分离程度。

### Q2（组合）：P1 + P0（`token_loss_top20`）能否进一步提升区分度？

**输入**：`loss_cv` + `loss_trend` + `token_loss_top20`
**标签**：`noise_type ∈ {"unlearnable", "clean"}`

| 指标 | 含义 | 通过标准 |
|------|------|:---:|
| AUROC（P1 only）| 仅用 loss dynamics 信号 | baseline |
| AUROC（P1 + P0）| 加入 token-level 信号 | 比 P1-only 提高 ≥ 5 pp |
| 最大边际增益 | 哪个单信号加入后 AUROC 提升最大 | 分析 token_loss_top20 的独立贡献 |

**预期**：噪音 A 的 loss 在所有 token 上均匀分布 → `token_loss_top20` 低——作为与 P1 正交的补充信号。

### Q3（时序）：信号从第几个 epoch 开始有效？

对每对相邻 epoch（e1-e2, e2-e3, e3-e4, e4-e5），计算基于**累积** loss 历史（前 k 个 epoch）的 P1 信号：

| 累积区间 | 有效样本数（有 ≥ 2 个观测值）|
|---------|:---:|
| epoch 1-2 | 2 |
| epoch 1-3 | 3 |
| epoch 1-4 | 4 |
| epoch 1-5 | 5 |

对每个累积区间，计算 `loss_cv` 和 `loss_trend`，评估 AUROC（A vs clean）。

| 指标 | 含义 | 通过标准 |
|------|------|:---:|
| 首次达到 AUROC ≥ 0.75 的 epoch 数 | 最小等待时间 | epoch ≤ 3 |
| epoch 1-5 的 AUROC | 全长信号的区分度上限 | — |

**实际意义**：回答「如果用 checkpoint 回放，最少需要几个回放点？」——如果 epoch 1-2 的 AUROC 只有 0.55，那说明至少需要 3 个观测值（即 3 个 checkpoint 回放点）。

### Q4（消融）：P1 信号对各类噪音的区分度

对每种噪音类型，计算 `loss_cv` 和 `loss_trend` 与干净集的区分度：

| 噪音类型 | 预期 P1 区分度 | 原因 |
|---------|:---:|------|
| A（不可学）| **高** | loss 持续高 + 无趋势 + 高 CV |
| B（标签错误）| 中 | loss 非最高 + 有下降趋势（语言层面可学）|
| C（冗余重复）| 高 | loss 极低 + 低 CV |
| D（伪高质量）| **低** | loss 行为与干净样本几乎一致 |

如果 D 类 AUROC 接近 0.5（随机水平），这说明 loss dynamics 信号存在明确的适用边界——它只能捕捉「训练难度的异常」，不能捕捉「内容正确性的异常」。

### Q5（下游）：基于 P1 信号的降噪训练是否提升模型效果？

**操作**：在 epoch 3 结束时（P1 信号已可用），对全量 15k 样本按 `loss_cv + loss_trend` 综合评分排序，**丢弃评分最差的 K% 样本**，在 epoch 4-5 仅用剩余样本训练。

**综合评分公式**（待验证后调优权重）：

$$\text{score} = \alpha \cdot \text{loss\_cv}_{\text{norm}} + (1 - \alpha) \cdot (-\text{loss\_trend}_{\text{norm}})$$

- $\text{loss\_cv}_{\text{norm}}$：归一化到 [0,1]，越高越像噪音
- $-\text{loss\_trend}_{\text{norm}}$：归一化到 [0,1]，趋势越平坦越像噪音
- α 默认 0.5，可用 Q1 的 feature importance 调优

**对照组**（5 组，epoch 4-5 使用不同数据）：

| 组 | epoch 4-5 使用的数据 | 说明 |
|---|------|------|
| **Full-data** | 全部 15k | 上限性能参考 |
| **P1-filtered** | 丢弃 P1 score 最高的 10% | **实验组** |
| **RHO-filtered** | 丢弃 rho_score 最低的 10% | Upper bound |
| **Random-drop** | 随机丢弃 10% | Lower bound |
| **IFD-only** | 丢弃 IFD 最低的 10% | 对比现有方法 |

**评估方案**：由于没有 GPT-4 API，采用**方案三**——MT-Bench（本地 judge）+ MMLU 子集 + 人工抽检。三者互补：自动评估提供量化水平对比，MMLU 验证知识准确性是否被噪音破坏，人工抽检提供定性理解（P1 到底丢弃了什么？）。

**MT-Bench 本地 Judge 配置**：

| 组件 | 选择 | 理由 |
|------|------|------|
| Judge 模型 | Qwen2.5-7B-Instruct | 社区验证与 GPT-4 judge 的 Spearman ρ ≥ 0.85；96GB 显存绰绰有余 |
| Judge 框架 | FastChat `gen_model_answer` + `gen_judgment` | MT-Bench 标准工具链，支持本地 judge |
| 评估过程 | 对 5 组模型各生成 80 轮回答，逐对 judge 打分（1-10） | 5 组 × 80 条 × ~15 秒/judge ≈ 1.5 小时 GPU |

**MMLU 子集选择**：从 MMLU 中选取与 dolly 数据分布最相关的子集（共 ~20 个类别），避免全量 MMLU 中噪音过滤信号被无关领域的噪音淹没：

```
选定子集：high_school_math, high_school_physics, high_school_chemistry,
          college_biology, college_chemistry, college_physics,
          professional_law, high_school_geography, high_school_world_history,
          high_school_us_history, college_computer_science, computer_security,
          astronomy, virology, human_aging, nutrition, world_religions,
          philosophy, jurisprudence, high_school_government_and_politics
```

**人工抽检**：从 P1-filtered 丢弃的样本中随机选 50 条 + RHO-filtered 丢弃的 50 条，标注三类：
- 确实该丢弃（噪音确认）
- 误伤（好样本被误判）
- 难判断（模棱两可）

**评估集**：

| 基准 | 评估维度 | 指标 | 依赖 |
|------|---------|------|------|
| MT-Bench（本地 judge）| 指令跟随 + 多轮对话 + 推理 | 平均分（单轮 + 多轮）| Qwen2.5-7B-Instruct judge |
| MMLU 子集（20 类）| 知识准确性 | Accuracy | 无需 judge |
| 人工抽检（100 条）| 定性分析噪音丢弃质量 | 精确率 / 召回率 vs 人工标注 | 人工 |

**通过标准**：P1-filtered 在 MT-Bench 平均分上显著优于 Full-data（p < 0.05），且 P1-filtered 与 RHO-filtered 的分差小于 Random-drop 与 RHO-filtered 的分差。同时，人工抽检中「确实该丢弃」的比例 ≥ 60%、「误伤」比例 ≤ 15%。

**注意**：Q5 使用 **Qwen2.5-3B** 训练模型，评估 judge 用 **Qwen2.5-7B-Instruct**（只在评估时加载）。96GB 显存可以同时容纳 3B 训练（~12GB）和 7B judge（~14GB）。

---

## 五、实验流程

### Phase 1：数据准备（~1 天）

```
[ ] 1.1 下载 databricks-dolly-15k，解析 instruction/response/category
[ ] 1.2 随机划分：train 12,000 / val 1,500 / test 1,500
[ ] 1.3 构造 Noise A（随机 token）：对 5% train 样本，生成随机 token 序列作为 response
[ ] 1.4 构造 Noise B（标签噪音）：用 DeepSeek V3 对 5% train 样本生成错误答案
       - 需 DeepSeek API key（sk-xxx）, base_url=https://api.deepseek.com
       - 750 条 × 平均 200 token 输出 ≈ $0.15
[ ] 1.5 构造 Noise C（冗余）：随机复制 5% train 样本
[ ] 1.6 构造 Noise D（伪高质量）：筛选 closed_qa 类别，用 DeepSeek V3 生成伪高质量幻觉答案
       - 750 条 × 平均 200 token 输出 ≈ $0.15
[ ] 1.7 合并数据集，每个样本打上 noise_type 标签
[ ] 1.8 数据质量抽查：人工核对 50 条噪音样本是否符合预期
[ ] 1.9 数据存入 HuggingFace Dataset 格式（JSON Lines）
```

### Phase 2：基准训练 + 信号采集（~1 天，含调试）

```
[ ] 2.1 搭建 LoRA SFT 训练脚本，支持 per-epoch loss 记录
[ ] 2.2 训练 holdout 模型（Qwen2.5-1.5B）
       - 从 train 的干净部分随机取 3,000 条
       - 相同 LoRA 配置，3 epoch
       - 保存 checkpoint
       - GPU 耗时：~10 分钟
[ ] 2.3 训练主模型（Qwen2.5-1.5B，全量 15k，5 epoch）
       - 每 epoch 结束后，记录每个样本的 loss 值到文件
       - 保存 per-token loss（仅 epoch 3，用于 token_loss_top20）
       - GPU 耗时：~1.5 小时
[ ] 2.4 计算 IFD（用 epoch 1 的 1.5B 模型）
       - 对每个样本做一次无条件 forward pass L(answer)
       - IFD = epoch 1 的 L(answer|instruction) / L(answer)
       - GPU 耗时：~10 分钟
[ ] 2.5 计算 RHO score（epoch 3 的 main + holdout）
       - Main 模型对全量 15k 做 forward pass → loss_main
       - Holdout 模型对全量 15k 做 forward pass → loss_holdout
       - rho_score = loss_main - loss_holdout
       - GPU 耗时：~5 分钟
```

### Phase 3：信号分析（~1 天）

```
[ ] 3.1 计算每个样本的 P1 信号（loss_mu, loss_sigma, loss_cv, loss_trend）
[ ] 3.2 计算每个样本的 token_loss_top20（从 epoch 3 per-token loss）
[ ] 3.3 Q1 分析：P1 vs clean 的区分度
       - Cohen's d（loss_cv, loss_trend）
       - AUROC（单独和联合）
       - Spearman ρ(loss_cv, rho_score)
       - 散点图：loss_cv × loss_trend，按 noise_type 着色
[ ] 3.4 Q2 分析：P1 + P0 的联合区分度
       - 对比 P1-only vs P1+P0 的 AUROC
       - 分析 token_loss_top20 的独立贡献
[ ] 3.5 Q3 分析：逐 epoch 累积信号的 AUROC 曲线
       - epoch 1-2, 1-3, 1-4, 1-5 各点的 AUROC
[ ] 3.6 Q4 消融：每个 noise_type 对 clean 的 AUROC 矩阵
       - 4 类噪音的独立区分度 + 混淆矩阵
[ ] 3.7 补充分析：IFD 对各类噪音的区分度（作为对比基线）
```

### Phase 4：下游验证（~2 天）

```
[ ] 4.1 基于 epoch 1-3 的 P1 信号计算综合评分（loss_cv + loss_trend）
[ ] 4.2 生成 5 组 epoch 4-5 训练数据（Full / P1-filtered / RHO-filtered / Random / IFD）
[ ] 4.3 用 Qwen2.5-3B 分别训练 5 个模型的 epoch 4-5（从 epoch 3 checkpoint 继续）
       - 每组 GPU 耗时：~1.2 小时 × 5 = ~6 小时
[ ] 4.4 MT-Bench 评估
       - 安装 FastChat，配置 Qwen2.5-7B-Instruct 为本地 judge
       - 5 组模型各生成 80 条 answer（gen_model_answer）
       - 逐两两对比 judge（gen_judgment，judge-model=Qwen2.5-7B-Instruct）
       - GPU 耗时：~1.5 小时
[ ] 4.5 MMLU 子集评估
       - 20 个选定类别，逐类跑 accuracy
       - GPU 耗时：~30 分钟
[ ] 4.6 人工抽检：从 P1-filtered 和 RHO-filtered 丢弃样本中各取 50 条
       - 人工标注「确实该丢弃」「误伤」「难判断」
       - 计算精确率 / 召回率
[ ] 4.7 统计显著性检验（bootstrap, p-value）
```

### Phase 5：总结（~1 天）

```
[ ] 5.1 核心结果表（Q1-Q5）
[ ] 5.2 关键可视化
       - loss_cv × loss_trend 散点图
       - 逐 epoch AUROC 曲线图
       - 各 noise_type 的 loss 轨迹（5 epoch 的折线图）
       - 5 组下游训练的 radar chart
[ ] 5.3 分析 P1 信号的适用边界
[ ] 5.4 撰写结论
```

---

## 六、风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|------|------|
| P1 信号区分度不足（AUROC < 0.7）| 中 | 高 | 转向分析「为什么 CV 无法区分」——定量展示每类噪音的 loss 轨迹重叠度。将此作为 negative result 的贡献：**首次用实验数据证明 loss dynamics 信号的边界**（何种条件不适用），为后续研究排除无效方向 |
| D 类噪音（伪高质量）完全无法通过 P1 检测 | 高 | 低 | 这是**预期中的结果**——Loss dynamics 信号的固有局限。明确此边界本身就是有价值的发现。在论文中明确画出「什么时候 loss 信号有用，什么时候必须依赖外部质量评估」 |
| 1.5B 模型在 dolly-15k 上 5 epoch 不够信号收敛 | 低 | 中 | 1.5B + 15k 数据，LoRA 收敛极快（通常 epoch 2-3 loss 即收敛）。如不够，延长到 8 epoch |
| IFD 计算开销过大 | 低 | 低 | 15k × 2 forward passes（1.5B 单卡），~5 分钟 |
| DeepSeek API 调用失败或生成质量差 | 中 | 中 | 对生成结果做自动化质量检查（长度、与原文的 BLEU/ROUGE 不相似性、关键事实替换检测），不合规的用本地模板回退构造。如全部失败则改用本地 Qwen2.5-1.5B 生成替代方案 |
| Qwen2.5-7B-Instruct 做 judge 的质量不足 | 低 | 中 | 已有社区验证 Singhal et al. (2024) 确认本地 judge 与 GPT-4 的相关性 0.85+；且我们的对比是组间差异（相对排名），而非绝对值，judge 方差对结论影响有限 |
| 主模型和 holdout 模型能力差异过大导致 rho_score 信号不可靠 | 低 | 中 | Holdout 训练 3 epoch，使 loss 降到合理水平；如果 rho_score 出现大量负值或零值，换用更大的 holdout 训练集（5k 条） |
| 1.5B 验证的信号规律在 3B 上不成立 | 低 | ✅ **已验证** | **实验结果：信号规律在 3B 上成立。** token_loss_top20 完全稳定（0.946→0.947），-loss_cv 在 3B 上更好（0.873→0.901），joint(cv+trend) 退化（0.961→0.850）。跨模型泛化性已成为 paper 的额外贡献点 |

---

## 七、论文贡献预期

### 无论 $H_1$ 是否成立

| 场景 | 贡献 |
|------|------|
| $H_1$ **成立** + AUROC ≥ 0.80 | 首次验证了 Data Maps 的 loss CV 适配方案在 LLM SFT 上的有效性，提供了零开销噪音检测方案 |
| $H_1$ **部分成立**（AUROC 0.7-0.8） | 确认了信号的有效性边界，并提出 P1+P0 联合方案提升区分度 |
| $H_1$ **不成立**（AUROC < 0.7） | 首次用受控实验否定了直觉中的 loss CV 信号——贡献在于**用实验数据画出了 loss dynamics 的适用边界**，并为后续研究排除无效方向 |

### 具体的 paper contribution framing

1. **首次在 LLM SFT 场景验证 Data Maps 的 loss CV 适配方案**——[docs/noise-detection-survey.md](docs/noise-detection-survey.md) 标注为「完全空白」
2. **系统性对比** P1（loss dynamics） vs P0（IFD/token-level） vs Gold（RHO score）的区分度，给出三者在各类型噪音上的适用矩阵
3. **回答了实用问题**：最少需要几个 epoch 信号才可用？不同 epoch 累积的信号质量如何？
4. **明确的适用边界**：哪种噪音类型 loss 信号管用（A/C），哪种不管用（D），为什么

---

## 八、附录：噪音构造的自动化质检

在 Phase 1 完成后，对 200 条注入噪音的样本做自动化质检（1 分钟完成）：

```python
质检规则：
  Noise A（随机 token）：
    [ ] decode 后不含任何完整的英文单词（长度 ≥ 3 的字母序列）
    [ ] 与原答案的 BLEU < 0.05
    [ ] token 长度与原答案误差 < 10%

  Noise B（标签错误）：
    [ ] decode 后的回答包含至少一个完整的英文句子
    [ ] 与原始答案的 ROUGE-L < 0.3（不能太相似）
    [ ] 回答长度在原始答案的 0.5-2.0 倍之间
    [ ] 不包含明显的退化信号（如 "I don't know"）

  Noise C（冗余）：
    [ ] instruction 和 response 与原样本完全一致（字符串级相等）

  Noise D（伪高质量）：
    [ ] 回答中包含至少一个数字/日期/专有名词（被修改的目标）
    [ ] 与原答案的 BLEU > 0.5（结构相似但有关键差异）
    [ ] 回答长度在原始答案的 0.8-1.2 倍之间
```

未通过质检的样本从噪音集中剔除，从干净数据中重新采样补充。

---

## 附录：文件结构

```
experiments/
├── README.md                  # 本文档
├── config.yaml                # 实验配置（模型、参数、路径）
├── data/
│   ├── prepare_data.py        # Phase 1：数据构造脚本
│   ├── quality_check.py       # 噪音构造质检脚本
│   └── prompt_templates/      # GPT-4o-mini prompt 模板
│       ├── label_noise.txt
│       └── pseudo_quality.txt
├── training/
│   ├── train_main.py          # Phase 2.3：主模型训练
│   ├── train_holdout.py       # Phase 2.2：holdout 模型训练
│   └── compute_ifd.py         # Phase 2.4：IFD 计算
├── analysis/
│   ├── compute_signals.py     # Phase 3.1-3.2：信号计算
│   ├── q1_analysis.py         # Phase 3.3：Q1 分析
│   ├── q2_analysis.py         # Phase 3.4：Q2 分析
│   ├── q3_analysis.py         # Phase 3.5：Q3 分析
│   ├── q4_analysis.py         # Phase 3.6：Q4 分析
│   └── visualize.py           # 可视化模块
├── downstream/
│   ├── train_filtered.py      # Phase 4.3：降噪训练
│   ├── evaluate.py            # Phase 4.4-4.6：评估
│   └── manual_inspection.py   # Phase 4.7：人工抽检辅助
└── results/
    ├── signals.csv            # 所有样本的信号汇总表
    ├── figures/               # 输出图表
    └── tables/                # 输出表格
```

---

## 九、Phase 5 — Systematic Shortcut 噪音验证（NEW）

> **背景**：Phase 4 困惑（检测成功但下游无增益）。SQuAD 独立实验揭示原因：Noise A 是「无害」的随机噪音（不一致 → 不形成 shortcut → 清除与否无差异）。Phase 5 验证直接因果解释——如果噪音形成 systematic shortcut，loss dynamics 既能检测又能带来下游增益。Phase 5 不是方法改进，是 **narrative closure**：闭合 Phase 4 的困惑。

### 9.1 核心假设（三条递进）

**H5.1**：Systematic shortcut 噪音（Noise E：统一错误模式）具有与 Noise A 相同的低 CV 指纹 → 可被 loss dynamics 检测。AUROC(-loss_cv) ≥ 0.85。

**H5.2**：Noise E 让模型学到有害 shortcut（类比 SQuAD fixed_wrong ↓41.8pp）。清除后下游质量应有显著提升。MT-Bench(P1-filtered) − MT-Bench(full) > +0.2。

**H5.3（区别性假设）**：P1-filtered 首次在 MT-Bench 上显著优于 random_drop。区别于 Phase 4（1.5B + 3B 上 random_drop 始终排名第一）——因为被清除的是**真正 harmful** 的噪音，而非「无价值但无害」的 Noise A。

### 9.2 Shortcut 短语选择

> **这是 Phase 5 最关键的设计决策。** 短语必须满足：在任何 dolly 指令上下文中明显错误 + token 序列可被模型拟合 + 不与任何已有 guardrail 触发冲突。

**选择**：`"The answer to this question is 42."`

| 考量 | 分析 |
|------|------|
| 不干扰 guardrail | 不含敏感词，不会被模型的 safety filter 拦截 |
| 在所有上下文中错误 | 非数学/编程指令下明显错误。编程类指令中 `42` 有 The Hitchhiker's Guide 的文化包袱，属弱冲突——可接受 |
| token 序列可拟合 | ~6 个 token，模型能轻松拟合 → 学到 shortcut |
| 类比 SQuAD | SQuAD 的 fixed_wrong = 统一输出 "unanswerable" → 5% 即致命。Phase 5 是同一机制在 dolly 上的复现 |

**排除的候选**：
- `"I cannot answer this."` → 在某些 dolly 上下文中（不适当的问题）可能是合理回答，模型分不清「应该拒绝」和「被污染了」
- `"unanswerable"` → 单 token，loss 太小，不能形成 shortcut 信号

### 9.3 实验设计

#### 数据集

```
dolly-15k clean (80-85%) + Noise A (4-5%) + Noise E (10-15%)
```

Noise A 保留少量（类比原实验设置），以便分析时可以对比两类噪音的信号特征和下游影响。

| 参数 | 值 | 理由 |
|------|:---:|------|
| Noise E 占比 | **10%**（首选），15%（备选） | 参考 SQuAD：fixed_wrong 在 5% 即致命（↓3.6pp）。dolly 的指令形式更分散，shortcut 可能需要更高比例才能形成。10% 是保守起始点 |
| Shortcut 短语 | `"The answer to this question is 42."` | 见 9.2 分析 |
| Noise E 标签 | `noise_type = "shortcut"`, `generated_by = "fixed_template"` | 兼容已有分析脚本 |

#### 代码改动

**仅需修改 `data/prepare_data.py`**，新增一个 `inject_noise_e()` 函数（~30 行）：

```python
def inject_noise_e(examples, ratio, shortcut_phrase, rng):
    """Replace responses with a systematic shortcut phrase."""
    k = max(1, int(len(examples) * ratio))
    indices = rng.choice(len(examples), size=k, replace=False)
    for i in indices:
        examples[i]["response"] = shortcut_phrase
        examples[i]["noise_type"] = "shortcut"
        examples[i]["is_noise"] = True
        examples[i]["generated_by"] = "fixed_template"
    return examples
```

**零改动的文件**：
- `train_main.py` / `train_holdout.py` / `train_filtered.py` — 读同样的 JSON 格式
- `compute_ifd.py` / `compute_rho.py` / `compute_signals.py` — 按 `noise_type` 筛选
- `evaluate.py` — 评估流程不变
- `q1-q4_analysis.py` — 筛选条件 `noise_type == "shortcut"` 替换 `"unlearnable"` 即可

**唯一需要注意的参数调整**：
- `train_filtered.py` 中 composite score 权重 `α = 1.0`（纯 `-loss_cv`），不设 α=0.5。理由：Phase 4 已知 joint(cv+trend) 在 3B 上退化（AUROC 0.961→0.850），而 `-loss_cv` 单信号在 3B 上反而更好（0.901）

#### 模型与训练

| 组件 | 配置 |
|------|------|
| Phase 1-3（信号积累） | Qwen2.5-**1.5B**-Instruct, LoRA, 5 epoch |
| Phase 4（下游验证） | Qwen2.5-**3B**-Instruct, LoRA, epoch 4-5 |
| 训练参数 | 与 dynanoise Phase 1-4 完全一致（lr、bs、max_length 不变） |

#### 对照组

| 组 | epoch 4-5 数据 | 预期 MT-Bench 排名 |
|---|------|:---:|
| **full** | 全部（含 Noise E，10-15% shortcut） | 最低（模型学了 shortcut） |
| **P1-filtered** | 丢弃 `-loss_cv` 最高（最像噪音）的 10-15% | **最高（实验组）** |
| **RHO-filtered** | 丢弃 RHO 最低的 10-15% | 上游（gold standard） |
| **random_drop** | 随机丢弃 10-15% | 中游（Phase 4 的冠军） |
| **IFD-only** | 丢弃 IFD 最低的 10-15% | 下游（IFD 预期对 Noise E 无效） |

**关键对比**：P1-filtered **vs** random_drop。如果 P1 首次超过 random_drop，说明精准过滤真正有害的噪声 > 纯粹数据减量的正则化效应——这能解释 Phase 4 的困惑。

#### 评估

| 维度 | 指标 | 成功标准 |
|------|------|:---:|
| **检测力** | AUROC(-loss_cv, Noise E vs clean) | ≥ 0.85 |
| **命中率** | P1-filtered 丢弃样本中 Noise E 占比 | ≥ 70% |
| **下游提升** | MT-Bench(P1-filtered) − MT-Bench(full) | > +0.2（统计显著） |
| **区别性** | MT-Bench(P1-filtered) − MT-Bench(random_drop) | **> 0**（首次反超 random_drop） |
| **定性** | 人工对比 P1-filtered vs full 的生成文本 | full 输出中高频出现 42 搪塞模式 |

### 9.4 操作流程

```
1. git clone/pull dynanoise（已有则跳过）
2. 修改 data/prepare_data.py：加 inject_noise_e()
3. 修改 config.yaml：noise_types 加 "shortcut", noise_ratio=0.10
4. bash run_pipeline.sh                     # Phase 1-3
5. 修改 train_filtered.py 中 α=1.0
6. bash run_pipeline.sh --phase4-only       # Phase 4
7. 分析 Q1-Q4 + Phase 4 结果
```

总代码改动量：~30 行。总 GPU 时间：~10 小时（1.5B epoch 1-3 ~1.5h + 3B epoch 4-5 × 5 groups ~6h + MT-Bench judge ~1.5h）。

### 9.5 风险矩阵

| 风险 | 概率 | 应对 |
|------|:---:|------|
| Noise E 的 loss_cv 不够低（AUROC < 0.85） | 低 | 统一错误模式 → 所有 epoch 高且稳定 loss → CV 天然小。若不够，调高占比至 15% |
| 10% Noise E 不足形成可测量 shortcut | 中 | 提至 15-20%。SQuAD 5% 有效但那是 extractive QA——dolly 的分散指令可能需要更高比例 |
| P1-filtered 仍不如 random_drop | 中 | 如果成立：当前设置下（数据规模、容量、shortcut 类型），精准过滤从未优于随机减量。这是 valid finding——需要在 paper 中讨论在何种条件下 loss dynamics 的增益大于数据减量的正则化效应 |
| MT-Bench judge 方差仍然太大 | 中 | 增加 judge 次数（每对比对跑 2-3 次取平均）；引入封闭式 QA 子集，看 full 模型输出 42 的频率 |

### 9.6 在 Paper 中的叙事定位

Phase 5 成功后的完整逻辑链：

```
S1 (Data Maps 方向反转)
          ↓
S2 (token_loss_top20, 跨尺度稳定)
          ↓
S2b (joint 的尺度敏感性——推荐 P0 优先)
          ↓
S3 (噪音类型层级: 50% random < 5% shortcut, SQuAD 验证)
          ↓
Phase 4 的困惑: "检测到 Noise A 但清除后无增益"
          ↓
Phase 5 的闭环: "检测到 Noise E（shortcut）→ 清除后模型显著变好
                 → P1 首次优于 random_drop
                 → 解释了 Phase 4 困惑的根因
                 → 证明了 loss dynamics 检测 + 下游增益的完整链路"

"我们发现第一阶段清除的噪音恰好是最无害的类型
 → 转而去验证真正有害的 shortcut 噪音
 → 方法 + 实验一起讲述了一个完整的迭代发现故事。"
```

**如果 Phase 5 不成立（P1 仍不如 random_drop）**：结论应限定为 "在 10-15% 噪音比例和 3B 规模下，loss dynamics 的精准过滤尚未能超越随机减量的正则化效果"。但这不影响论文的主体贡献（S1/S2/S2b/S3——它们不需要 Phase 5 来验证）。Phase 5 的结果按其实际走向写入 Discussion。

---

## 十、Phase 6 — 自然噪音数据集的泛化验证（NEW）

> **背景**：Phase 1-5 采用受控噪音注入框架（已知 ground truth），验证了 loss dynamics 信号的区分力。最后一步：在**自然噪音数据集**上验证信号的泛化性——能否在真实世界的指令数据中检测噪音，且检测结果与人类质量判断一致。

### 10.1 实验目标（两条独立验证线）

**验证线 A（定量）**：loss dynamics 信号与人类质量标注的 Spearman 相关性。回答：「我们检测到的噪音是人类也认为质量差的吗？」

**验证线 B（定性）**：在自然噪音数据上，信号标记为「噪音」的样本是什么样的？人工核查 Top/Bottom 50 条，分类噪音类型。回答：「信号在真实数据中捕捉到了什么？」

### 10.2 验证线 A：与人类质量标注的对比

#### 10.2A.1 评测数据集：HuggingFace DIBT

DIBT（Data Is Better Together）是人工标注的指令-响应对质量排名数据集。每条 prompt 有 4-5 个不同质量的 response，按 10 个维度标注：

| 维度 | 含义 | 预期 loss dynamics 相关性 |
|------|------|:---:|
| **fluency** | 语言流畅度 | 中——流畅度差的可能导致 token 分布异常 |
| **coherence** | 与 prompt 的逻辑一致性 | 中——不连贯的答案 loss 模式可能异常 |
| **factuality** | 事实准确性 | **低——语义正确性在 token 级别不可见**（已知盲区） |
| **relevance** | 与问题的相关度 | 中——离题答案可能产生异常 token 序列 |
| **completeness** | 回答的完整性 | 低——简短高质量 vs 详细低质量在 token 级难区分 |
| **conciseness** | 简洁度 | 低 |
| **helpfulness** | 有用性 | 中——综合性维度 |
| **safety** | 安全性 | **低——toxic 内容 token 统计正常**（已知盲区） |
| **diversity** | 多样性 | 低 |
| **overall** | 综合质量分数 | 中——综合维度 |

**核心假设**：loss dynamics 信号与 **fluency/coherence** 维度的相关性高于与 **factuality/safety** 维度的相关性——因为前者是统计异常，后者是语义异常。这是 paper 中「信号覆盖边界」论述的直接验证。

#### 10.2A.2 流程

```
1. 在 ShareGPT (90K) 或 WildChat (~1M) 上用 Qwen2.5-1.5B LoRA 训练 3 epoch
2. 记录每个训练样本的 loss（用于背景分布校准，非 DIBT 评估所必需）
3. 将 DIBT 所有 prompt-response pair 用训练好的模型做一次 forward pass
4. 对每条记录：提取 token_loss_top20, IFD, loss_cv（单点不可算，用 token_loss_top20 + IFD）
5. 对每条 prompt，计算各 response 的 loss dynamics 得分排序
6. 与 DIBT 人工质量排序做 Spearman 相关性
```

#### 10.2A.3 评估指标

| 指标 | 含义 | 通过与标准（1.5B/3B 受控实验已知 AUROC） |
|------|------|:---:|
| Spearman ρ(token_loss_top20, DIBT overall score) | P0 信号与人类质量的全局一致性 | 预期 ρ > 0.3（自然数据可能低于受控实验） |
| Spearman ρ(token_loss_top20, fluency) | P0 信号与流畅度 | 预期 ρ > 0.4（受控实验中 A 类噪音 AUROC 对应此） |
| Spearman ρ(token_loss_top20, factuality) | P0 信号与事实性 | 预期 ρ ≈ 0.05-0.1（已知盲区） |
| Spearman ρ(IFD, coherence) | IFD 与连贯性 | 预期 ρ > 0.3（IFD 测量的是 instruction-answer 相关性） |
| Spearman ρ(IFD, factuality) | IFD 与事实性 | 预期 ρ ≈ 0.1（与受控实验中 IFD 对 B/C/D 类 ≈ 随机一致） |

**如果 token_loss_top20 与 fluency 相关性 > 0.4**：受控实验中的高 AUROC 在真实数据上得到了人工评价的支撑。

**如果 token_loss_top20 与 factuality 相关性 ≈ 0.1**：精确量化了信号的已知盲区——「我们不能检测事实错误，但我们一开始就没声称能做到」。这就是诚实的方法界限陈述。

#### 10.2A.4 DIBT 数据集获取

```python
from datasets import load_dataset
dibt = load_dataset("DIBT/prompts_ranked", trust_remote_code=True)
# 约 1,000 prompts × 4-5 responses = ~5,000 ranked pairs
```

### 10.3 验证线 B：自然噪音的定性分析

#### 10.3B.1 数据集：ShareGPT 或 WildChat

| 候选 | 规模 | 噪音特征 | 推荐 |
|------|:---:|------|:---:|
| **ShareGPT** | ~90K 对话 | ChatGPT 真实对话——包含截断、幻觉、格式不一致 | ⭐ **首选**（规模适中，噪音类型多样） |
| WildChat | ~1M | 更大的规模，更完整的真实分布 | 备选（如 ShareGPT 噪音覆盖不足） |

#### 10.3B.2 流程

```
1. 用 Qwen2.5-1.5B LoRA 在 ShareGPT 上训练 3 epoch
2. 记录 per-sample loss（epoch 1-3）+ per-token loss（epoch 3）
3. 计算 token_loss_top20 + IFD + loss_cv（有跨 epoch 数据时）
4. 按 token_loss_top20 升序（最低分 = 最像噪音），各取 Top 50 + Bottom 50
5. 人工核查：
   - 每条看 instruction + response + token_loss_top20 / IFD 得分
   - 标注噪音类型：截断 / 格式错 / 事实错 / 流畅但离题 / 幻觉 / 正常
6. 统计：Top 50 中各类噪音的分布 vs Bottom 50 中高质量样本的分布
```

#### 10.3B.3 成功标准

| 指标 | 标准 | 含义 |
|------|:---:|------|
| Top 50（信号认为噪音）中截断/格式错占比 | ≥ 60% | 信号对结构噪音的 precision 足够 |
| Bottom 50（信号认为高质量）中事实错/幻觉占比 | ≥ 40% | 信号确实对不同类型错误有选择性盲区 |
| 人工标注噪音类型的分布一致性 | — | 提供 qualitative evidence for paper |

### 10.4 实验矩阵

| 参数 | 配置 |
|------|------|
| 模型 | Qwen2.5-1.5B-Instruct, LoRA (r=16) |
| 训练数据 | ShareGPT (90K) 或 WildChat 子集 (~100K) |
| Epoch | 3（与 Phases 1-5 的 epoch 数一致） |
| 评测数据 A | DIBT（~1,000 prompts × 4.5 responses = ~4,500 forward passes） |
| 评测数据 B | ShareGPT（从训练集中随机抽 100 条做人工核查） |
| GPU 时间 | 训练 ~3h + forward passes ~0.5h |

### 10.5 预期输出与 Paper 定位

#### 预期产出

| 产物 | 内容 |
|------|------|
| **Table X** | token_loss_top20 / IFD 与 DIBT 10 维度的 Spearman 相关性矩阵 |
| **Figure X** | 相关性的雷达图——fluency/coherence 高相关，factuality/safety 低相关→可视化信号边界 |
| **Table Y** | Token_loss_top20 Top/Bottom 50 的人工核查噪音类型分布 |
| **Qualitative examples** | 3-5 条典型样本：信号识别的结构噪音 vs 信号盲区的内容噪音 |

#### Paper 定位

> Section X: Generalization to Natural Noise
> 
> "我们前四组实验验证了信号在受控噪音注入环境下的区分力。本节验证两个问题：(1) 信号评分与人类质量判断是否一致？(2) 信号在真实数据中能捕捉什么类型的噪音？
> 
> 在 DIBT 人工质量排名数据集上，token_loss_top20 与 fluency/coherence 的 Spearman ρ = X.X，与 factuality 的 ρ = 0.X——精确复现了受控实验中的信号覆盖边界（Table 4 中 Noise B/D vs Noise A 的 AUROC 差异）。
> 
> 在 ShareGPT 的定性分析中，信号识别为「噪音」的 Top 50 样本以截断、格式错、编码损坏为主（占比 XX%）——这些恰好是真实数据中危害最大的 L2/L3 层级噪音（SQuAD 分析已验证）。信号未能检测到的低质量样本以事实错误为主——这些属于 L0 级别，对下游基本无害。"

### 10.6 风险与缓解

| 风险 | 概率 | 缓解 |
|------|:---:|------|
| DIBT 样本量太小（~4,500 forward passes），Spearman ρ 置信度低 | 中 | 增加 forward pass 次数（使用更大的 prompt set）；如果 ρ 低但方向一致，改为 qualitative correlation analysis |
| ShareGPT 经社区过滤后质量较好，低质量样本代表性不足 | 中 | 换用 WildChat（未过滤）或 WizardLM unfiltered |
| 信号在真实数据上的分布与受控实验差异大——无一组 AUROC 的 clean/noise 分离 | 高 | 这是**预期结果**——真实噪音不是二元的。改为展示 "信号评分分布 vs 人工扣留的极端样本"，用 qualitative evidence 代替 quantitative benchmarking |
| 人工核查成本高（100 条 ≈ 1-2h） | 低 | 可以接受，且核查结果直接用于 paper 的 qualitative examples |

