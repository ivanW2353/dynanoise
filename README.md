# dynanoise — 基于 Loss Dynamics 的 LLM 训练噪音检测

受控实验中验证 loss 轨迹信号（loss_cv、loss_trend、token_loss_top20、IFD）能否区分不可学习噪音与有价值困难样本，并实际验证清除噪音后的模型质量提升。

## 核心发现

- **H₁ 方向被证伪，但信号极强** — unlearnable 噪音的 loss_cv 低于 clean（0.013 vs 0.041），反转后 AUROC = 0.96
- **token_loss_top20 是最稳健信号** — 跨 1.5B/3B 两个模型尺度完全稳定（0.947 ± 0.001）
- **random_drop 始终优于精准过滤** — 三个独立实验中，随机丢弃 10% 数据的 MT-Bench 提升始终大于精准噪音过滤
- **Noise A 和 Noise E 指纹完全相反** — 需用 zscore 联合方案同时覆盖两类噪音

## 快速开始

```bash
cp config.yaml.example config.yaml          # 填入 DeepSeek API key
pip install -r requirements.txt

# 下载模型到本地
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/autodl-tmp/Qwen2.5-1.5B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct   --local-dir ~/autodl-tmp/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct   --local-dir ~/autodl-tmp/Qwen2.5-7B-Instruct

# Phase 1-3（数据 + 训练 + 信号分析）
python data/prepare_data.py --skip-api-calls   # 构造噪音数据集（本地）
python training/train_holdout.py               # holdout 模型
python training/train_main.py --model-size 1b  # 主模型
python analysis/compute_signals.py             # 信号计算
python analysis/q1_analysis.py                 # Q1-Q4 分析
```

## 项目结构

```
dynanoise/
├── config.yaml.example        # 配置模板（复制为 config.yaml 后填入 API key）
├── AGENTS.md                  # Agent 指南（代码陷阱 + 实验发现）
├── experiment-plan.md         # 完整实验方案文档
│
├── data/                      # Phase 1：数据构造
│   ├── prepare_data.py        #   噪音注入（4 类 + shortcut）
│   ├── preprocess_chat.py     #   LMSYS-1M 对话预处理
│   ├── quality_check.py       #   质检脚本
│   └── prompt_templates/      #   DeepSeek API prompt
│
├── training/                  # Phase 2：训练与信号采集
│   ├── train_main.py          #   主模型 LoRA 训练（per-epoch loss）
│   ├── train_holdout.py       #   Holdout 模型训练
│   ├── compute_all_losses.py  #   从 checkpoint 提取 per-sample loss
│   ├── compute_ifd.py         #   IFD 计算
│   └── compute_rho.py         #   RHO score 计算
│
├── analysis/                  # Phase 3：信号分析
│   ├── compute_signals.py     #   P1/P0 信号 + composite score
│   ├── q1_analysis.py         #   Q1 — CV/trend 区分度
│   ├── q2_analysis.py         #   Q2 — P1+P0 联合
│   ├── q3_analysis.py         #   Q3 — 时延分析
│   ├── q4_analysis.py         #   Q4 — 噪音消融
│   ├── phase6_dibt_correlation.py  # Phase 6A — DIBT 相关性
│   ├── phase6_extract_samples.py   # Phase 6B — 样本提取
│   └── generate_figures*.py        # 图表生成
│
├── downstream/                # Phase 4：下游验证
│   ├── train_filtered.py      #   5 组过滤训练
│   ├── evaluate.py            #   MT-Bench + MMLU 评估
│   └── manual_inspection.py   #   人工抽检辅助
│
├── docs/                      # 分析报告 + 图表
│   ├── phase1-4-analysis.md   #   1.5B 完整报告
│   ├── phase1-3-3b-analysis.md #  3B 报告（含跨模型对比）
│   ├── phase5-analysis.md     #   Phase 5 shortcut 报告
│   ├── figures/               #   1.5B 图表
│   ├── figures_3b/            #   3B 图表
│   └── figures_p5/            #   Phase 5 图表
│
├── run_pipeline.sh            # 一键运行 Pipeline
└── results/                   # 信号数据 + 表格（git 排除）
```

## 实验阶段总览

| Phase | 内容 | 模型 | 状态 |
|:---:|------|:---:|:---:|
| 1 | 数据构造（4 类噪音注入） | — | ✅ |
| 2 | 基准训练 + IFD + RHO | 1.5B | ✅ |
| 3 | Q1-Q4 信号分析 | 1.5B | ✅ |
| 4 | 下游过滤训练 + MT-Bench | 1.5B / 3B | ✅ |
| 5 | Systematic shortcut 验证 | 1.5B / 3B | ✅ |
| 6 | 自然噪音泛化（LMSYS-1M + DIBT） | 1.5B | 进行中 |

## 论文贡献

1. **首次在 LLM SFT 场景验证 Data Maps 的 loss CV 适配方案**
2. **系统性对比** P1（loss dynamics）vs P0（token-level）vs Gold（RHO）的区分度
3. **发现 token_loss_top20 的跨模型尺度稳健性**
4. **定义了 loss dynamics 信号的实用边界**（噪音浓度阈值 ~15-20%）

## 依赖

- Python 3.10+
- PyTorch 2.0+, Transformers 4.40+, PEFT 0.10+
- DeepSeek API（Noise B/D 构造需要，可选）
- GPU：≥24GB（推荐 32GB+，用于 7B judge 推理）

## 引用

查看 `docs/` 目录下的各阶段分析报告获取完整实验细节。
