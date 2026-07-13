# AGENTS.md — dynanoise 实验项目指南

## 环境设置

```bash
# 1. 复制配置（config.yaml 已从 git 排除）
cp config.yaml.example config.yaml
# 填入 DeepSeek API key

# 2. 安装依赖
pip install -r requirements.txt
pip install lm_eval==0.4.4

# 3. 模型必须下载到 ~/autodl-tmp/（不依赖 HF cache）
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/autodl-tmp/Qwen2.5-1.5B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct   --local-dir ~/autodl-tmp/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct   --local-dir ~/autodl-tmp/Qwen2.5-7B-Instruct
```

## 代码陷阱（容易猜错的）

### 1. DataCollator 必须用 `DataCollatorForSeq2Seq`
所有训练脚本（`train_main.py`, `train_holdout.py`, `train_filtered.py`）必须用 `DataCollatorForSeq2Seq(padding=True)` **而不是** `DataCollatorForLanguageModeling`。后者与 Qwen tokenizer 的 padding 不兼容，会报 `Unable to create tensor`。

### 2. PeftModel 加载 checkpoint 后需要 `enable_input_require_grads()`
`PeftModel.from_pretrained(base, ckpt)` 不会自动调用此方法，梯度会断开。只有 `get_peft_model()` 自动处理。`train_filtered.py` 已修复。

### 3. evaluate.py OOM 修复：两阶段加载
Judge 模型（7B, ~14GB）和目标模型（3B, ~12GB）不能同时加载。当前代码分两阶段：
- Phase 1：逐个加载目标模型生成答案（用完即 `del model, base; torch.cuda.empty_cache()`）
- Phase 2：只加载 judge 做 pairwise judging

### 4. config.yaml 已从 git 排除
有 `config.yaml.example` 作为模板。API key 不能提交到 git。

### 5. `compute_all_losses.py` 自动检测 steps_per_epoch
公式：`ceil(dataset_size / effective_bs)`，其中 effective_bs = per_device_bs × grad_accum。也可以显式指定 `--steps-per-epoch`。

### 6. composite score 模式
- `--composite-mode cv_trend`（默认）：`alpha * (-loss_cv) + (1-alpha) * (-loss_trend)` — 用于 Phase 1-4
- `--composite-mode zscore`：绝对 z-score 偏差 `|z_cv| + |z_t20| + |z_ifd|` — 用于 Phase 5+，同时检测 Noise A 和 Noise E

### 7. Phase 1 数据构造
```bash
# 原始 Noise A+B+C+D（需 DeepSeek API）
python data/prepare_data.py

# 仅 Noise A+C（跳过 API）
python data/prepare_data.py --skip-api-calls

# Phase 5 模式：仅 Noise A + Noise E shortcut
python data/prepare_data.py --phase5
```

### 8. 可忽略的警告
- `libgomp: Invalid value for environment variable OMP_NUM_THREADS` → autodl 环境，无害
- `grad_norm: 0` → `PeftModel.from_pretrained()` 导致的日志显示问题，梯度正常流动
- `torch_dtype is deprecated` → 用 `dtype` 即可，不影响运行

## 核心实验发现（避免重复犯错）

- **loss_cv 方向与 H1 假设相反**：unlearnable 噪音的 loss_cv **低**于 clean（0.013 vs 0.041），需用 `-loss_cv` 反转方向
- **Noise A 和 Noise E 的指纹完全相反**：A 是低 loss_cv + 高 IFD；E 是高 loss_cv + 低 IFD
- **token_loss_top20 是唯一跨模型尺度完全稳定的信号**（0.947 ± 0.001）
- **在 5-10% 噪音浓度下，random_drop 的过滤效果始终优于精准过滤**（三个独立实验一致结论）

## 验证 GPU 显存是否够

- 1.5B LoRA 训练：~6GB
- 3B LoRA 训练：~12GB
- 7B judge 推理：~14GB
- 1.5B + 7B 同时加载：~20GB
- 3B + 7B 同时加载：~26GB（需 ≥32GB GPU）

## 完整实验流程

```
Phase 1 → 数据准备（注入噪音，获 ground truth）
Phase 2 → 训练 + IFD + RHO 计算
Phase 3 → compute_signals.py + Q1-Q4 分析
Phase 4 → 过滤训练（train_filtered.py）+ MT-Bench 评估
Phase 5 → systematic shortcut 噪音验证
Phase 6 → 自然噪音泛化验证（lmsys-chat-1m + DIBT）
```
