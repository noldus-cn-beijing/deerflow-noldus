# SkillOpt 应用于 noldus-insight — 实施计划

## 整体路线

```
行为学专家产出 Golden Cases（eval benchmark）
    ↓
改造 SkillOpt 代码（自定义 EnvAdapter + 评分函数）
    ↓
跑 SkillOpt 优化循环 → best_skill.md（每个范式一个）
    ↓
用优化后的 skill 驱动 agent → 生成高质量 SFT 轨迹
    ↓
微调 Qwen3.6-35B-A3B → 模型内化 skill 知识（SFT→GRPO）
```

## 阶段 1：Golden Cases — 行为学专家建立 Benchmark

### 为什么必须专家来做

SkillOpt 能工作的前提是**每个 benchmark 任务有明确的对/错评分标准**。对 noldus-insight 来说，评分标准是：

- 指标数值是否在专家认可的范围内
- 统计方法选择是否正确（参数 vs 非参数）
- 效应量判断是否合理
- 表型推断是否符合领域共识
- 是否包含不该出现的错误结论（幻觉）

这些只有行为学专家能定义。synthetic fixture 只能作为初始占位，最终 golden case 必须来自专家。

### 目标

v0.1 的 6 个范式（EPM/OFT/LDB/FST/Zero Maze/TST），每个至少 2-3 个 golden case，覆盖：

- **正常 case**：典型 control vs treatment 数据，标准分析路径
- **边界 case**（至少 1 个）：小样本（n < 5）、异常值、缺失列等
- 总计目标：**12-18 个 golden cases**

### Golden case 结构（已有框架，直接使用）

```
golden-cases/case-XXX-<paradigm>-<描述>/
├── raw-data/                  # EthoVision XT 原始导出文件
├── metadata.yaml              # 范式、物种、分组、实验条件
├── expected-analysis.yaml     # ★ 专家期望的分析结论（可机读评分）
└── notes.md                   # 专家推理过程（供 SFT 用）
```

`expected-analysis.yaml` 中的评分字段（已在 `SCHEMA.md` 中定义）：

- `metrics[].expected_range` → 指标数值是否在范围内
- `findings[]` → 必须出现的分析结论类型（statistical_conclusion / phenotype_indication / confound_note 等）
- `required_keywords` / `forbidden_claims` → 关键词匹配
- `severity` → 错误严重程度

### 当前状态

框架就绪（SCHEMA.md + TEMPLATE/ + validate_golden_case.py），但 golden case 数量为零。这是阻塞项——没有 benchmark 就无法跑 SkillOpt。

## 阶段 2：改造 SkillOpt 代码

### 能复用的部分（不需要改动）

| SkillOpt 模块 | 作用 | 复用方式 |
|-------------|------|---------|
| `skillopt/engine/trainer.py` | ReflACTTrainer 训练循环 | 直接使用，6 阶段流程不变 |
| `skillopt/gradient/reflect.py` | 分析师 LLM 分析失败轨迹 | 复用，但需要改写 reflect prompt 让分析师理解行为学领域 |
| `skillopt/gradient/aggregate.py` | 合并多个 analyst 的编辑建议 | 直接使用 |
| `skillopt/optimizer/` | 学习率调度、编辑预算、更新策略 | 直接使用 |
| `skillopt/model/` | LLM 后端（已有 Anthropic Claude 支持） | 直接使用 |
| `skillopt/evaluation/` | 验证门控逻辑 | 直接使用 |

### 需要新建/改造的部分

#### 2.1 新建 `skillopt/envs/ethoinsight/` — 核心改造

这是最大的改动。需要实现 SkillOpt 的 `EnvAdapter` 接口，让训练循环能驱动 noldus-insight agent：

```
skillopt/envs/ethoinsight/
├── __init__.py          # 注册 ethoinsight benchmark 环境
├── adapter.py           # EnvAdapter 实现
├── dataloader.py        # 从 golden-cases/ 加载任务
├── rollout.py           # 驱动 agent 流水线执行单个任务
├── scorer.py            # 对比 agent 输出 vs expected-analysis.yaml → 分数
└── initial.md           # 种子 skill（当前手写版本的合并）
```

**`adapter.py` — EnvAdapter 实现**：

```python
class EthoInsightAdapter(EnvAdapter):
    """让 SkillOpt 训练循环能驱动 noldus-insight agent"""
    
    def setup(self, cfg):
        # 确保 agent 服务可访问（LangGraph + Gateway）
        ...
    
    def build_train_env(self, seeds, batch_size):
        # 从 golden-cases/ 加载训练集任务
        ...
    
    def build_eval_env(self, seeds, batch_size):
        # 从 golden-cases/ 加载验证集任务（held-out）
        ...
    
    def run_rollout(self, env, skill_md):
        # 核心：用当前 skill 驱动 agent 完成一个 golden case
        # 1. 将 skill_md 注入 agent 的 skill 配置
        # 2. 上传 raw-data 文件
        # 3. 发送分析请求
        # 4. 等待完整流水线完成（lead → code-executor → data-analyst → ...）
        # 5. 收集所有 subagent 输出
        # 6. 返回轨迹（trajectory）+ 原始输出
        ...
```

**`scorer.py` — 评分函数**：

对比 agent 最终输出与 `expected-analysis.yaml`：

- 指标数值匹配 → 每项 0-1 分（在 expected_range 内 = 1）
- 必需发现类型命中 → 每项 0-1 分
- 禁止声明违反 → 每项扣分
- 最终分数 = 加权平均

**`dataloader.py` — 数据加载**：

- 扫描 `golden-cases/case-*/` 目录
- 解析 `metadata.yaml` + `expected-analysis.yaml`
- 按范式分层划分 train/valid/test split

#### 2.2 改造 reflect prompt

SkillOpt 的 reflect 阶段用"分析师"LLM 看 agent 失败轨迹，提出 skill 编辑建议。当前 reflect prompt 是通用的，需要加入行为学领域上下文：

- 分析师需要理解：范式判读逻辑、handoff 契约、catalog resolve 流程
- 在 reflect prompt 中注入简短的领域知识摘要，让分析师能判断"这个失败是因为 skill 指令模糊，还是 agent 本身的问题"

#### 2.3 配置文件

```yaml
# configs/ethoinsight/epm.yaml
env: ethoinsight
paradigm: epm
train:
  num_epochs: 4
  batch_size: 8        # golden case 少，batch 不能太大
  train_size: 8        # 留 2-4 个做验证
optimizer:
  learning_rate: 0.3   # 编辑预算（文本改动量上限）
  lr_scheduler: cosine
model:
  optimizer_backend: anthropic_chat    # 分析师用 Claude
  optimizer_model: claude-sonnet-4-6
  target_backend: deepseek_chat        # agent 用 deepseek
  target_model: deepseek-v4-pro
evaluation:
  use_gate: true
  gate_metric: soft     # 行为学分析没有"精确匹配"，用软评分
```

### 改造工作量估计

| 组件 | 工作量 | 说明 |
|------|--------|------|
| `adapter.py` | 3-5 天 | 需要理解 SkillOpt EnvAdapter 接口 + noldus-insight agent API |
| `scorer.py` | 1-2 天 | 基于已有 SCHEMA.md 定义 |
| `dataloader.py` | 1 天 | 简单，golden-cases 结构已规范化 |
| `rollout.py` | 2-3 天 | 需要处理多 subagent 异步流水线 |
| reflect prompt 改造 | 1-2 天 | 注入领域上下文 |
| 配置文件 | 0.5 天 | 每个范式一个 YAML |
| **总计** | **8-14 天** | |

## 阶段 3：跑 SkillOpt 优化循环

### 前置条件

- 阶段 1 完成：每个目标范式至少 2 个 golden case
- 阶段 2 完成：ethoinsight EnvAdapter 通过基本冒烟测试

### 运行方式

```bash
# 对每个范式独立优化（skill 是 per-subagent 的，但优化可以按范式拆分）
python scripts/train.py --config configs/ethoinsight/epm.yaml
python scripts/train.py --config configs/ethoinsight/oft.yaml
python scripts/train.py --config configs/ethoinsight/fst.yaml
# ...
```

### 预期产出

每个范式一个 `best_skill.md`（或每个 subagent 一个），包含：

- 紧凑的领域知识（去除了未被 benchmark 验证为有用的内容）
- 经过验证的指令（每条指令都经历过"删掉它 benchmark 分数会降吗"的测试）
- 明确的成功标准和失败边界

### 成本和周期

- **一次训练**：4 epoch × 8 batch ≈ 32 次 agent 完整流水线运行 × 每次 2-5 分钟 ≈ 1.5-4 小时
- **LLM API 成本**：analyst reflect 调用（Claude Sonnet）+ agent rollout（deepseek）≈ $10-30/次训练
- **6 个范式**：可以并行跑，总计 1-2 天

## 阶段 4：用优化后的 Skill 生成 SFT 数据

阶段 3 产出的 `best_skill.md` 经过 benchmark 验证，用它驱动 agent 产生的轨迹质量上限更高。

1. 用优化后的 skill 替换当前 skill 文件
2. 在真实用户数据上运行 agent
3. Training data flywheel 采集的轨迹质量提升
4. 配合专家 feedback（correct/needs_fix/wrong），构建高质量 SFT 数据集
5. 达到 Phase 1 目标（1K SFT 样本 + 300 DPO 对）

## 阶段 5：微调模型

- 基座：**Qwen3.6-35B-A3B**（2026-06-30 拍板；历史曾写 Qwen3-30B-A3B-Instruct-2507 已作废）。SSOT 见 memory `project_base_model_qwen36_35b_a3b`。候选 Qwen3.5-35B-A3B 同构仅作退路（若 Gated DeltaNet 在 verl 训练时支持踩坑严重）
- 平台：火山引擎 verl 或 Fireworks（客户硬件 RTX 5090 32GB，多卡或 LoRA-RL）
- 数据：阶段 4 采集的高质量 SFT 数据
- 目标：模型内化 v0.1 核心能力，推理时减少 skill 注入

## 前置依赖和阻塞项

```
行为学专家产出 Golden Cases  ← 阻塞项 #1，无此无法启动阶段 2/3
    ↓
改造 SkillOpt 代码（EnvAdapter）  ← 可与阶段 1 部分并行（先用 synthetic 数据开发）
    ↓
跑优化循环  ← 需要阶段 1 + 2 都完成
    ↓
生成 SFT 数据 + 微调  ← 7 月启动
```

## 与现有计划的关系

- 阶段 1（Golden Cases）与 Phase 0 基线验证并行推进
- 阶段 2（代码改造）可以在等待专家时启动，用 synthetic fixture 做开发和测试
- 阶段 3（跑优化）在 Phase 0 收尾、Phase 1 SFT 开始之前完成
- 阶段 4-5 对应现有的 Phase 1 SFT 计划（7-8 月）
- v0.1 硬 deadline 9 月不变

## 风险

1. **Golden Cases 延迟**：行为学专家可能无法及时产出足够 case。缓解：先用 synthetic fixture 开发 + 验证代码通路，专家 case 就位后直接替换数据跑正式优化。
2. **多 subagent 流水线的 rollout 复杂性**：SkillOpt 设计为单轮 agent，noldus-insight 是多轮多 subagent。缓解：将整个流水线封装为一个"黑盒 rollout"，SkillOpt 只看最终输出和分数，不关心中间过程。
3. **评分函数的主观性**：行为学分析没有绝对的"对/错"，expected_range 可能过宽或过窄。缓解：软评分（soft gate），不要求精确匹配。
