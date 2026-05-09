---
name: ethovision-paradigm-knowledge
description: >
  EthoVision XT 19 模板知识库（20 大类 / 62 变体）+ 学术实验范式映射。
  用于在用户上传数据并请求分析时识别其使用的 EV19 模板变体（如 PlusMaze-AllZones），
  并把识别结果作为 ev19_template 字段写入 experiment-context.json。
  使用对话识别（read 用户消息 + 文件名 + 必要时 read raw txt meta），
  必要时通过 ask_clarification 反问，反问失败时按 default-template-fallback.md 兜底。
version: 0.1.0
author: noldus-insight
---

# EthoVision Paradigm Knowledge — EV19 模板识别 + 学术范式映射

## 何时使用此 skill

**必须使用**：用户提到任何 EthoVision 实验数据分析需求时（含上传 raw txt 文件 + 请求分析/统计/可视化/报告）。

**可跳过**：纯知识问答（无数据上传 + 概念性问题）；追问已有分析结果；闲聊。

## 核心原则

1. **EV19 模板 = 用户语言**（agent 与用户对话时使用），**学术范式 = 内部分析路径**（agent 调 set_experiment_paradigm 时填这个）。
2. **不要硬猜**——如果信息不足，**用 ask_clarification 给结构化选项**让用户选；不要瞎填导致下游分析跑错路径。
3. **反问最多 1 次**——LoopDetectionMiddleware 会在重复反问时强制中断；如果第一次反问后用户答 "不知道"，按 references/default-template-fallback.md 选默认值进入分析。
4. **反问前必读 raw 文件 meta**——用 read_file 读用户上传的第一个 raw txt 前 50 行，看单位（毫米=鱼 / 厘米=啮齿）、追踪点（单点/三点）、zone 列结构，把候选缩到 ≤3 个再问。

## Workflow

### Step 1: 收集证据

读以下信息（由 agent 综合判断）：
- 用户消息文本（"高架十字迷宫"、"EPM"、"焦虑测试" 等关键词）
- 上传文件名（"轨迹-EPM-Trial 1...txt" 等）
- 文件数量 + Subject 数（5 Subject = shoaling / 三箱社交，2 Arena = 三箱社交）
- 必要时 read_file 第一个 raw txt 前 50 行查 meta + 列结构

### Step 2: 决策

按 `references/identification-decision-tree.md` 决策：
- 候选 = 1 高置信度 → 直接 set_experiment_paradigm（不反问）
- 候选 2-3 → ask_clarification 给结构化选项 + 推荐项放第一位 + 默认值兜底说明
- 候选 0 或 ≥4 → ask_clarification 先问大实验类型

### Step 3: 调 set_experiment_paradigm

```
set_experiment_paradigm(
    paradigm="epm",                    # 学术范式 key（snake_case 英文）
    paradigm_cn="高架十字迷宫",         # 中文显示名
    category="anxiety",                # 大类
    subject="rodent",                  # rodent | fish | insect | other
    ev19_template="PlusMaze-AllZones", # EV19 变体 ID（白名单内）
)
```

工具会校验 `ev19_template` 在 62 变体白名单内；如不在，会返回错误 + 候选模板。

## 分析阶段使用本 skill（set_experiment_paradigm 之后）

模板识别完成（experiment-context.json 已写入 ev19_template + paradigm）后，**在派遣分析子代理前**，lead agent 必须 read 以下两个文件，把领域知识纳入派遣 prompt：

### 派遣 code-executor 之前

read `references/by-experiment/<paradigm>.md`，从中提取这两段写入 task() 的 prompt：

1. **"必须计算的指标"段** —— 拷贝到 code-executor 的 prompt 中，明确告诉它 compute_metrics 步骤要算哪些。
   - 例：派遣 EPM 分析时，从 `by-experiment/epm.md` 读"必须计算的指标"段，把"开臂时间百分比、开臂进入百分比、开臂进臂次数、开臂进臂时间、总进臂次数"作为指标清单传给 code-executor。

2. **"常见脱险点 / 数据质量风险"段** —— 同上拷贝，让 code-executor 在 assess_and_handoff 阶段做对应的质控判断。
   - 例：EPM 数据中"总进臂次数 < 8 时开臂指标降低可能为运动抑制"这条风险，code-executor 应在 handoff 中标记。

### 派遣 report-writer 之前

read `references/by-experiment/<paradigm>.md`，从中提取这两段写入 task() 的 prompt：

1. **"报告解读语言"段** —— 报告写作的术语规范（如 EPM 必须用"开臂滞留时间百分比"而非"开臂时间"）。
2. **"与其他实验的区分"段** —— 写 Discussion 时避免误读（如"EPM 不评估抑郁样行为，那是 FST/TST 的范畴"）。

### 模板变体差异查询（reactive，不强制）

如果 agent 在分析中需要解释为什么用此变体（用户追问"为什么选 PlusMaze-AllZones 而不是 -FewZones"）：read `references/by-template/<大类>.md`（例如 `by-template/PlusMaze.md`），找到该变体节的"这个变体相对其他变体的核心差异"段。

### 不要做的事

- ❌ 不要把整个 by-experiment/<paradigm>.md 文件原样塞进 task() 的 prompt — 只摘需要的段，避免 prompt 膨胀
- ❌ 不要 read MVP 范围外（如 shoaling/AquariumTrack3D 等）的 by-experiment/by-template — 同事尚未填写，read 到的是空占位
- ❌ 不要在每次 task() 派遣前都重 read — agent 应在一个 thread 内缓存第一次 read 的结果，跨 task() 复用

## 知识资源（按需 read_file 加载）

- `references/_facts.md` — 62 变体事实表（机器抽取的 arena/zone/subject 字段，最权威）
- `references/identification-decision-tree.md` — 决策流程详解 + 反问质量准则
- `references/default-template-fallback.md` — 范式 → 默认变体降级表（反问失败时用）
- `references/by-template/<大类>.md` — 单个 EV19 大类的变体差异 + 推荐场景（同事 PR 持续补充）
- `references/by-experiment/<范式>.md` — 单个学术范式的指标 / 模板候选 / 解读语言（同事 PR 持续补充）

**Token 节省提示**：不要一次性加载所有 references，按对话需要 read_file 单个文件。
