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

## 知识资源（按需 read_file 加载）

- `references/_facts.md` — 62 变体事实表（机器抽取的 arena/zone/subject 字段，最权威）
- `references/identification-decision-tree.md` — 决策流程详解 + 反问质量准则
- `references/default-template-fallback.md` — 范式 → 默认变体降级表（反问失败时用）
- `references/by-template/<大类>.md` — 单个 EV19 大类的变体差异 + 推荐场景（同事 PR 持续补充）
- `references/by-experiment/<范式>.md` — 单个学术范式的指标 / 模板候选 / 解读语言（同事 PR 持续补充）

**Token 节省提示**：不要一次性加载所有 references，按对话需要 read_file 单个文件。
