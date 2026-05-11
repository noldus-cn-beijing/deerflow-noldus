# Handoff: EV19 模板范式体系重定位 — 设计 + Review 包就绪

**日期**: 2026-04-29
**状态**: 设计完成、review 包就绪、文档已更新；**等行为学同事补充 P0 7 个文件**后进入实施
**上一会话**: [2026-04-29-event-loop-fix-v2-completed-handoff.md](2026-04-29-event-loop-fix-v2-completed-handoff.md)

---

## 1. 本会话目标

用户在 Event loop 修复后做端到端测试，发现 Gate 1（实验范式反问）没生效——agent 在用户上传文件之前没主动问"用了哪类模板"，上传文件后又因为文件名包含 `Shoaling behavior` 直接断言 paradigm，跳过两级 UI。

调查后定位到根本问题：**当前 prompt 用的"7 大类 18 范式"分类表与 EthoVision XT 19 真实模板（20 大类 62 变体）不对应**。

用户决定写设计文档，把范式体系从"学术分类"重定位到"EV19 模板 + 学术范式"双层。

---

## 2. 产出清单

### 2.1 设计文档
- [docs/plans/2026-04-29-ev19-template-paradigm-design.md](../plans/2026-04-29-ev19-template-paradigm-design.md)
- 内容：根因分析、三种方案对比、选项 B（双层）详细设计、三层架构（事实/知识/流程）、Gate 1 三步流程、8 步实施拆解
- **决策记录**（10 条）：双层 / 三步对话流 / 知识做成独立 skill / N:M 用 markdown 不用 dict / ...

### 2.2 起草脚本
- [scripts/build_ev19_template_index.py](../../scripts/build_ev19_template_index.py)（363 行）
  - 扫 `demodata/ev19 templates/` 解析 `templateMetaData.xml` + 目录名
  - 输出 `_facts.json`（机器可读事实表）+ 20 个 by-template/<大类>.md 草稿
- [scripts/build_ev19_experiment_drafts.py](../../scripts/build_ev19_experiment_drafts.py)（234 行）
  - 列出 20 个常见实验类型（基于 PRD 的范式 + EV demodata 反推）
  - 输出 20 个 by-experiment/<实验>.md 草稿

### 2.3 行为学同事 review 包
- [docs/review-packages/2026-04-29-ev19-templates/](../review-packages/2026-04-29-ev19-templates/)
- `README.md` — 给同事的填写说明 + P0 优先级
- `_facts.json` — 62 变体的机器可读事实表
- `by-template/` — 20 个大类草稿（🟢 字段已填，🟡 待补）
- `by-experiment/` — 20 个实验草稿（🟢 字段已填，🟡 待补）

### 2.4 文档更新（防止下一个 agent 走老路）
- `CLAUDE.md` 第 9 条后加第 10、11 条（范式体系重定位 + memory event-loop 修复）
- `docs/roadmap.md` Phase 0 加 M0.2 EV19 重定位里程碑，更新最近更新行
- `docs/architecture-diagram.md` 顶部加警告头
- `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md` Gate 0 加废弃警告头
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 文件级注释提示

---

## 3. 关键设计决策

### 3.1 三层架构（事实 / 知识 / 流程 分离）

```
层 1：静态注册表（事实）— packages/ethoinsight/ev19_templates.py
  ↓ 由扫描脚本自动生成，不含领域解读
层 2：渐进披露 skill（领域知识）— packages/agent/skills/custom/ethovision-paradigm-knowledge/
  ↓ by-template/ 和 by-experiment/ 双向索引，markdown 维护
层 3：Gate 1 三步流程（agent prompt + 中间件）
  ↓ prompt 描述流程，中间件检查 ev19_template + paradigm 双字段
```

**为什么三层而非合并**：事实数据用 Python dict（让代码直接 import），领域知识用 markdown（行为学同事维护），流程用 prompt + 中间件（让 deepseek 可控）。

### 3.2 Gate 1 三步对话流（manual 模式）

1. 第一步：选 EV19 大类（20 个，按使用频率排序前 5-7 个先显示）
2. 第二步：在该大类下选具体变体（1-15 个；只有 1 个时跳过）
3. 第三步（仅 ambiguous 时）：选学术范式（一个 EV 模板对多个学术实验时）

写入 `experiment-context.json` 时同时保存 `ev19_template` + `paradigm` 双字段。

### 3.3 文件名推断采"开门 + 补救"模式

agent 从文件名推断 paradigm 时**不阻断**，但工具返回的 ToolMessage 必须包含"如不正确请说'重选范式'"提示，给用户兜底。

---

## 4. 等待行为学同事的事

P0（先填这 7 个就能跑通 shoaling/EPM/OFT 三大范式）：

```
docs/review-packages/2026-04-29-ev19-templates/
├── by-experiment/shoaling.md          ← 已经做完一次 shoaling 端到端测试，最先要 ground truth
├── by-experiment/epm.md               ← 路线图 v0.1 必须完成
├── by-experiment/open_field.md        ← 路线图 v0.1 必须完成
├── by-template/OpenFieldRectangle.md  ← OFT 矩形版
├── by-template/OpenFieldCircle.md     ← OFT 圆形 + 鱼类相关
├── by-template/PlusMaze.md            ← EPM 模板
└── by-template/AquariumTrack3D.md     ← shoaling 的 3D 版
```

P1：剩余 17 个 by-experiment + 17 个 by-template（异步进行）。

填写规则见 [review 包 README](../review-packages/2026-04-29-ev19-templates/README.md)。

---

## 5. 实施步骤（接手 agent 用）

> ⚠️ Step 0 是行为学同事的事，等 P0 完成后才能开始 Step 1。期间技术侧可以并行做 Step 1 的事实表 Python 化和 Step 3 的 context schema 升级。

### Step 1：建模事实表（1 天）
- 把 `_facts.json` 转成 Python 字典 `EV19_VARIANTS` / `EV19_CATEGORIES`
- 写 `packages/ethoinsight/ethoinsight/ev19_templates.py`
- 单测：覆盖 20 大类、62 变体 round-trip

### Step 2：搬 review 包到 skill 目录（0.5 天）
- 创建 `packages/agent/skills/custom/ethovision-paradigm-knowledge/` 目录
- 写 `SKILL.md`（设计文档 4.1.3 节有结构示例）
- review 包的 `by-template/` 和 `by-experiment/` 移进 `references/`
- 写 CI 脚本：从 references/ 反向汇总生成 SKILL.md 里的两个索引段
- 写 CI 测试：双向一致性（template 提到的实验在 by-experiment 都有；by-experiment 推荐的模板在 _facts.json 都存在）
- 在 `extensions_config.json` 启用这个 skill

### Step 3：experiment-context schema 升级（0.5 天）
- 增加 `ev19_template` / `arena_template` / `zone_template` / `subject_type` 字段
- 增加 `_schema_version: 2`
- `experiment_context.py` 加新函数 `read_ev19_template()` / `is_ev19_template_set()`
- `set_experiment_paradigm` 工具签名扩展，加 `ev19_template` 参数

### Step 4：prompt 改写（1 天）
- 替换"7 大类 18 范式"表为 EV19 模板 20 大类列表
- 改写 Gate 1 描述为三步流程
- 加"上传前触发"分支说明
- 加"重选范式"补救机制说明
- 加"调 ethovision-paradigm-knowledge skill"的指令（什么时候调、调哪个 reference 文件）

### Step 5：GateEnforcementMiddleware 增强（0.5 天）
- 增加对 `inferred_from` 字段的检查
- 文件名推断或对话推断时附加 ToolMessage 提示

### Step 6：UI 端 ask_clarification 改造（1 天）
- 第一步：20 个大类（按使用频率排序）
- 第二步：只显示该大类下变体（1-15 个；1 个时跳过）
- 第三步（ambiguous 时）：显示该模板对应的多个学术范式
- 检查 frontend 是否支持"返回上一步"

### Step 7：迁移与回归（0.5 天）
- 老 thread 的 context.json 缺新字段 → fail open
- 跑全量测试，重点测 memory / gate enforcement / experiment_context
- E2E 烟测：用 demodata 各跑一次，确认每种模板都能走通

### Step 8：文档（0.5 天）
- 更新 [quality-gates.md](../../packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md) — 删警告头，重写 Gate 0 描述
- 更新 [paradigm-analysis-tools-spec.md](../specs/paradigm-analysis-tools-spec.md)（如果需要）
- 写本会话的"实施完成"交接文档
- 删本文 + 老 quality-gates 顶部警告头 + prompt.py 顶部维护提示
- 把 `docs/architecture-diagram.md` 顶部警告头改成"已重定位"

**总工时**：5-7 天工程工时（不含行为学同事 review 时间，那是异步进行的）

---

## 6. 风险与注意事项

| 风险 | 应对 |
|---|---|
| 行为学同事没时间 review | P0 只 7 个文件 ≈ 半天工作量；review 包已经把机器字段填好了，同事只需要补领域知识；可以分批，先填 shoaling 即可 unblock 第一次端到端测试 |
| paths.py 类的"受保护文件"在 4-29 已被改 | 本次实施不直接改 paths.py，但 deerflow 上游 sync 会触发它；做 Step 1-3 时尽量不动 paths.py |
| 老 thread 的 experiment-context.json 缺新字段 | fail open 策略；可加一次性迁移脚本（v0.1 后再做） |
| deepseek 仍可能跳步 | 三步流程 + 中间件代码兜底（task() 时 gate 检查 ev19_template 存在）；prompt 用正面指令（CLAUDE.md 第 6 条） |
| EV19 升级到 V20/V21 | 字典自动生成脚本写防御式：已知字段缺失时记 warning 不 crash |
| 用户用其他工具（DeepLabCut/Bonsai）的数据 | 暂不支持；context.json 的 `ev19_template` 字段允许填 `null` 或 `external_pipeline`；留待 v0.1 后 |

---

## 7. 与 4-28 已实施的 gate-enforcement 的关系

不动的部分：
- 中间件框架（GateEnforcementMiddleware）
- task() 边界拦截
- Gate 2 数据质量检查

需要回写的部分：
- `_check_gate1` 的判定从「context.json 是否存在」改为「context.json 是否有 ev19_template + paradigm 双字段」
- prompt 里的 Gate 1 段（"7 大类 18 范式"那张表）整个换掉
- `set_experiment_paradigm` 工具签名加 `ev19_template` 参数

---

## 8. 给下一个 agent 的注意事项

如果你接手这个任务，**第一件事是检查 review 包状态**：

```bash
ls -la docs/review-packages/2026-04-29-ev19-templates/by-experiment/shoaling.md
# 看顶部是否有 "**status**: reviewed by <名字> on YYYY-MM-DD"
# 如果没有 → 先确认行为学同事是否在填，如果还没填，向用户反馈
# 如果已填 → 继续看 P0 其他 6 个文件状态
```

如果 P0 7 个还有任何一个没填，**不要进入 Step 4 prompt 改写**（prompt 里要引用具体的 by-experiment/by-template 内容）。Step 1（事实表 Python 化）和 Step 3（context schema 升级）可以提前做。

完成 Step 4-7 后，**别忘了回头清理本文档 + 老 quality-gates 警告头 + prompt.py 文件级注释**——它们都是临时引导，实施完后应该删除避免误导。

---

## 9. 相关文件速查

| 内容 | 路径 |
|---|---|
| 设计文档 | [docs/plans/2026-04-29-ev19-template-paradigm-design.md](../plans/2026-04-29-ev19-template-paradigm-design.md) |
| Review 包入口 | [docs/review-packages/2026-04-29-ev19-templates/README.md](../review-packages/2026-04-29-ev19-templates/README.md) |
| 起草脚本 | [scripts/build_ev19_template_index.py](../../scripts/build_ev19_template_index.py)、[scripts/build_ev19_experiment_drafts.py](../../scripts/build_ev19_experiment_drafts.py) |
| EV19 demodata | [demodata/ev19 templates/](../../demodata/ev19%20templates/) |
| 旧 Gate 0 文档（带警告头） | [packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md](../../packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md) |
| 中间件实现 | [packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py) |
| 范式段 prompt | [packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)（约 L266-L350） |
| 现有 ethoinsight 模板 | [packages/ethoinsight/ethoinsight/templates/shoaling.py](../../packages/ethoinsight/ethoinsight/templates/shoaling.py) |
