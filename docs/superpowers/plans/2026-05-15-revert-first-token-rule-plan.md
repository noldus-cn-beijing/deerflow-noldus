# 回退 FIRST-TOKEN Emoji 规则 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 回退 commit `007fb390` 中加入的"FIRST-TOKEN emoji 规则"——该规则要求 lead 每条用户消息第一个非空白字符必须是预设 emoji 之一。实测在 deepseek 多步推理中不稳定（dogfood 只达成 0→1 emoji，期望 ≥4），且**副作用是 UX 生硬**（强制模板填充而非真过程透明）。本 plan 只回退 emoji 位置强制，**保留**同 commit 的输出宪法 / 违规扫描 / read constitution 改动（它们 G1 验证有效）。

**Architecture:** 纯 prompt 文件改动——只动一份文件 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`。两处精确回退（line 442-462 段的标题 + FIRST-TOKEN 描述 + emoji-必须-第一字符的反例 + line 1094 的 orchestration_guide 第 5 步）。保留宪法相关、违规扫描相关、播报表格本身。

**Tech Stack:** Python 3.12+ (agent backend)、无新增依赖、ruff (line length 240)

**Context links（实现时必读）：**
- 上下文：`docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`（说明 G4 状态：partial，0→1 emoji 未达标）
- 设计讨论：本会话上下文（用户已判定 FIRST-TOKEN 不是合适机制；G4 改走方案 C 前端自动播报，由另一份 plan 落地）
- 要回退的 commit：`git show 007fb390 -- packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- 项目约束：`CLAUDE.md`（TDD 强制 / 中文 commit / line length 240）

**Scope（明确不做）：**
- **不回退** `007fb390` 里的 output-constitution.md 文件创建——它是阶段 3 提前落地的核心，G1 验证有效
- **不回退** subagent prompt 加 "开工前 read constitution"——G1 修复关键机制
- **不回退** L1 handoff schema 加 `constitution_acknowledged` 字段（如有）——同上
- **不回退** lead prompt 第 493-507 行的"输出前强制扫描违规关键词表"——G1 关键防线，独立于 emoji 位置规则
- **不回退** `0. 先读输出宪法`（lead prompt 第 313 行新加的）——这是 lead 自己读宪法的入口
- **不改前端** UI 任何代码——方案 C 的 plan 单独写
- **不改任何测试**——FIRST-TOKEN 规则没带配套测试，回退无需改测
- **不删 line 442-454 的"播报表格"本身**——表格内容（emoji 对应触发动作）是有用的参考资料，作为"建议"保留而非"强制"。只回退"FIRST-TOKEN/必须/位置强制"措辞

**前置假设（执行前用 `git log -1` 验证）：**
- 当前在 `dev` 分支，工作目录 `/home/wangqiuyang/noldus-insight/`
- dev 已和 origin 同步（最新 commit 包含 `854adea0 docs(handoff): G1+G4 修复交接`）
- backend 测试在本基线全绿
- commit `007fb390` 已经入了 dev（用 `git show 007fb390 --stat` 验证）

---

## File Structure

**修改 1 个文件**：
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 两处回退（line 442-462 段 + line 1094 orchestration_guide 第 5 步）

**0 个新文件 / 0 个删除文件**

---

### Task 1: 回退过程透明原则段（line 442-462）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:442-462`

**改动思路**：把段落标题从"强制清单 — 每条消息的第一句话必须是播报"改回中性的"强制清单"；把"FIRST-TOKEN 规则"段（强调第一个非空白字符必须是 emoji）改回中性的"播报建议"措辞；删掉"自检"段；删掉"emoji 必须是第一句话"那条反例。**保留**触发动作 + emoji 对照表本身作为建议参考。

- [ ] **Step 1: 用 Edit 工具精确替换"过程透明原则"段**

文件：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

old_string（完整段落，包含 line 442-462，注意保留前后空行用作锚点）：

```
### 过程透明原则（强制清单 — 每条消息的第一句话必须是播报）

**FIRST-TOKEN 规则**：每次调用 `task`、`bash`、`ask_clarification`、`present_files` 之前，你面向用户的消息**第一个非空白字符必须是以下 emoji 之一**。不允许先说别的内容再补播报。

| 触发动作 | 第一句话模板（必须原样使用） |
|---------|--------------------------|
| 派 code-executor | "🧮 正在计算 N 个 <范式> 指标，预计 30-60 秒..." |
| 派 data-analyst | "🔬 指标已完成，正在请专家解读，预计 1-2 分钟..." |
| 派 report-writer | "📝 解读已完成，正在生成中文研究报告..." |
| 跑 `python -m ethoinsight.parse.dump_headers` | "📂 正在解析 EthoVision 文件结构..." |
| 跑 `python -m ethoinsight.catalog.resolve` | "📋 正在生成指标计划..." |
| ask_clarification 反问 | "⚠️ 我需要先确认一件事..." |
| subagent 完成后下一动作前 | "✅ <subagent> 已完成。<下一步>..." |

**自检**：每条消息发出前，在心里数一遍：这条消息里有没有上面这些 emoji？如果没有但包含 tool call，**立刻补上**再发送。

**反例**（thread 5046a6e6 实际发生的错误，永远不要这样）：
- 派完 code-executor 直接派 data-analyst，中间不向用户说一句 → **违规**
- subagent 完成后不汇报，直接派下一个 → **违规**
- 跑 bash 命令前不解释（用户看到一长串 SandboxAudit 命令但不知道为什么）→ **违规**
- 先说"好的，我来分析数据"再补"🧮 正在计算..."→ **违规**（emoji 必须是第一句话）
```

new_string：

```
### 过程透明原则（强制清单）

每次调用 `task` 派遣 subagent、调用 `bash` 跑 catalog/parse CLI、调用 `ask_clarification` 反问用户、调用 `present_files` 呈现文件之前，**必须**先用 1 条简短中文消息向用户播报状态。

**强制场景清单**（缺一不可）：

| 触发动作 | 推荐播报模板 |
|---------|--------------------------|
| 派 code-executor | "🧮 正在计算 N 个 <范式> 指标，预计 30-60 秒..." |
| 派 data-analyst | "🔬 指标已完成，正在请专家解读，预计 1-2 分钟..." |
| 派 report-writer | "📝 解读已完成，正在生成中文研究报告..." |
| 跑 `python -m ethoinsight.parse.dump_headers` | "📂 正在解析 EthoVision 文件结构..." |
| 跑 `python -m ethoinsight.catalog.resolve` | "📋 正在生成指标计划..." |
| ask_clarification 反问 | "⚠️ 我需要先确认一件事..." |
| subagent 完成后下一动作前 | "✅ <subagent> 已完成。<下一步>..." |

注：前端 UI 会在工具调用事件触发时自动展示状态指示条（独立于本播报，作为机制层兜底）。本段播报是面向用户对话的"自然语言陈述"——按推荐模板写但不必拘泥位置或固定措辞，自然衔接对话流即可。

**反例**（thread 5046a6e6 实际发生的错误，永远不要这样）：
- 派完 code-executor 直接派 data-analyst，中间不向用户说一句
- subagent 完成后不汇报，直接派下一个
- 跑 bash 命令前不解释（用户看到一长串 SandboxAudit 命令但不知道为什么）
```

**关键点**：
- 标题从"强制清单 — 每条消息的第一句话必须是播报"改回 commit `007fb390` 之前的"强制清单"
- 删除"FIRST-TOKEN 规则"段
- 表格列头从"第一句话模板（必须原样使用）"改为"推荐播报模板"
- 删除"自检"段
- 删除最后一条反例"先说'好的，我来分析数据'再补..."
- **新增一句**说明前端 UI 会自动展示状态条作为机制层兜底——这一句是连接方案 C 的语义桥，让 lead 知道"prompt 不强制 emoji 位置，因为有 UI 兜底"
- 表格本身保留作为参考

- [ ] **Step 2: 用 Edit 工具精确删除 orchestration_guide 第 5 步**

文件：同上

old_string（line 1086-1095 段，含上下文锚点）：

```
1. **立即调用**: `read_file("/mnt/skills/ethoinsight-planning/SKILL.md")`
2. **遵循 6 步规划流程**: 意图分类 → 完整性检查 → 选模板 → 质量门控 → 单行摘要 → 执行
3. **仅两种情况必须反问用户**:
   - 范式推断失败（文件名看不出范式）
   - 分组无法推断（无命名规律且用户未明示）
   - 其他情况走默认，**不要过度反问**
4. **输出单行计划给用户**（格式：`将对 <范式> 数据执行 <操作>，约 X 分钟`）
5. **每条用户消息的第一个 token 必须是阶段播报 emoji**：🧮/🔬/📝/📂/📋/⚠️/✅ 之一（详见本文档「过程透明原则」段），不允许先寒暄再补播报
6. **执行时遵循本文档后续的派遣流程**
```

new_string：

```
1. **立即调用**: `read_file("/mnt/skills/ethoinsight-planning/SKILL.md")`
2. **遵循 6 步规划流程**: 意图分类 → 完整性检查 → 选模板 → 质量门控 → 单行摘要 → 执行
3. **仅两种情况必须反问用户**:
   - 范式推断失败（文件名看不出范式）
   - 分组无法推断（无命名规律且用户未明示）
   - 其他情况走默认，**不要过度反问**
4. **输出单行计划给用户**（格式：`将对 <范式> 数据执行 <操作>，约 X 分钟`）
5. **执行时遵循本文档后续的派遣流程**（含「过程透明原则」段的播报建议）
```

**关键点**：删除原第 5 步（FIRST-TOKEN emoji 强制），把原第 6 步降为新第 5 步并合并指引。

- [ ] **Step 3: ruff lint 验证**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint
```

Expected: 无错误。如有错误（极不可能，因为只动 docstring 文本），按 ruff 提示修。

- [ ] **Step 4: 跑 backend 测试确认无回归**

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test
```

Expected: 全绿（除已知 6 个预存失败 auth/live/skill 与本 plan 无关）。**重点关注**：
- `test_lead_prompt_role_boundaries.py`（4 个，应该都过——它测的是 line 309-340 的角色边界硬约束，与 FIRST-TOKEN 规则无关）
- 任何匹配 `*lead_prompt*` / `*orchestration*` 的测试

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "$(cat <<'EOF'
revert: 回退 FIRST-TOKEN emoji 规则（commit 007fb390 G4 部分）

dogfood 实测 FIRST-TOKEN 规则在 deepseek 多步推理中不稳定（0→1 emoji，未达 ≥4），
且副作用是 UX 生硬——强制 emoji 作为消息第一个非空白字符让 lead 退化为模板填充
而非真过程透明。

本次回退：
- "过程透明原则"段标题从"强制清单 — 每条消息的第一句话必须是播报"改回中性"强制清单"
- 删除 FIRST-TOKEN 描述段（"第一个非空白字符必须是 emoji 之一"）
- 删除"自检：在心里数一遍"段
- 删除"先说寒暄再补播报 = 违规"反例
- 删除 orchestration_guide 第 5 步"每条用户消息的第一个 token 必须是阶段播报 emoji"
- 表格列头从"必须原样使用"改回"推荐模板"
- 加一句说明前端 UI 会展示状态条作为机制层兜底

保留：
- output-constitution.md（G1 修复核心，单独验证有效）
- subagent prompt 加"开工前 read constitution"（同上）
- L1 handoff schema 的 constitution_acknowledged 字段（如有）
- lead prompt 第 493-507 行违规关键词强制扫描（G1 关键防线，与 emoji 位置无关）
- 第 313 行"0. 先读输出宪法"指引

G4 真正修法走方案 C：前端在 tool_call event 时自动渲染状态条
（独立 plan：docs/superpowers/plans/2026-05-15-frontend-stage-broadcast-plan.md）
EOF
)"
```

- [ ] **Step 6: Push**

```bash
cd /home/wangqiuyang/noldus-insight
git push origin dev 2>&1 | tail -5
```

Expected: `<旧hash>..<新hash>  dev -> dev`，1 个 commit 入 origin。

---

## 验收（执行 agent 自检）

回退完成后，再次 `git show 007fb390 -- packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 看保留 vs 回退的对应关系：

| commit `007fb390` 中的改动 | 本次状态 |
|---|---|
| `0. 先读输出宪法 — read_file output-constitution.md` 加在角色边界 | **保留** ✅ |
| "过程透明原则"标题加"每条消息的第一句话必须是播报" | **回退** ❌ |
| "FIRST-TOKEN 规则"段（第一个字符必须 emoji）| **回退** ❌ |
| 表格列头改"必须原样使用" | **回退** ❌ |
| "自检"段 | **回退** ❌ |
| 反例加"先说寒暄再补播报 = 违规" | **回退** ❌ |
| "输出前强制扫描违规关键词表" | **保留** ✅（G1 防线）|
| orchestration_guide 第 5 步"FIRST-TOKEN" | **回退** ❌ |

回退后 lead prompt 比 `007fb390` 之前的版本**多出**：宪法 read + 违规扫描；**等同**：播报表格 + 强制场景清单（仅措辞从"FIRST-TOKEN"软化）。

---

## 不要做的事（防止越权）

- ❌ **不要修前端代码**——方案 C 单独的 plan
- ❌ **不要删 output-constitution.md** 或 subagent prompt 里的 "read constitution" 指引——G1 修复关键
- ❌ **不要删第 493-507 行违规扫描表**——G1 防线
- ❌ **不要 force push** —— 1 commit 直接推 push origin dev
- ❌ **不要用 `--no-verify`**
- ❌ **不要碰这 3 个无关文件**：
  - `docs/specs/llm-finetuning-strategy.md`
  - `docs/plans/2026-05-13-base-model-decision-memo.md`
  - `packages/agent/frontend/src/app/page.tsx`

---

## 实施完成后的状态

- 1 个 commit 入 origin
- lead prompt 中 FIRST-TOKEN emoji 位置强制规则**全部回退**
- 输出宪法 / 违规扫描 / read constitution 改动**全部保留**
- G4 修复路径明确转向方案 C（前端 tool_call 自动播报），独立 plan 处理
- backend 测试全绿
- 用户感知：lead 输出消息更自然，不再硬性 emoji 开头
