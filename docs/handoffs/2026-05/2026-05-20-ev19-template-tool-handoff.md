# 2026-05-20 EV19 模板识别 tool 化 Handoff

> **前置上下文**：承接 [2026-05-20-fst-e2e-dual-bug-fix-handoff.md](2026-05-20-fst-e2e-dual-bug-fix-handoff.md)。
> 上一轮修复了 Bug 1（复合语义 → E2E_FULL）和 Bug 2（多 subject 静默丢弃），
> 但今天的 E2E 验证仍然 FORCED STOP。根因定位为：LoopDetectionMiddleware 硬限 5 和
> 模板识别阶段必须的 read_file 次数（4-5 次）冲突。
>
> **状态**：实现完成，等待手动 PR + E2E 验证。
>
> **分支**：`worktree-fix+ev19-template-identify-tool`（从 dev 分出）

---

## 根因

`identification-decision-tree.md` 规定的模板识别流程要求 lead 在调 `set_experiment_paradigm` 之前自己 `read_file` 读取 4-5 个文件：

```
read_file(_facts.md)              → 1
read_file(identification-decision-tree.md) → 2
read_file(by-experiment/fst.md)   → 3
read_file(用户上传的txt)          → 4
read_file(by-template/PorsoltCylinder.md) → 5 = LoopDetectionMiddleware 硬限
```

Bug 1/2 修复后 subagent 派遣正确，但模板识别发生在派遣**之前**，仍然触发硬限。

## 修复方案

**将模板识别封装为 tool**（参照 `prep_metric_plan` 模式）。

Lead 从"5 次 read_file"变成"1 次 tool call"：

```
lead 调 identify_ev19_template(uploaded_files, user_message)
  │                              (1 tool call, 0 次 read_file)
  ├─ ev19_facts.py（内存，0 IO） → 62 变体 + 范式→模板映射
  ├─ parse_header (Python)      → 列结构检测 AllZones/NoZones
  ├─ Path.read_text (Python)    → by-experiment / by-template 领域知识
  └─ 返回 {"status": "ok"|"ambiguous"|"unknown", ...}
```

Tool 内部的所有文件读取走 Python `Path.read_text()`，**不计入** agent 的 `read_file` 配额。

## 改动清单（6 文件）

| 文件 | 改动 |
|---|---|
| `tools/builtins/identify_ev19_template_tool.py` | **新建**（300 行）— tool 实现 |
| `tools/builtins/__init__.py` | +2 行 — export 新 tool |
| `tools/tools.py` | +2 行 — `BUILTIN_TOOLS` 注册 |
| `skills/.../SKILL.md` | 模板识别场景改为"调 tool"而非"read_file" |
| `skills/.../identification-decision-tree.md` | 简化为 tool 手册，原决策树移除 |
| `lead_agent/prompt.py` | Gate 1 段改为调 `identify_ev19_template` tool |

## Tool 接口

```python
identify_ev19_template(runtime, uploaded_files: list[str], user_message: str) -> dict
```

**返回**：
- `status="ok"`：候选唯一，直接 `set_experiment_paradigm`
- `status="ambiguous"`：2-3 候选，带 `clarification_question`（lead 直接转发给 ask_clarification）
- `status="unknown"`：无法识别，带反问话术

## 测试验证

- ethoinsight 测试：324 passed, 9 failed（均为 worktree 路径 + pre-existing 的 parse fixture 问题）
- backend 测试：2630 passed, 39 failed（均为 pre-existing：provisioner/PVC、stepwise_gate、aio_sandbox 等）
- 工具核心逻辑手动验证：6 个 case 全部通过（paradigm 提取、subject 检测、zone 过滤、候选交叉排除）

## 下次 E2E 验证点

用同一句 "**帮我做大鼠强迫游泳的描述性分析和可视化**" + **上传 2 个 Arena 文件**，验证：

| # | 信号 | 期望值 |
|---|---|---|
| 1 | lead 第一个 non-read tool call | `identify_ev19_template`（不是 read_file） |
| 2 | tool 返回 status | `ok` 或 `ambiguous`（不应 hit FORCED STOP） |
| 3 | read_file count | ≤2 次（仅读 prompt reference，不读 skill 引用文件） |
| 4 | 后续流程 | `set_experiment_paradigm` → `prep_metric_plan` → code-executor → ... |
| 5 | 最终用户消息 | 无 `[FORCED STOP]`，有完整分析报告 |
