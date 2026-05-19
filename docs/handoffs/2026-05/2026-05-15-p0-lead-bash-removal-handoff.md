# 2026-05-15 P0 lead-retry-blocked-bash 修复 + Plan A/B sync 暂停 交接

> **状态**：P0 bug 诊断完成,修复方向已经经过充分 grill 拍板,但**实施 plan 没写出来**——本会话写 plan 时反复卡住(详见"会话状态"段)。新会话第一件事:写 plan,然后派执行 agent。

---

## 当前任务目标(给下一个 agent)

**第一优先**:把已经 grill 完毕的 P0 修复方案落成完整可执行 plan,保存到
`docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md`。

**第二优先**(plan 通过用户 review 后):派执行 agent 实施 plan,在专用 worktree
跑,完成后回报。

**第三优先**(P0 修复 + dogfood 通过后):派 Plan A 上游 deerflow sync(11 个安全
commit,plan 已在 `docs/superpowers/plans/2026-05-15-deerflow-upstream-sync-plan-A-safe-batch.md`)。
之后 Plan B(同目录 plan-B-...md)。

---

## 当前进展

### ✅ 已完成

1. **G4 方案 C 前端 stage broadcast** — 6 commits 已合 origin/dev (HEAD `3ee6bd07`)
2. **Spec 阶段 1 双层 handoff 协议** — 12 commits push 到分支
   `worktree-spec-phase-1-handoff`(tip `adbf3107`),含:
   - 10 个核心 commit(`2115ce53..092500a6`):catalog/projection.py、summarize CLI、
     task_tool handoff_suffix、HandoffIsolationProvider、4 个 subagent L1+L2、lead prompt
   - 1 个 symlink 修复 commit(`840b6551`):恢复 `packages/agent/backend/packages/ethoinsight`
     被 Task 1 误删的 symlink(历史上 `73237356` 已修过同问题,这次是第二次)
   - 1 个 live dogfood 验证 commit(`adbf3107`):5/7 §9 验收通过,§9.7 部分通过(暴露 P0)
3. **deerflow sync Plan A** 已写完(11 个安全 commit,workspace 隔离方案,base 切换说明)
4. **deerflow sync Plan B** 已写完(剩余 13 个 commit,Phase 1 DynamicContextMiddleware 链 +
   Phase 2 9 个独立 + 4 个 SKIP)
5. **P0 bug 完整诊断** — 见下文"关键发现 §A"。两层根因 + 修复方向已 grill 拍板。

### 🔴 进行中 / 阻塞

- **P0 修复 plan 未写出来**(本会话失败) — 见"会话状态"段
- **P0 修复 dogfood 未通过** → 阻塞:Spec 阶段 1 不能合 dev;Plan A 不能派
- **Spec 阶段 1 merge dev** 暂停,直到 P0 修复完成
- **Plan A / Plan B** 暂停,直到 P0 修复完成 + Spec 阶段 1 合 dev

---

## 关键上下文

### A. 项目状态

- 仓库根:`/home/wangqiuyang/noldus-insight/`
- 主分支:`dev`,HEAD `3ee6bd07`(merge G4 方案 C)
- 本地 dev 可能仍在 `fca62e33`(本会话没跑 git pull 同步 origin/dev),需要 `git pull`
- 项目愿景:EthoInsight 行为学 AI 分析助手,9 月 v0.1
- 详细架构:`CLAUDE.md`(项目根)+ `packages/agent/backend/CLAUDE.md`

### B. 关键 worktree 状态(`git worktree list`)

```
/home/wangqiuyang/noldus-insight                                                     主 dev 工作树
/home/wangqiuyang/noldus-insight/.claude/worktrees/feat+g4-frontend-stage-broadcast  G4 已 push 已合
/home/wangqiuyang/noldus-insight/.claude/worktrees/spec-phase-1-handoff              Spec 阶段 1 等修复
```

主工作树有 4 个 M / 多个 untracked 文件,交接文档明令"不要碰":
- M `docs/specs/llm-finetuning-strategy.md`
- M `packages/agent/frontend/src/app/page.tsx`
- M `packages/agent/scripts/serve.sh`
- M `packages/agent/skills/public/bootstrap/SKILL.md`

### C. 关键文件路径速查

| 用途 | 路径 |
|---|---|
| 已写的 Plan A | `docs/superpowers/plans/2026-05-15-deerflow-upstream-sync-plan-A-safe-batch.md` |
| 已写的 Plan B | `docs/superpowers/plans/2026-05-15-deerflow-upstream-sync-plan-B-dynamic-context-and-misc.md` |
| **本会话要写但没写出的 plan**(P0 修复) | `docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md`(**不存在,要新建**) |
| Spec 阶段 1 分支信息 | `docs/handoffs/2026-05/2026-05-15-spec-phase-1-and-g4-dual-track-handoff.md` |
| dogfood 报告(暴露 P0) | `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` (line 196 起 "2026-05-15 spec 阶段 1 + G4 方案 C live dogfood 验证") |
| differential test 报告(证伪 prompt 假设) | `docs/problems/2026-05-15-*.md`(Spec 阶段 1 agent 写的) |
| **G4 boundary 源码**(要删) | `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py` |
| **lead_agent 入口**(要改) | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` line 438 附近 |
| **lead prompt**(要改) | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` line 453-454 + line 1107-1123 |
| **LoopDetectionMiddleware**(要改) | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py` |
| **tool registry**(要加新 tool) | `packages/agent/backend/packages/harness/deerflow/tools/tools.py` line 14-18 `BUILTIN_TOOLS` |
| **ethoinsight parse 函数**(prep_metric_plan 内部要调) | `packages/ethoinsight/ethoinsight/parse/_core.py` (`detect_ethovision` / `parse_header` / `parse_trajectory`) |
| **ethoinsight resolve 函数**(同上) | `packages/ethoinsight/ethoinsight/catalog/resolve.py` (`resolve()` / `plan_to_dict()` / `ResolveError`) |
| **lead 实际 dogfood log** | `packages/agent/logs/langgraph.log`(thread `b8f48b43-c7a5-4ef7-a433-7a0ef197df6e`) |

### D. 关键 commit 速查

| Commit | 内容 |
|---|---|
| `3ee6bd07` | merge G4 方案 C(origin/dev HEAD)|
| `840b6551` | Spec 阶段 1 symlink 修复 |
| `adbf3107` | Spec 阶段 1 live dogfood 报告 |
| `092500a6` | Spec 阶段 1 双层 handoff 代码级验证 |
| `0c83bf9b` | Spec 阶段 1 lead prompt 改造(handoff_suffix 必填)|
| `cdf8b268` | Spec 阶段 1 catalog/projection.py — 该 commit 误删 symlink,被 840b6551 修 |

### E. 关键 4 条用户约束(不能违反)

1. **不要 push 任何东西**——除非用户明确说"push"
2. **不要 commit 未明确授权的文件**——尤其上面 4 个 M 文件
3. **本会话 review 角色**:Plan 写完 + 派 agent。不直接动代码
4. **不要打断 worktree 里跑的 agent**——目前 Spec 阶段 1 + G4 都已 push,不再有跑中的 agent

---

## 关键发现

### A. P0 bug 根因(双层,经过完整 grill)

**现象**:用户 dogfood EPM 单只描述性分析,lead 调
`cd /mnt/user-data/workspace && python -m ethoinsight.parse.dump_headers <路径> --output columns.json 2>&1`
被 G4 LeadAgentExecutionBoundaryProvider 拒,然后 lead 在 thinking 里反复尝试不同
quoting / 引号 / 路径写法直到 recursion limit=100 耗尽。

**触发层根因**:`guardrails/lead_execution_boundary_provider.py:36-38` 的正则
`^\s*python3?\s+-m\s+ethoinsight\.(parse|catalog)\.\w+(\s|$)` 只认裸 python 开头,
lead 写 `cd <dir> && python -m ethoinsight.parse.*` 因开头是 `cd` 被拒。
(文件路径含中文 + 多个空格 `Trial     1`,lead 倾向用 `cd && python` 避免 quoting。)

**放大层根因**:`LoopDetectionMiddleware` 已在 `lead_agent/agent.py:245+293` 注册,
但 lead 每次 retry 微调 command 字符串,tool_call hash 每次不同,中间件没识别为
"重复"。`_DEFAULT_TOOL_FREQ_HARD_LIMIT=50`(同 tool name 不看 args 50 次强制停)
理论上该触发但实际没触发——说明 tool_freq 计数实现要看。

**已经证伪的假设**:Spec 阶段 1 改 lead prompt 62 行(加 handoff_suffix 约束)。
经 differential test(将 prompt 回退到 spec 阶段 1 之前的 0acfd46b 跑同样 dogfood),
lead 仍反复重试。说明 prompt 改动**不是诱因**。

**G4 方案 C 那次 dogfood "成功 dispatch 2 次"是误读**——那次 lead 没尝试自己 bash,
直接派 code-executor。两次 dogfood 不构成 differential 对比。

### B. 已经 grill 拍板的修复方案(7 个细节决策已确认)

第一性原理:**用户感知到的快(不是结果快)+ 科研可复现性**。

核心:**lead 不再有 bash tool**。所有 lead 现在通过 bash 干的事(`parse.*`、
`catalog.*`、ls 验证产物)全部包成 deerflow 注册的 Python 工具。**subagent 保留 bash**
(它们用 `tools=[...]` 显式声明,跟 lead 的工具列表分开)。

#### 7 个 grill 确认细节

| Q | 决策 | 理由 |
|---|---|---|
| Q1 prep_metric_plan 一个 vs 两个工具 | **一个**(`prep_metric_plan`)| Step 0.5 两步本连续,合一 = 一次 tool_call = 用户感知更快 |
| Q2 返回 columns 颗粒度 | **2C 不返回 columns,只返回 columns_path** | lead 决策只依赖 plan_summary,columns 是 catalog 内部输入,节 context |
| Q3 错误处理协议 | **3B 结构化错误**({"status": "error", "error_code", "message", "hint"}) | 给 deepseek 看 stack trace 会触发 retry quoting,跟现 bug 一样;hint 字段直接引导下一步 |
| Q4 list_workspace_artifacts | **不写,信 code-executor handoff status + 必要时 read_file** | spec 阶段 1 双层 handoff 已有 L1 信号,二次 ls 冗余,少一个工具 = lead 工具列表更短 |
| Q5 lead 是否保留 read_file/write_file | **5A read_file 保留 + write_file/str_replace 删** | write_file 留着是另类漂移源(deepseek 可能自己 write plan.json 绕开工具) |
| Q6 LoopDetectionMiddleware 阈值 | **warn=3, hard=5 同 tool name 不看 args** | 默认 50 太宽松;改成按 tool name 计数(不看 args)避免 hash 漂移问题 |
| Q7 dogfood 验收 | **7B 主标准 + 7C 加分**:lead 能派 code-executor 就算过(code-executor 内部 sandbox bug 不归本 fix);加分:LoopDetection 没触发 | 全链路通过会无限延期,聚焦本 fix 的根因 |

### C. lead 实际 bash 使用现场(grep langgraph.log)

唯一一个**合法 bash 用例**:`ls -la /mnt/user-data/workspace/m_*.json`(lead 验证
code-executor 产物完整性)。Q4 决策不为它写新工具——信 handoff status 就行。

### D. deerflow 已有但没用的现成能力(CLAUDE.md 第 12 条复用原则)

- **LoopDetectionMiddleware** — 已注册,但需要改计数逻辑(按 tool name)
- **ClarificationMiddleware** — 已用,作为新工具 pattern 参考
- **DeferredToolFilterMiddleware** — 没用,潜在能力:动态删 lead 工具
- `@tool` 装饰器 — 写新 Python tool 的标准方式

### E. tool registry 注册机制

`packages/agent/backend/packages/harness/deerflow/tools/tools.py:14-18`:

```python
BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
    set_experiment_paradigm_tool,
]
```

新工具加这里。lead 通过 `get_available_tools()` 拿所有 `BUILTIN_TOOLS` + config.yaml
`tools[]`。**filtering for lead** 走 lead_agent/agent.py `tools=` 参数:

```python
# agent.py:438 现状:
tools=get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled),

# 改为:
_LEAD_EXCLUDED_TOOLS = {"bash", "write_file", "str_replace"}
all_lead_tools = get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled)
tools=[t for t in all_lead_tools if t.name not in _LEAD_EXCLUDED_TOOLS],
```

subagent 不受影响(它们用 `tools=[...]` 显式声明)。

### F. ethoinsight 函数签名(prep_metric_plan 内部要用)

```python
# parse:
from ethoinsight.parse._core import detect_ethovision, parse_header, parse_trajectory
# detect_ethovision(file_path: str) -> bool
# parse_header(file_path: str) -> dict  # 抛异常 = header_parse_failed
# parse_trajectory(file_path: str) -> pd.DataFrame  # 列名 = df.columns

# catalog:
from ethoinsight.catalog.resolve import resolve, plan_to_dict, ResolveError
# resolve(paradigm, columns, raw_files, workspace_dir, *,
#         virtual_workspace_dir=None, ev19_template=None, ...) -> Plan
# 抛 ResolveError(code, message, details);code 枚举:
#   unknown_paradigm / columns_missing / schema_violation / unknown_metric
# plan_to_dict(plan) -> dict
```

### G. 现状 lead Step 0.5 prompt(要改的内容)

`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`:

- line 453-454(transparency 表):
  ```
  跑 `python -m ethoinsight.parse.dump_headers` → "📂 正在解析 EthoVision 文件结构..."
  跑 `python -m ethoinsight.catalog.resolve`    → "📋 正在生成指标计划..."
  ```
  改成:
  ```
  调 `prep_metric_plan(...)` → "📋 正在生成指标计划..."
  ```

- line 1107(skill 调用段):删除"bash dump_headers" + "bash catalog.resolve",改为"调
  prep_metric_plan tool"

- line 1119-1123(Step 0.5 完整段):
  ```
  ### Step 0.5: 生成 metric_plan.json（**派遣 code-executor 前必做**）
  1. bash dump_headers 提取数据列名到 /mnt/user-data/workspace/columns.json
  2. write raw_files.json
  3. bash catalog.resolve 生成 /mnt/user-data/workspace/metric_plan.json
  ```
  改成:
  ```
  ### Step 0.5: 生成 metric_plan.json（**派遣 code-executor 前必做**）
  调 `prep_metric_plan(uploaded_file=<path>, paradigm=<id>)` 一步完成。
  内部直接调 Python 函数,无需 bash。返回 status="ok" 时 plan_summary 含 metric_count
  + metric_ids,继续派 code-executor;status="error" 时按 hint 字段建议处理。
  ```

### H. CLAUDE.md 教训(本次再次复现的)

- 第 6 条:"deepseek 不要用'禁止 X',会反向激活"
- 第 12 条:"自写中间件之前先看 agents/middlewares/ 和 tools/builtins/ 目录有没有现成的"
- 第 18 条 / 风险段:"单测和生产路径脱节"——G5 教训,本 P0 是第 N 次类似复现
- 陷阱 2 / 风险段:"prompt 自觉 → 机制层"——G1/G4/G5 都验证 prompt-only 不可靠

---

## 未完成事项(优先级排序)

### 🔴 P0 — 阻塞所有后续工作

**1. 写 P0 修复实施 plan**

保存到 `docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md`。

使用 `superpowers:writing-plans` skill,严格按它的模板。**Plan 必须含 4 个 task**:

#### Task 1:修 LoopDetectionMiddleware 计数(按 tool name 不看 args)

文件:`packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`

诊断步骤:
- 跑 `grep -i "LoopDetection" packages/agent/logs/langgraph.log` 看实际行为
- 读源码看 tool_freq 怎么计数(hash 还是 name)
- 看为什么 100 次重试没触发 `_DEFAULT_TOOL_FREQ_HARD_LIMIT=50`

修复:
- 加新 counter 只看 tool name 不看 args
- 阈值 warn=3, hard_limit=5
- warn 时注入 system message:"你已经 3 次调 bash 但都被拒,如果是分析任务请改用
  task(code-executor)。再尝试会被强制停止。"
- hard 时 strip 所有 tool_calls 强制 lead 出文字答复

TDD 测试(`tests/test_loop_detection_middleware.py` 加新用例):
- 同 tool name 3 次不同 args → warn
- 同 tool name 5 次不同 args → hard limit + strip
- 不同 tool name 各 4 次 → 不触发
- 同 bash 2 次 + 切别的 1 次 + 同 bash 再 1 次 → counter 不重置(因为我们要的就是"
  这个 tool 不该这么用")

单独 commit,**可独立止血**(即使 Task 2-4 卡住,这一个合 dev 就让 recursion 100 不再发生)。

#### Task 2:新建 `prep_metric_plan` Python 工具

文件:`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`(新建)

工具签名(完整代码,plan 里展开):

```python
"""prep_metric_plan tool — lead 一步生成 metric_plan.json,无需 bash。"""

import json
import logging
from pathlib import Path
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from ethoinsight.parse._core import detect_ethovision, parse_header, parse_trajectory
from ethoinsight.catalog.resolve import resolve, plan_to_dict, ResolveError

logger = logging.getLogger(__name__)


@tool("prep_metric_plan", parse_docstring=True)
def prep_metric_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    uploaded_file: str,
    paradigm: str,
) -> dict:
    """一步生成 metric_plan.json,无需 bash。

    Args:
      uploaded_file: 虚拟路径如 /mnt/user-data/uploads/xxx.txt
      paradigm: 范式 id 如 'epm' / 'oft' / 'fst' / 'ldb' / 'tst' / 'zero_maze' / 'shoaling'

    Returns:
      status="ok" 时:
        {"status": "ok",
         "plan_path": "/mnt/user-data/workspace/metric_plan.json",
         "plan_summary": {
             "paradigm": "epm",
             "metric_count": 5,
             "metric_ids": ["open_arm_time_ratio", ...]
         }}
      status="error" 时:
        {"status": "error",
         "error_code": "file_not_found" | "format_unrecognized" | "parse_failed"
                     | "unknown_paradigm" | "columns_missing" | "schema_violation",
         "message": str,
         "hint": str}
    """
    # 1. 虚拟路径 → 真实路径
    state = runtime.state
    sandbox = state.get("sandbox")
    if sandbox is None:
        return {"status": "error", "error_code": "no_sandbox",
                "message": "Sandbox not initialized", "hint": "等服务重启后再试"}
    real_path = sandbox.replace_virtual_path(uploaded_file)

    # 2. 检查文件存在
    if not Path(real_path).is_file():
        return {"status": "error", "error_code": "file_not_found",
                "message": f"文件不存在: {uploaded_file}",
                "hint": "用 ask_clarification 让用户重新上传数据文件"}

    # 3. 检查 EthoVision 格式
    if not detect_ethovision(real_path):
        return {"status": "error", "error_code": "format_unrecognized",
                "message": f"文件不是 EthoVision XT 导出格式: {uploaded_file}",
                "hint": "用 ask_clarification 让用户确认是从 EthoVision XT 导出的轨迹文件"}

    # 4. 解析列名
    try:
        df = parse_trajectory(real_path)
        columns = list(df.columns)
    except Exception as e:
        return {"status": "error", "error_code": "parse_failed",
                "message": f"无法解析数据文件:{e}",
                "hint": "数据文件可能损坏,用 ask_clarification 让用户重新导出"}

    # 5. 用 catalog.resolve 生成 plan
    real_workspace_dir = sandbox.replace_virtual_path("/mnt/user-data/workspace")
    try:
        plan = resolve(
            paradigm=paradigm,
            columns=columns,
            raw_files=[real_path],
            workspace_dir=real_workspace_dir,
            virtual_workspace_dir="/mnt/user-data/workspace",
        )
    except ResolveError as e:
        hints = {
            "unknown_paradigm": f"范式 '{paradigm}' 不在 catalog 内,用 ask_clarification 让用户确认范式",
            "columns_missing": "数据缺关键列(可能录制设置漏了),用 ask_clarification 让用户确认实验录制设置",
            "schema_violation": "catalog YAML 损坏 — 这是项目内部 bug,用 present_files 呈现错误让用户报 bug",
            "unknown_metric": "请求的指标不存在,用 ask_clarification 让用户确认指标列表",
        }
        return {"status": "error", "error_code": e.code,
                "message": str(e.message),
                "hint": hints.get(e.code, "请用 ask_clarification 反问用户")}

    # 6. 写 plan.json 到 workspace
    plan_dict = plan_to_dict(plan)
    real_plan_path = Path(real_workspace_dir) / "metric_plan.json"
    real_plan_path.parent.mkdir(parents=True, exist_ok=True)
    real_plan_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. 返回摘要(不返回完整 plan)
    return {
        "status": "ok",
        "plan_path": "/mnt/user-data/workspace/metric_plan.json",
        "plan_summary": {
            "paradigm": plan.paradigm,
            "metric_count": len(plan.metrics),
            "metric_ids": [m.id for m in plan.metrics],
        },
    }
```

注册步骤:
- 在 `packages/agent/backend/packages/harness/deerflow/tools/builtins/__init__.py` 导出
- 在 `packages/agent/backend/packages/harness/deerflow/tools/tools.py` 的 `BUILTIN_TOOLS` 加入(line 14)
- 不需要 config.yaml 改动(BUILTIN_TOOLS 自动给 lead)

TDD 测试(`packages/agent/backend/tests/test_prep_metric_plan_tool.py` 新建):
1. 正常路径:mock EthoVision 数据(用 conftest fixture),paradigm=epm → status=ok, metric_count>0
2. file_not_found
3. format_unrecognized:mock 非 EthoVision 文件
4. unknown_paradigm:paradigm="invalid"
5. columns_missing:mock 数据缺 in_zone_open_arms_* 列,paradigm=epm
6. 写盘验证:status=ok 后 plan_path 真实存在 + JSON 可读 + 含 metric_ids

单独 commit。

#### Task 3:从 lead 工具列表移除 bash + write_file + str_replace

文件:`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` line 438 附近

完整代码 diff(plan 里展开):

```python
# 现状 line ~438:
tools=get_available_tools(
    model_name=model_name,
    groups=lead_tool_groups,
    subagent_enabled=subagent_enabled,
)

# 改为:
_LEAD_EXCLUDED_TOOLS = {"bash", "write_file", "str_replace"}
all_lead_tools = get_available_tools(
    model_name=model_name,
    groups=lead_tool_groups,
    subagent_enabled=subagent_enabled,
)
filtered_lead_tools = [t for t in all_lead_tools if t.name not in _LEAD_EXCLUDED_TOOLS]
# 上面那个调用改为:
tools=filtered_lead_tools,
```

TDD 测试(`packages/agent/backend/tests/test_lead_tool_filtering.py` 新建):
1. lead get tools → bash 不在
2. write_file 不在
3. str_replace 不在
4. ls 在(保留)
5. read_file 在
6. prep_metric_plan 在
7. task / ask_clarification / present_files 在
8. code_executor.tools 仍含 bash(验证不影响 subagent)

单独 commit。

#### Task 4:删 G4 boundary + 改 prompt + 改 skill + dogfood 验证

拆 4a/4b/4c/4d 子任务,每个单独 commit。

**4a 删除 G4 boundary**:
- 删 `packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py`
- 删 agent.py 中所有 `LeadAgentExecutionBoundaryProvider` import + 注册(grep 找)
- 删配套 test:`packages/agent/backend/tests/test_lead_execution_boundary_provider.py`
  (如存在,grep 找)
- 改 `packages/agent/backend/packages/harness/deerflow/guardrails/__init__.py` 移除 export

**4b 改 lead prompt** `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`:
- line 453-454:transparency 表 bash dump_headers + bash catalog.resolve 行
  改为"调 prep_metric_plan → '📋 正在生成指标计划...'"
- line 1107:ethoinsight-metric-catalog skill 段说明改成"通过 prep_metric_plan tool"
- line 1119-1123:Step 0.5 完整段重写为"调 prep_metric_plan(uploaded_file=, paradigm=)
  一步完成"。具体新文本见上面"关键发现 §G"

**4c 改 skill 文档** `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`:
- 找 lead role 段(grep "lead" 或 "bash dump_headers"),把"bash dump_headers +
  bash catalog.resolve"改成"调 prep_metric_plan tool"
- 加详细 error_code 字段说明 + hint 用法

**4d dogfood 验证**:
- 建 worktree `git worktree add .claude/worktrees/p0-fix-test -b p0-lead-bash-removal dev`
- `cd packages/agent && make stop && make dev`
- 跑跟 P0 现场完全一样的 dogfood:上传 EPM 数据 → "请分析这份 EPM 数据" →
  "做单样本分析"
- 验证 7B+7C:
  * lead 调 prep_metric_plan 成功 → 返回 plan_summary
  * lead 派 code-executor 成功(不卡 boundary)
  * 没有 recursion 耗尽
  * (7C 加分项)`grep -i LoopDetection packages/agent/logs/langgraph.log` 没有触发
- 写 dogfood 报告 `docs/handoffs/2026-05/2026-05-15-p0-fix-dogfood-validation.md`
- 失败 → 单独 revert task 1/2/3 调查,不瞎合

**plan 的全局准备段**(放最前):

```
## Step P.1:建专用 worktree

cd /home/wangqiuyang/noldus-insight
git fetch origin
git pull origin dev  # 同步 G4 方案 C
git worktree add .claude/worktrees/p0-lead-bash-removal -b p0-lead-bash-removal dev

## Step P.2:跑基线测试
cd packages/agent/backend && source .venv/bin/activate
make test 2>&1 | tail -20  # 期望大致全绿(2315 passed / 3 failed 全预存)
make lint 2>&1 | tail -10

## Step P.3:跑 P0 复现验证(确认能复现 bug,这样后面 dogfood 通过才有意义)
cd ../  # 回 packages/agent
make stop && make dev
# 用 DeerFlowClient 跑 EPM dogfood,确认能复现 recursion 100 次
```

写完 plan 之后执行 self-review:
- 无 placeholder
- 类型一致性(prep_metric_plan 返回字段在 task 4b 引用处一致)
- 4 个 task 完整覆盖 P0 触发层 + 放大层

**Plan 完成后**:
1. commit plan 文件到 dev 主工作树(不要在 worktree commit)
2. 不要 push,等用户授权
3. 给用户回报 plan 完成 + 问要不要派执行 agent

### 🟡 第二优先:P0 修复 plan 通过用户 review 后,派执行 agent 实施

派 prompt 模板见下面"建议接手路径"段。

### 🟢 第三优先:P0 修复完成 + 合 dev 后

1. **Spec 阶段 1 合 dev**:把 `worktree-spec-phase-1-handoff` (HEAD `adbf3107`)
   合进 dev(可能需要 rebase 把 P0 fix 包含进去再合)
2. **派 Plan A** deerflow sync(11 个安全 commit)
3. Plan A 通过后派 Plan B(13 个剩余 commit)
4. 上面三步都搞完 → 整个 sync + spec 阶段 1 + P0 fix 完整闭环

---

## 建议接手路径

### 第一步:验证现状(2 分钟)

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin
git log -1 --format='%H %s' origin/dev    # 期望:3ee6bd07
git status --short                          # 看主工作树 dirty 状态(4 个 M 文件预期存在)
git worktree list                           # 看 worktree 状态
git log -1 --format='%H %s' worktree-spec-phase-1-handoff  # 期望:adbf3107
```

### 第二步:读关键文件确认上下文

```bash
# 项目根 CLAUDE.md
head -100 CLAUDE.md

# backend CLAUDE.md(架构 + harness/app 边界)
head -100 packages/agent/backend/CLAUDE.md

# 本交接文档(就你在读的这个)
cat docs/handoffs/2026-05/2026-05-15-p0-lead-bash-removal-handoff.md

# dogfood 报告(P0 现场)
grep -A 80 "spec 阶段 1.*live dogfood" docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md

# differential test 报告
ls docs/problems/2026-05-15-*.md
```

### 第三步:写 P0 修复 plan

调用 `superpowers:writing-plans` skill,严格按它模板。**Plan 内容已经在本交接的
"未完成事项 §P0" 段完整给出**——按那个内容展开成 plan 即可。

**特别注意**:不要重新 grill,所有 7 个细节决策已经拍板(本交接"关键发现 §B" 表格)。
不要重新做差分实验,根因诊断已完成(本交接"关键发现 §A")。

**Plan 文件路径**:
`docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md`

写完 plan 后:
1. 在主工作树 `git add docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md`
2. `git commit -m "docs: P0 lead bash removal 实施 plan"`(不 push)
3. 给用户回报"plan 已写完,问要不要派执行 agent"

### 第四步(plan 通过后):派执行 agent

执行 agent 的 prompt 模板(直接复制给用户用):

```
你是执行 P0 lead bash removal 修复的 agent。在 noldus-insight 仓库工作。

# 你的任务

按 docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md
完整执行 4 个 task。

# 必读资料(开工前)

1. docs/handoffs/2026-05/2026-05-15-p0-lead-bash-removal-handoff.md(根因 + 决策)
2. docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md(执行手册)
3. CLAUDE.md 项目根 第 6/12/18 条 + 风险段陷阱 1/2
4. packages/agent/backend/CLAUDE.md 架构 + middleware 链段

# 执行 skill

使用 superpowers:executing-plans 或 superpowers:subagent-driven-development。

# 硬约束

1. 必须在专用 worktree 跑(plan Step P.1 命令)
2. 不要 push 任何东西
3. 不要碰这 4 个文件:llm-finetuning-strategy.md / page.tsx / serve.sh /
   bootstrap/SKILL.md
4. Task 1 单独 commit(独立止血,即使后面卡住也合 dev)
5. TDD 强制:每个 task 先写 failing test 再实现
6. 遇到决策点立即 STOP,写进 progress 表的备注列,回报用户
7. dogfood 验证不通过:revert 单 task,不瞎合

# 完成标准

1. 4 个 task 全部 commit 落 p0-lead-bash-removal 分支
2. make test + make lint 全绿
3. dogfood 端到端通过(Q7 验收 7B+7C)
4. progress 文档写完
5. 回报用户:实际 task 数、commit hash 列表、dogfood 通过情况、等用户授权 merge

# 工作守则

- 你的改动只在 p0-lead-bash-removal 分支
- 每完成一个 task 回报一句话:"Task N 已完成,commit X"
- 卡住:STOP,回报,等回复

现在开始。先做 Step P.1-P.3 全局准备,然后从 Task 1 开始按顺序跑。
```

---

## 风险与注意事项

### 🚨 不要做的事

1. **不要重新诊断 P0**——根因已经确认(双层),不要再跑 differential test
2. **不要重新 grill 修复方案**——7 个细节已经拍板,直接按 grill 结果写 plan
3. **不要 force push / git reset --hard**——会丢 Spec 阶段 1 / G4 的工作
4. **不要在主工作树跑修复**——会跟那 4 个 M 文件混在一起
5. **不要修 P0 时把 Spec 阶段 1 也一起合 dev**——两件事解耦,各自走自己的合并流程
6. **不要绕开 LoopDetectionMiddleware 修改直接删它**——它是兜底机制,要修不要删

### ⚠ 容易混淆的点

#### 1. "G4 方案 C" vs "G4 boundary"

- **G4 方案 C** = 前端 stage broadcast,**已合 dev**(3ee6bd07)
- **G4 boundary** = `LeadAgentExecutionBoundaryProvider`(本次要删的)

两个完全不同的东西,review 时不要混淆。

#### 2. "spec 阶段 1" vs "P0 修复"

两件事都跟 lead prompt 有关,但:
- spec 阶段 1 是 handoff_suffix + 双层 handoff(已 push 未合 dev)
- P0 修复是 lead bash removal(待 plan 待实施)

P0 修复要在 dev 当前 tip(3ee6bd07,**不含**spec 阶段 1)上做。等 P0 fix 合 dev 后,
spec 阶段 1 rebase 到 P0 fix 之后再合 dev。

#### 3. differential test 结果

Spec 阶段 1 agent 跑过 differential test:把 prompt 回退到 spec 阶段 1 改之前
(`0acfd46b`),lead 仍反复重试 bash。这**证伪了**"spec 阶段 1 prompt 改动是 P0 诱因"
的假设。Plan A 之前 grill 推断 prompt 是诱因——错了,不要再纠结。

### 已经验证的错误方向

- ❌ 不要回退 spec 阶段 1 commit(根因不在那)
- ❌ 不要给 G4 boundary 正则扩白名单(治标不治本,模型还会找别的绕过方式)
- ❌ 不要用 prompt 加约束让 lead 切 task tool(G1/G4/G5 教训:prompt-only 不可靠)

### 已发现的认知陷阱(本会话踩过的)

#### 陷阱 1:假因果

我前面看到"G4 dogfood lead 能 dispatch / spec 阶段 1 dogfood lead 不能 dispatch"
就推断"spec 阶段 1 prompt 是诱因"。但 G4 dogfood 那次 lead **根本没尝试自己 bash**
(直接派 code-executor),两次不构成 differential。差点带歪修复方向。

教训:看到关联不要立刻推因果,先问"这两个场景在出问题的那个分支上是不是真的对等"。

#### 陷阱 2:表层根因 vs 深层根因

我先定位到"G4 boundary 正则没匹配 cd &&" — 看起来是个 5 分钟正则改动。但用户
问了一句"为什么 lead 自己在跑 ethoinsight CLI,不派 code-executor",才暴露
真根因:**lead 不应该有跑 ethoinsight CLI 的权限**。表层根因修了,深层 bug 还在
(下次 lead 写别的 quoting 还是会卡)。

教训:表层根因找到后,问"如果修了这层,下次会不会以另一种形式复发"。

#### 陷阱 3:本会话写 plan 卡住

我多次回复"开始写"但没真发 Write tool call。用户问"为什么卡住" — 我也不完全
确定,候选解释:plan 内容跨多子系统、5 个 task 长得完全不一样、每个都要单独
verify 前置事实。**对策:plan 拆分 Write 调用,每次写 1-2 个 task,中间不发文字
narration**。

这条认知陷阱**给下一位 agent 的提醒**:如果发现自己卡在"准备写"循环,直接
Write,**不要先发"我开始写了"的文字消息**。

---

## 下一位 Agent 的第一步建议

```bash
# 1. 读交接文档(就你在读这个)
cat docs/handoffs/2026-05/2026-05-15-p0-lead-bash-removal-handoff.md

# 2. 验证状态
cd /home/wangqiuyang/noldus-insight
git status --short
git log -1 --format='%H %s' origin/dev          # 期望 3ee6bd07
git log -1 --format='%H %s' worktree-spec-phase-1-handoff  # 期望 adbf3107

# 3. 读必要的源码上下文
cat packages/agent/backend/packages/harness/deerflow/guardrails/lead_execution_boundary_provider.py
sed -n '14,25p' packages/agent/backend/packages/harness/deerflow/tools/tools.py
sed -n '430,445p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
head -80 packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py
head -90 packages/ethoinsight/ethoinsight/catalog/resolve.py
head -90 packages/ethoinsight/ethoinsight/parse/dump_headers.py
```

**然后**:调 `superpowers:writing-plans` skill,按本交接"未完成事项 §P0"段的
4 个 task 内容,展开成完整 plan 写到
`docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md`。

**Plan 写完后**:commit plan(主工作树,不 push),回报用户"plan 已写完,是否
要派执行 agent"。

**问用户**:
> "P0 修复实施 plan 已写完 ❌(本会话失败,新会话已接手)/ ✅(本会话已完成)。
> Plan 路径 docs/superpowers/plans/2026-05-15-lead-bash-removal-and-tool-registration-plan.md。
> 含 4 个 task,Task 1 (LoopDetectionMiddleware fix) 可独立 commit 独立止血。
> 要不要派执行 agent 实施 plan?执行 agent prompt 模板已在交接文档'第四步'段。"

**不要**未经用户确认就开 commit / 派 agent / 改代码。

---

## 会话状态(为什么有这份交接)

本会话(2026-05-15 review 角色)做完了:
- ✅ G4 方案 C review + 决定合 dev
- ✅ Spec 阶段 1 review + symlink bug 发现 + 让 agent 修
- ✅ Spec 阶段 1 live dogfood review + 暴露 P0
- ✅ P0 根因诊断(differential test + log 分析)
- ✅ P0 修复方案 7 个细节 grill 拍板

**没做完**:
- ❌ P0 修复实施 plan 没写出来(多次尝试 Write 但实际没发出 tool call,具体原因
  未明,见"认知陷阱 §3")

新会话第一件事是写这份 plan。所有 grill 决策 + 根因诊断已经在本交接里完整保留,
**不需要重做**。
