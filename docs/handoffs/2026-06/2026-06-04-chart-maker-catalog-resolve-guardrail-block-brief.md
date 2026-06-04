# 调查 + 修复 brief：chart-maker 全链路被 guardrail 拦死（catalog.resolve 不在白名单）

> **面向接手的 AI agent**。本 brief 已**现场核实**（guardrail 源码正则 + git 史 + chart-maker SKILL.md + 失败 thread 解码）。**真根因高置信度锁定，是一个 harness guardrail 白名单遗漏的回归，大概率一行正则修复 + 单测。** 另有一个次要的 subagent 脑补步骤问题。
>
> 失败 thread：`ca86744f-39e6-4acb-95fa-bda8ec8593ed`（与画图反问循环 brief 同一 thread）。仓库 HEAD 应含 `e5a0d53b`。

---

## 1. 现象

O迷宫 dogfood 走到 chart-maker 出图时全部失败，前端还伴随一个前端 console error：

```
Unexpected tool message outside a processing group {}
  at getMessageGroups (src/core/messages/utils.ts:83)
```

chart-maker 最终 seal 了 `status=failed`、`charts_generated: 0`、`failed_charts: 3`，blocker 自报 `dump_headers_not_callable`。lead 收到失败后正确 ask_clarification 问用户怎么办。

---

## 2. 真根因（已核实，高置信度）

### 2.1 guardrail 白名单正则不含 catalog.resolve

`backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py:40`：

```python
_ALLOWED_PYTHON_PATTERN = re.compile(
    r"^\s*python\s+-m\s+ethoinsight\.scripts\.\w+\.\w+(\s|$)"
)
```

这是 chart-maker / code-executor 的 bash 唯一放行的 `python -m` 形式。它**只匹配 `ethoinsight.scripts.<paradigm>.<script>`**，不匹配：
- `python -m ethoinsight.catalog.resolve --mode charts ...`
- `python -m ethoinsight.parse.dump_headers ...`

`evaluate()` 里 `python -m` 的唯一 allow 分支就是 `_ALLOWED_PYTHON_PATTERN.match(cmd)`（L182），不匹配就落到末尾 `return GuardrailDecision(allow=False, ...)`（L221）。**没有任何分支放行 catalog.resolve。**

### 2.2 但 chart-maker 的工作流要求跑 catalog.resolve

`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md:22`（chart-maker subagent 的工作流 step 2）：

```
2. bash `python -m ethoinsight.catalog.resolve --mode charts --paradigm <p> --user-intent "<原话>" ... --output plan_charts.json`
```

`packages/agent/skills/custom/ethoinsight/references/execution-conventions.md:14` 也明确声明：

```
- python -m ethoinsight.catalog.resolve --mode charts ... — chart-maker 自跑
```

**契约自相矛盾**：执行宪法 + chart-maker SKILL 都说"chart-maker 自跑 catalog.resolve"，但 guardrail 从未把它放进白名单。

### 2.3 git 史证明：catalog.resolve 从未在白名单里

```bash
cd /home/wangqiuyang/noldus-insight
git log -p -S 'catalog.resolve' -- packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py
# → 空。catalog.resolve 从未出现在这个 guardrail 文件里。
```

guardrail 引入于 `4a061097`（ScriptInvocationOnlyProvider 白名单），`797e42b9` 扩过 chart-maker 白名单但**只扩了 file-ops 路径校验，没加 catalog.resolve**。

**结论：chart-maker 自这个 guardrail 落地起就从没能跑通 catalog.resolve 这一步。** 要么 chart-maker 在 guardrail 落地后从未被真正端到端 dogfood 过，要么早期走的是别的代码路径。这是个一直潜伏、这次 O迷宫 dogfood 才暴露的 harness 回归。

### 2.4 次要问题：subagent 脑补了 SKILL 里没有的 dump_headers 步骤

chart-maker SKILL.md 的工作流（L20-35）**没有 dump_headers / columns.json 这一步**——step 2 直接就是 catalog.resolve。但失败 thread 里 subagent 的 thinking 自己发明了"Step 7: write columns.json via dump_headers CLI"，反复试 `ethoinsight.parse.dump_headers` / `ethoinsight.scripts._common.dump_headers` / `ethoinsight.scripts.zero_maze.dump_headers`（后两个模块根本不存在），全被 guardrail 拦或报 No module。

这是 subagent 把别处工作流（可能是旧版或 code-executor 的）的步骤**脑补**进了 chart-maker 流程。即使 dump_headers 问题不存在，它接着跑正确的 catalog.resolve **也会**被 §2.1 的 guardrail 拦死。所以 §2.1 是主根因，dump_headers 脑补是叠加噪声。

> 关联记忆：[[feedback_skill_describing_tool_output_enables_hallucination]]（skill 详述易诱导脑补）；[[feedback_subagent_consumption_via_first_party_tool]]。

---

## 3. 修复方案

### 3.1 🔴 主修复（harness，一行正则 + 边界确认）

把 `ethoinsight.catalog.resolve` 加进 guardrail 白名单。**但不要简单粗暴地放开整个 `ethoinsight.*`**——白名单要精准，符合 provider 顶部"allowed shape is small and stable"的设计。

建议改 `script_invocation_only_provider.py:40` 的正则，新增一条精准放行（二选一，**优先方案 A**）：

**方案 A（推荐）：扩 `_ALLOWED_PYTHON_PATTERN` 容纳 catalog.resolve 与 parse.dump_headers**

```python
# scripts.<paradigm>.<script>（compute/plot 脚本）
# catalog.resolve（chart-maker / prep 内部 resolve；execution-conventions.md:14 声明 chart-maker 自跑）
# parse.dump_headers（若确认 chart-maker 工作流确实需要——见 §3.3）
_ALLOWED_PYTHON_PATTERN = re.compile(
    r"^\s*python\s+-m\s+ethoinsight\.(scripts\.\w+\.\w+|catalog\.resolve|parse\.dump_headers)(\s|$)"
)
```

**方案 B：单独加一条 allow 分支**（若想让拒绝消息/日志区分 resolve vs script）——在 L182 的 `if _ALLOWED_PYTHON_PATTERN.match(cmd)` 之后加：

```python
if re.match(r"^\s*python\s+-m\s+ethoinsight\.catalog\.resolve(\s|$)", cmd):
    return GuardrailDecision(allow=True)
```

**边界确认（实施前必做）**：
- catalog.resolve 会接收 `--raw-files-json` / `--groups-json` / `--output` 等参数，参数里有路径。确认放行 catalog.resolve 本身不绕过 §2.1 之外的路径安全校验——catalog.resolve 是只读 resolve + 写 plan_charts.json 到 workspace，不执行任意代码，放行它本身是安全的（它和 prep_metric_plan 工具内部调的是同一个 mode，只是 chart-maker 走 charts mode）。
- 确认 `--user-intent "<中文原话>"` 里的引号/空格不会让正则误判（正则只 anchor 开头 `python -m ethoinsight.catalog.resolve`，后面 `(\s|$)` 之后的 args 不卡，安全）。

### 3.2 🟡 次要修复（SKILL，防 dump_headers 脑补）

确认 chart-maker SKILL.md 工作流里**确实不需要 dump_headers**（catalog.resolve 自己读 raw_files 解析列）。若确实不需要：
- 不用改 SKILL（它本来就没写 dump_headers，是 subagent 脑补的）。
- 可考虑在 SKILL.md 工作流 step 2 附近加一句正向澄清：「catalog.resolve 会自行从 raw_files 解析列结构，本步骤无需单独 dump 列名」——堵住脑补缺口（遵守 §6 正面措辞）。

若 catalog.resolve **确实**需要预先的 columns.json（去 read resolve.py 的 `--mode charts` 是否要 `--columns-file`）：那 dump_headers 就是真需要的步骤，§3.1 必须把 `parse.dump_headers` 也放行，且要确认 `ethoinsight.parse.dump_headers` 模块真实存在可跑。**先确认 catalog.resolve charts mode 的真实入参契约再决定。**

### 3.3 验证 catalog.resolve charts mode 的真实契约（实施第一步）

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
.venv/bin/python -m ethoinsight.catalog.resolve --help 2>&1 | head -40
# 看 --mode charts 到底要哪些参数：是否需要 --columns-file？只要 --raw-files-json 够不够？
```

这一步决定 §3.2 走哪条路（dump_headers 要不要放行）。

### 3.4 🟢 前端 console error（可能是次生，先确认是否独立）

`frontend/src/core/messages/utils.ts:83` 报 `Unexpected tool message outside a processing group`。这很可能是 chart-maker 失败 + 反问循环（见姊妹 brief `2026-06-04-clarification-reask-loop-investigation-brief.md`）共同导致的消息分组异常——一堆连续无 tool_call AIMessage + 失败 ToolMessage 打乱了前端 group 逻辑。**先修后端两个 brief 的根因，再看这个前端报错是否自动消失**；若仍在，再单独查 `getMessageGroups` 对"孤立 ToolMessage"的容错。

---

## 4. 验证

1. **red 锚点单测**：`backend/tests/test_script_invocation_only_provider.py`（已存在，扩它）——加一条断言 `python -m ethoinsight.catalog.resolve --mode charts --paradigm zero_maze --user-intent "画图" --output x.json` 对 chart-maker agent_id **allow=True**；同时保留对 `python -c` / `pip install` / 任意脚本 **仍 deny** 的既有断言（确认没放开太多）。
2. **跑全量**（改共享 guardrail 必跑全量）：`cd backend && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`。
   - ⚠️ 已知 flake（与改动无关，别误判）：全量顺序下 `test_deferred_tool_registry_promotion.py` 2 条 + `test_inspect_gate_guardrail.py::...test_async_delegates_to_sync` + `test_paradigm_identification_gate.py::...test_async_delegates_to_sync`，单跑这 4 条 4/4 passed。
3. **重启 dev 后 dogfood 复测**（改 guardrail 必 `make stop && make dev`）：
   - O迷宫，发"分析数据" → 走到画图 → 选"画图" → **期望 chart-maker 跑通 catalog.resolve → plot 脚本 → 出 png**。
   - 盯 chart-maker thinking：catalog.resolve 不再被拦；若 subagent 仍脑补 dump_headers，记下来回到 §3.2。

---

## 5. 注意 / 红线

- **白名单精准放行，不要 `ethoinsight\.\w+` 整个放开**——provider 设计哲学是 small & stable 白名单，放太宽会让 `python -m ethoinsight.<任意>` 都能跑，破坏 script-per-metric 沙箱隔离。只加 `catalog.resolve`（+ 经确认的 `parse.dump_headers`）。
- **改 guardrail 是改共享判定逻辑**（用户铁律 [[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）：必跑全量 + grep 所有调用方（code-executor 也走这个 guardrail，确认放行 catalog.resolve 不会让 code-executor 误用——code-executor 不该调 charts mode，但放行 module 本身无害，它的工作流不会去调）。
- **先验 catalog.resolve --help 契约再动 SKILL**（§3.3）——别假设要不要 dump_headers。
- **正面措辞**（CLAUDE.md §6）：若改 SKILL 堵脑补，用"catalog.resolve 自行解析列结构"正向描述，不写"不要调 dump_headers"。
- **改完必重启 dev** 再 dogfood。

---

## 6. 关键路径速查

- guardrail（主修复点）：`backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py:40`（`_ALLOWED_PYTHON_PATTERN`）+ `:182`（allow 分支）
- guardrail 单测：`backend/tests/test_script_invocation_only_provider.py`
- chart-maker 工作流：`packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md:22`（catalog.resolve step）
- 执行宪法（声明 chart-maker 自跑 resolve）：`packages/agent/skills/custom/ethoinsight/references/execution-conventions.md:14`
- catalog.resolve 入口：`packages/ethoinsight/ethoinsight/catalog/resolve.py`
- 失败 thread：`.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/ca86744f-39e6-4acb-95fa-bda8ec8593ed/`（workspace 有 raw_files.json / groups.json 已写，但无 plan_charts.json / 无 png）
- checkpoint 解码：`langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`（非 msgpack）
- 姊妹 brief（同 thread 的反问循环问题）：`docs/handoffs/2026-06/2026-06-04-clarification-reask-loop-investigation-brief.md`
- 历史：`docs/handoffs/2026-05/2026-05-26-chart-maker-fst-contract-fix-handoff.md`（chart-maker 契约修复史，5 根因）
