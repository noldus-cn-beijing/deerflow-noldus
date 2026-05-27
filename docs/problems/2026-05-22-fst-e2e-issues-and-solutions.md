# 2026-05-22 FST E2E 问题诊断与解决方案

> **文档状态**:v2(2026-05-22 修订)
> v1 由初稿 agent 产出,v2 经事实核对 + 修正建议合并而成,**可直接交付实施 agent 执行**。
> 修订要点:P0 schema 补强、P1 三层防御细节打磨、P1 reasoning_effort 降级策略收敛、实施依赖顺序点明。

## 背景

2026-05-21 用户跑了 FST(强迫游泳实验)端到端流程,总耗时 ~19.5 分钟(5 个 run),期间触发 5 次 LoopDetection 警告。本文档基于 langgraph.log 的完整分析,梳理每个问题的前因后果,并给出利用 DeerFlow 现有 infra 的解决方案。

## 数据来源

- langgraph.log: `/home/wangqiuyang/noldus-insight/packages/agent/logs/langgraph.log`(2443 行)
- 用户提供的前端输出轨迹(完整 lead agent thinking 内容)
- **事实核对**:本文档所有"现状代码引用"已在 2026-05-22 复盘时逐条对照源码与日志确认(见各章节"事实核对"小节)

## 5 个 Run 的时间线

| Run | request_id | 耗时 | 用途 | reasoning_effort | is_plan_mode |
|-----|-----------|------|------|-----------------|-------------|
| 1 | 3088f64e | 49s | Gate 1 范式识别(EV19 模板反问) | high | True |
| 2 | 1fc03b72 | 70s | Gate 1 继续(set_experiment_paradigm + prep_metric_plan + 分组反问) | high | True |
| 3 | a6ec26df | 5.6min | code-executor 派遣 + Gate 2 质量门反问 | high | True |
| 4 | e873e318 | 5.9min | data-analyst 派遣(Gate 2 死锁导致 loop hell) | high | True |
| 5 | 658fb859 | 4min | chart-maker 派遣 | high | True |

**总 token 消耗**: ~120k input + ~6k output(est.,基于 TokenUsageMiddleware 日志求和)

---

## 问题 1(致命):Gate 2 死锁 — lead 无法写 experiment-context.json

### 现象

从日志提取的关键事件序列(Run 3 + Run 4,同一个 thread `189e7840`):

```
09:40:23  gate_check gate=gate2_quality → BLOCKED (1 critical warning, not acknowledged)
09:40:39  ClarificationMiddleware: Intercepted clarification request
          → 用户选择"是,继续做描述性比较即可"
          [Run 3 结束,耗时 336s]

09:46:43  Run 4 开始 (reasoning_effort=high, is_plan_mode=True)
09:47:06  gate_check gate=gate2_quality → BLOCKED (仍然!因为 experiment-context.json 没更新)
09:47:17  LoopDetection: read_file x3 warning
09:47:39  LoopDetection: Repetitive tool calls — injecting warning (read_file)
09:48:30  gate_check gate=gate2_quality → BLOCKED (第三次)
09:49:25  SandboxAudit: python -c "json.dump({...gate2_quality_acknowledged...})"
          → code-executor 终于写入了 experiment-context.json
09:49:51  LoopDetection: task(data-analyst) x3 warning  ← 用户看到的 LOOP DETECTED
09:49:51  gate_check gate=gate2_quality → ALLOWED (acknowledged)
09:52:39  ClarificationMiddleware: 图表反问
          [Run 4 结束,耗时 356s]
```

### 事实核对(2026-05-22)

| 声明 | 核对结果 |
|---|---|
| `_LEAD_EXCLUDED_TOOLS = {bash, write_file, str_replace}` | ✅ [agent.py:234](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py#L234) |
| `set_experiment_paradigm` 硬写 `gate_completed: ["gate1_paradigm"]` | ✅ [experiment_context.py:160](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py#L160) |
| Gate 2 阻塞消息让 lead 用 `write_file` | ✅ [gate_enforcement_middleware.py:99-100](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py#L99-L100) |
| lead 最终用 `python -c "json.dump(...)"` 派 code-executor 绕过 | ✅ 日志 09:49:25 SandboxAudit 实锤 |
| Gate 2 被 block 三次后才 allowed | ✅ 09:40:23 / 09:47:06 / 09:48:30 blocked → 09:49:51 allowed |

### 根因链

1. [gate_enforcement_middleware.py:99-100](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py#L99-L100) 的 Gate 2 阻塞消息告诉 lead:

```
用户确认后,调用 write_file 更新 experiment-context.json,
在 gate_completed 中添加 'gate2_quality_acknowledged',然后重新调用 task(data-analyst)。
```

2. 但 [agent.py:234](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py#L234) 把 `write_file` 排除了:

```python
_LEAD_EXCLUDED_TOOLS: frozenset[str] = frozenset({"bash", "write_file", "str_replace"})
```

3. `set_experiment_paradigm` 工具([experiment_context.py:160](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py#L160))只写 `gate_completed: ["gate1_paradigm"]`,没有追加 gate2 的能力。

4. lead 的挣扎路径:
   - 尝试 `task(data-analyst)` → Gate 2 阻塞
   - 尝试 `ask_clarification` → 用户确认
   - 再次 `task(data-analyst)` → **仍阻塞**(JSON 没更新)
   - 尝试 bash → 没有 bash 工具
   - 尝试 general-purpose subagent → 不存在
   - 派遣 code-executor 用 `python -c` 写 JSON → 成功(但已经浪费 ~3 分钟 + 触发 LoopDetection)

**根因本质**:gate 中间件给 lead 的指令("用 write_file 更新 JSON")与 lead 的实际能力(没有 write_file)不匹配。这是一个**中间件契约与工具能力的不一致**。

### 解决方案:给 `set_experiment_paradigm` 加 `acknowledge_quality` 参数,并将所有 Gate 1 字段改为可选

**为什么选这个方案**:

- `set_experiment_paradigm` 已经有写 experiment-context.json 的能力(通过 `runtime.state.thread_data.workspace_path` 解析 host 路径)
- lead 已经有这个工具(不在排除列表中)
- 它是 Gate 1 确认的同一个工具——Gate 2 确认用同一个工具符合用户心智模型
- 改动最小、不破坏 lead 工具隔离原则

**实施细则**(给实施 agent 的精确改造说明):

#### 改动 A:`set_experiment_paradigm_tool` 的 schema 调整

文件:[experiment_context.py:95-170](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py#L95-L170)

把所有 Gate 1 字段改为 `Optional[str] = None`,新增 `acknowledge_quality: bool = False`:

```python
@tool("set_experiment_paradigm", parse_docstring=True)
def set_experiment_paradigm_tool(
    paradigm: str | None = None,
    paradigm_cn: str | None = None,
    category: str | None = None,
    subject: str | None = None,
    ev19_template: str | None = None,
    acknowledge_quality: bool = False,
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Record the user's experiment paradigm choice and/or acknowledge data quality.

    Two usage modes:
      1) Gate 1 paradigm confirmation:
         set_experiment_paradigm(paradigm="forced_swim", paradigm_cn="...", category="...",
                                 subject="...", ev19_template="...")
         → creates experiment-context.json with gate_completed=["gate1_paradigm"]
      2) Gate 2 quality acknowledgement:
         set_experiment_paradigm(acknowledge_quality=True)
         → reads existing experiment-context.json, appends "gate2_quality_acknowledged"
           to gate_completed (preserving all other fields). Requires Gate 1 already done.

    Args:
        paradigm: English paradigm name key. Required for Gate 1 mode.
        paradigm_cn: Chinese display name. Required for Gate 1 mode.
        category: Category name. Required for Gate 1 mode.
        subject: Subject type. Required for Gate 1 mode.
        ev19_template: EthoVision 19 template variant ID. Required for Gate 1 mode.
        acknowledge_quality: Set True to acknowledge data quality warnings (Gate 2 mode).
                             When True, all paradigm fields may be omitted — the existing
                             experiment-context.json is read and only gate_completed is
                             updated.
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/".

    Returns:
        JSON confirmation with the updated context.
    """
```

#### 改动 B:tool 函数体改造

伪代码:

```python
# 1) resolve actual_workspace from runtime.state.thread_data.workspace_path (现有逻辑保留)
actual_workspace = ...

# 2) 读取已有 context(可能为 None)
existing = read_context(actual_workspace)

# 3) 分支:Gate 2 acknowledge 模式
if acknowledge_quality:
    if existing is None:
        return json.dumps({
            "status": "error",
            "message": "Cannot acknowledge quality before Gate 1. Call set_experiment_paradigm with paradigm fields first."
        }, ensure_ascii=False)
    gate_completed = existing.get("gate_completed", [])
    if not isinstance(gate_completed, list):
        gate_completed = []
    if "gate2_quality_acknowledged" not in gate_completed:
        gate_completed.append("gate2_quality_acknowledged")
    data = {**existing, "gate_completed": gate_completed,
            "gate2_acknowledged_at": datetime.now(UTC).isoformat()}
    # write back
    ...
    return json.dumps({"status": "ok", "path": ..., "gate_completed": gate_completed},
                      ensure_ascii=False)

# 4) 分支:Gate 1 创建/更新模式
# 必填校验
required = {"paradigm": paradigm, "paradigm_cn": paradigm_cn,
            "category": category, "subject": subject, "ev19_template": ev19_template}
missing = [k for k, v in required.items() if not v]
if missing:
    return json.dumps({"status": "error",
                       "message": f"Missing required fields for Gate 1: {missing}"},
                      ensure_ascii=False)

# ev19_template 校验(保留原有逻辑)
...

# 写入,保留已有 gate_completed 中的 gate2_quality_acknowledged(若存在)
prior_gate_completed = (existing or {}).get("gate_completed", []) if isinstance(existing, dict) else []
gate_completed = ["gate1_paradigm"]
if "gate2_quality_acknowledged" in prior_gate_completed:
    gate_completed.append("gate2_quality_acknowledged")
data = {
    "paradigm": paradigm,
    "paradigm_cn": paradigm_cn,
    "category": category,
    "subject": subject,
    "ev19_template": ev19_template,
    "paradigm_confirmed_at": datetime.now(UTC).isoformat(),
    "gate_completed": gate_completed,
}
```

**说明**:Gate 1 模式下保留已有 `gate2_quality_acknowledged`,是为了支持"用户改主意改 paradigm 之后,不必重新走 Gate 2 确认"——但若产品要求 Gate 1 重置一切,把这两行去掉即可。建议**保留**,因为重新走 Gate 2 反问对用户体验更差。

#### 改动 C:更新 Gate 2 阻塞消息

文件:[gate_enforcement_middleware.py:99-100](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/gate_enforcement_middleware.py#L99-L100)

把这段:

```
用户确认后,调用 write_file 更新 experiment-context.json,
在 gate_completed 中添加 'gate2_quality_acknowledged',然后重新调用 task(data-analyst)。
```

改成:

```
用户确认后,调用 set_experiment_paradigm(acknowledge_quality=True) 记录确认,
然后重新调用 task(data-analyst)。
```

#### 改动 D:测试

在 `backend/tests/` 下新增 `test_set_experiment_paradigm_gate2.py`,覆盖:
1. Gate 1 模式:正常创建 → gate_completed == ["gate1_paradigm"]
2. Gate 2 acknowledge 模式:已有 Gate 1 context → 追加 "gate2_quality_acknowledged"
3. Gate 2 acknowledge 模式:无 Gate 1 context → 返回 error
4. Gate 2 重复 acknowledge:不重复追加
5. Gate 1 后已有 Gate 2:Gate 1 重新调用保留 Gate 2 acknowledge

**备选方案(不推荐)**:

- B) 给 lead 恢复 `write_file` 工具:破坏工具隔离原则,lead 不应该直接写文件
- C) ClarificationMiddleware 回调自动更新 JSON:引入隐式副作用,调试困难

---

## 问题 2(严重):write_todos 每轮都重写

### 现象

从日志提取的 LoopDetection 警告:

```
09:35:59  LoopDetection: write_todos x3 frequency warning
09:56:59  LoopDetection: write_todos x3 frequency warning
```

从前端输出来看,lead 的典型行为模式是:

```
→ 思考一段
→ write_todos([...])    # 重写整个 todo list
→ 派遣 subagent
→ 收到结果
→ 思考一段
→ write_todos([...])    # 又重写整个 todo list,内容几乎一样
→ 派遣下一个 subagent
→ ...
```

每次 `write_todos` 调用消耗 ~200-400 input token + 1 轮 LLM round-trip(~3-10s)。在 5 个 run 中这个模式重复了约 6-8 次。

### 事实核对

- LoopDetection 全局默认阈值是 `warn_threshold=3, hard_limit=5`([loop_detection_middleware.py:30-35](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py#L30-L35))
- `LoopDetectionMiddleware` 已支持 `tool_freq_overrides`([loop_detection_middleware.py:178](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py#L178))
- `guardrails/` 目录已有同类 Provider 协议实现(`ev19_template_provider.py` 等),可直接类比
- lead prompt(`agents/lead_agent/prompt.py`)**目前没有** write_todos 使用段落 — Layer 3 是"新增段落"而非"修改"

### 根因

这**不是中间件的 bug,而是 prompt 导致的行为问题**。lead 的 system prompt 告诉它"update todos as you work",deepseek-v4-pro 解释为"每轮都要重新输出完整的 todo list"。

更深层的根因:**缺乏"规划阶段 vs 执行阶段"的契约**。TodoMiddleware 只防止 agent 在任务未完成时退出(completion reminder),但不防止 agent 反复重新规划。

### 解决方案(三层防御)

#### Layer 1(核心):`TodoPlanningDisciplineProvider` — 利用 GuardrailMiddleware 协议

新建 `packages/harness/deerflow/guardrails/todo_planning_discipline_provider.py`,实现 `GuardrailProvider` 协议。

**核心机制(v2 修正)**:

1. **前 2 次 write_todos** → allow(初始规划 + 一次调整)。
2. **之后每次** → 计算"内容签名",做实质性变化检测:
   - **签名定义**:`signature = [(item.get("content"), item.get("activeForm")) for item in todos]`
     **不包含 status** — 这是为了允许合法的状态转换(pending → in_progress → completed)无需带 reason 参数即可通过,因为 status 变化时 content/activeForm 通常不变,签名一致 → 走"无变化"分支。
   - **status 变化判定**:同时计算 `status_diff = (旧 status 集合) vs (新 status 集合)`。
     - 若 `signature` 不变 + `status_diff` 不为空 → **allow**(合法状态转换)
     - 若 `signature` 不变 + `status_diff` 也为空 → **deny**(纯重写,无实质变化)
     - 若 `signature` 变了 → **allow**(增删条目或改写描述)
3. **deny 时的 ToolMessage 文案**(中文,对齐 deepseek 习惯,**正面指令**):
   ```
   todo 列表已反映当前状态,无需重写。继续执行下一个任务即可。
   若确有合法状态变化或新增条目,请在 reason 参数中说明(下次调用会放行)。
   ```
4. **reason 参数兜底机制**:provider 检查 tool_call args 是否包含 `reason: str`(任意非空字符串)。若存在,**直接放行不做签名比较** — agent 明确声明意图时永远放行,避免误杀。这取代了原 v1 方案的"每 5 次 deny 后放行一次"魔数机制。
5. **状态按 thread_id 隔离**,通过 `runtime.context["thread_id"]` 获取。
6. **TodoListMiddleware 工具是否接受 `reason`**:需要在 TodoListMiddleware 的 write_todos schema 中加 `reason: str | None = None`(忽略它在功能上的作用,仅供 guardrail 读取)。改动量 ~3 行。

**为什么用 GuardrailMiddleware 而不是 LoopDetection**:

| 维度 | LoopDetection(现有) | GuardrailMiddleware + Provider(方案) |
|------|----------------------|--------------------------------------|
| 拦截时机 | after_model(事后警告) | wrap_tool_call(事前拦截) |
| 反馈方式 | 在 AIMessage 末尾追加警告文字 | 返回 error ToolMessage,agent 立即看到 |
| 状态追踪 | 仅计数 | 可比较内容签名 |
| 配置方式 | yaml 静态阈值 | Provider 代码,逻辑任意定制 |

**测试**:在 `backend/tests/` 下新增 `test_todo_planning_discipline_provider.py`,覆盖:
1. 前 2 次 allow
2. 第 3 次内容不变 + status 不变 → deny
3. 第 3 次内容不变 + status 变化(in_progress → completed) → allow
4. 第 3 次内容变化(新增/删除条目) → allow
5. 带 `reason` 参数 → 永远 allow
6. 多 thread 状态隔离

#### Layer 2(兜底):LoopDetectionMiddleware `tool_freq_overrides` 配置

在 `config.yaml` 加:

```yaml
loop_detection:
  tool_freq_overrides:
    write_todos:
      warn: 2
      hard_limit: 4
```

**说明(v2 修正)**:全局默认是 `warn=3, hard_limit=5`,所以 `write_todos` override 必须**比全局更严**才有意义。`2/4` 与 Layer 1 的"前 2 次免检"对齐——Layer 1 在第 3 次开始拦截,Layer 2 在第 4 次硬停。这是零代码改动,作为 Layer 1 实现 bug 时的兜底安全网。

#### Layer 3(根治):prompt 修改

在 [agents/lead_agent/prompt.py](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 的 lead system prompt **任务规划/todo 使用相关段落**新增以下中文段落(若该文件目前没有 todo 段落,在工具说明区或"工作方式"区新增):

```
**Todo 列表使用规则**:

制定计划后调用一次 write_todos 写入完整 todo 列表。之后专注于按顺序执行。

只有以下情况才再次调用 write_todos:
- 任务状态发生了真实变化(pending → in_progress → completed)
- 出现了原计划未覆盖的新任务需要追加
- 用户明确要求调整计划

todo 列表已反映当前状态时,继续执行下一个任务即可。如果确实需要在状态未变的情况下
重写列表,在 reason 参数中说明原因。
```

**v2 修正说明**:CLAUDE.md 第 6 条指出 deepseek 对"不要 X / 禁止 X"会反向激活。原 v1 文案 `Do NOT re-write the entire todo list every turn — this wastes time and context` 改为正面指令"已反映当前状态时,继续执行下一个任务即可"。

---

## 问题 3(中等):每个 run 都用 `reasoning_effort: high`

### 现象

5 个 run 全部 `reasoning_effort: high, is_plan_mode: True`。单个 run 的 reasoning token 消耗:

| Run | reasoning token 合计 |
|-----|---------------------|
| 1 | 378 + 194 = 572 |
| 2 | 764 + 86 + 33 + 138 = 1021 |
| 3 | 140 + 31 + 22 + 259 + 60 + 73 + 130 + 32 = ~747 |
| 4 | 46 + 250 + 56 + 33 + 55 + 48 + 53 + 301 + 391 + 349 + 487 + 28 + 427 = ~2524 |
| 5 | 57 + 53 + 79 + 47 + 95 + 54 = ~385 |

Run 4(data-analyst 派遣,含 gate 死锁挣扎)烧了 ~2500 reasoning token。这个 run 的核心决策只是"读 handoff JSON → 派遣 data-analyst",不需要 high reasoning。**注意**:Run 4 的高消耗主要由问题 1(Gate 2 死锁)驱动,P0 修完后这个 run 应当 ~1 分钟内完成,reasoning 自然显著下降——本问题的方案是在 P0 之上的**叠加优化**,不是冗余。

### 根因

`make_lead_agent` 在 [agent.py:397](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py#L397) 从 `config.configurable` 读取 `reasoning_effort`,整个 run 不变。调用方(前端)对所有 run 传相同的 `reasoning_effort: high`。

### 解决方案:在 `make_lead_agent` 中根据 thread 状态**降一级**(而非降到 None)

**v2 修正**:原 v1 方案在派遣阶段降到 `None`,有两个风险:
1. 用户在派遣阶段中途追问"这个数据看起来不对劲,能解释一下吗" — lead 仍处于"已派遣"判断分支,但其实需要分析推理
2. `None` 与 `high` 落差太大,引入回归不易定位

修正策略:**降一级**(high → medium → low),保留思考能力但降低消耗。

```
逻辑:
1. 从 config 获取 thread_id 和 配置值 configured_effort(默认 high)
2. 通过 thread_id 查找 workspace 路径(复用 ThreadDataMiddleware 的路径规则)
3. 检查 experiment-context.json:
   - 不存在/读取失败 → reasoning_effort = configured_effort(默认 high,fail-safe)
   - 存在且 gate1_paradigm 在 gate_completed 中 → 降一级
   - 存在且 gate2_quality_acknowledged 也在 → 再降一级
4. 降级函数 step_down(effort):
   - "high" → "medium"
   - "medium" → "low"
   - "low" → "low"(不再降)
   - None / 其他 → 保持
```

| 阶段 | 判断条件 | reasoning_effort(configured=high 时) |
|------|---------|:---:|
| 首次进入 | 无 experiment-context.json | high |
| Gate 1 完成 | 有 gate1_paradigm | medium |
| Gate 2 完成 | gate1 + gate2 都完成 | low |
| 旧 thread(无 workspace) | 找不到 workspace | high(fail-safe) |

改动位置:[agent.py:397 附近](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py#L397) 的 `make_lead_agent` 函数,~25 行(含读 context 的工具函数复用 `read_context()`)。

**测试**:在 `backend/tests/` 下新增 `test_make_lead_agent_reasoning_downgrade.py`,覆盖:
1. 无 workspace → 保持配置值
2. 有 workspace 无 context.json → 保持配置值
3. context.json gate_completed=["gate1_paradigm"] → high → medium
4. context.json gate_completed=["gate1_paradigm", "gate2_quality_acknowledged"] → high → low
5. 配置值为 None → 不降级(None step_down → None)
6. 配置值为 low → low(不再降)

**先决条件**:此问题修复必须在 P0(问题 1)上线**之后**才做,以便:
1. Run 4 死锁修好后,验证基线消耗
2. 避免叠加变更引入回归难定位

---

## 问题 4(低):read_file 重复读同一文件

### 现象

```
09:35:49  LoopDetection: read_file x3 frequency warning
09:47:17  LoopDetection: read_file x3 frequency warning
09:47:39  LoopDetection: Repetitive tool calls — injecting warning (read_file)
```

lead 反复读取 handoff_code_executor.json 和 experiment-context.json,即使内容没变。

### 根因

这是已知问题,handoff 文档(`2026-05-21-deerflow-sync-complete-and-next-handoff.md`)已经把 **ReadCacheMiddleware** 列为高优先级待办,设计已对齐。

### 解决方案

按 handoff 文档中的设计实现 ReadCacheMiddleware,**本文档不重复讨论**。在本次实施批次中,本问题**推迟到下一个工作单元**——本文档的 P0/P1 修复完成后,P0 修复会显著降低 lead 对 experiment-context.json 的重复读取,届时再评估 ReadCacheMiddleware 的优先级。

---

## 问题 5(低):Pydantic 序列化警告

### 现象

```
PydanticSerializationUnexpectedValue(Expected `none` - serialized value may not be
as expected [field_name='context', input_value={'model_name': 'deepseek-...'])
```

出现 2 次(Run 1 和 Run 2 的 tools node),不影响功能。

### 根因

Tool call 的 `context` 字段被传入了 dict(含 model_name 等),但 Pydantic schema 期望 `None`。可能是 LangChain/LangGraph 版本兼容性问题。

### 解决方案

低优先级,排查 tool call context 字段的来源,修正序列化路径。不影响功能,**本次实施批次推迟**。

---

## 实施计划(v2,含依赖顺序)

实施 agent 请**严格按以下顺序**执行,跨阶段之间需要回归验证。

### 阶段 A:P0 修复(必做且先做)

依赖:无。
变更:

| 步骤 | 文件 | 操作 | 估计行数 |
|------|------|------|:---:|
| A.1 | `experiment_context.py` | `set_experiment_paradigm_tool` schema 改为 Optional + acknowledge_quality | ~50 |
| A.2 | `experiment_context.py` | tool 函数体 分 Gate1/Gate2 两个分支 | (含在 A.1) |
| A.3 | `gate_enforcement_middleware.py` | 改阻塞文案,引导 lead 用 `set_experiment_paradigm(acknowledge_quality=True)` | ~3 |
| A.4 | `backend/tests/test_set_experiment_paradigm_gate2.py` | 新建,5 个 case | ~80 |

验证:`make test` 全绿,跑一次 FST E2E,确认 Gate 2 反问后 lead 直接调 `set_experiment_paradigm(acknowledge_quality=True)`,不再触发 LoopDetection。

### 阶段 B:P1 最便宜兜底(可与 A 并行,零代码)

依赖:无。
变更:

| 步骤 | 文件 | 操作 | 估计行数 |
|------|------|------|:---:|
| B.1 | `packages/agent/config.yaml` | 新增 `loop_detection.tool_freq_overrides.write_todos: {warn: 2, hard_limit: 4}` | ~5 |

验证:启动服务后跑 1 次任务,确认配置加载无报错(log 应出现 LoopDetection 初始化日志包含 overrides)。

### 阶段 C:P1 reasoning_effort 降级

依赖:**必须在 A 上线并验证后**才做(理由见问题 3 章节)。
变更:

| 步骤 | 文件 | 操作 | 估计行数 |
|------|------|------|:---:|
| C.1 | `agents/lead_agent/agent.py:397` 附近 | `make_lead_agent` 中根据 context.json 状态降级 | ~25 |
| C.2 | `backend/tests/test_make_lead_agent_reasoning_downgrade.py` | 新建,6 个 case | ~80 |

验证:`make test` 全绿,跑一次 FST E2E,对比 reasoning token 消耗(Run 3/4/5 应显著下降)。

### 阶段 D:P1 write_todos 三层防御(核心)

依赖:阶段 A 完成。
变更:

| 步骤 | 文件 | 操作 | 估计行数 |
|------|------|------|:---:|
| D.1 | `packages/harness/deerflow/guardrails/todo_planning_discipline_provider.py` | 新建,实现 GuardrailProvider 协议 | ~80 |
| D.2 | TodoListMiddleware write_todos 工具 | 添加 `reason: str \| None = None` 参数(供 guardrail 读取,本身忽略) | ~3 |
| D.3 | `make_lead_agent` 中注册新 provider | 与 ev19_template_provider 等并列 | ~5 |
| D.4 | `agents/lead_agent/prompt.py` | 新增 Todo 列表使用规则段落(正面指令) | ~10 |
| D.5 | `backend/tests/test_todo_planning_discipline_provider.py` | 新建,6 个 case | ~100 |

验证:`make test` 全绿,跑一次 FST E2E,确认 write_todos 调用次数从 ~6-8 次降到 ~2-3 次,且无误杀(合法状态转换不被拦)。

### 阶段 E(本文档不实施):P2

- 问题 4(ReadCacheMiddleware):按 handoff 文档单独实施
- 问题 5(Pydantic 警告):后续单独排查

### 总计

| 阶段 | 改动量 | 实施先后 |
|------|:---:|----|
| A(P0 Gate 2) | ~130 行 | 第 1 步 |
| B(LoopDetection override) | ~5 行 | 与 A 并行 |
| C(reasoning 降级) | ~105 行 | 在 A 之后 |
| D(write_todos 三层防御) | ~200 行 | 在 A 之后 |

**总计**:~440 行(其中测试约 260 行,生产代码约 180 行)。

---

## 不推荐的方向

- 给 lead 恢复 `write_file` 工具:破坏工具隔离,lead 不应直接操作文件
- 在 ClarificationMiddleware 中隐式更新 experiment-context.json:引入隐式副作用
- 完全禁用 `write_todos` 工具:agent 仍需追踪任务状态
- 不做 phase 判断、全局降 reasoning_effort:Gate 1 阶段确实需要 high reasoning
- write_todos 防御用"deny N 次后放行一次"的固定计数器:让 agent 学到"硬撞 N 次就能过",不如显式 reason 参数对齐意图(v2 修正)
- reasoning_effort 派遣阶段直接降到 None:用户中途追问场景下推理能力归零,且与 high 落差太大,改用"降一级"策略(v2 修正)

---

## v2 修订记录(2026-05-22)

| 章节 | 修订内容 | 理由 |
|------|---------|------|
| 问题 1 解决方案 | 补强 schema:所有 Gate 1 字段改为 Optional,Gate 2 模式可单独调用 | 否则 lead 被迫每次都重传 Gate 1 字段,出错率高 |
| 问题 1 改动 B | 增加"Gate 1 后保留 Gate 2 acknowledge"的合并逻辑 | 用户改主意 paradigm 后不必重走 Gate 2 |
| 问题 1 改动 D | 新增测试章节,明确 5 个 case | 给实施 agent 可执行清单 |
| 问题 2 Layer 1 | 签名只取 (content, activeForm),不取 status;增加 status_diff 旁路判定 | 避免误杀合法状态转换 |
| 问题 2 Layer 1 | 用 `reason` 参数显式放行,取代"deny 5 次放行一次"魔数 | 显式信号好于隐式计数 |
| 问题 2 Layer 1 | 增加 TodoListMiddleware 加 reason 字段的改动 | 否则 reason 走不到 provider |
| 问题 2 Layer 2 | 阈值改为 `warn=2, hard_limit=4`(原 v1 误写 `warn=2, hard_limit=4`,但理由表述含糊) | 与全局默认 3/5 拉开严格度,且与 Layer 1 "前 2 次免检"对齐 |
| 问题 2 Layer 3 | prompt 文案改为正面指令"已反映当前状态时,继续执行下一个任务即可" | CLAUDE.md 第 6 条:deepseek 对负面指令反向激活 |
| 问题 3 解决方案 | 派遣阶段从 `None` 改为"降一级"(high→medium→low) | 避免推理能力归零,且降低回归风险 |
| 实施计划 | 增加阶段 A/B/C/D 依赖顺序,C 必须在 A 之后 | 避免叠加变更引入回归难定位 |
| 各章节 | 新增"事实核对"小节,逐条对照源码+日志 | 给实施 agent 强引用,避免凭文档推测 |
