# Sprint 1 实施 spec — data_quality_warnings 结构化 + 浮上水面

**关联**：[2026-05-28 SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 1
**估期**：1.5 周
**前置**：**Sprint 0 必须先合 main**（Sprint 0 定义了 DataQualityWarning 的 code/evidence/blocks_downstream 三字段 + LEGACY 兜底；Sprint 1 需要这些字段就位才能填实际值，且必须清理掉 Sprint 0 留下的兜底）
**执行者**：交给独立 agent 执行

---

## 1. 背景与目标

### Sprint 0 之后留下的两个状态

1. **schema 已就位**：DataQualityWarning 含 `severity / metric / message / code / evidence / blocks_downstream` 6 字段
2. **过渡兜底在运行**：`seal_code_executor_handoff` 内对没有 code 的 warning 自动补 `code="LEGACY.UNCATEGORIZED"` + `evidence={}` + `blocks_downstream = (severity=="critical")`。所以 Sprint 0 上线之后，**所有 warning 实际进 handoff 时都已经有 6 字段**——但 code 是 "LEGACY.UNCATEGORIZED" 占位，evidence 是空 dict，blocks_downstream 按 severity 推断。

### Sprint 1 的两件事

1. **正本清源**：dispatcher.py 9 处 warning 端正填写真正的 code（4 一级分类）、真正的 evidence（数字证据）、真正的 blocks_downstream（按风险等级判定）
2. **删过渡兜底**：dispatcher 端真填了，seal tool 内的 LEGACY 兜底就该清掉；Sprint 0 spec §7.1 已经写明清理步骤

### 三个互锁的工作流

```
dispatcher.py 9 处 warning 升级（填 code/evidence/blocks_downstream）
        ↓
        ↓ data_quality_warnings 透传 handoff_code_executor.json (Sprint 0 seal tool 已就绪)
        ↓
data-analyst 读 warning、按 blocks_downstream 分级写入 method_warnings / quality_warnings
        ↓
lead 看 data-analyst 的 gate_signals.quality_warnings_critical_count
        ↓
前端按 blocks_downstream 渲染红字 / 橙字
        ↓
最后:删 LEGACY 兜底 + 验证 grep 空
```

---

## 2. 文件改动清单

### 2.1 改动 `dispatcher.py` 9 处 warning

**位置**：`packages/ethoinsight/ethoinsight/metrics/dispatcher.py:169-300`

**9 处 warning 映射表**（按代码顺序）：

| # | 行号 | 触发条件 | 旧结构 | 新 code | 新 evidence | blocks_downstream |
|---|---|---|---|---|---|---|
| 1 | 175-185 | 跨范式：组内 n<3 | severity=critical, metric=all | `SAMPLE.TOO_SMALL` | `{"n": <sample_n>, "threshold": 3, "group": <grp_name>}` | **true** |
| 2 | 192-202 | epm: 组内 n<5 | severity=warning, metric=all | `SAMPLE.UNDERPOWERED` | `{"n": <sample_n>, "threshold": 5, "paradigm": "epm", "group": <grp_name>}` | false |
| 3 | 204-216 | epm: subject total_entry_count<8 | severity=warning, metric=total_entry_count | `MOTOR.LOW_ENTRIES` | `{"subject": <name>, "total_entry_count": <int(te)>, "threshold": 8, "paradigm": "epm"}` | false |
| 4 | 223-233 | zero_maze: 组内 n<5 | severity=warning, metric=all | `SAMPLE.UNDERPOWERED` | `{"n": <sample_n>, "threshold": 5, "paradigm": "zero_maze", "group": <grp_name>}` | false |
| 5 | 236-252 | zero_maze: subject distance_moved<10 | severity=warning, metric=distance_moved | `MOTOR.LOW_DISTANCE` | `{"subject": <name>, "distance_moved": <td>, "threshold": <_ZM_LOW_DISTANCE_THRESHOLD>, "paradigm": "zero_maze"}` | false |
| 6 | 259-269 | light_dark_box: 组内 n<5 | severity=warning, metric=all | `SAMPLE.UNDERPOWERED` | `{"n": <sample_n>, "threshold": 5, "paradigm": "light_dark_box", "group": <grp_name>}` | false |
| 7 | 271-283 | light_dark_box: subject transition_count<4 | severity=warning, metric=transition_count | `SIGNAL.LOW_TRANSITION_COUNT` | `{"subject": <name>, "transition_count": <int(tc)>, "threshold": 4, "paradigm": "light_dark_box"}` | false |
| 8 | 286-300 | forced_swim/tail_suspension: 组内 n<5 | severity=warning, metric=all | `SAMPLE.UNDERPOWERED` | `{"n": <sample_n>, "threshold": 5, "paradigm": <paradigm>, "group": <grp_name>}` | false |

**注**：原代码中 fst/tst 的 8 号是一个共享 loop，会触发 2 次（一次 fst 一次 tst），但 paradigm 字段会区分。

**blocks_downstream 判定原则**：
- `SAMPLE.TOO_SMALL`（n<3）→ true（统计推断不可能可靠，必拦）
- `SAMPLE.UNDERPOWERED`（n<5）→ false（统计功效不足，但可跑描述）
- `MOTOR.*` / `SIGNAL.*`（运动/信号品质问题）→ false（可解读为混杂因素，data-analyst 自会标注 method_warnings）

**实施模板**（替换 1 号 warning）：

```python
# 旧（line 176-185）
data_quality_warnings.append(
    {
        "severity": "critical",
        "metric": "all",
        "message": (
            f"Group '{grp_name}' has n={sample_n} (<3). "
            "Statistical inference will be unreliable; descriptive statistics only."
        ),
    }
)

# 新
data_quality_warnings.append(
    {
        "severity": "critical",
        "code": "SAMPLE.TOO_SMALL",
        "metric": "all",
        "message": (
            f"Group '{grp_name}' has n={sample_n} (<3). "
            "Statistical inference will be unreliable; descriptive statistics only."
        ),
        "evidence": {
            "n": sample_n,
            "threshold": 3,
            "group": grp_name,
        },
        "blocks_downstream": True,
    }
)
```

其他 8 处按上表一一替换。**保持原 `message` 文本不变**（前端如果未升级展示，仍能渲染老 message；blocks_downstream 字段缺失也兼容老前端）。

**Sprint 2b 衔接**：dispatcher.py:235 的 `_ZM_LOW_DISTANCE_THRESHOLD = 10.0` 会在 Sprint 2a 搬到 zero_maze.yaml。Sprint 1 阶段保留为局部常量，evidence dict 里读这个常量即可。Sprint 2b 完工后会自动改为 catalog 传入。

### 2.2 dispatcher.py 单元测试

新建 `packages/ethoinsight/tests/test_dispatcher_warnings.py`（如果已存在则补充）：

| 测试 | 输入场景 | 期望 warning |
|---|---|---|
| `test_warning_sample_too_small` | epm/oft/任意范式，组内 n=2 | code=SAMPLE.TOO_SMALL, evidence={n:2, threshold:3, group:'X'}, blocks_downstream=True |
| `test_warning_epm_underpowered` | epm，组内 n=4 | code=SAMPLE.UNDERPOWERED, evidence.paradigm='epm', blocks_downstream=False |
| `test_warning_epm_low_entries` | epm，subject total_entry_count=5 | code=MOTOR.LOW_ENTRIES, evidence.subject=<name>, evidence.total_entry_count=5 |
| `test_warning_zm_low_distance` | zero_maze，subject distance_moved=8 | code=MOTOR.LOW_DISTANCE, evidence.distance_moved=8 |
| `test_warning_ldb_low_transitions` | light_dark_box，subject transition_count=3 | code=SIGNAL.LOW_TRANSITION_COUNT |
| `test_warning_fst_underpowered` | forced_swim，组内 n=4 | code=SAMPLE.UNDERPOWERED, evidence.paradigm='forced_swim' |
| `test_warning_no_legacy_remaining` | 跑各范式 critical case → assert 所有 warning 的 code != "LEGACY.UNCATEGORIZED" | **关键**：验证 dispatcher 端真填了，没退化到兜底 |

### 2.3 改动 data_analyst.py 读 warnings 分级

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

**当前状态**：read 现有 prompt + DataAnalystHandoff schema。需要把 warnings 真正"浮上水面"：

1. **subagent prompt workflow** 加一段（read_file 当前 prompt 上下文后插入）：

```
1.5 **读 quality warnings**: read_file handoff_code_executor.json,
   遍历 data_quality_warnings:
   - severity=critical AND blocks_downstream=true → method_warnings 前置一条:
     "[阻断级 {code}] {message}; 证据: {evidence}"
   - severity=critical AND blocks_downstream=false → method_warnings 加一条:
     "[严重 {code}] {message}"
   - severity=warning → method_warnings 加一条:
     "[提示 {code}] {message}"
   key_findings 首条若有阻断级警告,必须明示:
     "本次分析含 {critical_count} 条阻断级质量警告,统计结论的可靠性受限"
```

2. **DataAnalystHandoff** 在 Sprint 0 schema 基础上**额外加一个字段**（这是 Sprint 1 范围）：

   位置：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

   ```python
   class DataAnalystHandoff(BaseModel):
       # ... existing fields ...
       # === Sprint 1 新增 ===
       quality_warnings: list[DataQualityWarning] = Field(
           default_factory=list,
           description=(
               "从 handoff_code_executor.json 透传的 data_quality_warnings, "
               "保留完整结构供下游(report-writer / lead UI / 假设面板)按 code 分组渲染。"
           ),
       )
   ```

3. **seal_data_analyst_handoff tool 加 quality_warnings 参数**：

   位置：`packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`

   ```python
   @tool("seal_data_analyst_handoff", parse_docstring=True)
   def seal_data_analyst_handoff(
       status: str,
       key_findings: list[str] | None = None,
       outlier_findings: list[dict[str, Any]] | None = None,
       excluded_metrics: list[str] | None = None,
       method_warnings: list[str] | None = None,
       recommendations: list[str] | None = None,
       errors: list[str] | None = None,
       gate_signals: dict[str, Any] | None = None,
       # === Sprint 1 新增 ===
       quality_warnings: list[dict[str, Any]] | None = None,
       runtime: ToolRuntime[ContextT, ThreadState] = None,
   ) -> str:
       """..."""
       payload = {
           # ... existing fields ...
           "quality_warnings": quality_warnings or [],
       }
       return _seal_handoff(DataAnalystHandoff, "handoff_data_analyst.json", payload, runtime)
   ```

4. **GateSignals 加 critical_count**：

   位置：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

   ```python
   class GateSignals(BaseModel):
       model_config = ConfigDict(extra="allow")
       data_quality: dict[str, Any] = Field(default_factory=dict)
       statistical_validity: Literal["ok", "warning", "failed", "skipped"] = "ok"
       errors_count: int = 0
       # === Sprint 1 新增 ===
       quality_warnings_critical_count: int = Field(
           default=0,
           description="data-analyst 看到的 critical + blocks_downstream=true 警告数,lead 据此判断是否需要 ask_clarification (Sprint 5 in manual mode)。",
       )
   ```

### 2.4 改动 lead prompt 播报模板

**位置**：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

read 当前 prompt，在"派遣 data-analyst 后接收 handoff"或类似播报段加：

```
收到 data-analyst handoff 后,如果 gate_signals.quality_warnings_critical_count > 0,
向用户播报模板:
  "已收到 data-analyst 结果: {N} 条阻断级质量警告:
   - {critical_item_1}
   - {critical_item_2}
   ..."
不要把 evidence dict 原样念出来(那是给开发/调试用),用 message 字段呈现给用户。
```

具体 prompt 文本需要 read 当前 prompt 找到最合适的插入点（多数情况是 "data-analyst handoff 处理" 那一段）。

### 2.5 前端红字/橙字渲染规则

**位置**：`packages/agent/frontend/`（具体路径需要 read frontend/）

**渲染规则**（基于 DataQualityWarning 三字段联合判定）：

| severity | blocks_downstream | 颜色 | UI 标识 |
|---|---|---|---|
| critical | true | 🔴 红字 + 警示图标 | "阻断级警告" |
| critical | false | 🟠 橙字 + 警告图标 | "严重警告" |
| warning | — | 🟡 黄字 | "提示" |
| info | — | 🔵 蓝字 | "信息" |

**渲染源**：frontend 从 SSE stream 中读 `data_quality_warnings` 数组（通常通过 data-analyst handoff 透传或 code-executor handoff 直接读），每条按 severity + blocks_downstream 二维选颜色。

**展示字段**：title 用 `code`（如"阻断级警告 SAMPLE.TOO_SMALL"），body 用 `message`，可点击展开 `evidence` dict 看具体数字证据。

### 2.6 清理 Sprint 0/1 过渡兜底（最后一步）

**前置条件**：上面 §2.1 / §2.2 已完成且测试通过——dispatcher.py 已经正式填了 code/evidence/blocks_downstream。

**清理步骤**（严格按 Sprint 0 spec §7.1 执行）：

1. **删 `seal_handoff_tools.py` 的 LEGACY 兜底 block**
   ```bash
   # 定位（应该返回 1 个 block）
   grep -n "LEGACY.UNCATEGORIZED" packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py
   ```
   - 找到 `# === Sprint 0/1 过渡兜底...===` 到 `# === 过渡兜底结束 ===` 的整个 block
   - 删除整 block（包括上下注释）

2. **删 `DataQualityWarning._validate_code_namespace` 白名单的 LEGACY**
   - 位置：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`
   - 改 `allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD", "LEGACY"}` → `allowed = {"SAMPLE", "MOTOR", "SIGNAL", "METHOD"}`
   - 删该函数附近的 "Sprint 0/1 过渡期" 注释

3. **删过渡测试文件**
   - 整文件删 `packages/agent/backend/tests/test_seal_handoff_legacy_fallback.py`
   - 删 `test_data_quality_warning_schema.py` 中的 `test_warning_code_legacy_accepted_during_transition` 测试函数

4. **验证**：
   ```bash
   git grep -n "LEGACY.UNCATEGORIZED" packages/agent/backend/  # 必须返回空
   git grep -n "LEGACY" packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py  # 必须返回空
   ```

5. **如果以上 grep 非空，停下来排查**——可能 Sprint 0 实施过程中引入了额外兜底点（执行 agent 自由发挥）。

---

## 3. 测试要求

### 3.1 单元测试

| 文件 | 关键测试 |
|---|---|
| `tests/test_dispatcher_warnings.py` | 见 §2.2 全表 8 个测试 + `test_warning_no_legacy_remaining` |
| `tests/test_data_analyst_handoff_quality_warnings.py` | seal_data_analyst_handoff 传 quality_warnings list → 落盘后透传完整 |
| `tests/test_gate_signals_critical_count.py` | GateSignals 默认 critical_count=0；设置后能正确序列化 |

### 3.2 集成测试

| 测试 | 期望 |
|---|---|
| `test_dogfood_fst_with_small_sample` | mock fst 数据 n=2 → handoff_code_executor.json 含 code=SAMPLE.TOO_SMALL + blocks_downstream=true |
| `test_dogfood_epm_underpowered` | mock epm 数据 n=4 → handoff_code_executor.json 含 code=SAMPLE.UNDERPOWERED + blocks_downstream=false |
| `test_data_analyst_method_warnings_includes_critical` | 上游有 critical+blocks warning → data-analyst handoff.method_warnings 首条以 "[阻断级 SAMPLE.TOO_SMALL]" 开头 |
| `test_legacy_fallback_removed` | grep `LEGACY.UNCATEGORIZED` packages/agent/backend/ → 空（pre-commit 或 CI hook 验证） |
| `test_frontend_renders_red_for_blocks_downstream` | （需要前端测试基建）blocks_downstream=true → DOM 含 red css class |

### 3.3 测试基线

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight && .venv/bin/python -m pytest tests/ -q
cd /home/wangqiuyang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test
```

Sprint 0 完成后基线：ethoinsight ≥439 + agent backend ≥3063（Sprint 0 增 ~20 测试）。

Sprint 1 完成后基线：ethoinsight ≥445（Sprint 1 增 ~6） + agent backend ≥3065（增 ~2，减 ~3 legacy 测试）。

---

## 4. 实施顺序（task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | dispatcher.py 9 处 warning 改写（按 §2.1 映射表）| 1 天 |
| T2 | tests/test_dispatcher_warnings.py 编写 8 个测试（含 test_warning_no_legacy_remaining） | 1 天 |
| T3 | handoff_schemas.py 加 DataAnalystHandoff.quality_warnings + GateSignals.quality_warnings_critical_count | 0.25 天 |
| T4 | seal_data_analyst_handoff 加 quality_warnings 参数 + 单元测试 | 0.5 天 |
| T5 | data-analyst subagent prompt 加 §2.3 step 1.5 workflow | 0.5 天 |
| T6 | lead prompt 加播报模板（§2.4）| 0.25 天 |
| T7 | 前端红字/橙字渲染（§2.5）| 1 天 |
| T8 | 集成测试（§3.2 前 3 个）| 0.5 天 |
| T9 | **清理 LEGACY 过渡兜底**（§2.6 严格按 5 步）| 0.5 天 |
| T10 | 全量回归测试 + 修复退化（缓冲）| 0.5 天 |
| T11 | dogfood FST + EPM + ZM + LDB 4 范式各跑一次，确认红字/橙字正确渲染 | 0.5 天 |
| **合计** | | **6.5 天 ≈ 1.5 周** |

---

## 5. 风险与缓解

| 风险 | 缓解 |
|---|---|
| dispatcher.py 改完后 ethoinsight 历史测试退化 | T1 实施时只动 warning 结构、保留原 message 文本、不改触发条件；T2 跑测试 |
| data-analyst LLM 看到新 quality_warnings 字段不知道处理 | T5 prompt 明确步骤 1.5；上线后 dogfood 第一时间检查 method_warnings 首条是否真有 "[阻断级 ...]" |
| 前端展示退化（旧 thread 没有 blocks_downstream 字段）| frontend 读字段时用 `?? false` 兜底，旧数据走 warning 黄字默认值 |
| 清理 LEGACY 兜底过程中误删 | T9 严格按 5 步执行，第 4 步 grep 验证；若 grep 非空停下来排查（不强行删） |
| Sprint 0 实施 agent 给 LEGACY 加了 spec 没说的兜底点 | T9 step 5 抓得到（grep 非空）；处置：评估是否合理，合理则保留，不合理则删 |
| 微调阶段后 Sprint 2b 改 _ZM_LOW_DISTANCE_THRESHOLD 来源 | Sprint 1 evidence dict 里 threshold 用 `_ZM_LOW_DISTANCE_THRESHOLD` 变量读，不写死 10.0；Sprint 2b 改成从 catalog 读时只改一处 |

---

## 6. 验收 checklist

实施完成时，确认下列全部通过：

- [ ] dispatcher.py 9 处 warning 全部含 code/evidence/blocks_downstream 三新字段
- [ ] code 命名走 4 一级分类（SAMPLE/MOTOR/SIGNAL/METHOD），无 LEGACY
- [ ] blocks_downstream 判定：SAMPLE.TOO_SMALL → true，其他 → false（按 §2.1 表）
- [ ] DataAnalystHandoff schema 加 quality_warnings 字段
- [ ] GateSignals schema 加 quality_warnings_critical_count 字段
- [ ] seal_data_analyst_handoff 支持 quality_warnings 参数
- [ ] data-analyst subagent prompt 含步骤 1.5（读 warnings 分级写入 method_warnings）
- [ ] lead prompt 含 critical_count > 0 时的播报模板
- [ ] 前端按 blocks_downstream + severity 二维渲染颜色（红/橙/黄/蓝）
- [ ] 单元测试通过（`test_warning_no_legacy_remaining` 必绿）
- [ ] 集成测试 dogfood 跑通 4 范式 critical case
- [ ] **`git grep "LEGACY.UNCATEGORIZED" packages/agent/backend/` 返回空**
- [ ] **`git grep "LEGACY" packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` 返回空**
- [ ] `test_seal_handoff_legacy_fallback.py` 整文件已删
- [ ] 全量测试通过（ethoinsight ≥445 + agent backend ≥3065）

---

## 7. 不在 Sprint 1 范围

明确**不做**的事：

- ❌ DataQualityGuardrailProvider 拦截 task(data-analyst)（Sprint 5 做；本 sprint 只让 warning 浮上来，不阻断流程）
- ❌ parameter_audit 字段（Sprint 3 做：data-analyst 比对参数 vs 数据分布，本 sprint 只透传质量警告）
- ❌ analysis_config_id 实际计算（Sprint 4.5 做；Sprint 1 仍占位 "PENDING_SPRINT_4.5"）
- ❌ 调整 dispatcher.py 9 处 warning 的触发阈值/触发逻辑（保留原数字判定，只改结构；阈值调参属于 Sprint 2a 工作）
- ❌ 给 evidence 加更多上下文字段（仅按 §2.1 映射表填入最小必要字段；Sprint 3 / Sprint 7 假设面板可能进一步加字段）

---

## 8. 参考

- [SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) — Sprint 1 章节
- [Sprint 0 实施 spec](2026-05-28-sprint-0-handoff-schema-foundation-design.md) — §7.1 Sprint 1 完工清理步骤
- dispatcher.py 9 处 warning：`packages/ethoinsight/ethoinsight/metrics/dispatcher.py:169-300`
- DataQualityWarning schema：Sprint 0 已定义于 handoff_schemas.py
- DataAnalystHandoff schema：handoff_schemas.py:208-244
- data-analyst subagent：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- lead prompt：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- frontend SSE 渲染：`packages/agent/frontend/`（具体路径执行时 read）
