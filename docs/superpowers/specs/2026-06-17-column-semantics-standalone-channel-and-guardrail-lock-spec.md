# Spec: column_semantics 独立写入通道缺失 + guardrail 模板锁误判（Sprint 1 合入回归三 bug 链）

> 日期：2026-06-17
> 类型：bug 修复（架构对称缺口收口 — Sprint 1 列语义合入引入的回归）
> 前置：Sprint 1 列语义对齐（commit `91c75a93`/`76b2a3a9`，已合 dev）。本 spec 修该合入引入的三处链式回归。
> 状态：待 review → 批准后新 worktree 实施。
> 触发源：2026-06-17 本地 EPM dogfood（thread `c95df975`，28 个真实 FewZones 文件，列名 `open`/`closed`）。用户明确："反复出现！！！我们之前没有过这种错误！！！"

---

## 0. 背景与症状

2026-06-17 本地 EPM dogfood，真实 FewZones 数据（control=7 / treatment=21，自定义归属列 `open`/`closed`）。Gate 1 已完成（`ev19_template=PlusMaze-FewZones` 落盘）。agent 随后要落盘**列语义对齐**（`column_semantics`）和**分组信息**（`resolved_facts`），却陷入一连串卡死与状态丢失：

1. agent 调 `set_experiment_paradigm(column_semantics={...}, resolved_facts=[...])`（不带 paradigm 字段）→ **guardrail 拦截** `template_already_set`：「ev19_template 已设置为 'PlusMaze-FewZones'，不允许中途修改」。
2. agent 被迫加 `confirm_template_change=True` 重试（语义错误：它根本没改模板）。
3. 调用过了 guardrail，但 **column_semantics 没写进 experiment-context.json** → `prep_metric_plan` 反复报 `columns_missing: ['in_zone_open_arms_*']`。
4. agent 反复脑补排查（怀疑 resolves_to 写错、怀疑 binary 列、反复读 skill），最后改用**全字段 Gate 1 调用**塞 column_semantics 才写进去 —— 但这一下**把之前落盘的 `resolved`（分组）clobber 丢了** → "列语义已落盘，但分组信息丢失了"。
5. 螺旋继续：补分组又可能丢列语义，prep_metric_plan 始终缺一半。

**这不是 prompt 问题，也不是 agent 笨**。是 Sprint 1 合入时新加了 `column_semantics` 能力，却没在三个协作点把它接对：工具体没给它独立写入通道、guardrail 模板锁没认识这个新用法、Gate 1 重建 data 时会 clobber 既有状态。三者链式放大成"反复卡死 + 状态丢失"。

---

## 1. 根因：三处真 bug，链式放大（同源 = Sprint 1 列语义合入未补全协作面）

### Bug 1 — guardrail 模板锁误判：没碰模板的调用也被锁

**位置**：`packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py:98-115`

```python
ctx = self._read_context(workspace)
if ctx and ctx.get("ev19_template"):
    if not args.get("confirm_template_change"):
        return GuardrailDecision(allow=False, ... code="ethoinsight.template_already_set" ...)
```

**缺陷**：只要 `ev19_template` 已落盘且本次没传 `confirm_template_change`，**无条件拦**。但本次调用根本没传 `ev19_template`、没传 `paradigm`，只想加 `column_semantics`/`resolved_facts`。锁的语义本应是"防止**中途改模板**"，实际却拦成了"模板设过之后任何 set_experiment_paradigm 调用"。

**对照已有的正确写法**：line 69 早就给 `acknowledge_quality` 开了正交放行：

```python
# Gate 2 quality acknowledgement mode is orthogonal to template setting —
# it neither sets nor changes ev19_template / paradigm fields. Pass through ...
if args.get("acknowledge_quality") and not (args.get("ev19_template") or args.get("paradigm")):
    return GuardrailDecision(allow=True, ...)
```

→ 这个 pass-through 的判据"本次没传 ev19_template/paradigm = 不改模板 = 放行"**本身是对的**，但写死只认 `acknowledge_quality`。Sprint 1 加了 `column_semantics`、Spec B 加了 `resolved_facts`，都同样"正交于模板设置"，却没被纳入这个放行。

### Bug 2 — 工具体没有独立 column_semantics 写入通道（核心阻塞）

**位置**：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py:343-578`

工具体的早返通道布局：

| 通道 | 触发 | 是否要求 paradigm 全字段 | 行号 |
|---|---|---|---|
| Gate 2（`acknowledge_quality`） | `acknowledge_quality=True` | 否（读 existing 改 gate_completed） | 435 |
| Standalone `resolved_facts` | `resolved_facts` 且无 Gate1/2 | 否（读 existing 合并 resolved） | 469 |
| **column_semantics** | —— | **无独立通道** | —— |
| Gate 1（paradigm 确认） | 兜底 | **是**（缺一即 error，line 499-501） | 495 |

`column_semantics` **只在 Gate 1 路径里处理**（line 547-552）。所以 `set_experiment_paradigm(column_semantics={...})` 单独调用 → 落进 Gate 1 → `required` 缺 `paradigm/paradigm_cn/category/subject/ev19_template` → 返 `{"status":"error","message":"Missing required fields for Gate 1"}` → **column_semantics 根本没写盘**。

这是直接对称缺口：`acknowledge_quality` 和 `resolved_facts` 都有各自的独立 early-return，唯独 `column_semantics` 没有。这正是 `prep_metric_plan` 反复 `columns_missing` 的真因（symptom 3）。

### Bug 3 — Gate 1 重建 data 时 clobber 既有状态

**位置**：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py:533-543`

```python
data = {
    "paradigm": paradigm, "paradigm_cn": paradigm_cn, "category": category,
    "subject": subject, "ev19_template": ev19_template,
    "paradigm_confirmed_at": ..., "gate_completed": gate_completed,
    "parameter_overrides": overrides, "analysis_config_id": config_id,
}
# 之后只回填【本次】传入的 column_semantics（547）和 resolved_facts（555）
```

**缺陷**：Gate 1 从零 `data = {...}` 重建，只回填**本次调用**带的 `column_semantics`/`resolved_facts`，**丢掉之前已落盘的 `resolved`（分组）、`column_semantics`、`column_aliases`**。Bug 2 逼 agent 改用全字段 Gate 1 调用塞 column_semantics 时，正好触发这个 clobber → 分组丢失（symptom 4）。

注意 line 522-526 已经特意保留了 `gate2_quality_acknowledged`，说明作者**意识到 Gate 1 重调要保留既有状态**，但只保了 gate_completed 一项，漏了 `resolved`/`column_semantics`/`column_aliases`。

---

## 2. 为什么是回归 + 为什么测试没抓到

- **回归来源**：三处都指向 Sprint 1（`91c75a93`/`76b2a3a9`）。guardrail 模板锁（`a05f2de1`）比 Sprint 1 更早，Sprint 1 加 `column_semantics` 时没回头更新锁的放行条件；工具体加 column_semantics 时塞进 Gate 1 路径而没像 resolved_facts 那样开独立通道；Gate 1 clobber 是 column_semantics 入 Gate 1 路径后才暴露。用户"之前没有过"的直觉准确。
- **测试假绿**：现有 `tests/test_column_semantics.py:186` 的 `test_set_experiment_paradigm_with_column_semantics` **把 column_semantics 和全部 paradigm 字段一起传**（line 224-233），走 Gate 1 路径恰好能写盘 → 掩盖了 Bug 2（真实 agent 是**分开调**的）。且该测试**直接调 `.func()` 不过 guardrail** → 没覆盖 Bug 1。Bug 3 没有任何"先落盘 resolved 再 Gate1 重调"的测试。
- **教训**：测试要复刻 agent 真实调用序列（Gate1 → 单独 column_semantics → 单独 resolved_facts），不能把多步合成一步测。

---

## 3. 修法（治本：补全对称缺口，全部复用既有纯函数，零新方法论）

三处修复正交、可独立验证。

### 3.1 Bug 2 — 工具体加独立 column_semantics 写入通道

在 `set_experiment_paradigm_tool` 的 **Gate 2 通道之后、Standalone resolved_facts 通道之前**插入独立通道（方案 A，见下方 ⚠️ 为何必须在 resolved_facts 之前），与既有两通道对称。最终通道顺序：Gate2(acknowledge_quality) → **standalone column_semantics(含可选 resolved_facts)** → standalone resolved_facts(纯) → Gate1。

```python
# --- Standalone column_semantics path (Sprint 1: no Gate 1/2 params) ---
# Symmetric with the resolved_facts standalone path: read existing context,
# merge column_semantics + derive column_aliases, write back preserving all
# other fields. Allows the agent to align columns separately AFTER Gate 1,
# instead of being forced through the full-fields Gate 1 path (which would
# clobber resolved/groups — see Bug 3).
if column_semantics is not None and isinstance(column_semantics, dict):
    if existing is None:
        return json.dumps(
            {"status": "error",
             "message": "No experiment-context.json found. Call set_experiment_paradigm with paradigm fields first."},
            ensure_ascii=False,
        )
    data = dict(existing)                                  # 继承既有全部字段
    cs = _normalize_column_semantics(column_semantics)     # 复用既有纯函数
    data["column_semantics"] = cs
    aliases = _derive_column_aliases(cs)                   # 复用既有纯函数
    if aliases:
        data["column_aliases"] = aliases

    # resolved_facts 可与 column_semantics 同次传（与其他通道一致的组合语义）
    if resolved_facts:
        _apply_resolved_facts(data, resolved_facts)

    path = Path(actual_workspace) / "experiment-context.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    if resolved_facts:
        _persist_resolved_facts_to_memory(resolved_facts, _thread_id_from_runtime(runtime))

    resp: dict = {"status": "ok", "path": str(path), "column_semantics_saved": True}
    if resolved_facts:
        resp["resolved_facts_saved"] = len(resolved_facts)
    return json.dumps(resp, ensure_ascii=False)
```

**通道放置顺序**（重要，见 §6.1）：必须在 standalone `resolved_facts` 通道（line 469）**之前**，且本通道内部处理可选的 `resolved_facts`。否则 `set_experiment_paradigm(column_semantics=..., resolved_facts=...)` 会被 resolved_facts 通道（条件 `if resolved_facts:`）先截走、column_semantics 丢失。resolved_facts 通道保持原样，继续处理"只传 resolved_facts"的情况。

**组合语义对齐**：本通道处理"带 column_semantics 但不带 paradigm 字段"的调用。若同时带 paradigm 字段（即真 Gate 1）走 Gate 1 路径（3.3 修好后那里也不再 clobber）。判据用 `column_semantics is not None`，与 Gate 1 入口的 `required` 缺失判据互斥（Gate 1 要求 paradigm 全字段，本通道恰好是 paradigm 字段缺失时的归宿）。

### 3.2 Bug 1 — guardrail 模板锁只在"本次真要改模板"时触发

`ev19_template_provider.py` line 98-115 的锁前，把 line 69 的正交放行**泛化**为"本次没传 paradigm 字段 = 不改模板 = 放行模板锁检查"：

```python
# 现状（line 69，只认 acknowledge_quality）：
if args.get("acknowledge_quality") and not (args.get("ev19_template") or args.get("paradigm")):
    return GuardrailDecision(allow=True, ...)

# 改为：任何"不触碰模板"的 set_experiment_paradigm 调用都放行模板锁
# （acknowledge_quality / column_semantics / resolved_facts / parameter_overrides 等
#  正交用法，均不设置也不改 ev19_template / paradigm）
if not (args.get("ev19_template") or args.get("paradigm")):
    return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])
```

**语义**：模板锁的职责是"防止**中途切换 ev19_template**"。判据应是"本次调用是否试图设置/改变模板"，而**本次有没有传 ev19_template/paradigm** 就是这个判据的直接信号。没传 = 不改模板 = 放行（让工具体的 standalone 通道处理）。传了 = 可能改模板 = 走下面的 ambiguous + already_set 检查。

**注意保留**：line 74-96 的 **ambiguous template 检查**（identify 返回 2-3 候选时强制用户选）依然要在"传了 ev19_template"时生效——所以泛化放行必须放在 ambiguous 检查**之前**只对"没传模板字段"短路，传了模板字段的仍往下走 ambiguous + already_set。

> 实施细节：现有结构是 line 69 先 `acknowledge_quality` 短路，然后 line 72 起 `workspace is not None` 块内做 ambiguous + already_set。把 line 69 的短路条件改成上面的泛化版即可（删掉 `acknowledge_quality and` 前缀，只留 `not (ev19_template or paradigm)`）。ambiguous/already_set 都在"传了模板字段"分支内，自然不受影响。

### 3.3 Bug 3 — Gate 1 路径继承 existing，不 clobber

`experiment_context.py` line 533 的 `data = {...}` 改为先继承 existing 再覆写本次字段：

```python
# 现状：从零重建，丢既有 resolved / column_semantics / column_aliases
data = { "paradigm": paradigm, ... }

# 改为：继承既有上下文（保 resolved/column_semantics/column_aliases 等），
# 再用本次 Gate 1 字段覆写。与 line 522-526 已对 gate2_quality_acknowledged
# 做的保留同一思路，补全所有应保留字段。
base = dict(existing) if isinstance(existing, dict) else {}
data = {
    **base,
    "paradigm": paradigm,
    "paradigm_cn": paradigm_cn,
    "category": category,
    "subject": subject,
    "ev19_template": ev19_template,
    "paradigm_confirmed_at": datetime.now(UTC).isoformat(),
    "gate_completed": gate_completed,
    "parameter_overrides": overrides,
    "analysis_config_id": config_id,
}
```

**注意**：`{**base, ...}` 后，本次若也传了 `column_semantics`（line 547-552）/`resolved_facts`（line 555-556）会在其后覆写/合并 base 中的旧值——这是正确的（本次显式传的优先）。本次没传的，base 里的旧值被保留——这正是要修的。`gate_completed` 已由 line 522-526 单独算好（line 540 覆写 base 的旧 gate_completed，正确）。

**语义边界**：Gate 1 是"用户（重新）确认范式/模板"。重新确认时**保留**已对齐的列语义和已解析的分组是对的（用户没说要清掉它们）；若用户真要换模板（`confirm_template_change=True` 且新模板与旧不同），列语义可能失效——但这属于 follow-up 的"换模板是否该清列语义"决策，本 spec **不处理换模板场景的列语义失效**（那是另一个正交问题），只修"重调 Gate 1 不该静默丢既有状态"。

---

## 4. 测试（TDD，红→绿，复刻 agent 真实调用序列）

新增/扩展 `packages/agent/backend/tests/test_column_semantics.py`（或新建 `test_column_semantics_standalone.py`）。**所有测试必须复刻 agent 真实"分步调用"序列，不许把多步合一步。**

### 4.1 Bug 2 — 独立 column_semantics 通道

- `test_standalone_column_semantics_after_gate1_writes`：先 Gate1（全字段，无 column_semantics）→ 再**只传 column_semantics**（无 paradigm 字段）→ 断言返回 `status=ok` 且磁盘 `column_semantics`/`column_aliases` 已写、`paradigm`/`ev19_template` 仍在（继承未丢）。**这条在修复前必红**（现状返回 Missing required fields，column_semantics 不写盘）。
- `test_standalone_column_semantics_without_existing_errors`：无 existing 时只传 column_semantics → 返 error（提示先 Gate1），不崩。

### 4.2 Bug 1 — guardrail 不再误锁

- `test_guardrail_allows_column_semantics_only_when_template_set`：experiment-context.json 已有 ev19_template，构造 `GuardrailRequest(tool_name="set_experiment_paradigm", tool_input={"column_semantics": {...}})`（**不传 confirm_template_change**）→ `evaluate` 返 `allow=True`。**修复前必红**（现状返 template_already_set）。
- `test_guardrail_still_blocks_real_template_change`：`tool_input={"ev19_template": "OtherTemplate-AllZones"}` 无 confirm_template_change → 仍 `allow=False` code=`template_already_set`（守锁不被改松）。
- `test_guardrail_allows_resolved_facts_only_when_template_set`：纯 resolved_facts 调用同样放行（泛化的附带正收益，坐实）。
- `test_guardrail_ambiguous_check_still_fires_when_template_passed`：ambiguous template_candidates 存在 + 传了 candidate ev19_template + 无 user_confirmed_template → 仍 `allow=False` code=`template_not_confirmed`（守 ambiguous 检查不被泛化放行旁路）。

### 4.3 Bug 3 — Gate1 重调不 clobber

- `test_gate1_rerun_preserves_resolved_facts`：先 Gate1 → 再 standalone resolved_facts（落盘 groups）→ 再**全字段 Gate1 重调**（无 resolved_facts）→ 断言磁盘 `resolved` 仍在。**修复前必红**（现状被 clobber）。
- `test_gate1_rerun_preserves_column_semantics`：先 Gate1 → standalone column_semantics（落盘）→ 全字段 Gate1 重调 → 断言 `column_semantics`/`column_aliases` 仍在。**修复前必红**。
- `test_column_semantics_plus_resolved_facts_same_call`（方案 A 边界）：一次调用同传 column_semantics + resolved_facts（无 paradigm）→ 断言两者都写盘。

### 4.4 端到端链路（坐实三 bug 合修后 dogfood 路径通）

- `test_full_alignment_sequence_then_prep_succeeds`（可选，重量级）：Gate1 → standalone column_semantics(open→open_arms, closed→closed_arms) → standalone resolved_facts(groups) → 调 `prep_metric_plan` → 断言不再 `columns_missing`、plan_metrics 带 `parameters_in_use: {open_arm_zones:["open"]}`。这条把本 spec 与 PR#141 列对齐接线连起来验证。若过重可留 dogfood 复跑替代。

---

## 5. 红→绿验证 + 裸导入

1. **拆 red/green commit**：先加测试（4.1/4.2/4.3 的"修复前必红"项）跑红，再实施 3.1/3.2/3.3 跑绿。两 commit 留痕，证明测试真的咬住 bug（避免假绿，参考 memory `feedback_pr115_stage1_equivalence_baseline_is_hollow_error_string`）。
2. **改了 guardrails/ 和 middlewares/ 核心，必须裸导入两生产入口**（守 CLAUDE.md 导入环铁律）：
   ```bash
   cd packages/agent/backend
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 exit 0。
3. **worktree 独立 venv 铁律**（守 memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`）：若用 worktree，先 `cat .venv/.../_editable_impl_*.pth` 确认 editable 指向 worktree 再跑测试；不确定就 importlib 显式加载 worktree 源。
4. `make test`（backend）全绿（扣除已知 15 failed executor 基线债，见 handoff §2.4）。

---

## 6. 风险与边界

1. **通道顺序错位**（最大风险）：standalone column_semantics 通道若放错位置（放在 resolved_facts 之后），同传 column_semantics+resolved_facts 时 resolved_facts 通道会先截走、column_semantics 丢。**按方案 A 把 column_semantics 通道放 resolved_facts 之前并在内部处理 resolved_facts**，4.3 的同传测试坐实。
2. **guardrail 泛化别放过真·换模板**：泛化放行只对"没传 ev19_template/paradigm"短路；传了的仍走 ambiguous + already_set。4.2 的两条守护测试咬住。
3. **Gate1 继承别覆盖该更新的字段**：`{**base, ...}` 后本次 Gate1 字段在后覆写，确保 paradigm/ev19_template/analysis_config_id 是本次的新值，不被 base 旧值盖回。
4. **换模板时列语义失效不在本 spec**：3.3 保留既有 column_semantics 是"重确认同范式"的正确语义；真换模板（新旧不同）时旧列语义可能失效，属正交 follow-up，本 spec 不处理，在实施 commit message 注明。
5. **不碰 prep_metric_plan / resolve / dispatcher**：本 spec 只修"列语义/分组能正确落盘"这一前置环节。落盘后的列对齐计算由已合的 PR#141 负责，正交。
6. **SSOT 守恒**：column_semantics 的 SSOT 是 experiment-context.json，本 spec 只修写入路径不改 schema、不引第二存储。

---

## 7. 实施清单（给下一个 agent）

1. 读本 spec + memory `feedback_single_source_of_truth` + `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib` + CLAUDE.md「导入环铁律」。
2. 新 worktree（基线 dev `0b2be473` 或更新）。
3. 先写 4.1/4.2/4.3 必红测试 → 跑红留 commit。
4. 实施 3.1（方案 A：column_semantics 通道置于 resolved_facts 之前）→ 3.2（guardrail 泛化）→ 3.3（Gate1 继承）。
5. 跑绿 + 裸导入两入口 + make test。
6. push，提醒用户建 PR。
7. 合并后用本次 dogfood 同数据复跑（`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`，列名 open/closed），确认：Gate1 → 单独列语义 → 单独分组 → prep_metric_plan 不再 columns_missing、不再丢分组、guardrail 不再误锁。
