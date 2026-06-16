# Spec 2 — path_sequence guardrail 感知 n=1：n<2 时 data-analyst 非必需

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree（**独立于 Spec 1**）
> 来源：EPM n=1 dogfood（thread `1bda1847`）发现的 3.1
> 性质：guardrail 逻辑改动，**改动面比 Spec 1 大**（碰 path_sequence provider 的核心判定），需谨慎 + 充分回归。
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 背景与证据（已现场核实）

### 问题 3.1：n=1 场景下 guardrail 强制 data-analyst，与 lead 的 n=1 fast-path 规则冲突

**现象**（thread 1bda1847）：lead 在 n=1 下按 prompt 规则**跳过 data-analyst**、直接派遣 chart-maker，被 `path_sequence` guardrail 拒：
```
Guardrail denied: tool 'task' was blocked (ethoinsight.path_sequence_violation).
Reason: 按 E2E_FULL_ASKVIZ 路径，chart-maker 之前必须先完成 data-analyst。
请先 task(data-analyst) 完成该步骤。
```

**两条规则直接矛盾**（白纸黑字）：
- **lead prompt（n=1 fast-path）** `agents/lead_agent/prompt.py:1007-1016`：
  ```
  6. n=1 快速路径判定: prep_metric_plan 返回 plan_summary.subject_count...
     若任一组 n < 2（无法做组间统计检验）:
     - 正常派遣 code-executor
     - code-executor 完成后，**跳过 data-analyst**（专业判读在 n=1 时没有统计基础）
     - lead 自己写简短描述性摘要...
     - chart-maker 和 report-writer 仍可派遣
     - 流水线: code → **跳过 data** → lead 出描述性摘要 → ask(viz?) → ask(report?)
  ```
- **path_sequence 序列** `guardrails/path_registry.py:42-46`（`E2E_FULL_ASKVIZ`）：
  ```python
  "E2E_FULL_ASKVIZ": [
      Step("dispatch", "code-executor"),
      Step("dispatch", "data-analyst"),          # ← 无条件必需
      ...
      Step("dispatch", "chart-maker", condition="viz==yes"),
      ...
  ]
  ```
  data-analyst 是**无条件**前置步骤。guardrail 在派遣 chart-maker 时检查所有前序 dispatch 的 handoff 是否存在（`path_sequence_provider.py:187-200`），发现 `handoff_data_analyst.json` 不存在 → 拒。

**后果**：chart-maker 被拦 → lead 被迫派遣 data-analyst 满足 guardrail → data-analyst 在 n=1 无统计基础下又撞上 seal 问题（Spec 1 的 3.2）。**3.1 是 3.2 的触发上游**。

prompt 说"跳过 data-analyst"，guardrail 说"data-analyst 必需"。**这是 prompt 与 guardrail 的契约冲突**，须让 guardrail 感知 n=1。

---

## 1. ⚠️ 关键设计约束（执行 agent 必读，否则会走错路）

### 1.1 `Step.condition` 是死字段，不要靠它

`path_registry.py` 的 `Step` 有 `condition: str | None`（line 26-32），且 chart-maker 写了 `condition="viz==yes"`（line 46）。**但 `path_sequence_provider` 完全不读 `condition`**——它的 missing-predecessor 循环（`path_sequence_provider.py:189-197`）只判 `step.kind != "dispatch"` 就 skip，从不 evaluate `condition`。

**实证**：chart-maker 的 `viz==yes` 根本不是 path_sequence 在管，是另一个独立 provider `intent_post_step_ask_gate_provider.py`（viz ask gate）强制的。所以 `Step.condition` 在 path_sequence 里是**装饰性死代码**。

**结论**：**不能**简单"给 data-analyst step 加 `condition='n>=2'`"就指望生效——provider 不看 condition。要么（方案 A）让 provider 的循环新增"n<2 时把 data-analyst 当 optional 跳过"的特判，要么（方案 B）实现通用 condition 求值。见 §2 的方案选择。

### 1.2 n 的可靠数据源 = `handoff_code_executor.json`，不是 groups.json

**实证**（thread 1bda1847 workspace）：
- `groups.json` 是**空文件**（cat 无输出）——guardrail 不能靠它判 n。
- `plan_metrics.json` 的 `plan_summary` = `{}`，`inputs.groups` 无 → 也不可靠。
- **但 `handoff_code_executor.json` 有可靠 n 信号**，且 guardrail 拦 chart-maker 时它**一定已存在**（它是 chart-maker 的前置 precondition，见 `_check_plan_precondition` chart-maker 分支要求 handoff_code_executor.json 非空）：
  ```python
  per_subject keys: ['Subject 1']           # 1 个 subject
  metrics_summary['All'][metric]['n'] == 1  # 每组 n
  ```

**判 n<2 的稳健方法**（二选一或都用，取保守）：
- `len(handoff_code_executor["per_subject"])` —— 总 subject 数
- `max(mm[metric]["n"] for group, mm in metrics_summary.items() for metric in mm)` —— 各组 n 的最大值
- n<2 判据：**任一组最大 n < 2**（与 prompt 的"任一组 n<2"对齐）。单文件无分组时 group='All'，per_subject 长度=1 → n=1。

### 1.3 这是「让 data-analyst 变 optional」，不是「删 data-analyst」

n≥2 时 data-analyst 仍必需，序列不变。只有 n<2 时 path_sequence 在判 chart-maker/report-writer 的前序时，**不把缺失的 handoff_data_analyst.json 计入 missing**。

---

## 2. 方案选择（执行 agent 按推荐方案做，备选方案记录备查）

### ✅ 推荐方案：在 `path_sequence_provider` 的 missing-predecessor 循环里加 n=1 特判

最小侵入、最贴合现有代码结构。在 `path_sequence_provider.py:187-200` 的循环中，当某个前序 step 是 `data-analyst` 且**当前 workspace 的 handoff_code_executor.json 显示 n<2** 时，**跳过该 step**（不计入 missing）。

**为什么选这个**：
- 不动死字段 `condition`（避免假装它能用）。
- 不动 `path_registry.py` 的序列定义（序列仍是完整的，guardrail 在运行时按数据放宽）。
- 改动集中在一个函数，易测、易回滚。

### 备选方案（不推荐，记录理由）：
- **B1 实现通用 condition 求值**：给 path_sequence 加 `condition` evaluator，让 `Step("dispatch","data-analyst", condition="n>=2")` 生效。**否决**：改动面大（要定义 condition DSL + 求值上下文 + 让 viz==yes 也走这套），超出本次范围，且 viz 已被另一 provider 管，重复。
- **B2 在 prompt 层降级 intent 为 `E2E_FULL_ASKVIZ_N1`**（报告 §3.1 提的）：**否决**：要新增一条 intent + 路由逻辑 + guardrail 序列，且 intent 在派遣 code-executor 前就定了、那时未必知道 n（n 来自 code-executor handoff）。时序不对。

---

## 3. 改动清单（推荐方案，源码 1 处 + helper + 测试）

### 文件：`packages/agent/backend/packages/harness/deerflow/guardrails/path_sequence_provider.py`

**Step 1**：新增一个 helper，从 workspace 的 handoff_code_executor.json 读 n，判是否 n<2。放在模块函数区（`_check_plan_precondition` 附近）：

```python
def _is_single_subject_run(workspace: str) -> bool:
    """n<2 判定：读 handoff_code_executor.json，任一组最大 n < 2 → True。

    数据源理由：groups.json/plan_metrics.json 在本场景为空/无 n 字段；
    handoff_code_executor.json 在 chart-maker/report-writer 被拦时一定已存在
    （它是这两者的 plan precondition），且含可靠的 per_subject + metrics_summary.n。
    Fails open（返回 False = 不放宽）当文件缺失/解析失败/无 n 信号——
    即拿不准时维持原有"data-analyst 必需"行为，不误放行。
    """
    try:
        p = Path(workspace) / "handoff_code_executor.json"
        if not p.exists() or p.stat().st_size == 0:
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        # 优先 per_subject 长度
        per_subject = data.get("per_subject") or {}
        if isinstance(per_subject, dict) and len(per_subject) >= 2:
            return False
        # 再看 metrics_summary 各组 n 的最大值
        ms = data.get("metrics_summary") or {}
        max_n = 0
        for _group, metrics in ms.items():
            if not isinstance(metrics, dict):
                continue
            for _metric, stats in metrics.items():
                if isinstance(stats, dict) and isinstance(stats.get("n"), int):
                    max_n = max(max_n, stats["n"])
        # per_subject 给了 1，或 metrics 给了 max_n<2 → 单 subject
        if len(per_subject) == 1 or (max_n == 1):
            return True
        # per_subject 空 + 无 n 信号 → 拿不准，fail open（必需）
        if not per_subject and max_n == 0:
            return False
        return max_n < 2
    except Exception:
        return False
```
> ⚠️ **已核实**：该文件顶部有 `from pathlib import Path`（line 29），**但没有 `import json`**。执行 agent **必须在顶部 import 区补 `import json`**（放在 `import re` 旁，line 26 附近），否则 helper 抛 NameError。

**Step 2**：在 `evaluate()` 的 missing 循环（line 188-197）中，跳过 n<2 时的 data-analyst：

```python
        # Verify all preceding dispatch steps have completed (handoff file exists)
        missing: list[str] = []
        single_subject = _is_single_subject_run(workspace)  # 新增
        for i in range(target_idx):
            step = steps[i]
            if step.kind != "dispatch":
                continue
            # n<2 fast-path: data-analyst 在单 subject 时非必需（与 lead prompt n=1 规则对齐）
            if step.target == "data-analyst" and single_subject:   # 新增
                continue                                            # 新增
            handoff_name = to_handoff_name(step.target)
            handoff_path = Path(workspace) / f"handoff_{handoff_name}.json"
            if not handoff_path.exists():
                missing.append(step.target)
```

> 注意：`workspace` 在此处已确认非空（line 183-185 已 gate）。`_is_single_subject_run` 每次 evaluate 调一次、只读一个小 JSON，开销可忽略。

**Step 3（重要，别漏）**：`_check_plan_precondition` 里 chart-maker/report-writer 的 deny 消息（`path_sequence_provider.py` 约 line 119）写的是"请先按路径依次派遣 code-executor → **data-analyst** → chart-maker"。这个 precondition 只检查 `handoff_code_executor.json` 存在性，**不检查 data-analyst**，所以它本身不会误拦 n=1（只要 code-executor handoff 在）。**但消息文案在 n=1 时有误导**（让用户以为还要 data-analyst）。**本 spec 不改这条 precondition 的逻辑**（它没坏），文案是否调整留作可选 polish，不强制。

---

## 4. 测试（TDD，先写 red）

放 `packages/agent/backend/tests/`，新建 `test_path_sequence_n1.py`（或追加到现有 `test_path_sequence*.py`）。

测试要点：构造一个临时 workspace + mock contextvars（`_lead_workspace` / `_lead_messages`），覆盖：

```python
import json
from pathlib import Path
from deerflow.guardrails.path_sequence_provider import (
    PathSequenceProvider, _is_single_subject_run, _lead_workspace, _lead_messages,
)

def _write_ce_handoff(ws: Path, n: int, subjects: int):
    """造一个 handoff_code_executor.json，n 由 metrics_summary 给，subjects 由 per_subject 给。"""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "handoff_code_executor.json").write_text(json.dumps({
        "status": "completed", "summary": "x", "paradigm": "epm",
        "per_subject": {f"Subject {i+1}": {} for i in range(subjects)},
        "metrics_summary": {"All": {"open_arm_time": {"mean": 1, "n": n}}},
    }), encoding="utf-8")

class TestSingleSubjectDetection:
    def test_n1_detected(self, tmp_path):
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        assert _is_single_subject_run(str(tmp_path)) is True

    def test_n2_not_single(self, tmp_path):
        _write_ce_handoff(tmp_path, n=2, subjects=2)
        assert _is_single_subject_run(str(tmp_path)) is False

    def test_missing_handoff_fails_open(self, tmp_path):
        # 无 handoff → 不放宽（保守，data-analyst 仍必需）
        assert _is_single_subject_run(str(tmp_path)) is False

class TestN1ChartMakerAllowedWithoutDataAnalyst:
    """red 锚点：修复前 n=1 派 chart-maker 缺 data-analyst handoff 被拒；修复后放行。"""
    def _run(self, tmp_path, subagent):
        # 造 n=1 的 code-executor handoff（chart-maker 的 plan precondition），不造 data-analyst handoff
        _write_ce_handoff(tmp_path, n=1, subjects=1)
        tok_ws = _lead_workspace.set(str(tmp_path))
        # mock 出 E2E_FULL_ASKVIZ intent（按现有测试如何注入 intent 的方式；可能需 set _lead_messages）
        tok_msg = _lead_messages.set([_fake_human_msg_with_intent("E2E_FULL_ASKVIZ")])
        try:
            req = _fake_guardrail_request(tool_name="task", subagent_type=subagent)
            return PathSequenceProvider().evaluate(req)
        finally:
            _lead_workspace.reset(tok_ws); _lead_messages.reset(tok_msg)

    def test_n1_chartmaker_allowed(self, tmp_path):
        decision = self._run(tmp_path, "chart-maker")
        assert decision.allow is True   # 修复前 False（path_sequence_violation: 缺 data-analyst）

    def test_n2_chartmaker_still_requires_data_analyst(self, tmp_path):
        _write_ce_handoff(tmp_path, n=2, subjects=2)  # 覆盖成 n=2
        decision = self._run(tmp_path, "chart-maker")
        assert decision.allow is False  # n≥2 仍需 data-analyst（不回归）
        assert "data-analyst" in decision.reasons[0].message
```

> 执行 agent **必须先读现有测试样板**（已核实存在）：
> - `tests/test_path_sequence_provider.py`（9184 字节）—— 含 `_make_request(subagent_type)` helper（line 40）、`_lead_workspace.set(str(tmp_path))`（line 48）、`reset_contextvars` fixture、以及 `test_chart_maker_with_handoff_allowed`（line 89）/`test_chart_maker_missing_handoff_denied`（line 81）等**正是本 spec 要扩展的同款测试**。**直接复用 `_make_request` 和它的 intent 注入方式**，不要凭空发明 mock。
> - `tests/test_path_sequence_plan_precondition.py`（6894 字节）—— precondition 分支的测试样板。
>
> 上面测试代码里的 `_fake_*` 占位符，全部替换成 `test_path_sequence_provider.py` 现有的 `_make_request` + 现有 intent 注入手法（看它现有测试如何让 `_extract_latest_intent` 返回 `E2E_FULL_ASKVIZ`——大概率通过 `_lead_messages.set([...])` 注入特定格式的 message；照抄）。memory 教训：guardrail 测试必须复用现有 fixture，别另起炉灶。

---

## 5. 验收标准

1. 改动前：`test_n1_chartmaker_allowed` red（n=1 派 chart-maker 被拒）。
2. 改动后：
   - 新测试全绿：`cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_path_sequence_n1.py -v`
   - **n≥2 不回归**：`test_n2_chartmaker_still_requires_data_analyst` 绿（n≥2 仍强制 data-analyst）。
3. **全量回归**（改了 guardrail 核心判定，memory 教训必须跑全量 + grep 所有调用方）：
   - `cd packages/agent/backend && make test`
   - 重点确认现有 `test_path_sequence*.py` 全绿（没破坏 n≥2 序列保护、没破坏 plan precondition）。
   - 已知 4 个污染失败与本改动无关，不阻塞。
4. **不动 lead prompt**：prompt 的 n=1 fast-path 规则本来就对（教跳过 data-analyst），错的是 guardrail 不感知。改完 guardrail 后两者自洽。
5. **不动 `path_registry.py` 的序列定义**、**不动死字段 `condition`**。

---

## 6. 影响面与风险

- **放行边界**：只在 `_is_single_subject_run` 返回 True（handoff_code_executor.json 明确显示 n<2）时放宽 data-analyst。fail-open 设计为**保守方向**（拿不准 → 维持 data-analyst 必需），不会误放行 n≥2。
- **时序安全**：guardrail 拦 chart-maker/report-writer 时，handoff_code_executor.json 必已存在（是其 plan precondition），所以 n 信号一定可读。
- **report-writer 同理**：n=1 时 report-writer 的前序若含 data-analyst，同样被本特判跳过（推荐方案的循环对所有 target 生效，data-analyst step 在 n<2 一律不计 missing）。这符合预期（n=1 报告也不需要 data-analyst 判读）。
- **与 Spec 1 的关系**：本 spec 修好后，n=1 不再强制 data-analyst → 从源头消除 Spec 1 的 3.2 触发路径。但 **3.2 仍须独立修**（Spec 1）：n≥2 时 data-analyst 若想报 partial（某组质量差）一样会撞 schema。两 spec 正交、不互相阻塞、可并行。

---

## 7. 提交

- worktree 名建议：`worktree-guardrail-n1-path-awareness`
- commit message（中文）：`fix(guardrail): path_sequence 感知 n=1，单 subject 时 data-analyst 非必需（与 lead n=1 fast-path 对齐）`
- 全量测试绿（除已知 4 污染）后建 PR 合入 dev。

---

## 8. 关联

- dogfood 报告：`docs/handoffs/2026-06/2026-06-08-epm-dogfood-findings.md` §3.1
- Spec 1（3.2+3.3）：`docs/superpowers/specs/2026-06-08-handoff-status-partial-and-file-perms-spec.md`
- 死字段 `condition` 实证：`path_sequence_provider.py:189-197` 不读 condition；viz==yes 实际由 `intent_post_step_ask_gate_provider.py` 强制
- n 数据源实证：thread 1bda1847 的 groups.json 空、handoff_code_executor.json per_subject=['Subject 1']
- memory：guardrail 测试复用现有 fixture（`feedback_pr_merge_must_run_full_suite_on_shared_logic.md`、`feedback_ssot_skill_deployment_distinction.md` 同类纪律）
