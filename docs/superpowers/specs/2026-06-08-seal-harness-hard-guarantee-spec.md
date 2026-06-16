# Spec C — seal 黑洞 harness 层硬保障：有产出就有 handoff，不靠 LLM 记得调工具

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree（独立于 Spec A / Spec B）
> 来源：seal 死锁第三次复现（thread `7d4d9b8e` report-writer），底层机制问题
> 性质：**harness 核心改动**（subagent 生命周期）。⚠️ 高风险区，memory 有多条红线（见 §1.3），充分测试 + 谨慎。
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 一句话目标

seal 死锁已**第三次**以不同触发路径出现：
1. step 2.8 prompt 矛盾（`feedback_subagent_seal_deadlock_is_prompt_not_budget.md`）
2. schema 拒 partial（`feedback_dataanalyst_reportwriter_handoff_status_missing_partial.md`）
3. 违宪输入逼迫改写、烧光 turn（本次，thread 7d4d9b8e）

三次都修了**表面触发点**，但**底层机制**——"subagent 完成了实质工作（写了 report.md / 算了指标），却把 reasoning 当完成、没调 seal 工具就退出"——**从没被根治**。现有 1 轮 `seal-resume` 兜底**这次也没救回**（gateway.log: `seal-resume did not recover`）。

**根治方向**：当 subagent **已产出关键文件**但**没调 seal** 时，harness **确定性地用已有产出构造 handoff**（不依赖 LLM），而不是判 FAILED 重派。把"靠 LLM 自觉调 seal"换成"有产出就有 handoff 的机制保证"。

---

## 1. 背景、证据、红线

### 1.1 本次失败的确切数据（已核实）

- report-writer 第 1 次（trace=9876bd7a）：`captured #1-5`（远小于 max_turns）+ 无 `reached max_turns` + `seal-resume did not recover` → **(a) 叙述黑洞**（自然结束、没调 seal、1 轮 seal-resume 也没救回）。
- **关键**：第 1 次失败时 `outputs/report.md` **已经写好了**（6151 字节）。report-writer 完成了实质工作，**只差 seal 这个"封条"**。
- 第 2 次重派（trace=f1f14295，#1-8）才 seal 成功。**用户白等了一整轮**（14:21 失败 → 14:23 第二轮成功，中间 lead 重派开销）。

### 1.2 现有兜底链（已读代码，executor.py）

- `_attempt_seal_resume`（line 735-829）：subagent 没产 handoff 时，**补 1 轮**"请调 seal"。前置守卫：history 至少 1 条 AIMessage 才补（line 773）。**只补 1 轮**（Sprint 5.8 锁定，line 161 注释"深思过的决策"）。
- seal-resume 失败 → `_validate_handoff_emitted`（line 170-262）仍报错 → 直接 `try_set_terminal(FAILED)`（line 1064）→ lead 重派整个 subagent。
- **缺口**：seal-resume 失败、`_validate_handoff_emitted` 报错时，**没有检查"产出文件是否已存在"**。report.md 明明在，却判 FAILED 重派——浪费一整轮。

### 1.3 ⚠️ 红线（memory 实证，违反会引入新 bug）

执行 agent **必须遵守**，这些是踩过坑的：

1. **不要给 seal-resume / 兜底加 `tool_choice` 强制**（`feedback_subagent_seal_deadlock_is_prompt_not_budget.md` 实证：强制 tool_choice 会产空 args 的 seal，比漏调更糟）。
2. **不要调大 max_turns**（同上 memory：turn 预算不是根因，调大治标且掩盖问题）。
3. **不要"优化" `_attempt_seal_resume` 的 resume_prompt 措辞**（executor.py:757 明确标注"实施 agent 不要优化这段"，2026-05-29 dogfood 实证能跑通）。
4. **不要用 structured output 强制 seal**（executor.py:746：会 strip runtime 注入字段）。
5. **`_validate_handoff_emitted` / 兜底函数绝不能抛异常**（executor.py:183 ROBUSTNESS：调用点在 try/except 里，抛了会被误归因成通用 FAILED，clobber 诊断）。所有文件访问包 try/except，异常 fail-open。
6. **正面措辞**（CLAUDE.md §6 deepseek）。

---

## 2. 设计：确定性 auto-seal 兜底（核心改动）

### 2.1 原理

在 seal-resume 失败后、判 FAILED 之前，**插入一层"确定性 auto-seal"**：
- 检查该 subagent 的**关键产出文件**是否已存在（report-writer → `outputs/report.md`；其他见 §2.3 适用范围）。
- 若存在 → harness **不靠 LLM**，直接用"产出文件 + 上游 handoff + 确定性默认值"构造一个**能过 `_HANDOFF_CONTENT_CHECKS`** 的 handoff，落盘，标 `errors=["harness auto-seal: subagent 完成产出但未调用 seal 工具"]`，判 **COMPLETED**（带 auto-seal 标记）。
- 若产出文件**不存在** → 维持现状判 FAILED（subagent 真没干活，该重派）。

### 2.2 为什么这是治本而非治标

- 它不依赖"LLM 这次记得调 seal"（seal-resume 已证不可靠）。
- 它基于**客观事实**：产出文件在不在（确定性可查）。
- 它区分两种失败：**"干了活没封存"**（auto-seal 救）vs **"压根没干活"**（仍 FAILED 重派）。这正是 seal-resume 前置守卫（line 773 "no analysis to seal"）想区分但没做彻底的。

### 2.3 ⚠️ 适用范围（关键分寸，执行 agent 务必理解）

**不是所有 subagent 都能 auto-seal**。判据：**handoff 的核心内容能否从产出文件/上游 handoff 确定性推导**。

| subagent | handoff 核心字段 | 能否确定性构造 | auto-seal? |
|----------|-----------------|--------------|-----------|
| **report-writer** | `report_path` + `sections_written` | ✅ report_path=outputs/report.md（存在即知）；sections_written 可从 report.md 的 markdown 标题解析 | **✅ 适用** |
| **chart-maker** | `chart_files` | ✅ 可列 outputs/ 下的 plot_*.png | **✅ 适用**（次要，本次未失败，但同理） |
| **code-executor** | `metrics_summary`（嵌套 group→metric→{mean,std,n}） | ❌ 不能凭空造数值——必须从 m_*.json 聚合，逻辑复杂且易错 | **❌ 不适用**（保持 FAILED/seal-resume） |
| **data-analyst** | `key_findings`（专业判读结论） | ❌ 判读是 LLM 的认知产物，无产出文件可推 | **❌ 不适用**（保持 FAILED/seal-resume） |

**本 spec 只对 report-writer（+ chart-maker）做 auto-seal**。code-executor / data-analyst 的核心 handoff 是"无法从文件确定性重建"的认知产物，强行 auto-seal 会造垃圾 handoff（违反 Sprint 5.5 "空内容比漏调更糟"）。它们继续走现有 seal-resume → FAILED 链。

> 这个分寸是本 spec 的灵魂：**auto-seal 只填"机械可推导"的 handoff，不伪造"认知产物"**。

### 2.4 改动位置

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`

在 seal-resume 失败、`_handoff_error is not None` 之后、`try_set_terminal(FAILED)` 之前（约 line 1062-1064），插入 auto-seal 尝试：

```python
            if _handoff_error is not None:
                # Spec C: seal-resume 仍失败 → 在判 FAILED 前，尝试确定性 auto-seal。
                # 仅对"handoff 核心字段可从产出文件机械推导"的 subagent（report-writer/
                # chart-maker）生效；code-executor/data-analyst 的认知产物无法重建，跳过。
                _auto_sealed = _attempt_auto_seal_from_artifacts(
                    self.config.name, _workspace,
                )
                if _auto_sealed:
                    logger.warning(
                        "[trace=%s] Subagent %s: seal-resume failed but artifacts "
                        "exist; harness auto-sealed handoff deterministically.",
                        self.trace_id, self.config.name,
                    )
                    result.try_set_terminal(SubagentStatus.COMPLETED)
                else:
                    logger.warning(
                        "[trace=%s] Subagent %s terminated without emitting handoff "
                        "(seal-resume did not recover, no artifacts to auto-seal): %s",
                        self.trace_id, self.config.name, _handoff_error,
                    )
                    result.try_set_terminal(SubagentStatus.FAILED, error=_handoff_error)
            else:
                result.try_set_terminal(SubagentStatus.COMPLETED)
```

新增模块函数 `_attempt_auto_seal_from_artifacts`（放在 `_validate_handoff_emitted` 附近）：

```python
# Spec C: subagent 完成产出但漏调 seal 时，harness 用已有产出确定性构造 handoff。
# 仅限 handoff 核心字段可从文件机械推导的 subagent。绝不伪造认知产物（判读/指标数值）。
_AUTO_SEALABLE: dict[str, str] = {
    "report-writer": "handoff_report_writer.json",
    "chart-maker": "handoff_chart_maker.json",
}

def _attempt_auto_seal_from_artifacts(subagent_name: str, workspace_path: str | None) -> bool:
    """seal-resume 失败后的确定性兜底。返回 True=已 auto-seal，False=无法兜底（维持 FAILED）。

    ROBUSTNESS: 绝不抛异常（调用点在 executor try/except 内，见 _validate_handoff_emitted 同款约束）。
    任何异常 → 返回 False → 外层走 FAILED。
    """
    if subagent_name not in _AUTO_SEALABLE or not workspace_path:
        return False
    try:
        from pathlib import Path
        import json, hashlib, os

        ws = Path(workspace_path)
        handoff_path = ws / _AUTO_SEALABLE[subagent_name]
        if handoff_path.exists() and handoff_path.stat().st_size > 0:
            return False  # 已有 handoff（不该到这，保险）

        # outputs 目录：thread 的 user-data/outputs（workspace 是 user-data/workspace，同级）
        outputs = ws.parent / "outputs"

        if subagent_name == "report-writer":
            report = outputs / "report.md"
            if not report.exists() or report.stat().st_size == 0:
                return False  # 没产出 → 不兜底
            # 确定性构造：report_path + 从 md 标题解析 sections_written
            text = report.read_text(encoding="utf-8")
            sections = [ln.lstrip("# ").strip() for ln in text.splitlines()
                        if ln.strip().startswith("#")]
            payload = {
                "status": "completed",
                "report_path": "/mnt/user-data/outputs/report.md",  # 虚拟路径，与正常 seal 一致
                "sections_written": sections or ["（harness auto-seal：未能解析标题）"],
                "errors": ["harness auto-seal: report-writer 完成报告产出但未调用 seal_report_writer_handoff 工具；handoff 由 harness 依据 outputs/report.md 确定性构造"],
                "gate_signals": None,
            }
        elif subagent_name == "chart-maker":
            charts = sorted(p.name for p in outputs.glob("plot_*.png")) if outputs.exists() else []
            if not charts:
                return False
            payload = {
                "status": "completed",
                "paradigm": "",  # 可选；chart-maker schema paradigm 必填则从上游 handoff 读，见下
                "summary": "harness auto-seal：图表已生成但 subagent 未调用 seal 工具",
                "chart_files": charts,
                "failed_charts": [],
                "gate_signals": None,
            }
            # chart-maker schema 若 paradigm 必填：从 handoff_code_executor.json 读
            ce = ws / "handoff_code_executor.json"
            if ce.exists():
                try:
                    payload["paradigm"] = json.loads(ce.read_text(encoding="utf-8")).get("paradigm", "")
                except Exception:
                    pass
        else:
            return False

        # 注入 analysis_config_id（与 _seal_handoff 一致，从 workspace 读）+ atomic write
        # 复用 seal_handoff_tools 的写盘逻辑最稳妥——见 §2.5。
        # 这里直接构造，必须能过 _HANDOFF_CONTENT_CHECKS[subagent_name]。
        ...  # atomic write + chmod 0o644（参照 Spec1 教训）+ manifest（见 §2.5）
        return True
    except Exception:
        logger.exception("[auto_seal] %s: failed; falling back to FAILED", subagent_name)
        return False
```

### 2.5 实现要点（执行 agent 必读）

1. **复用 `seal_handoff_tools._seal_handoff` 的写盘 + 校验逻辑**，不要自己手写 json.dump。最干净的做法：让 `_attempt_auto_seal_from_artifacts` 调用对应的 `seal_*_handoff` 工具函数的**底层 helper**（`_seal_handoff(ModelCls, filename, payload, runtime)`），这样自动获得：Pydantic 校验（保证过 schema）、atomic write、manifest hash、analysis_config_id 注入、task_context 注入。
   - 但 `_seal_handoff` 需要 `runtime`（取 workspace）。executor 里有 `self.thread_data` / `self.sandbox_state`，需构造一个能让 `_resolve_workspace` 工作的 runtime-like 对象，或重构 `_seal_handoff` 接受显式 workspace 参数。**执行 agent 评估两条路**：(a) 重构 `_seal_handoff` 抽出一个 `_seal_handoff_to_workspace(model_cls, filename, payload, workspace: Path)` 纯函数（推荐，可测）；(b) 构造 runtime stub。倾向 (a)。
2. **构造的 payload 必须过 `_HANDOFF_CONTENT_CHECKS[subagent_name]`**（executor.py:219）。先读 `_HANDOFF_CONTENT_CHECKS` 里 report-writer / chart-maker 的判据，确保填的字段满足（report-writer 大概率查 report_path 非空 + sections_written 非空）。
3. **chmod 0o644**（Spec1 教训：文件权限）——若复用 `_seal_handoff` 且它已统一处理就不用；否则补。
4. **status 取值**：report.md 存在 → "completed"；若上游 handoff_code_executor 的 statistical_validity=skipped（n<2），仍可 "completed"（报告本身完成了）。不要纠结 partial，报告产出完整就是 completed。
5. **errors 字段务必写明是 auto-seal**：让下游 / 调试能看出这个 handoff 是 harness 兜底造的，不是 subagent 正常产的。

---

## 3. 测试（TDD，先 red）

放 `packages/agent/backend/tests/`，新建 `test_auto_seal_from_artifacts.py`。

```python
import json, os, stat
from pathlib import Path
from deerflow.subagents.executor import _attempt_auto_seal_from_artifacts

def _mk_thread(tmp_path):
    ws = tmp_path / "user-data" / "workspace"
    out = tmp_path / "user-data" / "outputs"
    ws.mkdir(parents=True); out.mkdir(parents=True)
    return ws, out

class TestAutoSealReportWriter:
    def test_report_exists_no_handoff_auto_seals(self, tmp_path):
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# 实验概况\n...\n## 结果\n...", encoding="utf-8")
        # 上游 analysis_config_id 来源（若 _seal_handoff 需要）
        (ws / "experiment-context.json").write_text(json.dumps({"analysis_config_id": "abc123"}), encoding="utf-8")
        ok = _attempt_auto_seal_from_artifacts("report-writer", str(ws))
        assert ok is True
        h = ws / "handoff_report_writer.json"
        assert h.exists()
        data = json.loads(h.read_text())
        assert data["status"] == "completed"
        assert data["report_path"].endswith("report.md")
        assert len(data["sections_written"]) >= 1          # 从标题解析
        assert any("auto-seal" in e for e in data["errors"])  # 标记
        # 文件权限 644（Spec1 教训）
        assert stat.S_IMODE(os.stat(h).st_mode) == 0o644

    def test_no_report_does_not_auto_seal(self, tmp_path):
        ws, out = _mk_thread(tmp_path)  # outputs 空
        ok = _attempt_auto_seal_from_artifacts("report-writer", str(ws))
        assert ok is False                                  # 没产出 → 不兜底 → 外层 FAILED

    def test_existing_handoff_not_overwritten(self, tmp_path):
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X", encoding="utf-8")
        (ws / "handoff_report_writer.json").write_text('{"status":"completed","report_path":"/x"}', encoding="utf-8")
        ok = _attempt_auto_seal_from_artifacts("report-writer", str(ws))
        assert ok is False                                  # 已有 handoff，不重造

class TestAutoSealScope:
    def test_data_analyst_never_auto_sealed(self, tmp_path):
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X", encoding="utf-8")  # 即便有文件
        ok = _attempt_auto_seal_from_artifacts("data-analyst", str(ws))
        assert ok is False                                  # 判读是认知产物，绝不 auto-seal

    def test_code_executor_never_auto_sealed(self, tmp_path):
        ws, _ = _mk_thread(tmp_path)
        ok = _attempt_auto_seal_from_artifacts("code-executor", str(ws))
        assert ok is False

    def test_no_workspace_fails_safe(self, tmp_path):
        assert _attempt_auto_seal_from_artifacts("report-writer", None) is False

class TestAutoSealRobustness:
    def test_never_raises_on_garbage(self, tmp_path):
        ws, out = _mk_thread(tmp_path)
        (out / "report.md").write_text("# X", encoding="utf-8")
        # 即便 _seal_handoff 内部出错也不抛（返回 False）
        # 构造一个会让构造失败的场景，断言不抛异常
        try:
            _attempt_auto_seal_from_artifacts("report-writer", "/nonexistent/\x00bad")
        except Exception as e:
            assert False, f"must not raise, got {e}"
```

> 执行 agent：先读 `_HANDOFF_CONTENT_CHECKS` + `_seal_handoff` 真实签名，按实际调整构造逻辑和测试。auto-seal 产的 handoff **必须能过 `_validate_handoff_emitted`**——加一条集成测试：auto-seal 后调 `_validate_handoff_emitted("report-writer", str(ws))` 应返回 None。

---

## 4. 验收标准

1. red：`test_report_exists_no_handoff_auto_seals` 在实现前 red（函数不存在）。
2. 实现后新测试全绿，**重点**：
   - report-writer/chart-maker 有产出无 handoff → auto-seal 成功，产物过 `_HANDOFF_CONTENT_CHECKS`。
   - data-analyst/code-executor → **永不** auto-seal（认知产物保护）。
   - 无产出 → 不兜底（维持 FAILED）。
   - 不抛异常（ROBUSTNESS）。
3. **全量回归**（改 harness 核心，memory 教训必须全量 + 不破坏现有 seal/executor 测试）：
   - `cd packages/agent/backend && make test`（已知污染 + config symlink）。
   - **重点确认**现有 `test_seal_resume.py` / `test_executor_handoff_emission.py` / `test_handoff_emission_validator.py` 全绿（没破坏现有 seal-resume → FAILED 链）。
4. **dogfood 验证**：重跑 EPM，若 report-writer 第 1 次又漏 seal（在 Spec A/B 修好前可能仍偶发）→ 期望 harness **auto-seal 兜底**，用户**不再看到"报告生成遇到问题，正在重新生成"的重派**，直接拿到报告 + handoff（带 auto-seal errors 标记）。
5. 守全部 §1.3 红线（不加 tool_choice / 不动 max_turns / 不改 resume_prompt 措辞 / 不用 structured output / 兜底不抛异常）。

---

## 5. 与 Spec A/B 的关系

- A（路由）+ B（术语）修好后，report-writer 不再收违宪输入 → seal 黑洞**触发概率大降**。
- 但 C 是**独立的底层兜底**：即使将来出现**新的**触发路径（第四次），只要 subagent 产出了文件，harness 都能兜底，不再让用户白等一轮重派。**C 是 seal 黑洞的最后一道防线，不依赖 A/B。**
- 三者正交、可并行、各自独立 PR。

---

## 6. 提交

- worktree 名建议：`worktree-seal-harness-auto-seal-fallback`
- commit message（中文）：`fix(harness): seal-resume 失败后用已有产出确定性 auto-seal（仅 report-writer/chart-maker），不再白等重派`
- 全量绿（除已知污染）+ 现有 seal 测试不破坏后建 PR 合入 dev。

---

## 7. 关联

- seal 死锁三次复现：`feedback_subagent_seal_deadlock_is_prompt_not_budget.md`（#1）、`feedback_dataanalyst_reportwriter_handoff_status_missing_partial.md`（#2）、`project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md`（#3）
- 现有兜底：executor.py `_attempt_seal_resume`（735）、`_validate_handoff_emitted`（170）、Sprint 5.5 内容非空检查、Sprint 5.8 seal-resume 1 轮决策
- seal 工具：`seal_handoff_tools.py` `_seal_handoff`（354）
- 红线 memory：`feedback_subagent_seal_deadlock_is_prompt_not_budget.md`（别加 tool_choice/别动 max_turns/别优化措辞）
- 文件权限：Spec1（chmod 0o644）
- Spec A / Spec B：同批
