# Handoff — chart-maker 不依赖 data-analyst 修复 + dogfood 验证

> 日期：2026-06-09 ｜ 写给下一位接手的 AI Agent

---

## 0. 一句话现状

在 dev 工作树上做了 3 文件修改（未 commit）：修复 chart-maker 被 path_sequence guardrail 误拦（要求 data-analyst 先完成）+ lead prompt 修复（用户要解读时不自动派 chart-maker）。全量测试 3832/3 baseline red 通过。dogfood 验证 TST 全流程跑通。

---

## 1. 仓库/分支坐标

- 主仓库 `dev` HEAD：**`00e7a2be`**
- **3 个文件有未提交改动**（在 dev 工作树上，没有新建分支）：
  - `packages/agent/backend/packages/harness/deerflow/guardrails/path_registry.py`
  - `packages/agent/backend/packages/harness/deerflow/guardrails/path_sequence_provider.py`
  - `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- 之前清理了 7 个 worktree（本地+远程），sync-21 分支也已删远程

---

## 2. 问题背景（从上一会话 dogfood 发现）

上一会话（2026-06-09 sync-21 handoff）末尾用户 dogfood 验证时发现两个问题：

### Bug 1: chart-maker 被 path_sequence guardrail 误拦

TST n=1 场景，用户要"解读"数据，lead 同时派了 data-analyst + chart-maker。chart-maker 被 `path_sequence_violation` 拒：
> "按 E2E_FULL_ASKVIZ 路径，chart-maker 之前必须先完成 data-analyst"

**根因**：path_sequence_provider 用"所有前序 dispatch step"做前置条件检查。`E2E_FULL_ASKVIZ` 路径中 chart-maker 排在 data-analyst 后面，chart-maker 的 missing-predecessor 检查要求 handoff_data_analyst.json 存在。

但 chart-maker 实际只需要 code-executor 的数据（handoff_code_executor.json）就能出图，不需要 data-analyst 的判读。

### Bug 2: 用户要解读时 lead 同时派了 chart-maker

用户回答 viz 反问时说"需要解读"，lead 理解为"解读+出图"，同时派了两个 subagent。

**根因**：lead prompt 没有明确"解读 ≠ 要图"，lead 把用户的解读请求理解为也同意出图。

---

## 3. 已完成的修复（✅）

### 3.1 path_registry.py — Step 新增 `prerequisites` 字段

```python
@dataclass(frozen=True)
class Step:
    kind: StepKind
    target: str
    condition: str | None = None
    prerequisites: tuple[str, ...] = ()  # NEW: 显式前置 subagent
```

PATHS 更新：
```python
"E2E_FULL_ASKVIZ": [
    Step("dispatch", "code-executor"),
    Step("dispatch", "data-analyst", prerequisites=("code-executor",)),
    Step("ask", "viz"),
    Step("dispatch", "chart-maker", condition="viz==yes", prerequisites=("code-executor",)),  # 不含 data-analyst!
    Step("ask", "report"),
],
"E2E_FULL": [
    Step("dispatch", "code-executor"),
    Step("dispatch", "data-analyst", prerequisites=("code-executor",)),
    Step("dispatch", "chart-maker", prerequisites=("code-executor",)),  # 不含 data-analyst!
    Step("ask", "report"),
],
```

### 3.2 path_sequence_provider.py — 从"所有前序"改为显式 prerequisites

missing-predecessor 检查从遍历所有前序 dispatch step 改为只检查 `Step.prerequisites`：

```python
# OLD: for i in range(target_idx): step = steps[i]; check all dispatch handoffs
# NEW:
step_prereqs = steps[target_idx].prerequisites
if step_prereqs:
    for prereq_name in step_prereqs:
        handoff_name = to_handoff_name(prereq_name)
        handoff_path = Path(workspace) / f"handoff_{handoff_name}.json"
        if not handoff_path.exists():
            missing.append(prereq_name)
```

- `_is_single_subject_run` 函数**保留**（用户明确要求——n=1 判定逻辑仍有价值，供 lead prompt fast-path 判定和其他 guardrail 使用）
- n=1 特殊 case hack 移除（不再需要——显式 prerequisite 从根本上解决了问题）

### 3.3 lead_agent/prompt.py — 不把"解读"当"出图"

两处改动：

**(a)** n=1 fast-path 段（~line 995）新增：
```
**用户要求解读时，只派 data-analyst，不派 chart-maker（用户没有要图）。
不要将"解读/洞察"理解为"同意出图"或"也画出图来"。**
```

**(b)** viz 反问段（~line 348）新增：
```
**重要**: 如果用户没有明确选择 A/B 选项，而是提出新需求（如"帮我解读一下""这些数据代表了什么"），
不要自动 set_viz_choice=yes。只响应用户的新需求（派 data-analyst），
可视化的选择留到下一次反问。不要把"解读数据"理解为"同意出图"。
```

---

## 4. 测试结果

- 全量 **3832 passed / 3 baseline red / 21 skipped**（42s）
- 3 baseline red = 已知 test isolation 污染（`test_chart_maker_config_basic_fields` + `test_async_delegates_to_sync`×2）
- 生产入口裸导入 0 退出；`test_gateway_import_no_cycle` 绿
- path_sequence 相关测试全部更新：59 passed（含新增测试）
- Dogfood 验证：TST n=1 全流程跑通，chart-maker 未被 guardrail 拦截

---

## 5. 未完成事项

### P0 — 用户操作
- **commit + push 本次 3 文件改动**（dev 分支上，未 commit）。建议 commit message（中文）：
  ```
  fix(guardrail): chart-maker 不依赖 data-analyst，path_sequence 改用显式 prerequisites

  - path_registry Step 新增 prerequisites 字段，chart-maker 只依赖 code-executor
  - path_sequence_provider 改从 Step.prerequisites 检查，不再遍历所有前序 dispatch
  - lead prompt 明确"解读≠出图"，用户要解读时不自动 set_viz_choice=yes
  - _is_single_subject_run 保留供 prompt 和其他 guardrail 使用
  ```
- **make dev 重启**后再次 dogfood 确认（Prompt 改动需要重启生效）

### P2 — 可选的后续改进
- `intent_post_step_ask_gate_provider.py` 的 viz gate 当前依赖"前一个 dispatch（data-analyst）完成"才触发；如果 chart-maker 可以在 data-analyst 之前跑，viz gate 的触发逻辑可能需要重新审视（目前 viz 反问仍由 prompt 驱动没问题，但 guardrail 层面的 viz gate 在 n=1 skip data-analyst 时不会触发——这是已有 gap，本次未改）
- report-writer 的 prerequisites 目前未显式定义（只有 plan precondition 检查 handoff_code_executor.json），如果未来要显式化 report-writer 的前置条件，可同样加 `prerequisites=("code-executor",)`

---

## 6. 下一位 Agent 的第一步建议

1. **先看 git status**：`cd /home/wangqiuyang/noldus-insight && git diff --stat` 确认 3 文件改动
2. **跑全量测试**（如果改了更多东西）：
   ```bash
   cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/ -q -p no:cacheprovider
   ```
   应 3832 passed / 3 baseline red
3. **跑裸导入验证**：
   ```bash
   PYTHONPATH=. uv run python -c "import app.gateway; from deerflow.agents import make_lead_agent; print('OK')"
   ```
4. **commit + push** 上述 3 文件改动
5. **如果用户说 dogfood**：`make dev` 重启后，用 `/home/wangqiuyang/DemoData/newdemodata/悬尾/` 中的 xlsx 文件跑 TST n=1 全流程，重点验证：
   - 用户说"解读"时只派 data-analyst
   - 之后说"画图"时 chart-maker 不被 guardrail 拦

---

## 7. 相关 memory（接手必读）

- `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import` — 改 harness 核心后必须跑裸导入
- `feedback_known_full_suite_test_pollution_4_tests` — 3 baseline red 别归因自己
- `feedback_sync_nonprotected_files_with_noldus_customization_overwritten` — 本次 session 开头读的 sync-21 handoff 中的教训
