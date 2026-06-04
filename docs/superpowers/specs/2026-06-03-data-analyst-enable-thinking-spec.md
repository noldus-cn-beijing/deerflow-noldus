# Spec：为 data-analyst（及 report-writer）启用 thinking —— 提升数据洞察质量

> **状态**：待实施 | **作者**：review/诊断 agent | **日期**：2026-06-03
> **实施者**：新 agent | **预计规模**：小（1 个 config 字段 + executor.py 1 行 + 2 个 subagent config + 测试 + dogfood 验证）
> **风险**：中（碰受保护文件 `executor.py`）

---

## 0. 这份 spec 解决什么、不解决什么

**解决**：**仅给 data-analyst 这一个 subagent 打开 thinking 通道**，提升其分析洞察的深度与结构化程度，并让其内部推理回到独立的 `reasoning_content` 字段（不污染对话历史 / handoff）。

> ⚠️ **本轮范围严格限定为 data-analyst 一个 subagent。** 用户明确要求：只开 data-analyst，**其他所有 subagent（code-executor / chart-maker / report-writer / general-purpose / bash 及任何上游同步进来的）一律保持不开 think**。report-writer 是否开 think 留作后续独立评估（见 §7），本轮**不动**。开关是**逐 subagent 属性**（每个 subagent 自己 config 的 `thinking_enabled`），不是全局开关——改 `executor.py` 只是让它去读各自 config 的值，真正决定开不开的是各 subagent config，本轮只有 data-analyst 一个显式设 True。

**明确不解决（不要混淆）**：
- ❌ **不是修 seal 卡死**。seal 卡死已由 2026-06-03 第 1 批纯 prompt 修复解决（commit `4caa78b8`，dogfood 第一次派遣即成功）。本 spec 与那个 bug **正交**——它追求的是"洞察质量"，不是"能不能 seal"。
- ❌ **不是因为"预算太少"**。11 次 deepseek 实验已证 `reasoning_content` 在关 think 时也始终存在、开 think 不显著减少 AIMessage 数。本 spec 的收益是"洞察质量"，**不要在 PR 里宣称它修了卡死或扩了预算**。
- 背景与实验证据见 [`docs/problems/2026-06-03-subagent-seal-deadlock-problem-statement.md`](../../problems/2026-06-03-subagent-seal-deadlock-problem-statement.md) 与 [`2026-06-03-reply-to-four-layer-proposal.md`](../../problems/2026-06-03-reply-to-four-layer-proposal.md)，memory `feedback_subagent_seal_deadlock_is_prompt_not_budget`。**实施前必读。**

---

## 1. 为什么开 think 对 data-analyst 有价值（动机）

- data-analyst 的职责本质是**深度推理洞察**（system prompt 第 217 行"主动提出洞察：不只复述统计数字，要告诉研究者这意味着什么"；要做方法学把关、反事实 leave-one-out、混杂因素排查）。这是 reasoning 模型最该用 thinking 通道的场景。
- 现状 `executor.py:642` 对**所有** subagent 硬编码 `thinking_enabled=False`，把 data-analyst 的推理逼成正文 `content`。开 think 后：
  1. 推理回到独立 `reasoning_content` 字段（`PatchedReasoningChatOpenAI` 已实现路由，见 `models/patched_reasoning.py`），**不混进 handoff JSON、不复述给下游**；
  2. deepseek 开 think 时推理更充分（实验实测 reasoning 长度 111+ vs 关 think 的 29 字符），洞察更结构化；
  3. code-executor / chart-maker 这类"按 plan 跑脚本、不需要洞察"的保持关 think（省 token、它们本就不靠推理）。

---

## 2. 现状机制（已核实，实施者据此改，不要重新假设）

| 事实 | 位置 | 含义 |
|---|---|---|
| subagent 一律 `thinking_enabled=False` | `executor.py:642` `create_chat_model(name=model_name, thinking_enabled=False)` | 要改成按 config 取值 |
| `SubagentConfig` 无 thinking 字段 | `subagents/config.py:11` dataclass | 要新增 `thinking_enabled: bool = False` |
| `create_chat_model(thinking_enabled=...)` 已支持 | `models/factory.py:50` | 传 True + 模型有 `when_thinking_enabled` 段即开启 |
| deepseek-v4-pro 已配 `when_thinking_enabled.extra_body.enable_thinking: true` | `config.yaml` models[0] | 基础设施就绪，只差打开开关 |
| reasoning 走独立字段 `additional_kwargs.reasoning_content` | `patched_reasoning.py:63-90` | 开 think 后推理不进 `content`、不进 handoff |
| `executor.py` 是受保护文件 | `scripts/sync-deerflow.sh:62` PROTECTED_FILES | 改它要 surgical、上游同步守护，本 spec 改动仅 1 行，成本低 |
| turn 计数不看 tool_calls | `executor.py:888-911` | 与本 spec 无关（不要动这里），仅说明开 think 不应被期望"减少 turn 数"——实验已证不显著 |

⚠️ **token 预算注意**：`config.yaml` 里 deepseek-v4-pro `max_tokens: 4096`。开 think 后 reasoning + 正文**共享**输出 token，4096 可能让复杂分析的"思考 + 正式输出"被截断。实施时观察 dogfood 是否出现输出截断；若有，评估给 data-analyst 单独的高 `max_tokens` 模型变体（见 §3 可选项）。

---

## 3. 实施步骤

### 步骤 1：`SubagentConfig` 新增 `thinking_enabled` 字段
`subagents/config.py`：在 dataclass 加
```python
thinking_enabled: bool = False  # reasoning 模型是否开扩展思考通道；洞察型 subagent(data-analyst/report-writer) 设 True
```
默认 `False` → **向后兼容**：所有未显式设置的 subagent（含上游同步进来的）行为不变。docstring 同步加一行说明。

### 步骤 2：`executor.py:642` 用 config 值替代硬编码
```python
# 改前
model = create_chat_model(name=model_name, thinking_enabled=False)
# 改后
model = create_chat_model(name=model_name, thinking_enabled=self.config.thinking_enabled)
```
**这是受保护文件的唯一 1 行改动**。改完在该行上方加一句注释说明"按 subagent config 决定是否开 think（洞察型开、执行型关）"，便于未来 sync 时 surgical 识别。

### 步骤 3：洞察型 subagent config 开 think（**本轮仅 data-analyst 一个**）
- `subagents/builtins/data_analyst.py` 的 `DATA_ANALYST_CONFIG`：加 `thinking_enabled=True`
- **其他所有 subagent 保持默认 False，本轮一律不加**：code-executor、chart-maker、**report-writer（本轮不开，留 §7 后续评估）**、general-purpose、bash。
- 即：本步骤**只改 data_analyst.py 一处** config，加一个 `thinking_enabled=True`。

### 步骤 4（可选，仅当步骤 5 dogfood 出现输出截断时才做）
若开 think 后 data-analyst 正式输出被 token 上限截断：评估给它配一个 `max_tokens` 更高的模型变体（如 config.yaml 加 `deepseek-v4-pro-thinking` 段，`max_tokens: 8192`，data_analyst config `model` 指向它）。**先不做，dogfood 观察到截断再说**，避免过度工程。

---

## 4. 测试要求（TDD 强制）

### 4.1 单元测试（`tests/`）
1. **config 字段默认值**：`SubagentConfig().thinking_enabled is False`（向后兼容锚点）。
2. **只有 data-analyst 开、其余全关（最关键的"不一开全开"锚点）**：断言 `DATA_ANALYST_CONFIG.thinking_enabled is True`，且 `CODE_EXECUTOR_CONFIG`、`CHART_MAKER_CONFIG`、`REPORT_WRITER_CONFIG`（以及 general-purpose / bash 等所有其他 builtin config）的 `thinking_enabled is False`。**遍历所有 builtin subagent config，断言除 data-analyst 外全部为 False**——这条测试就是用来防止未来有人"顺手"把别的也开了。
3. **executor 传递**：mock `create_chat_model`，断言 `_create_agent` 用 `thinking_enabled=self.config.thinking_enabled` 调用它（即 data-analyst executor 传 True、code-executor executor 传 False）。注意现有测试可能 `patch.object` 了 `_create_agent`，确认不冲突。

### 4.2 全量回归（共享逻辑改动铁律 [[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）
改了 `executor.py`（所有 subagent 共用的执行引擎），**合并前必跑后端全量**：
```
cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=<...>/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider -o addopts=""
```
预期：3612 passed + **2 pre-existing failed**（`test_inspect_gate_guardrail.py::...::test_async_delegates_to_sync` 和 `test_paradigm_identification_gate.py::...::test_async_delegates_to_sync`——event-loop 测试隔离问题，单独跑 2 passed，**非回归**，不要误判为本改动引入）。若 failed 数 > 2 或换了别的测试，必须查清。

### 4.3 关键验证实验（开 think 是否真不破坏 seal + 是否提升洞察）
**必跑**（环境能调真 deepseek）：
1. 重启 dev（`cd packages/agent && make stop && make dev`——改 executor/config 必须重启，[[feedback_subagent_seal_deadlock_is_prompt_not_budget]]）。
2. 跑 FST dogfood（同 n=1 Porsolt 数据），确认：
   - data-analyst **仍第一次派遣即 seal 成功**（开 think 不能把已修好的卡死带回来）；
   - `handoff_data_analyst.json` 的 `reasoning_content` 不混进任何字段（推理隔离正确）；
   - 对比开 think 前后的 `key_findings` / `method_warnings` 质量（开 think 后是否更具体、更有洞察——这是本 spec 的成败标准）。
3. **建议再跑一个非 FST 范式（EPM 或 TST）的 dogfood**，确认开 think 的收益泛化、不仅限 FST。

---

## 5. 验收标准（review 时逐条核对）

- [ ] `SubagentConfig.thinking_enabled` 字段加好，默认 `False`，docstring 说明。
- [ ] `executor.py:642` 改为 `self.config.thinking_enabled`，仅此 1 行 + 1 行注释，未碰 turn 计数/seal-resume/其他逻辑。
- [ ] **仅 data-analyst** config 设 `thinking_enabled=True`；**其他所有 subagent（code-executor / chart-maker / report-writer / general-purpose / bash）保持 False**——有遍历型测试断言这一点。
- [ ] 单元测试 4.1 三项齐全且绿。
- [ ] 后端全量：3612 passed + 2 pre-existing failed（不多不少、就是那 2 个）。
- [ ] dogfood：data-analyst 仍第一次 seal 成功（**未把卡死带回**）；reasoning 未污染 handoff；洞察质量有可观察提升。
- [ ] PR 描述如实：本改动是"洞察质量优化"，**不宣称**修了卡死/扩了预算。
- [ ] `executor.py` 改动在 PR 里标注"受保护文件 surgical 改动"，便于未来 sync 守护。

---

## 6. 禁区 / 注意

- ❌ **不动 turn 计数逻辑**（`executor.py:888-911`）——本 spec 与它无关，实验已证开 think 不靠减 turn 数获益。
- ❌ **不给 seal-resume 加 tool_choice**（`executor.py:712` 探针证明产空 args）。
- ❌ **不动 step 2.8 / step 3 的 prompt**（那是已生效的卡死修复 commit `4caa78b8`，本 spec 不碰）。
- ❌ **不改 code-executor / chart-maker 的 think 设置**（它们该保持关）。
- ⚠️ 开 think 增加延迟 + token 成本（reasoning token 通常 2-4× output）。data-analyst `timeout_seconds=600` 当前应够；若 dogfood 超时，提到 900。
- ⚠️ **config.yaml 里有明文 API key**（与本 spec 无关，但实施 agent 若动 config.yaml 切勿把 key 带进 git diff 暴露；建议单独 issue 处理 key 外移到环境变量）。
- ✅ 全程正面措辞（CLAUDE.md §6）。

---

## 7. 回滚

若 dogfood 发现开 think 引入问题（截断/超时/洞察反而变差/卡死复现）：把 data_analyst/report_writer config 的 `thinking_enabled` 改回 `False`（或删字段）即可即时回滚，`executor.py` 的 `self.config.thinking_enabled` 读到 False 等价于原行为。**先验证再扩到 report-writer**——可先只给 data-analyst 开、dogfood 确认收益后再开 report-writer。
