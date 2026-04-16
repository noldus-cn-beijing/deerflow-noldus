# EthoInsight 会话交接文档

> 日期: 2026-04-16
> 上一份交接: `docs/handoffs/2026-04-15-model-selection-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了两件事**：

1. **诊断 Agent 无限循环问题** — 行为学同事用 EPM（高架十字迷宫）数据测试 agent 时，agent 陷入循环（context 被压缩 6+ 次，每次丢失进度重来）
2. **修复 Agent 鲁棒性** — 从 prompt 契约、工具返回值、middleware 协同三个层面修复，让 agent 在遇到不支持的范式时优雅降级

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 诊断循环根因（5 个缺口叠加） | ✅ |
| 2 | Fix 1: 强化 lead agent prompt 契约 — 失败后强制 ask_clarification | ✅ 已提交 |
| 3 | Fix 2: run_paradigm_analysis 未注册范式返回结构化失败 | ✅ 已提交 |
| 4 | Fix 3: loop detection 与 summarization middleware 协同 | ✅ 已提交 |
| 5 | 单元测试（3 个新测试） | ✅ 已提交 |
| 6 | 全量 backend 测试 | ✅ 1537 passed, 2 pre-existing failures |
| 7 | EPM 范式模板创建 (`epm.py`) | ❌ 未开始（本次聚焦鲁棒性，非功能补全） |
| 8 | EPM 指标补全 (`metrics.py`) | ❌ 未开始 |
| 9 | read_file UTF-16 fallback | ❌ 未开始（非循环根因） |
| 10 | 人工 E2E 验证（启动 agent 测试 EPM 降级） | ❌ 未做 |

**提交**: `6bf51adc` ("修复了循环问题")，branch: `feature/etho-skills`

---

## 3. 本次改动的文件

| 文件 | 改动 |
|------|------|
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 重写契约表格"失败降级"→"失败处理"；新增"失败处理规则"段落（必须 ask_clarification）+ "失败后的特殊情况"段落；绝对禁止静默重试 |
| `packages/ethoinsight/ethoinsight/templates/tool.py` | `run_paradigm_analysis_core` 入口增加范式校验 gate：不在 `get_available_paradigms()` 中的范式立即返回 `status="failed"` + `available_paradigms` 列表；修正 docstring 移除虚假的 "epm, open_field" 支持声明 |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py` | `ArchivingSummarizationMiddleware` 新增 `loop_detection` 可选参数；summarization 后调用 `reset(thread_id)` 清除 loop detection 状态 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | `LoopDetectionMiddleware` 实例提前创建，引用传给 summarization middleware；消除两个 middleware 的状态不一致 |
| `packages/ethoinsight/tests/test_template_tool.py` | 新增 `TestRunParadigmAnalysisCoreGate` 类（3 个测试：unknown/epm/error message 校验） |

---

## 4. 关键架构决策

### 循环根因分析（5 个缺口叠加）

| # | 缺口 | 位置 | 修复方式 |
|---|------|------|---------|
| 1 | skill fallback 假设 get_analysis_template 必然成功 | `fallback-workflow.md` | Fix 2: 入口 gate，不让 code-executor 走到 fallback |
| 2 | prompt 降级规则是建议不是强制 | `prompt.py:276` | Fix 1: 改为硬规则 + ask_clarification |
| 3 | context 压缩后丢失失败记录 | `archiving_summarization.py` | Fix 3: summarization 后 reset loop detection |
| 4 | loop detection 只看 hash，不看语义 | `loop_detection_middleware.py` | 未改（Fix 1 是更有效的防线） |
| 5 | task tool 返回字符串，无结构化失败码 | `task_tool.py:189-209` | 未改（Fix 2 让失败信息足够明确） |

### 核心设计原则确认

> **Lead agent 是 human-in-the-loop 的判断者，subagent 只是执行者。**
> subagent 失败后如实上报 → lead agent 用 ask_clarification 向用户澄清 → 用户给方向 → lead agent 再决策。
> 不在 skill 层加终止条件（那是在教 subagent 放弃，搞反了角色）。

### 工程化方向确认

> **框架是 DeerFlow 的，领域智能是你的。**
> 不需要做额外的框架层工程化（如 Celery 等），聚焦在：
> 1. 范式模板 + 指标补全
> 2. 领域知识丰富
> 3. Subagent prompt 调优
> 4. 微调数据采集

---

## 5. 关键发现

### 问题全景（来自同事的 EPM 测试记录）

测试记录在 `docs/problems/大鼠高架十字迷宫实验.md`，核心症状：
- agent 陷入循环，context 被压缩 6+ 次
- 出现大量重复的 "Here is a summary of the conversation to date" 摘要重放
- code-executor 多次派遣但无法产出有效文件
- `run_paradigm_analysis` 和 `get_analysis_template` 都不完全支持 EPM
- read_file 工具因 UTF-8 硬编码无法预览 EthoVision UTF-16 文件

### 当前范式支持状态

| 范式 | templates/ 模板 | metrics.py 指标 | assess.py 阈值 | 实际可用 |
|------|----------------|----------------|----------------|---------|
| shoaling | ✅ `shoaling.py` | ✅ 完整 | ✅ | ✅ |
| epm | ❌ 不存在 | 部分（1/7 指标） | 部分（1/3 阈值） | ❌ 被 gate 拦截 |
| open_field | ❌ 不存在 | 部分 | 部分 | ❌ 被 gate 拦截 |
| 其他 10 个范式 | ❌ 不存在 | ❌ | ❌ | ❌ 被 gate 拦截 |

`prompt.py:897` 列出了 11 个范式名称，但只有 shoaling 真正可用。Fix 2 的 gate 确保不可用范式返回明确错误而非"半成功"。

### Pre-existing 测试失败

`test_client.py` 中 2 个测试失败（与本次改动无关）：
- `TestClientInit::test_default_params` — `assert client._subagent_enabled is False` 但实际是 True
- `TestEnsureAgent::test_reuses_agent_same_config` — config key 不匹配导致 agent 被重建

这是之前将 `subagent_enabled` 默认值改为 True 但未更新对应测试导致的。

---

## 6. 未完成事项

### 高优先级

1. **人工 E2E 验证** — 启动 agent，上传 EPM 数据，验证现在能优雅降级而不是循环
2. **提交上次 43 个文件** — 上上次交接文档记录的未提交改动，需要确认是否已提交（检查 `git log`）
3. **429 重试策略优化** — `llm_error_handling_middleware.py` 中当前 1s/2s 太短，建议 5s/15s/30s
4. **修复 2 个 pre-existing 测试** — `test_client.py` 中 `subagent_enabled` 默认值断言需更新

### 中优先级（功能补全）

5. **创建 EPM 模板** — `packages/ethoinsight/ethoinsight/templates/epm.py`，参照 `shoaling.py` 结构
6. **补全 EPM 指标** — `metrics.py` 中添加 closed_arm_time_ratio、center_time、entries 等 6 个函数
7. **补全 EPM 评估阈值** — `assess.py` 中 `_DEFAULT_THRESHOLDS["epm"]` 添加 open_arm_entry_ratio 和 total_arm_entries
8. **read_file UTF-16 fallback** — `local_sandbox.py:337` 加 BOM 检测（减少 agent 浪费轮次）
9. **微调数据采集启动** — 按 `docs/plans/2026-04-15-fine-tuning-data-checklist.md`
10. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json` 为 `"enabled": true`

### 低优先级

11. **更多范式模板** — open_field, novel_object, y_maze 等
12. **范式模板验证** — open_field 和 epm 的 demo data
13. **更新微调设计文档** — `docs/plans/2026-04-13-fine-tuning-small-model-design.md`

---

## 7. 建议接手路径

### 如果要验证本次修复

```bash
cd /home/qiuyangwang/noldus-insight

# 1. 确认修复已提交
git log --oneline -3
# 应看到 6bf51adc "修复了循环问题"

# 2. 跑测试确认
cd packages/agent/backend && uv run pytest tests/test_loop_detection_middleware.py ../../ethoinsight/tests/test_template_tool.py -v

# 3. 启动 agent，手动测试 EPM 降级
cd /home/qiuyangwang/noldus-insight && make dev
# 上传 demo-data/DemoData/高架十字迷宫/ 下的文件
# 发送 "帮我分析这个高架十字迷宫数据"
# 预期：agent 用 ask_clarification 告知 EPM 暂不支持，而不是循环
```

### 如果要补全 EPM 范式

```bash
# 1. 查看 shoaling 模板作为参考
cat packages/ethoinsight/ethoinsight/templates/shoaling.py

# 2. 查看已有的 EPM metrics 代码
grep -n "epm\|open_arm" packages/ethoinsight/ethoinsight/metrics.py

# 3. 创建 epm.py（照搬 shoaling.py 结构，改 PARADIGM 和 METRICS_TO_COMPUTE）

# 4. 补全 metrics.py（添加 closed_arm_time_ratio 等 6 个函数）

# 5. 跑测试
cd packages/agent/backend && uv run pytest ../../ethoinsight/tests/ -v
```

### 如果要修复 pre-existing 测试失败

```bash
# 查看失败的断言
grep -n "subagent_enabled.*False" packages/agent/backend/tests/test_client.py
# 将 False 改为 True（因为默认值已改为 True）
```

---

## 8. 风险与注意事项

1. **代码可能仍有未提交改动** — 上上次交接记录了 43 个文件未提交，请用 `git status` 确认
2. **noldus-kb 临时禁用** — `extensions_config.json` 中 `"enabled": false`，不要提交
3. **skills/custom/ 是 gitignored** — skill 文件不在 git 中
4. **GLM-5.1 API 不稳定** — 429 限流和 500 错误频繁
5. **Fireworks.ai 不支持 MoE 微调** — 不要尝试在 Fireworks 上微调 Qwen3.5-9B
6. **Fix 1 依赖 LLM 遵守 prompt** — 失败处理规则写在 prompt 中，LLM 仍可能不完全遵守。如果测试中发现仍有循环，可能需要在 middleware 层面加硬性中断（如：检测到 task tool 返回 "failed"/"timed out" 后自动注入 ask_clarification）
7. **`run_paradigm_analysis_core` 的 gate 基于模板文件** — 即使 metrics.py 有某范式的指标代码，只要 templates/ 下没有对应 .py 文件，该范式就被 gate 拦截。添加新范式时需要同时创建模板文件

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. `git log --oneline -5` 确认最新提交
3. 人工 E2E 验证：启动 agent → 上传 EPM 数据 → 确认优雅降级
4. 根据优先级选择下一个任务：EPM 范式补全 or 429 重试优化 or 微调数据采集
