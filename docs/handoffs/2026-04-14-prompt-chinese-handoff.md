# Prompt 中文化 + E2E 测试 — 交接文档

> 日期: 2026-04-14
> 上一份交接: `docs/handoffs/2026-04-14-upstream-sync-handoff.md`

---

## 1. 当前任务目标

**将 prompt.py 中的英文 system prompt 改为中文**，解决 GLM-5.1 被英文 prompt 影响导致前端输出全英文的问题。

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | DeerFlow 上游同步（27 commits） | ✅ commit `5fb824d3` |
| 2 | 修复 20 个失败测试（1539 passed） | ✅ |
| 3 | 创建 `scripts/sync-deerflow.sh` + SOP 文档 | ✅ |
| 4 | prompt.py 英文 → 中文改写 | ❌ 未开始 |
| 5 | E2E 端到端测试 | ❌ 部分完成，流水线能跑但有 API 间歇性错误 |

---

## 3. 待完成任务：Prompt 中文化

### 3.1 需要改的文件

`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（917 行）

### 3.2 需要中文化的段落

以下是 prompt.py 中仍为英文的段落（Noldus 调度规则部分已经是中文）：

| 行号范围 | 内容 | 优先级 |
|---------|------|--------|
| ~190-280 | `_build_subagent_section()` — subagent 调度指令（DECOMPOSE/DELEGATE/SYNTHESIZE、并发限制、批次执行） | **高** — 直接影响 GLM 输出语言 |
| ~318-340 | `SYSTEM_PROMPT_TEMPLATE` 开头 — role、thinking_style | **高** |
| ~341-400 | `<clarification_system>` — 澄清工作流、场景描述（部分已改中文，部分仍是英文） | **中** |
| ~400-430 | `<working_directory>` — 工作目录说明 | **中** |
| ~430-500 | 输出格式、citations、critical_reminders | **中** |

### 3.3 不需要改的段落

- Noldus 调度规则（`noldus_rules`）— 已经是中文
- Noldus subagent 描述（`noldus_descriptions`）— 已经是中文
- orchestration_guide — 已经是中文
- 中文澄清示例 — 已经是中文

### 3.4 改写原则

1. **保持语义完全一致** — 不要趁机改逻辑，只改语言
2. **保留 XML tag 名称为英文** — `<subagent_system>`、`<clarification_system>` 等不改
3. **保留模板变量** — `{n}`、`{available_subagents}`、`{skills_section}` 等不改
4. **保留代码示例中的代码** — `task()`、`ask_clarification()` 等函数调用不改
5. **用正面指令** — 记住 memory: `feedback_positive_prompting.md`，GLM 对"禁止X"会反向激活

---

## 4. 关键上下文

### 项目结构
- 主项目: `/home/qiuyangwang/noldus-insight`
- prompt 文件: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- 上游 remote: `deerflow` → `https://github.com/noldus-cn-beijing/deerflow-noldus.git`
- prompt.py 是受保护文件（在 `scripts/sync-deerflow.sh` 的 PROTECTED_FILES 列表中）

### GLM-5.1 行为特征
- system prompt 语言会强烈影响输出语言
- 对"禁止X"指令会反向激活（正面指令更有效）
- JSON 输出偶尔包含非法控制字符

### API 状态
- `newapi.noldusapi.com/v1` — 可用但有 ~10% 间歇性 400 错误（upstream_error）
- 主流程正常，memory updater 和 subagent 偶尔受影响
- 不影响 prompt 中文化工作

---

## 5. E2E 测试现状

上次 E2E 测试（commit `5fb824d3`）观察到：
- ✅ `subagent_enabled: True` — 已生效
- ✅ subagent tools (task) 已加载
- ✅ lead agent 成功派遣了 code-executor subagent
- ✅ subagent 连续执行多轮成功
- ⚠️ API 间歇性 400（3/30 次，~10% 失败率）— newapi 代理层问题
- ❌ 前端输出语言混杂英文 — 需要 prompt 中文化解决

---

## 6. 未完成事项（按优先级）

### 高优先级

1. **prompt.py 中文化** — 本文档的核心任务
2. **中文化后 E2E 测试** — 验证 GLM 输出是否变为中文
3. **回归测试** — `cd packages/agent/backend && make test`

### 中优先级

4. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复
5. **微调工作启动** — 按 `docs/plans/2026-04-13-fine-tuning-small-model-design.md`

### 低优先级

6. **prompt.py 抽离重构** — 将 Noldus 逻辑抽到独立文件（长期改善同步体验）
7. **修复 API 间歇性失败** — 联系 newapi 侧排查

---

## 7. 建议接手路径

```bash
# 1. 读取本文档
# 2. 读取 prompt.py
cat packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py

# 3. 按 3.2 节的优先级逐段中文化
# 重点改 _build_subagent_section() 和 SYSTEM_PROMPT_TEMPLATE 开头

# 4. 改完后跑测试
cd packages/agent/backend && make test

# 5. 启动服务验证
cd /home/qiuyangwang/noldus-insight && make dev
# 在前端发消息，确认输出是中文

# 6. 提交
git add -A && git commit -m "prompt.py: 英文 system prompt 中文化"
```

---

## 8. 风险与注意事项

1. **prompt.py 是受保护文件** — 改动越大，下次上游同步冲突越大。但这次中文化是必要的
2. **不要改 XML tag 名** — `<subagent_system>` 等 tag 名可能被代码解析
3. **正面指令** — GLM 对"不要做X"会反向激活，改成"请做Y"
4. **skills/custom/ 是 gitignored** — skill 文件不在 git 中
5. **noldus-kb 临时禁用** — `extensions_config.json` 中 `"enabled": false`
6. **`subagent_enabled` 默认值** — 已在 commit `5fb824d3` 中改为 True，确认已生效

---

## 下一位 Agent 的第一步建议

1. 读取本文档
2. 读取 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
3. 从 `_build_subagent_section()` 函数（~line 190）开始中文化
4. 然后改 `SYSTEM_PROMPT_TEMPLATE`（~line 318）
5. 改完跑 `make test`，确认无破坏
6. `make dev` 验证前端输出语言
