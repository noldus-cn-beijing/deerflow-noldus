# Prompt 中文化 + 空响应修复 — 交接文档

> 日期: 2026-04-14
> 上一份交接: `docs/handoffs/2026-04-14-prompt-chinese-handoff.md`

---

## 1. 当前任务目标

两个任务：
1. **prompt.py 英文 system prompt 中文化** — 解决 GLM-5.1 被英文 prompt 影响导致前端输出全英文
2. **修复 GLM-5.1 空响应问题** — 前端上传文件后 agent 返回空白，卡住不动

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | prompt.py 英文 → 中文改写（全部 6 个区域） | ✅ |
| 2 | file_pattern 校验强化（orchestration_guide Step 1） | ✅ |
| 3 | 测试修复（test_lead_agent_skills.py 断言更新） | ✅ |
| 4 | 回归测试（1528 passed） | ✅ |
| 5 | 空响应根因排查 | ✅ 根因已定位 |
| 6 | UploadsMiddleware 修复（list content 扁平化） | ✅ 代码已改 |
| 7 | 重启服务 + E2E 验证 | ❌ 未完成 |
| 8 | 提交 commit | ❌ 未完成 |

---

## 3. 已完成：Prompt 中文化

### 3.1 改动的文件

**`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`**:
- `_build_subagent_section()` — 全部英文框架指令改为中文（DECOMPOSE→分解、DELEGATE→委派 等）
- `SYSTEM_PROMPT_TEMPLATE` — `<role>`、`<thinking_style>` 中文化
- `<clarification_system>` — 工作流说明、5 个场景、使用示例；禁止式指令改为正面 ✅ 格式
- `<working_directory>`、`<response_style>` 中文化
- `<citations>` — 引用规则和示例中文化
- `<critical_reminders>` 中文化
- `_build_skill_evolution_section()` — "Skill Self-Evolution" → "技能自进化"
- `apply_prompt_template()` — 非 Noldus 场景的 subagent_reminder/thinking 中文化

**`packages/agent/backend/tests/test_lead_agent_skills.py`**:
- 3 处断言 `"Skill Self-Evolution"` → `"技能自进化"`

### 3.2 file_pattern 强化

**位置**: orchestration_guide Step 1（prompt.py ~line 825）

**旧写法**（声明式，GLM 容易忽略）:
```
CRITICAL: 文件路径必须使用正确的 glob 模式！
- 正确: /mnt/user-data/uploads/轨迹*.txt
- 错误: /mnt/user-data/uploads/.txt
```

**新写法**（正面步骤式指令）:
```
文件路径构造方法（请按以下步骤执行）：
1. 从 <uploaded_files> 中读取完整文件名
2. 提取文件名公共前缀
3. 构造 glob 模式
4-5. 具体示例
请确认你构造的文件路径包含文件名前缀和 * 通配符。
```

---

## 4. 已完成：空响应根因 + 修复

### 4.1 根因

**`UploadsMiddleware`**（上游同步 commit `5fb824d3` 引入的变更）将前端发送的 multimodal list content 保留为 list 格式：

```python
# 旧版（正常工作）
content = f"{files_message}\n\n{original_content}"  # 总是字符串

# 新版（导致空响应）
updated_content = [files_block, *original_content]  # list 格式
```

**GLM-5.1 不支持 OpenAI 的 multimodal content list 格式**（`[{type: "text", text: "..."}]`）。收到 list content 后返回 200 OK 但内容为空。

### 4.2 修复

**文件**: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py` ~line 264

修复逻辑：
- 如果 list 中有非 text 类型的 block（如 image）→ 保留 list 格式（为将来支持 multimodal 的模型保留）
- 如果 list 中全是 text block → **扁平化为纯字符串**（兼容 GLM-5.1 等不支持 list content 的模型）

测试状态：`test_uploads_middleware_core_logic.py` 37 passed ✅

### 4.3 排查过程关键证据

```
# LangGraph SDK 获取的 thread 消息结构
Message 0 (human): content type: list, blocks=2  ← 这就是问题
Message 1 (ai): content type: str, len=0          ← GLM 返回空

# curl 直接调用 API（字符串 content）→ 正常返回
# DeerFlow 模拟调用（字符串 content）→ 正常返回
# 前端 thread（list content）→ 空响应
```

---

## 5. 关键上下文

### 项目结构
- 主项目: `/home/qiuyangwang/noldus-insight`
- prompt 文件: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- uploads middleware: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py`
- 计划文档: `docs/plans/2026-04-14-prompt-chinese-plan.md`

### GLM-5.1 行为特征
- system prompt 语言强烈影响输出语言
- 对"禁止X"指令会反向激活（正面指令更有效）
- **不支持 OpenAI multimodal content list 格式** — 收到 list 后返回空内容
- JSON 输出偶尔包含非法控制字符
- newapi.noldusapi.com 有 ~10% 间歇性 400 错误

### API 状态
- `newapi.noldusapi.com/v1` — 基本可用，间歇性 400
- `usage.input_tokens` 和 `output_tokens` 字段始终为 0（newapi 代理层问题，不影响功能）
- memory updater 偶尔因 API 400 失败（非阻塞）

---

## 6. 未完成事项（按优先级）

### 高优先级

1. **重启服务 + E2E 验证** — `make stop && make dev`，新建 thread，上传 Shoaling 数据测试
   - 确认 GLM 不再返回空响应
   - 确认 file_pattern 包含正确的文件名前缀（`轨迹*.txt`）
   - 确认输出是中文
2. **提交 commit** — 三个文件一起提交：
   - `prompt.py` — 中文化 + file_pattern 强化
   - `test_lead_agent_skills.py` — 断言更新
   - `uploads_middleware.py` — list content 扁平化修复

### 中优先级

3. **空响应重试机制** — 在 `llm_error_handling_middleware` 中添加对 "200 OK 但 content 为空" 的检测和重试（当前只处理 HTTP 错误异常）
4. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复
5. **微调工作启动** — 按 `docs/plans/2026-04-13-fine-tuning-small-model-design.md`

### 低优先级

6. **prompt.py 抽离重构** — 将 Noldus 逻辑抽到独立文件（长期改善同步体验）
7. **修复 API 间歇性失败** — 联系 newapi 侧排查

---

## 7. 建议接手路径

```bash
# 1. 读取本文档
# 2. 重启服务
cd /home/qiuyangwang/noldus-insight && make stop && make dev

# 3. 在前端新建 thread，上传 Shoaling 数据，发送分析请求
# 验证：GLM 不再返回空响应，file_pattern 正确，输出中文

# 4. 如果验证通过，提交
cd /home/qiuyangwang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py
git add packages/agent/backend/tests/test_lead_agent_skills.py
git commit -m "prompt.py 中文化 + 修复 list content 导致 GLM 空响应"

# 5. 如果仍有空响应，检查前端发送的消息 content 格式
# 在 langgraph.log 中搜索 "token usage: input=0 output=0"
# 用 LangGraph SDK 检查 thread 消息结构：
# python3 -c "from langgraph_sdk import get_client; ..."
```

---

## 8. 风险与注意事项

1. **prompt.py 是受保护文件** — 改动大，下次上游同步冲突大（但中文化是必要的）
2. **中文化不影响其他模型** — Qwen/Claude/GPT 的逻辑理解不受中文 prompt 影响
3. **list content 扁平化** — 只对纯 text list 扁平化，包含 image 等非 text block 时保留 list（为将来 multimodal 模型保留）
4. **newapi usage 字段** — `input_tokens`/`output_tokens` 始终为 0，是代理层问题，token_usage_middleware 日志会显示 `input=0 output=0`，不影响实际功能
5. **skills/custom/ 是 gitignored** — skill 文件不在 git 中
6. **noldus-kb 临时禁用** — `extensions_config.json` 中 `"enabled": false`

---

## 下一位 Agent 的第一步建议

1. 读取本文档
2. `cd /home/qiuyangwang/noldus-insight && make stop && make dev` 重启服务
3. 在前端新建 thread → 上传 Shoaling 数据 → 发送分析请求
4. 查看 `packages/agent/logs/langgraph.log` 确认：
   - 无 `token usage: input=0 output=0`（空响应已修复）
   - 有 `Subagent code-executor` 日志（subagent 被正确派遣）
   - file_pattern 包含 `轨迹*`（不是 `.txt`）
5. 验证通过后提交 commit
