# 交接文档：GLM-5.1 主模型升级 + Summarization 调优

> 日期: 2026-04-08
> 前序交接: `docs/handoffs/2026-04-08-knowledge-assistant-and-mcp-guard-handoff.md`

---

## 1. 当前任务目标

1. 将主模型从 GLM-5 升级为 GLM-5.1（编码能力 +28%，thinking 默认开启）
2. 调优 Summarization 配置，防止长时间追问导致上下文溢出
3. 将 GLM-5 保留为摘要专用模型，降低主模型算力浪费

---

## 2. 当前进展

### 已完成 ✅

| # | 改动 | 详情 |
|---|------|------|
| 1 | 主模型升级为 GLM-5.1 | `config.yaml` models 第一项改为 `glm-5.1`，`supports_thinking: true` |
| 2 | GLM-5 保留为第二模型 | `config.yaml` models 第二项，`supports_thinking: false`（摘要不需要） |
| 3 | Summarization 触发条件调优 | 从 `tokens: 8000` 改为 `messages: 15 + fraction: 0.6`（OR 逻辑） |
| 4 | Summarization 保留消息数调优 | 从 `keep: 6` 改为 `keep: 8`（追问场景需更多上下文） |
| 5 | Summarization 指定摘要模型 | `model_name: glm-5`（之前为 null，即用主模型） |
| 6 | 确认无硬编码引用 | `grep glm-5 *.py` 无结果，DeerFlow 通过 config 列表第一项选主模型 |

### 已验证 ✅

- GLM-5.1 API 兼容性：端点 `open.bigmodel.cn/api/coding/paas/v4` 相同，模型名 `glm-5.1`
- GLM-5.1 上下文窗口 204,800 tokens，最大输出 131,072 tokens
- Coding Plan 全等级（Lite/Pro/Max）均支持 GLM-5.1
- GLM-5.1/GLM-5 消耗倍率：高峰 3x，低峰 2x（4月内暂为 1x）

---

## 3. 关键架构知识

### 模型配置

| 角色 | 模型 | 用途 | thinking |
|------|------|------|----------|
| 主模型 | glm-5.1 | lead agent + 所有 subagent | ✅ |
| 摘要模型 | glm-5 | SummarizationMiddleware | ❌ |

### DeerFlow Summarization 机制

- **位置**：`SummarizationMiddleware`，middleware 链第 6 位
- **触发**：消息数 ≥ 15 OR 上下文占比 ≥ 60%（OR 逻辑，任一满足即触发）
- **行为**：保留最近 8 条消息，将更早的压缩为一条 HumanMessage 摘要
- **保护**：不拆分 AI/Tool 消息对
- **配置位置**：`packages/agent/config.yaml` → `summarization` 段

### DeerFlow 三层上下文管理

| 层 | 机制 | 状态 |
|----|------|------|
| Layer 1 | 工具输出截断（MCP 4096 chars + sandbox 20K/50K chars） | ✅ 已启用 |
| Layer 2 | SummarizationMiddleware（15 轮或 60% 上下文触发） | ✅ 已启用并调优 |
| Layer 3 | 长期记忆 MemoryMiddleware（跨 thread 持久化） | ✅ 已启用 |

### Coding Plan 配额注意

- GLM-5.1 / GLM-5 / GLM-5-Turbo 消耗 **3x 倍率**（高峰），2x（低峰），4 月内暂为 1x
- GLM-4.7 及以下为 1x 倍率
- 如果配额吃紧，可将摘要模型降级为 `glm-4.7`（改 `summarization.model_name`）

---

## 4. 关键发现

1. **GLM-5.1 是 post-training 优化**：同架构（744B MoE / 40B 活跃），编码 benchmark 35.4 → 45.3
2. **API 完全兼容**：GLM-5 和 GLM-5.1 参数一致，无需改 LangChain adapter
3. **DeerFlow 不硬编码模型名**：`config.yaml` models 列表第一项即为默认主模型
4. **Summarization 之前配置过于激进**：8000 tokens 触发 + 只保留 6 条消息，追问场景容易丢失关键上下文

---

## 5. 未完成事项

### 高优先级

- [ ] **重启服务验证 GLM-5.1 加载**：`make stop && rm -f packages/agent/backend/.deer-flow/checkpoints.db* && rm -rf packages/agent/backend/.deer-flow/threads/* && make dev`
- [ ] **端到端功能验证**：上传数据 → 分析 → 追问，确认 GLM-5.1 正常工作
- [ ] **Summarization 验证**：连续追问 15+ 轮，观察日志确认摘要触发
- [ ] **knowledge-assistant 路由验证**：追问场景 + 纯知识问题场景

### 中优先级（继承自前序交接）

- [ ] code-executor 性能优化（3 个方向）
- [ ] 数据质量关卡验证
- [ ] 新图表类型端到端验证（raincloud_plot）
- [ ] noldus-kb MCP search_knowledge 500 错误排查

### 低优先级

- [ ] 更新 README.md / CLAUDE.md
- [ ] 评估是否需要 title 生成也指定 glm-5 模型（当前 `model_name: null` 用主模型）
- [ ] 考虑 memory 模块也指定轻量模型（当前 `model_name: null`）

---

## 6. 建议接手路径

### 启动

```bash
cd /home/qiuyangwang/noldus-insight
make stop
rm -f packages/agent/backend/.deer-flow/checkpoints.db*
rm -rf packages/agent/backend/.deer-flow/threads/*
make dev
```

### 验证顺序

1. `tail -f packages/agent/logs/langgraph.log` — 确认日志中出现 `glm-5.1` 而非 `glm-5`
2. Ultra 模式 → 上传数据 → "请分析" → 确认端到端正常
3. 同 thread → 连续追问 15 轮以上 → 观察日志确认 SummarizationMiddleware 触发
4. 新 thread → "什么是高架十字迷宫？" → 确认 knowledge-assistant 路由

### 关键文件

| 文件 | 用途 |
|------|------|
| `packages/agent/config.yaml` | 模型 + summarization + memory 配置 |
| `packages/agent/backend/packages/harness/deerflow/config/summarization_config.py` | Summarization 配置解析 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | Middleware 链注册 |
| `packages/agent/backend/packages/harness/deerflow/agents/memory/` | 长期记忆模块 |
| `packages/agent/logs/langgraph.log` | 运行时日志 |

---

## 7. 风险与注意事项

1. **不要把 glm-5 模型项删掉** — summarization 依赖 `model_name: glm-5` 引用它
2. **Coding Plan 配额** — GLM-5.1 和 GLM-5 都是 3x 倍率（高峰），注意监控使用量
3. **如果 GLM-5.1 不可用** — 回退方案：将 models 第一项的 `model` 字段改回 `glm-5`
4. **title 和 memory 的 model_name 仍为 null** — 意味着用 glm-5.1 主模型，可能浪费算力，后续可优化为轻量模型
5. **继承前序风险**：不要改 `executor.py:136` 的 `object.__setattr__`、has_noldus_agents 有两处需同步修改
