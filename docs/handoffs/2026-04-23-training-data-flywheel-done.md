# 2026-04-23 训练数据飞轮 — 交接文档

> **给下一个 AI Agent：** 你无法访问本次会话上下文。这份文档让你快速理解我们在 2026-04-23 做了什么，以及接下来怎么继续推进。
>
> **读取顺序**：本文档 → [CLAUDE.md](../../CLAUDE.md) → [docs/sop/training-data-flywheel-sop.md](../sop/training-data-flywheel-sop.md)

---

## 1. 会话概览

**主题**：实现"训练数据飞轮"——让行为学专家正常使用 EthoInsight 时，自动沉淀 Fireworks SFT/DPO 格式训练样本，同时通过前端三按钮收集专家质量反馈。

**最终状态**：全部 17 个 Task 完成，手动 E2E 验证通过，1694 后端测试全绿，`pnpm typecheck` 干净。

---

## 2. 本次实现的内容

### Phase A — 后端录制中间件（已完成）

| 文件 | 说明 |
|------|------|
| `packages/harness/deerflow/agents/middlewares/training_data_middleware.py` | 新建 `TrainingDataMiddleware`，在 `after_agent` 钩子抽取 lead/subagent 样本落到 `.deer-flow/training-data/auto-collected/<thread_id>.jsonl` |
| `packages/harness/deerflow/agents/lead_agent/agent.py` | 在 `_build_middlewares()` 中 MemoryMiddleware 之后注册 TrainingDataMiddleware |
| `tests/test_training_data_middleware.py` | 8 个单测，含质量过滤和鲁棒性兜底测试 |
| `tests/test_lead_agent_training_middleware.py` | 验证注册顺序正确 |

**关键设计决策**：
- `before_agent`/`after_agent` 都有 try/except 兜底，任何 IO 或提取异常只 warning log，不上抛，保证 agent 鲁棒性
- 质量过滤：空输出、含 `"error":` / `"timed_out"` / `HTTP 429` 的 ToolMessage 自动过滤

### Phase B — Gateway 反馈 API（已完成）

| 文件 | 说明 |
|------|------|
| `app/gateway/routers/feedback.py` | `POST /api/threads/{id}/feedback`（verdict 枚举校验）+ `GET /api/threads/{id}/feedback`（返回历史反馈列表） |
| `app/gateway/app.py` | 注册 feedback router |
| `tests/test_feedback_router.py` | 7 个单测，含磁盘失败兜底测试 |

### Phase C — 前端三按钮 UI（已完成）

| 文件 | 说明 |
|------|------|
| `src/components/feedback/feedback-buttons.tsx` | `<FeedbackButtons>` 组件，✅ 直接提交，⚠️/❌ 展开 textarea |
| `src/core/api/api-client.ts` | 新增 `submitFeedback` / `listFeedback` 方法 |
| `src/components/workspace/messages/message-list-item.tsx` | assistant 消息末尾挂 FeedbackButtons（非 streaming、非 human） |
| `src/components/workspace/messages/subtask-card.tsx` | subtask 完成后挂 FeedbackButtons，messageId=`subtask-<taskId>` |
| `src/components/workspace/messages/message-list.tsx` | 向 MessageListItem 和 SubtaskCard 传递 `threadId` prop |

**关键 bug 已修复**：Task 11/12 初版用 `useParams()` 读 `thread_id`，在 `/workspace/chats/new` 路由下会返回字面量 `"new"` 导致反馈文件命名错误。已改为从 `MessageList` 向下传 prop（`threadId` 来自 `useThreadChat()` 生成的真实 UUID）。

### Phase D — 后处理 + 仪表板（已完成）

| 文件 | 说明 |
|------|------|
| `scripts/extract_e2e_sessions.py` | 读 auto-collected + feedback，join 成 SFT/DPO JSONL，输出 stats.json |
| `scripts/training_dashboard.py` | 格式化进度日报 |
| `Makefile` 新增 `training-stats` target | `make training-stats` 打印进度条 |

### Phase E — 验证（已完成）

- **手动 E2E**：本地启动后跑了一次真实 Shoaling 分析，录制了 3 条样本（1 lead + 2 subagent），专家打了 2 条反馈（1 correct + 1 needs_fix with revision），`make training-stats` 输出 `sft_count=3`
- **全量后端回归**：1694 passed, 14 skipped, 0 failed

---

## 3. 新增文件清单

```
packages/agent/backend/
├── packages/harness/deerflow/agents/middlewares/training_data_middleware.py  [新建]
├── app/gateway/routers/feedback.py                                            [新建]
├── scripts/__init__.py                                                        [新建]
├── scripts/extract_e2e_sessions.py                                            [新建]
├── scripts/training_dashboard.py                                              [新建]
└── tests/
    ├── test_training_data_middleware.py                                       [新建]
    ├── test_lead_agent_training_middleware.py                                 [新建]
    ├── test_feedback_router.py                                                [新建]
    ├── test_extract_e2e_sessions.py                                           [新建]
    └── test_training_dashboard.py                                             [新建]

packages/agent/frontend/src/
├── components/feedback/feedback-buttons.tsx                                   [新建]
└── core/api/api-client.ts                                                     [修改：新增 submitFeedback/listFeedback]

docs/
└── sop/training-data-flywheel-sop.md                                         [新建]
```

**修改的受保护文件（DeerFlow fork）**：
- `packages/harness/deerflow/agents/lead_agent/agent.py` — 1 行 import + 5 行注册，影响极小
- `app/gateway/app.py` — 2 行（import + include_router）

---

## 4. 已知问题 / 后续工作

### 4.1 反馈-样本关联精度（v0.1 限制）

`extract_e2e_sessions.py` 当前用"第一条反馈应用于整个 thread 所有样本"的简单 join。这是 v0.1 临时方案，注释中已标注。改进方向：前端 `message_id` 已正确传入（格式：LangGraph run ID 或 `subtask-<taskId>`），后端 join 逻辑需要按 message_id 精确匹配。

### 4.2 云部署时的数据收集

训练数据落在服务器本地 `.deer-flow/training-data/`。云部署时有以下选项：

**选项 A（推荐 v0.1）**：挂载持久化卷 + 定期 rsync
```bash
# 在服务器上
rsync -avz user@server:/path/to/.deer-flow/training-data/ ./local-backup/
```

**选项 B**：`extract_e2e_sessions.py` 输出后上传到 S3/OSS
```bash
python scripts/extract_e2e_sessions.py && aws s3 sync .deer-flow/training-data/processed/ s3://your-bucket/
```

**选项 C（长期）**：把 feedback router 改为写 S3/数据库，middleware 也直接写远端存储——但这需要引入额外依赖，超出 v0.1 范围。

### 4.3 下一阶段

- 两个月内目标：800 条 SFT + 300 对 DPO（当前：3 条 SFT，飞轮刚启动）
- 每周一运行 `make training-stats` 看进度，每周一运行 `extract_e2e_sessions.py` 合成当周数据集
- 达标后进入 Phase 1 微调（Qwen3-8B + Fireworks.ai）

---

## 5. 测试基线

| 项目 | 数量 |
|------|------|
| 后端测试（本次新增） | +18（从 1676 → 1694） |
| 前端 typecheck | 通过 |
| 手动 E2E | ✅ 录制 + 反馈 + 仪表板全链路验证 |
