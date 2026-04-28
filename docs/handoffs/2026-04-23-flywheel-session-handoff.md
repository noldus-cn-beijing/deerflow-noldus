# 交接文档 — 训练数据飞轮完成 + 下一步

> 生成时间：2026-04-23
> 当前分支：`dev`
> 工作目录：`/home/qiuyangwang/noldus-insight`

---

## 1. 本次会话完成的工作

**训练数据飞轮（`docs/plans/2026-04-23-training-data-flywheel.md`）全部 17 个 Task 已完成并验证。**

飞轮已在本地手动 E2E 跑通：真实 Shoaling 分析录制了 3 条样本 + 专家打了 2 条反馈，`make training-stats` 输出正确。

### 完成的 commits（本次会话）
```
4a4c27f4 fix(lint): ruff auto-fix 格式修正
f494f70c docs: 训练数据飞轮 handoff 文档
4e5e0dd6 fix: 修复 feedback thread_id=new 的 bug + lint 修复 + 测试隔离
326e3684 docs: 训练数据飞轮 SOP + CLAUDE.md 更新
7e7db19f feat(training): 新增 make training-stats 日报命令
b7151697 feat(training): 新增 extract_e2e_sessions 后处理脚本
85e57ed7 feat(frontend): SubtaskCard 下挂 FeedbackButtons
2bccf09e feat(frontend): assistant 消息下挂 FeedbackButtons
fbe118a9 feat(frontend): 新增 FeedbackButtons 组件
2942d0ed feat(frontend): API client 增加 feedback 方法
c08819b0 feat(feedback): 新增 GET 反馈列表接口
3bc38c19 feat(feedback): 在 Gateway 注册 feedback router
b696b75b feat(feedback): Gateway 新增 feedback 路由（含磁盘失败兜底）
9448bbd4 feat(training): 在 lead agent 注册 TrainingDataMiddleware
b5e4f3f4 feat(training): 过滤低质量样本 + 录制异常兜底（保持 agent 鲁棒性）
902f31cf feat(training): 录制 subagent tool calling 样本
ae7e667c feat(training): 录制 lead agent 输入输出样本
cf14189b feat(training): 新增 TrainingDataMiddleware 骨架
```

---

## 2. 当前状态

### 后端
- **测试**：1694 passed, 14 skipped, 0 failed（`make test` 全绿）
- **Lint**：`make lint` 仍有 2 个 pre-existing 错误（`prompt.py:207` F841, `prompt.py:306` E501）——这是本次之前就存在的，不是我们引入的，不用处理
- **新增文件**：
  - `packages/harness/deerflow/agents/middlewares/training_data_middleware.py`
  - `app/gateway/routers/feedback.py`
  - `scripts/extract_e2e_sessions.py`
  - `scripts/training_dashboard.py`

### 前端
- **typecheck**：`pnpm typecheck` 通过
- **新增文件**：`src/components/feedback/feedback-buttons.tsx`
- **修改文件**：`src/core/api/api-client.ts`, `message-list-item.tsx`, `subtask-card.tsx`, `message-list.tsx`

### 数据目录
```
packages/agent/backend/.deer-flow/training-data/
├── auto-collected/
│   └── 55c0aad9-5582-46e8-b443-acbf6835cc68.jsonl  ← 真实录制，3条
└── feedback/
    └── new.jsonl  ← 手动测试时 bug 留下的（thread_id="new"，已修复）
```
`new.jsonl` 这个文件是测试时留的，可以删掉：
```bash
rm packages/agent/backend/.deer-flow/training-data/feedback/new.jsonl
```

---

## 3. 未完成 / 下一步工作

### 3.1 飞轮运营（不需要写代码）
- 每周让行为学同事多用几次，参考 [docs/sop/training-data-flywheel-sop.md](docs/sop/training-data-flywheel-sop.md)
- 每周一运行 `cd packages/agent/backend && make training-stats` 看进度
- 两个月目标：800 SFT + 300 DPO → 进 Phase 1 微调

### 3.2 反馈关联精度改进（v0.2，低优先级）
- 当前 `extract_e2e_sessions.py` 用"第一条反馈应用于整个 thread" 的简单 join
- 前端已经在发真实 `message_id`（如 `lc_run--019db86d-...`），后端只需按 message_id 精确匹配
- 文件：`scripts/extract_e2e_sessions.py` 第 74 行附近的 `fb = next(iter(feedbacks), None)`

### 3.3 云部署数据收集（如果需要）
训练数据落在本地 `.deer-flow/training-data/`，云部署时：
- **简单方案**：挂持久化卷 + `rsync -avz user@server:/path/.deer-flow/training-data/ ./backup/`
- **自动化方案**：在 `extract_e2e_sessions.py` 后加 `aws s3 sync .deer-flow/training-data/processed/ s3://bucket/`

### 3.4 路线图中的下一个 Phase 工作
读 [docs/roadmap.md](docs/roadmap.md) 和最新 handoff [docs/handoffs/2026-04-23-training-data-flywheel-done.md](docs/handoffs/2026-04-23-training-data-flywheel-done.md) 确认下一优先级。当前 Phase 0 还有 EPM/OFT 范式待补全。

---

## 4. 关键架构提醒

- **agent 鲁棒性优先**：`TrainingDataMiddleware` 的所有 IO 都在 try/except 内，失败只 warning log，绝不上抛
- **受保护文件**：`lead_agent/agent.py` 和 `gateway/app.py` 是 DeerFlow fork 的受保护文件，改了要在 `scripts/sync-deerflow.sh` 里标人工判断
- **GLM-5.1 正面提示**：prompt 里不要用"禁止X"，用正面指令
- **前端 threadId 来源**：一定用 `useThreadChat().threadId`（真实 UUID），不要直接用 `useParams().thread_id`（`/new` 路由下会是字面量 `"new"`）

---

## 5. 接手的第一步

1. 读 [docs/roadmap.md](docs/roadmap.md) 确认当前 Phase 优先级
2. 删掉测试留下的脏数据：`rm packages/agent/backend/.deer-flow/training-data/feedback/new.jsonl`
3. 确认测试全绿：`cd packages/agent/backend && make test`
4. 根据 roadmap 选下一个任务开始

---

## 6. 快速命令参考

```bash
# 后端测试
cd packages/agent/backend && source .venv/bin/activate && make test

# 飞轮进度
cd packages/agent/backend && source .venv/bin/activate && make training-stats

# 数据处理
cd packages/agent/backend && source .venv/bin/activate && PYTHONPATH=. python scripts/extract_e2e_sessions.py

# 启动服务
cd packages/agent && make dev  # → http://localhost:2026

# 前端类型检查
cd packages/agent/frontend && pnpm typecheck
```
