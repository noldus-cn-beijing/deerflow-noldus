# 2026-05-12 反馈按钮 CSRF + auth bypass + 路由升级 完成交接

## 背景

前置交接：[2026-05-12-feedback-button-csrf-and-upstream-sync-handoff.md](2026-05-12-feedback-button-csrf-and-upstream-sync-handoff.md)
设计稿：[../../superpowers/specs/2026-05-12-feedback-csrf-fix-and-run-scoped-design.md](../../superpowers/specs/2026-05-12-feedback-csrf-fix-and-run-scoped-design.md)
实施计划：[../../superpowers/plans/2026-05-12-feedback-csrf-fix-and-run-scoped-plan.md](../../superpowers/plans/2026-05-12-feedback-csrf-fix-and-run-scoped-plan.md)

## 解决的问题

1. ✅ 反馈按钮无反应（CSRF header 缺失）
2. ✅ 后端 feedback router auth bypass（任何登录用户可向任意 thread 提交）
3. ✅ nginx 非标端口下 `Host $host` 丢端口号导致 CSRF cookie 域不匹配
4. ✅ JSONL 反馈与上游 SQLite FeedbackRepository 平行存在的架构碎片化
5. ✅ 反馈未带 run_id，无法精确定位训练样本

## 主要改动

### 后端
- 扩展 `FeedbackRow`：加 `verdict (String(16))` + `revised_text (Text)`，`rating` 改 nullable，UniqueConstraint 加 `message_id`
- `FeedbackRepository.upsert` 加 verdict/revised_text/message_id 参数；上游 rating-only 路径保留兼容
- 重写 `routers/feedback.py`：URL 切换到 `/api/threads/{tid}/runs/{rid}/feedback`；加 `@require_permission`
- Alembic migration scaffold（生产部署用）
- 15 条 pytest 测试覆盖（ORM + 路由 + 兼容性）

### 前端
- `submitFeedback/listFeedback` 签名加 `runId`；改 `csrfFetch` 自动 CSRF
- `useThreadStream` 在 `onLangChainEvent` 捕获 message → run_id 映射并暴露
- `FeedbackButtons` 加 `runId` 必传 prop；失败时按钮下方红字提示 3s 自动消失

### 基础设施
- cherry-pick 上游 `70737af7`：nginx `Host $host` → `$http_host`（14 处）

## QA 验证结果

自动验证：
1. ✅ Backend `make test`: 2231 passed, 5 pre-existing failures (auth/ethoinsight/lead_prompt, unrelated)
2. ✅ Frontend `pnpm typecheck`: passes
3. ✅ Frontend `pnpm lint`: pre-existing issues in test config files only
4. ✅ Dev startup: nginx proxies health check (200 OK)
5. ✅ DB feedback table schema: verdict VARCHAR(16), revised_text TEXT, rating INTEGER (nullable), uq_feedback_thread_run_user_message
6. ✅ Feedback route mounted: 401 response confirms AuthMiddleware active

手动浏览器 QA 清单（待执行）：
- [ ] 1. 发一条 message → 点 ✅ → 按钮变 "已反馈（✅ 正确）"
- [ ] 2. F12 Network 查看 POST 200、X-CSRF-Token 非空、csrf_token cookie
- [ ] 3. 点 ⚠️ → 修订文本框 → 提交 → 按钮变 "已反馈（⚠️ 需修正）"
- [ ] 4. DB 验证：SELECT verdict, revised_text, message_id, run_id FROM feedback
- [ ] 5. 故障注入 1：删除 csrf_token cookie → 红字 "提交失败，请重试" (3s 消失)
- [ ] 6. 故障注入 2：跨用户访问 → 403/404
- [ ] 7. 同一 message 二次反馈 → upsert 不增行

## 后续未完成项

- 🟡 **PR-B 安全批**：拉上游 6 条 sandbox/upload/auth 安全修复（详见 spec §11 与设计阶段分类清单）
- 🟡 **PR-C 稳定性批**：拉上游 32 条小 bug fix
- 🟡 **PR-D 增强批**：拉上游 12 条 feat/refactor（高风险，需 surgical merge）
- 🟡 **导出脚本**：`scripts/export_feedback_jsonl.py` 离线导 SQLite → Fireworks JSONL 供训练流水线使用（待实现）
- 🟡 **subtask 反馈**：subtask-card.tsx 中的反馈按钮因 subtask 上下文无 run_id 临时不渲染，需后续设计 subtask 反馈数据模型
- 🟡 **浏览器 QA**：上述 7 项手动验证待执行

## 验收清单

- [x] `make test`（backend）全绿（5 pre-existing failures, unrelated）
- [x] `pnpm check`（frontend）typecheck 通过（lint 有 pre-existing issues）
- [ ] 浏览器 7 项 QA 全 PASS（待手动执行）
- [x] DB feedback 表含 verdict/revised_text 字段
- [x] CLAUDE.md / SOP 已更新

## Commit 历史

```
220c0d9f docs: 反馈走 SQLite + run-scoped URL；CLAUDE.md 标注多用户/Tier 4 已合状态
fe78b551 feat(frontend): 反馈按钮接入 run-scoped 路由 + csrfFetch + 错误提示
710dc346 fix(nginx): cherry-pick 上游 70737af7 — Host $http_host 修非标端口 CSRF
10a864f7 test(feedback): C1 兼容性回归——thread_runs messages 端点仍能挂 feedback
a818b164 test(feedback): 路由集成测试覆盖 CSRF / run 校验 / verdict 422 / GET / 多 message
7abf1434 feat(feedback): 路由切换到 run-scoped + 闭 auth bypass
d658fa83 feat(feedback): Alembic migration 加 verdict / revised_text / message_id unique
ba4ed91f feat(feedback): Repository.upsert 支持 verdict / revised_text / message_id 复合 key
2301ccb1 feat(feedback): 扩展 FeedbackRow 加 verdict + revised_text + message_id 复合 unique
```
