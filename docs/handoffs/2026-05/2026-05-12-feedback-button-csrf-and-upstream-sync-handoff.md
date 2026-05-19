# 2026-05-12 反馈按钮无反应问题诊断 + DeerFlow 上游 sync 评估

## 反馈按钮：根因分析

### 现象

聊天页面 assistant 消息下方的反馈按钮（✅ 正确 / ⚠️ 需修正 / ❌ 错误）点击后无任何反应。

### 根因

前端 `submitFeedback()` 使用原生 `globalThis.fetch` 发送 POST 请求，缺少 `X-CSRF-Token` header。后端 `CSRFMiddleware` 拦截并返回 403，前端 catch 块 silently ignore。

**链路**：

```
submitFeedback() [api-client.ts:95]
  → 使用 globalThis.fetch，没有 X-CSRF-Token header
  → 也没有 credentials: "include"（auth cookie 可能也不带）
CSRFMiddleware [csrf_middleware.py:73]
  → POST 请求触发 should_check_csrf()
  → 缺少 cookie csrf_token 或 header X-CSRF-Token → 403
FeedbackButtons [feedback-buttons.tsx:53]
  → catch 块 // silently ignore
  → 用户看到的：按钮无变化
```

### 涉及文件

| 文件 | 问题 |
|------|------|
| `packages/agent/frontend/src/core/api/api-client.ts:95-99` | `submitFeedback` 用原生 fetch，缺 CSRF header + credentials |
| `packages/agent/frontend/src/core/api/api-client.ts:104-108` | `listFeedback` 同样问题 |
| `packages/agent/frontend/src/core/api/fetcher.ts` | 已有封装好的 CSRF-aware `fetch` 函数，但 api-client.ts 没有用它 |
| `packages/agent/backend/app/gateway/csrf_middleware.py` | CSRF 中间件正常工作，不是 bug |

### 修复方案

将 `submitFeedback` 和 `listFeedback` 改为使用 `fetcher.ts` 中封装的 `fetch`（自动注入 `X-CSRF-Token` + `credentials: "include"`）：

```typescript
// api-client.ts 顶部增加 import
import { fetch as csrfFetch } from "./fetcher";

// submitFeedback 中
const res = await csrfFetch(`/api/threads/${threadId}/feedback`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
```

改动量：1 行 import + 替换 2 处 `fetch` → `csrfFetch`。

### 如果修完后依然无反应

检查后端 Gateway 日志，看请求是否到达 `/api/threads/{thread_id}/feedback`：
- 401 → auth cookie 问题
- 403 → CSRF 问题（已修应该不会再出现）
- 500 → `_base_dir()` 路径问题（检查 `config/paths.py`）

## DeerFlow 上游 sync 评估

### DeerFlow 上游（bytedance/deer-flow）反馈系统

| 层面 | 文件 | 状态 |
|------|------|------|
| ORM 模型 | `persistence/feedback/model.py` | `FeedbackRow`：feedback_id, run_id, thread_id, user_id, message_id, rating(+1/-1), comment |
| Repository | `persistence/feedback/sql.py` | 完整 CRUD：create, upsert, delete, list_by_run, list_by_thread, aggregate_by_run |
| Read API | `thread_runs.py:305` | `GET /api/threads/{id}/messages` 自动 attach feedback 到消息 |
| Write API | — | **不存在**，上游没有 POST/PUT/DELETE 反馈的 REST 端点 |

### 对比 Noldus 自定义反馈

| 维度 | DeerFlow 上游 | Noldus 自定义 (`feedback.py`) |
|------|--------------|------------------------------|
| 存储 | SQLite (ORM) | JSONL 文件 |
| 数据类型 | 二元 rating (+1/-1) + comment | 三元 verdict (correct/needs_fix/wrong) + revised_text |
| Write API | 无 | POST + GET `/api/threads/{id}/feedback` |
| Read API | 有（消息列表附带） | 有（GET /api/threads/{id}/feedback） |
| 用途 | 通用用户反馈 | 训练数据飞轮（SFT 种子收集） |

### 结论

Noldus 自定义反馈不能直接替换为上游方案，因为：
1. 上游没有 write API（无法从前端提交反馈）
2. Noldus 的三元 verdict + revised_text 比上游的二元 rating 更适合训练数据收集

后续可优化方向：将 Noldus feedback 的存储层从 JSONL 迁移到上游 `FeedbackRepository`（SQLite），需扩展上游 model（verdict 三分类 + revised_text）。

### 上游 sync 待办

dry-run 摘要（自 `f0dd8cb` 起累积 129 commits / 147 文件，但因 Tier 1-4 手动 sync 已部分覆盖）：

- **13 新文件** — 可安全合入（`dynamic_context_middleware.py`、`loop_detection_config.py`、`skills/manager.py` 等）
- **124 安全文件** — 可批量合入
- **10 受保护文件** — 需逐文件 surgical merge（见 `/tmp/deerflow-sync-report/`）

关注安全修复（未合入）：
- `74081a85` 沙箱 Docker 端口绑定到 loopback
- `6bd88fe1` 阻止沙箱 bash 路径穿越逃逸
- `39c5da94` 防止 local custom mount 符号链接逃逸
- `af8c0cfb` 约束 view_image 只能访问 thread data 路径

建议等当前工作稳定后再批量 sync，避免引入 auth/Tier 4 体系变更导致回归。

## 未完成事项

1. 🔴 修复反馈按钮 CSRF bug（1 行 import + 2 处替换，改动小）
2. 🟡 后续评估：将 Noldus feedback 存储从 JSONL 迁移到上游 FeedbackRepository（扩展 model 支持三分类 verdict + revised_text）
3. 🟡 DeerFlow 上游 sync：129 commits 待合入（10 个受保护文件需 surgical merge）

---

## 状态更新（2026-05-12 晚）

✅ **本 handoff 已闭环** — 见后续完成交接 [2026-05-12-feedback-csrf-fix-completion-handoff.md](2026-05-12-feedback-csrf-fix-completion-handoff.md)。

但**问题边界比本 handoff 描述大很多**：实际修复涵盖（1）前端 CSRF + 错误兜底，
（2）后端 auth bypass 闭洞，（3）nginx 非标端口 Host 丢端口号修复，
（4）反馈架构从 JSONL 全面切到上游 SQLite FeedbackRepository，
（5）路由升级到 `/api/threads/{tid}/runs/{rid}/feedback` 加 run_id 维度
（专家反馈未来可精确定位训练样本）。

也就是说，原计划"1 行 import + 2 处替换"的最小修远远不够——更深的 auth + 架构
问题在调研中被发现并一起解决。详情看完成 handoff 的「解决的问题」段。
