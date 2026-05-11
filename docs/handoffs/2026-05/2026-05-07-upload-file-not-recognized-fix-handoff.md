# 文件上传 Agent 无法识别 — Bug 修复交接

**日期**: 2026-05-07
**类型**: Bug 修复（端到端测试发现）
**状态**: 🟢 已修复，待用户验证

---

## 0. TL;DR

用户端到端测试发现：上传文件后，agent 在沙箱中 `find /mnt/user-data -type f` 找不到文件。

**根因**: Round 2 wrap-up commit `9e542bb6` 在 `uploads/manager.py` 中为上传路径加了 `user_id` 参数（per-user 文件系统隔离），但 `ThreadDataMiddleware`、`UploadsMiddleware`、`archiving_summarization.py` 中的路径计算未同步更新。

**修复**: 4 个文件，统一在路径计算时传入 `user_id=get_effective_user_id()`。

**测试**: 2 failed, 1877 passed, 14 skipped（仅 2 个 pre-existing 失败）。

---

## 1. 问题链路

| 步骤 | 代码位置 | 路径 (修复前) |
|------|---------|--------------|
| **上传** | `uploads/manager.py:43` | `sandbox_uploads_dir(tid, user_id=get_effective_user_id())` → `{base}/users/default/threads/{tid}/user-data/uploads/` |
| **Agent 路径解析** | `thread_data_middleware.py:60` | `sandbox_uploads_dir(tid)` → `{base}/threads/{tid}/user-data/uploads/` |

当前 auth 模式 `noop`，`get_effective_user_id()` 返回 `"default"`。上传和读取使用不同目录，agent 自然找不到文件。

## 2. 修改的文件

| 文件 | 修改 |
|------|------|
| `packages/harness/deerflow/agents/middlewares/thread_data_middleware.py` | `_get_thread_paths()` / `_create_thread_directories()` 增加 `user_id` kwarg；`before_agent()` 传入 `get_effective_user_id()` |
| `packages/harness/deerflow/agents/middlewares/uploads_middleware.py` | `sandbox_uploads_dir()` 调用增加 `user_id=get_effective_user_id()` |
| `packages/harness/deerflow/agents/middlewares/archiving_summarization.py` | `sandbox_work_dir()` + `thread_dir()` 调用增加 `user_id=get_effective_user_id()` |
| `tests/test_uploads_middleware_core_logic.py` | `_uploads_dir()` helper 同步使用 `user_id` |

## 3. 未修改的已知不一致（非本轮范围）

以下位置也未传 `user_id`，但不影响当前 local sandbox 模式的端到端测试：

- `aio_sandbox_provider.py:_get_thread_mounts()` — Docker sandbox 静态方法，需较大重构
- `gateway/routers/uploads.py:96, 177` — 响应元数据字段，实际文件存储路径已正确
- `channels/manager.py` / `channels/feishu.py` — IM 通道模块

## 4. 验证步骤

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend

# 跑测试（确认只有 2 个 pre-existing 失败）
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5

# Lint
PYTHONPATH=. uv run ruff check packages/harness/deerflow/agents/middlewares/thread_data_middleware.py \
  packages/harness/deerflow/agents/middlewares/uploads_middleware.py \
  packages/harness/deerflow/agents/middlewares/archiving_summarization.py

# 重启 agent 服务，然后在 UI 中上传文件测试
make dev
```

## 5. 当前 git 状态

修改未 commit，工作区有改动：

```bash
git diff --name-only
# packages/harness/deerflow/agents/middlewares/archiving_summarization.py
# packages/harness/deerflow/agents/middlewares/thread_data_middleware.py
# packages/harness/deerflow/agents/middlewares/uploads_middleware.py
# tests/test_uploads_middleware_core_logic.py
```

## 6. 接手路径

1. 等用户重启 agent 服务并验证端到端测试
2. 如果验证通过，commit 这 4 个文件
3. 如果用户还发现其他路径不一致，按同样模式修复（加 `user_id=get_effective_user_id()`）
