# 交接文档：Event loop is closed 终极修复完成

**日期**: 2026-04-29
**执行分支**: `fix/event-loop-sync-memory` (基于 `dev`)
**Commit**: `f23e2770`

---

## 1. 实际改动清单

```
 .../harness/deerflow/agents/memory/queue.py        |  11 ++
 .../harness/deerflow/agents/memory/storage.py      |  57 +++---
 .../harness/deerflow/agents/memory/updater.py      | 201 ++++++++++++---------
 .../agents/middlewares/memory_middleware.py        |   6 +
 .../packages/harness/deerflow/config/paths.py      | 104 +++++++----
 .../harness/deerflow/runtime/user_context.py       | 167 +++++++++++++++++
 packages/agent/backend/tests/test_memory_queue.py  |   2 +
 .../agent/backend/tests/test_memory_updater.py     |  72 +-------
 8 files changed, 410 insertions(+), 210 deletions(-)
```

| 文件 | 改动性质 |
|---|---|
| `runtime/user_context.py` | **新增**（167 行）— `get_effective_user_id()`, `AUTO`, `resolve_user_id()` |
| `config/paths.py` | **手动合并**（受保护文件）— 加 `_validate_user_id`, `user_dir`/`user_memory_file`/`user_agent_memory_file` 方法，thread_dir 系列加 `user_id` keyword-only 参数。保留本地 `shared_dir` / `SHARED_PATH_PREFIX` 定制 |
| `agents/memory/updater.py` | **全量同步上游** — sync `model.invoke()` 替代 `asyncio.run(coro)` |
| `agents/memory/queue.py` | **全量同步上游** — user_id 参数透传 |
| `agents/memory/storage.py` | **全量同步上游** — 缓存 key → `(user_id, agent_name)` tuple |
| `agents/middlewares/memory_middleware.py` | **全量同步上游** — enqueue 时捕获 `get_effective_user_id()` |
| `tests/test_memory_updater.py` | 删除 4 个过时测试 + 修复 `ainvoke` → `invoke` 断言 + 修复 `user_id=None` 参数 |
| `tests/test_memory_queue.py` | 修复 2 个测试的 `user_id=None` 期望参数 |

## 2. 测试结果

| 指标 | 修复前（基线） | 修复后 |
|---|---|---|
| Passed | 1725 | 1721 |
| Failed | 2 | 2 |
| Skipped | 14 | 14 |

两个 pre-existing 失败（未变）：
- `test_planning_skill_is_enabled_in_config`
- `test_usage_example_shows_ask_clarification_between_analyst_and_writer`

Passed 减少 4 = 删除的 4 个 `_evict_provider_async_client_caches` 测试。

## 3. E2E 烟测

**待用户手动执行：**

1. `cd packages/agent && make dev`
2. 浏览器 http://localhost:2026
3. 创建新 thread（manual mode）
4. 上传 `demo-data/shoaling/` 数据
5. 让 agent 跑分析直到结束
6. 检查：`grep "Event loop is closed" packages/agent/logs/langgraph.log` — 应无新增

## 4. 遗留事项

- [ ] **Gateway auth 与 user_context 集成** — 当前 `get_effective_user_id()` fallback 到 `"default"`。如需真正区分用户，要在 Gateway auth 中间件里调 `set_current_user()`。本次不做，留 follow-up
- [ ] **上游其他安全文件的同步** — `scripts/sync-deerflow.sh` 里 65 个安全文件 + 9 个其他受保护文件仍待合入
- [ ] **Phase 0 P0** — `set_experiment_paradigm` 和 `/mnt/shared` 问题由其他 handoff 处理

## 5. 上游 issue 状态

上游 deerflow 已在 `7289d2bb` (Merge branch 'main' into fix-2615) 正式修复。本次同步已将该 fix 完整带入。之前在 [2026-04-28-deerflow-upstream-issue-draft.md](2026-04-28-deerflow-upstream-issue-draft.md) 草拟的上游 issue 可以标记为"已被上游修复"。

## 6. 回滚

如需回滚：

```bash
git checkout dev
git branch -D fix/event-loop-sync-memory
# 或 git reset --hard HEAD~1  (如果仍在 fix 分支上)
```

`/tmp/memory-backup/` 里有修复前 4 个 memory 文件的原始副本。
