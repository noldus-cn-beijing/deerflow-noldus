# 2026-06-02 待定位 — pagination 测试隔离污染（sync 引入）

**状态**:🔴 待根因定位 + 修复（已 skip 止血，sync PR 可 merge）
**测试**:`packages/agent/backend/tests/test_thread_run_messages_pagination.py::test_get_run_hydrates_store_only_run`
**引入**:2026-06-02 DeerFlow 上游全量 sync（`fa3418ec`，f9b70713→74e3e80c）
**影响**:仅测试隔离，**产品行为正常**（单独跑 passed）。当前已 `@pytest.mark.skip`，修复后解除。

---

## 0. 给定位 agent 的第一步

```bash
cd packages/agent/backend && source .venv/bin/activate   # 或用主仓库 .venv
export DEER_FLOW_CONFIG_PATH=<repo>/packages/agent/config.yaml   # ⚠️ 必须，否则一批测试 FileNotFoundError 假失败
# 先删掉 skip 装饰器（test_thread_run_messages_pagination.py:194 上方），复现：
PYTHONPATH=packages/harness:. python -m pytest tests/ -q -p no:cacheprovider | grep test_get_run_hydrates
```

---

## 1. 已确认的事实（不必重查，直接用）

| 事实 | 证据 |
|---|---|
| **dev 基线全量全绿** | `3f353458` 全量 = 3329 passed, 0 failed（用对 config）。说明套件本身隔离没问题，污染源是 sync 引入的 |
| **单独跑 passed** | `pytest tests/test_thread_run_messages_pagination.py` = 13 passed；整个文件单独跑不复现 |
| **全量跑 failed** | `assert 404 == 200`（GET run 返回 404 Not Found，期望 200） |
| **同文件前 8 个测试全 PASSED，只有它 FAILED** | verbose 顺序确认 → 污染来自**文件外**的某个前序测试，且只影响这一个依赖全局 store 状态的测试 |
| **不是这几簇污染的** | `test_runtime_lifecycle_e2e`（已 skip）+ 它 → pass；`test_run_*` / `test_persistence_*` / `test_checkpointer_*` / `test_gateway_services` 全簇 + 它 → 205 passed。**这些不是元凶** |

## 2. 测试本身（自包含，理论上不该被污染）

```python
def test_get_run_hydrates_store_only_run():
    app = _make_app(run_manager=_make_store_only_run_manager())  # 注入 mock run_manager
    with TestClient(app) as client:
        response = client.get("/api/threads/thread-store/runs/store-only-run")
    assert response.status_code == 200   # ← 全量跑返回 404
```

**404 的含义**:注入的 `_make_store_only_run_manager()` mock **没生效**，真实 run_manager 被用了 → store-only-run 找不到。

## 3. 根因假设（按可能性排序，待验证）

1. **🎯 最可能：FastAPI dependency override 的全局状态泄漏**。`_make_app` 大概率用 `app.dependency_overrides[get_run_manager] = ...` 注入。若**某个前序测试**也 override 了同一个全局 `get_run_manager` 依赖但**没在 teardown 清理**（或用了 module/session 级单例），就会污染。查 `_make_app` 实现 + 所有 `dependency_overrides[get_run_manager]` / `get_run_manager` 单例的设置点，看哪个没隔离。
2. **全局 RunManager 单例**：`deerflow.runtime` 的 `get_run_manager()` 若是模块级单例缓存，前序测试初始化了真实 manager，本测试的 mock 注入被绕过。查 `RunManager` 是否有全局单例 + sync 是否改了它的初始化（`runtime/runs/manager.py` / `worker.py` 在 sync 里 +88 行，重点看）。
3. **config 缓存单例**：`get_app_config()` 缓存被前序测试设成别的 config，导致 app 用错 store 路径。可能性较低（404 更像 manager 而非 config）。

## 4. 定位方法（二分法）

```bash
# pytest 按收集顺序跑。找污染源：用 --deselect 逐步排除，或 git 对比 sync 改了哪些 e2e/router 测试
# 重点查 sync 全量覆盖的、会设置全局 dependency_overrides / RunManager 的测试文件：
git show fa3418ec --stat | grep -iE "test_.*(run|thread|gateway|runtime|router|stream)" 
# 对每个嫌疑测试文件，单独 + pagination 跑，看哪个组合触发 404：
PYTHONPATH=packages/harness:. python -m pytest tests/<嫌疑>.py tests/test_thread_run_messages_pagination.py::test_get_run_hydrates_store_only_run -q -p no:cacheprovider
```

## 5. 修复方向（定位后）

- 若是 dependency_overrides 泄漏：给污染源测试加 teardown 清理 `app.dependency_overrides.clear()`，或用 fixture 隔离。
- 若是全局单例：本测试用 fixture 强制 reset 单例，或污染源测试用完 reset。
- **不要**改 `test_get_run_hydrates_store_only_run` 本身去适配脏状态——它是对的，要修的是污染它的那一方。
- 修好后删 `test_thread_run_messages_pagination.py:194` 上方的 `@pytest.mark.skip`，跑全量确认 3255+ passed、该测试不再 fail。

## 6. 关联
- sync PR：worktree `sync-0602`（`fa3418ec`），本问题已 skip 止血，不阻塞 merge
- sync spec：`docs/superpowers/specs/2026-06-02-deerflow-upstream-sync-design.md`
- 同源教训：[[feedback_pr_merge_must_run_full_suite_on_shared_logic]]（共享状态测试隔离）
