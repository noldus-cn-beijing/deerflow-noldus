# 2026-05-08 Subagent ContextVar 跨线程丢失修复 — 完成交接

## TL;DR

修复 subagent 跑在独立 ThreadPoolExecutor + 独立 event loop 时丢失父任务 ContextVar
（user_id）的 bug。症状：data-analyst 看不到 code-executor 写的文件，因路径
fallback 到 users/default/。

通过合入上游 deerflow 的 contextvars.copy_context() + 持久 isolated event loop
设计修复。改动范围：1 个核心文件 + 1 个回归测试文件 + 1 个已有测试适配。

## 改动清单

- `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` — surgical merge
  - 新增 import: `atexit`, `Callable`, `Coroutine`, `Context`, `copy_context`
  - 新增: 持久 isolated event loop 基础设施（`_isolated_subagent_loop` 等 6 个函数/变量）
  - 删除: `_isolated_loop_pool` ThreadPoolExecutor + `_execute_in_isolated_loop` 方法
  - 改写: `execute()` 使用 `copy_context()` + `_submit_to_isolated_loop_in_context`
  - 改写: `execute_async()` 在 `_scheduler_pool.submit` 前抓 `parent_context`
- `packages/agent/backend/tests/test_subagent_user_context_propagation.py` — 新增 6 个回归测试
- `packages/agent/backend/tests/test_subagent_executor.py` — 适配 `test_timeout_does_not_overwrite_cancelled`

## 验证

- [x] `make test` 通过: 5 failed (pre-existing), 2148 passed, 14 skipped (baseline 2142 + 6 new)
- [x] `ruff check` 0 error
- [x] `ruff format` 通过
- [ ] 端到端 shoaling pipeline 跑通，data-analyst 看到产物（待用户执行 §Step 7）

## 与上游的关系

合入内容（来自 `deerflow/main`）:
- `_isolated_subagent_loop` 持久 loop 基础设施
- `_submit_to_isolated_loop_in_context` + `copy_context()` 入口
- `_shutdown_isolated_subagent_loop` + `atexit.register`

未合入（留给下轮 deerflow sync）:
- `tool_policy.py` 重构 / `Skill.allowed_tools`
- `resolve_subagent_model_name`
- `_create_agent` 签名变化（`tools` 参数）
- `_build_initial_state` 异步化（`async + tuple 返回`）
- `SystemMessage` 引入（上游 skill 注入方式）

## 经验沉淀

1. ContextVar 不跨 ThreadPoolExecutor 自动传——这是 Python 标准库设计，不是 bug。
   任何自创线程池都需要 copy_context() 显式拷贝。
2. silent fallback 是这一类 bug 的放大器：get_effective_user_id() 静默回到
   "default"，让"未认证"和"已认证但状态丢失"两种语义合并到一条路径，问题只有
   在文件系统出现两份目录时才暴露。下一轮基础设施加固应考虑改成 require 抛错。
3. better-auth 同步引入的 user_context ContextVar 体系，每个独立线程入口都需要
   桥接一次。lead_agent 已修（02547092），subagent 是本次修，未来如果加新的
   subprocess/线程入口（IM channels 异步处理、batch worker 等）要重新评估。

## 后续 / 不在本次范围

- 下轮 deerflow sync 时考虑整覆盖 executor.py，引入 tool_policy.py / Skill.allowed_tools
- 单独 issue：评估把 get_effective_user_id() 改成 require_effective_user_id() 抛错
- 端到端验证：重启服务后跑 shoaling pipeline，确认 data-analyst 产物在真实 user_id 目录下
