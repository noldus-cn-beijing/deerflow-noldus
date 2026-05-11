# DeerFlow 上游 Tier 2/3/4 同步 - 轮 1 完成交接

**日期**: 2026-05-07
**交接人**: Claude (本会话, opus-4-7-1m)
**接手对象**: 下一位 AI Agent
**任务状态**: ✅ 轮 1 完成 (3 个 commit 已落地, 测试全绿, 等用户 review)
**前置依赖**: [2026-05-07-deerflow-tier234-execution-plan.md](2026-05-07-deerflow-tier234-execution-plan.md)

---

## 0. TL;DR

按计划执行了 33 个 commit 的轮 1 同步, 实际只需要应用约 12 个核心 patch — 剩下的 21 个上游 commit 在前期 squash sync 或 Tier 1 sync 中已经吸收到 noldus。

**3 个 commit 已落地**:
- `08275b04` Tier 1-sibling A 阶段 (19 commit)
- `0753456b` Tier 2-keep B 阶段 (9 commit, 含 6bd88fe1 大型 surgical merge)
- `413bb4d8` Tier 3 ⭐⭐ C 阶段 (5 commit, 含 1 个延后)

**测试**: 1714 → 1811 passed (+97 net, 主要来自 sandbox audit 测试覆盖 75→147 + logging level 14 + view_image 16 + setup_agent 11), 2 pre-existing failures 不变。

**唯一延后**: C.5 (`f9ff3a69` summarization skill rescue), 600+ 行新功能, 与 BeforeSummarizationHook 协调风险大, 转 round 2。

---

## 1. 已完成的 commit 清单

### A 阶段: T1-sibling 19 commit (commit 08275b04)

| # | SHA | 作用 | 状态 |
|---|---|---|---|
| 1 | `e543bbf5` | 拒绝 symlinked upload | ✅ 应用 (uploads/manager.py + channels + uploads.py) |
| 2 | `707ed328` | skill archive scan | ✅ 应用 (installer.py + __init__.py + skills 路由) |
| 3 | `f7dfb88a` | aio-sandbox redact env logs | ✅ 应用 (local_backend.py) |
| 4 | `80e210f5` | 文档转换需 opt-in | ✅ 应用 (uploads.py + 测试新增) |
| 5 | `3b3e8e1b` | bash 命令审计加强 | ✅ 已在 noldus, 仅取上游测试 |
| 6 | `a664d2f5` | checkpointer parent dir | ✅ 已在 noldus |
| 7 | `0948c7a4` | codex streamed output | ✅ 已在 noldus |
| 8 | `24fe5fbd` | mcp get_cached fix | ✅ 已在 noldus |
| 9 | `29817c3b` | memory tz-aware UTC | ✅ 已在 noldus |
| 10 | `1df389b9` | web_fetch readability async | ✅ 已在 noldus |
| 11 | `718dddde` | sandbox file lock leak | ✅ 已在 noldus |
| 12 | `0b6fa8b9` | sandbox orphan cleanup | ✅ 已在 noldus |
| 13 | `ad6d934a` | clarification options coerce | ✅ 已在 noldus |
| 14 | `f4c17c66` | present_files thread id | ✅ 已在 noldus |
| 15 | `e4f896e9` | todo middleware premature exit | ✅ 已在 noldus |
| 16 | `bb8b234d` | codex token usage | ✅ 已在 noldus |
| 17 | `b1aabe88` | client stream token deltas | ✅ 已在 noldus |
| 18 | `f80ac961` | legacy skills fallback | ⏸️ 跳过 (noldus 默认即 legacy 路径) |
| 19 | `eba3b9e1` | log_level 统一 | ✅ 应用 (app_config.py + gateway/app.py) |

**测试**:
- 新增 `test_logging_level_from_config.py` (14 tests)
- 替换 `test_sandbox_audit_middleware.py` (147 tests, 上游 superset)
- 更新 `test_uploads_router.py` (3 个新测试 + 2 个修改)
- 更新 `test_client_e2e.py` (添加 `_allow_skill_security_scan` fixture)

### B 阶段: T2-keep 9 commit (commit 0753456b)

| # | SHA | 作用 | 状态 |
|---|---|---|---|
| B.1 | `a62ca5dd` | httpx.ReadError retriable | ✅ 已在 noldus |
| B.5 | `ca1b7d5f` | ls_tool path masking | ✅ 应用 (sandbox/tools.py 1 行 + thread_data 初始化) |
| B.7 | `5ba1dacf` | present_file → present_files | ✅ 应用 (lead_agent/prompt.py 2 处) |
| B.8 | `2176b2bb` | bootstrap agent name 校验 | ✅ 应用 (lead_agent/agent.py + 已有 setup_agent_tool) |
| B.4 | `af8c0cfb` | view_image 限制 thread data 路径 | ✅ 应用 (view_image_tool 重写 + factory.py + sandbox/tools.py 公开 API) |
| B.3 | `6bd88fe1` | sandbox bash traversal 防御 | ✅ 应用 (常量 + 8 个 helper + 重写 validate_local_bash_command_paths) |
| B.6 | `4d4ddb3d` | LLM circuit breaker | ✅ 已在 noldus |
| B.2 | `e5b14906` | event loop fix v1 | ✅ 已在 noldus |
| B.2' | `7dea1666` | isolated loop refactor | ⏸️ 跳过 (e5b14906 已解决根本问题, refactor 风险大) |

**测试**:
- 新增 `test_setup_agent_tool.py` (11 tests, 修改了 3 个 user_context 路径假设)
- 新增 `test_view_image_tool.py` (16 tests)
- 测试 `test_create_deerflow_agent.py` (2 处 `sandbox=False` → `sandbox=True`)

**风险提示**: B.3 是大型 surgical merge (200+ 行), 引入了:
- 新常量: `_URL_*_PATTERN`, `_DOTDOT_*`, `_LOCAL_BASH_CWD_COMMANDS`, `_SHELL_*` 等
- 新函数: `_is_non_file_url_token`, `_split_shell_tokens`, `_validate_local_bash_shell_tokens`, `_is_allowed_local_bash_absolute_path`, `_next_cd_target`, `_validate_local_bash_cwd_target`, `_looks_like_unsafe_cwd_target`, `_validate_local_bash_root_path_args`
- `validate_local_bash_command_paths` 主体逻辑重写

**保留 noldus 全部定制**: `{{shared://}}` 占位符, `SHARED_PATH_PREFIX`, `extra_env`, `mask_local_paths_in_output`。

83 个 sandbox security 测试全过, 318 个 sandbox 测试全过。

### C 阶段: T3 ⭐⭐ 5 commit (commit 413bb4d8)

| # | SHA | 作用 | 状态 |
|---|---|---|---|
| C.1 | `f514e35a` | clarification 幂等性 | ✅ 已在 noldus |
| C.2 | `5db71cb6` | dangling tool-call provider raw | ✅ 已在 noldus |
| C.3 | `ec8a8cae` | gate deferred MCP execution | ✅ 应用 (deferred_tool_filter_middleware + tool_search.py) |
| C.4 | `11f557a2` | trace run_name | ✅ 应用 (suggestions + security_scanner; memory + title 已在 noldus) |
| C.5 | `f9ff3a69` | summarization skill rescue | ⏸️ **延后到 round 2** (600+ 行大功能) |

---

## 2. 跳过项及原因

### 2.1 永久跳过

- **`f80ac961`** (A.3): noldus `get_skills_root_path()` 默认就是 backend.parent/"skills" — 即上游所谓"legacy"路径。无需引入 fallback 逻辑。
- **`7dea1666`** (B.2'): noldus 已合 `e5b14906` (用 `_isolated_loop_pool` ThreadPoolExecutor 隔离), 它解决了 #1965 描述的根本问题。`7dea1666` 改用持久 event loop + atexit 清理是 refactor, 引入新的 `_isolated_subagent_loop_thread` 全局状态, 在 noldus 重定制的 executor.py 上风险大。

### 2.2 延后到轮 2

- **`f9ff3a69`** (C.5) summarization skill rescue:
  - 600+ 行新功能, 涉及:
    - `summarization_config.py` 新增 4 个 config 项 (`preserve_recent_skill_count`, `preserve_recent_skill_tokens`, `preserve_recent_skill_tokens_per_skill`, `skill_file_read_tool_names`)
    - `summarization_middleware.py` 新增完整的 skill 抢救逻辑 (lift skill bundles out before summarization)
    - `lead_agent/agent.py` 工厂函数注入 skills_container_path
  - **风险**: noldus `DeerFlowSummarizationMiddleware` 已经有 `BeforeSummarizationHook` 协议 (memory_flush_hook, archive hook), 引入 skill rescue 的初始化路径可能与现有 hook 链冲突
  - **建议**: 轮 2 与 E (skill storage 重构) 一起做, 协调 skill 文件路径检测逻辑

---

## 3. 关键状态验证

### 3.1 测试基线
```
Before (8b19c667): 2 failed, 1714 passed, 14 skipped
After  (413bb4d8): 2 failed, 1811 passed, 14 skipped  (+97 tests)
```

2 个 pre-existing failures 与本次同步无关:
- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

### 3.2 受保护文件状态
- `agents/lead_agent/prompt.py`: 仅 2 处 `present_file` → `present_files`, 中文调度规则/Gate 反问/EV19 模板路径 全保留
- `agents/lead_agent/agent.py`: 仅 1 处 `validate_agent_name(cfg.get("agent_name"))`, 中间件链全保留
- `subagents/executor.py`: 0 改动 (e5b14906 已在, 7dea1666 跳过), recursion_limit/max_turns/{{shared://}} 全保留
- `sandbox/tools.py`: B.3 + B.5 + B.4 surgical merge, `mask_local_paths_in_output` / `{{shared://}}` 占位符 / `SHARED_PATH_PREFIX` 全保留
- `agents/middlewares/llm_error_handling_middleware.py`: 0 改动, 总超时 + 多种 timeout 关键字 + circuit breaker 全保留

### 3.3 git log
```
413bb4d8 sync deerflow upstream Tier 3 ⭐⭐ 精选: 5 个高价值改进
0753456b sync deerflow upstream Tier 2-keep: 受保护文件 surgical merge 9 commit
08275b04 sync deerflow upstream Tier 1-sibling: 安全修复 + bug fix + 增强 19 commit
8b19c667 update from upstream tier 1 before may
```

---

## 4. 轮 2 backlog (下次会话)

### E. Skill storage 重构 (`1ad1420e`)
- 设计文档 §6 已详述, 4-6 小时
- 与 C.5 (summarization skill rescue) 协调

### D. Tier 4 BC 持久化层 (11 commit)
- 设计文档 §6 已详述, 5-7 小时
- `persistence/`, `runtime/checkpointer/`, `runtime/events/` 3 个新目录

### 与 round 1 关联性
- C.5 留给轮 2 是因为 skill rescue 的核心是 "tool name + container path 检测", 与 E 的 skill storage 路径系统强相关 — 一起做更安全。

---

## 5. 完成度统计

```
105 个上游 commit
├── T1-DONE 11 (上次会话已合)
├── 轮 1 已合 33 = T1-sibling 19 + T2-keep 9 + T3 5
│   ├── 实际应用 patch 12 个
│   └── 验证 0-diff (already in noldus) 21 个
├── 留 round 2 1 (C.5)
├── 留轮 2/3 backlog 60
└── 永久跳过 2 (f80ac961 in noldus / 7dea1666 refactor risk)
```

---

## 6. 下一位 Agent 的第一步

1. 读这份文档 + 读上一份 `2026-05-07-deerflow-tier234-execution-plan.md` §6 (轮 2 backlog)
2. 跑当前测试基线确认 2/1811:
   ```bash
   cd /home/wangqiuyang/noldus-insight/packages/agent/backend
   PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
   ```
3. 选择切入点:
   - **方案 A** (推荐): 先做 E (skill storage 重构) + C.5 (summarization rescue), 这两个是配套的
   - **方案 B**: 先做 D (Tier 4 BC 持久化), 引入 persistence/ 目录但用 memory backend, 不破坏现状
4. 每完成一个 phase commit + 测试, 不要 push origin

---

## 7. 风险与已知问题

### ⚠️ B.3 大型 surgical merge — 后续监控

`validate_local_bash_command_paths` 重写后增加了:
- shlex 词法分析
- token 级语义检测 (cd/pushd target 检查, 根路径检测, command wrappers)
- URL span 排除 (避免 https:// 被当作绝对路径)

测试全过 (83/83), 但生产环境可能遇到:
- 某些 LLM 生成的 bash 命令组合触发新的 false positive
- shlex 在异常 quoting 下 fallback 到 split() — 可能漏检某些情况

**建议**: 在 v0.1 上线前观察 1-2 周生产 bash 工具调用日志, 看 `Unsafe absolute paths in command` / `path traversal detected` / `Unsafe working directory change` 三类错误的分布。

### ⚠️ skill 安装现在需要 LLM 调用

`707ed328` 引入的 `scan_skill_content` 在 install 路径中是必经步骤, 调用 LLM (`config.skill_evolution.moderation_model_name` 或默认 model)。如果 LLM endpoint 不可用, skill 安装会失败。测试用 `test_client_e2e.py` 的 `_allow_skill_security_scan` fixture mock 掉了。

**生产环境**: 确保 model server 可用, 否则 skill 安装会 503/timeout。

### ⚠️ debug.py 没合 eba3b9e1

noldus 的 `backend/debug.py` 是简单 REPL, 与上游 debug 子系统结构不同。`apply_logging_level` 没引入到 debug.py — 对 LangGraph + Gateway 启动的日志级别不影响 (那是 gateway/app.py 控制), 但 `python debug.py` 直接启动时不会读 `config.yaml log_level`。如果未来需要, 在 debug.py 中调一次 `apply_logging_level` 即可。

---

## 8. 不要 push

所有 commit 留在本地 `dev` 分支, 等用户决定是否 push。

```bash
git log --oneline 8b19c667..HEAD  # 查看本次 3 个 commit
```
