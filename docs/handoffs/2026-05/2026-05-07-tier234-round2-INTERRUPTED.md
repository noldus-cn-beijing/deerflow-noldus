# 轮 2 DeerFlow 上游同步 — 中断交接

**日期**: 2026-05-07
**中断点**: Phase E 完成，但全量测试 24 failed（22 来自上游测试文件的 Tier 4 依赖），用户选择暂停交接
**当前 HEAD**: `2c0db62b docs: 轮 1 deerflow 同步完成交接文档`（未提交任何改动）

---

## 1. 已完成：Phase E (Skill Storage 重构)

### E.1 备份
- `/tmp/noldus-skills-backup-20260507/custom/` — 5 个 custom skill 目录
- `/tmp/noldus-skills-code-backup/` — 8 个旧代码文件

### E.2 新增 storage/ 目录
- `skills/storage/__init__.py` (83 行)
- `skills/storage/skill_storage.py` (254 行, ABC)
- `skills/storage/local_skill_storage.py` (195 行, 实现)
- `config/runtime_paths.py` (新增依赖)

### E.3 更新 5 个 skills/ 文件
- `parser.py`, `security_scanner.py`, `types.py`, `validation.py`, `__init__.py` — 全部整文件覆盖（0 noldus 定制）

### E.4 清理旧文件 + 更新 installer
- 删除 `manager.py` ✅
- 删除 `loader.py` → 但创建了 compat shim `loader.py`（见下）
- `installer.py` 整文件覆盖上游版本 + 追加 compat 函数：
  - `get_skills_root_path()` — 委托到 `get_or_new_skill_storage()`
  - `scan_skill_content` re-export
- `loader.py` compat shim — 提供 `load_skills()` 函数（委托到新 storage API）

### E.5 Surgical merge: prompt.py
- Line 14: `from deerflow.skills import load_skills` → `from deerflow.skills.storage import get_or_new_skill_storage`
- Line 29: `load_skills(enabled_only=True)` → `get_or_new_skill_storage().load_skills(enabled_only=True)`
- 新增 `AppConfig` import
- `get_skills_prompt_section` 签名加 `app_config` 参数
- Noldus 中文规则验证: 16 refs (≥5) ✅

### E.6 其他调用方
- `client.py` — surgical edit: 6 处 import/call 变更（保留 noldus 版本）
- `skills_config.py` — 整文件覆盖
- `subagents/executor.py` — 2 行 surgical edit（noldus 定制 6 refs ≥5 ✅）
- `skill_manage_tool.py` — 整文件覆盖
- `gateway/routers/skills.py` — 整文件覆盖
- `gateway/app.py` — 加 `app.state.config = cfg`
- `gateway/deps.py` — 加 `get_config` 函数

### E.7 测试
- `test_local_skill_storage_write.py` ✨ 新增 162 行
- 10 个测试文件处理：conftest.py, test_client.py, test_client_e2e.py, test_lead_agent_prompt.py, test_lead_agent_skills.py, test_local_sandbox_provider_mounts.py, test_skill_manage_tool.py, test_skills_custom_router.py, test_skills_installer.py, test_skills_loader.py
- `test_ethoinsight_planning_skill.py` — import 路径修复

---

## 2. 当前状态

### 全量测试
```
24 failed, 1834 passed, 14 skipped
```

| 类型 | 数量 | 说明 |
|------|------|------|
| Pre-existing（不应修） | 2 | test_ethoinsight_planning_skill / test_lead_prompt_interactive_pipeline |
| test_client.py（上游版） | 19 | Tier 4 依赖：runtime/checkpointer, artifact API, model API |
| test_client_e2e.py（上游版） | 3 | upload/artifact 差异 |
| **Skill 相关测试** | **75/75** ✅ | 全部通过 |

### 失败根因分析

`test_client.py` 和 `test_client_e2e.py` 使用了**上游版本**。因为 noldus 原版测试 mock 的路径（`deerflow.skills.loader.load_skills`, `deerflow.skills.installer.get_skills_root_path`）在 Phase E 重构后不再有效。

**尝试过但不可行的方案**：
1. 保留 noldus 原版 + 修正 mock 路径 — 需要改 30+ 处 mock target，且语义变化（函数→方法）需 wrapper Mock
2. loader.py compat shim — 已创建，但 client.py 不 import loader，mock 不生效
3. batch 替换 mock target → 引入了 return_value 语义错误

**结论**：使用上游 test_client.py / test_client_e2e.py，接受 22 个 Tier 4 相关 failure。这些失败会在 Phase D（引入 runtime/checkpointer/ + persistence/）后自动消失。

---

## 3. 未完成

- [ ] Phase C.5: Summarization skill rescue (f9ff3a69)
- [ ] Phase D: Tier 4 BC 持久化层 (11 commits)
- [ ] Phase F: 收尾 (handoff + deprecate 4-23 plan)

---

## 4. 给下个会话的建议

### 继续路径
```
cd /home/wangqiuyang/noldus-insight
git stash pop   # 当前改动在 working tree
```

或者直接继续 Phase C.5（无需恢复 stash，改动在 working tree）。

### Phase C.5 要点
- `summarization_config.py` — 加 4 个新字段
- `summarization_middleware.py` — surgical merge, **保留** `BeforeSummarizationHook` / `memory_flush_hook` / archive hook
- `lead_agent/agent.py` — surgical merge, **保留** noldus 中间件链

### Phase D 要点
- 新增 persistence/ + runtime/checkpointer/ + runtime/events/ 三个目录
- `database: { backend: memory }` 默认值
- worker.py — 整文件覆盖（noldus 0 定制）
- D.3 per-user isolation — 大部分 noldus 已吸收，剩余 surgical merge
- D.2 checkpointer mv — 创建 compat shim 在 agents/checkpointer/

### 受保护文件清单（不可整覆盖）
- `agents/lead_agent/prompt.py` (1051 行 noldus diff)
- `subagents/executor.py` (455 行 noldus diff)
- `agents/lead_agent/agent.py` (257 行 noldus diff)
- `agents/middlewares/summarization_middleware.py`
- `sandbox/tools.py` (`{{shared://}}`, `SHARED_PATH_PREFIX`)
- `config/paths.py` (`/mnt/shared`, `shared_dir()`)
- `gateway/routers/threads.py` (614 行 noldus diff)

### 规则重申
- 每 commit 后跑全量测试
- 失败数 > 2 立即停
- ❌ 永远不 push origin
- ❌ 永远不修 2 个 pre-existing failures
- ❌ 永远不整文件覆盖受保护文件

---

## 5. 应急参考

| 卡点 | 对策 |
|------|------|
| ImportError from runtime/checkpointer | Phase D.2 才引入，暂时用 compat shim 或 mock |
| 测试失败数 > 2 | 区分：pre-existing vs Phase D 依赖 vs 真实问题 |
| Surgical merge 丢 noldus 定制 | `git checkout HEAD -- <file>` 恢复重做 |
| make dev 启动失败 | 检查 persistence/ 是否在顶层 import sqlalchemy（应 lazy） |
