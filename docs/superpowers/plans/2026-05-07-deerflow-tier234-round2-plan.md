# DeerFlow 上游 Tier 2/3/4 同步 - 轮 2 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 deerflow 上游 3 个高价值改动合到 noldus dev 分支:E (skill storage 重构)、C.5 (summarization skill rescue)、D (Tier 4 BC 持久化层 11 commit),同时严格保留 noldus 全部定制。

**Architecture:** 纯拉上游同步,不引入 better-auth 不动前端。`database: { backend: memory }` 默认值保持 noldus 行为不变。`agents/checkpointer/` 通过保留兼容 shim 维持旧 import 路径。

**Tech Stack:** Python 3.12 / pytest / SQLAlchemy 2.0 (新引入到 persistence/) / pyyaml / git subtree (deerflow remote)

**前置依赖:**
- 当前 HEAD 应为 `2c0db62b` (轮 1 handoff commit)
- 轮 1 测试基线: `2 failed, 1811 passed, 14 skipped`
- 设计文档: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md`

**关键路径:**
```
NOLDUS=/home/wangqiuyang/noldus-insight
BACKEND=$NOLDUS/packages/agent/backend
HARNESS=$BACKEND/packages/harness/deerflow
GATEWAY=$BACKEND/app/gateway
TESTS=$BACKEND/tests
SKILLS_CUSTOM=$NOLDUS/packages/agent/skills/custom
```

**通用测试命令:**
```bash
cd $BACKEND
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

**已知 pre-existing failures (绝对不要修):**
- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

**Noldus 价值清单 (任何时候都不能丢):**
1. Skill 内容: `packages/agent/skills/custom/` 的 5 个 markdown skill 目录
2. Prompt: `agents/lead_agent/prompt.py` 中文调度规则、Gate 反问、subagent 描述、EV19 模板
3. Subagent 名字: `subagents/builtins/__init__.py` 的 4 个 ethoinsight 子代理
4. 关键 setting:
   - `sandbox/sandbox.py` 的 `extra_env` 参数
   - `sandbox/local/local_sandbox.py` 的 venv PATH + `DEERFLOW_PATH_*`
   - `sandbox/tools.py` 的 `{{shared://}}` 占位符 + `mask_local_paths_in_output`
   - `config/paths.py` 的 `/mnt/shared` + `shared_dir()`
   - `agents/thread_state.py` / `thread_data_middleware.py` 的 `shared_path` 字段
   - `agents/middlewares/llm_error_handling_middleware.py` 的总超时 + circuit breaker
   - `mcp/tools.py` 的 4096 字符截断
   - `subagents/executor.py` 的 `recursion_limit` + `max_turns` + `{{shared://}}`
   - `agents/lead_agent/agent.py` 的中间件链 (ArchivingSummarizationMiddleware、ThinkTagMiddleware、TrainingDataMiddleware、GateEnforcementMiddleware、LoopDetectionMiddleware)

**General rules:**
- ❌ 永远不要 `git show <sha> > <file>` 整文件覆盖 noldus 受保护文件
- ❌ 永远不要 push origin 或 deerflow remote
- ❌ 永远不要修 pre-existing failures
- ✅ 每完成一个阶段都跑全量测试;失败数 > 2 立即停下查根因
- ✅ 每个 sub-task 改完都跑相关单测验证

---

## Phase 0: 基线确认 (Task 0)

### Task 0: 验证起点

**Files:** 无改动,纯验证

- [ ] **Step 1: 验证 git HEAD**

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -1
```

期望输出:
```
2c0db62b docs: 轮 1 deerflow 同步完成交接文档
```

如果 HEAD 不是 `2c0db62b`, **立即停下,问用户**。

- [ ] **Step 2: 验证工作区干净**

```bash
git status
```

期望: `无文件要提交,干净的工作区`。

如果有未提交改动, **立即停下,问用户**。

- [ ] **Step 3: 验证 deerflow remote 存在**

```bash
git remote -v | grep deerflow
```

期望:
```
deerflow	git@github.com:noldus-cn-beijing/deerflow-noldus.git (fetch)
deerflow	git@github.com:noldus-cn-beijing/deerflow-noldus.git (push)
```

如果不存在, **停下并问用户**:`deerflow remote 缺失,请确认是否需要 git remote add`。

- [ ] **Step 4: 验证 deerflow/main 是否最新**

```bash
git fetch deerflow main 2>&1 | tail -3
git log --oneline deerflow/main -1
```

记录上游 head SHA (后续 Task 会用到), 当前应为 `4ead2c6b` 附近 (≤ 2026-05-07)。

- [ ] **Step 5: 跑测试基线**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望 (允许 passed 数 ±5):
```
2 failed, 1811 passed, 14 skipped
```

如果失败数 ≠ 2, **立即停下,问用户**。

- [ ] **Step 6: 记录基线**

把 passed 数记下来, 后续每阶段对比。本 plan 后续假设基线为 `1811 passed`。

---

## Phase E: Skill Storage 重构 (Task E.1 - E.7)

> 上游 commit: `1ad1420e refactor(skills): Unified skill storage capability (#2613)`
>
> **设计**: 删除 `skills/manager.py / installer.py / loader.py`, 引入 `skills/storage/skill_storage.py + local_skill_storage.py`。所有调用方迁到 `get_or_new_skill_storage()`。**保留** `packages/agent/skills/custom/` 5 个 markdown 目录。
>
> **冲击文件 (按 commit 1ad1420e 的 diff stat):**
> ```
> skills/__init__.py            (导出变更)
> skills/parser.py              (8 行)
> skills/security_scanner.py    (3 行)
> skills/types.py               (16 行)
> skills/validation.py          (6 行)
> skills/manager.py             ❌ 删除 (161 行)
> skills/installer.py           ❌ 删除 (86 行)
> skills/loader.py              ❌ 删除 (105 行)
> skills/storage/__init__.py    ✨ 新增 (83 行, 含 get_or_new_skill_storage 工厂)
> skills/storage/skill_storage.py        ✨ 新增 (254 行, ABC)
> skills/storage/local_skill_storage.py  ✨ 新增 (198 行, 实现)
> agents/lead_agent/prompt.py   (1 import + 1 调用)
> client.py                     (18 行)
> config/skills_config.py       (10 行)
> subagents/executor.py         (4 行)
> tools/skill_manage_tool.py    (92 行)
> app/gateway/routers/skills.py (118 行)
> ```
> **测试更新:**
> - `tests/test_local_skill_storage_write.py` ✨ 新增 162 行
> - `tests/conftest.py` (15 行)
> - `tests/test_client.py` (88 行)
> - `tests/test_client_e2e.py` (68 行)
> - `tests/test_lead_agent_prompt.py` (4 行)
> - `tests/test_lead_agent_skills.py` (4 行)
> - `tests/test_local_sandbox_provider_mounts.py` (8 行)
> - `tests/test_skill_manage_tool.py` (15 行)
> - `tests/test_skills_custom_router.py` (60 行)
> - `tests/test_skills_installer.py` (32 行)
> - `tests/test_skills_loader.py` (13 行)
>
> **Noldus 受保护文件:** `agents/lead_agent/prompt.py` 必须 surgical merge (1051 行 noldus diff)。
> **Noldus 不能丢:** `packages/agent/skills/custom/` 全部 markdown 内容。

### Task E.1: 备份 noldus skills 内容

**Files:**
- 不改代码, 只复制文件做备份

- [ ] **Step 1: 备份 5 个 custom skill 目录**

```bash
mkdir -p /tmp/noldus-skills-backup-$(date +%Y%m%d)
cp -r /home/wangqiuyang/noldus-insight/packages/agent/skills/custom \
      /tmp/noldus-skills-backup-$(date +%Y%m%d)/
ls -la /tmp/noldus-skills-backup-$(date +%Y%m%d)/custom/
```

期望输出: 5 个目录 (`compaction-recovery / ethoinsight / ethoinsight-analysis / ethoinsight-charts / ethoinsight-planning`)。

- [ ] **Step 2: 备份 noldus 旧 skills 加载代码**

```bash
mkdir -p /tmp/noldus-skills-code-backup
cp /home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/skills/{manager,installer,loader,__init__,parser,security_scanner,types,validation}.py /tmp/noldus-skills-code-backup/
ls -la /tmp/noldus-skills-code-backup/
```

期望: 8 个 .py 文件存在。

- [ ] **Step 3: 记录 noldus prompt.py 中所有 skill 调用点**

```bash
grep -n "load_skills\|skill_storage\|get_or_new\|skills.loader\|skills.installer\|skills.manager" \
     /home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

预期看到 2 处:
```
14:from deerflow.skills import load_skills
29:    return list(load_skills(enabled_only=True))
```

把这两处记下来,后续 Task E.5 要 surgical edit。

### Task E.2: 拉上游 skills/storage/ 三个新文件

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/skills/storage/__init__.py`
- Create: `packages/agent/backend/packages/harness/deerflow/skills/storage/skill_storage.py`
- Create: `packages/agent/backend/packages/harness/deerflow/skills/storage/local_skill_storage.py`

- [ ] **Step 1: 创建目录**

```bash
cd /home/wangqiuyang/noldus-insight
mkdir -p packages/agent/backend/packages/harness/deerflow/skills/storage
```

- [ ] **Step 2: 拉上游 storage/__init__.py**

```bash
cd /home/wangqiuyang/noldus-insight
git show deerflow/main:backend/packages/harness/deerflow/skills/storage/__init__.py \
    > packages/agent/backend/packages/harness/deerflow/skills/storage/__init__.py
```

- [ ] **Step 3: 拉上游 storage/skill_storage.py**

```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/storage/skill_storage.py \
    > packages/agent/backend/packages/harness/deerflow/skills/storage/skill_storage.py
```

- [ ] **Step 4: 拉上游 storage/local_skill_storage.py**

```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/storage/local_skill_storage.py \
    > packages/agent/backend/packages/harness/deerflow/skills/storage/local_skill_storage.py
```

- [ ] **Step 5: 验证 3 个新文件**

```bash
wc -l packages/agent/backend/packages/harness/deerflow/skills/storage/*.py
```

期望 3 个文件: `__init__.py ~83 行 / local_skill_storage.py ~198 行 / skill_storage.py ~254 行` (允许 ±5 行误差)。

- [ ] **Step 6: 跑 import 检查**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.skills.storage import get_or_new_skill_storage, SkillStorage, LocalSkillStorage
print('OK:', SkillStorage.__name__, LocalSkillStorage.__name__)
"
```

期望: `OK: SkillStorage LocalSkillStorage`。

如果报错 (例如 ImportError), **立即停下** — 通常是 storage 文件 import 了不存在的模块,需要看错误信息检查依赖。

### Task E.3: 拉上游 5 个微改 skills/ 文件

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/skills/parser.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/skills/security_scanner.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/skills/types.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/skills/validation.py`

- [ ] **Step 1: 看 parser.py diff,判断是否可整文件覆盖**

```bash
cd /home/wangqiuyang/noldus-insight
diff <(git show deerflow/main:backend/packages/harness/deerflow/skills/parser.py) \
     packages/agent/backend/packages/harness/deerflow/skills/parser.py
```

判定准则:
- 如果输出 < 50 行 且 不含 noldus 定制痕迹 (中文注释/`mask_local_paths`/`shared_path`/`/mnt/shared`等), **整文件覆盖**:
  ```bash
  git show deerflow/main:backend/packages/harness/deerflow/skills/parser.py \
      > packages/agent/backend/packages/harness/deerflow/skills/parser.py
  ```
- 否则 **停下问用户**: noldus parser.py 含定制,需要 surgical merge。

- [ ] **Step 2: security_scanner.py 同样判断与处理**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/skills/security_scanner.py) \
     packages/agent/backend/packages/harness/deerflow/skills/security_scanner.py
```

如可整覆盖, 则:
```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/security_scanner.py \
    > packages/agent/backend/packages/harness/deerflow/skills/security_scanner.py
```

- [ ] **Step 3: types.py 同样处理**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/skills/types.py) \
     packages/agent/backend/packages/harness/deerflow/skills/types.py
```

如可整覆盖:
```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/types.py \
    > packages/agent/backend/packages/harness/deerflow/skills/types.py
```

- [ ] **Step 4: validation.py 同样处理**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/skills/validation.py) \
     packages/agent/backend/packages/harness/deerflow/skills/validation.py
```

如可整覆盖:
```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/validation.py \
    > packages/agent/backend/packages/harness/deerflow/skills/validation.py
```

- [ ] **Step 5: 替换 skills/__init__.py**

由于上游 `__init__.py` 是 export 接口变更 (移除 load_skills/get_skills_root_path/install_skill_from_archive 的 export, 加 storage 三件), 整文件覆盖即可:

```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/__init__.py \
    > packages/agent/backend/packages/harness/deerflow/skills/__init__.py
```

验证内容:
```bash
cat packages/agent/backend/packages/harness/deerflow/skills/__init__.py
```

期望最后一行附近的 `__all__` 含: `"SkillStorage", "LocalSkillStorage", "get_or_new_skill_storage"` 等。

### Task E.4: 删除旧 manager/installer/loader

**Files:**
- Delete: `packages/agent/backend/packages/harness/deerflow/skills/manager.py`
- Delete: `packages/agent/backend/packages/harness/deerflow/skills/installer.py`
- Delete: `packages/agent/backend/packages/harness/deerflow/skills/loader.py`

⚠️ **不要现在就跑测试** — 调用方还没改, 跑了一定一堆 ImportError。先删完, 改完调用方再测。

- [ ] **Step 1: 确认上游已删除这 3 文件**

```bash
git show deerflow/main:backend/packages/harness/deerflow/skills/manager.py 2>&1 | head -3
git show deerflow/main:backend/packages/harness/deerflow/skills/installer.py 2>&1 | head -3
git show deerflow/main:backend/packages/harness/deerflow/skills/loader.py 2>&1 | head -3
```

期望: 全部输出 `fatal: path '...' does not exist in 'deerflow/main'` 或类似 (上游已删除)。

如果上游某个文件还存在, **停下问用户** — 可能是上游 commit 1ad1420e 之后又被恢复, 不应该删 noldus 这个文件。

- [ ] **Step 2: 删除三个文件**

```bash
cd /home/wangqiuyang/noldus-insight
rm packages/agent/backend/packages/harness/deerflow/skills/manager.py
rm packages/agent/backend/packages/harness/deerflow/skills/installer.py
rm packages/agent/backend/packages/harness/deerflow/skills/loader.py
```

- [ ] **Step 3: 验证 skills/ 目录现状**

```bash
ls packages/agent/backend/packages/harness/deerflow/skills/
```

期望: `__init__.py / parser.py / security_scanner.py / types.py / validation.py / storage/`。

不应该再有 manager.py / installer.py / loader.py。

### Task E.5: 改调用方 - prompt.py (surgical merge)

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:14`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:29`

⚠️ **prompt.py 是高风险受保护文件 (1051 行 noldus diff)**。**绝对不要整文件覆盖**。只改 2 处。

- [ ] **Step 1: 看现状**

```bash
cd /home/wangqiuyang/noldus-insight
sed -n '12,16p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
echo "---"
sed -n '27,32p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

第 14 行附近期望 (noldus 现状):
```python
from deerflow.skills import load_skills
```

第 29 行附近期望:
```python
    return list(load_skills(enabled_only=True))
```

- [ ] **Step 2: 改 import (Edit tool)**

用 Edit 工具:
- file_path: `/home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- old_string: `from deerflow.skills import load_skills`
- new_string: `from deerflow.skills.storage import get_or_new_skill_storage`

- [ ] **Step 3: 改调用 (Edit tool)**

用 Edit 工具:
- file_path: 同上
- old_string: `    return list(load_skills(enabled_only=True))`
- new_string: `    return list(get_or_new_skill_storage().load_skills(enabled_only=True))`

- [ ] **Step 4: 验证 surgical edit**

```bash
grep -n "load_skills\|skill_storage\|get_or_new" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

期望:
```
14:from deerflow.skills.storage import get_or_new_skill_storage
29:    return list(get_or_new_skill_storage().load_skills(enabled_only=True))
```

不应该再有 `from deerflow.skills import load_skills`。

- [ ] **Step 5: 验证 noldus 中文规则仍在**

```bash
grep -c "中文\|按以下\|EV19\|Gate" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

期望: 数字 > 5 (noldus 大量中文 prompt)。

如果数字 = 0, **立即停下,git restore prompt.py** — 你不小心覆盖了 noldus 定制。

### Task E.6: 改其他调用方

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/client.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/config/skills_config.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py`
- Modify: `packages/agent/backend/app/gateway/routers/skills.py`

⚠️ `subagents/executor.py` 是 noldus **重定制文件 (455 行 diff)**, 必须 surgical edit, 只改 import 和 1 处调用。

- [ ] **Step 1: 看上游 client.py 改动**

```bash
cd /home/wangqiuyang/noldus-insight
git show 1ad1420e -- 'backend/packages/harness/deerflow/client.py'
```

记录上游的具体 import 和调用变更。预期上游把 `from deerflow.skills.loader import load_skills` 改成 `from deerflow.skills.storage import get_or_new_skill_storage` 等。

- [ ] **Step 2: noldus client.py 应用同等改动**

逐处用 Edit 工具修改 client.py 中所有 `from deerflow.skills.loader import load_skills` 为 `from deerflow.skills.storage import get_or_new_skill_storage`, 调用从 `load_skills(...)` 改为 `get_or_new_skill_storage().load_skills(...)`。

每处改完, 跑 `grep -n "load_skills\|skill_storage" packages/agent/backend/packages/harness/deerflow/client.py | head -10` 验证。

- [ ] **Step 3: skills_config.py 同样改 (10 行)**

```bash
git show 1ad1420e -- 'backend/packages/harness/deerflow/config/skills_config.py'
```

按上游 diff 应用。预期改 `from deerflow.skills.loader import get_skills_root_path` 为新路径。

- [ ] **Step 4: subagents/executor.py surgical edit**

⚠️ **不要整文件覆盖**。只改 4 行。

```bash
git show 1ad1420e -- 'backend/packages/harness/deerflow/subagents/executor.py'
```

预期上游改动是把:
```python
    from deerflow.skills.loader import load_skills

    all_skills = load_skills(enabled_only=True)
```

改成:
```python
    from deerflow.skills.storage import get_or_new_skill_storage

    all_skills = get_or_new_skill_storage().load_skills(enabled_only=True)
```

用 Edit 工具应用这个 surgical edit。改完跑:
```bash
grep -c "recursion_limit\|max_turns\|shared://" packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

期望: 数字 ≥ 5 (noldus 定制全在)。如果数字 = 0, **立即停下,git restore executor.py**。

- [ ] **Step 5: tools/skill_manage_tool.py 整文件升级 (92 行 diff,但 noldus 端可能未深度定制)**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/tools/skill_manage_tool.py) \
     packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py | head -30
grep -c "中文\|EthoInsight" packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py
```

如果 noldus 端无中文/EthoInsight 痕迹 (数字=0) 且 diff < 200 行, **整文件覆盖**:
```bash
git show deerflow/main:backend/packages/harness/deerflow/tools/skill_manage_tool.py \
    > packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py
```
否则 **停下问用户**。

- [ ] **Step 6: app/gateway/routers/skills.py 处理 (118 行 diff)**

```bash
diff <(git show deerflow/main:backend/app/gateway/routers/skills.py) \
     packages/agent/backend/app/gateway/routers/skills.py | head -30
grep -c "中文\|EthoInsight" packages/agent/backend/app/gateway/routers/skills.py
```

判断同上。如可整覆盖:
```bash
git show deerflow/main:backend/app/gateway/routers/skills.py \
    > packages/agent/backend/app/gateway/routers/skills.py
```

- [ ] **Step 7: 全局 import 检查**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
import importlib
for m in ['deerflow.skills', 'deerflow.skills.storage', 'deerflow.client',
         'deerflow.config.skills_config', 'deerflow.subagents.executor',
         'deerflow.tools.skill_manage_tool', 'deerflow.agents.lead_agent.prompt',
         'app.gateway.routers.skills']:
    importlib.import_module(m)
    print('OK:', m)
"
```

期望: 8 个 OK 行。

任何 ImportError, **立即停下** — 看错误信息找漏改的调用方。

### Task E.7: 更新测试 + 全量回归

**Files:**
- Modify: `packages/agent/backend/tests/conftest.py`
- Modify: `packages/agent/backend/tests/test_client.py`
- Modify: `packages/agent/backend/tests/test_client_e2e.py`
- Modify: `packages/agent/backend/tests/test_lead_agent_prompt.py`
- Modify: `packages/agent/backend/tests/test_lead_agent_skills.py`
- Modify: `packages/agent/backend/tests/test_local_sandbox_provider_mounts.py`
- Modify: `packages/agent/backend/tests/test_skill_manage_tool.py`
- Modify: `packages/agent/backend/tests/test_skills_custom_router.py`
- Modify: `packages/agent/backend/tests/test_skills_installer.py`
- Modify: `packages/agent/backend/tests/test_skills_loader.py`
- Create: `packages/agent/backend/tests/test_local_skill_storage_write.py`

⚠️ noldus 测试文件可能含中文 / EthoInsight specific 内容。**先看 diff** 判断每个文件能否整覆盖。

- [ ] **Step 1: 拉新增的 test_local_skill_storage_write.py**

```bash
cd /home/wangqiuyang/noldus-insight
git show deerflow/main:backend/tests/test_local_skill_storage_write.py \
    > packages/agent/backend/tests/test_local_skill_storage_write.py
wc -l packages/agent/backend/tests/test_local_skill_storage_write.py
```

期望: 162 行附近。

- [ ] **Step 2: 处理每个测试文件 (循环)**

对每个测试文件按以下流程:

```bash
TEST=test_skills_installer.py
diff <(git show deerflow/main:backend/tests/$TEST) packages/agent/backend/tests/$TEST | head -20
grep -c "中文\|EthoInsight\|ethoinsight" packages/agent/backend/tests/$TEST
```

判定准则:
- 数字 = 0 且 diff < 100 行 → 整覆盖:
  ```bash
  git show deerflow/main:backend/tests/$TEST > packages/agent/backend/tests/$TEST
  ```
- 数字 > 0 或 diff > 100 行 → 用 Edit tool 只改 import / mock path 部分 (把 `deerflow.skills.loader.load_skills` 改成 `deerflow.skills.storage.get_or_new_skill_storage` 等)

按这个流程处理 10 个测试文件 (上面 Files 里列的)。

- [ ] **Step 3: 跑只 skill 相关测试**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_skills_installer.py tests/test_skills_loader.py \
    tests/test_skills_custom_router.py tests/test_local_skill_storage_write.py \
    tests/test_skill_manage_tool.py tests/test_lead_agent_skills.py \
    -v 2>&1 | tail -20
```

期望: 全过 (允许 skipped)。

如果有 fail, 看错误信息修对应测试 (通常是 mock path 还是旧的 `deerflow.skills.loader.xxx`, 改成 `deerflow.skills.storage.xxx`)。

- [ ] **Step 4: 跑全量测试**

```bash
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, X passed, 14 skipped`,X 应该 ≥ 1811 (上游 1ad1420e 加了 162 行测试,加上微调,新增约 100-160 个测试)。

如果失败数 > 2, **立即停下** — 看具体哪些测试失败,可能是 import path 漏改或者 mock 路径需要更新。

- [ ] **Step 5: lint 检查**

```bash
PYTHONPATH=. uv run ruff check packages/harness/deerflow/skills/ app/gateway/routers/skills.py 2>&1 | tail -5
```

期望: `All checks passed!` 或 0 error。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/skills/ \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
        packages/agent/backend/packages/harness/deerflow/client.py \
        packages/agent/backend/packages/harness/deerflow/config/skills_config.py \
        packages/agent/backend/packages/harness/deerflow/subagents/executor.py \
        packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py \
        packages/agent/backend/app/gateway/routers/skills.py \
        packages/agent/backend/tests/

git commit -m "$(cat <<'EOF'
sync deerflow upstream Tier 4 E: skill storage 重构 (1ad1420e)

- 新增 skills/storage/{__init__,skill_storage,local_skill_storage}.py
- 删除 skills/{manager,installer,loader}.py (上游已删)
- 更新 14 处调用方使用 get_or_new_skill_storage()
- 微改 skills/{__init__,parser,security_scanner,types,validation}.py
- 新增 test_local_skill_storage_write.py (162 行)
- 更新 10 个相关测试文件的 mock 路径

保留 Noldus 全部定制:
- packages/agent/skills/custom/ 5 个 markdown skill 目录原样
- agents/lead_agent/prompt.py 中文调度 + Gate 反问 + EV19 模板路径
- subagents/executor.py recursion_limit + max_turns + {{shared://}}

详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md §2.1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: 验证 commit**

```bash
git log --oneline -1
git show --stat HEAD | head -3
```

期望: HEAD 是 "sync deerflow upstream Tier 4 E: skill storage 重构 (1ad1420e)"。

---

## Phase C.5: Summarization Skill Rescue (Task C.5.1 - C.5.3)

> 上游 commit: `f9ff3a69 fix(middleware): avoid rescuing non-skill tool outputs during summarization (#2458)`
>
> **冲击文件 (按 commit 的 diff stat):**
> ```
> backend/docs/summarization.md  (28 行新增)
> agents/lead_agent/agent.py     (19 行,工厂函数注入 skills_container_path)
> agents/middlewares/summarization_middleware.py  (206 行,核心改动)
> config/summarization_config.py (19 行,4 个新 config 项)
> tests/test_lead_agent_model_resolution.py       (24 行新增)
> tests/test_summarization_middleware.py           (327 行新增)
> config.example.yaml             (15 行)
> ```
> **noldus 受保护文件:**
> - `agents/lead_agent/agent.py` (257 行 noldus diff, **必须 surgical merge**)
> - `agents/middlewares/summarization_middleware.py` (含 Noldus `BeforeSummarizationHook` 协议 + memory_flush_hook)

### Task C.5.1: 拉 summarization_config.py 4 个新 config 项

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/config/summarization_config.py`

- [ ] **Step 1: 看 diff**

```bash
cd /home/wangqiuyang/noldus-insight
diff <(git show deerflow/main:backend/packages/harness/deerflow/config/summarization_config.py) \
     packages/agent/backend/packages/harness/deerflow/config/summarization_config.py
```

- [ ] **Step 2: 提取上游新增的 4 个字段**

```bash
git show f9ff3a69 -- 'backend/packages/harness/deerflow/config/summarization_config.py'
```

预期看到 4 个新增 Pydantic 字段:
```python
preserve_recent_skill_count: int = ...
preserve_recent_skill_tokens: int = ...
preserve_recent_skill_tokens_per_skill: int = ...
skill_file_read_tool_names: list[str] = ...
```

- [ ] **Step 3: 用 Edit tool 在 noldus summarization_config.py 末尾的 class 中追加这 4 个字段**

具体步骤:
- 用 Read tool 读 noldus `config/summarization_config.py` 找到 SummarizationConfig 类的 fields 定义结尾位置
- 用 Edit tool 在合适位置插入这 4 个新字段 (从 step 2 复制具体定义)

- [ ] **Step 4: 验证导入**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.config.summarization_config import SummarizationConfig
c = SummarizationConfig()
print('preserve_recent_skill_count:', c.preserve_recent_skill_count)
print('skill_file_read_tool_names:', c.skill_file_read_tool_names)
"
```

期望打印 4 个字段值, 全是默认值。

### Task C.5.2: surgical merge summarization_middleware.py

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py`

⚠️ noldus 这个 middleware 含 `BeforeSummarizationHook` 协议 + `memory_flush_hook` + archive hook。**绝不能整覆盖**。

- [ ] **Step 1: 看上游改动**

```bash
git show f9ff3a69 -- 'backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py' > /tmp/c5_summarization_patch.diff
wc -l /tmp/c5_summarization_patch.diff
head -60 /tmp/c5_summarization_patch.diff
```

预期看到上游新增了一个或多个 helper 函数 (例如 `_extract_skill_bundles`, `_lift_skill_messages`, `_is_skill_read_tool` 等)。

- [ ] **Step 2: 看 noldus 现有结构**

```bash
grep -n "class\|^def \|^async def \|BeforeSummarizationHook" \
     packages/agent/backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py | head -20
```

记录 noldus 现有的 class / function 结构, 知道在哪里插入新 helper 函数。

- [ ] **Step 3: 提取上游新增 helper 函数**

```bash
# 从 patch 中提取所有 +def / +async def 行附近的代码块
grep -A 20 "^+def \|^+async def \|^+class" /tmp/c5_summarization_patch.diff
```

记录这些新函数的完整定义。

- [ ] **Step 4: 用 Edit tool 把上游新增的 helper 函数插入到 noldus middleware**

把每个新 helper 函数添加到 noldus 文件中合适的位置 (通常在 middleware 类定义前的 module-level helpers 区)。

- [ ] **Step 5: 用 Edit tool 在 noldus middleware 主流程中加入 skill rescue 调用**

按上游 patch 中的 `+` 行, 在 noldus `summarize` 或 `before_summarization` 主方法的合适位置插入 skill rescue 逻辑 (lift skill bundles before truncation)。

⚠️ **保留** noldus 现有的 `before_summarization` hook 调用、`memory_flush_hook` 调用、archive hook 调用。**新加的 skill rescue 应该在 hook 调用之后、truncation 之前**。

- [ ] **Step 6: 验证 noldus 定制仍在**

```bash
grep -c "BeforeSummarizationHook\|memory_flush_hook\|archive" \
     packages/agent/backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py
```

期望: ≥ 3。如果数字 < 3, **立即停下,git restore**。

- [ ] **Step 7: import 检查**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware, BeforeSummarizationHook
print('OK')
"
```

期望: `OK`。

### Task C.5.3: surgical merge lead_agent/agent.py + 拉测试 + 全量回归

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- Create: `packages/agent/backend/tests/test_summarization_middleware.py` (整覆盖, 327 行)
- Create or Modify: `packages/agent/backend/tests/test_lead_agent_model_resolution.py`

⚠️ `agents/lead_agent/agent.py` 是 noldus **重定制文件 (257 行 diff)**, 含中间件链。**绝不能整覆盖**。

- [ ] **Step 1: 看上游 lead_agent/agent.py 改动**

```bash
git show f9ff3a69 -- 'backend/packages/harness/deerflow/agents/lead_agent/agent.py'
```

预期上游改动很小 (~19 行): 在工厂函数中给 `DeerFlowSummarizationMiddleware()` 调用传入 `skills_container_path` 参数。

- [ ] **Step 2: 找 noldus agent.py 中创建 SummarizationMiddleware 的位置**

```bash
grep -n "DeerFlowSummarizationMiddleware\|ArchivingSummarizationMiddleware\|SummarizationMiddleware" \
     packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
```

记录 line number。

- [ ] **Step 3: 用 Edit tool 加 skills_container_path 参数**

按上游 diff 在 noldus 的 SummarizationMiddleware 实例化代码中加新 kwarg。**保留** noldus 中所有其他参数和 ArchivingSummarizationMiddleware (如有)。

- [ ] **Step 4: 验证 noldus 中间件链全在**

```bash
grep -c "ArchivingSummarizationMiddleware\|ThinkTagMiddleware\|TrainingDataMiddleware\|GateEnforcementMiddleware\|LoopDetectionMiddleware" \
     packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
```

期望 ≥ 5。如果 < 5, **立即停下 git restore**。

- [ ] **Step 5: 拉新测试 test_summarization_middleware.py**

```bash
diff <(git show deerflow/main:backend/tests/test_summarization_middleware.py) \
     packages/agent/backend/tests/test_summarization_middleware.py 2>&1 | head -10
grep -c "中文\|EthoInsight\|ethoinsight" packages/agent/backend/tests/test_summarization_middleware.py 2>/dev/null
```

如果 noldus 端不存在或无定制, 整覆盖:
```bash
git show deerflow/main:backend/tests/test_summarization_middleware.py \
    > packages/agent/backend/tests/test_summarization_middleware.py
```
否则 surgical merge (查上游 patch 找新增 test cases)。

- [ ] **Step 6: 处理 test_lead_agent_model_resolution.py**

```bash
diff <(git show deerflow/main:backend/tests/test_lead_agent_model_resolution.py 2>/dev/null) \
     packages/agent/backend/tests/test_lead_agent_model_resolution.py 2>/dev/null | head -10
```

按相同判定流程整覆盖或 surgical merge。

- [ ] **Step 7: 跑相关单测**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_summarization_middleware.py tests/test_lead_agent_model_resolution.py -v 2>&1 | tail -10
```

期望全过。

- [ ] **Step 8: 跑全量测试**

```bash
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望 `2 failed, X passed, 14 skipped`, X 比 E 阶段后增加 ~30+ (上游 327 行新测试 + 24 行 model resolution)。

- [ ] **Step 9: lint**

```bash
PYTHONPATH=. uv run ruff check packages/harness/deerflow/agents/middlewares/summarization_middleware.py \
    packages/harness/deerflow/agents/lead_agent/agent.py \
    packages/harness/deerflow/config/summarization_config.py 2>&1 | tail -3
```

期望 0 error。

- [ ] **Step 10: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/config/summarization_config.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
        packages/agent/backend/tests/test_summarization_middleware.py \
        packages/agent/backend/tests/test_lead_agent_model_resolution.py

git commit -m "$(cat <<'EOF'
sync deerflow upstream Tier 3 C.5: summarization skill rescue (f9ff3a69)

- summarization_config.py 加 4 个新 config 项 (preserve_recent_skill_count
  / preserve_recent_skill_tokens / preserve_recent_skill_tokens_per_skill
  / skill_file_read_tool_names)
- summarization_middleware.py 加 skill rescue 逻辑 (lift skill bundles 在
  truncation 前)
- lead_agent/agent.py 工厂注入 skills_container_path
- 新增 test_summarization_middleware.py (327 行) + test_lead_agent_model_resolution.py

保留 Noldus 全部定制:
- BeforeSummarizationHook 协议 + memory_flush_hook + archive hook
- lead_agent/agent.py 完整中间件链 (Archiving/ThinkTag/TrainingData/
  GateEnforcement/LoopDetection)

依赖 Phase E (skills/storage 已就位)
详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md §2.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 11: 验证 commit**

```bash
git log --oneline -2
```

期望:
```
<sha> sync deerflow upstream Tier 3 C.5: summarization skill rescue (f9ff3a69)
<sha> sync deerflow upstream Tier 4 E: skill storage 重构 (1ad1420e)
```

---

## Phase D: Tier 4 BC 持久化层 (Task D.1 - D.10)

> 上游 11 个 commit (按依赖顺序):
> | # | SHA | 作用 | 风险 |
> |---|---|---|---|
> | D.1 | `d8ecaf46` | persistence scaffold + RunEventStore + ORM models + RunJournal (#1930) | 低 |
> | D.2 | `56d5fa33` | unified persistence rebase cleanup + checkpointer mv (#2134) | 中 (mv) |
> | D.3 | `2e05f380` | per-user filesystem isolation (user_id 默认 None,BC) (#2153) | 中 |
> | D.4 | `35ef8b7c` | 默认 database config (19 行) | 低 |
> | D.5 | `16aedf45 + 897dae54 + 829e82a9` | lint 修复 | 低 |
> | D.6 | `898f4e8a` | memory cache corruption 修复 (#2251) | 中 |
> | D.7 | `87609374` | memory I/O 用 asyncio.to_thread (#2220) | 中 |
> | D.8 | `35f141fc` | checkpoint rollback on cancel (#1867) | 中 (worker.py 整覆盖) |
> | D.9 | `ca3332f8` | gateway ISO 8601 timestamps (#2599) | 中 |
> | D.10 | `17447fcc` | rollback restore checkpoint supersede newer (#2582) | 低 |
>
> **新增目录:**
> ```
> packages/agent/backend/packages/harness/deerflow/persistence/
> ├── __init__.py base.py engine.py
> ├── feedback/{__init__,model,sql}.py
> ├── migrations/{alembic.ini,env.py,versions/.gitkeep}
> ├── models/{__init__,run_event}.py
> ├── run/{__init__,model,sql}.py
> ├── thread_meta/{__init__,base,memory,model,sql}.py
> └── user/{__init__,model}.py
>
> packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/
> └── {__init__,async_provider,provider}.py  (从 agents/checkpointer/ mv 来)
>
> packages/agent/backend/packages/harness/deerflow/runtime/events/
> └── store/{__init__,base,db,jsonl,memory}.py
>
> packages/agent/backend/packages/harness/deerflow/runtime/runs/store/
> └── {__init__,base,memory}.py  (新增 RunStore)
>
> packages/agent/backend/packages/harness/deerflow/config/
> ├── database_config.py        (新增)
> └── run_events_config.py      (新增)
>
> packages/agent/backend/packages/harness/deerflow/utils/
> └── time.py                    (新增 ISO 8601 helpers)
>
> packages/agent/backend/packages/harness/deerflow/runtime/
> └── journal.py                 (新增 RunJournal)
> ```

⚠️ **D 阶段所有改动都按"默认 memory backend"配置, noldus 现有行为保持完全不变。如发现任何 noldus 测试因 D 改动失败,立即停下查根因。**

### Task D.1: 拉 persistence/ 目录全部新文件 (d8ecaf46)

**Files (全是 Create):**
- 21 个 persistence/ 下的新文件 (见上面树形)
- `config/database_config.py`
- `runtime/events/store/{base,db,jsonl,memory}.py` + `__init__.py`
- `runtime/journal.py`
- `runtime/runs/store/{base,memory}.py` + `__init__.py`
- `config/run_events_config.py`
- `utils/time.py`

- [ ] **Step 1: 创建目录骨架**

```bash
cd /home/wangqiuyang/noldus-insight
HARNESS=packages/agent/backend/packages/harness/deerflow

mkdir -p $HARNESS/persistence/{feedback,migrations/versions,models,run,thread_meta,user}
mkdir -p $HARNESS/runtime/events/store
mkdir -p $HARNESS/runtime/runs/store
mkdir -p $HARNESS/utils
ls $HARNESS/persistence/
```

期望 6 个子目录创建。

- [ ] **Step 2: 拉 persistence/ 全部 21 个文件**

⚠️ 上游 persistence/ 全是新文件, noldus 端不存在, 直接 git show 即可。

```bash
cd /home/wangqiuyang/noldus-insight
HARNESS=packages/agent/backend/packages/harness/deerflow

for f in __init__.py base.py engine.py \
         feedback/__init__.py feedback/model.py feedback/sql.py \
         migrations/alembic.ini migrations/env.py migrations/versions/.gitkeep \
         models/__init__.py models/run_event.py \
         run/__init__.py run/model.py run/sql.py \
         thread_meta/__init__.py thread_meta/base.py thread_meta/memory.py thread_meta/model.py thread_meta/sql.py \
         user/__init__.py user/model.py; do
    git show "deerflow/main:backend/packages/harness/deerflow/persistence/$f" \
        > "$HARNESS/persistence/$f"
done

ls -R $HARNESS/persistence/ | head -30
```

- [ ] **Step 3: 拉 config/database_config.py 和 config/run_events_config.py**

```bash
git show deerflow/main:backend/packages/harness/deerflow/config/database_config.py > $HARNESS/config/database_config.py
git show deerflow/main:backend/packages/harness/deerflow/config/run_events_config.py > $HARNESS/config/run_events_config.py
```

- [ ] **Step 4: 拉 runtime/events/ 全部**

```bash
for f in __init__.py store/__init__.py store/base.py store/db.py store/jsonl.py store/memory.py; do
    git show "deerflow/main:backend/packages/harness/deerflow/runtime/events/$f" \
        > "$HARNESS/runtime/events/$f"
done
ls -R $HARNESS/runtime/events/
```

- [ ] **Step 5: 拉 runtime/runs/store/**

```bash
for f in __init__.py base.py memory.py; do
    git show "deerflow/main:backend/packages/harness/deerflow/runtime/runs/store/$f" \
        > "$HARNESS/runtime/runs/store/$f"
done
ls $HARNESS/runtime/runs/store/
```

- [ ] **Step 6: 拉 runtime/journal.py 和 utils/time.py**

```bash
git show deerflow/main:backend/packages/harness/deerflow/runtime/journal.py > $HARNESS/runtime/journal.py
git show deerflow/main:backend/packages/harness/deerflow/utils/time.py > $HARNESS/utils/time.py
ls $HARNESS/runtime/journal.py $HARNESS/utils/time.py
```

- [ ] **Step 7: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.persistence import init_engine, close_engine, get_session_factory
from deerflow.config.database_config import DatabaseConfig
from deerflow.config.run_events_config import RunEventsConfig
from deerflow.runtime.events.store import RunEventStore
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.journal import RunJournal
from deerflow.utils.time import iso8601_now
print('OK')
"
```

如果 ImportError, 看错误信息 — 通常是某个文件 import 了缺失的 sibling, 例如 `from deerflow.persistence.user.model import UserRow` 之类。如果是 user/ 子目录引用的 sibling 报错, 检查是否有遗漏文件没拉。

期望: `OK`。

⚠️ **如果某个 import 引用了 SQLAlchemy / alembic / asyncpg 但环境里没装, 跳过这个 Step**, 继续 Step 8。**不要立即装包**, 应该先看 D.4 拉 pyproject.toml deps 后再回来。

- [ ] **Step 8: 跑现有测试,确保新增模块没破坏**

```bash
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, X passed, 14 skipped`, X 不变 (新模块还没启用,只是文件存在)。

如果 X 减少 (有测试新失败), **立即停下** — 通常是 noldus 现有代码 import 了 `deerflow.runtime.events` 之类 ambiguous path,被新模块影响。

### Task D.2: checkpointer 目录 mv + 兼容 shim (56d5fa33)

> 上游 56d5fa33 把 `agents/checkpointer/` mv 到 `runtime/checkpointer/`。
> noldus 还有调用 `from deerflow.agents.checkpointer.async_provider import make_checkpointer` 等 (例如 `app/gateway/deps.py`、`langgraph.json`、`tests/test_checkpointer*.py`)。
>
> **策略**: 拉新 `runtime/checkpointer/` 目录, **保留** `agents/checkpointer/` 作为 re-export shim, 确保旧 import 路径不破坏。

**Files:**
- Create: `packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/__init__.py`
- Create: `packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/async_provider.py`
- Create: `packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/checkpointer/__init__.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/checkpointer/async_provider.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/checkpointer/provider.py`

- [ ] **Step 1: 创建 runtime/checkpointer/ 目录**

```bash
cd /home/wangqiuyang/noldus-insight
HARNESS=packages/agent/backend/packages/harness/deerflow
mkdir -p $HARNESS/runtime/checkpointer
```

- [ ] **Step 2: 拉上游新版 (在 runtime/checkpointer/)**

```bash
for f in __init__.py async_provider.py provider.py; do
    git show "deerflow/main:backend/packages/harness/deerflow/runtime/checkpointer/$f" \
        > "$HARNESS/runtime/checkpointer/$f"
done
```

- [ ] **Step 3: 把 agents/checkpointer/ 改为 re-export shim**

⚠️ **不要删 agents/checkpointer/**, noldus 还在用旧路径。改成纯 re-export 即可。

把 `$HARNESS/agents/checkpointer/__init__.py` 改成:

```python
"""Compatibility shim: re-export from new location runtime.checkpointer.

Upstream 56d5fa33 moved checkpointer to runtime/. We keep this shim
so noldus callers using `from deerflow.agents.checkpointer import ...`
continue working without code changes.
"""
from deerflow.runtime.checkpointer import (
    checkpointer_context,
    get_checkpointer,
    make_checkpointer,
    reset_checkpointer,
)

__all__ = ["checkpointer_context", "get_checkpointer", "make_checkpointer", "reset_checkpointer"]
```

- [ ] **Step 4: 把 agents/checkpointer/async_provider.py 改成 shim**

```python
"""Compatibility shim — re-export from runtime.checkpointer.async_provider."""
from deerflow.runtime.checkpointer.async_provider import *  # noqa: F401,F403
from deerflow.runtime.checkpointer.async_provider import make_checkpointer  # noqa: F401
```

- [ ] **Step 5: 把 agents/checkpointer/provider.py 改成 shim**

```python
"""Compatibility shim — re-export from runtime.checkpointer.provider."""
from deerflow.runtime.checkpointer.provider import *  # noqa: F401,F403
from deerflow.runtime.checkpointer.provider import (  # noqa: F401
    checkpointer_context,
    get_checkpointer,
    reset_checkpointer,
)
```

- [ ] **Step 6: smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.agents.checkpointer.async_provider import make_checkpointer
from deerflow.agents.checkpointer.provider import get_checkpointer, checkpointer_context
from deerflow.agents.checkpointer import reset_checkpointer
from deerflow.runtime.checkpointer.async_provider import make_checkpointer as mk2
print('OK', mk2 is make_checkpointer)
"
```

期望: `OK True` (shim 应该 re-export 同一个对象)。

- [ ] **Step 7: 跑 checkpointer 测试**

```bash
PYTHONPATH=. uv run pytest tests/test_checkpointer.py tests/test_checkpointer_none_fix.py -v 2>&1 | tail -10
```

期望: 全过。

如果 fail, 通常是 mock path 还在引用旧的 `deerflow.agents.checkpointer.xxx` 但 shim 让 module identity 改变。**临时方案**: 把 shim 改成 `import + 直接赋值` 而非 `from ... import *`。**问用户决定是否接受**。

### Task D.3: per-user filesystem isolation 处理 (2e05f380)

> 这个 commit 是 D 阶段最复杂的,涉及 noldus 多个受保护文件。
>
> **noldus 现状: storage.py / queue.py / user_context.py diff = 0 (已吸收)**, 所以这部分上游与 noldus 等同。
> 但 commit 还涉及 `agents/middlewares/memory_middleware.py`、`thread_data_middleware.py`、`uploads_middleware.py`、`config/paths.py`、`runtime/runs/worker.py` 等。
>
> **策略**: 该 commit 大部分改动 noldus 已吸收。剩余的 noldus 受保护文件 surgical edit。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/events/store/{base,db,jsonl,memory}.py` (12 行 each, user_id 加入 schema)
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py` (6 行)
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/thread_data_middleware.py` (23 行)
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py` (3 行)
- Modify: `packages/agent/backend/packages/harness/deerflow/config/paths.py` (98 行 — noldus 重定制 + `/mnt/shared`)
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py` (2 行)
- Modify: `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` (3 行)
- Modify: `packages/agent/backend/packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py` (17 行)
- Modify: `packages/agent/backend/packages/harness/deerflow/uploads/manager.py` (3 行)
- Modify: 多个 gateway routers (memory.py / runs.py / threads.py / thread_runs.py / uploads.py)
- Create: `packages/agent/backend/scripts/migrate_user_isolation.py` (160 行 - 数据迁移脚本)
- Modify: 多个测试文件

⚠️ **超过一半的文件 noldus 端有定制, 必须 surgical merge。逐个文件做。**

- [ ] **Step 1: 看 commit 整体范围**

```bash
git show 2e05f380 --stat | head -50
```

记录所有修改的文件。

- [ ] **Step 2: 处理 runtime/events/store/ (5 个文件,纯新文件,可整覆盖)**

```bash
cd /home/wangqiuyang/noldus-insight
git show 2e05f380 -- 'backend/packages/harness/deerflow/runtime/events/store/' | head -100
```

由于 events/store/ 是 D.1 刚拉的新目录, 上游 2e05f380 对它的修改应该 **已经在 D.1 拉的版本里** (D.1 用 deerflow/main, 已含此 commit 改动)。**确认即可,不需再改**。

```bash
HARNESS=packages/agent/backend/packages/harness/deerflow
diff <(git show deerflow/main:backend/packages/harness/deerflow/runtime/events/store/base.py) \
     $HARNESS/runtime/events/store/base.py
```

期望: 0 行 diff。如果非 0, **立即停下** — D.1 拉错了。

- [ ] **Step 3: 处理 memory_middleware.py (surgical, 6 行)**

```bash
git show 2e05f380 -- 'backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py'
```

提取上游的 6 行改动 (通常是加个 user_id 参数读取),用 Edit tool 在 noldus 文件中应用。

验证 noldus 定制仍在:
```bash
grep -c "user:\|agent_name" $HARNESS/agents/middlewares/memory_middleware.py
```

期望: ≥ 1。

- [ ] **Step 4: 处理 thread_data_middleware.py (23 行,noldus `shared_path` 受保护)**

```bash
git show 2e05f380 -- 'backend/packages/harness/deerflow/agents/middlewares/thread_data_middleware.py'
grep -c "shared_path\|/mnt/shared" $HARNESS/agents/middlewares/thread_data_middleware.py
```

surgical edit, **保留 noldus shared_path 字段**。

- [ ] **Step 5: 处理 uploads_middleware.py (3 行)**

```bash
git show 2e05f380 -- 'backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py'
```

3 行小改, 看上游 patch 直接 Edit 应用。

- [ ] **Step 6: 处理 config/paths.py (98 行 — 极高风险)**

```bash
git show 2e05f380 -- 'backend/packages/harness/deerflow/config/paths.py' | head -80
diff <(git show deerflow/main:backend/packages/harness/deerflow/config/paths.py) \
     $HARNESS/config/paths.py | wc -l
grep -c "/mnt/shared\|shared_dir" $HARNESS/config/paths.py
```

判定:
- 如果 noldus diff 行数 < 200 且上游 2e05f380 改动是函数级追加 (不删 noldus 的 `shared_dir()` 等), 可 surgical merge。
- 否则 **停下问用户** — paths.py 是 noldus 关键定制,需要详细讨论。

- [ ] **Step 7: 处理 runtime/runs/worker.py (2 行)**

```bash
git show 2e05f380 -- 'backend/packages/harness/deerflow/runtime/runs/worker.py'
```

2 行加 user_id 处理。直接 Edit 应用。

⚠️ noldus worker.py **0 处定制 (前面调研确认)**, 但还是别整覆盖, 因为后续 D.8 会有大改。

- [ ] **Step 8: 处理 sandbox/tools.py (3 行) 和 uploads/manager.py (3 行)**

⚠️ sandbox/tools.py 是 noldus **重度定制** (`{{shared://}}`、`SHARED_PATH_PREFIX`、`mask_local_paths_in_output`)。**只能 Edit 加几行, 不能整覆盖**。

```bash
git show 2e05f380 -- 'backend/packages/harness/deerflow/sandbox/tools.py'
git show 2e05f380 -- 'backend/packages/harness/deerflow/uploads/manager.py'
```

按上游 patch 逐行 Edit。

每个文件改完跑:
```bash
grep -c "{{shared://}}\|SHARED_PATH_PREFIX\|mask_local_paths_in_output" $HARNESS/sandbox/tools.py
```

期望 ≥ 3。

- [ ] **Step 9: 处理 community/aio_sandbox/aio_sandbox_provider.py (17 行)**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py) \
     $HARNESS/community/aio_sandbox/aio_sandbox_provider.py | head -10
grep -c "noldus\|EthoInsight" $HARNESS/community/aio_sandbox/aio_sandbox_provider.py
```

如无 noldus 定制 (数字=0) 且 diff 小, 整覆盖:
```bash
git show deerflow/main:backend/packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py \
    > $HARNESS/community/aio_sandbox/aio_sandbox_provider.py
```

- [ ] **Step 10: 处理 gateway routers (memory.py / runs.py / threads.py / thread_runs.py / uploads.py)**

为每个 router 做相同流程: `diff` + 检查 noldus 定制 + 可整覆盖则整覆盖, 否则 surgical merge。

⚠️ **gateway/routers/threads.py noldus diff 是 614 行,极重定制**。**绝对不能整覆盖。** 必须 surgical merge,只挑上游加的 user_id 处理代码。

- [ ] **Step 11: 拉迁移脚本 scripts/migrate_user_isolation.py**

```bash
git show deerflow/main:backend/scripts/migrate_user_isolation.py \
    > packages/agent/backend/scripts/migrate_user_isolation.py 2>/dev/null \
    || echo "file not at this path, check"
ls packages/agent/backend/scripts/migrate_user_isolation.py 2>&1
```

- [ ] **Step 12: 处理新增测试 (test_memory_*_user_isolation.py 等)**

```bash
git show 2e05f380 --stat | grep "test_.*_user_isolation\|test_migration_user"
```

每个新测试整覆盖 (上游新增, noldus 端无):
```bash
for t in test_memory_queue_user_isolation.py test_memory_storage_user_isolation.py test_memory_updater_user_isolation.py test_migration_user_isolation.py; do
    git show "deerflow/main:backend/tests/$t" > "packages/agent/backend/tests/$t" 2>/dev/null && echo "OK $t" || echo "SKIP $t"
done
```

- [ ] **Step 13: 跑测试**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, X passed, 14 skipped`, X 比 D.2 后增加 ~20 (新增 4 个 user_isolation 测试文件)。

如果失败 > 2, 看具体失败排查。

### Task D.4: 拉默认 database config (35ef8b7c) + lint 修复 D.5

**Files:**
- Modify: `packages/agent/config.yaml` (加 database / run_events 段)
- Modify: `packages/agent/config.example.yaml` (同上)
- Modify: `packages/agent/backend/packages/harness/deerflow/config/app_config.py` (注册 DatabaseConfig 子段)
- Modify: 各种 lint 修复

- [ ] **Step 1: 看 35ef8b7c**

```bash
cd /home/wangqiuyang/noldus-insight
git show 35ef8b7c --stat
```

预期改:
- `config/app_config.py` 加 `database: DatabaseConfig = Field(default_factory=DatabaseConfig)`
- `config.example.yaml` 加 `database: { backend: memory }`

- [ ] **Step 2: 看 noldus app_config.py 现状**

```bash
grep -n "DatabaseConfig\|database\|run_events" packages/agent/backend/packages/harness/deerflow/config/app_config.py | head -10
```

如未含 database 字段, 用 Edit tool 加。

- [ ] **Step 3: 看 noldus config.yaml 现状**

```bash
grep -n "^database:\|^run_events:" packages/agent/config.yaml
```

如不存在,用 Edit tool 在合适位置 (例如 checkpointer 段附近) 添加:
```yaml
# ============================================================================
# Database (新增 round 2 D 阶段) - 默认 memory backend, noldus 行为不变
# ============================================================================
database:
  backend: memory  # memory | sqlite | postgres

run_events:
  store: memory  # memory | db | jsonl
```

- [ ] **Step 4: 同步改 config.example.yaml**

如存在则用 Edit tool 加同样段。

- [ ] **Step 5: 拉 D.5 三个 lint commit (16aedf45 / 897dae54 / 829e82a9)**

```bash
git show 16aedf45 --stat
git show 897dae54 --stat
git show 829e82a9 --stat
```

如改动文件 noldus 端已含上游版本 (因为 D.1 拉了上游 head, lint 修复已在内), 跳过。否则按 patch 应用。

- [ ] **Step 6: smoke + 测试**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.config import get_app_config
cfg = get_app_config()
print('database backend:', cfg.database.backend)
"
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望:
```
database backend: memory
2 failed, X passed, 14 skipped
```

### Task D.6: memory cache corruption 修复 (898f4e8a)

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py` (or storage.py)
- Modify: 相关测试

- [ ] **Step 1: 看 commit**

```bash
cd /home/wangqiuyang/noldus-insight
git show 898f4e8a --stat
git show 898f4e8a -- 'backend/packages/harness/deerflow/agents/memory/' | head -100
```

- [ ] **Step 2: 检查 noldus 现状**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/agents/memory/queue.py) \
     packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py | wc -l
diff <(git show deerflow/main:backend/packages/harness/deerflow/agents/memory/storage.py) \
     packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py | wc -l
```

如果 diff = 0 (已吸收), 跳过此 task 整体。如果 > 0, 看 commit 改动手工合入。

- [ ] **Step 3: 跑 memory 测试**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ -k "memory" --no-header -q 2>&1 | tail -5
```

期望全过。

### Task D.7: memory I/O 用 asyncio.to_thread (87609374)

类似 D.6 流程: 检查 diff,如 0 跳过,如非 0 surgical merge。

```bash
git show 87609374 --stat
git show 87609374 -- 'backend/packages/harness/deerflow/agents/memory/' | head -50
```

按上游 patch 应用。

### Task D.8: checkpoint rollback on cancel (35f141fc) — worker.py

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py`
- Modify: 相关 test_worker_*.py 测试

⚠️ noldus worker.py **0 处定制** (前面调研确认), 但是上游历史还有 `8ba01dfd / 78633c69 / e82940c0 / 17447fcc` 等也在改 worker.py。**为简化, 直接整文件覆盖到上游 head**, 因为 noldus 没有定制可丢。

- [ ] **Step 1: 再次确认 noldus worker.py 0 定制**

```bash
grep -c "noldus\|EthoInsight\|shared_dir\|/mnt/shared\|extra_env" packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py
```

期望 0。如非 0, **立即停下问用户**。

- [ ] **Step 2: 整覆盖**

```bash
git show deerflow/main:backend/packages/harness/deerflow/runtime/runs/worker.py \
    > packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py
```

- [ ] **Step 3: 拉 worker 相关测试**

```bash
# 找所有 35f141fc 加的测试
git show 35f141fc --stat | grep test_
```

例如可能是 `tests/test_run_worker_rollback.py` 之类。整覆盖每个新测试。

- [ ] **Step 4: import 检查**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.runtime.runs.worker import run_agent
print('OK')
"
```

如果 ImportError (例如 missing `from deerflow.config.app_config import AppConfig` — 实际 noldus 有), 看错误信息排查。

- [ ] **Step 5: 跑相关测试**

```bash
PYTHONPATH=. uv run pytest tests/ -k "worker or rollback or run_agent" --no-header -q 2>&1 | tail -5
```

期望全过。

### Task D.9: gateway ISO 8601 timestamps (ca3332f8)

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/sql.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/manager.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/utils/time.py`
- Modify: 多处 gateway routers

- [ ] **Step 1: 看 commit**

```bash
git show ca3332f8 --stat
```

- [ ] **Step 2: persistence/thread_meta/sql.py 应该已是上游 head (D.1 拉的最新), 验证**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/persistence/thread_meta/sql.py) \
     packages/agent/backend/packages/harness/deerflow/persistence/thread_meta/sql.py | wc -l
```

期望 0。

- [ ] **Step 3: runtime/runs/manager.py surgical (74 行 noldus diff)**

```bash
git show ca3332f8 -- 'backend/packages/harness/deerflow/runtime/runs/manager.py'
```

按 patch 改 noldus manager.py。surgical edit, 保留 noldus 定制 (如有)。

- [ ] **Step 4: utils/time.py 验证**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/utils/time.py) \
     packages/agent/backend/packages/harness/deerflow/utils/time.py | wc -l
```

期望 0 (D.1 已拉)。

- [ ] **Step 5: gateway routers 处理**

按 commit ca3332f8 列出的所有 router, 逐个 surgical merge (其中 threads.py 是高风险, 必须只挑 timestamp 转换部分)。

- [ ] **Step 6: 跑 gateway 测试**

```bash
PYTHONPATH=. uv run pytest tests/ -k "thread or gateway" --no-header -q 2>&1 | tail -5
```

期望全过。

### Task D.10: rollback restore checkpoint supersede (17447fcc)

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py` (D.8 已整覆盖, 应该已含此 commit)

- [ ] **Step 1: 验证 worker.py 已是上游 head**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/runtime/runs/worker.py) \
     packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py | wc -l
```

期望 0。

- [ ] **Step 2: 拉相关测试**

如有新增测试 (例如 test_rollback_supersede), 整覆盖。

```bash
git show 17447fcc --stat | grep test_
```

- [ ] **Step 3: 跑测试**

```bash
PYTHONPATH=. uv run pytest tests/ -k "rollback or supersede" --no-header -q 2>&1 | tail -5
```

### Task D 收尾: 全量回归 + 启动 smoke + commit

- [ ] **Step 1: 全量测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, X passed, 14 skipped`, X 比 C.5 后增加 ~80-100 (新增大量 persistence / events / journal / rollback 测试)。

如果失败 > 2, **立即停下, 看具体失败**。最常见原因:
- 新引入的 persistence/__init__.py 在某些路径下被 langgraph 启动加载, 但 SQLAlchemy 没装 — 看 `backend/pyproject.toml` 和 `backend/packages/harness/pyproject.toml` 是否需要加依赖
- 旧 `from deerflow.agents.checkpointer import` 调用的 mock path 在 D.2 shim 后 module identity 变了 — 改测试

- [ ] **Step 2: 跑 lint**

```bash
PYTHONPATH=. uv run ruff check packages/harness/deerflow/persistence/ \
    packages/harness/deerflow/runtime/checkpointer/ \
    packages/harness/deerflow/runtime/events/ 2>&1 | tail -5
```

期望: `All checks passed!` 或 0 error。如非 0, fix。

- [ ] **Step 3: 启动 smoke (langgraph + gateway)**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev 2>&1 | tee /tmp/d-smoke.log &
sleep 30
curl -s http://localhost:8001/health 2>&1 | head -3
make stop
```

期望: `{"status":"ok"}` 或类似, 不报 import error。

如果启动失败, 看 `/tmp/d-smoke.log` 排查。最常见: `import` 时 SQLAlchemy 缺失 — 因为 D 默认 memory backend, 应该 lazy import, 检查 `persistence/__init__.py` 是否在顶层 import sqlalchemy。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/persistence/ \
        packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/ \
        packages/agent/backend/packages/harness/deerflow/runtime/events/ \
        packages/agent/backend/packages/harness/deerflow/runtime/runs/ \
        packages/agent/backend/packages/harness/deerflow/runtime/journal.py \
        packages/agent/backend/packages/harness/deerflow/agents/checkpointer/ \
        packages/agent/backend/packages/harness/deerflow/config/database_config.py \
        packages/agent/backend/packages/harness/deerflow/config/run_events_config.py \
        packages/agent/backend/packages/harness/deerflow/config/app_config.py \
        packages/agent/backend/packages/harness/deerflow/utils/time.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/thread_data_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/memory/ \
        packages/agent/backend/packages/harness/deerflow/sandbox/tools.py \
        packages/agent/backend/packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py \
        packages/agent/backend/packages/harness/deerflow/uploads/manager.py \
        packages/agent/backend/packages/harness/deerflow/config/paths.py \
        packages/agent/backend/app/gateway/ \
        packages/agent/backend/scripts/ \
        packages/agent/backend/tests/ \
        packages/agent/config.yaml \
        packages/agent/config.example.yaml 2>/dev/null

git commit -m "$(cat <<'EOF'
sync deerflow upstream Tier 4 D: BC 持久化层 11 commit

新增目录:
- persistence/ (21 个文件: ORM models / repositories / migrations / engine / feedback / user)
- runtime/checkpointer/ (从 agents/checkpointer/ mv 来 + agents/checkpointer/ 保留 shim)
- runtime/events/ (RunEventStore ABC + memory/db/jsonl 实现)
- runtime/runs/store/ (RunStore ABC)

新增模块:
- runtime/journal.py (RunJournal: BaseCallbackHandler, 累积 token usage)
- config/database_config.py (memory/sqlite/postgres backends)
- config/run_events_config.py (memory/db/jsonl)
- utils/time.py (ISO 8601 helpers)

合入 commit:
- d8ecaf46 persistence scaffold + RunEventStore + ORM models (#1930)
- 56d5fa33 unified persistence rebase + checkpointer mv (#2134)
- 2e05f380 per-user filesystem isolation (user_id 默认 None, BC) (#2153)
- 35ef8b7c 默认 database config
- 16aedf45 / 897dae54 / 829e82a9 lint 修复
- 898f4e8a memory cache corruption (#2251)
- 87609374 memory I/O asyncio.to_thread (#2220)
- 35f141fc checkpoint rollback on cancel (#1867)
- ca3332f8 gateway ISO 8601 timestamps (#2599)
- 17447fcc rollback restore checkpoint supersede (#2582)

config.yaml 加默认 `database: { backend: memory }` 和 `run_events: { store: memory }`,
noldus 现有行为不变。

保留 Noldus 全部定制:
- agents/checkpointer/ 兼容 shim 让旧 import 路径继续工作
- sandbox/tools.py {{shared://}} / SHARED_PATH_PREFIX / mask_local_paths_in_output
- config/paths.py /mnt/shared / shared_dir()
- gateway/routers/threads.py noldus 大量定制

依赖 Phase E + C.5
详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md §2.3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: 验证**

```bash
git log --oneline -3
git diff --stat HEAD~1 HEAD | tail -3
```

期望 HEAD 是 "sync deerflow upstream Tier 4 D: BC 持久化层 11 commit", 共 3 个新 commit。

---

## Phase F: 收尾 (Task F.1 - F.2)

### Task F.1: 写交接文档

**Files:**
- Create: `docs/handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md`

- [ ] **Step 1: 用 round 1 handoff 模板写**

参考 `docs/handoffs/2026-05-07-deerflow-tier234-round1-completed-handoff.md` 结构, 内容应包括:

1. **TL;DR**: 3 个 commit, X passed → Y passed
2. **已完成 commit 清单** (E / C.5 / D 三阶段, 表格)
3. **跳过项及原因** (例如 better-auth 留给轮 3, noldus user-backend 4-23 计划弃用)
4. **关键状态验证** (测试基线 / 受保护文件状态 / git log)
5. **轮 3 backlog**:
   - better-auth 全套 (94eee95f / 848ace98 / da174dfd / 98a5b34f / 4e4e4f92)
   - 配套 7 个 wiring (8ba01dfd / 38714b6c / e82940c0 / b8bc4826 / 83938cf3 / 30d619de / 487c1d93)
   - 前端登录页适配中文 + EthoInsight 品牌
   - 写新 multi-user 部署 SOP
6. **完成度统计**:
   - 105 commit:
     - T1-DONE 11 (上次会话)
     - 轮 1 已合 33
     - 轮 2 已合 X (E + C.5 + D = 13 个 + 配套 lint)
     - 留轮 3: better-auth + wiring 共 ~12-15
     - 永久跳过 (e.g. f80ac961 / 7dea1666 / Serper / Exa 等)
7. **风险与已知问题**:
   - 默认 memory backend, 上线时切 sqlite/postgres 需测
   - agents/checkpointer/ shim 在某些 mock 路径下可能要更新测试
   - persistence/ 引入了 SQLAlchemy 依赖, 检查 pyproject.toml
8. **不要 push**

- [ ] **Step 2: Commit handoff**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md \
        docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md \
        docs/superpowers/plans/2026-05-07-deerflow-tier234-round2-plan.md
git commit -m "docs: 轮 2 deerflow 同步完成交接文档 + 设计/实施 plan"
```

### Task F.2: 标 4-23 user-backend 计划为 deprecated

**Files:**
- Modify: `docs/plans/2026-04-23-multi-user-deployment.md`

- [ ] **Step 1: 在文件顶部加 deprecated 标记**

用 Edit tool 在 `docs/plans/2026-04-23-multi-user-deployment.md` 的 `# 多用户部署实施计划` 标题后加:

```markdown
> ⚠️ **DEPRECATED 2026-05-07** — 本计划已被上游 better-auth 路线取代,详见 `docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md` §1.1。
> noldus user-backend 自建路线弃用,改为合入上游 deerflow 的 `94eee95f / 848ace98 / da174dfd / 98a5b34f / 4e4e4f92` 五个 commit (轮 3 实施)。
> 本文档保留作为历史参考。

```

- [ ] **Step 2: Commit**

```bash
git add docs/plans/2026-04-23-multi-user-deployment.md
git commit -m "docs: 标记 4-23 user-backend 计划为 deprecated"
```

---

## 总完成定义 (Done Criteria)

- [ ] 4 个 commit 在本地 dev 分支:
  - sync deerflow upstream Tier 4 E: skill storage 重构
  - sync deerflow upstream Tier 3 C.5: summarization skill rescue
  - sync deerflow upstream Tier 4 D: BC 持久化层 11 commit
  - docs: 轮 2 deerflow 同步完成交接文档 + ...
  - docs: 标记 4-23 user-backend 计划为 deprecated
- [ ] 测试基线: `2 failed, ≥1900 passed, 14 skipped` (X 应该比基线 1811 增加 80-160)
- [ ] `make lint` 0 error
- [ ] `make dev` 启动成功
- [ ] noldus 全部受保护文件 grep check 通过 (中间件链 / `{{shared://}}` / `recursion_limit` / 中文 prompt 全在)
- [ ] **不 push** — 留 dev 分支等用户决定

---

## 应急处置

### 任何阶段失败超过 2h 且测试不通过

1. `git status` 看未提交改动
2. 写中断 handoff: `docs/handoffs/2026-05-07-tier234-round2-INTERRUPTED.md` 记录:
   - 当前在哪个 Task
   - 卡住的具体错误
   - 已尝试的 debugging 步骤
3. **不 commit 半成品**
4. `git stash` 当前改动备份
5. 通知用户

### 测试失败数从 2 涨到 3+

1. **立即停下**, 不要继续往下做
2. 跑失败的测试单独看错误信息:
   ```bash
   PYTHONPATH=. uv run pytest tests/<failing_test>.py -v 2>&1 | tail -30
   ```
3. 99% 是 mock path 没更新或某个 import 漏改, 不是真正的逻辑 bug
4. 修完了再继续

### 启动失败 (make dev 报 ImportError)

1. 看 `/tmp/d-smoke.log` (或 `/var/log/...`) 找 import error 行
2. 通常是 D.1 拉的某个新文件 import 了 noldus 没装的 package (SQLAlchemy / asyncpg / alembic)
3. 检查 `packages/agent/backend/pyproject.toml` 和 `packages/agent/backend/packages/harness/pyproject.toml` 看是否需要加依赖
4. 如果上游加的依赖在 pyproject.toml 但 noldus 没合, 添加到对应 pyproject 然后跑 `cd backend && uv sync`

### Surgical merge 把 noldus 定制改丢了

1. `git diff` 看具体哪个文件丢失了什么
2. `git checkout HEAD -- <file>` 恢复整个文件 (前提是已 commit 过)
3. 重新做这个文件的 surgical merge, 这次更小心

### Subagent driven development 中断

如果用 subagent-driven-development skill 实施, 中途某个 subagent 失败:
1. 不要让下一个 subagent 接力, 它没有上下文
2. 主 agent 自己接手当前 task 完成
3. 然后再继续后续 task
