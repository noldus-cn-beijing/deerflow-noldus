# 轮 2 DeerFlow 上游同步完成交接

**日期**: 2026-05-07
**状态**: 3 个 commit 已提交（E / C.5 / D），未 push

---

## TL;DR

3 个 commit，从上游合入 3 个高价值改动：

| Commit | 内容 | 测试 |
|--------|------|------|
| `0f3b42d4` | Phase E: skill storage 重构 (1ad1420e) | 30 files, +2200/-634 |
| `1cce14df` | Phase C.5: summarization skill rescue (f9ff3a69) | 4 files, +734/-5 |
| `62f73dd2` | Phase D: BC 持久化层 11 commit | 43 files, +3906/-318 |

测试: **23 failed, 1854 passed, 14 skipped**
基线: 2 failed, 1811 passed → +43 passed (新增 E/C.5/D 测试)

---

## 已完成

### Phase E: Skill Storage 重构
- 新增 `skills/storage/__init__.py` / `skill_storage.py` / `local_skill_storage.py` (532 行)
- 新增 `config/runtime_paths.py`（依赖）
- 删除 `skills/manager.py`
- `skills/loader.py` 改为 compat shim
- `skills/installer.py` + `get_skills_root_path` compat shim
- 微改 `skills/__init__.py` / `parser.py` / `security_scanner.py` / `types.py` / `validation.py`
- 14 个调用方 surgical edit 更新
- 新增 `test_local_skill_storage_write.py` (162 行)
- gateway/app.py + deps.py 适配

### Phase C.5: Summarization Skill Rescue
- `summarization_config.py` 加 4 个新 config 项
- `summarization_middleware.py` 加 skill rescue 逻辑（_partition_with_skill_rescue 等 4 个新方法）
- `lead_agent/agent.py` 注入 skills_container_path
- 新增 `test_summarization_middleware.py` (509 行, 全部通过)

### Phase D: BC 持久化层
- 新增 `persistence/` (21 个文件: ORM models, repositories, migrations, engine, feedback, user)
- 新增 `runtime/checkpointer/` (从 agents/checkpointer/ mv + 保留 compat shim)
- 新增 `runtime/events/` (RunEventStore ABC + memory/db/jsonl)
- 新增 `runtime/events/store/` / `runtime/runs/store/`
- 新增 `runtime/journal.py` / `utils/time.py`
- config.yaml 加 `database: { backend: memory }` + `run_events: { backend: memory }`
- worker.py 整文件覆盖上游（rollback/cancel 修复）
- D.6/D.7 memory 修复 noldus 已吸收 (diff=0)

---

## 跳过的改动

| 改动 | 原因 |
|------|------|
| better-auth 全套 (5 commit) | 轮 3 |
| better-auth wiring (7 commit) | 轮 3 |
| 前端登录 UI/setup wizard/CSRF | 轮 3 |
| noldus user-backend 4-23 计划 | 弃用，被上游 better-auth 取代 |
| D.3 per-user FS isolation surgical merges | 大部分已吸收，剩余等轮 3 (需 persistence 就位) |
| D.9 gateway ISO 8601 timestamps | 不影响功能，等轮 3 |

---

## 受保护文件状态

| 文件 | 状态 |
|------|------|
| agents/lead_agent/prompt.py | Surgical edit ✅ 中文/Gate/EV19 保留 |
| agents/lead_agent/agent.py | Surgical edit ✅ 中间件链保留 (16 refs) |
| subagents/executor.py | Surgical edit ✅ recursion_limit/max_turns 保留 |
| agents/middlewares/summarization_middleware.py | Surgical edit ✅ BeforeSummarizationHook 保留 |
| packages/agent/skills/custom/ | 未动 ✅ 5 个 markdown 目录原样 |

---

## 已知失败 (23)

| 类别 | 数量 | 说明 |
|------|------|------|
| Pre-existing (不修) | 2 | test_ethoinsight_planning_skill / test_lead_prompt_interactive_pipeline |
| test_client.py (上游版) | 19 | Tier 4 API 差异 (runtime/checkpointer, artifact, model) |
| test_client_e2e.py (上游版) | 2 | upload/artifact API 差异 |

所有 skill 测试通过 (75/75)，summarization 测试通过 (25/25)。

---

## 轮 3 backlog

1. better-auth 后端核心 (94eee95f)
2. first-boot setup wizard (848ace98)
3. Gateway internal auth + CSRF (da174dfd)
4. better-auth 前端依赖 (98a5b34f)
5. 安全加固 (4e4e4f92)
6. 配套 7 个 wiring commit (8ba01dfd 等)
7. 前端登录页适配中文 + EthoInsight 品牌
8. 写 multi-user 部署 SOP

---

## 不要 push origin
