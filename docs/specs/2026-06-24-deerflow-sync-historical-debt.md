# 2026-06-24 DeerFlow 同步历史债清单

> 触发：sync PR #191（e418d729→11415875）深挖发现 —— 本地 harness 在 e418d729 **之前**就漏合了上游两批重构 + app 层几个独立 feature。`.deerflow-sync-state` 标的 e418d729 是**名义同步点**，实际本地多处落后。
>
> 来源：PR #191 合入上游测试时，5 个测试文件（test_lead_agent_prompt / test_gateway_services / test_subagent_executor / test_uploads_router / test_feishu_parser）的上游版假设了本地没有的接口，整文件覆盖后暴露这些债。已回退这 5 个到本地版，债单独立项清理。

## 债 1：app_config 参数贯通（commit 8ba01dfd，纯参数通路）

**根因 commit**：`8ba01dfd refactor: thread app_config through lead and subagent task path`

| 函数 | 本地签名 | 上游签名 | Noldus 冲突 |
|---|---|---|---|
| `_get_memory_context` | `(agent_name)` | `(agent_name, *, app_config=None)` | 无（纯数据层，行 730-772） |
| `apply_prompt_template` | `(…, paradigm, user_id, thread_id, …)` | `(…, app_config=None, …)` | 有（函数体 1052-1185 含中文调度/Gate/反问，但 app_config 仅 1159/1164 惰性读取，可 surgical） |
| `_build_acp_section` | `()` | `(*, app_config=None)` | 无（行 979-997） |
| `_build_custom_mounts_section` | `()` | `(*, app_config=None)` | 无（行 999-1019） |

**修复**：surgical。4 个签名加 `app_config` 参数 + `agent.py` 2 处调用点补传 + 函数体内 `get_app_config()` 改用参数。Noldus 业务段原样保留。**规模：小**。

**风险文件**：`agents/lead_agent/prompt.py`（受保护）、`agents/lead_agent/agent.py`（受保护）

## 债 2：skills cache 重构（commit 0ee9ad9e，2026-05-11，基础设施重构）

本地缺 4 类符号：
- `_enabled_skills_by_config_cache: dict[int, tuple[object, list[Skill]]] = {}`（模块级 per-AppConfig 缓存）
- `_build_self_update_section(agent_name)` → custom agent 自更新 prompt 段
- `get_enabled_skills_for_config(app_config)` / `get_cached_enabled_skills()` → per-config skills 加载
- `_build_available_subagents_description(available_names, bash_available, *, app_config)` → 动态 subagent 描述

**核心断裂**：`apply_prompt_template` 签名本地是 `paradigm/user_id/thread_id`（Noldus 注入 prior_corrections + resolved_facts），上游是 `app_config` → 两套参数体系冲突。

**修复**：中等。cherry-pick 4 符号 + `apply_prompt_template` 同时支持两种 config 获取（参数 > 全局 `get_app_config`）+ `get_skills_prompt_section(available_skills, app_config=)` 对齐。**保留** Noldus EthoInsight subagent rendering（`_build_subagent_section` 大幅定制）+ orchestration_guide + clarification_system。

**风险文件**：`agents/lead_agent/prompt.py`、`agents/lead_agent/agent.py`、`skills/storage.py`

## 债 3：ToolRuntime `tools` 参数（langchain 库升级）

本地 `ToolRuntime.__init__` 不接受 `tools`，上游接受。**17 个调用点全在测试代码**（test_prep_metric_plan_tool / test_tool_args_schema_no_pydantic_warning / test_analysis_config_id / test_task_tool_core_logic 等），非 harness 核心。

**修复**：小。先确认是 langchain/langgraph 库版本差异还是 deerflow 源码改动 —— 查 pyproject.toml 锁定版本。若库升级引入，逐处加 `tools=None`；若源码改动，定位 commit。

**风险文件**：无 harness 受保护文件，纯测试调用点。

## 债 4：executor async 化（`_build_initial_state` + `_load_skills`）

- 本地 `_build_initial_state`（行 1029）**sync** 返回 tuple → 上游（main 行 440）**async**
- 本地无 `_load_skills` → 上游 `async _load_skills`（main 行 378）
- 本地用 `_load_skill_contents`（sync helper，行 824）替代上游 `_load_skill_messages`

**修复**：小。2 个方法改 async + `_aexecute`（行 1259）补 `await` + helper 替换。**Noldus 定制正交不受影响**：`_attempt_auto_seal_from_artifacts` / `_preset_handoff_template_if_needed` / `recursion_limit` 计算 / handoff validation 都是同步或惰性 import。

**风险文件**：`subagents/executor.py`（受保护，但 Noldus 定制段与 async 化正交）

## 债 5：app 层三类小缺口（非 harness，sync 脚本不覆盖 app/）

| 符号 | 上游 commit | 作用 | 修复 |
|---|---|---|---|
| `_make_file_sandbox_readable` | `f9b70713` | Docker sandbox 文件加组/其他读权限（S_IRGRP\|S_IROTH） | surgical，与本地 `_make_file_sandbox_writable` 协调调用时序（上游写入后循环外调 readable，sync_to_sandbox 分支调 writable） |
| `FeishuChannel.receive_file/_receive_single_file(user_id=)` | `aa015462` | IM 多用户（#3487） | surgical，逐处补 user_id 传递（manager.py 调用点 + 内部认证头） |
| `apply_checkpoint_to_run_config` + `INTERNAL_OWNER_USER_ID_HEADER_NAME` + `get_trusted_internal_owner_user_id` + `create_internal_auth_headers(owner_user_id=)` | `ca9428d0` / `aa015462` | regenerate 答案（checkpoint）+ 内部认证 owner | surgical，新增函数 + 路由调用点协调 |

**规模：中等**，3 类都需调用链协调。

**风险文件**：`app/gateway/routers/uploads.py`、`app/channels/feishu.py`、`app/gateway/services.py`、`app/gateway/internal_auth.py`（都在 app/ 层，非 harness 受保护清单，但需单独同步）

## 执行建议

按风险/收益排序：
1. **债 1（app_config 贯通）**：小、纯参数通路、解锁上游测试 → **先做**，让 test_lead_agent_prompt / test_gateway_services 能合入上游版
2. **债 4（executor async 化）**：小、正交无冲突 → 顺带做，解锁 test_subagent_executor
3. **债 3（ToolRuntime tools）**：小、纯测试 → 确认根因后做，解锁 17 个测试
4. **债 2（skills cache 重构）**：中等、有签名冲突需仔细 surgical → 单独 PR
5. **债 5（app 层）**：中等、3 类独立 → 可拆 3 个小 PR 或合并一个

完成后本地 harness 真正追平 e418d729 基线，后续 sync（11415875 之后）不会再因预存偏离暴露红测试。
