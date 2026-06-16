# DeerFlow 上游 sync spec — f92a26d5 → 2d5f0787（11 commit）

> 执行 agent：本 spec 指导你把 deerflow 上游自 `f92a26d5`（2026-06-09 上次 sync 基准）到 `2d5f0787`（当前上游 HEAD）的 **11 个 commit** 合入本仓库 dev。
> 写于 2026-06-11。已逐 commit head-to-head 评估，分类 + surgical 风险点 + 验证步骤都在下方。
>
> **前置铁律（CLAUDE.md「同步核心规则」）**：deerflow 是 infra 底座，**默认全量合**；唯一 surgical（逐处对比、绝不整文件覆盖）的是含 Noldus 定制的**受保护文件**。本 spec 已替你判好每个 commit 属哪类。

---

## 0. 一句话现状

- **基准**：`.deerflow-sync-state` 记录 `last_sync_commit: f92a26d5`（2026-06-09）。
- **上游 HEAD**：`2d5f0787`（`git fetch deerflow` 后 `git rev-parse deerflow/main`）。
- **delta**：harness + app 层 **11 个 commit**、34 个文件。
- **分类结论**：**7 个 🟢 纯安全/新增（可全量合）** + **4 个 🟡 触受保护文件（surgical）**。
- **无一个 commit 依赖我们没有的子系统**（unified auth/better-auth 等）——本仓库已吃下 Tier 4，全部可合。

---

## 1. 执行前准备

```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow
git rev-parse deerflow/main          # 应为 2d5f0787…，若不同说明上游又前进了，本 spec 的 commit 清单需重新核对
git status --short                   # 工作区必须干净（只允许未跟踪 docs）
git checkout dev && git pull         # 确保在最新 dev
```

**新建 sync 分支**（不直接在 dev 上改）：

```bash
git checkout -b chore/sync-deerflow-2d5f0787
```

> 本仓库无独立 LANGGRAPH 容器，跑 Gateway-embedded。改完必须裸导入两生产入口验证（见 §4）。

---

## 2. 逐 commit 合入清单

> 标记含义：🟢 = 纯安全/新增文件，全量 `git checkout`/`git show >` 接受即可；🟡 = 触受保护文件，必须 surgical（先 `diff` 看上游本次改了什么，只把上游改动点手工合入，保留所有 Noldus 定制）。
> 上游路径前缀 `backend/packages/harness/deerflow/`（harness）或 `backend/app/`（app 层）；本地对应 `packages/agent/backend/...`。

### 🟢 高价值，优先合（纯安全文件，零冲突）

#### 2.1 `8db16bb3` fix(config): coerce null config.yaml list sections to empty list
- **价值（高）**：修 `config.example.yaml` 首次 `cp` 到 `config.yaml` 后、把 `models:`/`tools:`/`tool_groups:` 全注释掉时 PyYAML 解析成 `None` → pydantic `Input should be a valid list` 崩溃。**本仓库已实证此报错**：2026-06-11 裸导入 `app.gateway` 时 skill 预热阶段就抛这个（`models Input should be a valid list`，因测试环境无 config.yaml）。
- **触受保护**：`config/app_config.py`（受保护——含 Noldus `handoff_strict_mode` 字段）。
- **surgical 操作**：上游本次只在 `AppConfig` 加一个 `@field_validator("models","tools","tool_groups", mode="before")` 方法（`_coerce_null_list_sections`，返回 `[] if value is None else value`）+ `from_file` 末尾加一段 `if not result.models: logger.warning(...)` + import 加 `field_validator`。这三处与我们的 `handoff_strict_mode` 字段**正交**，直接把这三段手工加进本地 `app_config.py`，**不碰**任何 Noldus 定制行。
- **配套测试**：上游带了测试，确认无（本 commit 的 test 在 `tests/test_app_config*.py`，按需拿过来）。

#### 2.2 `b62c5a7b` fix(agents): offload blocking filesystem IO in the custom-agent router off the event loop
- **价值（高）**：把 custom-agent router 里阻塞的文件 IO 挪出 event loop。**与我们 2026-05-28 修的 `async handler 里 read_bytes() 阻塞 event loop` 同类**（见 MEMORY [[feedback_async_io_blocks_event_loop]]）——这类 event-loop 阻塞在 Gateway-embedded 下会互锁 SSE，是我们迟早会踩的坑。
- **触文件**：`app/gateway/routers/agents.py` + `app/gateway/services.py`（app 层，非受保护）。
- **操作**：全量接受（grep 确认这两个文件无 Noldus 定制；若有 ethoinsight/shared_path 等定制则降级 surgical）。

#### 2.3 `ba9cc5e9` fix(gateway): enforce thread ownership on stateless run endpoints
- **价值（高，安全相关）**：无状态 run 端点强制 thread 归属校验——多用户研究助手的越权防护（本仓库是多用户，Tier 4 已吃下）。
- **触文件**：`app/gateway/` 下若干（app 层，非受保护）。
- **操作**：全量接受。**部署相关注意**：这是权限收紧，合入后确认线上前端调用链带正确的用户上下文（理论上 Tier 4 已通，但部署后验一次 thread 列表/历史能正常加载）。

#### 2.4 `a57d05fe` fix runtime journal run lifecycle events
- **价值（中）**：修 runtime journal 的 run 生命周期事件。journal 是 Tier 4 子系统，本仓库已有。
- **触文件**：`packages/harness/deerflow/runtime/journal.py`（非受保护）。
- **操作**：全量接受（grep 确认 journal.py 无 Noldus 定制）。

#### 2.5 `ae9e8bc0` fix(sandbox): make missing sandbox.mounts host_path a loud ERROR
- **价值（中）**：sandbox.mounts 的 host_path 缺失时从静默变响亮 ERROR（符合我们「响亮故障优于哑故障」纪律）。
- **触文件**：`packages/harness/deerflow/config/sandbox_config.py`（非受保护）。
- **操作**：全量接受。**注意**：确认不影响我们 `local_sandbox.py` 的 `extra_env`/`DEERFLOW_PATH_*` 定制（那是另一个文件，受保护，本 commit 不碰）。

#### 2.6 `b3c2cc42` fix(agents): require config.yaml in resolve_agent_dir to skip memory-only directories
- **价值（中）**：`resolve_agent_dir` 要求有 config.yaml 才认作 agent 目录，跳过纯 memory 目录。
- **触文件**：`packages/harness/deerflow/config/agents_config.py`（非受保护）。
- **操作**：全量接受。

#### 2.7 `37337b77` feat(models): add StepFun reasoning model adapter
- **价值（低，但零害）**：新增 StepFun reasoning model 适配器（`models/patched_stepfun.py` 新文件）。我们用 deepseek，用不上——但 CLAUDE.md 铁律：纯新增文件落非受保护区，**合进来零害**（不走那条路不触发）且消除下次 sync 冲突点。
- **触文件**：`packages/harness/deerflow/models/patched_stepfun.py`（新增）+ models 注册（确认注册点是否受保护）。
- **操作**：全量接受新文件；若注册点在受保护的聚合 import/`__all__` 块，surgical 加注册行。

### 🟡 触受保护文件，surgical（逐处对比）

#### 2.8 `0fb18e36` refactor(lead-agent): make build_middlewares public 【纯机械重命名】
- **价值（中，降低未来 sync 摩擦）**：上游把 `_build_middlewares` → `build_middlewares`（去掉下划线，提为公开 API），因为 `client.py` 跨模块调它。**纯机械重命名，零行为变化**。
- **触受保护**：`agents/lead_agent/agent.py`（受保护——含 Noldus 定制中间件链）。
- **本地现状（已核实）**：本地 `client.py:36` + `client.py:250` import 并调用 `_build_middlewares`；**5 个测试文件** monkeypatch/patch `_build_middlewares`（`test_lead_agent_skills.py:113`、`test_checkpointer.py:750/784`、`test_lead_agent_model_resolution.py:86/135`、`test_client_e2e.py:147`）。
- **surgical 操作（二选一，建议 A）**：
  - **A（跟随上游重命名）**：在 `agent.py` 把 `def _build_middlewares` 改名 `build_middlewares`（保留 Noldus 定制中间件链全部不动，只改函数名 + 两处内部调用）；同步改 `client.py` 的 import + 调用、5 个测试文件的 patch 目标。好处=与上游对齐，下次 sync 这个文件少一个分叉点。**改完必须裸导入验证（§4）+ 跑这 5 个测试文件**。
  - **B（不跟随，保留 `_build_middlewares`）**：跳过本 commit 的重命名。代价=本地 agent.py 与上游永久分叉一个符号名，下次上游再动 `build_middlewares` 周边会多一处 surgical。**不推荐**（违反「为'用不上'主动跳过=人为制造分叉」纪律，且这是零行为改动的廉价对齐）。
- **判据**：A 是 CLAUDE.md「默认全量合」的正解；这是机械重命名，surgical 成本极低（改名 + 同步调用方）。

#### 2.9 `167ef451` feat(memory): add memory.token_counting config to avoid tiktoken network dependency
- **价值（高，离线部署相关）**：加 `memory.token_counting` 配置，避免 tiktoken 联网下载 encoding（内网/离线 ECS 部署可能拉不到）。
- **触受保护**：`agents/lead_agent/prompt.py`（6 行）+ `agents/memory/prompt.py`（110 行）——两者都受保护（prompt.py 含中文调度规则；memory/prompt.py 含 Noldus topOfMind/history 隔离的 67 行定制，见 MEMORY [[feedback_sync_nonprotected_files_with_noldus_customization_overwritten]]）。
- **surgical 操作**：
  - 非受保护部分全量合：`config/memory_config.py`（新增 token_counting 字段，+13 行）、`client.py`（+1 行）、`app/gateway/app.py`、`app/gateway/routers/memory.py`、新测试 `test_tiktoken_cache_and_count_tokens.py`（+204 行，直接拿）。
  - **受保护部分逐处**：`diff <(git show deerflow/main:backend/packages/harness/deerflow/agents/memory/prompt.py) packages/agent/backend/packages/harness/deerflow/agents/memory/prompt.py`，**只合上游本次为 token_counting 加的逻辑**（git show 167ef451 -- 该文件 看本次净改），**保住 Noldus 的 `_format_memory` topOfMind/history 隔离 67 行**。`lead_agent/prompt.py` 的 6 行同理。
- **风险**：memory/prompt.py 是 2026-06-09 sync 被 full-follow 洗掉过定制的文件（血泪教训），**务必 head-to-head，绝不整文件覆盖**。

#### 2.10 `16391e35` fix(skills): harden slash skill activation across chat channels 【大 feature，评估后再定】
- **价值（低-中，IM channel 场景）**：支持 `/skill` slash 激活贯穿 IM channel（Feishu/Slack/Telegram/Discord/Dingtalk/Wechat/Wecom）+ 前端 slash 自动补全 + uploads 保留。**我们 v0.1 主要走 web 对话，IM channel 非主路径**。
- **触受保护**：`agents/lead_agent/agent.py` + `agents/lead_agent/prompt.py`（都受保护）+ 大量 app/channels/* + 新中间件 `skill_activation_middleware.py` + frontend。
- **建议**：**本 commit 体量大、触受保护多、价值与我们主路径弱相关**——
  - **选项 A（推荐：本轮先跳过，独立评估）**：本次 sync 跳过 16391e35，在交接文档/issue 记录"slash skill 激活留独立 PR 评估"。理由：它把 slash 激活逻辑塞进受保护的 agent.py + prompt.py，surgical 面大且需求弱；不像其余 commit 是"地基修复"。跳过它**不制造分叉风险**的前提是后续 commit 不依赖它——本区间它是叶子 feature（无下游依赖），可安全延后。
  - **选项 B（跟随）**：若团队确实要 IM channel 的 slash 能力，则 surgical 合入：新中间件 `skill_activation_middleware.py`（新文件，直接拿）；agent.py 中间件链插入点 surgical（保 Noldus 链顺序）；prompt.py surgical；app/channels/* 全量；frontend 全量。**成本高，仅在明确需要时做。**
- **决策点**：执行 agent 遇到此 commit 时，**默认选 A 跳过并记录**，除非用户/交接文档明确要 IM slash 能力。这是本 spec 唯一一个"建议跳过"的 commit，且理由是 surgical 成本 vs 价值，非"用不上就跳"。

---

## 3. 推荐合入顺序

先把 7 个 🟢 全合（零冲突、快速建立增量），再做 3 个 🟡 surgical（0fb18e36 / 167ef451；16391e35 默认跳过）：

```
批次1（🟢 全量，一起合）：8db16bb3 b62c5a7b ba9cc5e9 a57d05fe ae9e8bc0 b3c2cc42 37337b77
  → 每个文件全量 git show deerflow/main:<上游路径> > <本地路径>（非受保护确认后）
  → 跑 make test，确认无回归
批次2（🟡 surgical）：0fb18e36（机械重命名，同步 client.py + 5 测试）
  → 裸导入 + 5 测试文件
批次3（🟡 surgical）：167ef451（token_counting，memory/prompt.py + lead prompt.py head-to-head）
  → make test + memory 相关测试
批次4：16391e35 默认跳过（记录待评估）；如需则单独 surgical
```

> 也可用 `./scripts/sync-deerflow.sh --dry-run` 交叉验证脚本视角的分类（脚本只追 harness 路径，app 层需手工，见 §5）。

---

## 4. 验证（每批次后 + 全部完成后）

```bash
cd packages/agent/backend
source .venv/bin/activate

# 1) 全量测试
make test
#   注意区分 CLAUDE.md 记录的 4 个已知污染失败（deferred_tool_registry_promotion×2 +
#   inspect_gate_guardrail/paradigm_identification_gate 的 test_async_delegates_to_sync），非本次引入。

# 2) 裸导入两生产入口无环（CLAUDE.md 铁律，改受保护 agent.py/prompt.py 后必做）
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
#   两者 0 退出才算过。0fb18e36 改了 agent.py 函数名 → 这步会抓重命名没同步的 ImportError。

# 3) 顶层 ethoinsight import grep（防 sync 引入非惰性 import）
grep -rn "^from ethoinsight" packages/harness/deerflow/ | grep -v "def " || echo "无顶层 ethoinsight import ✓"

# 4) ruff
make lint
```

**0fb18e36 专项**：合后必跑 `PYTHONPATH=. python -m pytest tests/test_lead_agent_skills.py tests/test_checkpointer.py tests/test_lead_agent_model_resolution.py tests/test_client_e2e.py -q`（这 5 个文件 patch 了被重命名的符号）。

**167ef451 专项**：合后跑 `tests/test_memory_prompt_injection.py tests/test_lead_agent_prompt.py tests/test_tiktoken_cache_and_count_tokens.py`，并**人工 diff** `agents/memory/prompt.py` 确认 Noldus topOfMind/history 隔离 67 行仍在（`grep -n "topOfMind\|recentMonths\|_format_memory" agents/memory/prompt.py`）。

---

## 5. ⚠️ 受保护文件 head-to-head 强制步骤（绝不整文件覆盖）

对 2.8/2.9/2.10 涉及的每个受保护文件，**必须**：

```bash
# 看上游本次 commit 对该文件的净改动（不是全量 diff，是这次提交改了什么）
git show <commit> -- backend/packages/harness/deerflow/<file>
# 看本地与上游当前版本的全量差异（含所有 Noldus 定制）
diff <(git show deerflow/main:backend/packages/harness/deerflow/<file>) \
     packages/agent/backend/packages/harness/deerflow/<file>
```

判据：把**上游本次净改动**手工合进本地，本地所有 Noldus 定制行（中间件链顺序、中文 prompt、topOfMind 隔离、handoff_strict_mode 等）**原样不动**。

**血泪教训复核（MEMORY）**：
- `agents/memory/prompt.py` + `config/app_config.py` 在 2026-06-09 sync 被 full-follow 洗掉过定制（[[feedback_sync_nonprotected_files_with_noldus_customization_overwritten]]）。本次它们都在 🟡 名单，**必 surgical**。
- 任何受保护文件改完后跑裸导入（§4 step 2），conftest mock 会藏循环导入假绿（[[feedback_conftest_mock_hides_circular_import_verify_bare_prod_import]]）。

---

## 6. 收尾

```bash
# 1) 更新 sync 基准
echo 推进 .deerflow-sync-state 的 last_sync_commit 到 2d5f0787、last_sync_date、commits_count、prs
# 文件格式见现有 .deerflow-sync-state（保留顶部历史注释）

# 2) commit（中文 message）
git add -A
git commit -m "sync deerflow upstream f92a26d5→2d5f0787（7 安全全量 + 0fb18e36 重命名 + 167ef451 token_counting；16391e35 slash skill 暂跳过）"

# 3) push + 建 PR 到 dev
git push -u origin chore/sync-deerflow-2d5f0787
```

**交接文档**：在 `docs/handoffs/2026-06/` 记录本次 sync——尤其 **16391e35 slash skill 激活被跳过**（待评估是否需要 IM channel slash 能力），以及 0fb18e36 重命名后本地所有调用方已同步。

---

## 7. 决策点小结（供执行 agent / 用户拍板）

| commit | 默认动作 | 需用户拍板？ |
|--------|----------|--------------|
| 7 个 🟢 | 全量合 | 否，直接做 |
| `0fb18e36` 重命名 | surgical 跟随（选项 A） | 否（机械、低成本、对齐上游） |
| `167ef451` token_counting | surgical 合 | 否（高价值、离线部署相关） |
| `16391e35` slash skill | **默认跳过 + 记录** | **是** — 若团队要 IM channel slash 能力则改为 surgical 合入 |

> 本 spec 的唯一开放决策 = 16391e35 是否要。其余 10 个 commit 判断已闭合，执行 agent 可直接按 §2/§3 推进。
