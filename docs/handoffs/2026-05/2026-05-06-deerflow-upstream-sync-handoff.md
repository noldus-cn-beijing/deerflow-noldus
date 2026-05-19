# DeerFlow 上游同步交接文档

**日期**: 2026-05-06
**交接人**: Claude (本会话, opus-4-7-1m)
**接手对象**: 下一位 AI Agent / 开发者
**任务状态**: 🟡 Tier 1 完成已实施未 commit;Tier 2 受保护文件待手动合并;CLAUDE.md 已加规则

---

## 1. 当前任务目标

用户原始诉求:**「从上游拉取最新的改动代码,但是上游仓库和真正的开源上游仓库有代码冲突问题」**

具体目标:
1. 解决 `deerflow-noldus`(用户 fork)与 `bytedance/deer-flow`(真正官方上游)的代码冲突
2. 把上游 176 个新 commit 中**安全 + 高价值的部分**拉到 `noldus-insight` 主项目
3. 同时**保留所有 Noldus 定制**(prompt、subagent 名字、自定义中间件、sandbox 接口扩展等)

---

## 2. 当前进展

### ✅ 已完成

#### 2.1 修复 deerflow-noldus fork 与 upstream 的冲突 — 已 push

**位置**: `/home/wangqiuyang/deerflow-noldus`

**问题诊断**:
- fork main 上有 3 个 noldus commit (`5bfd23d5` / `82731aeb` / `7289d2bb`),都是 memory event-loop 修复
- 这些修复**已被官方上游吸收**(成为 `c0da2782`,PR #2627),并且上游合并时叠加了 PR #2153 的 `user_id` 参数(per-user filesystem isolation)
- 直接 merge 会冲突,因为 fork 是「PR 合入前中间态」,upstream 是「合入后最终版」

**已执行**:
1. 打了备份 tag `pre-reset-backup-20260506` (本地 + 已通过 push 保留远程历史)
2. `git reset --hard upstream/main` — fork main 完全对齐官方上游 4ead2c6b
3. `git push origin main --force-with-lease` — GitHub fork 已同步

**验证**: 本地 main = origin/main = upstream/main(0 commits 差)

#### 2.2 noldus-insight 选择性同步 — Tier 1 已实施未 commit

**位置**: `/home/wangqiuyang/noldus-insight`

按 Tier 1-5 分级评估了 105 个影响 harness 的 commit(详见 §3.2)。**Tier 1 中只有以下 7 个文件实际改动**(其余「想拉取的」要么 noldus 已经领先要么有 Tier 4 依赖被回滚了):

| 文件 | 上游来源 commit | 价值 |
|---|---|---|
| `agents/middlewares/loop_detection_middleware.py` | `5b633449` + `c3170f22` + `e8675f26` | per-tool-type 频率检测 + 稳定 hash + tool-call 配对保护 |
| `agents/middlewares/title_middleware.py` | `c91785dd` | 剥 `<think>` 标签(deepseek 受益) |
| `community/jina_ai/jina_client.py` | `e8572b9d` | 瞬时失败用 WARNING 不堆 traceback |
| `models/claude_provider.py` | `1f59e945` | prompt cache 断点限 4 防 API 400 |
| `models/factory.py` | `194bab46` + `616caa92` + `c99865f5` | when_thinking_disabled / 重复 reasoning_effort / openai 流式 usage |
| `models/openai_codex_provider.py` | `1b74d845` + `866d1ca4` | serialized kwargs / Codex usage metadata |
| `tests/test_loop_detection_middleware.py` | 配套上游测试(47 case 全过) |

**测试结果**: 1714 通过 / 14 skipped / 2 failed(均为 pre-existing,经 `git stash` 验证与本次同步无关)

#### 2.3 CLAUDE.md 已新增「上游同步核心规则」

[CLAUDE.md L123-L174](../../CLAUDE.md) 写入了 70 行规则:取长补短、不直接覆盖含 Noldus 定制的文件;明确列出 9 个 Tier 4 import 黑名单;血泪教训 3 条。

下次同步时这块会被直接看到,避免再踩同样的坑。

### ⚠️ 待处理

- **未 commit**: 上述 7 个文件改动**还没 commit**。状态为 working tree dirty。
- **Tier 2 受保护文件未合并**: 4 个文件(`lead_agent/agent.py`、`subagents/executor.py`、`llm_error_handling_middleware.py`、`sandbox/tools.py`)有上游高价值改动但混入 Tier 4 噪音,需要手工 surgical merge

---

## 3. 关键上下文

### 3.1 三层仓库链路

```
bytedance/deer-flow (官方开源,2.0-rc 多用户/auth/persistence 路线)
        ↓ 手动 merge upstream/main
noldus-cn-beijing/deerflow-noldus (用户 fork,本次已对齐)
        ↓ scripts/sync-deerflow.sh (subtree 选择性合入)
noldus-insight (主项目,EthoInsight v0.1 单用户研究助手)
```

### 3.2 105 个 commit 的 Tier 分级(关键决策记录)

详细评估在本次会话上下文里,精简版:

- **Tier 1 强烈拉取**: 安全修复 + event-loop 修复 + bug fix(纯文件,无依赖) — 16 commits
- **Tier 2 建议拉取**: 影响受保护文件,需手工合并 — 13 commits
- **Tier 3 可选**: 中等改动,价值/成本要权衡 — 11 commits
- **Tier 4 不要拉**: 架构级重构(persistence + multi-user + auth + skill storage 重构) — 8 个大 PR
- **Tier 5 与项目无关**: MindIE / Serper / Setup Wizard / DingTalk / Container workflow — ~10 commits

### 3.3 Tier 4 黑名单(任何 import 这些的上游文件都不能直接拉)

```python
from deerflow.runtime.user_context import ...     # per-user filesystem isolation
from deerflow.persistence import ...              # SQLAlchemy 持久化层
from deerflow.runtime.events import ...           # event store
from deerflow.runtime.checkpointer import ...     # 新 checkpointer 抽象
from deerflow.runtime.journal import ...          # auth-related journal
from deerflow.utils.time import ...               # 与 persistence 配套的 ISO8601
from deerflow.config.database_config import ...
from deerflow.config.run_events_config import ...
from deerflow.skills.storage import ...           # Tier 4 重构的 skill storage
```

EthoInsight v0.1 是**单用户研究助手**,2026-09 硬指标是把 EPM/OFT 范式跑通,引入这些只会拖慢进度。

### 3.4 sync-deerflow.sh 的盲区(必须知道)

[scripts/sync-deerflow.sh](../../scripts/sync-deerflow.sh) 把文件分类成「安全/受保护/新增」**只看本地是否改过**,**不识别间接依赖**。所以它会把以下文件标为「安全可直接合入」,但实际上拉了会炸:

| 文件 | 实际上不能直接拉的原因 |
|---|---|
| `runtime/user_context.py` | 是 Tier 4 入口,本身就是 per-user 体系 |
| `runtime/runs/manager.py` | import `runtime.runs.store.base`(Tier 4 新增) |
| `runtime/__init__.py` | re-export `checkpointer` + `store`(Tier 4) |
| `agents/memory/storage.py` | 上游版本依赖 `user_id` 参数(Tier 4) |
| `agents/memory/queue.py` | 同上 |
| `tools/builtins/setup_agent_tool.py` | import `user_context`(Tier 4) |
| `tools/builtins/present_file_tool.py` | import `user_context`(Tier 4) |
| `agents/middlewares/uploads_middleware.py` | import `user_context`(Tier 4) |
| `tools/builtins/view_image_tool.py` | 引用 sandbox/tools.py 中**还没存在**的 `resolve_and_validate_user_data_path` 函数 |
| `sandbox/local/local_sandbox.py` | 不接受 noldus 定制的 `extra_env` 参数(test_client_live 立刻报错) |
| `sandbox/local/list_dir.py` | 与 local_sandbox 配套 |
| `config/agents_api_config.py` | 上游 auth 体系 |

**如果直接跑 `./scripts/sync-deerflow.sh --auto-apply`,以上文件都会被覆盖,然后 import 链炸,然后跑不起来。**

### 3.5 Noldus 定制改动清单(任何人想 sync 都要保护的)

详细见 CLAUDE.md L130-L137,核心:

- **Prompt**: `lead_agent/prompt.py`(中文调度规则、Gate 反问、subagent 描述)
- **Subagent 名字**: `subagents/builtins/__init__.py`(注册 4 个 ethoinsight 子代理)
- **自定义中间件**: `ArchivingSummarizationMiddleware`、`ThinkTagMiddleware`、`TrainingDataMiddleware`、`GateEnforcementMiddleware` — 出现在 `lead_agent/agent.py` 中间件链
- **Sandbox 扩展**: `sandbox.py` 的 `extra_env` 参数、`local_sandbox.py` 的 venv PATH + `DEERFLOW_PATH_*`、`sandbox/tools.py` 的 `{{shared://}}` 占位符
- **Shared workspace**: `config/paths.py` 的 `/mnt/shared`、`thread_state.py` 的 `shared_path` 字段
- **错误处理增强**: `llm_error_handling_middleware.py` 的总超时 + 多种 timeout 关键字
- **MCP 截断**: `mcp/tools.py` 的 4096 字符截断
- **Subagent executor 修复**: `subagents/executor.py` 的 `recursion_limit` + `max_turns`

---

## 4. 关键发现

### 4.1 fork 上游的修复其实是冗余的

deerflow-noldus 的 3 个 commit 与上游 c0da2782 是**同一个作者(Willem Jiang)同一个 PR(#2627)的不同阶段**。fork 等于卡在 PR 中间状态,所以 reset 到上游不丢任何独有逻辑。

### 4.2 上游正在走 v2.0-rc 多用户/auth 路线

176 个 commit 中相当大比例(38 个 +)是 persistence + multi-user + auth + skill storage 重构。从 commit 时间线看,这是 bytedance 团队在做 SaaS 化。**EthoInsight 不需要,且引入会破坏 v0.1 进度**。

### 4.3 sync 脚本的「安全文件」分类不可信

(已在 §3.4 详述)。CLAUDE.md 已新增规则警告未来 agent 不要无脑跑 `--auto-apply`。

### 4.4 部分「想拉」的 commit 实际上 noldus 已经领先

第一次复制 20 个文件后,git diff 显示有 11 个文件「无差异」(noldus 已经是最新,要么之前同步过,要么定制版本恰好和上游一致)。这是好事,说明先前的同步并不全无效果。

### 4.5 测试套发现的 2 个 pre-existing 失败

```
tests/test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config
tests/test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer
```

第一个找不到 `packages/agent/extensions_config.json`(可能符号链接或路径配置问题)
第二个断言「需要 APA 格式报告」字符串在 prompt 中,实际不在(prompt 改过没同步测试)

**两个都不是本次同步引入,但建议下次单独修。** 经 `git stash` 实测拉取前后均失败。

---

## 5. 未完成事项

### P0 — 阻塞下一步

- [ ] **commit 当前 7 个文件改动**。建议 commit message:
  ```
  sync deerflow upstream: 拉取上游 Tier 1 安全修复和 bug fix

  - loop_detection: per-tool-type 频率检测 + 稳定 hash + tool-call 配对保护
  - title_middleware: 剥 <think> 标签
  - jina_client: 瞬时失败 WARNING 不堆 traceback
  - claude_provider: prompt cache 断点限 4 (1f59e945)
  - models/factory: when_thinking_disabled / 重复 reasoning_effort / openai 流式 usage
  - openai_codex_provider: serialized kwargs / Codex usage metadata
  - test_loop_detection: 47 case 配套上游版本(含 noldus 自己加的稳定 hash 测试)

  跳过 Tier 4 (persistence + multi-user + auth + skill storage 重构),
  保留所有 noldus 定制。详见 docs/handoffs/2026-05-06-deerflow-upstream-sync-handoff.md
  ```

### P1 — Tier 2 手工合并(高价值,但需仔细)

按价值/难度排序:

1. **`agents/middlewares/llm_error_handling_middleware.py`** (95 行 diff,最简单)
   - 上游加: `httpx.ReadError` + `RemoteProtocolError` 识别(PR #2309 / #2095)
   - noldus 已有: 总超时 `retry_total_timeout_s = 180.0`、多种 timeout 关键字、`get_app_config()` fallback
   - **合并方法**: 把 `"ReadError"` 和 `"RemoteProtocolError"` 加到 noldus 的 retriable error class 列表中,**保持 noldus 其他改动不动**。两边改动正交,易合并。

2. **`subagents/executor.py`** (455 行 diff)
   - 上游有: `e5b14906` (event-loop conflict fix) + `7dea1666` (avoid temp event loops) + `83938cf3` (user context propagate)
   - noldus 有: `recursion_limit` 修复 + `max_turns` 硬限制 + `{{shared://}}` 占位符
   - **合并方法**: 重点提取 e5b14906 + 7dea1666 的 event-loop 部分(关键代码块),跳过 user context 部分(Tier 4)。仔细 diff,这个文件最容易出 bug。

3. **`agents/lead_agent/agent.py`** (257 行 diff)
   - 上游大多是 `app_config` 显式传参重构(Tier 4 PR #2666)+ `BeforeSummarizationHook` 概念(配套 Tier 4 memory flush)
   - noldus 有: 自定义中间件链(`ArchivingSummarizationMiddleware`、`ThinkTagMiddleware`、`TrainingDataMiddleware`、`LoopDetectionMiddleware._loop_detection` 注入)
   - **合并方法**: **建议跳过**。上游改动几乎全是 Tier 4 噪音,价值低,风险高。等 noldus 自己改这个文件时顺手参考。

4. **`sandbox/tools.py`** (422 行 diff)
   - 上游有: `6bd88fe1` (bash traversal 防御 — 大量常量定义)+ `af8c0cfb` (view_image 路径限制 — `resolve_and_validate_user_data_path`)
   - noldus 有: `{{shared://}}` 占位符解析、`shared:// → /mnt/shared` 路径映射、定制 mask
   - **合并方法**: 把上游的 `_URL_*_PATTERN`、`_LOCAL_BASH_*` 常量、`resolve_and_validate_user_data_path` 函数 patch 进 noldus 版本。注意上游引入了 `from deerflow.runtime.user_context import get_effective_user_id`,**这一行不能合**(Tier 4),要把对应代码改成 noldus 现有的 user_id 处理方式。

### P2 — 跳过的 commit 中可能值得回头看的

- `c0da2782` memory event-loop fix — 已被上游吸收的 noldus 自己 PR,但上游版本带 `user_id` 参数(Tier 4)。noldus 的 `agents/memory/updater.py` 是早期版本,可能也需要 event-loop fix,但 surgical merge 难度高
- `87609374` memory file I/O 用 `asyncio.to_thread` — 同上
- `35f141fc` checkpoint rollback on cancel — 用户体验改善,但依赖 `runtime/runs/worker.py` 的 store(Tier 4)

### P3 — 长期事项

- [ ] 修两个 pre-existing 测试失败(见 §4.5)
- [ ] 把 sync-deerflow.sh 升级:加上「Tier 4 import 检测」,自动把 import `runtime.user_context` 等的文件归到「需人工判断」,不再误导
- [ ] 等 v0.1 发布后,评估是否要采纳上游的 multi-user 体系(可能用作 SaaS 化基础)

---

## 6. 建议接手路径

### 6.1 验证当前状态干净

```bash
cd /home/wangqiuyang/noldus-insight
git status -s | grep -v "^??"
# 应该看到 7 个文件 modified
```

期望输出:
```
 M packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py
 M packages/agent/backend/packages/harness/deerflow/agents/middlewares/title_middleware.py
 M packages/agent/backend/packages/harness/deerflow/community/jina_ai/jina_client.py
 M packages/agent/backend/packages/harness/deerflow/models/claude_provider.py
 M packages/agent/backend/packages/harness/deerflow/models/factory.py
 M packages/agent/backend/packages/harness/deerflow/models/openai_codex_provider.py
 M packages/agent/backend/tests/test_loop_detection_middleware.py
```

### 6.2 跑测试再确认

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, 1714 passed, 14 skipped`(2 个 pre-existing,见 §4.5)

### 6.3 commit

按 §5.P0 的 commit message 提交。**不要** push 到远程,等用户决定。

### 6.4 (可选) 开始 P1.1 — `llm_error_handling_middleware.py` 合并

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py) \
     packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py | less
```

只把 `"ReadError"` 和 `"RemoteProtocolError"` 加到 retriable error class 列表中,跑 `make test` 验证。

### 6.5 如果接手者决定不继续 Tier 2

把 §5.P1 内容存档为 GitHub issue,标记为「上游同步 backlog」,以备后续。

---

## 7. 风险与注意事项

### ⚠️ 不要直接覆盖任何含 Noldus 定制的文件

详见 [CLAUDE.md L123-L174](../../CLAUDE.md) 新增的「上游同步核心规则」。

### ⚠️ 不要无脑跑 `./scripts/sync-deerflow.sh --auto-apply`

脚本的「安全文件」分类不识别**间接 Tier 4 依赖**。详见 §3.4。

### ⚠️ Tier 4 体系不是 bug,是上游的有意设计

上游正在走 v2.0-rc 多用户 SaaS 路线(persistence + auth + per-user FS isolation)。Noldus 不要这条路,**不是因为它坏,是因为目标不一致**。如果未来 EthoInsight 要做 SaaS 化,届时再大批量合入。

### ⚠️ deerflow-noldus 已 force-push,远程历史已变

`pre-reset-backup-20260506` tag 还在本地,但远程 main 是新的。**任何之前 clone 过 deerflow-noldus 的人都需要 `git fetch && git reset --hard origin/main`**。

### ⚠️ 当前 7 个文件未 commit,任何 reset/checkout 都会丢失

如果暂不 commit,记得 `git stash` 保存。

### ✅ CLAUDE.md 已加规则,未来 agent 会看到

下次任何 AI agent 做上游同步,会被那 70 行规则提醒「不要直接覆盖含 Noldus 定制的文件」。

---

## 8. 下一位 Agent 的第一步建议

1. **读这份 handoff** ✅(你正在做)
2. **读 CLAUDE.md L123-L174 的上游同步规则** — 这是本次会话提炼的关键经验
3. **跑 §6.1 + §6.2 验证当前状态**
4. **如果验证通过 + 用户同意 → §6.3 commit**
5. **如果用户想推进 Tier 2 → §6.4 从最简单的 llm_error_handling_middleware 入手**
6. **不要轻易动 Tier 4 文件**(persistence、user_context、skill storage 等)

---

## 9. 附录:关键资源链接

| 资源 | 路径 |
|---|---|
| 本次新增的规则 | [CLAUDE.md L123-L174](../../CLAUDE.md) |
| 上一份交接(前端打磨 + 子任务 bug) | [docs/handoffs/2026-05-06-frontend-polish-and-subtask-bug-handoff.md](2026-05-06-frontend-polish-and-subtask-bug-handoff.md) |
| 同步脚本 | [scripts/sync-deerflow.sh](../../scripts/sync-deerflow.sh) |
| 同步 SOP | [docs/sop/deerflow-sync-sop.md](../sop/deerflow-sync-sop.md) |
| deerflow-noldus fork(已对齐 upstream) | `/home/wangqiuyang/deerflow-noldus`,tag `pre-reset-backup-20260506` 是 reset 前备份 |
| 受保护文件清单 | sync-deerflow.sh L33-L52(`PROTECTED_FILES` 数组) |
| 上游 commit 范围 | `f0dd8cb..deerflow/main`(176 个 commit,其中 105 个影响 harness) |
