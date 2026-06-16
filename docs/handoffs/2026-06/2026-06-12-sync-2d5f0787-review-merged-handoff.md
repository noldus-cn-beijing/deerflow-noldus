# Handoff: DeerFlow sync 2d5f0787 review + merge（PR #124 已合 dev）

> 交接对象：下一个接手的 AI Agent
> 日期：2026-06-12
> 上一会话产出：review `chore/sync-deerflow-2d5f0787` → 发现并修复 3 处缺陷 → PR #124 已 merge 进 dev

---

## 1. 当前任务目标（已完成）

用户要求 review sync worktree `origin/chore/sync-deerflow-2d5f0787`（DeerFlow 上游同步 f92a26d5→2d5f0787，9 commit：7 全量 + 2 surgical + 1 跳过），"有问题直接在 worktree 上改"。

**最终状态：review 完成、3 缺陷已修、PR #124 已 merge 进 dev。本任务闭环。** 下一 Agent 只需处理 2 个低优先级收尾项（见 §5）。

---

## 2. 当前进展 ✅

- ✅ **完整 review 了 sync commit `784f86ce`**：手工核对 + 5 维度对抗式 workflow（10 agent，每发现双重验证）
- ✅ **4 个维度核实干净**（无需改动）：
  - 中间件链承重墙：`_build_middlewares→build_middlewares` 重命名仅改名，30+ append 序列**字节一致**
  - `memory/prompt.py` 的 topOfMind/history 隔离 + token_counting 特性：字节一致保住/搬全
  - StepFun 适配器（`patched_stepfun.py`）：字节一致、模式一致、16 测试绿
  - `app_config.py` null-coercion（修的正是 `models: null` 崩溃类）：忠实、保 Noldus 定制
  - skip `16391e35`（slash skill）：核实干净可分离、无悬空依赖、无半应用
- ✅ **发现并修复 3 缺陷**（commit `3775d1c6`，已在 dev）：见 §4
- ✅ **全量测试基线钉死**：6 failed / 4783 passed，6 个全是**预存在测试债**（4 个本 sync 未碰 + 3 sandbox 字节一致分歧 + 2 async_delegates 单跑绿=isolation 污染），**无新增回归**
- ✅ **生产入口裸导入过**：`import app.gateway` + `from deerflow.agents import make_lead_agent`
- ✅ **PR #124 merge 进 dev**（2026-06-12 02:33，merge commit `90223f68`）

---

## 3. 关键上下文

- **仓库**：`/home/wangqiuyang/noldus-insight`（subtree fork，不是普通 git repo——`packages/agent/` 是 DeerFlow 上游）
- **当前分支/HEAD**：`chore/sync-deerflow-2d5f0787` @ `90223f68`（= `origin/dev`，已 fast-forward）
- **dev = `90223f68`**：已含 sync `784f86ce` + review-fix `3775d1c6` + merge commit
- **后端测试 cwd 必须是** `/home/wangqiuyang/noldus-insight/packages/agent/backend`，用 `PYTHONPATH=. .venv/bin/python -m pytest ...`（无 `python`，只有 venv 内的）
- **运行时形态**：Gateway-embedded（`make dev` 经 `serve.sh`，无独立 langgraph 进程），Gateway 进程 `cd backend` 启动
- **上游路径前缀差异**：上游树用 `backend/...`，本仓库用 `packages/agent/backend/...`（做三方 diff 时注意）

---

## 4. 本次修复的 3 缺陷（已在 dev，仅供理解，无需重做）

1. **`memory/prompt.py` 受保护文件回植退化（medium）** — 上游 167ef451 合入时把 BASE 的强约束伞句 `MUST NOT be written into any memory field (workContext, personalContext, topOfMind, recentMonths, earlierContext, longTermBackground, or facts)` 降级成上游窄句（只管 upload）+ 弱 `Additionally` 子级嵌套，并引入 `(facts or facts)` typo。**12 条 CJK 示例串字节一致没丢→"grep 串还在"会假绿放行**。修复=恢复 BASE 原句、去嵌套、消 typo，块现与 BASE 字节一致；token_counting 特性不动。
2. **`test_stateless_runs_owner_isolation.py` 8 个安全测试 hollow-red** — fixture 只设全局 `app_config` 单例，没设 request-scoped `app.state.config`→`start_run` 经 `get_run_context→get_config` 全部 **503 在 owner check 之前**→测试从不真正校验跨用户隔离。补 `app.state.config` 后 8 测试真实通过（404/409/internal-bypass）。
3. **`0fb18e36` 重命名漏改（low）** — `_build_middlewares→build_middlewares` 漏改 `scripts/tool-error-degradation-detection.sh`（line 38/191，跑会 `ImportError`）+ `backend/CLAUDE.md:407` + `docs/plan_mode_usage.md`。补全。`executor._build_middlewares`（不同符号）+ `rfc-create-deerflow-agent.md`（上游有意保留旧名）不动。

> 教训已落 memory：`feedback_sync_protected_file_paraphrase_merge_weakens_constitution.md`（受保护 prompt 文件 review 必须三方字节对账块，不能只 grep 串）。

---

## 5. 未完成事项（按优先级）

### 🟡 P1 — 回填 `.deerflow-sync-state` 的 PR 号
`/home/wangqiuyang/noldus-insight/.deerflow-sync-state` 第 15 行仍是占位：
```
last_sync_prs: "待填实际 PR 号 (9 commit sync: 7 全量接受 + 2 surgical; 16391e35 slash skill 跳过待独立评估)"
```
现已知是 **PR #124**。改成实际号（如 `last_sync_prs: "#124"`），commit 进 dev。否则下次 sync 脚本记录留 TODO。

### 🟡 P2 — 为跳过的 `16391e35`（slash skill activation）建独立 issue
本次**故意跳过**且核实干净可分离，但：
- 上游标题是 `fix(skills): harden slash skill activation across chat channels (#3466)`（上游归类为 fix，但它 harden 的是 Noldus 从未有过的 slash-skill 子系统，所以跳过不会重开任何 Noldus 回归）
- **它在 Noldus 的 divergent lead_agent base 上不能直接 cherry-pick**——假设了上游 `_make_lead_agent` / `app_config`-threaded `build_middlewares` 重构，Noldus 是 `make_lead_agent`（无下划线）。未来要捡起来需 **surgical re-port**，不是直接 apply。
- 建议建 issue 挂着免遗忘，记录"需 surgical re-port"这点。

### 🟢 P3（可选）— 3 个 sandbox mount 测试债
`test_local_sandbox_provider_mounts.py` 的 3 个失败是**预存在的 Noldus string-vs-list 命令分歧**（已证 BASE/MERGED 字节一致，非本 sync 引入）：
- `test_execute_command_path_replacement`（上游断言 list-command `["/bin/sh","-c",...]`，Noldus 用 string + `shell=True`）
- `test_list_dir_..._symlink_like_directory`（上游期望目录尾斜杠，Noldus 无）
- `test_rejects_path_outside_virtual_prefix_and_logs_error`（上游期望日志 `"outside allowed directory"`，Noldus 措辞不同）

不阻塞任何东西，是独立测试卫生债。若要清，应**改测试适配 Noldus 实现 shape**（不是改实现）。与本 sync 无关，可单独处理。

---

## 6. 风险与注意事项

- **别把"6 failed"当回归**：全量 `make test` 这 6 个红是确定性的预存在债——3 sandbox 分歧 + 1 chart_maker_config 既定 + 2 `test_async_delegates_to_sync`（inspect_gate / paradigm_identification，**单跑绿、全量跑红=test isolation 污染**，见 memory `feedback_known_full_suite_test_pollution_4_tests.md`）。看到这 6 红别归因自己改坏。
- **受保护文件 review 别只 grep 关键串**：本次最大发现就是"CJK 串全在但约束被悄悄削弱"的假绿。三方字节对账块（`diff <(git show BASE:file | sed -n '/块起/,/块止/p') <(同范围 MERGED)`）才靠谱。
- **改 harness 核心后必跑裸导入**：`PYTHONPATH=. .venv/bin/python -c "import app.gateway"` + `... "from deerflow.agents import make_lead_agent"`（conftest mock 会藏循环导入，pytest 假绿）。
- **worktree 卫生**：`git worktree list` 里那个 `.claude/worktrees/chore+sync-deerflow-2d5f0787` 指向**别的分支**（`worktree-chore+...` @ `529e5c2f`，与 pr115-stage4 共用），**不是**本次 sync 的实施 worktree（实施在主 checkout）。是历史残留，可单独清理。

---

## 7. 下一位 Agent 的第一步建议

如果用户让你继续收尾：

1. **回填 sync-state（P1，2 分钟）**：
   ```bash
   cd /home/wangqiuyang/noldus-insight
   # 编辑 .deerflow-sync-state 第15行 last_sync_prs 改为 "#124"
   git add .deerflow-sync-state && git commit -m "chore: 回填 sync 2d5f0787 的 PR 号 #124" && git push
   ```
   注意：当前在 `chore/sync-deerflow-2d5f0787` 分支但它已等于 dev。先确认是否该直接在 dev 上提（`git checkout dev` 或新建小分支）——按项目规范"所有 commit 先进 dev"，可直接 commit 到 dev 再 push。

2. **建 16391e35 issue（P2）**：`gh issue create`，标题如"slash skill activation (上游 16391e35) 需 surgical re-port 到 Noldus divergent lead_agent base"，正文引用 §5-P2。

3. **若用户问"sync 还有没有问题"**：答已 review 完、3 缺陷已修已合（PR #124），剩 2 个低优先级收尾（PR 号回填 + slash skill issue），无功能阻塞。

如果用户开新任务：本 sync 已闭环，无需回头看，直接进新任务即可。

---

## 关键文件路径速查

- sync 实施 spec：`docs/superpowers/specs/2026-06-11-deerflow-sync-f92a26d5-to-2d5f0787-spec.md`
- sync 实施 handoff（11日，前序）：`docs/handoffs/2026-06/2026-06-11-deerflow-sync-f92a26d5-to-2d5f0787-handoff.md`
- sync 状态文件（待回填）：`.deerflow-sync-state:15`
- 本次修的 3 文件：`packages/agent/backend/packages/harness/deerflow/agents/memory/prompt.py`、`packages/agent/backend/tests/test_stateless_runs_owner_isolation.py`、`packages/agent/scripts/tool-error-degradation-detection.sh`（+ CLAUDE.md / plan_mode_usage.md 文档）
- review 教训 memory：`~/.claude/.../memory/feedback_sync_protected_file_paraphrase_merge_weakens_constitution.md`
