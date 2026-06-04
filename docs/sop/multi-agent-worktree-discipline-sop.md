# 多 Agent 并发 + Worktree 纪律 SOP

**适用场景**：多个 Claude Code agent 并发开发同一个项目（共享主仓 + 各自 worktree）。
**来源**：2026-05-29 一次真实事故——主仓 dev 上一条 shoaling-removal 的 `git stash pop` 冲突未解决，卡住了所有想在主仓提交的 agent；同时 `git worktree list` 堆到 8 个（多为已合并未清理的残留）。
**核心原则**：**主仓 dev 是「只进不出的干净集成点」，所有实质开发在 worktree 隔离，提交走分支 → PR。**

---

## 1. 主仓 dev 绝不 stash / pop

**事故根因**：有人在主仓 `dev` 上 `git stash pop` 冲突了没收拾。stash 是**无主、不显式**的——它不在任何分支历史里，`git status` 不会主动提醒，pop 冲突后会留下 `UU` 文件**卡死后续所有 commit**（即使你只想提交无关的 docs）。

**规则**：
- ❌ 主仓 dev 上**绝不** `git stash`。需要临时挪开改动？用 worktree 或备份到 `/tmp`。
- ✅ 主仓 dev 只用来：`git pull` 拉取 + 提交 docs/小改 + 集成。
- ✅ 若发现主仓有残留 stash（`git stash list` 非空且不是你刚存的），**先查清是谁的、内容是什么**（`git stash show -p stash@{N}`），确认可丢弃才 `git stash drop`；不确定就保留并报告，绝不替别人解决。

---

## 2. Worktree 用完即删，不堆积

**事故现象**：8 个 worktree，其中 6 个分支早已合并 dev 但 worktree 没删。worktree 越多 → 共享主仓被污染概率越大 + 自己记不清哪个在跑。

**规则**：
- ✅ 一个 sprint/PR **合并后立刻** `git worktree remove <path>` + 删对应分支 `git branch -D <branch>`。
- ✅ 定期 `git worktree list` 自检：列表应只剩「当前真正在跑的 agent」+ 主仓。
- ✅ 删之前**两个安全检查**（缺一不可）：
  1. 分支已合并 dev：`git merge-base --is-ancestor <branch> origin/dev`（成功=已合，可删）
  2. 工作区干净：`git -C <path> status --short`（空=干净，可删）
- ❌ **绝不删**未合并 / 进行中（有未提交改动）/ 别的 agent 正在跑的 worktree。

---

## 3. 每个并发 agent 严格隔离，提交走分支 → PR，绝不直推主仓 dev

**规则**：
- ✅ 每个 agent 在自己的 worktree（`EnterWorktree` 或 `git worktree add`）独立分支开发。
- ✅ 提交 → push 自己分支 → 开 PR → merge 到 dev。**主仓 dev 永远是集成结果，不是开发现场。**
- ❌ agent 永远不在主仓 dev 直接改业务代码 / 提交。
- ✅ 派并发 agent 前，先核实它们的**文件改动面是否相交**（head-to-head：`git diff --stat` 各自分支）。相交的（如都改 `lead_agent/prompt.py` / `data_analyst.py`）必须约定谁先合、后者 rebase；不相交的可全速并行。
- ✅ 已知热点文件（多 sprint 共改）：`agents/lead_agent/prompt.py`、`subagents/builtins/data_analyst.py`。改它们加 workflow step 后必须 `grep` 编号唯一性（防 5.7 seal bug 复发）。

---

## 4. 残留 stash 当天处理，不留「WIP 挂着」

**规则**：
- ✅ stash 当天消费（pop 解决）或显式丢弃（`git stash drop`）。
- ✅ stash message 写清来源（`git stash push -m "WIP: <什么> for <哪条线>"`），便于后人判断。
- ❌ 不留无说明、跨多次会话的 stash——它们会在某次 pop 时炸，且没人敢动。

---

## 5. 提交前先 `git status` 看全貌，发现非自己的脏状态就停手报告

**这是 agent 的硬纪律**（2026-05-29 事故中正确执行的一条）：
- ✅ 在共享工作区提交前，**先 `git status`** 看全貌。
- ✅ 任何「不是我造成的未完成状态」——`UU`（未合并）/ stash 残留 / detached HEAD / 别的分支的未提交改动——**停手，报告给用户，不擅自解决别人的活**。
- ✅ 自己的改动先备份到 `/tmp`（纯拷贝，零风险），再做任何 git 操作。
- ❌ 不在不了解来龙去脉时解决冲突 / drop stash / 删分支——可能毁掉别人正在做的工作。

**呼应项目根原则**：「对你不了解、不是你创建的东西，先surface再动，不直接覆盖/删除」。

---

## 速查清单（agent 提交/清理前过一遍）

```
提交前：
  □ git status —— 有非我造成的 UU / stash 残留 / detached？→ 停手报告
  □ 我的改动已备份 /tmp？
  □ 显式 git add <我的文件>，确认无 .env* 等敏感文件进暂存

清理 worktree 前（每个都查）：
  □ git merge-base --is-ancestor <br> origin/dev —— 已合并？
  □ git -C <path> status --short —— 工作区干净？
  □ 不是别的 agent 正在跑的？
  三者皆是 → 可 remove + branch -D

stash：
  □ git stash list 非空且非我刚存 → 查清来源，不确定就保留报告
```
