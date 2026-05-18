# 2026-05-14 E2E 测试问题：「补充轨迹图和汇总表格」阶段卡死

> **目的**：把一份独立、可验证的诊断材料交给另一位 agent 做 co-analysis（共审根因 + 共定修复方向）。我已经做了 systematic-debugging 的 Phase 1（root cause）和 Phase 2（pattern），还没动代码。
>
> **请你做的事**：
> 1. 不要绕过我描述的根因直接修代码 — 我要的是**独立的根因复核**
> 2. 用本文件列出的"证据指针"自己 grep / Read 一遍源头，确认我没看错
> 3. 给出你独立的根因判定：同意 / 部分同意 / 反对 + 理由
> 4. 给出你独立的修复优先级建议（prompt 修 vs sandbox 修 vs spec 修）
> 5. 指出我**遗漏的根因**（如果有）
>
> **不要做的事**：
> - 不要动代码（用户没授权）
> - 不要 commit 任何东西
> - 不要重新跑 dogfood 测试（数据已经在文件系统里）
> - 不要把这份文档当 plan 来执行

---

## 0. 上下文（30 秒了解项目）

- 项目：EthoInsight（行为学 AI 分析助手）位于 `~/noldus-insight/`
- 后端：基于 DeerFlow（LangGraph fork），lead_agent + 4 个 ethoinsight subagent
- 此前会话产物：handoff 双层协议 spec（draft）+ dogfood-fix-plan 全部 11 个 commit 已被另一 agent 跑完
- 用户正在做"端到端测试 dogfood 修复"——这次测试就是验证那 11 个 commit 是否真的解决了上一次 thread 5046a6e6 暴露的架构问题
- 本次失败的 thread：`b0d3a611-071e-41a5-a952-36c3772c167f`

强烈建议先快速读一下：
- `~/noldus-insight/CLAUDE.md`（项目总览，含「常用命令 / 范式重构现状 / Tier 4 已合入」三条关键信号）
- `~/noldus-insight/docs/handoffs/2026-05/2026-05-14-spec-handoff-protocol-handoff.md`（上轮会话交接 — 解释为什么这次端到端测试很重要）

---

## 1. 用户做了什么

打开 `http://localhost:2026`，新建 thread，上传：

```
/mnt/user-data/uploads/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt
```

（EthoVision XT 19 导出的小鼠 EPM 单只轨迹，UTF-16-LE 编码，注意文件名里有多个连续空格）

对话路径（来自 `~/noldus-insight/docs/e2e/小鼠高架迷宫数据分析.json`）：

| 消息 idx | 角色 | 内容摘要 |
|---|---|---|
| 0 | human | "我刚做完小鼠高架迷宫的实验，你可以帮我分析一下数据吗" |
| 1-4 | ai/tool | lead 走 Gate 1 反问 |
| 5 | human | 上传数据 + "这是数据" |
| 6-9 | ai/tool | lead 第二次 ask_clarification（问组别 / 是否要洞察 / 是单只还是多只） |
| 10 | human | "只有 Subject 1 的数据，先做单样本分析（轨迹图 + 开臂时间等）" |
| 11-50 | ai/tool | lead 派 code-executor → data-analyst → report-writer，输出五个核心指标（开臂时间 23.56s / 开臂时间比率 7.99% / ...），**但没出轨迹图、没出汇总表格** |
| 51 | human | **"需要！补充轨迹图和汇总表格图表"** ← 触发点 |
| 52-87 | ai/tool | **故障区** — lead 折腾 18 轮（read_file → bash → write_file → bash → str_replace → bash），4 次 sandbox 错误、2 次 FileNotFoundError、归档 7 次，**最终 outputs/ 目录至今为空** |

---

## 2. 直接观察到的症状（5 条事实）

### 症状 1：lead 第一次 bash 调用就被 sandbox 拒绝
- e2e JSON msg 55：`Error: Unsafe absolute paths in command: //5000. Use paths under /mnt/user-data`
- msg 54 的 bash 命令是 `python3 << 'PYEOF' ...` 的 heredoc，里面有 Python 整除 `n // 5000`

### 症状 2：第二次 bash 同样被拒
- msg 57：`Error: Unsafe absolute paths in command: //. Use paths under /mnt/user-data`
- lead 在 thinking 里自己也认了"路径检查误触发"（msg 56 thinking）

### 症状 3：lead 改用 write_file 写脚本（核心转向点）
- msg 58：lead 写了 `/mnt/user-data/workspace/gen_charts.py`
- msg 60：`python3 /mnt/user-data/workspace/gen_charts.py` 触发 `FileNotFoundError`（msg 61，是 `pd.read_csv` 找不到文件）

### 症状 4：诡异的"1-liner 能找文件，脚本运行不能"
- msg 64：`python3 -c "import glob; ..."` 1-liner 在 bash 里**能**找到文件
- msg 74：`python3 -c "import os; print(os.listdir(...))"` 1-liner 也**能**找到文件
- msg 75 输出：`Files: ['轨迹-Elevated...txt']` 字面证据
- msg 82-83：作为脚本文件 `python3 gen_charts.py` 跑就 `FileNotFoundError: '/mnt/user-data/uploads'`
- msg 86-87：再跑一次 `python3 /mnt/user-data/workspace/gen_charts.py` 还是 `FileNotFoundError: '/mnt/user-data/uploads'`

### 症状 5：上下文反复被压缩，token 大量浪费
- langgraph.log 里在故障窗口内 `ArchivingSummarizationMiddleware.before_model` 触发了 7 次（行 599、622、646、669、695、722）
- 每次归档 7-8 条消息，写入 `conversation_summary.md`
- 这段子任务最终一张图都没生成

---

## 3. 我推断的根因（请你独立复核）

### 根因 A：lead 违反职责边界，绕过 code-executor 自己写脚本

**直接证据**（e2e JSON msg 52 的 thinking 字段）：
> "The user wants the trajectory plot and summary table chart. **Since these weren't in the plan, I can generate them with a simple Python script. Let me do this directly rather than re-running the full code-executor pipeline.**"

**对照事实**：
- `~/noldus-insight/packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/b0d3a611-071e-41a5-a952-36c3772c167f/user-data/workspace/handoff_code_executor.json` 里的 `errors` 字段已经明确告诉 lead 该怎么走：
  ```json
  "errors": [
    "用户请求的轨迹图（trajectory plot）未在 plan.charts 中，已跳过；如需生成请通过 lead 重新规划",
    "用户请求的指标汇总表格未在 plan 中，已跳过；如需生成请通过 lead 重新规划"
  ]
  ```
- 这是 code-executor 在第一轮 plan 时就识别出来的：用户首轮要的"轨迹图"不在 `metric_plan.charts`，所以它跳过了并写下提示让 lead 重新规划。
- 但 lead 没有按这条提示走「update metric_plan.json → 重派 code-executor」，而是 `read_file → 自己写 Python 脚本 → 自己跑`。

**关联**：
- 这正是 PR #7 反馈 Gap #4「编写代码耗时过久」/「lead 自己 write_file 写脚本的反模式」
- dogfood-fix Issue #2 commit `24715250` 在 lead prompt 里加了「禁判读/品系/常模/金标准」**但没禁"自己写代码做指标外的图表"**
- 也就是说 dogfood-fix Issue #2 **修复不充分**，本次端到端测试是把这个"未修补的角落"暴露出来的第一手证据

**请你复核**：
- 看 `~/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（或本地化的 prompt）里"角色边界"那段，看它是否真的没禁止 lead 自己写分析脚本
- 看 dogfood-fix-plan Task 2 的实际 diff（commit `24715250`）确认现状

### 根因 B：sandbox 路径替换只对 bash 命令字符串生效，不对脚本文件内容生效

**机制**（来自 `~/noldus-insight/packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`）：

| 入口 | 行为 |
|---|---|
| `bash` tool 收到命令字符串 | `validate_local_bash_command_paths` 拦校验（line 917）→ `replace_virtual_paths_in_command` 替换（line 959）→ 实际执行（line 1273）|
| `write_file` tool 写文件内容 | **不做路径替换**——content 字符串直接落盘 |

所以：
- `bash python3 -c "print(os.listdir('/mnt/user-data/uploads'))"` 工作 ✅
  - 因为 `/mnt/user-data/uploads` 是 bash 命令字符串里的字面量，会被 `replace_virtual_paths_in_command` 替换成 `~/noldus-insight/packages/agent/backend/.deer-flow/users/cd95effa.../threads/b0d3a611.../user-data/uploads/`
- `bash python3 /mnt/user-data/workspace/gen_charts.py` 不工作 ❌
  - 命令字符串里只有脚本路径会被替换；脚本**文件内容**里的 `upload_dir = '/mnt/user-data/uploads'` 是文件落盘时的字符串，Python 进程实际去访问的是字面意义的 `/mnt/user-data/uploads`，宿主机上根本没这个目录

**唯一逃生口**：`_build_path_env` (line 507-520) 注入了 `DEERFLOW_PATH_MNT_USER_DATA_UPLOADS` 等环境变量，脚本要主动读 `os.environ['DEERFLOW_PATH_...']` 才能拿到真实路径。但**这套约定 lead 不知道**——lead prompt / ethoinsight-code skill / code-executor 上下文里都没有显式告诉它"写到文件里的脚本必须从 env var 读路径"。

**证据**：
- `gen_charts.py` 文件内容（在 `~/.../workspace/gen_charts.py`）开头几行：
  ```python
  upload_dir = '/mnt/user-data/uploads'
  all_files = os.listdir(upload_dir)
  ```
- 显式硬编码字面量，没用 env var 兜底

**请你复核**：
- 读 `sandbox/tools.py:507-520`（`_build_path_env`）和 `sandbox/tools.py:436-499`（`replace_virtual_path` 系列）确认我理解正确
- 检查 `~/noldus-insight/packages/agent/skills/custom/ethoinsight-code/`（如果存在）有没有提示"脚本里用 DEERFLOW_PATH_*"
- 想想这个语义差是设计意图还是疏漏 —— bash 命令字符串能替换 vs 脚本文件不能替换，是有意区分还是漏洞？

### 触发点：误报让 lead 从安全的 1-liner 转向有 bug 的脚本文件

**为什么 lead 第一次用 heredoc 而不是 write_file？** 看 msg 54 的 thinking——他原本想用 `python3 << 'PYEOF' ... PYEOF` 一次跑完。如果不被误判 `Unsafe paths: //`，他就不会被逼到 write_file 路径，根因 B 就不会触发。

**误报来源**：`sandbox/tools.py:840-854`（`_validate_local_bash_root_path_args`）
- `_LOCAL_BASH_ROOT_PATH_COMMANDS = {"awk","cat","cp","du","find","grep","head","less","ln","ls","more","mv","rm","sed","tail","tar"}`
- python3 不在这个集合里，但它扫描 token 时会把 heredoc 里的 `n // 5000`（即两个 `/` 相邻）当成"裸 `/`"参数误判
- 第二次报错信息 `Unsafe absolute paths in command: //` 印证了它确实把 `//` 当成了路径
- 这是**误报**——`//5000` 是 Python 整除，不是任何文件路径

**所以两个独立 bug 的链式叠加是这样的**：
```
误报（路径校验过严）
  → 逼 lead 从 heredoc 转 write_file
    → 触发"脚本文件路径不被替换"的 bug
      → 反复 FileNotFoundError
        → lead 不理解为什么 1-liner 行而文件不行
          → 反复打 ls / xxd / glob / os.listdir 各种猜测，token 烧光
            → ArchivingSummarizationMiddleware 归档 7 次，关键上下文被压缩
              → 直至卡死
```

请你确认这个因果链是否成立。

---

## 4. 关键证据指针（可独立 grep / Read 验证）

| 类别 | 位置 | 你需要看什么 |
|---|---|---|
| 前端 e2e 完整记录 | `~/noldus-insight/docs/e2e/小鼠高架迷宫数据分析.json` | msg 52 thinking（lead 决定自己写脚本的原话）、msg 55/57/61/83/87（4 次错误的字面文本） |
| 后端 thread workspace | `~/noldus-insight/packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/b0d3a611-071e-41a5-a952-36c3772c167f/user-data/workspace/` | `handoff_code_executor.json`（errors 字段）、`gen_charts.py`（硬编码 /mnt 字面量）、`metric_plan.json`（看 plan.charts 是否真的没列轨迹图） |
| outputs 目录（应该有图但没有）| `~/noldus-insight/packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/b0d3a611-071e-41a5-a952-36c3772c167f/user-data/outputs/` | 应为空 — 确认确实没产出任何图 |
| Langgraph 服务端日志 | `~/noldus-insight/packages/agent/logs/langgraph.log` | grep `b0d3a611` 看 sandbox audit verdict、Archived messages 次数 |
| Sandbox 路径系统源码 | `~/noldus-insight/packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` | 行 38-58（`_LOCAL_BASH_ROOT_PATH_COMMANDS`）、436-499（`replace_virtual_path`）、507-520（`_build_path_env`）、840-854（`_validate_local_bash_root_path_args`）、917-1004（bash 命令处理流程）、1269-1273（实际调用顺序） |
| lead 当前 prompt | `~/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 找"角色边界 / 禁止 / role / boundary"看是否真的没禁止 lead 自己写分析脚本 |
| dogfood-fix Issue #2 修复 | `git show 24715250` （在 `~/noldus-insight/` 仓库里）| 看 commit 实际 diff，对照是否覆盖了"自己写代码做指标外的图表"|
| Spec 草案 | `~/noldus-insight/docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` | 第 2 章双层 handoff、第 5 章 output-constitution、第 6 章 catalog modifier |

---

## 5. 我推测的修复方向（请你独立判断优先级）

| 候选修复 | 改动位置 | 我的判断 |
|---|---|---|
| (F1) lead prompt 加"补充图表/分析必须 update metric_plan → 重派 code-executor" 边界 | `lead_agent/prompt.py` | 治本：直接断了 lead 自己写脚本的路径，对根因 A 是充分条件 |
| (F2) 修 sandbox 路径校验误报，让 heredoc 里的 `//` 不被误判 | `sandbox/tools.py:840-854` | 治标但消除触发点：即使 lead 退化到自己写代码，也能让他在 heredoc 里完成 |
| (F3) write_file 也跑路径替换 / 或者让 ethoinsight-code skill 显式要求 `DEERFLOW_PATH_*` | `sandbox/tools.py` 或 skill 文档 | 修真 bug：根因 B 是 sandbox 路径系统的语义漏洞，长期一定要修 |
| (F4) 在 spec 第 5 章 output-constitution 加一条强约束 | spec 文档 | 把这次的经验沉淀到协议层，避免下次 plan 跑偏 |

**我的偏好（请你挑战）**：F1 + F4 优先（治本）→ F3（语义修真 bug）→ F2（消除触发点，可选）。

但我担心的事：
- F1 改 prompt 可能让 lead 在"用户问指标外问题"时反应过于刚性（"我不能做"），破坏体验
- F3 改 sandbox 可能改坏现有跑通的范式（shoaling 已经依赖现状）
- 这两个 risk 需要你独立评估

---

## 6. 你给我的产出

请你写一份 < 600 字的回复，包含：

1. **复核结论**：根因 A / B / 触发点 / 因果链——分别 ✅ / ⚠️ / ❌ + 理由
2. **遗漏的根因**：如果有
3. **修复优先级**：F1/F2/F3/F4 的顺序 + 你不同意的地方
4. **风险**：你看到的 F1/F3 副作用
5. **下一步建议**：是直接修代码，还是先把 spec 改了再写 plan？

不需要给 plan，只要诊断。

---

## 7. 元信息

- 你的角色：co-analysis agent，独立复核
- 我的角色：原诊断 agent（不会再继续推进，等你回信对齐后用户决定下一步）
- 端到端测试的发起人：用户本人（手工 dogfood）
- 截至本文档时间：2026-05-14
- 关联 thread：`b0d3a611-071e-41a5-a952-36c3772c167f`
- 关联 commit 范围：dev 比 origin/dev 领先 11 个 commit（未 push）
