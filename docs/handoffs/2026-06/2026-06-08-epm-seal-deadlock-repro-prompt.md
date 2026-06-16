# Dogfood 复现 + 取证任务：code-executor seal 死锁定性 (a 叙述黑洞 vs b max_turns 切断)

> 交给另一个 agent 执行。这是一次**受控复现 + 日志取证**任务，目标是把一个已知 harness bug 的**根因定性**钉死，不是修 bug。
> 工作目录：`/home/wangqiuyang/noldus-insight/packages/agent`

---

## 0. 一句话目标

重新跑一次 EPM dogfood，**复现 code-executor「算对了但 3 次都没封存 handoff」**，并**保住 agent-runtime 进程日志**，用日志里的「`captured AI message #N` 计数 + seal-resume 结局」把根因定性为下面两者之一：

- **(a) 叙述黑洞**：subagent 自然结束（`N` 远小于 max_turns=40），LLM 把"写 handoff JSON"当成"已完成"、输出 `OK: handoff written` 但**从没调 `seal_code_executor_handoff` 工具**，连 seal-resume 兜底也没救回。
- **(b) max_turns 切断**：subagent 撞到 `max_turns=40`（`N == 40`）在到达封存步骤前被硬切断。

这俩的修法完全不同，所以**必须先定性再谈修复**。

---

## 1. 背景（已由上一会话核实的硬事实，不要重新质疑，直接用）

- 本次要复现的故障：thread `9b27adae` 的 EPM dogfood，code-executor 被派遣 **3 次**，每次都 `terminated without emitting 'handoff_code_executor.json'`。**5 个指标其实全算出来了、值非 null**（`open_arm_time=23.56` 等），坏的只是**封存**这一步。
- **这不是某个 PR 的回归**，是 memory `feedback_subagent_seal_deadlock_is_prompt_not_budget.md` 记录的老 seal 死锁。
- 代码路径已核实：
  - code-executor 的 `max_turns=40`（`backend/packages/harness/deerflow/subagents/builtins/code_executor.py:201`）。
  - 封存工具 `seal_code_executor_handoff`，**只有 3 个必填参数** `status/summary/paradigm`，其余全可选（`backend/.../tools/builtins/seal_handoff_tools.py:401`）。所以封存本身**极易**，不是"参数太难填"。
  - max_turns 硬切断 WARNING：`executor.py:944` → `Subagent code-executor reached max_turns=40 AI messages, terminating early`。
  - seal-resume 兜底（只补 1 轮）：`executor.py:735 _attempt_seal_resume`，失败 WARNING 在 `executor.py:1059` → `terminated without emitting handoff (seal-resume did not recover)`。
  - 每个 subagent AI turn 会打 `executor.py:940` → `Subagent code-executor captured AI message #N`。
- **当前 dev 是 Gateway 模式**（只有 :8001 在听，没有 :2024 langgraph）。**所以 agent runtime 跑在 Gateway 进程里，executor 的 structlog 日志进 `logs/gateway.log`**。
  - ⚠️ 注意：`make stop` / `make dev-daemon` 会用 `> ../logs/gateway.log 2>&1` **截断** gateway.log。上一会话就是因为修 502 反复重启 dev，把那次 dogfood 的决定性 WARNING 冲掉了。**本任务最大的纪律就是：开跑前先把 gateway.log 实时另存，全程不能丢。**

### (a)/(b) 判据（这是整个任务的核心，务必理解）

参考 2026-06-02 langgraph.log 里一段**同类** data-analyst seal 失败的真实日志形态（结构一样，可作模板）：
```
[trace=cf80346f] Subagent data-analyst captured AI message #1
... #2
... #3                      ← 只到 #3/#4，远小于 max_turns
[trace=cf80346f] Subagent data-analyst: handoff missing, attempting seal-resume (1 focused turn)
[trace=cf80346f] Subagent data-analyst terminated without emitting handoff (seal-resume did not recover): ...
```
注意：它**没有** `reached max_turns` 那行，且 `#N` 只到 3-4。**这就是 (a) 的签名**——自然结束、没封存、seal-resume 也没救回。

判据表（对每一次 code-executor 派遣，从日志数它的 `captured AI message #N` 最大值 + 看有没有 `reached max_turns`）：

| 观察到的日志 | 定性 |
|---|---|
| 有 `reached max_turns=40 ... terminating early`，且 `#N` 数到 40 | **(b) max_turns 切断** |
| **没有** `reached max_turns`，`#N` 明显 < 40（如 3~12），后跟 `attempting seal-resume` + `seal-resume did not recover` | **(a) 叙述黑洞** |
| 出现 `seal_code_executor_handoff` 真实工具调用但工具报错（schema validation failed 等） | **(c) 工具报错**（第三种，概率低，但要能识别） |

**还要进一步抓 (a) 的直接物证**：把 subagent 最后一条 AIMessage 的文本捞出来，看它是不是输出了 `OK: handoff written`（或在 thinking 里写"已封存/已完成"）却没有伴随 seal 工具调用。能拿到这条就实锤 (a)。

---

## 2. 前置准备（开跑前做完，缺一不可）

1. **确认 dev 在线且健康**（应该已经在跑，:2026 nginx / :8001 gateway / :3000 frontend）：
   ```bash
   ss -ltn | grep -E ':(2026|8001|3000)\b'
   curl -s -o /dev/null -w '%{http_code}\n' -L --max-time 15 http://localhost:2026/   # 期望 200
   ```
   若没起来：`cd /home/wangqiuyang/noldus-insight/packages/agent && make dev-daemon`，等 :3000 起来再继续。
   （注：本机 inotify 上限已临时抬到 524288，前端能起。若你重启后前端崩报 `OS file watch limit reached`，先 `cat /proc/sys/fs/inotify/max_user_watches` 确认是不是被重置回 65536。）

2. **🔴 最关键一步：开一个持续 tail 把 gateway.log 另存到不会被截断的文件**。在后台起一个 monitor，全程运行，直到 dogfood 跑完：
   ```bash
   # 另存到 /tmp，文件名带时间，避免被 make stop/dev 截断 gateway.log 时丢证据
   mkdir -p /tmp/seal-repro
   # 用 tail -F 跟随（即使 gateway.log 被 rotate/截断也继续跟）
   nohup tail -F /home/wangqiuyang/noldus-insight/packages/agent/logs/gateway.log \
     > /tmp/seal-repro/gateway-capture-$(date +%H%M%S 2>/dev/null || echo run).log 2>&1 &
   echo "capture pid $!"
   ```
   - **注意**：你（执行 agent）在 Claude Code 环境里，`Date.now()`/时间函数受限的是 workflow 脚本，普通 bash 里 `date` 可用；但稳妥起见文件名给个固定后备。
   - 另外，强烈建议**同时**用 harness 的 `Monitor` 工具盯这个 capture 文件的关键行（见 §3 step 3），这样 seal 失败一发生你就知道。

3. **记下基线**：跑前 gateway.log 的行数 / 当前最大的 `captured AI message` 出现位置，便于事后只看"本次新增"的部分：
   ```bash
   wc -l logs/gateway.log
   grep -c 'captured AI message' logs/gateway.log
   ```

---

## 3. 执行 dogfood（用 Playwright，UI 全程走真实用户路径）

> 上传数据：`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28/` 下 **28 个 xlsx**（已确认存在）。
> 注意：你要复现的是 **n>1 的真实分组场景**（28 文件），比上次单文件 n=1 更接近真实，更容易触发完整 code-executor 流程。

1. 用 Playwright（MCP `mcp__plugin_playwright_playwright__*`）打开 `http://localhost:2026`，进 `/workspace`，新建对话。
2. 上传那 28 个 xlsx（一次性多选上传）。
3. 发送：`analyze the data for me`（英文即可，和上次一致，触发 `E2E_FULL_ASKVIZ` 意图）。
4. **正常应答反问**：lead 会 Gate 反问（EV19 模板 + 分组）。按 EPM 标准范式回答，例如选 `PlusMaze-FewZones（标准 EPM）`，分组就按文件里的 Treatment 字段（lead 通常能自动从表头提取，若问就让它用文件自带分组）。目标是让流水线**走到 code-executor 派遣**。
5. **盯着 code-executor 阶段**：UI 会显示「🧮 正在计算指标…子任务失败」。一旦看到**失败**（大概率会，因为这就是要复现的 bug），让它**自动重试到 3 次**（lead 会自己重试）。不要手动干预、不要替 lead 写脚本。
6. **全程让 Monitor / capture 文件记录**。每次 code-executor 失败，gateway.log 里就会多一组 `captured AI message #...` + seal-resume 行。

### 用 Monitor 工具盯关键信号（建议）

开一个 persistent monitor 盯 capture 文件，第一时间捕获判据行：
```
Monitor:
  command: tail -n +1 -F /tmp/seal-repro/gateway-capture-*.log 2>/dev/null | grep -E --line-buffered "captured AI message #|reached max_turns|attempting seal-resume|seal-resume did not recover|terminated without emitting|seal_code_executor_handoff|OK: handoff written"
  description: code-executor seal 死锁判据行
  timeout_ms: 600000
```
这样每条判据行一出现就通知你，能实时看出是 (a) 还是 (b)。

---

## 4. 取证与定性（dogfood 跑完后）

1. **锁定本次 thread_id**（从 Playwright URL `/workspace/chats/<thread_id>` 拿，或从 gateway.log 最近的 `/api/threads/<id>/runs` 拿）。

2. **从 capture 文件里抽出本次 3 次 code-executor 派遣的完整证据**。每次派遣有独立 `trace=XXXX`。对每个 trace：
   ```bash
   CAP=/tmp/seal-repro/gateway-capture-*.log
   # 列出所有 code-executor 的 trace
   grep -oE 'trace=[0-9a-f]+\] Subagent code-executor' $CAP | sort -u
   # 对每个 trace TraceID：
   grep "trace=<TraceID>" $CAP | grep -E "captured AI message #|reached max_turns|seal-resume|terminated without emitting|seal_code_executor_handoff"
   ```
   对每次派遣记录：
   - `captured AI message #` 的**最大 N**（=该次 subagent 跑了几个 AI turn）
   - 有没有 `reached max_turns=40`
   - 有没有 `attempting seal-resume` → 结局是 `did not recover` 还是成功
   - 有没有出现 `seal_code_executor_handoff` 作为**真实工具调用**（不是 prompt 文本）

3. **捞 subagent 最后一条 AIMessage 文本**（实锤 (a) 的关键物证）。subagent 内部 turn **不进 checkpoints.db**（在内存里跑），但**会进 agent-runtime 的 structlog**。在 capture 文件里找该 trace 最后一段 AIMessage 内容，看是否有 `OK: handoff written` / "已封存" / "已完成" 字样而**无** seal 工具调用。
   - 若 structlog 没打全文本，退而求其次：看 `training-data/auto-collected/<thread_id>.jsonl` 里该 subagent 的 `output` 字段（但注意上次发现它只存 envelope，可能只有 error 串——以 structlog 为准）。

4. **按 §1 判据表定性**。三次派遣大概率同一种签名。给出 **(a)/(b)/(c)** 的明确结论 + 每次派遣的 `#N` 计数表 + 至少一条直接引文。

---

## 5. 产出（写成一份新 handoff：`docs/handoffs/2026-06/2026-06-08-epm-seal-deadlock-rootcause-handoff.md`）

必须包含：

1. **定性结论**：(a) / (b) / (c)，一句话 + 证据强度（实锤/强证据/推断）。
2. **每次派遣证据表**：
   | 派遣 | trace | captured #N 最大值 | reached max_turns? | seal-resume 结局 | 最后 AIMessage 摘录 |
   |---|---|---|---|---|---|
3. **决定性引文**：至少 1 条来自 capture 文件的原始日志行（带 trace + 时间戳）。
4. **本次 thread_id + capture 文件路径**（留证，别删 `/tmp/seal-repro/`）。
5. **5 个指标是否仍全部算出**（确认这次也是"算对了只是没封存"，而不是别的新问题）。检查 workspace：
   ```bash
   # 注意真实路径含 users/<uid>/ 层
   find /home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow -path '*<thread_id>*/user-data/workspace' -type d
   # 然后 ls 该目录，确认 m_*.json 有 5 个、handoff_code_executor.json 是否缺失
   ```
6. **针对定性结论的修复方向建议**（先别实施，给方向 + 关联 memory）：
   - 若 **(a) 叙述黑洞**：根因在 code-executor prompt（`code_executor.py`）**措辞矛盾**——step6 说"调 seal tool"，但第 102/104/123 行反复用"写/落盘/已写盘 handoff"+ 把 `OK: handoff written` 当完成标志，诱导 LLM 用"写 JSON"的心智模型跳过工具调用。修法是**消除"写文件"叙事、把"调 seal 工具"设为唯一完成动作**，且 `OK: handoff written` 这种话术不应能在没调工具时被说出。**禁止**给 seal-resume 加 `tool_choice` 强制（memory 实证会产空 args）、**禁止**乱调 max_turns。关联 `feedback_subagent_seal_deadlock_is_prompt_not_budget.md`、`feedback_skill_describing_tool_output_enables_hallucination.md`。
   - 若 **(b) max_turns 切断**：根因是 40 轮预算被 28 文件 × 多指标的 bash 编排吃光，没留出封存余量。修法方向是**降 bash 调用数**（batch 并行，prompt 第 40/112 行已有 batch 提示但可能没生效）或**给封存留硬预算**，而非简单调大 max_turns。
   - 若 **(c) 工具报错**：贴出 schema validation 报错，对 `CodeExecutorHandoff` schema 修字段。

7. **顺带记账（不要在本任务修）**：上次还发现一个**独立**的 lead 侧 bug——lead 用**过期的 `ls` 快照**误报"只有 2/5 指标"（真相是第 3 次派遣已写齐 5 个，但 lead 没重新 ls）。本次 dogfood 若复现，记录证据即可，留作单独 fix。

---

## 6. 纪律红线（务必遵守）

- 🔴 **gateway.log 全程另存**，跑完前别 `make stop`/`make dev-daemon`（会截断）。如必须重启，先确认 capture 文件已存好。
- 🟡 **不要手动替 lead 写脚本 / 不要手动调 seal 工具去"救"这次 dogfood**——那会污染复现。要的就是它**自然失败**的真实轨迹。
- 🟡 **不要改任何源码**。这是取证任务，产出是 handoff + 定性结论，不是 PR。
- 🟡 **subagent 内部 turn 不在 checkpoints.db**（在内存 SubagentExecutor 里跑），别去解 1.15GB 的 checkpoints.db 找 subagent 轨迹——白费功夫，证据在 **gateway.log 的 structlog**。
- 🟢 解码/取证用 `cd backend && uv run python`（项目 venv 有依赖；裸 python3 没有）。
- 🟢 grep structlog 时注意行里有 ANSI 颜色码（`\x1b[...m`），必要时 `sed 's/\x1b\[[0-9;]*m//g'` 去色再 grep。

---

## 附：判据速查（贴墙上）

```
对每次 code-executor 派遣，看它的 structlog：
  captured AI message #N 的最大 N 是多少？
  有没有 "reached max_turns=40 ... terminating early"？

  N==40 + 有 reached max_turns   → (b) max_turns 切断
  N<40  + 无 reached max_turns
        + "seal-resume did not recover"
        + 最后 AIMessage 说"OK: handoff written"却没调工具  → (a) 叙述黑洞  ← 最可能
  出现真实 seal_code_executor_handoff 调用 + 工具报错        → (c) 工具报错
```
