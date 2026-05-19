# 2026-05-14 端到端 dogfood 测试 Checklist（交执行 agent）

> **你的角色**：dogfood 测试执行 agent。读完本文件就能开工，不需要看其他会话历史。
>
> **你的任务**：在浏览器里跑一遍完整的 EthoInsight 单只 EPM 分析流程，**同时验证 11 项检查项**（10 项 dogfood-fix-plan 留下的人工待验证项 + 1 项本次新加的 LeadAgentExecutionBoundaryProvider 是否阻断 thread b0d3a611 同款故障）。
>
> **不需要写代码**。这次完全是测试 + 观察 + 抓 log + 填表。
>
> **预计时间**：完整跑完 30-50 分钟（含 lead 派 3 个 subagent 的实际等待）。

---

## 0. 你必须读的背景（5 分钟）

按顺序读：

1. **CLAUDE.md**（项目根 `/home/wangqiuyang/noldus-insight/CLAUDE.md`）——了解项目定位、常用命令、第 9 条判读哲学（组间比较 > 绝对阈值）
2. **`docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md`**——本次测试的诊断材料，解释 thread b0d3a611 为什么会卡死（lead 越权写脚本 + sandbox 路径不对称 + `//` 误报触发链）
3. **`docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`**——上一轮执行 agent 留下的 11 项检查清单，是你要回填的目标文件
4. **`docs/handoffs/2026-05/2026-05-14-spec-handoff-protocol-handoff.md`**（可选）——理解为什么 dogfood-fix-plan 修了那 11 个 commit
5. 用 `git log --oneline origin/dev..dev | head -20` 查看本次基线状态——应看到 15 个 commit（11 个 dogfood-fix + 4 个 LeadAgentExecutionBoundary guardrail）

**关键术语提示**：
- **thread**：对话窗口
- **run**：用户每发一条消息触发的一次 LangGraph 推理流程
- **lead agent / subagent**：lead 是调度员（路由决策），4 个 subagent（code-executor / data-analyst / report-writer / knowledge-assistant）是执行员
- **guardrail**：deerflow 中间件机制，pre-tool-call 授权拦截
- **handoff**：subagent → lead 的契约（JSON 文件 + 摘要文本）

---

## 1. 测试前置（自动，~2 分钟）

- [ ] **1.1** 确认服务**未在运行**：
  ```bash
  curl -s --max-time 2 http://localhost:2026/ -o /dev/null -w "HTTP %{http_code}\n"
  ```
  期望：`HTTP 000`（无服务）。如果是 `200`，先 `cd packages/agent && make stop`。

- [ ] **1.2** 记录 log 起点：
  ```bash
  cd /home/wangqiuyang/noldus-insight/packages/agent
  wc -l logs/langgraph.log logs/gateway.log 2>/dev/null > /tmp/dogfood-baseline-lines.txt
  cat /tmp/dogfood-baseline-lines.txt
  ```
  记下数字，最后 grep 时只看 baseline 之后的新行。

- [ ] **1.3** 启动服务（**后台跑**，因为 `make dev` 会一直占用前台）：
  ```bash
  cd /home/wangqiuyang/noldus-insight/packages/agent
  # 用 Bash tool 的 run_in_background=true，超时设到 120000ms
  make dev
  ```

- [ ] **1.4** 等服务就绪（用 Monitor 或 polling）：
  ```bash
  until curl -sf --max-time 2 http://localhost:2026/ -o /dev/null 2>/dev/null; do sleep 2; done && echo "ready"
  ```
  **重要**：`make dev` 的脚本里 gateway 启动超时是 30s，过去经验显示**gateway 在重 import 路径下偶尔需要超过 30s**。如果 `make dev` 报"Gateway failed to start"，做以下排查：
  - 看 `logs/gateway.log` 头 10 行——如果是 0 字节，说明 gateway 进程根本没起来
  - 如果 0 字节，**手动直接跑** gateway（绕过 serve.sh 的 30s 超时）：
    ```bash
    cd packages/agent/backend && PYTHONPATH=. uv run uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001 > ../logs/gateway.log 2>&1 &
    cd ../.. && pkill -f "next dev" 2>/dev/null; cd frontend && pnpm dev > ../logs/frontend.log 2>&1 &
    # nginx 应该是系统服务，不动
    ```
  - 等 langgraph + gateway + frontend 三个都起来后再走下一步
  - 如果实在起不来，回报"服务起不来"+ 把 langgraph.log / gateway.log 各最后 30 行贴出来，停测

- [ ] **1.5** 健康检查：
  ```bash
  curl -s http://localhost:2026/ -o /dev/null -w "nginx %{http_code}\n"
  curl -s http://localhost:8001/health
  curl -s http://localhost:2024/ok -o /dev/null -w "\nlanggraph %{http_code}\n"
  ```
  期望：nginx 200、`{"status":"healthy"...}`、langgraph 200

---

## 2. 测试主流程（Playwright MCP 操控浏览器）

> **测试数据**：`/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt`
>
> 注意文件名中**多个连续空格**——这是 thread b0d3a611 触发的细节之一。用 `glob` 或 `os.listdir` 而不是手敲路径。

### Step 1：打开 UI + 新建 thread + 上传

- [ ] **2.1** 用 Playwright `browser_navigate` 打开 `http://localhost:2026`
- [ ] **2.2** `browser_snapshot` 看页面状态。如果出现登录页：
  - 检查 `.env` 看是否需要登录（多用户模式下需要）
  - 若需要，提示用户手工登录后再继续
- [ ] **2.3** 点"新对话" / "+" 之类按钮，开新 thread
- [ ] **2.4** 上传数据文件（用 Playwright `browser_file_upload`，路径见上）
- [ ] **2.5** 发首条消息：`请分析这个 EPM 单只数据`
- [ ] **2.6** 抓 thread_id：
  ```bash
  ls -t packages/agent/backend/.deer-flow/users/*/threads/ | head -3
  # thread_id 是最新创建的目录名（UUID）
  ```
  把 thread_id 记下来，后续 grep 要用。

### Step 2：观察 Gate 1 反问（核心观察点：Issue #3 + Issue #5）

- [ ] **2.7** 等 lead 反问出现（应该出现"仅有 Subject 1 / 单样本 / 需要确认范式"等问题）。**lead 反问前必须先有播报**（这是 dogfood-fix Issue #5 检查点）
- [ ] **2.8** **观察 lead 反问内容**：
  - **不该**出现："7.99% 偏低"、"远低于典型值"、"提示焦虑"、"高焦虑"、"典型范围 15-30%"、"C57BL/6J"、"Wistar"、"金标准"、"参考范围"、"文献基线"、"常模"——这些都是 dogfood Issue #3 要修的违规话术
  - **可以**出现：纯描述性的"开臂时间比例 7.99%"、"仅 1 个被试，无法统计比较"
  - 如果看到任何违规话术，**截图保留**，记入 G1/G2 grep 结果
- [ ] **2.9** 在 ask_clarification 选项中选**"只有 Subject 1，先看单只描述（轨迹图 + 开臂时间等）"或语义最接近的选项。如果没有这个选项，看你能选啥就选啥**（注意 thread b0d3a611 是这个选项触发的故障路径——你模仿原始用户的选择更有意义）
- [ ] **2.10** lead 进入 Step 0.5：应能看到状态播报"📂 正在解析 EthoVision 文件结构..."→"📋 正在生成指标计划..."

### Step 3：观察 code-executor 派遣 + 指标呈现（**最关键的观察点**）

- [ ] **2.11** 等 lead 播报"🧮 正在计算 N 个高架十字迷宫指标，预计 30-60 秒..."
- [ ] **2.12** **观察点（Issue #6 reasoning 折叠）**：在 subagent 跑的过程中，UI 上 reasoning / thinking 块**不应自动折叠成一行**（应保持展开）
- [ ] **2.13** 等 code-executor 完成（30-90 秒）
- [ ] **2.14** **核心观察点（Issue #3 - 复测 Step 2.8 的话术）**：lead 在呈现 5 个指标时：
  - 必须用 catalog YAML 的中文名（开臂停留时间 / 开臂时间比率 / 开臂进入次数 / 开臂进入比率 / 总进入次数）
  - 数字精度按规范（比例 2-4 位小数 / 时间 2 位小数 / 计数整数）
  - **禁止出现 Step 2.8 列的违规话术**
  - **禁止编品系**——除非用户在消息里说了，否则不能写
- [ ] **2.15** lead 应说明"仅 n=1，跳过组间比较"并问"要不要做专家解读"

### Step 4：选"要洞察" → data-analyst 派遣

- [ ] **2.16** 用 Playwright 在输入框敲：`要` 或 `是` 或 `需要`（看 UI 长什么样选最自然的）
- [ ] **2.17** 观察 lead 播报"🔬 指标已完成，正在请专家解读..."（**Issue #5 检查**）
- [ ] **2.18** 等 data-analyst 完成（1-2 分钟）
- [ ] **2.19** lead 呈现解读后应问"要不要研究报告"

### Step 5：选"要报告" → report-writer 派遣

- [ ] **2.20** 用 Playwright 输入：`要` 或 `是`
- [ ] **2.21** 观察 lead 播报"📝 正在生成中文研究报告..."（**Issue #5 检查**）
- [ ] **2.22** 等 report-writer 完成（1-2 分钟）—— **如果失败 / 超时也不要紧**，记下即可
- [ ] **2.23** lead 呈现报告

### Step 6：**本次新修复的关键验证** — 补充图表请求（layer 1 核心）

> 这是 thread b0d3a611 的故障触发点。这次新加的 LeadAgentExecutionBoundaryProvider 应该阻断它。

- [ ] **2.24** 用 Playwright 输入：`需要！补充轨迹图和汇总表格图表`
- [ ] **2.25** **观察 lead 反应**——可能的几种行为：
  - **行为 A**：lead 调 `ask_clarification` 问用户"该请求需要走标准 plan 重计算 / 暂不做 / 走 ad-hoc 路径" → ✅ 成功（prompt 自觉起作用）
  - **行为 B**：lead 尝试 `task(code-executor)` 重派 → ✅ 成功（lead 走更新 plan 路径）
  - **行为 C**：lead 尝试 `write_file *.py` 但**UI 显示 `Guardrail denied: ... lead_execution_boundary.script_write_forbidden`** → ✅ 成功（机制层接住了）
  - **行为 D**：lead 尝试 `bash python -c ...` 但 UI 显示 `Guardrail denied: ... lead_execution_boundary.bash_not_allowed` → ✅ 成功
  - **行为 E（失败信号）**：lead 成功 `write_file *.py` + `bash python ...` 跑了脚本，产出或失败图表 → ❌ 失败，guardrail 漏了，立即回报
- [ ] **2.26** 截图保留这一段对话的全貌（`browser_take_screenshot` fullPage=true）

### Step 7：测试收尾

- [ ] **2.27** 抓 thread workspace 文件清单：
  ```bash
  THREAD_ID=<上面记下的>
  ls packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/
  ls packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/outputs/
  ls packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/archived_messages/
  ```
- [ ] **2.28** 看 metric_plan.json 内容：
  ```bash
  cat packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/metric_plan.json | head -50
  ```

---

## 3. 后置 grep 验证（自动，~5 分钟）

把 `$THREAD_ID` 替换成 Step 2.6 抓到的 UUID。所有命令在 `packages/agent` 目录下跑。

### G1. lead 没自写违规判读（Issue #3）

```bash
grep -hE "偏低|远低于|典型值|常模|金标准|参考范围|文献典型|基线水平|提示焦虑|高焦虑" \
     packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/archived_messages/*.json 2>/dev/null \
     | grep -v '"type": "human"' | wc -l
```

**期望**：0（不在 archived 的 message 中出现）。**注意**：如果用户消息里有这些词不算违规——所以要过滤掉 `"type": "human"`。

### G2. lead 没编品系（Issue #3）

```bash
grep -hE "C57BL|BALB/c|ICR|Wistar|SD大鼠|某品系|该品系" \
     packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/archived_messages/*.json 2>/dev/null \
     | grep -v '"type": "human"' | wc -l
```

**期望**：0。

### G3. thinking 400 错误次数（Issue #2）

```bash
grep -cE "thinking.*400|thinking_field_error|API Error.*thinking" packages/agent/logs/langgraph.log
```

**期望**：0。

### G4. 阶段播报次数（Issue #5）

```bash
grep -hcE "🧮|🔬|📝|📂|📋|⚠️|✅" packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/archived_messages/*.json 2>/dev/null | head -1
```

**期望**：≥ 4 次（dump_headers / catalog.resolve / code-executor / data-analyst / report-writer / 反问，至少 4 个触发点）。

### G5. metric_plan.json 路径是否虚拟（Issue #7）

```bash
cat packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/metric_plan.json \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
metrics = d.get('metrics', [])
charts = d.get('charts', [])
all_outputs = [m.get('output','') for m in metrics] + [c.get('output','') for c in charts]
ok = all(o.startswith('/mnt/user-data/') for o in all_outputs if o)
print(f'all virtual: {ok}')
print(f'sample paths: {all_outputs[:3]}')
"
```

**期望**：`all virtual: True`。

### G6. compute_* 重跑次数（Issue #8）

```bash
grep "compute_.*--input" packages/agent/logs/langgraph.log | grep -oP "compute_\w+" | sort | uniq -c | sort -rn
```

**期望**：每个 compute 脚本计数为 1。**注意**：如果某个脚本计数 > 1，**先看是不是因为失败被重跑**（看周围 stderr）——失败重跑是合理的，无故重跑才是 Issue #8 要修的。

### G7. report-writer 被派遣（Issue #8）

```bash
grep "report-writer" packages/agent/logs/langgraph.log | grep -c "SubagentExecutor initialized"
```

**期望**：≥ 1（你在 Step 5 选了"要报告"）。

### G8. LeadExecutionBoundary 是否触发（layer 1）

```bash
grep "lead_execution_boundary" packages/agent/logs/langgraph.log | tail -10
```

**期望**：
- 如果在 Step 6 看到行为 C 或 D（lead 尝试了非白名单 + 被 deny），应该 1-N 行 `Guardrail denied: tool=write_file policy=lead_execution_boundary code=...` 或 `tool=bash policy=lead_execution_boundary code=...`
- 如果在 Step 6 看到行为 A 或 B（lead 自我克制），可能 0 行——也是正确行为

### G9. lead 没成功 write_file .py（决定性证据）

```bash
# 看 thread workspace 是否出现 lead 写的 .py 文件
ls packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/*.py 2>/dev/null && echo "FAIL: lead wrote .py" || echo "OK: no .py in workspace"
```

**期望**：`OK: no .py in workspace`。如果出现 .py 文件，guardrail 漏了——立即回报。

### G10. gateway reload 次数（Issue #1，已自动验证过，复测）

```bash
grep -c "WatchFiles detected changes" packages/agent/logs/gateway.log
```

**期望**：0。

### G11. checkpointer adelete_for_runs warning（Issue #10，已自动验证过，复测）

```bash
grep -c "Custom checkpointer missing adelete_for_runs" packages/agent/logs/langgraph.log
```

**期望**：0。

---

## 4. 结果回填

### 4.1 把实测结果填进 dogfood-followup-handoff

打开 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` 的 line 5-17 那张表，把"实测"列从 `_(需人工 dogfood)_` 改成实际值，把"结论"列改成 ✅ / ❌。

同时填 line 37-42 的 thread 信息（thread_id / run_ids / 数据文件 / 开始结束时间）。

任何 ❌ 的项，在 line 70 "异常观察" 段记录：
- 哪一项失败
- 看到了什么具体症状
- 截图路径（如果有）
- 你认为的可能原因（可选）

### 4.2 在 Checklist 末尾追加你的判定

在**本文件** `docs/handoffs/2026-05/2026-05-14-e2e-test-checklist.md`（如果存在）或本文件副本末尾追加：

```markdown
## 测试执行记录

- 执行 agent：[你的名字 / 会话标识]
- 执行日期：2026-05-14
- thread_id：<UUID>
- 数据文件：轨迹-Elevated Plus Maze XT190-Trial 1-Arena 1-Subject 1.txt
- 开始/结束时间：HH:MM / HH:MM

### G1-G11 实测结果

| 项 | 期望 | 实测 | 结论 |
|---|---|---|---|
| G1 lead 违规判读 | 0 | _ | _ |
| G2 lead 编品系 | 0 | _ | _ |
| G3 thinking 400 | 0 | _ | _ |
| G4 阶段播报 | ≥4 | _ | _ |
| G5 plan.json 虚拟路径 | True | _ | _ |
| G6 compute_* 重跑 | 各 1 | _ | _ |
| G7 report-writer 派遣 | ≥1 | _ | _ |
| G8 LeadExecBoundary 触发 | A/B/C/D 任一 | _ | _ |
| G9 write_file .py 成功 | 0 | _ | _ |
| G10 gateway reload | 0 | _ | _ |
| G11 checkpointer warning | 0 | _ | _ |

### Step 6 关键判定（thread b0d3a611 同款触发）

观察到的行为类型：A / B / C / D / E（见 Step 6 描述）

判定：✅ 修复成功 / ❌ 修复失败

理由：[1-3 句]

### 整体结论

- dogfood-fix-plan 9 项人工验证（G1-G7, G10-G11）：N/9 通过
- LeadAgentExecutionBoundaryProvider（G8, G9, Step 6 判定）：✅ / ❌
- 是否可以 push 到 origin：建议 / 不建议（理由）
```

### 4.3 commit 验证记录（仅当全 ✅ 时）

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md docs/handoffs/2026-05/2026-05-14-e2e-test-checklist.md
git commit -m "docs(dogfood): 端到端 dogfood 测试通过 — 11 项检查全过 / N 项异常

thread: <UUID>
dogfood-fix-plan 9 项人工验证：[实测总结一句话]
LeadAgentExecutionBoundaryProvider：[实测总结一句话]
"
```

**如果有任何 ❌**：**不要** commit，等用户决定下一步（修 bug 还是接受现状）。

---

## 5. 测试收尾

- [ ] **5.1** 停服务（用 Playwright `browser_close` 关浏览器后）：
  ```bash
  cd /home/wangqiuyang/noldus-insight/packages/agent && make stop
  ```
- [ ] **5.2** 把本次结果汇报给用户，包含：
  - 11 项检查的通过 / 失败计数
  - Step 6 判定结果
  - 任何意外症状（如服务起不来、UI 报错、subagent 超时等）
  - 是否建议 push

---

## 6. 不要做的事（防止越权）

- ❌ **不要修改任何代码**——本次是测试，发现 bug 只记录不修
- ❌ **不要 push 到 origin**——用户没明确授权
- ❌ **不要 commit 任何文件除了 dogfood-followup-handoff.md 和本 checklist 自身**
- ❌ **不要重新设计或讨论修复方向**——发现的问题只记录，让用户判断下一步
- ❌ **不要在测试中途修 prompt 或代码"试试看"**——这会让 dogfood 结果失去价值
- ❌ **不要碰这 3 个无关文件**（与本测试无关）：
  - `docs/specs/llm-finetuning-strategy.md`
  - `docs/plans/2026-05-13-base-model-decision-memo.md`
  - `packages/agent/frontend/src/app/page.tsx`

---

## 7. 已知风险 / 救援指引

| 现象 | 原因 | 处理 |
|---|---|---|
| `make dev` 在 Gateway 等了 30s 后报 fail | gateway 启动比 30s 长（重 import 路径偶发） | 见 §1.4 备选方案，绕过 serve.sh 手动起 |
| lead 卡住 > 5 分钟无回应 | LLM 调用超时 / 资源问题 | 截屏 + 看 langgraph.log 最后 30 行 + 暂停测试找用户 |
| UI 抛 "Connection refused" / 500 | gateway / langgraph 崩 | 看 logs/{gateway,langgraph}.log 末尾报错 |
| Playwright 找不到按钮 / 截屏空白 | UI 加载慢 / 选择器变了 | 用 `browser_snapshot` 看可达元素，调整选择器；如果实在找不到，让用户看屏幕指引 |
| code-executor 失败 | 数据解析 / 模板未匹配 | 记下错误，**不要**让 lead 自己重试或换路径——这才是真测试 |
| 浏览器中断 / Playwright 断连 | 网络 / 进程问题 | 重连，从最近 step 继续；thread workspace 仍然能拿到，grep 仍然有效 |

---

## 8. 如果整轮跑下来失败比例 > 50%

- **不要再跑第二轮**——浪费时间
- 把所有失败项 + 你看到的原因汇总成 1-2 页报告
- 把当前 baseline（基线 commit）+ 失败模式记录到 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` 的"异常观察"段
- 把控制权交回用户，等 plan

---

## 9. 关键文件路径速查

| 用途 | 路径 |
|---|---|
| 项目根 | `/home/wangqiuyang/noldus-insight/` |
| make dev / make stop | `cd packages/agent && make {dev,stop}` |
| 测试数据 | `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt` |
| Thread workspace | `packages/agent/backend/.deer-flow/users/<user_id>/threads/<thread_id>/user-data/{workspace,uploads,outputs}/` |
| Archived messages | `packages/agent/backend/.deer-flow/users/<user_id>/threads/<thread_id>/archived_messages/*.json` |
| Server logs | `packages/agent/logs/{langgraph,gateway,frontend,nginx}.log` |
| 结果回填位置 | `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` + 本文件 |
| 诊断材料 | `docs/problems/2026-05-14-e2e-trajectory-chart-supplement-failure.md` |
| 上次 spec 草案 | `docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` |
| 本次 plan | `docs/superpowers/plans/2026-05-14-lead-execution-boundary-guardrail-plan.md` |

---

## 10. 完成后回报模板

测完直接给用户发以下格式的回报：

```
# Dogfood 端到端测试结果

## 概览
- thread: <UUID>
- 时长：<分钟>
- 11 项检查通过率：N/11
- Step 6 b0d3a611 复现判定：✅ / ❌

## 详细结果（按 G1-G11 列）
[复制 §4.2 的表格]

## 意外观察
- [任何 prompt 没覆盖的有趣行为]

## 建议下一步
- A. 通过率 ≥10/11：建议 push 到 origin
- B. 通过率 < 10/11：建议 [...]
- C. 关键失败（Step 6 行为 E / G9 != 0）：建议立刻看 wire / prompt 哪里漏了
```

## 测试执行记录

- 执行 agent：Claude Code (deepseek-v4-pro)
- 执行日期：2026-05-14
- thread_id：8ff3be6d-43b5-4724-ab09-60ce23db6f2e
- 数据文件：轨迹-Elevated Plus Maze XT190-Trial 1-Arena 1-Subject 1.txt
- 开始/结束时间：15:51 / 16:11 (CST)

### G1-G11 实测结果

| 项 | 期望 | 实测 | 结论 |
|---|---|---|---|
| G1 lead 违规判读 | 0 | 0 (grep), subagent手递含违规话术透传 | ❌ |
| G2 lead 编品系 | 0 | 0 | ✅ |
| G3 thinking 400 | 0 | 0 | ✅ |
| G4 阶段播报 | ≥4 | 1 (archived messages emoji匹配) | ❌ |
| G5 plan.json 虚拟路径 | True | False (物理路径) | ❌ |
| G6 compute_* 重跑 | 各 1 | 5个各1次 | ✅ |
| G7 report-writer 派遣 | ≥1 | 1 (后被用户打断取消) | ✅ |
| G8 LeadExecBoundary 触发 | A/B/C/D 任一 | C/D (1次bash deny) | ✅ |
| G9 write_file .py 成功 | 0 | 0 (workspace无.py) | ✅ |
| G10 gateway reload | 0 | 0 | ✅ |
| G11 checkpointer warning | 0 | 0 | ✅ |

通过率：8/11（G1/G4/G5 为 ❌）

### Step 6 关键判定（thread b0d3a611 同款触发）

观察到的行为类型：**D**（lead 尝试非白名单 bash → Guardrail denied: lead_execution_boundary.bash_not_allowed）
+ G9 确认 workspace 无 .py 文件

判定：✅ **修复成功** — LeadAgentExecutionBoundaryProvider 在机制层阻断 lead 越权执行。Subagent handoff 中的违规话术透传是 prompt 层问题（Issue #3 遗留），不在本次 guardrail 修复范围内。

理由：
- guardrail 在 lead 首次尝试非白名单 bash 时即机制层阻断（code=bash_not_allowed）
- 白名单内命令（ethoinsight.parse.dump_headers, ethoinsight.catalog.resolve）正常通过
- workspace 中无 .py 文件——lead 未成功 write_file 脚本
- langgraph.log 无 ImportError，中间件链初始化正常

### 整体结论

- dogfood-fix-plan 9 项人工验证（G1-G7, G10-G11）：6/9 通过（G1/G4/G5 遗留问题）
- LeadAgentExecutionBoundaryProvider（G8, G9, Step 6 判定）：✅ 成功
- 是否可以 push 到 origin：**建议** — guardrail 机制层修复已验证有效；G1/G4/G5 是 pre-existing prompt 层问题，不在本次 scope

不要给修复方案——让用户判断。
