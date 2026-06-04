# Playwright 多范式 dogfood 端到端测试指南

> **面向执行的 Claude Code agent**。目标：在本地 `make dev` 起来后，用 Playwright 驱动前端，对 6 个哺乳动物范式逐个跑「上传数据 → 帮我分析数据 → 走完全链路」的 dogfood，最快暴露 seal-deadlock / zone override / 反问诱导 / 流水线断裂 等问题。
>
> **本次重点关注**：`data-analyst terminated without emitting 'handoff_data_analyst.json'`（seal-deadlock）。2026-06-04 刚修了 O迷宫触发的「非空参数路」复发点（`data_analyst.py` step 2.8）。本轮要验证修复是否在真实 LLM 行为层生效，并顺带扫其他范式有没有同类或新问题。

---

## 0. 前置：环境必须就绪

### 0.1 启动服务（如果还没起）

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop && make dev
```

> ⚠️ **铁律：backend 改过工具/subagent 定义后必须重启 dev**。tool 与 subagent 的 prompt 在 agent 创建时就构建进内存，不重启跑的是旧代码 → 白测。本次的修复改的就是 `data_analyst.py` 的 system prompt，**没重启就测 = 验旧代码**。

### 0.2 确认服务起来了

前端入口（nginx 统一反代）：**`http://localhost:2026`**

```bash
# 确认端口在监听
ss -ltn | grep -E ':(2024|2026|3000|8001)'
# 期望：2026 (nginx) / 3000 (frontend) / 8001 (gateway) / 2024 (langgraph) 都在
```

> 注：开发者本地可能把前端开在 **2027**（见上次 thread URL `localhost:2027/workspace/...`）。**以实际监听端口为准**——先 `ss -ltn` 看，再决定 Playwright `navigate` 到 2026 还是 2027。

### 0.3 浏览器内确认可用

Playwright 打开 `http://localhost:2026`（或 2027），确认能看到聊天输入框 + 文件上传按钮，再开始。

---

## 1. 测试数据（唯一数据源）

全部在 `/home/wangqiuyang/DemoData/newdemodata/`。**每个范式上传对应的 `原始数据-*.xlsx`**（不传 `轨迹-*.txt`，那是逐帧轨迹，xlsx 才是 EV19 导出含元数据/分组的主文件）。

| # | 范式 | 中文目录 | 上传文件（xlsx 绝对路径） | 份数 | 备注 |
|---|------|---------|--------------------------|------|------|
| 1 | EPM 高架十字迷宫 | `高架十字迷宫_小鼠_三点` | `原始数据-Elevated Plus Maze XT190-Trial     1.xlsx` | **1** | 单试验 → n=1，验**不误伤**（命名列齐全，**不该**报 zone_unnamed） |
| 2 | OFT 旷场 | `旷场_小鼠_三点` | `原始数据-Open Field test XT190-Trial     1.xlsx`、`...Trial     2.xlsx` | 2 | 验 OFT center_zone（str 路径） |
| 3 | LDB 明暗箱 | `明暗箱` | `原始数据-ethoInsightDemo-ldb-试验     1.xlsx` … `试验     6.xlsx` | 6 | 验 LDB light_zone（str 路径）+ 多文件 |
| 4 | FST 强迫游泳 | `强迫游泳_大鼠` | `Mobility_1/原始数据-Porsolt forced swim test XT190-Trial     1.xlsx`、`Mobility_10/原始数据-...Trial     1.xlsx` | 2 | xlsx **嵌在 Mobility_* 子目录里**；验 mobility_state 路径（parameters_used 应为空 `{}` → 走空参数快速路） |
| 5 | O迷宫 Zero Maze | `O迷宫` | `原始数据-oMaze-试验     1.xlsx`、`...试验     2.xlsx`、`...试验     3.xlsx` | 3 | **本次根因范式**，必跑。验 zero_maze open_zones（list 路径）+ **非空参数路 seal** |
| 6 | TST 悬尾 | `悬尾` | `原始数据-tstHelperDemoVideo-试验     1.xlsx`、`...试验     2.xlsx` | 2 | 验 TST mobility/pendulum 路径 |

> ⚠️ **文件名里有连续空格**（如 `Trial     1`）。Playwright `setInputFiles` / 上传时用**绝对路径原样**传，别手动改空格数量。
>
> ⚠️ **n=1 注意**：EPM(1)、FST(每组1)、O迷宫(每组1) 都是「每组 n=1」——会触发 `SAMPLE.TOO_SMALL` critical 警告（`blocks_downstream=false`）。这是**预期行为，不是 bug**；data-analyst 应把它降级成 method_warnings 并照常出解读。**真正的失败信号是 data-analyst 没 seal 或链路中断**，不是这条警告本身。

---

## 2. 跑法：串行单范式（推荐先这样，定位最准）

**为什么先串行不并行**：dogfood 失败几乎都在 LLM 行为层（seal 漏调、反问诱导、流水线断裂），需要**逐 thread 盯链路 + 核 workspace 文件**。并行跑会让多个 thread 的后端 log 交织，定位变难。先串行把每个范式跑清楚；6 个范式各开**独立 thread**（每个范式点「新建对话」或新 thread URL），互不污染。

> 想加速：可以开多个浏览器 context / 多 tab **并行起多个 thread**，但**每个范式仍是独立 thread**，且失败时回到串行复查那一个。并行只省墙钟时间，不改变「一范式一 thread + 逐个核验」的判定方式。

### 2.1 每个范式的标准动作（Playwright 脚本骨架）

对每个范式重复：

1. **新开 thread**：navigate 到前端根，点「新建对话」（或直接打开一个新 thread URL）。**记下 thread URL 里的 thread_id**（形如 `.../chats/<thread_id>` 或 `.../workspace/chats/<thread_id>`）——核验时要用。
2. **上传该范式的全部 xlsx**：点上传按钮 → `setInputFiles([...该范式所有 xlsx 绝对路径...])` → 等待上传完成（文件名出现在输入区上方）。
3. **发指令**：在输入框输入 **`帮我分析数据`** → 回车。
4. **等待并盯链路**（见 §3 验收点）。单范式从发指令到出报告，正常 3-8 分钟（deepseek + 多 subagent）。给足超时，**别提前判失败**。
5. **判定 + 记录**：按 §3 checklist 逐项打勾；无论成功失败，按 §4 核验 workspace 落盘文件（这是硬证据，**不要只信前端文字**）。
6. **下一个范式 → 回到 1**。

### 2.2 Playwright 提示

- 上传控件大概率是隐藏的 `<input type="file">`；用 `page.locator('input[type=file]').setInputFiles([...])` 而非点按钮走系统弹窗。
- 等待用**文本/状态**等待，别用固定 sleep：等「🧮 正在计算指标」「🔬 正在请专家解读」「报告」等关键文案出现/消失。
- 中途如果 agent **反问**（弹出选项卡，如「in_zone=1 代表开放臂还是封闭臂」），见 §3.3 怎么答——**答错会污染结果**。

---

## 3. 验收点 checklist（每个范式逐项核）

### 3.1 通用全链路（所有范式都该走通）

- [ ] **inspect**：lead 调 `inspect_uploaded_file`，识别出范式（EPM/OFT/LDB/FST/zero_maze/TST）+ 自动从 EV19 元数据提取分组（Treatment/Dose）。
- [ ] **set_experiment_paradigm**：范式写入 experiment-context.json。
- [ ] **prep_metric_plan**：生成 `plan_metrics.json`（指标清单）。
- [ ] **code-executor**：算出指标，产 `handoff_code_executor.json`，封存成功（log 出现 `OK: handoff written to .../handoff_code_executor.json`）。
- [ ] **data-analyst**：✅ **本次核心**——产 `handoff_data_analyst.json`，**一次 seal 成功**，**不出现** `terminated without emitting 'handoff_data_analyst.json'`。
- [ ] **report-writer**：出最终报告（markdown / 前端展示）。
- [ ] 全程**无死循环**（同一工具被反复调 ≥5 次撞安全限 FORCED STOP = 失败）。

### 3.2 🔴 seal-deadlock 专项（这是本轮第一优先信号）

任何范式只要前端出现：

```
Error: Subagent 'data-analyst' terminated without emitting 'handoff_data_analyst.json'.
```

→ **记为该范式 FAILED**，并立刻按 §4 抓证据。同理留意 `code-executor` / `chart-maker` / `report-writer` 的同句式 `terminated without emitting`。

**重点观察**：

- O迷宫、FST、EPM 这几个 n=1 / 含 `parameters_used` 的范式，data-analyst 的 **step 2.8 参数审计** 是高危段。修复后**期望**：遇到 `open_zones`（O迷宫）这类离散参数 + 范式无判据时，data-analyst **直接记一条 info finding 就 seal**，不在 thinking 里反复纠结 Phase2/Phase1/mismatch_kind。
- 如果还卡：看后端 log，data-analyst 的 reasoning 是不是又在 step 2.8 把整个 handoff JSON 草稿写在 thinking 里、写完「现在调用 seal」却没发出 tool_call —— 那就是同款复发，**记下完整 reasoning** 交回。

### 3.3 🔴 zone 反问专项（O迷宫 / 可能 OFT/LDB）

O迷宫会反问 **「in_zone=1 代表开放臂还是封闭臂」**。

- **几何实测定论：O迷宫 `in_zone=1` = 开放臂**（动物天然回避，占时仅 6-14%）。
- 反问时**答「开放臂（Open Arm）」**。
- ✅ 期望：lead 反问时**带占时证据**（「各组占比仅 6.4%–14.2%」）正向引导用户选开放臂，**不该**诱导用户选「封闭臂」。
  - 如果 lead 的反问话术在**诱导答封闭臂** → 记为话术问题（即使数据层能 override 修正，话术错了用户会答反）。
- 答完后期望：写入 `parameter_overrides={"anonymous_zone_is":"in_zone"}` → 翻译成 `open_zones=["in_zone"]` → compute 出真值（open_zone_time_ratio ≈ 0.06~0.14）。
- ❌ **不该出现** `~in_zone` / `open_zone:"~in_zone"` 这种「取反」脑补语法（那是修复前的死循环根因）。

### 3.4 不误伤专项（EPM）

EPM 命名列齐全（开放臂/封闭臂都有正式列名）。

- ✅ 期望：**不报** `zone_unnamed`，**不反问** zone 归属，直接出 5 个指标。
- ❌ 若 EPM 也弹「未命名分析区」反问 → 误伤，记下来。

---

## 4. 硬证据核验（失败/成功都要做，不要只信前端）

每个范式跑完，用 thread_id 去核 workspace 落盘文件。路径模板：

```
packages/agent/backend/.deer-flow/users/<user_id>/threads/<thread_id>/user-data/workspace/
```

> `<user_id>` 当前本地 dogfood 通常是 `cd95effa-d595-441a-bc44-29db0f3e259d`（以实际为准；可 `ls .../users/` 看哪个 user 下有你的 thread_id）。

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
TID=<你的thread_id>
WS=$(find .deer-flow/users -type d -path "*threads/$TID/user-data/workspace")
echo "workspace: $WS"
ls -la "$WS"

# 关键落盘文件应当齐全：
#   experiment-context.json     ← 含 paradigm + (O迷宫)parameter_overrides
#   plan_metrics.json           ← 指标计划
#   handoff_code_executor.json  ← code-executor 封存（应存在）
#   handoff_data_analyst.json   ← data-analyst 封存（★ seal-deadlock 时【缺失】= 铁证）
#   handoff_chart_maker.json / handoff_report_writer.json（若链路走到）
```

判定：

- **`handoff_data_analyst.json` 缺失** = data-analyst seal 失败的硬证据（哪怕前端文字写了「封存成功」也不算）。
- **O迷宫专项**：`cat "$WS/experiment-context.json"`，确认 `parameter_overrides` 是 `{"anonymous_zone_is":"in_zone"}`（统一 key），**不该**有 `~in_zone`。
- **核 parameters_used**：`grep -o '"parameters_used":[^}]*}' "$WS/handoff_code_executor.json" | sort -u`
  - O迷宫期望 `{"open_zones": "in_zone"}`（非空 → 走修复的非空参数路）。
  - FST 期望多为 `{}`（空 → 走空参数快速路）。

### 4.1 失败时额外抓后端 log + reasoning

```bash
# 后端 log（路径以实际 make dev 输出为准，常见在 backend/ 或 logs/）
# 搜本 thread 的 seal / zone / 死循环关键字：
grep -E "$TID|terminated without emitting|seal_data_analyst|anonymous_zone_is|open_zones|FORCED STOP" <log文件>
```

把以下三样一起带回来（这是诊断 seal-deadlock 的最小证据集）：
1. 前端 transcript（data-analyst 的完整 thinking，尤其 step 2.8 那段）。
2. `handoff_data_analyst.json` 是否存在 + `handoff_code_executor.json` 的 `parameters_used`。
3. 后端 log 里该 thread 的 seal / 死循环相关行。

---

## 5. 优先级与最小集

时间有限时按此顺序：

1. **🔴 O迷宫**（本次根因 + 最复杂：非空参数路 seal + zone 反问 + list 翻译）—— **必跑**。
2. **🔴 FST**（上次同类故障范式，验空参数快速路没回退）—— 必跑。
3. **🟡 EPM**（验不误伤）+ **LDB**（验 str 路径 + 多文件）。
4. **🟢 OFT** + **TST**（补全 6 范式覆盖）。

全绿判据：6 范式各自走完 inspect→plan→code→data-analyst→report，**无一例** `terminated without emitting`，O迷宫 zone 反问被正向引导且 override 落对，EPM 不误伤。

---

## 6. 报告回填模板

跑完把结果按此表回填交回：

| 范式 | thread_id | inspect识别对 | code seal | **data-analyst seal** | report出 | zone反问(仅O迷宫) | 误伤(仅EPM) | 备注/异常 |
|------|-----------|--------------|-----------|----------------------|----------|------------------|------------|-----------|
| O迷宫 | | | | | | | — | |
| FST | | | | | — | — | |
| EPM | | | | | — | | |
| LDB | | | | | — | — | |
| OFT | | | | | — | — | |
| TST | | | | | — | — | |

> **特别提醒**：LLM 行为是非确定性的（deepseek）。**单次跑通 ≠ 已修复**，单次失败也可能是偶发。对 data-analyst seal 这种概率性故障，**O迷宫和 FST 建议各跑 2-3 次**（每次新 thread），看是否稳定 seal。若 3/3 都 seal 成功，修复可信度高；若偶发失败，记下失败那次的完整证据（§4.1）交回。
