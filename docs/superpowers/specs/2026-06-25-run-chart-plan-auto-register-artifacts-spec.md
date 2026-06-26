# Spec：run_chart_plan 自动登记 artifacts —— 修复「113 张图只显示 1 张」产物丢失

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-25
> 代码基线：dev HEAD `0ff55a98`
> 性质：🔴 高 · 产物丢失修复（结构性，守"LLM 提议、确定性门定生死"）
> 受保护文件（sync surgical）：`subagents/builtins/chart_maker.py`、`tools/tools.py`、`tools/builtins/__init__.py`（本 spec 主要改 `run_chart_plan_tool.py` 非受保护，但 chart_maker.py prompt 文案受保护）
> **承继关系**：本 spec 修的是 [2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md](2026-06-24-chart-maker-run-chart-plan-deterministic-execution-spec.md) 的设计遗留洞。那份 spec 建了 `run_chart_plan` 工具，但 §146 把「图进 state」设计成「run_chart_plan 返回后 chart-maker 再调 present_files 登记」——本 spec 证实这一步因 LLM 逐张漏调而丢图，改为 run_chart_plan **自己确定性登记**。

---

## ⚠️ 〇、根因取证（生产数据 + 亲读代码双重坐实）

> 守 memory `feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected`（先取证再写 spec）+ `feedback_code_has_fix_not_equal_bug_eliminated`（问"现象还会不会发生"）。下列每条均有证据，**接手 agent 可信不必重查**（但实施前建议复跑「验证」节确认基线未变）。

**现象**：研究员跑完 EPM 分析（生成 113 张图：1 aggregate + 112 per_subject），前端画廊只显示 1 张。

**取证链**（实测 thread `0e72d605-1f51-4dbd-800f-6738bab39a69`，owner `qiuyang.wang@noldus.com.cn`）：

| 证据 | 数据 | 来源 |
|---|---|---|
| 磁盘真实产物 | **113 张 png** 全在 | `find .../outputs -name '*.png' \| wc -l` = 113 |
| chart-maker 自报 | `handoff_chart_maker.json` `chart_files` **113 项全渲染**、`status=completed`、`sealed_by=run_plan` | workspace handoff |
| plan 声明 | `plan_charts.json` `charts` **113 项**（1 aggregate + 112 per_subject） | workspace plan |
| **state.artifacts（前端数据源）** | **只有 2 项**：`plot_box_open_arm.png` + `report.md`（旧 string 格式） | langgraph API `/threads/{tid}/state` + checkpoints.db 解码 |
| present_files 调用记录 | 被调 **5 次，每次只传 1 个文件**（`filepaths=[1,1,1,1,1]`） | checkpoints.db `writes` 表解码 |

**根因**：`run_chart_plan` 工具落盘 113 张图 + 写 handoff，但**返回普通 dict、从不更新 `thread.values.artifacts`**（`run_chart_plan_tool.py` 末尾 `return result`）。artifacts 的更新 100% 依赖 chart-maker 之后**逐张调 present_files**——但 LLM 只调了 5 次每次 1 张，**112 张 per_subject 图根本没传给 present_files**。属 seal/present 漏调家族（SKILL 写了 `present_files(<png列表>)` 但 LLM 不可靠地逐张调且没调完）。

**这不是前端 spec（#2/#3/#6）引入的**：出问题 thread 是 #3 之前跑的（`state.artifacts` 旧 string 格式，没经 #3 的 ArtifactMeta 路径）。该登记缺陷自 `run_chart_plan` 工具诞生（前序 spec）就存在。**#3 画廊只是暴露了它**（画廊忠实渲染 state，state 只有 1 张就显示 1 张；画廊源码本身已验证全绿）。

**修复方向**：让 `run_chart_plan` 落盘的 `chart_files` **必然全部进 artifacts**，不再依赖 LLM 逐张 present。

---

## 一、命门事实（修复有效性依据，已坐实）

1. **subagent 工具返回的 `Command(update={"artifacts":...})` 能上行到 lead thread state**。证据：present_files 在 chart-maker（subagent）里调用返回 `Command(update={"artifacts":[...]})`（`present_file_tool.py:307-315`），生产 checkpoints 里这些写入 `checkpoint_ns=''`（= lead 主 thread state，前端读得到那 2 张）。
2. **subagent 子图不产生独立 namespace**：`executor.py:1029` subagent 用 `checkpointer=False` + `state_schema=ThreadState` + `astream(state, config={configurable:{thread_id}})`，工具 Command 更新的 state 上行到 lead thread。
3. **run_chart_plan 与 present_files 在同一 chart-maker subagent 里、返回同类 Command、写同一 `artifacts` 字段、过同一 `merge_artifacts` reducer**（`thread_state.py:160` + 按 path 去重）→ 走完全相同的上行路径，113 张必然进 lead state。
4. **`view_image_tool.py:49-116` 是现成 1:1 模板**：`@tool` + `tool_call_id: Annotated[str, InjectedToolCallId]` + `-> Command` + `Command(update={"messages":[ToolMessage(...,tool_call_id=tool_call_id)]})`。**Command 返回类型变更零风险**（同款工具已在生产跑）。

---

## 二、改动清单

### 1. `packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py`（主改，非受保护）

**1.1 顶层 import 补**（langchain/langgraph 包，非 deerflow 内部模块，不参与 harness 环，安全提顶层）：
```python
from typing import Annotated
from langchain.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
```

**1.2 `present_file` 三 helper 必须惰性 import（函数体内）**——它们来自 `present_file_tool`（harness 内部模块，顶层 import 链含 `sandbox.tools`/`config.paths`/`runtime.user_context`/`tools.types`，与 run_chart_plan 交叉成环）。放在现状已有惰性 import 处（`run_chart_plan_tool.py:139-143` 同区，那里已惰性 import `_build_path_env`/`replace_virtual_path`/`_seal_handoff_to_workspace`）。**CLAUDE.md 铁律：改 tools/builtins 顶层别加 deerflow import。**

**1.3 签名变更**：
```python
@tool("run_chart_plan", parse_docstring=True)
def run_chart_plan_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],   # ← 新增；位置必须在 runtime 后、plan_path(有默认值)前
    plan_path: str = "/mnt/user-data/workspace/plan_charts.json",
    only_chart_ids: list[str] | None = None,
    on_error: str = "continue",
) -> Command:                                            # ← dict → Command
```
- **位置坑**：`tool_call_id` 无默认值，Python 语法要求它在有默认值的 `plan_path` 之前（view_image 放最后是因它前面参数也无默认值，不能照搬位置）。
- InjectedToolCallId 从 args_schema 剔除，**LLM 看不到、调用习惯不变**（仍只传 plan_path/only_chart_ids/on_error）。docstring `Args:` 段不动；`Returns:` 段改述「返回 Command，把 artifacts/charts_status 写入 state；ToolMessage content 含 status/n_rendered/failures 供决策树」。

**1.4 6 个错误早退点也要返回 Command**（`run_chart_plan_tool.py:148/157/162/167/175/301`）——签名 `-> Command` 后混返 dict 会让 ToolMessage 无 tool_call_id 绑定、行为退化。把 `_error_result(dict)` 改造成：
```python
def _error_command(tool_call_id: str, code: str, message: str) -> Command:
    result = {"status":"failed", "error_code":code, "message":message,
              "handoff_path":"/mnt/user-data/workspace/handoff_chart_maker.json",
              "n_total":0, "n_rendered":0, "n_failed":0, "failures":[],
              "gate_signals": {...同现状 _error_result...}}
    return Command(update={"messages":[ToolMessage(json.dumps(result, ensure_ascii=False), tool_call_id=tool_call_id)]})
```
6 个早退点改 `return _error_command(tool_call_id, code, msg)`。错误路径**不写 artifacts**。`_seal_failed` 不变（只写盘）。

**1.5 末尾正常返回**（替换 `run_chart_plan_tool.py:303-316`）：
```python
    # ---- Step 10: 自动登记 artifacts（产出+交付合一，不再依赖 chart-maker 逐张 present）----
    from deerflow.tools.builtins.present_file_tool import _build_artifact_meta, _build_charts_status, _load_plan_charts

    artifact_plan = _load_plan_charts(thread_data)   # {paradigm, by_output}；缺/坏→None。复用 SSOT，别手重建 by_output
    artifact_metas: list[str | dict[str, Any]] = [
        _build_artifact_meta(vp, artifact_plan, thread_data, generate_thumb=True)  # 缩略图策略见 §三
        for vp in chart_files                          # 已核磁盘的虚拟路径（Step 7）
    ]
    result = {"status":status, "handoff_path":"/mnt/user-data/workspace/handoff_chart_maker.json",
              "n_total":n_total, "n_rendered":n_rendered, "n_failed":n_failed,
              "failures":failed_charts, "gate_signals":payload["gate_signals"]}
    update: dict[str, Any] = {
        "artifacts": artifact_metas,
        "messages": [ToolMessage(json.dumps(result, ensure_ascii=False), tool_call_id=tool_call_id)],
    }
    cs = _build_charts_status(thread_data)             # ← 连带修：失败/截断摘要也只在 present 时进 state，必须一起写
    if cs is not None:
        update["charts_status"] = cs
    return Command(update=update)
```
- **ToolMessage content 用 `json.dumps(result)`**（与现状 chart-maker 拿到的 dict 序列化字节同形）→ chart-maker 决策树（`chart_maker.py:64-72` parse status/failures）**零感知变化**。

**1.6 加注释钉死与 `run_metric_plan` 的不对称**（防 catastrophic forgetting）：run_chart_plan 返 Command（图进 artifacts 给画廊/IM）；其孪生 `run_metric_plan_tool` 仍返 dict（指标给下游 subagent 读 JSON handoff，**不该进 artifacts**）。**合理不对称，勿"修回"一致**——否则下个 sync/重构者会以为漏改、照 run_metric_plan 改回 dict 回归本 fix。

### 2. `packages/agent/backend/tests/test_run_chart_plan.py`（同步测试，TDD 先行）

- **改 `_call`**（`test_run_chart_plan.py:188-190`）：传 `tool_call_id="test-tcid"` + 解包 `Command.update["messages"][0].content`(json) → 旧 result dict。这样所有读 `res["status"]`/`res["n_rendered"]`/`res["failures"]`/`res["error_code"]` 的既有断言（T1-T14 + F1 + M2）**零改动**：
  ```python
  def _call(runtime, **kwargs):
      cmd = _TOOL.run_chart_plan_tool.func(runtime, tool_call_id="test-tcid", **kwargs)
      msgs = cmd.update.get("messages", [])
      assert msgs, f"Command 缺 ToolMessage: {cmd.update}"
      return json.loads(msgs[0].content)
  def _call_command(runtime, **kwargs):   # 需直接断言 artifacts/charts_status 的测试用
      return _TOOL.run_chart_plan_tool.func(runtime, tool_call_id="test-tcid", **kwargs)
  ```
- **新增 `TestArtifactsRegistered`**：(a) 113 charts → `cmd.update["artifacts"]` 113 个 meta、aggregate 那张 `output_mode=="aggregate"`、全 `/mnt/user-data/outputs/` 前缀、`kind=="chart"`；(b) status/n_rendered 从 ToolMessage 取、tool_call_id 正确透传；charts_status 失败时进 state（`_runner_fail_on`）；错误路径返 Command 且 `"artifacts" not in cmd.update`。
- **⚠️ 缩略图断言坑**：测试 runner 写假 PNG header（`test_run_chart_plan.py:133` `b"\x89PNG..."`），Pillow 打不开 → `_generate_thumbnail` 返 None → meta 无 `thumb_path`。**不要断言"113 张都有 thumb_path"**（会红）——只验 `generate_thumb=True` 传下去 + Pillow 失败优雅退化不 crash。缩略图真实性是 present_file 职责的测试范畴。

### 3. `chart_maker.py` + `ethoinsight-chart-maker/SKILL.md`（prompt 文案，第一版「A 弱化版」）

- **present_files 步骤保留**（`chart_maker.py` 工具白名单不动）——因 IM 集成 `manager.py:377-409` `_extract_artifacts` **只扫 present_files tool_call 的 filepaths 提取产物，不读 state.artifacts**（已读码坐实 `:404` `tc.get("name")=="present_files"`）。纯删 present_files 会让飞书/Slack 渠道拿不到图（Web 读 state.artifacts 不受影响，IM 会回归）。
- **文案改成正面描述新事实**（deepseek 正面指令，**不写"不要再 present"**——反向激活 + 等于加提醒规则违 HarnessX 自检）。第 7 步（present_files 那步）改为：
  > 「图表已由 run_chart_plan 自动登记并呈现给用户（前端画廊直接可见）。再调一次 `present_files(<run_chart_plan 落盘的 png 列表>)` 把同批图登记进消息通道——IM 渠道（飞书/Slack）据此把图作为附件推送；Web 端已由 run_chart_plan 呈现，此步为幂等补充（reducer 按 path 去重不会重复）。」
- **SSOT 三镜像点必须同步改防 drift**（memory：`385f8989` 漏同步三处镜像被 PR#175 补）：`SKILL.md:37` + `chart_maker.py:74`(system_prompt 第 10 步) + `chart_maker.py:191-198`(output_contract)。**改前跑** `grep -rn "present_files" docs/.../ethoinsight-chart-maker/SKILL.md packages/.../subagents/builtins/chart_maker.py` 核全部镜像点（工具白名单那处只保留不改文案）。

---

## 三、缩略图策略

第一版用 `generate_thumb=True` **同步串行**（复用 `_build_artifact_meta` 内部 `_generate_thumbnail`，113 张 ~2.3s）。**否决"跳过/延迟给 present_files"**——根因就是 present_files 不可靠，延迟等于让 112 张图照样没缩略图（前端画廊退化成加载 1–2.7MB/张原图，正是 #3 spec §3.1.6 要砍的成本）。`_generate_thumbnail` 有 `if not thumb_real.exists()` 短路，重跑子集不重做。若 dogfood 实测 2.3s 扎眼，第二版改 ProcessPool 并行降到亚秒（需 pickle 友好 worker：worker 只做纯 Pillow resize 返真实路径、主进程再还原虚拟；第一版不吃这复杂度）。

---

## 三-bis、state 体积权衡（"把图写进 state 会不会爆" — 实测回答）

> 把完整 ArtifactMeta 写进 state 是否使 state 过大？**本量级（113~500 张）安全，但要理解为什么、以及何时该升级。**

**实测数据**（thread `0e72d605`，checkpoints.db 解码）：

| 项 | 大小 |
|---|---|
| 单个 ArtifactMeta（path/chart_id/output_mode/metric/subject/thumb_path…） | **379 bytes** |
| 113 个 artifacts（本修复新增） | **42 KB** |
| 极端 500 张 | **185 KB** |
| 同一 checkpoint 里 `messages` 字段（对比大头） | **119.6 KB** |
| 当前 `artifacts`（只 2 项） | 0.1 KB |

**为什么安全（关键是写放大，不是绝对大小）**：
1. 42 KB 比 `messages`（120 KB）还小，前端读一次 state 多 42 KB 无感。
2. **LangGraph checkpointer 每个 super-step 重写变更的 channel**（该 thread 跑 290 个 checkpoint、累计写 22 MB）——但 `artifacts` 用 `merge_artifacts` **reducer**，只在 `run_chart_plan` 那**一次** Command 触发写入，之后 step 不改它就不重写（`writes` 表按 channel 增量）。**42 KB 只写一次，不随 step 放大。**
3. **path 是真相、ArtifactMeta 是缓存**：图在磁盘 + plan_charts.json 是唯一真相，state 里的 meta 是"方便前端用的缓存视图"，丢了可重建。state 不承担"唯一存储"职责。

**最佳实践对照（业界 reference-in-state, payload-in-store 三档）**：
- **A. 完整对象进 state**（本 spec 选）：适合几十~几百个小对象。**本量级用 A。**
- **B. 轻量引用进 state**（state 只存 path[]，元数据走 `/api/threads/{tid}/artifacts/meta` 按需取）：单 thread 出现**几千张图**且多轮累积致 state 达 MB 级时才升级。
- **C. state 不存清单**（前端直接问后端 artifacts API 列目录）：海量场景。

> 注意：本系统已在用 B/C 思想——`plan_charts.json`/`handoff_*.json` 是「重数据在文件、state 不存」。把完整 ArtifactMeta 进 state 略偏离该惯例，但**本量级 42 KB 扛得住，且前端 #3 画廊已按 `ArtifactMeta[]` 完整对象设计 + 虚拟化渲染**——改 B 等于推翻 #3 刚做完的契约，得不偿失。

**升级触发条件（写进 spec 供未来参考，本次不做）**：若监控发现单 thread `artifacts` 字段 > ~1 MB（约 2500+ 张，或多轮追问累积），升级 B——state 存 `path[]`，新增只读端点 `/api/threads/{tid}/artifacts/meta?paths=` 让前端按需取 ArtifactMeta，前端 `normalizeArtifact` 已是归一边界、改造面小。**本次量级远未触及，A 即可。**

---


## 四、风险与注意（Plan agent 核出，已坐实）

| 风险 | 结论 |
|---|---|
| 112 张同 chart_id per_subject 互覆盖 | **不成立**：`by_output` 用 output **路径**做 key（`present_file_tool.py:130-135`），112 张各有唯一 `_sN.png`，都命中、都升级、都不互吞（artifacts 去重键也是 path） |
| chart_files 路径未经 F1 改写、与 by_output key 不匹配 | **不成立**：`chart_files` 存 plan 原始 `output`（`run_chart_plan_tool.py:255`），未经 F1（F1 只改喂 worker 的 args），与 by_output key 字节相等必命中（`test_f1_t5` 已断言全 `/mnt/user-data/outputs/` 前缀） |
| Command 返回类型变更破坏 @tool/决策树/测试矩阵 | **零风险**：view_image 同款已在生产；ToolMessage content 同形 JSON → 决策树零感知；测试经 `_call` 解包后 T1-T14 零改 |
| artifacts 写入需中间件配合 | **不需要**：artifacts 只有 reducer 无消费中间件，前端经 LangGraph values 快照直读，写完即生效 |
| 重读盘解析 plan（_load_plan_charts 第二次读） | **接受**：守"别复制 SSOT" > 省一次毫秒级 parse；run_chart_plan 本就跑 ProcessPool 画图（秒级），parse 是噪声。**不要**手重建 by_output |

---

## 五、验证

1. **TDD**：先改 `test_run_chart_plan.py`（`_call` 解包 + 新断言）→ `make test` **应红** → 改 `run_chart_plan_tool.py` → 全绿（T1-T14 + F1 + M2 + 新断言）。
2. **裸导入两生产入口（强制，CLAUDE.md 铁律——新增惰性 import present_file 必须不闭环）**：
   ```bash
   cd packages/agent/backend
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 0 退出才算过。**`pytest` 全绿是假绿**（conftest mock 了 executor，那条惰性 import 环不触发）；以裸导入 + 测试 T12（subprocess 裸导入）为准。
3. `make lint`。
4. **dogfood 真 EPM 数据**（owner 账号登录跑一次多 subject 分析）：前端画廊出 113 张 + charts_status 出失败摘要。验收看**真产物**不看 LLM 自述（HarnessX reward hacking 自检）。
5. **存量 thread 不回填**：本 fix 只对新跑分析生效；`0e72d605` 等老 thread 的 state.artifacts 已固化（除非重跑），不在范围。

---

## 六、关键文件

- `packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py`（主改：签名→Command、6 错误路径→Command、末尾登记 artifacts+charts_status+ToolMessage、惰性 import present_file 三 helper、注释钉死与 run_metric_plan 不对称）
- `packages/agent/backend/tests/test_run_chart_plan.py`（`_call` 解包 + tool_call_id + `TestArtifactsRegistered`）
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py`（prompt 第 10 步 + output_contract 文案「A 弱化版」，工具白名单不动）
- `packages/agent/skills/custom/ethoinsight-chart-maker/SKILL.md`（第 37 行第 7 步文案，与 chart_maker.py 同步防 drift）
- **参考模板（不改，照抄模式）**：`view_image_tool.py:49-116`（`@tool`+`InjectedToolCallId`+`-> Command` 1:1 模板）、`present_file_tool.py`（复用 `_load_plan_charts`/`_build_artifact_meta`/`_build_charts_status`/`_generate_thumbnail` SSOT）、`app/channels/manager.py:377-409`（IM 只认 present_files tool_call 的依据，决定 present_files 必须保留）

---

## 七、落地顺序

1. 改 `test_run_chart_plan.py`（`_call` 解包 + 新断言）→ `make test` 应红。
2. 改 `run_chart_plan_tool.py`（顶层 import + 签名 + 6 错误路径 + 末尾登记，缩略图先同步方案）。
3. `make test` 全绿。
4. **裸导入两入口必过。**
5. 改 `chart_maker.py` + `SKILL.md` prompt（A 弱化版），grep 三镜像点防 drift。
6. `make lint`。
7. dogfood 真 EPM 数据验前端画廊 + charts_status。

---

*依据：生产 thread `0e72d605` 取证（checkpoints.db 解码 + langgraph API state）+ 亲读 `run_chart_plan_tool.py`/`present_file_tool.py`/`executor.py`/`task_tool.py`/`manager.py` + Explore×3 + Plan agent 设计（核出 7 个考虑不周点，关键：错误路径也返 Command、IM 只认 present_files 故 present_files 保留）+ memory `feedback_code_has_fix_not_equal_bug_eliminated`/`feedback_chart_reconcile_loop_key_must_be_unique`/`feedback_conftest_mock_hides_circular_import`/`feedback_deterministic_batch_execution_tool_pattern`。承继 2026-06-24 run_chart_plan deterministic-execution spec。*
