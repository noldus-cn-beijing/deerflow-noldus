# Handoff: 2026-06-16/17 三 spec 写作+review（io-boundary 善后 / statistics 列对齐 / seal 漏调结构性根治）

> 会话主线：接 io-boundary review handoff → 修第三层 bug → 写 statistics 列对齐 spec → 写 seal 漏调结构性根治 spec → review 两个已实施分支（statistics + seal-gate）。
> 交接对象：下一个接手的 AI Agent。
> 当前位置：主仓 `/home/wangqiuyang/noldus-insight`（分支 dev，HEAD `046a4f3a`）。

---

## 1. 当前状态速览（三条线）

| 工作 | 分支 | commit | 状态 |
|---|---|---|---|
| 第三层 file→subject 桥接 | feat/ethoinsight-io-boundary-and-aggregator | 9d84d6f4 | ✅ **已合 dev**（PR#139） |
| statistics 路径列对齐 | feat/statistics-path-column-alignment | 7cf58145 | ✅ **已合 dev**（PR#141），我 review 通过 |
| seal 漏调结构性根治（SealGateMiddleware） | feat/seal-gate-middleware | 7aa577a3 | ⚠️ **已 push，review 通过，待用户建 PR** |

**dev 现状 HEAD = `046a4f3a`**（已含前两条）。**seal-gate 是唯一待建 PR 的**。

---

## 2. 未完成事项（按优先级）

### 🔴 高优先级

1. **seal-gate 建 PR 合 dev**：`feat/seal-gate-middleware`（7aa577a3）已 push、我 review 忠实通过（详见 §4）。基线 `046a4f3a`，零文件冲突。用户建 PR 即可。

2. **4 个 untracked spec 入库**：主仓 dev 工作区有 4 个 untracked spec（本会话 + 上轮产出），需 commit 进 dev（参考前例：handoff 入库 commit）：
   - `docs/superpowers/specs/2026-06-16-io-boundary-symmetry-and-aggregator-spec.md`
   - `docs/superpowers/specs/2026-06-16-statistics-path-column-alignment-spec.md`
   - `docs/superpowers/specs/2026-06-16-seal-gate-middleware-engineering-spec.md`（**主 spec**，两层工程方案）
   - `docs/superpowers/specs/2026-06-16-seal-produce-deliver-merge-spec.md`（**已被取代**，纯 prompt 版，顶部已标注被 seal-gate spec 取代，保留作 L2 prompt 细节参考）

3. **seal-gate 合并后 dogfood 复跑（治本终验）**：用本次失败的同一数据 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（control=7/treatment=21，列名 open/closed），跑三场景（正常 statistics / 空 statistics / n<3），确认 data-analyst **三种都一次发出 seal tool_call**、不再 `terminated without emitting`。观测 gateway.log 里 `SealGateMiddleware: seal reminder` 注入次数（应随 L2 prompt 改善趋近 0 = L2 见效探针）。

### 🟡 中优先级

4. **memory 沉淀**（用户问过"要不要沉淀"，我还没写）：本会话三个值得沉淀的点：
   - **seal 漏调结构性根治已落地**：真根因=ReAct"无 tool_call=结束"+ prompt 把"产出分析(思考成文)"与"交付(调seal誊抄)"拆两步、seal 参数=产出内容→模型产出完就以为完成跳过冗余 seal。历史 4 次 prompt/skill/schema 修复都在打地鼠（修诱因）。治本=SealGateMiddleware（after_model + jump_to='model' 拦在终止前，结构性 100%）+ L2 prompt 合一。复刻已有 ParadigmIdentificationGateMiddleware 模式。**关键认知：gate 是主防线非兜底（结束前拦≠结束后补）**。
   - **review 铁律再实证**：worktree **有独立 venv 时必须用它跑测试**（不是主仓 venv）——我两次用错（主仓 venv 跑得 ModuleNotFound / 错误 import 方式得假"假绿"）。seal-gate worktree `/home/wangqiuyang/noldus-insight-seal-gate/packages/agent/backend/.venv` editable 指 worktree。statistics 的 backend 测试用 `spec_from_file_location(__file__ 相对路径)` 显式加载 worktree 源（正确范例）。
   - **test_subagent_executor.py 15 failed 是 dev 基线债**：主仓干净 venv 同样 15 failed（`ModuleNotFoundError: deerflow.agents.middlewares is not a package`，是测试隔离副作用，单独 import experiment_context OK，非生产问题）。回退 executor 到 base 失败数不变=非任何分支引入。下次 review 改 executor 的分支别被这 15 failed 误导。

### 🟢 低优先级（已知遗留，非本会话范围）

5. **FewZones 列对齐第四轴的 EPM 专属指标**：statistics 列对齐 spec 已修"接线"，但真实 FewZones 数据的 EPM 专属指标（open_arm_time_ratio 等）算对，依赖列对齐参数已传到 dispatcher（PR#141 已做）。第三层桥接（PR#139）+ 列对齐（PR#141）叠加后，真实数据 comparisons 才完整——**这两个已合，理论上闭环，待 dogfood 复跑确认**。

---

## 3. 关键发现（本会话核心结论）

### seal 漏调的真根因（最重要，反复确认）
- subagent 是 ReAct agent（`create_agent`），终止条件=模型输出一条**无 tool_call** 的消息。模型写"分析完成"纯文本就退出，seal 没机会调。
- prompt 的"step 3 必须 seal"无法强制——ReAct 不认 step 顺序，只认有没有 tool_call。
- data-analyst prompt 有 **7+ 处**"seal 必达/别忘"哀求补丁 = 结构病指纹（产出/交付分离）。
- **唯一能改终止条件的位置 = `after_model` hook**（langchain 1.2.3 原生 `@hook_config(can_jump_to=["model"])`，本仓库 `ParadigmIdentificationGateMiddleware` 是活范例）。
- code-executor 已经"产出=交付合一"（run_metric_plan 一个工具即算即落盘 handoff），结构上不漏——是治本的正面范例，**故意不改它**。

### 用户的关键立场（贯穿会话，必须遵守）
- **否决兜底**（"别想办法兜底写屎山代码"）：不加 auto-seal、不把 data-analyst 加进 `_AUTO_SEALABLE`、不改 seal-resume。gate 是"结束前拦"，本质不同于兜底"结束后补"。
- **要工程化落地不要概率性**（"改 prompt 99% 不够，要真正工程化"）：所以 L1 gate 是主防线（结构性 100%），L2 prompt 是辅助（让模型少被拦、省 turn）。

---

## 4. seal-gate review 已验证项（全通过，可信）

| 项 | 结果 | 证据 |
|---|---|---|
| L1 SealGateMiddleware 6 判据 | ✅ 忠实 | 复刻 ParadigmIdentificationGate；判据 1/2/3/4/5/6 全到位 |
| executor 挂载 | ✅ | append 在 LoopDetection 后 + 惰性 import（守导入环铁律） |
| L2 data-analyst 合一 | ✅ | step 2.7"思考只推导素材"+ step 3"产出=调seal"；删 seal 哀求补丁、保留实质约束 |
| L2 chart/report | ✅ | "产出后下一动作就是 seal" |
| 单元测试 | ✅ 17 passed | **用 worktree 独立 venv 跑**（editable 指 worktree） |
| 红→绿 | ✅ | 去掉判据 6 注入→核心测试红；恢复→绿 |
| 裸导入两入口 | ✅ exit 0 | app.gateway + make_lead_agent；import 环 guard（subprocess）passed=无新环 |
| 15 failed executor 测试 | ✅ 非本分支 | 主仓干净 venv 同样 15 failed=dev 基线债 |

---

## 5. 关键上下文（文件指针）

- **seal-gate worktree**：`/home/wangqiuyang/noldus-insight-seal-gate`（有独立 `.venv`，editable 指 worktree）
- **核心新文件**：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/seal_gate_middleware.py`（184 行）
- **范例模式**：`agents/middlewares/paradigm_identification_gate_middleware.py`（已有的活范例，SealGate 复刻它）
- **L2 改的三 prompt**：`subagents/builtins/{data_analyst,chart_maker,report_writer}.py`
- **测试**：`tests/test_seal_gate_middleware.py`（17 测试）
- **真实测试数据**：`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（28 文件，列名 open/closed，本会话 dogfood 失败源数据）
- **失败 dogfood thread**：`51ff5eb8`（statistics 空 + data-analyst 漏 seal，本会话诊断起点）

---

## 6. 风险与注意事项

1. **worktree 独立 venv 铁律**（本会话踩两次）：seal-gate worktree 有自己的 `.venv`，跑测试/裸导入**必须 `source <worktree>/.venv/bin/activate`**，不能用主仓 venv（editable 指主仓→ModuleNotFound 或假绿）。先 `cat .venv/lib/python3.12/site-packages/_editable_impl_deerflow_harness.pth` 确认指向。
2. **15 failed executor 测试是基线债**，别当回归。判据：主仓干净 venv 同样失败 + 回退 executor 到 base 失败数不变。
3. **既有 task_tool↔subagents 导入环**：手动裸 `import deerflow.subagents.executor` 会命中（主仓同样），但两**真生产入口** exit 0=无新环。验导入环看 app.gateway/make_lead_agent，别看手动裸 import。
4. **uv.lock 不进 commit**：worktree 新建 venv 改了 uv.lock（cryptography 依赖，dev 基线 sync 带的），review/commit 前确认它不被误提交。
5. **seal-produce-deliver-merge-spec.md 已被取代**：别照那份纯 prompt 版实施——它缺 L1 主防线。以 seal-gate-middleware-engineering-spec.md 为准（已实施）。

---

## 7. 下一位 Agent 的第一步建议

1. 读本 handoff + `docs/superpowers/specs/2026-06-16-seal-gate-middleware-engineering-spec.md`（了解两层方案）。
2. 确认 seal-gate 是否已建 PR（`git log --oneline origin/dev | grep seal`）——若未合，提醒用户建 PR。
3. **优先做 §2.2**（4 个 spec 入库 commit 进 dev）+ **§2.4**（memory 沉淀，用户问过）。
4. seal-gate 合并后做 §2.3 dogfood 复跑（治本终验）。
5. 若用户给新 dogfood 失败：先按 §3 真根因框架判断（是诱因还是结构），别再写"提醒 seal"的 prompt 补丁。
