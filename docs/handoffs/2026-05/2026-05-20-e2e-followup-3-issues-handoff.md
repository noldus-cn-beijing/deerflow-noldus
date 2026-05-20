# 2026-05-20 stepwise 删除后 E2E 暴露的 3 个独立问题 Handoff

> **前置上下文**:本文档承接 [2026-05-19-stepwise-gate-redesign-handoff.md](2026-05-19-stepwise-gate-redesign-handoff.md) 的 2026-05-20 决议(commit `d5b52738`)。stepwise gate 已确认 v0.1 不做,4 个原 bug(B0/B1/B2/B7)全部消失。**本文档是删除后 E2E 验证暴露的 3 个新问题,跟 stepwise 完全无关,独立处理**。
>
> **每个问题应作为独立 PR**,**不要捆绑**。

---

## E2E 验证背景

2026-05-20 删除 stepwise gate 后跑了一次完整 EPM 单样本 E2E:

- 用户:"我刚做完高架十字迷宫实验,帮我分析下数据"
- 上传:`轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt`(注意文件名中**有 5 个连续空格**)
- thread_id: `8dfe4943-6280-418b-8b2a-40a3d8622e37`
- run trace 完整跑通:Gate 1 范式确认 → prep_metric_plan → code-executor(5/5 指标)→ 并行 data-analyst + chart-maker → report-writer → 交付 `report.md` + `trajectory_plot.png`

完整 trace + log 在用户本机 `packages/agent/logs/langgraph.log`。

---

## 问题 1 — Lead 派 task 时传宿主机绝对路径,subagent 撞墙才改沙盒虚拟路径

### 症状

trace 中 **code-executor 和 chart-maker 各跑了一次"先用宿主机路径 → 失败 → 改沙盒路径 → 成功"的浪费循环**。

code-executor 第一次的命令(直接来自 log,`SandboxAudit verdict=pass`,**真的执行了**):

```bash
python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio \
  --input "/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/cd95effa.../user-data/uploads/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt" \
  --output /mnt/user-data/workspace/m_open_arm_time_ratio.json
```

UI 文字说"路径被拦截了",但 backend log **verdict=pass**,**不是 Guardrail 拦截**。需要 debug 才能确认真正失败点 — 候选根因:

1. **文件名含 5 个连续空格**(EthoVision 命名常见),宿主机绝对路径在 shell 里 escape 后 Python `open()` 拿到的字符串不对
2. **EthoVision parser 不接受软链接**(sandbox local 模式是否把 host 路径暴露成软链接需要确认)
3. **lead 输出路径时把宿主机路径填进 `--input`,subagent 收到后,sandbox 路径翻译 `replace_virtual_paths_in_command` 不识别宿主机路径**(只识别 `/mnt/user-data/...` 虚拟路径)→ shell 直接拿宿主机路径执行 → 失败原因取决于 sandbox 实现细节(local vs aiosandbox 差异)

### 根因来自哪一步

不是 subagent 自己拼错,是 **lead 在派遣 task 时**就把宿主机路径写进了 task prompt:

```
lead AIMessage: task(subagent_type="code-executor", prompt="...请用以下文件:
/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/.../user-data/uploads/<file>")
```

subagent 收到 prompt 后照抄进 `--input`。

### Lead prompt 已声明 uploads 路径

[`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:403`](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L403):

```
- User uploads: `/mnt/user-data/uploads` - Files uploaded by the user (automatically listed in context)
```

**但是** `<uploaded_files>` 注入(由 `UploadsMiddleware` 完成)给出的是哪种路径?需要核 — 如果是宿主机绝对路径,lead 照抄,问题出在 UploadsMiddleware。如果是 `/mnt/user-data/uploads/<file>` 虚拟路径,lead 还是拷宿主机路径,问题出在 prompt 强度 + LLM 自觉。

### 推荐 debug 路径

1. **复现**:用同样含连续空格的 EthoVision 文件名跑一次 E2E,看 `<uploaded_files>` 注入的具体字符串
2. **定位**:`grep -n "uploaded_files\|UploadsMiddleware" packages/agent/backend/packages/harness/deerflow/agents/middlewares/uploads_middleware.py`,确认注入格式
3. **修复(候选)**:
   - 如果注入的是宿主机路径 → 改 UploadsMiddleware 注入沙盒虚拟路径 `/mnt/user-data/uploads/<file>`
   - 如果已经是虚拟路径但 lead 还是写宿主机路径 → lead prompt 强约束"派 task 时 `--input` 必须用 `/mnt/user-data/uploads/<file>` 沙盒路径,绝不写宿主机路径"
4. **写 test**:`tests/test_uploads_middleware.py` 加一个 case 验证注入路径以 `/mnt/user-data/uploads/` 开头

### 工程量估算

1-3h。**修复完后跑 E2E 验证两个 subagent 都不再撞墙**。

### 严重程度

中。E2E 能跑通,只是浪费 1 个 tool call + 几秒。但**对用户体验显得"agent 在摸索"** — 跟 lead 应该展现的"专业性"冲突。

---

## 问题 2 — EPM 范式 chart catalog 缺图,fallback 只覆盖 1/3 用户需求

### 症状

chart-maker trace:

```
charts 为空(0 个 catalog 命中),fallback 有 2 个:trajectory_plot 和 timeseries_plot
用户要 3 张图:
  - 轨迹图(分区域着色:开臂/闭臂/中心区)        ✅ trajectory_plot 命中
  - 开臂时间占比图                              ❌ 无对应脚本
  - 各区域进入次数分布图                        ❌ 无对应脚本
最终 charts_generated: 1, failed_charts: 2
```

### 根因

**EPM catalog yaml 注册了 chart,但没有对应的 plot 脚本**。

[`packages/ethoinsight/ethoinsight/catalog/epm.yaml:65-67`](../../packages/ethoinsight/ethoinsight/catalog/epm.yaml#L65):

```yaml
charts:
  - chart_id: <something>
    script: ethoinsight.scripts.epm.plot_box_open_arm
```

具体内容需要打开 yaml 看,但可以确认 EPM **只有 1 个 chart 注册**,而且 `plot_box_open_arm` 是组间箱线图(单样本不适用,被 catalog.resolve 过滤掉了)。

`_common` fallback([`_common.yaml`](../../packages/ethoinsight/ethoinsight/catalog/_common.yaml)):
- `plot_trajectory`(轨迹图)
- `plot_timeseries`(时间序列)

**单样本场景下,catalog 0 命中 + fallback 只能给 1 张轨迹图**。缺 EPM 最基础的 2 张图:
1. **开臂时间占比单值图**(柱状或饼图,Subject 1 vs 参考区间)
2. **各区域(开臂/闭臂/中心)进入次数分布图**(柱状图)

### 推荐修复路径

属于 **EPM 范式补全 scope**(CLAUDE.md 第 4 条 Phase 0 / 第 10 条范式补全)。

1. 在 `packages/ethoinsight/ethoinsight/scripts/epm/` 下新建:
   - `plot_open_arm_time_ratio_bar.py`(读 `m_open_arm_time_ratio.json`,生成柱状图,可叠加参考线)
   - `plot_zone_entry_distribution.py`(读 `m_open_arm_entry_count.json` + 总入次,生成区域入次分布柱状图)
2. 在 [`epm.yaml`](../../packages/ethoinsight/ethoinsight/catalog/epm.yaml) `charts:` 段注册这两个 chart,**支持单样本场景**(catalog 字段需要有"single-subject ok"标记 — 需要核 catalog schema)
3. 同步检查 OFT 范式(`oft.yaml`)是否同样缺 chart
4. 跑 E2E 验证 chart-maker `charts_generated: 3, failed_charts: 0`

### 工程量估算

2-4h(2 个 matplotlib 脚本 + catalog 注册 + 测试)。可能涉及 catalog schema 扩展(看现有是否区分单样本/组间 chart),如需要 schema 改动 +1-2h。

### 严重程度

中。报告里有 trajectory_plot 一张图也能交付,但用户期望 EPM 标配图缺失 → 报告完整性打折扣。**v0.1 9 月里程碑要求 EPM 范式完整**,这是范式补全的一部分。

### 关联文档

- [docs/specs/paradigm-analysis-tools-spec.md](../../docs/specs/paradigm-analysis-tools-spec.md)
- [docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md](../../docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md)

---

## 问题 3 — 单样本场景 `statistical_validity` 字段语义不准确(轻)

### 症状

code-executor 跳过 statistics 步骤(单样本无法做组间检验,**这是正确行为**),但 handoff JSON 里写:

```
[gate_signals] statistical_validity: ok
```

**没做统计 ≠ 统计 OK**。应该是 `skipped` / `not_applicable` / 类似明确的 "不适用" 状态。

### 当前 contract 定义

[`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py:70`](../../packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py#L70):

```python
statistical_validity: Literal["ok", "warning", "failed"] = "ok"
```

**只有 3 个枚举值,没有 `skipped`**。

[`code_executor.py:81`](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py#L81) prompt 解释:
```
- ok = 统计结果可用
- warning = 警告(如 n<5)
- failed = 统计完全失败
```

**没有 "单样本不做统计" 的状态**。

### 推荐修复

1. 扩 enum:`Literal["ok", "warning", "failed", "skipped"]`
2. 更新 [`code_executor.py:81-89`](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py#L81) prompt 解释,加 `skipped = 单样本或 n_per_group < 2,无可比组,未运行统计检验`
3. 同步 [`data_analyst.py:110-114`](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py#L110)、[`report_writer.py:225`](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py#L225) prompt 提到 statistical_validity 处加 `skipped` 含义
4. 写 test:`tests/test_handoff_schemas.py` 加单样本 case,验证 `statistical_validity="skipped"` 合法
5. 检查 report-writer 模板是否在 `statistical_validity == "skipped"` 时正确生成"不做组间推断"的局限性段落(目前看 trace 报告里有"局限性"段落,但表述需要 align)

### 工程量估算

30min-1h(纯 schema + prompt 微调 + 1 个 test case)。

### 严重程度

轻。**不影响功能正确性**,只是 handoff 语义更精确。当前 lead 看到 `ok` 也没误判(report 里写了"单样本无法做统计推断"),所以不紧急。

---

## 三个问题独立性 + 优先级建议

| # | 问题 | 严重 | 工程量 | scope |
|---|---|---|---|---|
| 1 | Lead 派 task 传宿主机路径 | 中 | 1-3h | 横切(影响所有 subagent)|
| 2 | EPM chart 补全 | 中 | 2-4h | EPM 范式补全(v0.1 硬指标)|
| 3 | statistical_validity 加 skipped | 轻 | 30min-1h | handoff contract 精化 |

**互不依赖**,建议**三个独立 PR**:
- **优先 #1**(影响用户体验最直接,且影响所有范式不止 EPM)
- 然后 **#2**(v0.1 9 月硬指标依赖 EPM 完整)
- 最后 **#3**(锦上添花,可顺手做)

---

## 上下文 / 不要混淆

- **B0(lead 并行派 2 task)是设计意图,不是 bug**(见 2026-05-19 stepwise handoff 决议章节);如果新 agent 看到 trace 里 lead 一次派 2 个 task,**不要"修"**
- **stepwise gate 不要重新实现**,v0.x 才考虑;真要做时设计指针在 2026-05-19 stepwise handoff 末尾
- **PR #14 的 A-update**(lead 收到 task ToolMessage 后必须先用一行 progress 报告关键数字 + 下一步)已经合入,**保留**;它是 auto mode 的进度透明感来源,不要回退
- 路径问题(#1)**不是** sandbox guardrail 拦截 — log 显示 `verdict=pass`,真实失败点在 Python 端(parser 或文件 IO),debug 要从 stderr / Python 异常入手,不是查 guardrail policy

---

## 参考

- 上一份 handoff:[2026-05-19-stepwise-gate-redesign-handoff.md](2026-05-19-stepwise-gate-redesign-handoff.md)(末尾"2026-05-20 决议"章节)
- stepwise 删除 commit: `d5b52738 revert(stepwise-gate): v0.1 不做 stepwise gate,删 PR-2 实现`
- E2E thread: `8dfe4943-6280-418b-8b2a-40a3d8622e37`(本机 `.deer-flow/threads/` 下保留完整 trace)
- E2E log: `packages/agent/logs/langgraph.log`(本机)
