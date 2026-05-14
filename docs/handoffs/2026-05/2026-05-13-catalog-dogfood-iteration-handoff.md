# 2026-05-13 catalog 架构 dogfood 修复迭代交接

## 当前任务目标

完成 catalog 架构（spec 2026-05-13）的端到端 dogfood 验证，修复每一轮真实使用暴露的 bug，让整条流水线 lead → code-executor → data-analyst → report-writer 跑通真数据闭环。

**当前阶段**：lead 侧 + code-executor 侧已验证打通。**data-analyst / report-writer 侧尚未观测**。

## 当前进展

### ✅ 已完成 — 本会话产生的修复（按时间顺序）

| commit | 性质 | 修复内容 |
|--------|------|----------|
| `62d31893` | docs | 5-13 同事反馈校正 5-12 review 包（review.html 加红色横幅、README + 5-12 交接文档加 blockquote 校正声明） |
| `6aaa76ba` | refactor | 删 `assess.py` 的 `_DEFAULT_THRESHOLDS`、`_load_thresholds`、`_assess_thresholds` 死阈值代码（同事 Q1/Q4 哲学硬要求）。worktree agent 漏 commit、从工作树捞回 |
| `19086d67` | chore | ruff format 全仓 68 文件、零逻辑变化、纯格式化 |
| `edcd425d` | test | 新增 `tests/test_metric_catalog_live.py`（5 个断言层次的 live e2e、CI skip） |
| `232424c5` | fix | feedback 接口 404 第一层：拆掉 run_store 存在性校验（run_store 全仓零 producer、永远 None） |
| `9d28bbe1` | fix | feedback 接口 404 第二层：装饰器去掉 `require_existing=True`（threads_meta 在 LangGraph 直连模式下永远空） |
| `bb8bf627` | fix | memory 跨 thread 幻觉：注入侧砍 topOfMind + history.* + 写入侧加禁令列表（覆盖文件上传、单次分析、pending action、数值发现 4 类） |
| `a0fe5074` | fix | GuardrailMiddleware 重名（langchain 升级引入唯一性约束、subagent 100% 派遣失败 "Please remove duplicate middleware instances"） |
| `b36e315a` | fix | code-executor max_turns 12→20 + workflow 加 `<critical_rules>` 段（critical_rules 禁令"不要探索 plan.json 以外的脚本"、"plan.charts=[] 不是邀请补图"、"优先写 handoff"） |

### ✅ 验证通过 — catalog 架构端到端机制层

通过浏览器 dogfood 验证、对话 + langgraph.log 双源确认：

1. **Gate 1 范式识别**：lead 通过 ev19_template + paradigm 锁定 → ✅
2. **Gate 2 数据完整性 + 反问链路**：单只无对照 → ask_clarification → ClarificationMiddleware 拦截中断 turn → ✅
3. **lead bash 触发 catalog 工作流**：dump_headers → columns.json → catalog.resolve → metric_plan.json → ✅
4. **code-executor 派遣 + 严格按 plan 走**：read plan.json → 逐条 bash 5 个 `python -m ethoinsight.scripts.epm.compute_*` → 不再探索 charts skill → ✅
5. **handoff_code_executor.json 写盘 + [gate_signals] 输出**：✅
6. **lead 收 gate_signals 后正确决策**：检测 statistical_validity=failed → 没盲目派遣下游 subagent → ask_clarification 让用户选择方向 → ✅
7. **memory 跨 thread 隔离**：新 thread 不再幻觉旧 thread 的上传文件 → ✅
8. **判读哲学**：lead 未对单只数据下"焦虑表型偏高/偏低"绝对判读 → ✅

### ⏸️ 等待观测 — data-analyst / report-writer 侧

最后一轮 dogfood 用户选了"暂时只看单只描述性结果就够了"作为 ask_clarification 的答复。**这次回复后 lead 应该派遣 data-analyst + report-writer 走"单只描述性"路径**，但**新会话开始前还没继续**。

需要观察：

1. **data-analyst 是否真的 read catalog YAML 取 `direction_for_anxiety`** 描述指标语义（"开放臂时间比例越低焦虑样回避越明显"），还是凭印象写
2. **report-writer 是否真的 read catalog YAML 取 `display_name_zh / unit_zh / one_liner`** 翻译指标 id，还是脑补中文
3. **data-analyst 判读语言**：单只无对照场景下会不会**违规**说"7.99% 偏低"。这是哲学测试关键场景 — 单只数据正确做法是**只描述数值不下方向判断**

## 关键上下文

### 项目状态

- **路线图**：v0.1 9 月可用版本（[docs/roadmap.md](docs/roadmap.md)）
- **当前阶段**：Phase 0 末尾 - EPM 范式 + 鲁棒性 + 基础设施
- **catalog 架构实施**：完成 + dogfood 验证中
- **dev 分支**：领先 origin/dev 9 个 commit（本会话的修复 + assess.py 清扫 + ruff format + e2e 测试）

### 仓库结构关键路径

```
docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md  # 当前 catalog spec
docs/superpowers/plans/2026-05-13-metric-catalog-implementation.md       # 实施 plan（已执行）
docs/review-packages/2026-05-12-feedback.md                              # 同事 5-13 反馈核心
docs/review-packages/2026-05-12-real-data-metrics/review.html            # review 包（已加 5-13 校正横幅）
docs/handoffs/2026-05/2026-05-13-metric-catalog-implementation-handoff.md # worktree agent 上一份交接
docs/handoffs/2026-05/2026-05-12-real-data-metrics-verified-handoff.md   # 5-12 真数据交接（已加 5-13 校正补遗）

packages/ethoinsight/ethoinsight/catalog/   # catalog 模块（YAML × 7 + schema/loader/resolve/cli）
packages/ethoinsight/ethoinsight/parse/     # parse 包（dump_headers CLI + _core.py）
packages/agent/skills/custom/ethoinsight-metric-catalog/  # 新 skill（按 role 分段读取指引）
packages/agent/backend/packages/harness/deerflow/guardrails/middleware.py  # 本次 a0fe5074 改的 GuardrailMiddleware
packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py  # 本次 b36e315a 改的 max_turns + critical_rules
packages/agent/backend/packages/harness/deerflow/agents/memory/prompt.py  # 本次 bb8bf627 改的 format_memory_for_injection + MEMORY_UPDATE_PROMPT
packages/agent/backend/app/gateway/routers/feedback.py  # 本次 232424c5 + 9d28bbe1 改的反馈路由
```

### 当前 dogfood thread 状态

- thread_id: `0cde783d-6367-4225-8043-5748209879d1`（如果用户继续在原 thread）
- 已上传文件：`/mnt/user-data/uploads/轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt`（注意：文件名含多个连续空格）
- workspace 已生成：`columns.json`、`raw_files.json`、`groups.json`、`metric_plan.json`、`m_open_arm_*.json` × 5、`handoff_code_executor.json`
- EPM Subject 1（Drug 组）实测数值：
  - open_arm_time_ratio = 7.99%
  - open_arm_time = 23.56s
  - open_arm_entry_count = 6
  - open_arm_entry_ratio = 28.57%
  - total_entry_count = 21
- 反馈表已记录 1 条（verdict=correct）— SQLite `feedback` 表

### 工作树未提交内容（接手者**不要动**）

- `docs/specs/llm-finetuning-strategy.md`（Qwen3-30B 升级备忘的引子段）
- `docs/plans/2026-05-13-base-model-decision-memo.md`（新 memo 文件）
- `packages/agent/frontend/src/app/page.tsx`（别人的工作）

这三个文件**跟本会话工作无关**，让用户自己 commit 或决定。

## 关键发现

### A. 修复分类的硬规则

dogfood 暴露的 5 个 bug 性质分类：

| Bug | 性质 | 修复后是否彻底解决 |
|-----|------|---------------------|
| feedback 404 × 2 层 | 确定性架构 bug | 彻底解决 |
| GuardrailMiddleware 重名 | langchain 升级引入的硬约束 | 彻底解决 |
| code-executor max_turns 不够 | 确定性预算问题 | 彻底解决 |
| code-executor 探索 chart 浪费 turn | **LLM 随机性** | 概率压制 + 余量兜底，**不保证 100% 不复发** |

**意思**：前 4 个永远不会再触发；最后一个未来仍可能少量浪费 turn，但 max_turns=20 兜底后不会再致命。

### B. catalog 架构的 single source of truth 已严格执行

- catalog YAML × 7 范式：`packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml`
- Q6 白名单（EPM/OFT/FST）由 `tests/test_catalog.py::test_catalog_default_metrics_match_q6_whitelist` 反退化测试锁死
- 每个 catalog script 字段 importable 由 `test_all_catalog_scripts_are_importable` 反退化测试锁死
- 任何未来的修改如果想动 Q6 白名单或脚本路径，**必须显式改测试** — 这是设计就要的

### C. Memory 隔离设计哲学

- **写入侧（MEMORY_UPDATE_PROMPT）**：禁止 4 类会话级状态进 memory（文件上传 / 单次分析 / pending action / 数值发现）
- **读取侧（format_memory_for_injection）**：硬砍 `topOfMind` 和 `history.*` 字段、只注入 `user.workContext / personalContext / facts`
- **数据库**：`memory.json` 仍保留这些字段（updater 输出 schema 兼容），但**不进入 system prompt**
- 现有 `memory.json` 已清理污染（topOfMind / recentMonths 清空），备份在 `memory.json.bak-2026-05-13`

## 未完成事项

### 🔴 阻塞 — 需要立刻验证的事

1. **data-analyst 派遣 + 输出验证**：让 lead 在当前 thread 继续推进（用户已答"暂时只看单只描述性结果就够了"）。观察 data-analyst 是否真读 catalog 取 `direction_for_anxiety`、判读语言是否符合"组间比较哲学"
2. **report-writer 派遣 + 中文展示验证**：观察 report-writer 是否真读 catalog 取 `display_name_zh / unit_zh / one_liner`，还是凭印象翻译

### 🟡 中优先级 — 已知未做但 spec 列了的事

3. **shoaling 多文件场景支持**：当前 `catalog.resolve` 用 `raw_files[0]`，shoaling 多文件 wrapper JSON 未实现（spec §10.2 列了、本会话未做）
4. **OFT 列名歧义反问场景**：spec 设计了"裸 in_zone → resolve 报 `columns_missing` → lead 反问"路径，本会话只用 EPM 数据 dogfood，**这条路径未观测**。下次用真 OFT 数据（`旷场_小鼠_三点/`）测
5. **多只对照场景**：本会话只测了 n=1，**完整端到端 + 真统计** 路径未观测。需要上传 EPM 2 只 Drug + 2 只 Saline 测组间分析

### 🟢 低优先级

6. **TodoList plan mode 跳过"trivial 问答"**：用户上一轮注意到"两次 think"现象，决定先不改。如果未来端到端 UX 优化时回来做
7. **memory.json `workContext` 末尾的"刚做完 EPM 实验"措辞**：仍有轻微会话状态残留。已通过 dogfood 验证下次 memory updater 会自己重写更干净，不需要手动改
8. **golden-cases 落盘**：本次 EPM 真数据指标（7.99% / 23.56s / 21 entries）可固化为 golden-case 候选。等同事 review 数字合理性后再做

## 建议接手路径

### 第一步（最重要）— 继续 dogfood 验证 data-analyst / report-writer

**前置**：用户上次回复"暂时只看单只描述性结果就够了"是关键 ask_clarification 答复。新会话起来后，第一步应该是：

1. 读这份交接 + 5-13 spec + 5-13 implementation handoff
2. 看 langgraph.log 确认上一次 lead 收到这个回复后做了什么（**很可能 lead 已经派遣了 data-analyst 但用户还没看到结果**）：

```bash
tail -300 packages/agent/logs/langgraph.log | grep -E "data-analyst|report-writer|reached max|completed" | tail -20
```

3. 让用户在浏览器**刷新当前 thread**或**重新发一条引导消息**让 lead 继续，例如：
   > "刚才那个回复看到了吗？请继续走单只描述性分析路径"

### 第二步 — 观察 data-analyst 行为

重点观察 reasoning 面板：
- ✅ data-analyst 是否 `read_file packages/ethoinsight/ethoinsight/catalog/epm.yaml`（或类似路径）取 `direction_for_anxiety`
- ❌ data-analyst 是否凭印象写"open_arm_time_ratio 越低越焦虑"（这是 catalog 字段、不该硬编码进 prompt）
- ❌ data-analyst 是否对单只数据违规说"7.99% 偏低 → 高焦虑表型"（违反同事 Q1/Q4 哲学）

如果发现 data-analyst 没读 catalog，可能需要在它的 system prompt 加更强的"必读 catalog"指引。当前 prompt（`subagents/builtins/data_analyst.py`）只加了一段"判读时 read catalog YAML 取 direction_for_anxiety / statistical_default"。

### 第三步 — 观察 report-writer 行为

重点观察：
- ✅ report-writer 是否 `read_file ... epm.yaml` 取 `display_name_zh`
- ✅ 报告中 `open_arm_time_ratio` 翻译为 catalog 里写的"开放臂时间比例（比例）"还是别的
- ❌ 报告是否含违禁词"正常范围 / Reference range / 文献典型 / 常模"（`test_metric_catalog_live.py::test_report_does_not_use_absolute_threshold_language` 自动化测试覆盖这一项）

### 第四步 — 上传真分组数据测组间

如果单只路径跑通了，让用户上传 EPM 4 只（2 Drug + 2 Saline）测**真组间统计路径** — 这才会触发 `run_groupwise_stats` + Shapiro + t/Mann-Whitney + Cohen's d → data-analyst 真正消费组间结果做判读。

### 关键命令速查

```bash
# 看 git 历史
git log --oneline -15

# 看 dogfood 跑的脚本顺序
tail -300 packages/agent/logs/langgraph.log | grep -E "SandboxAudit|subagent|completed"

# 看反馈表内容（训练数据飞轮）
cd packages/agent/backend && make training-stats

# 看 memory 状态
cat packages/agent/backend/.deer-flow/.ethoinsight/memory.json | head -50

# 手动验证 catalog CLI
cd packages/ethoinsight && python -m ethoinsight.catalog.resolve \
    --paradigm epm \
    --columns-file /tmp/columns.json \
    --raw-files-json /tmp/raw_files.json \
    --workspace-dir /tmp/ws \
    --output /tmp/plan.json
```

## 风险与注意事项

### 🚨 千万不要做的事

1. **不要 commit 工作树里的 3 个无关文件**（Qwen3 备忘 / page.tsx）— 不是本会话工作
2. **不要重启时强制 `make stop` 后没等够时间就 `make dev`**— uvicorn `--reload` 需要几秒清缓存。重启后第一次请求可能仍走旧代码
3. **不要修改 catalog YAML 的 Q6 白名单字段**而不同步改 `test_catalog_default_metrics_match_q6_whitelist`— 反退化测试会立刻 fail，但这是 design 就要的硬约束
4. **不要在 data-analyst / report-writer 的 system prompt 里硬编码任何指标的中文名 / 方向性** — 这违反 single source of truth（用户 5-13 反复强调"千万不要有两份知识打架"）
5. **不要把 memory `topOfMind` 或 `history.*` 注入回 system prompt** — 这是跨 thread 隔离的核心防御

### ⚠️ 容易混淆的点

1. **三个文件夹里都有 EPM 相关数据**：
   - `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/` — 同事 5-12 给的真数据
   - `/mnt/user-data/uploads/` — sandbox 虚拟路径、映射到 thread workspace
   - `packages/agent/backend/.deer-flow/users/.../threads/.../user-data/uploads/` — 物理对应路径
2. **文件名含连续空格**：`轨迹-Elevated Plus Maze XT190-Trial     1-Arena 1-Subject 1.txt`（Trial 后面 5 个空格）。bash 时记得加引号
3. **两个 backup 文件**：
   - `memory.json.bak-2026-05-13`（步骤 a 清理时备份）
   - `.deer-flow.bak.round3-20260507-185139/` 等老备份目录里有大量 EPM 训练数据样本，**不要污染当前 grep 结果**（grep 时用 `--exclude-dir=.bak`）
4. **plan mode 设计的"两次 think"现象不是 bug** — 用户上一轮已确认接受这个 UX

### 已被推翻的判断 / 不要回头做的事

- ❌ 不要按 5-12 review 包的 Q4 假设去补 `mobility_continuous` 算法路径（整题作废）
- ❌ 不要按 5-12 review 包的 Q5 把 Mobility_1/Mobility_10 当阈值处理（真相是采样率）
- ❌ 不要在 OFT 的 `_find_center_zone_column` 加回裸 `in_zone` silent fallback（同事 Q2 明确禁止）
- ❌ 不要给 `feedback.py` 加回 `require_existing=True` 或 run_store 校验（threads_meta / run_store 都没 producer）

## 下一位 Agent 的第一步建议

```bash
# 1. 读这份交接（你正在做）+ 关键设计文档
cat docs/handoffs/2026-05/2026-05-13-catalog-dogfood-iteration-handoff.md
cat docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md | head -100
cat docs/review-packages/2026-05-12-feedback.md

# 2. 看 git 历史确认状态
git log --oneline -15

# 3. 看 dogfood 跑到哪儿了
tail -300 packages/agent/logs/langgraph.log | grep -E "subagent|completed|reached max" | tail -20

# 4. 看反馈表 + memory 状态
cat packages/agent/backend/.deer-flow/.ethoinsight/memory.json | head -30

# 5. 让用户继续 dogfood（在原 thread 重新引导）
# 用户原 thread_id: 0cde783d-6367-4225-8043-5748209879d1
# 关键观察点见 §"建议接手路径"
```

完成第 1-4 步之后，向用户提议第 5 步：**"我已掌握上次 catalog dogfood 状态，建议你在原 thread 继续推进，重点观察 data-analyst 是否真读 catalog 取 direction_for_anxiety、判读语言是否符合组间比较哲学"**。
