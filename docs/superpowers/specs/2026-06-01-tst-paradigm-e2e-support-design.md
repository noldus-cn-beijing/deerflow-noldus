# 2026-06-01 设计 spec — 新增 TST（悬尾实验）范式端到端支持

**类型**：可实施版（已对 dev HEAD `065b4180` 核验现状；实施前 `git pull` 复核行号）
**对应**：CLAUDE.md「v0.1 支持 5 个范式」→ 扩为 6 个（新增 TST）
**估期**：工程部分 ~1-1.5 天（库层已就位，主要是工程接头 + 文档搬迁）；领域内容部分依赖行为学同事
**前置**：无硬前置（库层 catalog/metrics/scripts 已 100% 就位）

> **本 spec 的独特背景**：TST **不是从零实现**。盘点（2026-06-01 explore + 人工核实）发现 **ethoinsight 库层 100% 就位**（catalog/tst.yaml 8 指标、metrics/tst.py、scripts/tst/ 全套 10 脚本、loader 已注册 `tail_suspension→tst` 别名）。缺的全在**工程接头 + 范式知识部署**。这是典型的 [[feedback_ssot_skill_deployment_distinction]]「文件齐了 ≠ subagent 能用」。

---

## 0. 目标与原则

**目标**：让 agent 对悬尾实验（TST）的支持达到和 FST 完全相同的端到端水平：用户上传 TST 数据 → Gate 1 识别为「已支持范式」→ 指标计算 → 统计 → 解读 → 报告全链路跑通。

**真实数据**（实施 + dogfood 用）：`/home/wangqiuyang/DemoData/newdemodata/悬尾/`
- `原始数据-tstHelperDemoVideo-试验 1.xlsx` / `试验 2.xlsx`（EV19 导出，多观察区，~450KB）
- `轨迹-tstHelperDemoVideo-试验 1-观察区 1-对象 1.txt` / `试验 2-...txt`

**铁律**：
- **领域知识归同事/review-packages**，工程只做通路和搬迁（[[feedback_ssot_lives_in_review_packages]] / [[feedback_single_source_of_truth]]）。本 spec 涉及一处「文档矛盾」（见 §3.2），**工程不自行裁决领域问题，标记给同事**。
- **解读按范式独立，即使算法共用**（用户 2026-06-01 锁定 + forced_swim.md L34 同事铁律）。TST/FST 共用 pendulum 算法，但**各自一份独立的 pendulum 参数判据**。

---

## 1. 现状盘点（实施前必读 — 已核实）

### ✅ 已 100% 就位（库层，不需动）

| 环节 | 证据 |
|---|---|
| `catalog/tst.yaml` | 8 指标，direction_for_anxiety / statistical_default / chart 注册齐全，与 fst.yaml 对称 |
| `metrics/tst.py` | 3 函数，与 fst.py 对称 |
| `scripts/tst/` | 10 脚本（3 compute + 4 plot + run_groupwise_stats + __init__），比 FST 还全（多 plot_activity_intensity / plot_struggle_distribution） |
| catalog loader | `loader.py:61` 已注册 `"tail_suspension": "tst"` 别名；`load_all_catalogs()` 含 tail_suspension |
| EV19 模板映射 | `ev19_facts.py:62` `"tail_suspension": ["NoTemplate"]` |
| `experiment_context.py` 工具链 | `set_experiment_paradigm(paradigm="tail_suspension", ev19_template="NoTemplate")` 可接受、兼容性检查通过 |
| skill `by-experiment/tail_suspension.md` | 存在，42 行，段落结构完整（但有一处模板矛盾，见 §3.2） |

### 🔴 缺口（本 spec 要补）

| # | 层 | 缺口 | 文件 |
|---|---|---|---|
| G1 | EV19 识别 | **`SUPPORTED_PARADIGMS_V01` 白名单不含 `tail_suspension`** → Gate 1 判为「v0.1 未实现」 | `packages/ethoinsight/ethoinsight/ev19_facts.py`（`:80-86`） |
| G2 | Lead prompt | **Gate 1「已支持范式」列表写死 5 个不含 TST** + 反问文案「现已支持 5 个…EPM/OFT/LDB/FST/Zero Maze」不含 TST + TST 被列在「暂不支持…TST 等」 | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（约 `:265-282`，**受保护文件**，surgical 改） |
| G3 | 范式知识部署 | **TST 的 pendulum 参数判据未进 skill**。源在 `docs/review-packages/2026-0521-feedbacks/tstYoyo/tst-pendulum-algorithm.md`（真实存在 9683 字节），但 skill references 里没有 → subagent 在 sandbox 只能 read `/mnt/skills/`，read 不到 docs/ | 目标：`packages/agent/skills/custom/ethovision-paradigm-knowledge/references/...`（见 §3.3） |

### 🟡 质量完善（建议同批做，部分依赖同事）

| # | 缺口 | 文件 | 归属 |
|---|---|---|---|
| Q1 | `tail_suspension.md:13` 说模板是「PorsoltCylinder-NoZones」，但 `ev19_facts.py:62` 映射「NoTemplate」，且真实数据 tstHelperDemoVideo 非 Porsolt 圆筒 — **领域矛盾** | `by-experiment/tail_suspension.md` | **同事裁决**（见 §3.2） |
| Q2 | `by-template/NoTemplate.md` 所有 🟡 字段留白待补 | skill `by-template/NoTemplate.md` | 同事 |
| Q3 | 无 TST golden-case | `golden-cases/tail_suspension/` | 同事（用真实数据标注） |
| Q4 | CLAUDE.md（根 + backend）「v0.1 支持 5 个」声明 | 两个 CLAUDE.md | 工程（本 spec 末做） |

---

## 2. 实施前核验清单（实施 agent 开工必做）

1. `git pull && git log --oneline origin/dev -5` — 确认 dev HEAD，行号会漂
2. **跑通库层（不碰 agent，先证明分析内核对真实 TST 数据能跑）**：
   - 用真实数据手动跑一条 `python -m ethoinsight.scripts.tst.compute_immobility_time --input "<TST xlsx>::<sheet>" --output /tmp/x.json --parameters-json '{...}'`，确认脚本对真实 TST 数据不报错、产出合理（不动时间不应是 0.5s 那种异常值 — 若异常，可能命中和 FST 同款 pendulum 参数问题，见 §6 与 data-analyst spec 的关系）
3. `ev19_facts.py` 的 `SUPPORTED_PARADIGMS_V01` 真实结构（确认怎么加 tail_suspension）
4. `lead_agent/prompt.py` Gate 1 段的真实文案结构（确认「已支持 5 个」「暂不支持 TST 等」的确切行 + 怎么 surgical 改不动其他中文调度逻辑）
5. **真实 TST 数据走 `inspect_uploaded_file` 会识别成什么 EV19 模板** — 这决定 Q1 矛盾怎么解（NoTemplate vs PorsoltCylinder）。这是 go 的关键核实

---

## 3. 设计与实施锚点

### 3.1 G1 — SUPPORTED_PARADIGMS_V01 加 tail_suspension

`ev19_facts.py:80-86` 的 frozenset 加一行 `"tail_suspension"`。**1 行改动**。
- 配套测试：加 `test_tail_suspension_is_supported`（断言 `"tail_suspension" in SUPPORTED_PARADIGMS_V01` + `get_default_template_for_paradigm("tail_suspension")` 返回 NoTemplate）

### 3.2 G2 — Lead prompt Gate 1 范式列表（受保护文件，surgical）

`lead_agent/prompt.py`（约 `:265-282`）三处：
1. 「已支持范式」列表加「悬尾实验 TST」
2. 反问文案「现已支持 5 个范式:EPM/OFT/LDB/FST/Zero Maze」→「6 个范式:EPM/OFT/LDB/FST/Zero Maze/TST」
3. 「暂不支持…TST 等」里移除 TST（改用其他未实现范式举例，如 MWM/Barnes/Y-maze）

> **🔴 受保护文件纪律**（[[feedback_sync_protected_files_registry_loss]]）：prompt.py 含大量中文调度规则。**只改 Gate 1 这三处范式文案，逐字 diff 确认没动其他中间件/Gate/subagent 描述**。用 deepseek 正面提示（CLAUDE.md §6）：写「现支持 6 个范式…」而非「不要说不支持 TST」。

### 3.3 G3 — TST pendulum 判据进 skill（**FST/TST 各自一份**）

**用户决策（2026-06-01）：FST/TST 各自一份独立 pendulum 判据，哪怕内容相似。** 本 spec 只负责 **TST 那一份**（FST 那份归 data-analyst spec，见 §6）。

- **源**：`docs/review-packages/2026-0521-feedbacks/tstYoyo/tst-pendulum-algorithm.md`（同事 tstYoyo 写的真判据，§3/§4 含 periodicity 钟摆段>0.5/挣扎段<0.3、ANALYSIS_WINDOW 覆盖 3~5 钟摆周期、§7.3 品系/体重影响 PERIOD_MIN/MAX）
- **目标**：搬进 skill references，供 TST 解读时 subagent 能 read。建议路径：
  `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/tail_suspension-pendulum-params.md`
  （或在 tail_suspension.md 末尾追加 `## 🟡 pendulum 参数判据` 段 — 与同事确认哪种，见下方边界）
- **🔴 SSOT 边界**（[[feedback_ssot_lives_in_review_packages]]）：搬迁 = 把同事已写的内容复制进 skill 部署位，**不改其判据数字**。这是「部署」不是「创作」。若要改写格式以适配 skill，保持判据数值原样、标注来源 `docs/review-packages/.../tst-pendulum-algorithm.md`。
- **更新 tail_suspension.md 指引**：在 by-experiment/tail_suspension.md 加一句「pendulum 参数判据见 [配套文档]」，让 data-analyst 按 §2.6「按图索骥读当前范式文档」时能找到。

### 3.4 Q1 — 模板矛盾（标记给同事，工程不裁决）

`tail_suspension.md:13`（PorsoltCylinder-NoZones）vs `ev19_facts.py:62`（NoTemplate）vs 真实数据（tstHelperDemoVideo）三者不一致。
- **工程做**：§2 核验第 5 步跑 `inspect_uploaded_file` 看真实数据识别成什么，把**事实**记录下来
- **同事裁决**：到底 TST 标准模板是 NoTemplate 还是 PorsoltCylinder-NoZones。在 review-packages 开 issue / 标注，**不工程拍脑袋改领域文档的模板结论**
- **若核实发现 ev19_facts 映射错**（真实数据识别成 X 但映射写 NoTemplate）→ 那是工程 bug，可改 ev19_facts；**若只是文档措辞**→ 归同事

---

## 4. 验收

- [ ] **G1**：`"tail_suspension" in SUPPORTED_PARADIGMS_V01`；新测试绿
- [ ] **G2**：lead prompt Gate 1 列 6 个范式含 TST；反问文案含 TST；逐字 diff 确认没动其他 prompt 逻辑
- [ ] **G3**：TST pendulum 判据在 skill references 可被 subagent read（路径在 `/mnt/skills/...` 下）；tail_suspension.md 有指引
- [ ] **端到端 dogfood（最高价值，证明真能跑）**：用 `/home/wangqiuyang/DemoData/newdemodata/悬尾/` 真实数据跑一次完整会话 → Gate 1 识别为已支持 → 指标算出（不动时间/潜伏期/次数）→ 报告产出。**不应出现「v0.1 未实现」拦截**
- [ ] 全量 `make test`（agent backend）+ `pytest tests/`（ethoinsight）不退化（基线先取真值）
- [ ] CLAUDE.md（根 + backend）「v0.1 支持 6 个范式」更新（Q4）
- [ ] **Q1 矛盾已记录并标记同事**（不静默跳过）

---

## 5. 实施顺序（TDD）

| Task | 内容 | 估时 |
|---|---|---|
| T0 | 核验清单 §2（含真实数据跑库层 + inspect 看模板）| 0.5h |
| T1 | G1 SUPPORTED_PARADIGMS_V01 + 测试 | 0.25h |
| T2 | G2 lead prompt surgical 改 + diff 核验 | 0.5h |
| T3 | G3 TST pendulum 判据搬进 skill + tail_suspension.md 指引 | 0.5h |
| T4 | 端到端 dogfood（真实数据）| 1h |
| T5 | Q1 矛盾记录 + CLAUDE.md 更新 + 全量回归 | 0.5h |
| **合计** | | **~3.25h** |

---

## 6. 与其他工作的关系（实施 agent 必读）

- **🔴 与「data-analyst pendulum 卡死」spec（`2026-06-01-data-analyst-pendulum-audit-fix-design.md`）的依赖**：
  - 两份 spec **都涉及 pendulum 判据进 skill**。用户决策 FST/TST **各自一份**，所以**不冲突**：本 spec 管 TST 那份，data-analyst spec 管 FST 那份 + step 2.8 降级逻辑
  - **但有顺序关联**：TST 端到端 dogfood（本 spec T4）会触发 data-analyst 的 step 2.8 参数审计。**若 data-analyst 卡死 bug 未修，TST dogfood 会和 FST 一样卡死**（共用 pendulum 路径 + 若 TST 数据也 n 小）。
  - **建议**：data-analyst spec **先实施或同期**；本 spec T4 dogfood 时若卡死，确认是否 data-analyst bug 未修所致，不要误判为 TST 实现问题
- **真实 TST 数据可能命中 FST 同款 pendulum 异常**（不动时间异常低 0.5s）：§2 核验第 2 步先单独跑库层脚本验证。若异常，是 pendulum 参数对该批数据不适配（领域/参数问题，归 issue #63 + data-analyst spec 的降级），**不是 TST 实现 bug**
- **prompt.py 是受保护文件**：下次 deerflow sync 仍按 surgical-merge（[[feedback_sync_protected_files_registry_loss]]）

---

## 7. 不在范围

- ❌ 工程编写 TST 的 pendulum 判据**数值**（领域，归同事/review-packages；本 spec 只搬迁同事已写的）
- ❌ 工程裁决 Q1 模板矛盾的领域结论（标记同事）
- ❌ 填 NoTemplate.md 的领域字段（Q2，归同事）
- ❌ 造 TST golden-case 的专家标注（Q3，归同事）
- ❌ 修改 catalog/tst.yaml 的指标/参数（库层已就位，不动）
