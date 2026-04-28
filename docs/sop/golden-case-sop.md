# Golden-Case 标注 — 行为学同事使用 SOP

> 2026-04-22 启动。目标：每个要做的范式积累 2-5 个 golden case，作为 agent 能力的"黄金标准"。
> 当前状态：case-001 (shoaling) 工程侧初稿完成，待专家补齐判断字段。

## 一句话说明

**你标注一份数据的"期望分析结论"，agent 未来必须达到这个水平**。你的标注同时服务三件事：**告诉 agent 怎么做**（领域知识）、**验证 agent 没退步**（回归测试）、**训练 agent 变强**（微调种子数据）。

## 一个 case 包含什么

```
golden-cases/case-XXX-<范式>-<标签>/
├── raw-data/                  # EthoVision 原始轨迹文件（工程准备）
├── metadata.yaml              # case 身份信息（工程填大部分）
├── expected-analysis.yaml     # ★ 你的期望结论（机器用来断言）
└── notes.md                   # ★ 你的推理过程（未来作为 agent 学习材料）
```

**你只需要填两个文件**：`expected-analysis.yaml`（结构化判断）和 `notes.md`（自由表述的推理链）。结构字段定义见 [SCHEMA.md](../../golden-cases/SCHEMA.md)。

## 你参与的流程（一个 case 15-60 分钟）

1. 工程搭骨架：选一份 demo 数据，填好 `metadata.yaml` 和 `expected-analysis.yaml` 的数值字段，在需要专家判断的地方标 `TODO(行为学同事)`
2. 工程在 Slack/微信告诉你"case-XXX 准备好了"并**附上具体问题清单**
3. 你打开 case 目录，**回答每个 TODO**（通常是"这种情况你怎么判读？"）
4. 你写 `notes.md`，按 6 段大纲记录你的推理过程（重点是推理链，不是结论）
5. 工程跑 `python3 scripts/validate_golden_case.py golden-cases/case-XXX-*/` 确认结构合法
6. 合入仓库，这个 case 就永久成为 agent 的"测试题"

## 你做好标注的三个原则

1. **推理过程比结论重要** — 写 notes.md 时，把你"为什么这么判断"的每一步都写下来。这些推理链会作为 agent 的学习材料。
2. **组间比较，不用绝对阈值** — 这是我们和你确认过的判读哲学。不写"开放臂时间 <10% 就是焦虑"，而写"处理组 vs 对照组显著降低，且不伴随运动能力差异，因此判读为焦虑样行为"。
3. **写出 agent 不该说什么** — `should_not_contain` 和 `forbidden_claims` 字段用来捕获 agent 的幻觉和过度推断。比如"样本量只有 5 不可能有 p<0.001"，"没有药物处理不能说'药物效应显著'"。

## 质量参考（让 case 真正有用）

- **`expected_findings` 至少 1 条 `statistical_conclusion`**（说清你的最终结论）
- **每条 finding 填 `reasoning` 字段**，哪怕只有一句话——这是 agent 未来要学会的"思考方式"
- **`required_keywords` 不要留空**，否则这条断言没有意义
- **notes.md 300-800 字**，太短推理链丢失，太长增加负担

## 什么时候该建新 case

- **新范式首次落地**：每个新范式至少建 2 个 case（一个基线，一个有异常）
- **Agent 翻车了**：专家看到 agent 输出明显错误时，就把那份数据+正确分析做成 case
- **边缘情况有代表性**：n<5 样本量、离群个体、混杂因素、品系差异等，都值得单独一个 case

## 技术说明（给工程）

| 组件 | 位置 |
|------|------|
| Schema 定义 | `golden-cases/SCHEMA.md` |
| 空白模板 | `golden-cases/TEMPLATE/` |
| 校验脚本 | `python3 scripts/validate_golden_case.py` |
| 案例存放 | `golden-cases/case-XXX-<paradigm>-<tag>/` |

### Golden-case 与其他系统的关系

- **vs 训练数据飞轮**（`training-data-flywheel-sop.md`）：飞轮采集**真实使用**中的 agent 输入输出 + 三按钮反馈（量大、质量不均）；golden-case 是**离线人工标注**的黄金标准（量少、质量高）。两者配合：飞轮训练规模，golden-case 校准质量。
- **vs 范式分析工具**（`docs/specs/paradigm-analysis-tools-spec.md`）：工具规格定义 agent **能做什么**（技术能力边界）；golden-case 定义 agent **应该做成什么样**（能力达成标准）。
- **vs `ethoinsight/assess.py` 里的阈值**：阈值只作为"批次质检参考"，**不作判读依据**。实际判读按组间比较，规则写在每个 case 的 `expected_findings` 里。

### 自动化回归测试（待实现）

当前 `validate_golden_case.py` 只做 schema 校验。未来需要 `run_golden_case_regression.py`：让 agent 对 raw-data 实际跑一次分析，比对 agent 输出与 `expected-analysis.yaml`（数值落在 `expected_range`、输出命中 `required_keywords`、不含 `forbidden_claims` 和 `should_not_contain`）。这个脚本等 case-001 完整填好后再实现，避免用假数据写歪测试框架。
