# Spec：ETHO-5 分组检测单文件外推 —— prompt 强制全量 per_file_grouping，禁止以偏概全

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23
> 性质：🟡 中 · 纯 prompt 指引修复。lead 只 inspect 单文件就断言「Group 均为 XX」，后才逐个扫出多组。工具侧 `identify_ev19_template` 本就一次扫全部文件返回 `per_file_grouping`（全量正确），缺口在 lead prompt 只「优先用」是软建议。
> **刻意轻量**：本条根因是**纯 prompt 指引缺口**——结构层（工具）已正确，不该上结构门（HarnessX §6.6 ALFWorld：prompt 编辑在某些场景就够；over-engineer 反而犯 under-exploration 的反面）。修法只改 prompt 措辞 + 加病理自检，不动工具、不加 middleware。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-5；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - 工具侧已正确：`identify_ev19_template_tool.py:518` `for f in uploaded_files` 全量扫描返回 `per_file_grouping`。
> - **框架**：DeerFlow（LangGraph）原生——只改 lead prompt（受保护文件），零机制变更。正面提示（memory `feedback_skill_describing_tool_output_enables_hallucination`：用正面指令不用"禁止 X"反向激活）。
> - 受保护文件：`agents/lead_agent/prompt.py` sync surgical。

---

## 〇、给实施 agent 的一句话

`prompt.py:493-494` 现状：
```
identify_ev19_template 现在返回 per_file_grouping ...，lead 构造分组时**优先用它**...，
不需要逐个调 inspect_uploaded_file 试探边界。
仅在以下情况调 inspect_uploaded_file：per_file_grouping 为空 ... 或不足以推断分组时 ...
```
「优先用」是软建议，给了 lead「看一个文件就外推」的偷懒入口。对比：范式推断有「禁止猜」硬约束（`prompt.py:476` 一带 HITL 铁律），**分组判定没有对称的「禁止单文件外推」约束**。

**修法**：把「优先用」升级为「分组判定**必须**基于全量 per_file_grouping」+ 加一条对称硬约束「分组结论必须覆盖全部文件，禁止用单个/少数文件 inspect 结果对全部文件下分组结论」。正面措辞为主。**只改 prompt，不碰工具。**

---

## 一、根因（逐字节）

### 1.1 现象

Run2 lead 只 inspect 单文件就断言「Group 均为 XX」，后才逐个扫出 4 组。分组结论以偏概全。

### 1.2 真因：prompt 软建议 + 缺对称硬约束

- 工具侧正确：`identify_ev19_template_tool.py:518` `for f in uploaded_files` 一次扫全部、返回 `per_file_grouping`（每文件分组字段）。批量能力早有。
- prompt 侧软（`prompt.py:493`）：「**优先用**」per_file_grouping，给了偷懒入口（看一个 inspect 就外推）。
- **不对称**：范式推断有硬约束（`prompt.py:476` 范式失败必须 ask、禁猜），分组判定没有「禁止单文件外推」对应约束。lead 在压力下走最省 token 的路（inspect 一个就下结论）。

### 1.3 为什么是纯 prompt（不上结构门）

工具已经全量正确——问题纯粹是 lead **不严格用它**。这不是"结构缺失"（结构在），是"指引不够硬"。加 middleware 强制检查"lead 是否用了全量分组"会 over-engineer（要解析 lead 的分组结论比对全量，复杂且脆）。**prompt 硬约束 + 现有工具足矣**（HarnessX：不是所有问题都上结构门）。

---

## 二、设计

### 2.1 修法：prompt.py 分组段升级（正面硬约束）

改 `prompt.py:493-494` 段：

```
**分组判定必须基于全部上传文件的 per_file_grouping：**
identify_ev19_template 一次返回所有文件的 per_file_grouping（每个文件名 → 分组字段）。
构造 control/treatment 映射时，读完整 per_file_grouping、对全部文件下分组结论——
分组判定覆盖全部文件，每个文件都依据它自己的 per_file_grouping 条目归组。
单个或少数文件的 inspect_uploaded_file 结果只反映那几个文件，不能外推为全部文件的分组结论。

仅在 per_file_grouping 为空（EV19 头无分组字段）或值不直观（如 "aa"/"bb"）时，
才 fallback 到 inspect_uploaded_file 看数据预览——且此时也要看够能覆盖全部分组的文件，
不是看一个就定全局。
```

放在现有分组段（`prompt.py:485-494` 一带），与范式「禁止猜」硬约束（`prompt.py:476`）形成对称。

### 2.2 不改的东西

- **不改** `identify_ev19_template_tool.py`（已全量正确）。
- **不加** middleware/guardrail 检查 lead 分组结论（over-engineer）。
- **不加** 「禁止 X」反向激活措辞（用正面"必须覆盖全部文件"）。

---

## 三、改动清单（change manifest）

### 3.1 `agents/lead_agent/prompt.py` —— 分组段硬约束
- **编辑**：`prompt.py:493-494` 「优先用」→「必须基于全量 per_file_grouping + 禁止单文件外推」（正面措辞）。
- **预期改善**：lead 不再 inspect 单文件就外推全局分组（Run2 现象消除）。
- **可能回归**：措辞过硬导致 lead 在 per_file_grouping 真为空时也不敢 fallback inspect → 保留「仅在为空/不直观时 fallback」分支。
- **病理**：under-exploration 的反面——这条**故意不上结构门**，因为结构（工具）已对，只缺 prompt 硬度。

---

## 四、测试（prompt 契约，红→绿）

> ⚠️ prompt 契约测试用 importlib 加载被测源（防主仓旧 prompt 假绿，memory 多次踩坑）。

测试文件：`tests/test_lead_prompt_grouping_full_scan.py`（新增）。

1. **test_prompt_requires_full_per_file_grouping**（红→绿）：渲染 lead prompt，断言含「分组判定必须基于全部上传文件 / per_file_grouping」+「不能外推」语义串。改前红（只有「优先用」软串）、改后绿。
2. **test_prompt_keeps_fallback_branch**（守边界）：断言仍保留「per_file_grouping 为空/不直观时 fallback inspect」分支（不把 fallback 删死）。
3. **test_prompt_no_negation_framing**（正面提示原则）：断言修改段不含「禁止猜」式反向激活措辞（用正面"必须覆盖全部"）。

---

## 五、验收标准

1. manifest 完整（§三）。
2. 红→绿：测试 1 改前红改后绿；2/3 绿。
3. import 环：虽只改 prompt，仍裸导入 `make_lead_agent` 0 退出（prompt 渲染不报错）。
4. 回归：lead prompt 相关测试邻域绿。
5. scope：未改 identify 工具；未加 middleware；未删 fallback 分支。

---

## 六、风险与注意事项（三大病理自检）

1. **reward hacking**：不适用（prompt 指引，无验收可被 hack）。
2. **catastrophic forgetting**：改分组段会不会削弱邻近的范式/列对齐约束？→ 只改分组段，diff 时确认不动 `prompt.py:476` 范式铁律 / `prompt.py:486` 列对齐段（PR#175 类 drift 教训：改一段别碰邻段）。
3. **under-exploration（反向自检）**：这条**正确地选择了 prompt 而非结构**——因为工具已全量正确，结构没缺。不是逃避结构改动（那是 under-exploration），是"结构已在、只缺指引硬度"。若将来发现 prompt 硬约束仍压不住（lead 持续外推），才升级到结构门（解析分组结论比对全量）——但 prod 现状是 prompt 软导致，先修 prompt。
4. **受保护文件**：prompt.py sync surgical 守护这段分组硬约束。
5. **prompt 契约测试读被测源**：否则假绿。

---

## milestone 建议
- 「EPM dogfood 图表/交互流水线打磨」track：ETHO-5（分组全量）与 ETHO-2（批量扫描路径指引）同属「lead 不严格用现成全量工具 → prompt 讲硬」的指引缺口系列。
