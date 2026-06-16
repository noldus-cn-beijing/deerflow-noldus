# Pydantic schema 修复 + Excel 引擎切换（请核实代码改动是否正确）

> 纯代码问题，本文档自包含 —— 核实只需读本文，无需打开任何其他文件。

---

## 背景

一个多进程数据处理流水线：子进程各自把计算结果写成 JSON 文件，父进程把很多份聚合成一个大 dict，用 Pydantic 模型校验后落盘。其中一个嵌套字段是「每条结果实际用到的参数」的映射表。

```python
class ResultStat(BaseModel):
    model_config = ConfigDict(extra="allow")
    params_used: dict[str, float | int | str | None] = Field(default_factory=dict)
```

producer→consumer 链是**逐字透传**这个 dict（没有 consumer 对这些值做算术）。

---

## 一、Bug：schema 太窄，拒绝合法的 list 值

producer 最近开始产出 **list 型的值**，例如 `{"selected_cols": ["a"]}`。Pydantic 对每个 list 值依次匹配 union 的 3 个标量分支（float/int/str）全部失败，N 个 list 值 → N×3 个 ValidationError，整个聚合校验失败、被判 FAILED——**尽管数据是对的**。

实证 stacktrace 形态（N=28 个 list 参数 → 84 errors）：
```
params_used.selected_cols.float
  Input should be a valid number [input_value=['a'], input_type=list]
params_used.selected_cols.str
  Input should be a valid string [input_value=['a'], input_type=list]
...（每个 list 值 ×3 分支）
```

**这是「过窄 schema 拒绝如实数据」**：schema 当初为标量参数（`threshold=30.0`）设计，后来上游开始合法地产出 list 值（一组选中的列名），schema 没同步更新。数据正确，schema 装不下。

**一个干扰项**：上层框架在校验失败时打了句 WARNING「producer finished but forgot to call the seal tool」，这句是**误导**——真因是它上方的 ValidationError，不是 producer 漏调工具。

**影响面**：≥2 个不同上游调用路径都会产出 list 型参数，所以不是单点问题。

---

## 二、慢：Excel 默认引擎慢

同一流水线里大量读 `.xlsx`。pandas 默认用 `openpyxl`（纯 Python 解 XML）。实测（real file 9301 行 × 12 列，742KB，best of 3）：

| 操作 | openpyxl | calamine | 加速 |
|------|---------|----------|------|
| `pd.read_excel` 全量 | 0.651s | 0.081s | **8.1×** |
| 一个聚合工具（读 header + 全量）| 2.03s ×28 文件 ≈57s | 预估 ≈8s | — |
| 一个计算脚本读盘 | 1.35s ×140 次 | ≈0.1s | 省 ~170s |

calamine（Rust，pandas 3.0 官方支持的 `engine="calamine"`）输出与 openpyxl **逐格 `astype(str).equals` 一致**（已在一个真实文件验证）。

---

## 三、修复方案

### A. schema bug

**A1 — 抽共享类型别名，两个字段共用**（治本：消除「同一值域两份独立声明各自漂移」）：

```python
# 模块顶部
ParamValue = float | int | str | list[str] | None

# 字段 1（出问题的）
params_used: dict[str, ParamValue]          # 原 dict[str, float|int|str|None]

# 字段 2（下游一个 finding 模型，producer 被指示把上面的值逐字拷进来）
used_value: ParamValue                      # 原 float | int | str
```

- 字段 2 也放宽的理由：另一个 producer（非确定性，是个 LLM）被指示把 `params_used[k]` 的真实值逐字填进 `used_value`。字段 1 放宽后 list 值即合法，所以字段 2 是**确定触发的潜伏 bug**——下一跳就会同样炸。不是投机。
- Pydantic v2 smart-union 中 `str` 与 `list[str]` **互不坍缩**（`"a"` 精确匹配 str 立即返回；`["a"]` 精确匹配 list[str]；v2 绝不 list↔str 互转）。
- **不设** `union_mode="left_to_right"`（float 在前 + lax 会把 `"3.5"` 吞成 3.5）。
- **不加** `bool`、**不放宽到** `list[int]`（`[1,2]` 仍想要响亮失败）。
- 字段 2 现有一个 `model_validator(mode="before")` 把 `used_value=None → ""`。`ParamValue` 含 `None`，与此归一有哲学冲突。**保守保留该映射 + 加注释说明它现在是 documented 例外**（删它=改一条未坏路径的行为）。

**A2 — 测试**（TDD 先红后绿，普通断言）：单元（`ResultStat(params_used={"selected_cols":["a"]})` 改前红改后绿）+ 端到端（仿真实 seal 路径写含 list 值的 JSON 文件，复现 N×3 error，转绿）。

### B. Excel 引擎

- 加**硬依赖** `python-calamine`（非 optional——装不上=build 时响亮失败）+ pin 版本 + lockfile 固化。
- 单一常量 `EXCEL_ENGINE = "calamine"`，所有 `pd.read_excel` 显式传 `engine=EXCEL_ENGINE`（不靠 pandas 默认，上游默认选择逻辑可能变）。
- **无运行时 fallback**（关键，已采纳上轮论证：`try calamine except openpyxl` 对「calamine 成功但值不同」永不触发，只在异常时触发——纯负收益，还掩盖 engine bug、逃逸的是 openpyxl 异常误导 debug、坏文件解析两遍）。失败带 path+engine 响亮 raise。
- openpyxl 留作 **test/dev 依赖**（calamine 只读，测试用 openpyxl 写 fixture；另有一处无关代码也 import 它）——**不删**。
- 测试：语料级 `pd.testing.assert_frame_equal(check_dtype=True)`（不止单文件 `astype(str)`）+ 合成边界探针 fixture（日期格式 / int-vs-float dtype / 空 cell-vs-空串-vs-NaN / error cell / bool / 前导零文本数字 / 尾部全空行）。calamine 不可用时 skip。

### 不改动（已核实下游容忍 list）
- 聚合处逐字透传 dict，无标量假设。
- 另一个校验工具从不读这个字段。
- 其他几个聚合模型不内嵌 `ResultStat`。

---

## 四、想请核实的点（纯代码）

1. **A1 的 `ParamValue` 共享别名**是否正确收口了 bug 类？别名 vs 改两处裸 union，前者更对吗？
2. **字段 2（`used_value`）放宽**是否过度（它还没在生产炸过）？还是确属同一 bug 的下一跳、该一起修？「修整个 bug 类」与「别动没坏的」的界线在哪？
3. **Pydantic v2 union** 里加 `list[str]` 有没有我没注意到的坑（smart-union 排序、lax 强转、`extra="allow"`+`default_factory=dict` 的 fail-open 面）？
4. **「无 fallback」**是否正确落实了上轮论证？引擎选择还有没有更好的可观测性手段？
5. **calamine 等价性测试**用语料级 `assert_frame_equal` + 合成探针够不够？哪个分歧轴最该测？
6. 有没有**我判断错或漏掉**的地方？
