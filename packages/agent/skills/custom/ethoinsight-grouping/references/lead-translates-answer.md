# Lead 落地分组答案

**适用**: lead 收到用户上传数据后,准备调 `prep_metric_plan` 时。

## 黄金工作流(2026-05-28 更新)

```
1. 上传完成 → identify_ev19_template / set_experiment_paradigm
2. 对每个 uploaded_file 调 inspect_uploaded_file(file=...)
   → 拿到 ev19_metadata.grouping_fields (含 Treatment / Dose / Animal ID 等)
3. 根据 inspect 返回判断分组场景:
   场景 A: 每个文件/sheet 都有 Treatment 字段 → 直接构造 groups dict,无需反问
   场景 B: 没有 Treatment 字段但有其他元数据 → 给用户列出 inspect 结果反问对应关系
   场景 C: 文件结构异常 → 反问用户人工分组
4. 调 prep_metric_plan(uploaded_files, paradigm, groups=...)
```

## 强制规则

- **必须**在调 `prep_metric_plan` **之前** 调 `inspect_uploaded_file` 探查每个上传文件
- **如果** inspect 返回 ev19_metadata.grouping_fields 含 `Treatment` 字段:**优先使用** Treatment 值构造 groups dict,**不要**反问用户
- **绝不允许**用自然语言"分组信息在 drug 列"塞进派遣 prompt 而不传 groups dict
- **不要**幻觉派遣 general-purpose / knowledge-assistant 去探查文件结构(它们做不了),只用 inspect_uploaded_file

## 三种典型场景

### 场景 A:EV19 文件头自带 Treatment 字段(最常见,无需反问)

**典型样本**:每个 EV19 txt 或 xlsx sheet 头部含

```
"Treatment";"Drug"
"Dose";"5 mg/L"
"Animal ID";"1"
```

**inspect_uploaded_file 返回示例**(xlsx 多 sheet):

```json
{
  "status": "ok",
  "file": "/mnt/.../原始数据.xlsx",
  "format": "xlsx",
  "sheets": [
    {
      "name": "轨迹-Arena 1-Subject 1",
      "virtual_path": "/mnt/.../原始数据.xlsx::轨迹-Arena 1-Subject 1",
      "ev19_metadata": {
        "arena": "Arena 1",
        "subject": "Subject 1",
        "grouping_fields": {"Treatment": "Drug", "Dose": "5 mg/L", "Animal ID": "1"}
      }
    },
    {
      "name": "轨迹-Arena 2-Subject 1",
      "virtual_path": "/mnt/.../原始数据.xlsx::轨迹-Arena 2-Subject 1",
      "ev19_metadata": {
        "arena": "Arena 2", "subject": "Subject 1",
        "grouping_fields": {"Treatment": "Saline", "Dose": "0 mg/L", "Animal ID": "2"}
      }
    }
  ]
}
```

**lead 应该直接构造 groups dict**(用 Treatment 值作 group_name):

```python
prep_metric_plan(
  uploaded_files=[
    "/mnt/.../原始数据.xlsx::轨迹-Arena 1-Subject 1",
    "/mnt/.../原始数据.xlsx::轨迹-Arena 2-Subject 1",
  ],
  paradigm="forced_swim",
  groups={
    "/mnt/.../原始数据.xlsx::轨迹-Arena 1-Subject 1": "Drug",
    "/mnt/.../原始数据.xlsx::轨迹-Arena 2-Subject 1": "Saline",
  },
)
```

**也可以**:在派遣前给用户**简短确认**(可选,推荐):

```
ask_clarification(
  question="数据自带 Treatment 标签:\n
   - Arena 1 Subject 1: Treatment=Drug, Dose=5 mg/L\n
   - Arena 2 Subject 1: Treatment=Saline, Dose=0 mg/L\n
  按此分组(Drug=实验组, Saline=对照组)继续吗?",
  options=["按 Treatment 字段分组", "我要手动指定分组"]
)
```

### 场景 B:文件没有 Treatment 字段(降级反问)

**inspect 返回** `grouping_fields={}` 或缺关键字段时,**列出文件清单 + 列名给用户反问**:

```python
ask_clarification(
  question="未在文件头部检测到 Treatment 字段。请告诉我分组:\n
   - File 1: /mnt/.../arena1.txt (列: x_center, y_center, mobility_state, ...)\n
   - File 2: /mnt/.../arena2.txt (...)\n
  请直接说: '文件 1 是 control,文件 2 是 treatment' 之类。",
)
```

用户答完后 lead **必须**翻译成 dict 调 prep_metric_plan(groups=...)。

### 场景 C:用户说"第一/第二组"模糊用语

**用户说**:

> "第一个是实验组,第二个是对照组"

**lead 行动**:

1. 先调 inspect_uploaded_file 看每个文件的 ev19_metadata
2. 如果 inspect 出 Treatment 字段:**优先用 Treatment**(数据可信度高于用户口语)
3. 如果 inspect 无 Treatment:按 uploaded_files **列表顺序**映射(第一个=列表 [0])
4. **务必**反问确认对应关系再调 prep_metric_plan:

```python
ask_clarification(
  question="确认分组对应关系:\n
   - 文件 1 (Arena 1, Subject 1): 实验组 (treatment)\n
   - 文件 2 (Arena 2, Subject 1): 对照组 (control)\n
  对应错了请告诉我哪两个互换。",
  options=["对应正确", "调换 1 和 2"]
)
```

## groups dict 的精确格式

| 字段 | 要求 | 例子 |
|---|---|---|
| key | uploaded_files 列表中的某个完整路径(含 `::sheet` 后缀如有) | `"/mnt/.../foo.xlsx::轨迹-Arena 1-Subject 1"` |
| value | group_name 任意字符串,建议用业内通用名 | `"Drug"` / `"Saline"` / `"treatment"` / `"control"` / `"vehicle"` |

**prep_metric_plan 实测校验**:

- key 不在 uploaded_files 中 → 返回 `schema_violation` 错误,列出可用键
- groups 为空 dict `{}` 与 `None` 等价(都按无分组处理)
- value 不限于英文,中文 group 名也接受("药物组" / "对照组")

## 常见 pitfall

| pitfall | 后果 | 应对 |
|---|---|---|
| 不调 inspect_uploaded_file 直接反问"drug 列是什么" | 用户也不知道 drug 列具体值,死循环 | 必须先 inspect,EV19 头部已经有 Treatment 字段 |
| 派遣 general-purpose subagent 去 Python pandas 读 xlsx | subagent 类型不存在,task() 报错 | 改用 inspect_uploaded_file 工具 |
| inspect 出 Treatment 字段后仍然反问用户 drug 值 | 浪费 turn,用户可能误填 | 直接信 EV19 头部 |
| 把 Treatment 当列数据看(`drug` 列每行一个值)而忽略 header | trajectory 行里每行都是同一个 Treatment 值,与头部冗余 | 信头部不信行 |
| 把分组信息塞进 set_experiment_paradigm 的 subject 字段 | 该字段是范式描述(rodent/zebrafish),不是 group | 用 prep_metric_plan(groups=...) |

## 与 EV19 元数据的对应

EV19 头部出现以下任一字段时,inspect 都会提取并放在 `grouping_fields`:

- `Treatment` / `treatment` - Noldus 默认分组字段(最常见)
- `Group` / `group` / `组别`
- `Drug` / `drug` - 注意区别 `Drug` 头部字段 vs 数据中 `Drug` 列(可能并存)
- `Condition` / `condition`
- `Dose` / `dose` / `剂量` - 配合 Treatment 一起描述实验组
- `Compound` / `compound`
- `Animal ID` / `动物编号` - 不是分组字段但 subject identifier,inspect 也会返回

lead 自行判断哪个字段最能体现组间对比(优先 Treatment > Group > Condition > Drug)。
