# 设计：从「学术范式」迁移到「EV19 模板」体系（Gate 1 重定位）

**日期**: 2026-04-29
**状态**: 设计阶段 + 已生成 review 包，等行为学同事补充
**触发**: E2E 测试发现 Gate 1 主动反问没生效；进一步追问发现 prompt 里"7 大类 18 范式"分类表与 EthoVision XT 19 真实模板体系（62 个）不对应；行为学同事会维护「模板 ↔ 实验」N:M 对应关系

**关联**：
- Review 包：[docs/review-packages/2026-04-29-ev19-templates/](../review-packages/2026-04-29-ev19-templates/)
- 起草脚本：[scripts/build_ev19_template_index.py](../../scripts/build_ev19_template_index.py)、[scripts/build_ev19_experiment_drafts.py](../../scripts/build_ev19_experiment_drafts.py)

---

## 1. 问题陈述

### 1.1 用户最初的诉求

> 「在用户上传数据之前，agent 并没有反问用户'你的实验模板'之类的问题。」

### 1.2 调查路径

第一次调查方向：以为是 Gate 1 中间件没生效，结果发现：

- 中间件正常装上、正常拦截
- 中间件只在 agent 调 `task()` 时检查，对 lead 自己的首轮回应不拦截
- 而且 prompt 第 310 行明确允许「用户提到范式名 → 跳过两级 UI 直接 set_experiment_paradigm」，第 1098 行明确教 agent「从文件名推断范式（Shoaling = shoaling）」

第二次调查发现真正分歧：用户期望 agent 问的是 **EthoVision 软件里的模板**（62 个），但当前 prompt 用的是**学术范式分类**（7 大类 18 范式）。这两个层面根本不对应。

### 1.3 两套体系对比

| 维度 | 当前 prompt 用的「学术范式」 | 用户期望的「EV19 模板」 |
|---|---|---|
| 数量 | 18（扁平） | **20 个大类，每个大类下 1-15 个变体，总共 62 个** |
| 抽象层级 | 行为学概念（焦虑迷宫 / 抑郁绝望） | EV 软件里的具体配置（arena 形状 + zone 定义 + 动物种类） |
| 命名方式 | 学术名词 | EV 软件参数组合（如 `OpenFieldRectangle-AllZones`） |
| 用途 | 报告写作的语言 | 决定数据结构（导出列、zone 列、subject 类型）+ 部分分析步骤 |
| 数据来源 | 我们自己定义 | EV19 软件 demodata 目录 `demodata/ev19 templates/`（每个目录有 `templateMetaData.xml`） |
| 现状字段 | `paradigm`, `paradigm_cn`, `category`, `subject` | 无 |

### 1.4 EV19 模板的真实层级（20 大类 → 62 变体）

**结构**：每个大类是一种 arena 类型，下面的变体围绕**三个独立轴**变化：

- **Zone 配置**：`AllZones` / `NoZones` / `NovObjZones` / `FewZones` / `AFewZones` / `FeedingShelter` / `Subdivided2x2/3x3/4x4`
- **动物种类**（仅部分大类有这个轴）：`Mice` / `RatOther` / `Fish` / `Insects` / `Rodents-Other`
- **阵列规模**（仅 PhenoTyper 和 UgoBasileFCS 有）：单个 / `Quad` / `16x` / `1cubicle` / `4cubicles`

**完整清单**：

| 大类 | 变体数 | 变体 |
|---|---|---|
| OpenFieldCircle | 5 | AllZones / NovObjZones / NoZones-Fish / NoZones-Insects / NoZones-Rodents-Other |
| OpenFieldRectangle | 7 | AllZones / NovObjZones / NoZones / NoZonesFishInsects / Subdivided2x2/3x3/4x4 |
| PhenoTyper | 15 | (单/Quad/16x) × (AllZones-Mice / AllZones-RatOther / FeedingShelter-Mice / FeedingShelter-RatOther / NoZones) |
| MWM | 3 | AllZones / AFewZones / NoZones |
| PlusMaze | 3 | AllZones / FewZones / NoZones |
| T-Maze | 4 | (Fish/Rodents-Other) × (AllZones/NoZones) |
| WellPlate | 4 | (Circle/Rectangle) × (AllZones/NoZones) |
| BarnesMaze | 2 | 20Holes / NoZones |
| Cross Maze-Fish | 2 | AllZones / NoZones |
| PorsoltCylinder | 2 | AllZones / NoZones |
| Radial-8-arm | 2 | AllZones / NoZones |
| Sociability | 2 | AllZones / NoZones |
| UgoBasileFCS | 2 | 1cubicle / 4cubicles |
| Y-Maze | 2 | AllZones / NoZones |
| ZeroMaze | 2 | AllZones / NoZones |
| AquariumTrack3D | 1 | (单变体) |
| FlightChamberTrack3D | 1 | (单变体) |
| DanioVision DVOC 004x | 1 | 96w-circ |
| UgoBasileActiveAvoidance | 1 | (单变体) |
| NoTemplate | 1 | (无模板，自由轨迹) |

合计：**20 大类，62 变体**。

### 1.5 实际 EV19 模板示例（节选）

```
OpenFieldCircle-AllZones / -NoZones / -NovObjZones / -NoZones-Fish / -NoZones-Insects / -NoZones-Rodents-Other
OpenFieldRectangle-AllZones / -NoZones / -NovObjZones / -Subdivided2x2/3x3/4x4 / -NoZonesFishInsects
PhenoTyper-16x-AllZonesMice / -AllZonesRatOther / -FeedingShelterMice / -FeedingShelterRatOther / -NoZones / ...
PlusMaze-AllZones / -FewZones / -NoZones
ZeroMaze-AllZones / -NoZones
MWM-AllZones / -AFewZones / -NoZones
BarnesMaze-20Holes / -NoZones
WellPlate-Circle-AllZones / -NoZones; WellPlate-Rectangle-AllZones / -NoZones
T-Maze-Fish-AllZones / -NoZones; T-Maze-Rodents-Other-AllZones / -NoZones
Y-Maze-AllZones / -NoZones
Cross Maze-Fish-AllZones / -NoZones
PorsoltCylinder-AllZones / -NoZones
Sociability-AllZones / -NoZones
Radial-8-arm-AllZones / -NoZones
AquariumTrack3D / FlightChamberTrack3D / DanioVision DVOC 004x-96w-circ
UgoBasileActiveAvoidance / UgoBasileFCS-1cubicle / UgoBasileFCS-4cubicles
NoTemplate
```

每个模板文件夹里 `templateMetaData.xml` 有：

```xml
<m_strArenaTemplate>Open field, square</m_strArenaTemplate>      <!-- 竞技场类型 -->
<m_strZoneTemplate>Center, border, corners</m_strZoneTemplate>    <!-- zone 定义 -->
<m_vecRodentSubTypes>...</m_vecRodentSubTypes>                    <!-- 适用动物：rodent -->
<m_vecFishSubTypes>...</m_vecFishSubTypes>                        <!-- 适用动物：fish -->
<m_vecInsectSubTypes>...</m_vecInsectSubTypes>                    <!-- 适用动物：insect -->
<m_bOtherTypes>1</m_bOtherTypes>                                  <!-- 其他动物 -->
```

注：`Shoaling` 不是 EV19 一个独立模板。鱼群行为研究在 EV19 里用 `AquariumTrack3D` 或 `OpenFieldCircle-NoZones-Fish` 等模板做。我们 `packages/ethoinsight/ethoinsight/templates/shoaling.py` 里的 shoaling **是 ethoinsight 自己的分析模板**（专为鱼群指标如 IID/NND/polarity 设计的脚本），与 EV 模板是不同概念。

### 1.6 用户在 EthoVision 里的真实选择路径

EV19 用户实际的工作流：先选 **arena 大类**（旷场？plus 迷宫？...），然后在该 arena 下面选**具体配置**（要不要分 zone？什么动物？是不是阵列？）。我们 Gate 1 应该镜像这个工作流——而**不是**把 62 个变体平铺给用户挑。

---

## 2. 关键概念澄清

```
┌─────────────────┐
│ EV19 模板（62） │  ← 用户在 EthoVision 里实际选的；决定数据结构
└────────┬────────┘
         │ 1:N 映射
         ↓
┌─────────────────┐
│ 学术范式（18）   │  ← 行为学话语；决定报告解读语言
└────────┬────────┘
         │ 1:N 映射
         ↓
┌──────────────────────┐
│ ethoinsight 分析模板 │  ← packages/ethoinsight/.../templates/*.py；决定分析脚本
└──────────────────────┘
```

举例：
- 用户在 EV19 选 `OpenFieldRectangle-AllZones` → 数据有 Center / Border / Corners 4 个 zone 列 → 学术范式是 `open_field` → 分析模板用 `open_field.py`（待写）
- 用户在 EV19 选 `OpenFieldCircle-NoZones-Fish` → 数据无 zone 列、动物是 fish → 学术范式可能是 `shoaling`（多鱼）或 `aquatic_open_field`（单鱼）→ 这一步**需要 agent 进一步问"是几条鱼还是单鱼"才能定**

注意：**EV19 模板 → 学术范式不是一一对应**。同一个 EV 模板（如 `OpenFieldCircle-NoZones-Fish`）在不同实验目的下对应不同范式。

---

## 3. 设计选项

> ✅ 用户已选**选项 B（保留学术范式 + 增加 EV19 模板字段）**，且**领域知识做成独立 skill**而非塞 prompt。其他选项保留供回顾。

### 选项 A：完全切换到 EV19 模板体系

**做法**：experiment-context.json 字段全部改成：
```json
{
  "ev19_template": "OpenFieldRectangle-AllZones",
  "arena_template": "Open field, square",
  "zone_template": "Center, border, corners",
  "subject_type": "rodent"
}
```

prompt 里的"7 大类 18 范式"表整张换成"62 个 EV19 模板分组列表"。

**优点**：
- 与 EthoVision 用户语言完全对齐，体验最直观
- 数据结构识别可以基于模板自动完成（zone 列、subject 列已知）

**缺点**：
- 学术范式那层完全丢失，**报告写作时 agent 不知道怎么解读**（"在 OpenFieldRectangle-AllZones 中焦虑水平显著降低"——不通顺，因为术语层级错了）
- 62 个模板 UI 太长，必须分组（按 arena 大类）
- ethoinsight/ 库里 `shoaling.py` 等模板是按学术范式组织的，需要重写映射

### 选项 B：保留学术范式 + 增加 EV19 模板字段（双层）

**做法**：experiment-context.json 字段扩展为：
```json
{
  "ev19_template": "OpenFieldCircle-NoZones-Fish",
  "arena_template": "Open field, circle",
  "zone_template": null,
  "subject_type": "fish",

  "paradigm": "shoaling",
  "paradigm_cn": "斑马鱼鱼群行为",
  "category": "zebrafish"
}
```

Gate 1 流程改为：
1. 第一步：问 EV19 模板（按 arena 大类分组的 22 组）
2. 第二步：根据模板自动推断学术范式；当 1:N 对应时再问一次细化
3. 第三步：set_experiment_paradigm（带两套字段）

**优点**：
- 用户语言（EV 模板）与系统内部语言（学术范式 → 决定分析模板和报告解读）双向打通
- 现有 ethoinsight 库（按学术范式组织）不需要重写
- 推断关系明确（EV 模板 + 用户回答细化 → 唯一学术范式）

**缺点**：
- 字段多，state 管理复杂度上升
- 需要维护一份 **EV19 模板 → 学术范式映射表**
- UI 流程从两步可能变成三步

### 选项 C：保留学术范式，prompt 里增加"如何识别 EV19 模板"段落（轻量）

**做法**：context.json 不变，prompt 加一段："如果用户的文件来自 EV19 软件，可以根据导出文件名 / zone 列 / 模板名推断学术范式"。

**优点**：
- 改动最小
- 不破坏现有 state 结构

**缺点**：
- 治标不治本——deepseek 看到文件名还是会跳过两级 UI
- 用户最初的诉求（agent 主动问 EV 模板）没实现
- 学术范式那 18 个不能完整覆盖 62 个 EV 模板的判别需求

### 推荐：选项 B（双层）

理由：
- 用户期望感受到的是 EV19 体验，但内部分析逻辑必须保持学术范式（不然报告写不出来）
- ethoinsight 库已经按学术范式组织（shoaling.py、未来的 epm.py / open_field.py），重写成本高
- 双层设计与 architecture-diagram.md 里的"上层价值 + 下层技术"分层思路一致

---

## 4. 详细设计（基于选项 B）

### 4.0 三层架构（事实 / 知识 / 流程 分离）

```
┌─────────────────────────────────────────────────────────────────┐
│ 层 1：静态注册表（事实）                                          │
│ packages/ethoinsight/ev19_templates.py                           │
│   - EV19_VARIANTS（62 变体）：arena_template / zone_template /   │
│     subject_types / array_size 等机器字段                         │
│   - EV19_CATEGORIES（20 大类）：display_name / variants 列表      │
│ 由扫描脚本自动生成，**不含任何领域解读**                          │
│ 触发更新：EV19 升级或 demodata 改动时重跑脚本                     │
└────────────────┬────────────────────────────────────────────────┘
                 │ 提供事实数据
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 层 2：渐进披露 skill（领域知识）                                  │
│ packages/agent/skills/custom/ethovision-paradigm-knowledge/      │
│   SKILL.md             ← 概览 + 双向索引（~300 行）               │
│   references/                                                    │
│     by-template/<大类>.md  ← 20 个文件，每个大类的所有变体        │
│     by-experiment/<实验>.md ← 20 个文件，每个常见实验的取舍      │
│ 行为学同事用 markdown 维护                                        │
│ Agent 渐进披露：用户问到哪个就加载哪个 references 文件            │
│ CI 测试：每个 by-template 提到的实验在 by-experiment 里都有文件； │
│           每个 by-experiment 推荐的模板都在层 1 的 _facts 里存在  │
└────────────────┬────────────────────────────────────────────────┘
                 │ Agent 调用本 skill 获取领域解读
                 ↓
┌─────────────────────────────────────────────────────────────────┐
│ 层 3：Gate 1 三步流程（agent prompt + 中间件）                    │
│ - prompt 描述三步对话流程                                         │
│ - GateEnforcementMiddleware 在 task() 边界检查                    │
│   experiment-context.json 里 ev19_template + paradigm 双字段      │
│ - 流程中 agent 调 ethovision-paradigm-knowledge skill 查           │
│   （而不是把所有解读硬编码进 prompt）                             │
└─────────────────────────────────────────────────────────────────┘
```

**为什么是三层而不是合并**：

- 层 1 用 Python 字典而非 markdown：保证可被代码（中间件、parse.py、报告生成）直接 import 使用
- 层 2 用 markdown 而非字典：行为学同事不写 Python；渐进披露能减小 prompt token
- 层 3 复用层 1（中间件查 ev19_template 是否存在）+ 层 2（agent 查领域解读）

### 4.1 新数据结构

#### 4.1.1 EV19 模板注册表

新文件 `packages/ethoinsight/ethoinsight/ev19_templates.py`：

```python
"""EthoVision XT 19 template registry.

Source: demodata/ev19 templates/ (每个目录的 templateMetaData.xml)

20 大类，62 变体。每个变体在三个独立轴上变化：
- zone_config: AllZones / NoZones / NovObjZones / FewZones / AFewZones / FeedingShelter / Subdivided{2x2|3x3|4x4} / None
- subject_type: rodent / fish / insect / mouse / rat_other / mixed (在 EV 里 vec*SubTypes 决定)
- array_size: single / quad / 16x / 1cubicle / 4cubicles / None (仅 PhenoTyper、UgoBasileFCS 有)
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class EV19Variant:
    template_id: str          # 目录名，如 "OpenFieldRectangle-AllZones"
    arena_category: str       # 大类，如 "OpenFieldRectangle"
    arena_template: str       # m_strArenaTemplate，如 "Open field, square"
    zone_template: str | None # m_strZoneTemplate，如 "Center, border, corners"
    zone_config: str | None   # 解析出的 zone 维度：AllZones / NoZones / ...
    subject_types: tuple[str, ...]  # 解析出的动物维度：("rodent",) / ("fish",) / ...
    array_size: str | None    # 阵列维度：single / Quad / 16x / ...

@dataclass(frozen=True)
class EV19Category:
    """大类。包含所有变体。"""
    category_id: str          # "OpenFieldRectangle"
    display_name_cn: str      # "矩形旷场"
    variants: tuple[str, ...] # 该大类下所有变体的 template_id

EV19_CATEGORIES: dict[str, EV19Category] = {
    "OpenFieldRectangle": EV19Category(
        category_id="OpenFieldRectangle",
        display_name_cn="矩形旷场",
        variants=(
            "OpenFieldRectangle-AllZones",
            "OpenFieldRectangle-NoZones",
            "OpenFieldRectangle-NovObjZones",
            "OpenFieldRectangle-NoZonesFishInsects",
            "OpenFieldRectangle-Subdivided2x2",
            "OpenFieldRectangle-Subdivided3x3",
            "OpenFieldRectangle-Subdivided4x4",
        ),
    ),
    # ... 全 20 个大类
}

EV19_VARIANTS: dict[str, EV19Variant] = {
    "OpenFieldRectangle-AllZones": EV19Variant(
        template_id="OpenFieldRectangle-AllZones",
        arena_category="OpenFieldRectangle",
        arena_template="Open field, square",
        zone_template="Center, border, corners",
        zone_config="AllZones",
        subject_types=("rodent", "other"),
        array_size=None,
    ),
    # ... 全 62 个变体
}
```

注意：`EV19_CATEGORIES` 和 `EV19_VARIANTS` 应当用一个 build 脚本扫描 `demodata/ev19 templates/` 下每个 `templateMetaData.xml` 自动生成，**不要手写**。手写容易抄错，且 EV 升级时无法自动适配。

#### 4.1.2 模板 ↔ 实验映射（不再用 Python 字典）

> ⚠️ 设计变更：早期草稿设计用 `DIRECT_MAPPING` + `AMBIGUOUS_MAPPING` 两个 Python dict 维护这个映射。**已废弃**。改为放进 skill 的 markdown 里（见 4.1.3）。

**理由**：
- 映射不是"从数据中能机械抽出来的事实"，而是行为学同事的**领域判断**
- 同一个模板对多个实验时（ambiguous），区分依据需要文字说明，dict 表达不出来
- markdown 让行为学同事直接维护，不用涉及 Python
- 层 2 的 by-template/by-experiment 双索引天然就是 N:M 关系，比 dict 更适合

**Agent 怎么"查映射"**：

```
场景：用户选了 ev19_template = "OpenFieldCircle-NoZones-Fish"
agent 调 ethovision-paradigm-knowledge skill 加载
  references/by-template/OpenFieldCircle.md
读到 OpenFieldCircle-NoZones-Fish 那一节的"对应学术范式"段
  - shoaling（多鱼测群体度）
  - aquatic_open_field（单鱼测探索）
agent 检查到这是 ambiguous → 触发第三步 ask_clarification
```

CI 跑一个测试：每个 `by-template/X.md` 里"对应学术范式"列出的实验，都必须在 `by-experiment/<实验>.md` 里有反向条目。这个测试避免双向不一致。

#### 4.1.3 ethovision-paradigm-knowledge skill 结构

新文件 `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md`：

```markdown
---
name: ethovision-paradigm-knowledge
description: 用户问"X 实验该用哪个 EV 模板"或"Y 模板能做什么实验"时调用。
   或在 Gate 1 三步流程的第二步（确认变体）和第三步（区分 ambiguous 实验）时调用。
allowed-tools: Read, Glob
---

# EthoVision XT 19 模板与实验知识

20 大类 EV19 模板，N:M 对应到学术实验范式。

## 何时加载本 skill 的 references/

| 用户问 | 加载 |
|---|---|
| "shoaling 用什么模板？" | references/by-experiment/shoaling.md |
| "OpenFieldCircle 能做什么？" | references/by-template/OpenFieldCircle.md |
| Gate 1 第二步：用户已选大类 OpenFieldCircle | references/by-template/OpenFieldCircle.md |
| Gate 1 第三步：用户已选模板，agent 要判断是否 ambiguous | references/by-template/<大类>.md |

## 模板 → 实验 索引（自动生成段，禁止手改）

| EV19 大类 | 变体数 | 适用实验（详见 by-template/<大类>.md） |
|---|---|---|
| OpenFieldRectangle | 7 | open_field, novel_object（详见） |
| OpenFieldCircle | 5 | open_field, shoaling, aquatic_open_field（详见） |
| ... |

## 实验 → 模板 索引（自动生成段，禁止手改）

| 实验 | 推荐模板（按优先级） | 详见 |
|---|---|---|
| shoaling | OpenFieldCircle-NoZones-Fish > AquariumTrack3D | by-experiment/shoaling.md |
| epm | PlusMaze-AllZones > PlusMaze-FewZones > PlusMaze-NoZones | by-experiment/epm.md |
| ... |

> 两个索引由 CI 任务从 references/by-template/ 和 references/by-experiment/ 反向汇总生成，
> 行为学同事不要手改这两节，改对应的 references 文件即可。
```

`references/` 目录见 review 包：[docs/review-packages/2026-04-29-ev19-templates/](../review-packages/2026-04-29-ev19-templates/)。补充完成后整体迁入 `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/`。

#### 4.1.4 experiment-context.json 字段扩展

```json
{
  "ev19_template": "OpenFieldCircle-NoZones-Fish",
  "arena_template": "Open field, circle",
  "zone_template": null,
  "subject_type": "fish",

  "paradigm": "shoaling",
  "paradigm_cn": "斑马鱼鱼群行为",
  "category": "zebrafish",

  "paradigm_confirmed_at": "2026-04-29T...",
  "gate_completed": ["gate1_paradigm"],

  "_schema_version": 2
}
```

旧 thread 的 context 文件（无 `_schema_version` 或 `< 2`）：
- 读取时 `experiment_context.read_context()` 在缺字段时返回 None / 默认值
- 不强制升级（fail open）
- middleware 见到旧 schema 跳过新增检查

### 4.2 流程改造

#### 4.2.1 Gate 1 三步对话流（manual mode）

```
用户上传文件并请求分析
    ↓
Lead agent 检测 experiment-context.json 不存在
    ↓
[第一步] ask_clarification — 选大类（20 个）
  question: "您在 EthoVision XT 19 里用的是哪个 arena 大类？"
  options: [
    "矩形旷场（OpenFieldRectangle）",
    "圆形旷场（OpenFieldCircle）",
    "高架十字迷宫（PlusMaze）",
    "零迷宫（ZeroMaze）",
    "Morris 水迷宫（MWM）",
    "Barnes 迷宫（BarnesMaze）",
    "T 迷宫（T-Maze）",
    "Y 迷宫（Y-Maze）",
    "十字迷宫-鱼（Cross Maze-Fish）",
    "8 臂迷宫（Radial-8-arm）",
    "PhenoTyper 居家观察",
    "鱼缸 3D 跟踪（AquariumTrack3D）",
    "DanioVision 96 孔板",
    "飞行舱 3D 跟踪（FlightChamberTrack3D）",
    "孔板（WellPlate）",
    "强迫游泳（PorsoltCylinder）",
    "社交偏好（Sociability）",
    "主动回避（UgoBasileActiveAvoidance）",
    "恐惧条件化（UgoBasileFCS）",
    "无模板 / 自定义（NoTemplate）",
  ]
    ↓
用户选择 arena_category = "OpenFieldCircle"
    ↓
[第二步] ask_clarification — 在该大类下选具体变体
  仅显示该大类下的变体（少则 1 个，多则 15 个）。
  当大类只有 1 个变体时（AquariumTrack3D / FlightChamberTrack3D / DanioVision /
  UgoBasileActiveAvoidance / NoTemplate）跳过此步。

  示例（OpenFieldCircle 5 个变体）：
  question: "请进一步选择具体配置（决定 zone 列和适用动物）"
  options: [
    "AllZones（带 Center/Border/Corners 等分区，啮齿/其他）",
    "NovObjZones（带新物体识别 zones，啮齿/其他）",
    "NoZones-Fish（无分区，鱼）",
    "NoZones-Insects（无分区，昆虫）",
    "NoZones-Rodents-Other（无分区，啮齿/其他）",
  ]
    ↓
用户选择 ev19_template = "OpenFieldCircle-NoZones-Fish"
    ↓
agent 调 ethovision-paradigm-knowledge skill 加载 references/by-template/OpenFieldCircle.md
读取 OpenFieldCircle-NoZones-Fish 那一节的"对应学术范式"
判断：列出 2 个范式 → ambiguous
    ↓
[第三步（仅 ambiguous 时）] ask_clarification
  question: "这条数据的实验目的是？"
  options: [
    "斑马鱼鱼群行为（多鱼同竞技场）",
    "鱼旷场（单鱼测探索）"
  ]
    ↓
用户选 paradigm = "shoaling"
    ↓
set_experiment_paradigm 写入 experiment-context.json（含 ev19_template + paradigm 双字段）
    ↓
后续 task() 调用通过 GateEnforcementMiddleware 检查
```

#### 4.2.2 中间件约束（兼容选项 "开门 + 补救"）

当前会话用户已表达倾向：「文件名包含明确范式信号时允许 agent 推断，但要在响应中告知用户可修正」。

**实现方式**：

1. `set_experiment_paradigm` 工具增加 `inferred_from` 参数（值如 `"user_clarification"` / `"filename_inference"` / `"context_inference"`）
2. 当 `inferred_from != "user_clarification"` 时，工具返回的 ToolMessage 文本里强制包含一段提示：
   ```
   ⚠️ 我从「{evidence}」推断出范式为 {paradigm_cn}（EV19 模板 {ev19_template}）。
   如果不正确，请直接说「重选范式」我会弹出选择菜单。
   ```
3. agent 后续若收到"重选范式"消息，必须删除 experiment-context.json 后重走 Gate 1 三步流

注：这个补救机制需要在 prompt 里专门教 agent 怎么响应"重选范式"。

#### 4.2.3 文件触发改造

当前 prompt 「文件上传后才触发 Gate 1」。改为：

- **触发条件 A**：用户消息提到实验/分析意图（即便没上传）→ agent 提示"请上传 EthoVision XT 导出数据"+ 顺便说明会先确认模板
- **触发条件 B**：检测到上传文件 → 执行 Gate 1 三步流

注意：触发条件 A 是 **lead 自己回应**，不经过 task()，所以 GateEnforcementMiddleware 拦截不到。这一步**只能靠 prompt** 约束（这是上一会话用户问的"上传前反问"问题的实际触发点）。

---

## 5. 实现拆解

### Step 0：行为学同事补充 review 包（异步进行，2-5 天看同事节奏）

> ✅ 已完成准备工作。Review 包路径：[docs/review-packages/2026-04-29-ev19-templates/](../review-packages/2026-04-29-ev19-templates/)

行为学同事填以下文件（按优先级）：

- P0（先填这 7 个就能跑通 shoaling/EPM/OFT 三大范式）：
  - `by-experiment/shoaling.md`
  - `by-experiment/epm.md`
  - `by-experiment/open_field.md`
  - `by-template/OpenFieldRectangle.md`
  - `by-template/OpenFieldCircle.md`
  - `by-template/PlusMaze.md`
  - `by-template/AquariumTrack3D.md`
- P1：剩余 17 个 by-experiment + 17 个 by-template

详见 [review 包 README](../review-packages/2026-04-29-ev19-templates/README.md)。

### Step 1：建模事实表（1 天）

> ✅ 起草脚本已就绪：[scripts/build_ev19_template_index.py](../../scripts/build_ev19_template_index.py)

- 把 `_facts.json` 转成 Python 字典 `EV19_VARIANTS` / `EV19_CATEGORIES`
- 写 `packages/ethoinsight/ethoinsight/ev19_templates.py`
- 单测：覆盖 20 大类、62 变体的 round-trip

### Step 2：搬 review 包到 skill 目录（0.5 天）

行为学同事完成 P0 的 7 个文件后：

- 创建 `packages/agent/skills/custom/ethovision-paradigm-knowledge/` 目录
- 写 `SKILL.md`（参见 4.1.3 的结构）
- 把 review 包的 by-template/ 和 by-experiment/ 移进 references/
- 写 CI 脚本：从 references/ 反向汇总生成 SKILL.md 里的两个索引段
- 写 CI 测试：双向一致性（template 提到的实验在 by-experiment 都有；by-experiment 推荐的模板在 _facts.json 都存在）
- 在 `extensions_config.json` 里启用这个 skill

### Step 3：experiment-context schema 升级（0.5 天）

- 增加 `ev19_template` / `arena_template` / `zone_template` / `subject_type` 字段
- 增加 `_schema_version: 2`
- `experiment_context.py` 加新函数 `read_ev19_template()`、`is_ev19_template_set()`
- `set_experiment_paradigm` 工具签名扩展，加 `ev19_template` 参数

### Step 4：prompt 改写（1 天）

- 替换"7 大类 18 范式"表为 EV19 模板分组列表
- 改写 Gate 1 描述为三步流程
- 加"上传前触发"分支说明
- 加"重选范式"补救机制说明
- 加"调 ethovision-paradigm-knowledge skill"的指令（什么时候调、调哪个 reference 文件）

### Step 5：GateEnforcementMiddleware 增强（0.5 天）

- 增加对 `inferred_from` 字段的检查
- 文件名推断或对话推断时附加 ToolMessage 提示

### Step 6：UI 端 ask_clarification 改造（1 天）

- 第一步显示 20 个大类（按使用频率排序：OpenFieldRectangle / OpenFieldCircle / PlusMaze / ZeroMaze / MWM 优先）
- 第二步只显示该大类下的变体（1-15 个，少于等于 1 个时跳过）
- 第三步（仅 ambiguous 时）显示该模板对应的多个学术范式
- 检查 frontend 是否支持"返回上一步"

### Step 7：迁移与回归（0.5 天）

- 老 thread 的 context.json 缺新字段 → fail open
- 跑全量测试，重点测 memory / gate enforcement / experiment_context
- E2E 烟测：用 demodata 各跑一次，确认每种模板都能走通

### Step 8：文档（0.5 天）

- 更新 [quality-gates.md](packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md)
- 更新 [paradigm-analysis-tools-spec.md](docs/specs/paradigm-analysis-tools-spec.md)
- 写交接文档

**总工时**：5-7 天工程工时（不含行为学同事 review 时间，那是异步进行的）

---

## 6. 风险与开放问题

### 6.1 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 模板 → 范式映射准确性 | 错误映射会导致报告解读用错语言 | 行为学同事必 review；在 ambiguous 段宁可多问一步 |
| 20 个大类 UI 不算少 | 用户体验下降 | 按使用频率排序前 5-7 个，"更多"折叠 |
| PhenoTyper 第二步 15 个变体太多 | 该大类下选择困难 | 第二步可拆成两小步（先选阵列规模，再选 zone 配置）；或加搜索框 |
| 老 thread 兼容 | 已有用户的 context.json 是旧 schema | fail open 策略；可加一次性迁移脚本 |
| ethoinsight library 重命名风险 | 现在 `shoaling.py` 是按学术范式命名 | 选项 B 不需要改 ethoinsight 库 |
| deepseek 仍可能跳步 | 即使三步流程也可能被 LLM 简化 | 用代码兜底（task() 时 gate 中间件检查 ev19_template 存在）|
| EV19 升级（V20 / V21）字段变化 | 字典自动生成脚本可能失配 | 生成脚本写防御式：已知字段缺失时记 warning 不 crash |

### 6.2 开放问题（需要用户/行为学同事回答）

1. **EV19 之外的旧版本数据怎么办？**（EV15、EV16…）
   - 是否要支持？还是只支持 EV19？

2. **没有用 EV 模板的研究怎么办？**
   - 例如用户用 DeepLabCut 或 Bonsai 自己的 pipeline 生成轨迹
   - context.json 里 `ev19_template` 应该填什么？`null`？还是给个 `external_pipeline`？

3. **Shoaling 在 EV19 里实际用哪个模板？**
   - 是 `AquariumTrack3D`、`OpenFieldCircle-NoZones-Fish` 还是 `NoTemplate`？
   - 行为学同事确认

4. **Sample 时 control n=2 这种情况，是否要加在 Gate 2？**
   - 当前 Gate 2 只看 critical warnings；样本量可以做成另一个独立 Gate
   - 但今天用户已经把 n=2/3 接受为"可分析"，所以保留现状

5. **"重选范式"触发字符串**
   - 用户说"重选"、"我选错了"、"换一个"等都要触发？
   - 还是只接受精确字符串？（前者要 LLM 判断；后者更刚性）

---

## 7. 与之前会话的关联

### 不变的部分

- 上一会话（4-28-gate-enforcement）建立的中间件框架不动；只是 `_check_gate1` 的判定从「context.json 是否存在」改为「context.json 是否有 ev19_template + paradigm 双字段」
- Gate 2 数据质量检查不动
- task() 边界拦截不动

### 需要回写的部分

- 4-28 交接文档里 P1 标记的"加强 Gate 1 prompt"任务可以合并到本设计实施
- 4-28 上游 issue 草稿（event loop fix）已经独立提交，与本设计无冲突

---

## 8. 决策记录

- ✅ 选**选项 B（双层）**：用户语言走 EV19，内部走学术范式，互不替代
- ✅ Gate 1 改成**三步对话流**：arena_group → ev19_template → paradigm（仅 ambiguous）
- ✅ 上传前触发条件 A 仅 prompt 约束，不靠中间件
- ✅ 文件名推断走"开门 + 补救"模式（用户已选）
- ✅ **领域知识做成独立 skill**（`ethovision-paradigm-knowledge`），渐进披露 by-template/by-experiment 双向索引；不放进 prompt，也不放进 ethoinsight 库 Python 字典
- ✅ **三层架构**：层 1 静态注册表（事实，Python）、层 2 skill（领域知识，markdown）、层 3 流程（prompt + 中间件）
- ✅ **行为学同事用 markdown 维护**领域知识，CI 测双向一致性
- ❌ 暂不改 ethoinsight library 的命名（保持按学术范式组织）
- ❌ 暂不支持 EV15/16 / DeepLabCut（留待 v0.1 后）
- ❌ **废弃旧设计**：用 `DIRECT_MAPPING` + `AMBIGUOUS_MAPPING` 两个 Python dict 做映射（理由见 4.1.2）

---

## 9. 验收标准

设计实施完后，下面三个场景都必须通：

### 场景 1：用户没上传，先问

```
用户: 我刚做完测试想分析数据
agent: 好的。在开始前请上传 EthoVision XT 导出文件。同时请告诉我：您在 EV19 里用的是哪类模板？
       [选项: 旷场 / 高架迷宫 / 零迷宫 / ...]
```

### 场景 2：用户上传 + 文件名带模板

```
用户: 这是数据，1 和 2 是对照组，3、4、5 是实验组（上传 5 个文件）
agent: 我从文件名「Shoaling behavior」推断模板是 OpenFieldCircle-NoZones-Fish，范式是斑马鱼鱼群行为。
       ⚠️ 如果不正确请说"重选范式"。
       现在分析这两个组的差异...（如果用户不反对则继续）
```

### 场景 3：用户上传但文件名无线索

```
用户: 帮我分析这些数据（上传 trial_001.txt 等无意义命名）
agent: 数据已收到。请告诉我您在 EV19 里用的是哪类模板？
       [选项: 旷场 / 高架迷宫 / ...]
用户: 旷场
agent: 请进一步选择：
       [选项: OpenFieldRectangle-AllZones / OpenFieldRectangle-NoZones / OpenFieldCircle-AllZones / ...]
```

---

## 10. 下一步

1. 用户 review 本文档（关注：选项 B 是否正确方向 / 6.2 开放问题答案）
2. 行为学同事 review：6.2 问题 3，以及 DIRECT_MAPPING 草稿
3. 若同意 → 按 Step 1-7 执行
4. 若分歧 → 重新讨论选项 A 或 C
