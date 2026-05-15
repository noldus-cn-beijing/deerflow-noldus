# EthoVision XT 原始数据（Raw Data）格式说明

> 适用版本：EthoVision XT 16 / 18（Noldus Information Technology）
> 文档日期：2026-04-28

---

## 1. 概述

本目录中的轨迹文件是从 **Noldus EthoVision XT** 软件通过「导出 → 原始数据（Raw Data）」生成的。每个 Trial 的每只被试（Subject）在每块观察区（Arena）独立输出为一个 `.txt` 文件，文件之间以文件名中的 `Trial`、`Arena`、`Subject` 编号作为区分。

### 10 组实验概览

| 目录 | 范式 | 物种 | 追踪方式 | EthoVision 版本 |
|------|------|------|----------|-----------------|
| 旷场_小鼠_三点 | Open Field | 小鼠 | 三点追踪 | XT160 |
| 高架十字迷宫_小鼠_三点 | Elevated Plus Maze | 小鼠 | 三点追踪 | XT180 |
| 巴恩斯迷宫_小鼠_三点 | Barnes Maze | 小鼠 | 三点追踪 | XT180 |
| 新物体识别_小鼠_三点 | Novel Object Recognition | 小鼠 | 三点追踪 | XT180 |
| 三箱社交_小鼠_三点 | Three-Chamber Social | 小鼠 | 三点追踪 | XT180 |
| 社交互动_小鼠_三点 | Social Interaction | 小鼠 | 三点追踪 | XT180 |
| 精细行为_小鼠_三点 | Behavior Recognition | 小鼠 | 三点追踪 | XT180 |
| 精细行为_大鼠_三点 | Behavior Recognition | 大鼠 | 三点追踪 | XT180 |
| 群聚行为_斑马鱼成鱼_单点 | Shoaling | 斑马鱼成鱼 | 单点追踪 | XT180 |
| 高通量_斑马鱼幼鱼_单点 | DanioVision 96-well | 斑马鱼幼鱼 | 单点追踪 | XT180 |

---

## 2. 文件整体结构

### 2.1 编码与分隔符

| 属性 | 值 |
|------|-----|
| 字符编码 | **UTF-16LE（带 BOM）** |
| 字段分隔符 | 分号 `;` |
| 行分隔符 | CR+LF（`\r\n`） |

> **Python 读取建议**：使用 `open(path, encoding='utf-16')` 即可自动处理 BOM 和行尾。

### 2.2 文件整体布局

每个轨迹文件分为三个区域：

```
第 1 行              → 标题行数声明（值 = "39"，即 39 行头信息）
第 2–37 行           → 元数据（36 组 Key;Value 对）
第 38 行             → 列名（Column Headers）
第 39 行             → 单位（Units）
第 40 行及之后       → 数据行（Data Rows），每行一条采样记录
```

> 注：虽然第 1 行声明的标题数为 39，这 39 行包含了第 1 行自身、元数据、列名行和单位行。数据区从第 40 行开始。

### 2.3 总体结构示意图

```
┌──────────────────────────────────────────────┐
│ Line 1:   "标题行数："; "39"                    │  ← 头信息计数器
├──────────────────────────────────────────────┤
│ Line 2:   "实验"; "Open Field test XT160"      │
│ Line 3:   "系统 自变量"; ""                      │
│   ...                                         │  ← 36 条元数据（Key;Value）
│ Line 37:  " "; ""                             │
├──────────────────────────────────────────────┤
│ Line 38:  Column1;Column2;...;ColumnN         │  ← 列名（因实验/范式而异）
├──────────────────────────────────────────────┤
│ Line 39:  Unit1;Unit2;...;UnitN              │  ← 单位
├──────────────────────────────────────────────┤
│ Line 40:  Value;Value;...;Value              │
│ Line 41:  Value;Value;...;Value              │  ← 逐行采样数据
│ ...                                           │
└──────────────────────────────────────────────┘
```

---

## 3. 元数据头（Metadata Headers）

每个文件的第 2–37 行包含 36 组元数据键值对，以分号分隔。所有文件共享相同结构，仅值不同。

### 完整元数据字段列表

| 序号 | 字段名（中文） | 字段名（英文等效） | 说明 |
|------|---------------|-------------------|------|
| 1 | 实验 | Experiment | EthoVision XT 中的实验名称 |
| 2 | 系统 自变量 | System Independent Variable | 系统级自变量，通常为空 |
| 3 | 试验名称 | Trial Name | 如 `Trial     1`（含多个空格） |
| 4 | 试验 ID | Trial ID | 数值 ID，通常为 0 |
| 5 | 观察区名称 | Arena Name | 如 `Arena 1`，三箱社交有 Arena 1 和 Arena 2 |
| 6 | 观察区 ID | Arena ID | 数值 ID，通常为 0 |
| 7 | 对象名称 | Subject Name | 如 `Subject 1` |
| 8 | 对象 ID | Subject ID | 数值 ID，通常为 0 |
| 9 | 观察区设定 | Arena Settings | 引用观察区设置名称 |
| 10 | 检测设置 | Detection Settings | 引用检测方法名称（如 `Dark mouse`） |
| 11 | 试验控制设置 | Trial Control Settings | 试验控制规则名称 |
| 12 | 开始时间 | Start Time | 试验开始时间戳，格式 `MM/DD/YYYY HH:MM:SS.fff` |
| 13 | 试验持续时间 | Trial Duration | 完整试验时长，格式 `+ HH:MM:SS.fff` |
| 14 | 记录在以下时间之后 | Record After | 延迟多久才开始记录（用于排除起始干扰） |
| 15 | 录制时长 | Recording Duration | 实际采集的数据时长（≤ 试验持续时间） |
| 16 | 轨迹 | Track File | 内部 .trk 文件名 |
| 17 | 轨迹追踪源 | Track Source | 通常为空 |
| 18 | 视频文件 | Video File | 源视频完整路径 |
| 19 | 视频开始时间 | Video Start Time | 视频文件时间戳 |
| 20 | 试验状态 | Trial Status | `已采集`（Acquired） |
| 21 | 采集状态 | Acquisition Status | `已采集` |
| 22 | 跟踪状态 | Tracking Status | `已采集` |
| 23 | 视频文件状态 | Video File Status | `外部`（External） |
| 24 | 同步状态 | Sync Status | `不可用`（Not available） |
| 25 | 参考时长 | Reference Duration | 参考信号时长 |
| 26 | 参考时间 | Reference Time | 参考信号时间戳 |
| 27 | Sof 文件 | Sof File | 通常为空 |
| 28 | 缺失样本 | Missing Samples | 百分比，如 `0.0 %` |
| 29 | 未找到对象 | Subject Not Found | 百分比，如 `2.1 %` |
| 30 | 插值样本 | Interpolated Samples | 百分比，如 `0.0 %` |
| 31 | 后处理器计算结果 | Post-processor Result | 如 `Tr::eIdrUndefined` |
| 32 | LED 图案 | LED Pattern | 通常为空 |
| 33 | 从 SAT 导入的数据 | Data From SAT | `否` |
| 34 | 用户定义的 自变量 | User-defined Independent Variable | 用户在实验中添加的自定义变量，可为空 |
| 35 | [自定义字段] | [User-defined Field] | 实验者自定义字段，不同实验内容不同（见 3.1） |
| 36 | [空行] | [Empty] | 固定空行，无值 |

### 3.1 第 36 号自定义字段示例

不同实验在此位置存储了不同信息：

| 实验 | 字段名 | 示例值 |
|------|--------|--------|
| 旷场 | `Mouse ID` | `123` |
| 三箱社交 | `Type` | `WT` |
| 精细行为（小鼠） | `Animal ID` | `A001` |
| 精细行为（大鼠） | `Phase` | `Habituation` |
| 高通量 | `Treatment` | `Control` |
| 其他 | `<User-defined 1>` | 空 |

---

## 4. 列名详解——三点追踪（Three-Point Tracking）

三点追踪模式追踪动物的三个身体点：**鼻点（Nose）**、**中心点（Center）** 和 **尾点（Tail）**。

### 4.1 基础列（所有三点追踪实验共有）

| 列号 | 列名（中文） | 列名（英文） | 单位 | 说明 |
|------|-------------|-------------|------|------|
| 0 | 试用时间 | Trial time | 秒 | 从 Trial 开始计的时间（含延迟） |
| 1 | 录制时间 | Recording time | 秒 | 从实际开始记录计的时间（Trial time - 延迟时间） |
| 2 | X 中心 | X center | 厘米 | 动物中心点 X 坐标 |
| 3 | Y 中心 | Y center | 厘米 | 动物中心点 Y 坐标 |
| 4 | X 鼻点 | X nose | 厘米 | 鼻尖 X 坐标 |
| 5 | Y 鼻点 | Y nose | 厘米 | 鼻尖 Y 坐标 |
| 6 | X 尾 | X tail | 厘米 | 尾根 X 坐标 |
| 7 | Y 尾 | Y tail | 厘米 | 尾根 Y 坐标 |
| 8 | 区域 | Area | 厘米² | 动物轮廓的像素面积（校准后） |
| 9 | 面积变化 | Area change | 厘米² | 与上一帧的面积差值 |
| 10 | 伸长 | Elongation | 无 | 身体伸长率 = 长轴/短轴，越大表示身体越细长 |
| 11 | 方向 | Direction | 度 | 动物朝向角度（鼻到尾的方向与水平线夹角，范围约 -180° 至 +180°） |

> 注意：某些实验（如社交互动）基础列的 8–9 为「移动距离」和「速度」而非「区域」和「面积变化」。

### 4.2 各实验特有列

#### 旷场（Open Field）— 15 列

| 列号 | 列名 | 说明 |
|------|------|------|
| 12 | In zone | 是否在特定区域内（0/1，- 表示无） |
| 13 | Dark mice | 区域名称（本实验自定义区域） |
| 14 | (空) | 尾部空列 |

#### 高架十字迷宫（Elevated Plus Maze）— 17 列

| 列号 | 列名 | 说明 |
|------|------|------|
| 12 | In zone(Open arms /中心点) | 中心点是否在开臂区域 |
| 13 | In zone(Closed arms /中心点) | 中心点是否在闭臂区域 |
| 14 | When In open arms > 5 s | 累计在开臂超过 5 秒的标记 |
| 15 | Result 1 | 结果变量 1 |
| 16 | (空) | 尾部空列 |

#### 巴恩斯迷宫（Barnes Maze）— 16 列

| 列号 | 列名 | 说明 |
|------|------|------|
| 12 | Velocity | 瞬时速度 |
| 13 | Distance moved | 累积移动距离 |
| 14 | Control | 对照组标识 |
| 15 | (空) | 尾部空列 |

#### 新物体识别（Novel Object Recognition）— 26 列

本实验定义了多个物体区域（Familiar object 1/2，Novel object），列 12–25 为物体交互指标：

| 列号 | 列名 | 说明 |
|------|------|------|
| 12 | Nose within object zone(Familiar object 1 /鼻尖) | 鼻尖是否在熟悉物体 1 区域 |
| 13 | Nose within object zone(Familiar object 2 /鼻尖) | 鼻尖是否在熟悉物体 2 区域 |
| 14 | Nose within object zone(Novel object /鼻尖) | 鼻尖是否在新物体区域 |
| 15 | Nose within object zone(Familiar object /鼻尖) | 鼻尖是否在熟悉物体区域（泛指） |
| 16–19 | Distance to objects(...) | 鼻尖到各物体的距离 |
| 20 | Head directed to object 1 | 朝向物体 1 |
| 21 | Head directed to object 2 | 朝向物体 2 |
| 22 | Head directed to Familiar object | 朝向熟悉物体 |
| 23 | Head directed to Novel object | 朝向新物体 |
| 24 | Result 1 | 结果变量 1 |
| 25 | (空) | 尾部空列 |

#### 三箱社交（Three-Chamber Social）— 18 列

每个 Trial 有 **两个 Arena**（Arena 1 = Social zone 侧，Arena 2 = Control zone 侧），各生成一个文件：

| 列号 | 列名 | 说明 |
|------|------|------|
| 12 | In zone(Social zone /鼻尖) | 鼻尖是否在社交区域 |
| 13 | In zone(Control zone /鼻尖) | 鼻尖是否在对照区域 |
| 14 | Head directed to Social zone | 头朝向社交区域 |
| 15 | Head directed to Control zone | 头朝向对照区域 |
| 16 | [自定义字段名，如 `WT`] | 实验者自定义分组标签 |
| 17 | (空) | 尾部空列 |

#### 社交互动（Social Interaction）— 31 列

每 Trial 追踪 2 只小鼠（Subject 1 和 Subject 2），各自独立文件。此实验基础列略有不同（第 8 列为「移动距离」，第 9 列为「速度」）：

| 列号 | 列名 | 说明 |
|------|------|------|
| 0–7 | 试用时间;录制时间;X/Y中心;X/Y鼻点;X/Y尾 | 标准三点坐标 |
| 8 | 移动距离 | Distance moved（替代了区域） |
| 9 | 速度 | Velocity（替代了面积变化） |
| 10–11 | 伸长;方向 | 标准列 |
| 12–15 | Side by side(...) | 并排行为检测（Subject 1/2 × 同向/反向） |
| 16–27 | Proximity Subject 1/2(...) | 趋近检测（鼻尖到另一只的中心点/鼻尖/尾根） |
| 28 | JavaScript state - Subject 1 approaches Subject 2 | JS 状态机：Subject 1 趋近 Subject 2 |
| 29 | JavaScript state - Subject 2 approaches Subject 1 | JS 状态机：Subject 2 趋近 Subject 1 |
| 30 | Result 1 | 结果变量 1 |

#### 精细行为（小鼠）— 30 列

| 列号 | 列名 | 说明 |
|------|------|------|
| 0–11 | 试用时间–方向 | 标准三点基础列 |
| 12 | Unknown | 未识别行为类别（0/1） |
| 13 | Grooming | 理毛（0/1） |
| 14 | Rearing supported | 扶壁站立（0/1） |
| 15 | Rearing unsupported | 无支撑站立（0/1） |
| 16 | Resting | 静止（0/1） |
| 17 | Sniffing | 嗅探（0/1） |
| 18 | Walking | 行走（0/1） |
| 19 | Hopping | 跳跃（0/1） |
| 20–27 | Behavior probability(...) | 对应 8 类行为的识别概率（0–1 连续值） |
| 28 | Result 1 | 结果变量 1 |
| 29 | (空) | 尾部空列 |

#### 精细行为（大鼠）— 34 列

比小鼠多了 `Drinking`、`Eating`、`Jumping`、`Twitching` 四个行为类别（共 10 类 + Unknown = 11 个行为列 + 对应概率列）：

| 列号 | 列名 | 说明 |
|------|------|------|
| 12 | Rearing unsupported | 无支撑站立 |
| 13 | Rearing supported | 扶壁站立 |
| 14 | Drinking | 饮水 |
| 15 | Eating | 进食 |
| 16 | Grooming | 理毛 |
| 17 | Jumping | 跳跃 |
| 18 | Resting | 静止 |
| 19 | Sniffing | 嗅探 |
| 20 | Twitching | 抽动 |
| 21 | Walking | 行走 |
| 22–31 | Behavior probability(...) | 对应 10 类行为的概率 |
| 32 | Result 1 | 结果变量 1 |
| 33 | (空) | 尾部空列 |

---

## 5. 列名详解——单点追踪（Single-Point Tracking）

单点追踪仅追踪动物的 **中心点（Center Point）**，无鼻点/尾点。适用于斑马鱼实验。

### 5.1 基础列（所有单点追踪实验共有）

| 列号 | 列名（中文） | 列名（英文） | 单位 | 说明 |
|------|-------------|-------------|------|------|
| 0 | 试用时间 | Trial time | 秒 | 从 Trial 开始计的时间 |
| 1 | 录制时间 | Recording time | 秒 | 从实际记录开始计的时间 |
| 2 | X 中心 | X center | 毫米 | 中心点 X 坐标 |
| 3 | Y 中心 | Y center | 毫米 | 中心点 Y 坐标 |
| 4 | 区域 | Area | 毫米² | 动物轮廓面积 |
| 5 | 面积变化 | Area change | 毫米² | 面积变化量 |
| 6 | 伸长 | Elongation | 无 | 身体长轴/短轴 |
| 7 | 移动距离 | Distance moved | 毫米 | 与上一帧之间的移动距离 |
| 8 | 速度 / Velocity | Velocity | 毫米/秒 | 瞬时移动速度 |
| 9 | [自定义] | — | — | 实验者定义的附加列 |

### 5.2 群聚行为（Shoaling）— 11 列

每个 Trial 追踪 **5 条鱼**（Subject 1–5），各输出一个文件。

| 列号 | 列名 | 说明 |
|------|------|------|
| 9 | Result 1 | 结果变量 1 |
| 10 | (空) | 尾部空列 |

### 5.3 高通量（DanioVision 96-well）— 11 列

每个 Trial 有 **96 个孔（Well）**，对应 96 个轨迹文件。

| 列号 | 列名 | 说明 |
|------|------|------|
| 9 | Control | 对照组标识（本实验中的 Treatment 名） |
| 10 | (空) | 尾部空列 |

---

## 6. 文件命名规范

### 6.1 轨迹文件（`轨迹-*.txt`）

```
轨迹-{实验名称}-Trial     {N}-Arena {A}-Subject {S}.txt
```

- `轨迹-` 前缀：标识这是轨迹数据文件。
- `{实验名称}`：EthoVision XT 中的 Experiment 名称，含空格和大小写。
- `Trial     {N}`：**Trial 后跟 5 个空格**，再跟试验编号（如 `Trial     1`、`Trial     2`）。
- `Arena {A}`：观察区编号（通常为 1；三箱社交有 Arena 1 和 Arena 2）。
- `Subject {S}`：被试编号。
- 文件名中**不含**扩展名以外的多余内容。

**示例**：
```
轨迹-Open Field test XT160-Trial     1-Arena 1-Subject 1.txt
轨迹-Social Approach test with Deep Learning XT180-Trial     2-Arena 2-Subject 1.txt
```

### 6.2 高通量实验的特殊文件类型

高通量（DanioVision）目录包含**三种**文件，由前缀区分：

#### 轨迹文件
```
轨迹-DanioVision with 96 wells XT180-Trial     {N}-{Well}-Subject 1.txt
```
- 96 个孔各一个轨迹文件，`Subject 1` 固定不变。
- `{Well}` 范围为 1–96。

#### 硬件事件文件
```
硬件-DanioVision with 96 wells XT180-Trial     {N}-{Well}.txt
```
- 记录硬件 I/O 事件（如白光刺激 ON/OFF）。
- 列结构：试用时间, 录制时间, 设备, 命令/信号, 名称, 值。

#### 试验控制事件文件
```
试验控制-DanioVision with 96 wells XT180-Trial     {N}-{Well}.txt
```
- 记录试验控制规则触发事件（如 Start-stop trial）。
- 列结构：试用时间, 录制时间, 事件, 规则, 条件, 操作, 运算符, 参考, 变量, 值, 以及 4 组参考/重复列。

### 6.3 各实验的文件数量特征

| 实验 | Trials | Arena/Well 数 | Subjects/文件 | 总轨迹文件数 |
|------|--------|--------------|--------------|-------------|
| 旷场 | 2 | 1 | 1 | 2 |
| 高架十字迷宫 | 1 | 1 | 1 | 1 |
| 巴恩斯迷宫 | 3 | 1 | 1 | 3 |
| 新物体识别 | 3 | 1 | 1 | 3 |
| 三箱社交 | 2 | **2** | 1 | 4 (=2×2) |
| 社交互动 | 2（推测） | 1 | **2** | 4 (=2×2) |
| 精细行为（小鼠） | ≥1 | 1 | 1 | ≥1 |
| 精细行为（大鼠） | ≥1 | 1 | 1 | ≥1 |
| 群聚行为 | ≥1 | 1 | **5** | ≥5 |
| 高通量 | ≥1 | **96** | 1 | ≥96 |

---

## 7. 解析指南

### 7.1 Python 示例代码

```python
# -*- coding: utf-8 -*-
def read_ethovision_raw(filepath):
    """读取 EthoVision XT raw data 文件，返回 (metadata_dict, columns, units, data_rows)"""
    with open(filepath, 'r', encoding='utf-16') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 第 1 行：header 计数
    n_header = int(lines[0].split(';')[1].strip('"'))

    # 元数据（lines[1] 到 lines[n_header-3]）
    metadata = {}
    for line in lines[1:n_header-2]:
        parts = line.split(';')
        if len(parts) >= 2:
            key = parts[0].strip('"')
            val = parts[1].strip('"')
            metadata[key] = val

    # 列名（lines[n_header-2]）
    columns = [c.strip('"') for c in lines[n_header-2].split(';')]

    # 单位（lines[n_header-1]）
    units = [u.strip('"') for u in lines[n_header-1].split(';')]

    # 数据行（lines[n_header:] 起）
    data_rows = []
    for line in lines[n_header:]:
        parts = line.split(';')
        row = []
        for p in parts:
            p = p.strip('"')
            # 尝试转为数字
            try:
                row.append(float(p))
            except ValueError:
                # 保留原始字符串（如 "-", "WT", "Trial 1" 等）
                row.append(p if p != '' else None)
        data_rows.append(row)

    return metadata, columns, units, data_rows


# 使用示例
meta, cols, units, data = read_ethovision_raw(
    '轨迹-Open Field test XT160-Trial     1-Arena 1-Subject 1.txt'
)
print(f"实验：{meta.get('实验')}")
print(f"列数：{len(cols)}")
print(f"数据行数：{len(data)}")
print(f"第一条数据：{data[0]}")
```

### 7.2 关键注意事项

1. **编码**：必须使用 `utf-16`（UTF-16LE with BOM）打开文件，否则会出现乱码。
2. **分隔符**：分号 `;`，不是逗号、Tab 或空格。
3. **列尾空列**：所有文件的最后一列均为空列（空字符串），读取后可丢弃。
4. **缺失值**：字符串 `"-"` 表示该帧该列无有效数据（如首帧无速度/移动距离）。
5. **文件名空格**：`Trial` 后跟 **5 个空格**，字符串匹配时注意使用通配符或处理多个连续空格。
6. **单位差异**：小鼠实验坐标单位为厘米（cm），斑马鱼实验为毫米（mm），需根据元数据中「实验」字段判断。
7. **社交互动列名**：社交互动的第 8、9 列为「移动距离」和「速度」（而非标准的「区域」「面积变化」），解析时注意实验类型。
8. **Zone 名称**：`In zone` 后面的括号内容（如 `Social zone /鼻尖`）为 EthoVision 自动生成，格式为 `Zone名称 /追踪点名称`。
9. **Trial time vs Recording time**：Trial time 从 Trial 开始计时（含延迟），Recording time 从实际开始记录后计时。两者差值等于元数据中「记录在以下时间之后」的值。

---

## 8. 坐标系与区域说明

- X 轴正方向为右，Y 轴正方向为上。
- 坐标原点在视频画面左上角。
- 坐标值已经过 EthoVision 的校准（Calibration），从像素值转换为实际距离（厘米或毫米）。
- `In zone` 列的值：`1` = 在区域内，`0` = 不在区域内，`-` = 不适用/未定义。
- 方向角（Direction）以度为单位，朝向角度范围约 -180° 至 +180°，具体取决于 EthoVision 版本的 angular 模式设置。
