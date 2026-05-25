# MouseGPT 论文借鉴分析

> 论文: Xu et al., "MouseGPT: A Large-scale Vision-Language Model for Mouse Behavior Analysis", bioRxiv 2025.03.27.645630
>
> 阅读日期: 2026-05-22

## 论文核心

MouseGPT 用 VLM 从原始视频帧直接生成自然语言行为描述（Overall/Head/Limb/Torso/Keywords 五段 JSON），再通过 text embedding → UMAP 聚类 → LLM 精炼，实现行为画像、细粒度分析、新颖行为发现、表型预测。配套 LangChain Agent 提供自然语言交互界面。

技术栈: 8 相机 4K 60fps → YOLOv5 + HRNet 姿态估计 → 3D 三角化 → GPT-4o 自动标注 270K 帧 → SFT 两个 VLM (76B InternVL2+Llama3 / 8B MiniCPM)

## 与 EthoInsight 的关系

**上下游互补，非竞争**。EthoInsight 处理的是 EthoVision XT 导出的量化轨迹数据（CSV），MouseGPT 处理的是原始视频 → 行为描述。MouseGPT 完全不涉及统计检验、APA 报告、范式专业知识。两者可以形成 pipeline：视频 → MouseGPT 行为描述 → EthoInsight 统计分析。

## 值得借鉴的设计（按优先级）

### P0: 行为描述的结构化层次输出

**论文做法**: 每帧输出 5 层 JSON — Overall → Head → Limb → Torso → Keywords。层次化描述让下游分析既能看到全貌又能下钻到身体部位级别。

**EthoInsight 应用**: 丰富 `catalog/<paradigm>.yaml` 和 report-writer 输出。

具体动作:
1. 在 `ethoinsight/catalog/<paradigm>.yaml` 中增加可选的 `description_schema` 字段，定义该范式需要产出的行为语义维度
2. report-writer 的报告模板增加 "行为语义总结" 段，对 code-executor 产出的数值指标补充自然语言解读
3. 不改变现有 metric_plan.json 主流程，description_schema 作为 opt-in 增强

### P1: Two-stage LLM-guided clustering for data-analyst

**论文做法**: Stage I — UMAP 降维 + 层次聚类形成基类簇；Stage II — LLM 基于关键词评估是否合并相似簇。解决了纯数值聚类缺乏语义可解释性的问题。

**EthoInsight 应用**: 增强 data-analyst subagent 的洞察能力。

具体动作:
1. data-analyst 新增 capability: 多组实验的行为模式对比聚类（当前只做统计方法审核）
2. 输入: 各组的 ethoinsight 指标值 + catalog 中的行为关键词
3. 输出: 聚类的行为模式分组 + LLM 语义标签
4. 优先级: v0.1 后 (当前 data-analyst 的统计审核能力更基础)

### P1: embedding-based 行为语义搜索

**论文做法**: 行为描述 → text-embedding-3-large (3072d) → 余弦相似度搜索。用户输入自然语言查询 "mouse lying down"，返回语义最匹配的帧。

**EthoInsight 应用**: 跨实验的语义检索能力。

具体动作:
1. golden-case 的 expert_notes 和预期结论文本做 embedding 存储
2. 新分析任务的行为描述与历史 golden-case 做语义相似度匹配
3. 输出: "该 treatment 组的行为模式与 golden-case #003 (LSD 0.2mg/kg) 相似度 0.87"
4. 优先级: v0.1 后 (需要足够 golden-case 积累)

### P2: Python REPL 动态代码生成

**论文做法**: Agent toolbox 不够用时，Agent 动态生成 Python 代码在虚拟环境执行。

**EthoInsight 借鉴**: 当前用 metric_plan.json 的严格预规划模式更安全可控，但遇到研究员开放式追问时缺乏灵活性。

具体动作:
1. 在 code-executor 中增加一个 "adhoc" 模式，允许 lead 对计划外的追问生成单次 Python 脚本
2. 约束: adhoc 模式不能修改 workspace 状态文件，只能读已有数据和输出临时结果
3. 安全: sandbox 已有隔离，adhoc 脚本与 plan 脚本使用同一 sandbox 但不同输出路径
4. 优先级: v0.1 后 (当前预规划模式能满足 Phase 0 需求)

### P2: Novelty detection via Isolation Forest

**论文做法**: 在 embedding 空间用 Isolation Forest 检测异常行为帧，多轮迭代识别新行为类别。

**EthoInsight 借鉴**: 质量审核关卡可以加入异常检测维度。

具体动作:
1. Gate 2 (数据质量检查) 后增加 novelty flag: 当某组数据指标模式明显偏离 golden-case baseline 时标记
2. 不是自动下结论，而是提醒 data-analyst "该组数据存在异常模式，建议人工关注"
3. 优先级: v0.1 后

## 不应借鉴的部分

- **视频直接输入**: EthoInsight 定位是 "已有 EthoVision XT 数据后"的分析，加视频输入会彻底改变产品形态和部署复杂度
- **SFT 专用 VLM**: 32×A100 训练 + 76B 推理，v0.1 单机 ECS 完全不可行
- **8 相机采集硬件**: 学术研究级别设备，不是产品化方向
- **GPT-4o 自动标注链路**: 存在系统性偏差，EthoInsight 有 golden-case 专家标注作为更强的 ground truth

## 战略启示

论文验证了 "自然语言行为描述 → embedding → 语义分析" 在行为学领域优于纯 kinematic-only 方法。EthoInsight 三项资产暗合此趋势:
1. **golden-case 专家标注**: 未来可作为语义空间的锚点
2. **report-writer 自然语言输出**: 每条报告都可向量化，构成知识底座
3. **范式知识体系**: 已按 Overall→身体部位→统计的层次组织，天然适配层次化描述 schema
