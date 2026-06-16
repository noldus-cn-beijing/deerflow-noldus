# Handoff:行为学范式书 PDF → skill 重构 + reward 判据 + MinerU 共享安装

**日期**:2026-06-16
**会话主题**:把《动物行为实验指南》(680页 PDF)转化为 EthoInsight 可用资产——服务两个目标:① 当前 agent skill(已落地);② 未来 agentic RL reward 判据(已结构化)
**commit**:`2a84c762`(dev 分支,6 个 skill 文件)
**状态**:MinerU 安装完成 + 书提取完成 + 6 范式 skill 重构已 commit;reward 判据 YAML + 原料切分在项目外 `~/behavioral-book/`

---

## 一、本会话完成的工作(四块)

### 1. MinerU 服务器共享安装(`/home/shared/mineru/`)
- **工具**:MinerU 3.3.1(开源 Apache 2.0,pipeline backend,GPU 加速)
- **隔离方案**:miniforge + conda 环境,数据装 `/home/shared/mineru/`,软链 `/opt/mineru`
- **共享权限**:`gpu` 组(wangqiuyang/yuxiaofei/quruoheng/ningning/yangpenghan)setgid 2775
- **权重**:`OpenDataLab/PDF-Extract-Kit-1.0`(2.4G,从 ModelScope 下,放共享位置)
- **关键坑**:torch 默认装太新(2.12.0)超过 12.8 驱动 → 重装 `torch 2.11.0+cu128` 解决
- **性能**:RTX 5090,680 页书 25 秒处理完(26 it/s)
- **未做**:全局 wrapper(`/usr/local/bin/mineru` 软链)——用户决定不做共享 wrapper,自己用即可。如后续要给其他用户用,补第 6 步 wrapper + `MINERU_TOOLS_CONFIG_JSON` 共享配置

### 2. 书的提取与切分(`~/behavioral-book/`,项目外公共资产)
- **raw/**:MinerU 全量产出(book_full.md 1.5MB/14416行 + 674图 + content_list.json)
- **by-paradigm/**:125 个范式按 17 大类切分,两层结构
  - 切分用**书目录页的 125 范式清单作 ground truth**,正文严格匹配 `## N.名称` 标题,0 未匹配
  - 脚本:`split_by_paradigm.py`(可重跑)
- **质量**:中文干净、公式转 LaTeX、表格结构化、章节层级正确
- **分类有小瑕疵**(位置法把部分边界划偏),但**内容切分准确**,不影响使用

### 3. ⭐ 6 个 v0.1 范式 skill 重构(已 commit `2a84c762`)
**repo 改动**(6 个文件,从 ~1.1KB 占位 → ~5.5KB 实质):
- `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/`
- 范围:EPM / OFT / FST / TST / LDB / Zero Maze

**统一结构**:范式定位 / 模型概述 / 设计原理 / 指标与判读 / 脱险点 / 范式区分

**遵循的规范**:
- skill-creator 官方规范(渐进披露、解释 why 不堆 MUST、references 按需加载)
- 避坑:不描述 `identify_ev19_template` 工具输出格式(防 LLM 脑补冒充,memory 教训)
- SSOT:指标公式指向 `catalog/*.yaml`,skill 不重复
- 第9条铁律:组间比较口径,不用绝对阈值/绝对程度词
- **无厂商名**(上海欣软/VisuTrack/XR- 全删)
- 保留项目特有:FST/TST 的 `pendulum-params.md` 指针、Zero Maze"犹疑次数"指标

### 4. ⭐ 6 个 reward 判据 YAML(项目外 `~/behavioral-book/reward-criteria/`,给未来 RL)
- `epm.yaml` / `open_field.yaml` / `forced_swim.yaml` / `tail_suspension.yaml` / `light_dark_box.yaml` / `zero_maze.yaml`
- 结构:`design_criteria`(hard/soft) / `metric_completeness`(required/recommended) / `interpretation_direction` / `confound_checks` / `forbidden`
- **设计**:skill(叙述层,给 agent)与 reward YAML(规则层,给 RL)物理分离,避免互相干扰

---

## 二、关键决策记录(供后续 agent 理解)

1. **用户最终目标**:让微调后的 Qwen3-30B-A3B 模型内化行为学决策能力,而非靠 skill 硬收敛。RL 路线 = LoRA + GRPO + verl,真实 EthoVision 数据做 rollout,RTX 5090 32GB
2. **书做高起点原料,owner 豁免"同事守门"**:用户重新评估后决定由我们直接写进 skill(原 SSOT 第8条"同事守门"的豁免)
3. **第8条不违反**:书的方法论(为什么这么设计/指标原理)≠ 专家判读结论,前者可进 by-experiment skill,后者仍走 golden-cases
4. **skill 与 reward 判据物理分离**:agent 读叙述理解 why;RL 需机器可判读规则;两消费者需求不同
5. **review 包早期格式(指标对照表)被弃用**:用户指出真正价值是"模型概述/设计第一性原理/指标洞察",不是 catalog 对照。重构改为洞察导向六章节

---

## 三、下一步(未做,按优先级)

### 待验证(优先)
- [ ] **skill 改动跑 dogfood**:6 个 skill 从占位变详细,需确认没引发脑补冒充工具调用等回归(memory 教训:skill 详细 → LLM 脑补)。建议跑 EPM/OFT dogfood 看行为
- [ ] 行为学同事审核 6 个重构后的 skill(尤其绝对阈值→组间比较的口径转换是否到位)

### RL 路线(中期)
- [ ] reward YAML → verl reward function 实现(等 RL pipeline 启动)
- [ ] 基座最终确认:Qwen3-30B-A3B-Instruct MoE(CLAUDE.md 第5条有 8B Dense vs 30B MoE 待对齐)
- [ ] 真实 EthoVision 数据做 GRPO rollout 的数据量评估(关联 Issue #90 golden-case 阻塞)

### 扩展(低优先级)
- [ ] 其他 119 个范式的 skill 重构(v0.1 只做 6 个)
- [ ] MinerU 全局 wrapper(若其他 gpu 用户要用)
- [ ] `/home/shared/mineru/book-full/`(182MB 原始产出)的清理——切分验证无误后可删

---

## 四、重要资产位置

| 资产 | 位置 |
|---|---|
| MinerU 安装 | `/home/shared/mineru/`(软链 `/opt/mineru`) |
| 书提取原料 | `~/behavioral-book/raw/` |
| 125 范式切分 | `~/behavioral-book/by-paradigm/` |
| reward 判据 YAML | `~/behavioral-book/reward-criteria/` |
| 重构规范 | `~/behavioral-book/SKILL-REFACTOR-SPEC.md` |
| 切分脚本 | `~/behavioral-book/split_by_paradigm.py` |
| 重构后的 skill(repo) | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/{epm,open_field,forced_swim,tail_suspension,light_dark_box,zero_maze}.md` |

---

## 五、踩坑备忘(给后续 agent)

1. **torch 版本 vs 驱动**:MinerU 默认装 torch 2.12(CUDA runtime 太新),RTX 5090 驱动 12.8 不够 → 必须装 `torch+cu128`。诊断:`python -c "import torch; print(torch.cuda.is_available())"`
2. **`| tail` 吞实时输出**:mineru 长任务用 `| tail -50` 看不到进度(缓冲),改用 `| tee log`
3. **首次跑卡在 Fetching files**:不是卡死,是补下 onnx 权重,耐心等
4. **切分用目录页作 ground truth**:正文标题格式不统一(大类标题 MinerU 提取不全),靠正则猜不可靠;书目录页(L35-175)的 125 范式清单最准
5. **MinerU 包名**:新版 3.3.1 是 `mineru`(不是旧版 `magic-pdf`),下载命令 `mineru-models-download`

## milestone 建议

本次会话让"行为学知识建设"这个隐性 track 到达了一个 checkpoint:
- v0.1 六范式 skill 从空壳变为实质内容(此前是 CLAUDE.md 提到的"等行为学同事补"的阻塞项之一,现由书提取部分化解)
- 为 agentic RL 建立了 reward 判据的结构化原料

**建议**:在 `docs/milestone/` 下记录一个"behavioral-knowledge-from-book"track,状态 = skill 重构已合 dev(commit 2a84c762),下一步 = dogfood 验证 + 同事审核 + reward YAML 转 verl function。但这不解除 CLAUDE.md 第13条的两个真实阻塞(结构聚合 #98 + golden-case #90)——那些仍等行为学专家方法论。本次只是用书这部分化解了 skill 内容空缺。
