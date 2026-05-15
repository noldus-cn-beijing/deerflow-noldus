# EV19 模板 + 实验知识 review 包

**给行为学同事**：这一份是 EthoInsight Agent 后续要用的 EV19 模板与实验知识库的草稿。我们先把机器能解析的字段（每个 EV 模板的 arena 名、zone 配置、subject 类型）自动填好了，剩下的领域知识需要你们补充。

---

## 1. 这个 review 包是什么

EthoInsight Agent 在帮研究员分析数据之前，需要先确认两件事：

1. **用户在 EthoVision XT 19 里用的是哪个模板**（决定数据结构）
2. **用户做的是什么实验**（决定怎么解读和写报告）

这两件事是 N:M 对应——一个模板可以做多种实验，一个实验可以用多种模板。比如：

- `OpenFieldCircle-NoZones-Fish` 既能做 shoaling（多鱼测群体度），也能做 aquatic open field（单鱼测探索）
- shoaling 既能用 `OpenFieldCircle-NoZones-Fish`（2D），也能用 `AquariumTrack3D`（3D），还能用 `DanioVision DVOC 004x-96w-circ`（96 孔板）

所以我们准备了两份索引：

- **`by-template/`** — 每个 EV 模板大类一个文件（共 20 个），从模板视角写："这个模板能做什么实验？"
- **`by-experiment/`** — 每个实验类型一个文件（共 20 个），从实验视角写："这个实验该用哪些模板？"

---

## 2. 你们要填什么

### 优先级最高的几个文件（先填这些）

```
by-experiment/shoaling.md          ← 我们已经做完一次 shoaling 端到端测试，最先要 ground truth
by-experiment/epm.md               ← 路线图 v0.1 必须完成
by-experiment/open_field.md        ← 路线图 v0.1 必须完成
by-template/OpenFieldRectangle.md  ← 矩形旷场（矩形 OFT 的核心）
by-template/OpenFieldCircle.md     ← 圆形旷场 + 鱼类相关
by-template/PlusMaze.md            ← EPM 模板
by-template/AquariumTrack3D.md     ← shoaling 的 3D 版
```

把这 7 个填完，agent 就可以跑 shoaling / EPM / OFT 三大范式的端到端流程了。

### 后续填（可以异步）

`by-experiment/` 剩下 17 个；`by-template/` 剩下 17 个。

---

## 3. 文件里的 🟢 和 🟡 是什么意思

每个文件里：

- **🟢 标记的字段** — 已经从 EV19 demodata 的 `templateMetaData.xml` 自动抽取出来。**不要修改**。如果你觉得这些字段错了，那是脚本 bug，告诉技术同事改脚本，不要改 markdown。
- **🟡 标记的字段** — 需要你们补充的领域知识。这些是 markdown 注释（`<!-- ... -->`）形式的提示，写完后**保留 🟡 标题，删除注释**即可。

举例：在 `by-template/OpenFieldCircle.md` 的 `OpenFieldCircle-NoZones-Fish` 变体下面：

```markdown
### 🟢 EV19 机器字段（自动抽取，请勿修改）   ← 不要动这一节

- **arena_template**：`Open field, round`
- **zone_template**：`No zone template`
- ...

### 🟡 推荐的实验场景（待补充）              ← 需要你们填
<!-- 例：「啮齿动物焦虑测试...」 -->         ← 这是提示，参考用，写完删掉

斑马鱼群体行为研究的首选 2D 模板...           ← 你写的内容
```

---

## 4. 怎么填——具体规则

### 4.1 by-template/<大类>.md 怎么填

每个文件从两层填：

**大类层（文件顶部）**：

- **🟡 这个大类用来做什么？** — 一两句话说清楚 EV 软件里这个模板大类的定位
- **🟡 何时不该选这个大类？** — 这个大类的边界，避免用户选错（比如"鱼不要选 OpenFieldRectangle"）
- **🟡 关键参考文献** — 1-3 篇代表性方法学论文

**变体层（每个变体一节）**：

- **🟡 这个变体相对其他变体的核心差异** — 为什么有 AllZones / NoZones / NovObjZones 这些差异，本质区别是什么
- **🟡 推荐的实验场景** — 什么情况下选这个变体（不是大类，是这个具体变体）
- **🟡 不该用这个变体的场景** — 关键边界
- **🟡 对应学术范式** — 这个变体最常对应的实验类型；如果一对多就写多个并说明区分依据

### 4.2 by-experiment/<实验>.md 怎么填

- **🟡 一句话定义** — 行为学定义。一句话。
- **🟡 适用模板（按推荐顺序 + 取舍说明）** — **核心字段**。我们脚本根据目录名猜了几个候选，你们必须 review、删除错的、补充缺的、说清楚每个的取舍
- **🟡 必须计算的指标** — 这个实验报告里必须有哪几个量化指标
- **🟡 常见脱险点 / 数据质量风险** — 数据多脏 / 样本量多少不能下结论 / 哪些容易被误读
- **🟡 报告解读语言** — 写报告时要用什么术语，避免什么误读
- **🟡 关键参考文献** — 1-3 篇
- **🟡 与其他实验的区分** — 如果有相近实验（如 shoaling vs aquatic_open_field），怎么界定

---

## 5. 不要做的事

- ❌ **不要修改 🟢 字段**（机器自动生成的 arena_template / zone_template 等）
- ❌ **不要把 by-template 和 by-experiment 的内容混着写**（前者按模板组织，后者按实验组织；同一条知识可能两边都引用，但要按对应的视角写）
- ❌ **不要写一大段教科书式的总论**——agent 是按文件加载这些知识的，写得越长 token 越贵；理想长度：每个 🟡 段落 1-3 句话，必要时配 bullet list
- ❌ **不要追求"完美"**：先把 7 个高优先级文件填到能用就行；后续我们做 E2E 测试发现哪里不准，再回来改
- ❌ **不要列举过多文献**：1-3 篇有代表性的就够了；agent 现在不会自动去读这些文献全文，文献只是作为人类 review 时的参考

---

## 6. 跨文件一致性

如果 `by-experiment/shoaling.md` 里推荐了 `OpenFieldCircle-NoZones-Fish`，那么 `by-template/OpenFieldCircle.md` 里 `OpenFieldCircle-NoZones-Fish` 变体的"对应学术范式"也应该列上 `shoaling`。

**Self-check 简单做法**：

- 填 `by-experiment/X.md` → 同步看一眼涉及的几个 `by-template/Y.md` 里那个变体节
- 不需要追求两边一字不差，但**关键判断要对齐**（推荐 vs 不推荐、能做 vs 不能做）

---

## 7. 完成后告诉技术同事

填完后告诉技术同事，他们会：

1. 把这些 markdown 移进 `packages/agent/skills/custom/ethovision-paradigm-knowledge/` 作为 agent 的渐进披露 skill
2. 写一个 CI 测试：每个 `by-template/` 提到的实验在 `by-experiment/` 里都有文件；每个 `by-experiment/` 推荐的模板都在 `_facts.json` 里存在
3. 改造 Gate 1 流程让 agent 实际用这些知识

---

## 8. 文件清单与状态跟踪

可以在每个文件填完后在文件顶部加一行 `**status**: reviewed by <你的名字> on YYYY-MM-DD`，方便追踪。

| 文件 | 优先级 | 状态 |
|---|---|---|
| `by-experiment/shoaling.md` | P0 | ⬜ 未填 |
| `by-experiment/epm.md` | P0 | ⬜ 未填 |
| `by-experiment/open_field.md` | P0 | ⬜ 未填 |
| `by-template/OpenFieldCircle.md` | P0 | ⬜ 未填 |
| `by-template/OpenFieldRectangle.md` | P0 | ⬜ 未填 |
| `by-template/PlusMaze.md` | P0 | ⬜ 未填 |
| `by-template/AquariumTrack3D.md` | P0 | ⬜ 未填 |
| 其他 by-experiment（17 个） | P1 | ⬜ |
| 其他 by-template（17 个） | P1 | ⬜ |

---

## 9. 有疑问？

- 关于"这个模板到底能不能做 X 实验" — 在文件里直接写"待技术同事和我一起跑一次端到端确认"
- 关于"我没听过的模板" — 跳过即可，技术同事在 demodata 里也只能看到这些模板名
- 关于"我们没用 EV 的研究怎么办" — 这一版只覆盖 EV19；EV15/16 / DeepLabCut / Bonsai 不在这次范围

---

## 10. 起草来源说明（供技术同事追溯）

- 模板大类与变体清单：自动扫描 `demodata/ev19 templates/`（共 20 大类、62 变体）
- 机器字段：解析每个目录的 `templateMetaData.xml`
- 实验清单：基于 `packages/ethoinsight/templates/` 现有模板 + CLAUDE.md 路线图 v0.1 必须范式 + EV demodata 大类反推
- 起草脚本：`scripts/build_ev19_template_index.py`、`scripts/build_ev19_experiment_drafts.py`
- 重新生成：在仓库根目录跑这两个脚本会**重置**所有 markdown 草稿；如果同事已经填了内容，请确认备份后再跑
