# classification-rules（doc-sync Step 6 参考）

Step 6 把每条漂移/事实归入三类(mechanical / semantic / new)，决定待遇。本文件给判定
细则、边界 case、和"新建判定门"。

## 判定主表

| 类别 | 判定要害 | 待遇 |
|------|---------|------|
| **mechanical** | 改动是**值替换**：换指针/状态/哈希/emoji，不动语义结构。无需在两个意思间权衡。 | 直接写，报告列出，不高亮 |
| **semantic** | 改动需要**语义判断**：改写过期段落、择新舍旧、整段收敛、挪表(完成归档)。 | 写，报告**高亮**+给理由 |
| **new** | 改动是**新建文件**：起 milestone / 起 ADR。 | 写，但必须过"新建判定门"+受配额 |

## mechanical vs semantic 的边界 case

- **状态 emoji 切换**(📋→✅)：若 git/gh 已坐实 done → **mechanical**(只是换符号)。
- **issue OPEN→CLOSED**：gh 坐实 → **mechanical**(全文搜该 issue 号统一替换)。
- **完成挪表**(活跃表删行 + 已完成表加行 + 填日期)：需要判断"确实 done 了"+重组表格 → **semantic**(虽是机械动作，但涉及判断 done 与否，高亮)。
- **过期段落改写**(如微调方案段落)：需要在两个意思间择新 → **semantic**。
- **handoff 指针换链接**：纯值替换 → **mechanical**。
- **删掉与 SSOT 冲突的手工枚举**(skill 列表)：删+改声明 → **semantic**(收敛，高亮)。

判定口诀：**只换值不换意 = mechanical；换了意思或删了内容 = semantic。**

## semantic vs new 的边界 case

- **某 track 到达 checkpoint，但已有对应 milestone 文件** → **semantic**(更新该 milestone 的"当前状态"段)，**不新建**。
- **某 track 到达 checkpoint，且无对应 milestone 文件** → **new**(起新 milestone)，过判定门。
- **架构决策锁定**：若已有 ADR 覆盖 → **semantic** 更新；若无 ADR 且满足"反直觉/血泪教训/防重提" → **new** 起 ADR。
- **新 feature 立项(spec 入库但未实施)** → 一般 **semantic**(在 milestone/README 活跃表加一行 📋)，**不起新 milestone 文件**直到真有 checkpoint 内容。

判定口诀：**能塞进既有文件 = semantic；必须起新文件 = new。默认是 semantic，new 是需论证的例外。**

## 新建判定门（new 类必须全过）

新建一个 milestone 或 ADR 前，逐条过：

### 门 1 — 不是旧 track 延续？
该事实能否塞进现有 milestone / CLAUDE.md 段落？
- 能 → 走 semantic，**不新建**。
- 不能 → 给出"为什么这不是某个现有 track 的延续"的判断依据(写进报告)。

### 门 2 — (ADR 专属)是否锁定决策 + 反直觉 + 防重提？
ADR 只记**已锁定 + 有反直觉/血泪教训 + 防止后人重新提议被否方案**的决策。
- 普通进度更新(如"Sprint 2 完成")**不配 ADR**，走 milestone。
- 例：ADR 候选——"桌面化选 Electron 不选 Tauri"(反直觉：Tauri 更轻，但 SSR 依赖让 Tauri 需 3-4 周改造)、"sync 默认全量跟随上游"(防重提：别再建议"挑着合")。
- 看 `docs/adr/0001-seal-resume-not-forced-tool-choice.md` 作为 ADR 写法范例(状态/决策简述/为什么不用 X 含实证证伪/关键约束/已知未覆盖/参考)。

### 门 3 — (milestone 专属)是否到达 checkpoint？
milestone 是 feature track 的 **checkpoint 总结**，不是每个 PR 一个。
- 判据：完成 / 阶段切换(Sprint N done) / 阻塞解除 / 重大架构落地。
- 仅"立项(spec 入库未实施)"**不配**独立 milestone 文件——在 README 活跃表加行即可。

### 门 4 — 配额未超？
- 单次回流：新建 **ADR ≤ 1**、**milestone ≤ 2**。
- 超额 → 分多次回流，每次超额在报告写"为何本期要新建多个"。
- 配额存在的理由：**防止 skill 自己变成"无止境添加"源**——这是用户要消灭的病。

## 新建时的文件命名

- milestone：`docs/milestone/<track-kebab>.md`，结构抄 `docs/milestone/TEMPLATE.md`(状态/时间跨度/dev HEAD/做了什么/关键节点表/当前状态/相关 handoff)。
- ADR：`docs/adr/<NNNN>-<title-kebab>.md`，编号 = 现有最大编号 +1(现仅 0001，下一个是 0002)，结构抄 0001。

## 一个反模式：把回流变成 PR 级 milestone 工厂

最危险的失败：每个合 dev 的 PR 都起新 milestone。结果 milestone 数爆炸、活跃表只增不减。
**正确的尺度**：milestone 对应"一个 feature track 的阶段checkpoint"，可能横跨多个 PR。
多数 PR 合入只是机械更新某 milestone 的"关键节点表"加一行 + dev HEAD，**不起新文件**。
