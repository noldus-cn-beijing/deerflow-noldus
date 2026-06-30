# drift-signatures（doc-sync Step 5 参考）

Step 5 检测"文档里写的" vs "handoff/git 暴露的最新真相"之间的漂移。本文件按文档区域
列出**已知的漂移模式**，作为种子——首次运行就能命中真问题，后续随项目演进扩充。

漂移三类：**过期**(文档写了已被推翻的)、**缺失**(事实有文档没落地的)、**矛盾**(文档内自相冲突)。

## CLAUDE.md 已知漂移模式

### 1. skill 列表自相矛盾（矛盾型）
- **现象**：CLAUDE.md"重要注意事项"第 1 条既说"**不在此处手工枚举以免漂移**"(权威源是 `extensions_config.json`)，**又紧跟着手工枚举**了 9 个 skill 名。
- **修法**(semantic)：删掉手工枚举，只保留"权威清单以 extensions_config.json 为准"这句 SSOT 声明。新增 skill 不要往 CLAUDE.md 加。
- **判据**：任何"声明不枚举却枚举了"或"枚举与 SSOT 源不一致"都是这类。

### 2. 微调方案新旧并存（矛盾/过期型）
- **现象**：第 5 条"微调方案已锁定 Qwen3-8B Dense + Fireworks"，同条又说"2026-05-13 提出升级改 Qwen3-30B-A3B + RTX 5090…待团队对齐前仍按原锁定方案执行"。两套方案并存，"当前到底用哪个"模糊。
- **修法**(semantic)：先 git/gh 查 `docs/plans/2026-05-13-base-model-decision-memo.md` 或后续 handoff 有无"团队已对齐"的坐实。若有 → 改写为单一锁定方案；若仍"待对齐" → 保留双轨但**明确标注"截至 YYYY-MM-DD 仍未对齐，执行态=原锁定"**，不要让读者猜。
- **判据**：任何"锁定 X"+"提议改 Y"+"待对齐"三者并存的段落。

### 3. issue 状态滞后（过期型）
- **现象**：CLAUDE.md 某处写"Issue #N 阻塞"，但 #N 已 CLOSED(git/handoff 坐实)。例：#98 结构聚合——CLAUDE.md 顶部已写"#98 CLOSED"，但若别处仍按 OPEN 描述就是漂移。
- **修法**(mechanical/semantic)：`gh issue view <N> --json state` 核实 → 全文搜索该 issue 号 → 统一改为坐实状态。
- **判据**：同一 issue 号在文档不同位置状态不一致，或文档状态滞后于 gh。

### 4. 日期/"当前"描述陈旧（过期型）
- **现象**：含"当前阶段"、"今年 X 月"、具体年份的描述性段落，随时间过期。
- **修法**(semantic)：对照最新 handoff 的日期，更新"当前"指向。

## milestone/README.md 已知漂移模式

### 5. 活跃表 track 完成未挪表（过期型 —— 最常见堆叠源）
- **现象**：活跃表某 track 状态已是"✅ 已合 dev"或 handoff/git 显示 done，但**还挂在活跃表**没挪到已完成表。
- **修法**(semantic，强制挪表)：从活跃表删行 + 已完成表加行 + 填完成日期。
- **判据**：活跃表行 = "状态列已 done" 或 "最新 handoff 指针显示该 track 收口"。

### 6. "最新 handoff"指针陈旧（过期型）
- **现象**：某 track 的"最新 handoff"列链接停在旧日期(如 5/28)，但 6 月有更新的相关 handoff。
- **修法**(mechanical)：换链接到最新相关 handoff。
- **判据**：指针日期 < 该 track 最近一次实际推进的 handoff 日期。

### 7. 状态 emoji 与坐实不符（过期型）
- **现象**：📋 立项但实际已合 dev；🔴 阻塞但阻塞已解除。
- **修法**(mechanical)：git/gh 坐实后切 emoji。

### 8. 阻塞描述滞后（过期型）
- **现象**：前置 blockquote 写"唯一硬阻塞 Golden Cases"，结构聚合 #98 已 CLOSED 解除——确认描述与最新一致；若又出现新硬阻塞而 blockquote 没更新就是漂移。

## milestone/<track>.md 已知漂移模式

### 9. "当前状态"段滞后（过期型）
- **现象**：track milestone 文件"当前状态"段写"遗留项：X"，但 X 已在后续 handoff 做掉。
- **修法**(semantic)：git 核实 X 是否 done → 更新"完成项/遗留项/下一 milestone"。

### 10. dev HEAD hash 过期（过期型）
- **现象**：milestone 记的 dev HEAD 是旧 commit。
- **修法**(mechanical)：`git log dev` 取最新相关 commit。

## docs/adr/ 已知漂移模式

### 11. ADR 状态未更新（过期型）
- **现象**：ADR 标"proposed"，但对应决策早已落地执行(代码/handoff 坐实)。
- **修法**(mechanical)：改为 accepted + 落地日期。
- **注**：ADR **不轻易新建**(配额≤1/次)，除非是"已锁定 + 反直觉 + 防重提"的架构决策。普通进度走 milestone/CLAUDE.md，不配 ADR。

## 检测顺序建议

1. 先扫**矛盾型**(1、2)——这些是文档自己和自己打架，最该修，且不依赖外部信号。
2. 再用事实清单 + git/gh 校准扫**过期型**(3、5、6、7、8、9、10、11)。
3. 最后扫**缺失型**——事实清单里有、文档没落地的重大事实(如新 track 立项、新决策锁定)，这些走"新建判定门"或塞进既有段落。

## 扩充本文件

运行中遇到新的漂移模式(不在上表)，**先修了，再把模式补进本文件对应区域**。这是本 skill
的领域知识积累点——drift-signatures 越全，下次检测越准。
