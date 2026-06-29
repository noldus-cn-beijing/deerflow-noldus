# Spec: 修复 HTML 报告内联图全坏（嵌套 `<img>` 冲突）

> 状态：待实施（确定性 bug，根因已坐实，活体复现）
> 归属：#234 回归修复（2026-06-29）
> 优先级：高——#234 的核心目标（HTML 报告内联代表性图）当前**完全失效**，每份报告图全废。

## Context（活体证据）

dogfood thread `73b41dc3` 跑出 `report.html`（124KB，结构正确：6 个 `<h2>` + table，无 `<script>`），但**所有内联图都是坏的**：
```html
<img src="&lt;img src=" data:image png;base64,iVBOR...">
```
- 我的探针：`data:image/png;base64,` 匹配 **0**（因 `/` 被破坏成空格 `data:image png`）；前端渲染该 src 无法显示 → 报告里**一张图都看不到**。

## 根因（已坐实——#234 自身 prompt 与 substitution 契约打架）

两段代码在 `packages/agent/backend/.../subagents/builtins/report_writer.py` 和 `tools/builtins/seal_handoff_tools.py`：

1. **prompt 让 LLM 把占位符写进 `src` 属性内**（report_writer.py:77/88/141/149/151）：
   ```html
   <figure><img src="{{img:plot_box_open_arm.png}}"><figcaption>...</figcaption></figure>
   ```
2. **seal 的替换把 `{{img:X}}` 换成「整个 `<img>` 元素」**（seal_handoff_tools.py:458 `_resolve_html_report_image_placeholders._replace`）：
   ```python
   return f'<img src="data:image/png;base64,{b64}" alt="{basename}">'
   ```

→ `<img src="{{img:X}}">` 替换后变成**嵌套**：
```html
<img src="<img src="data:image/png;base64,...">">
```
→ 随后 `sanitize_report_html`（seal_handoff_tools.py:392，HTMLParser）解析这堆嵌套：外层 `src` 属性吃进 `<img src="`，把内层 `<` 转义成 `&lt;`，`data:image/png` 在未引号边界处被拆 → 产出 `<img src="&lt;img src=" data:image png;base64,...">`。

**两处对「占位符是元素还是属性值」的假设相反**，是 #234 内部矛盾。markdown 模式不犯此病（markdown 的 `![](...)` 占位符替换成路径，不是整 img 标签）。

## 方案（二选一，推荐 A）

### 方案 A（推荐）：substitution 只产 data-URI 值，不产整个 `<img>`
保持 prompt 现状（LLM 写 `<img src="{{img:X}}">`——这是最自然、最不易错的写法），把 `_replace`（line 458）改成**只返回 data URI 字符串**：
```python
return f"data:image/png;base64,{b64}"
```
则 `<img src="{{img:X}}">` → `<img src="data:image/png;base64,...">`，正确且无嵌套。
- 缺图 stub（line 446/456）当前返回 `[图表 'X' 未找到]` 这类文本——在 src 属性里会变成坏 src。**需配套**：缺图时让占位符所在 img 失效但不破坏 HTML（如返回空串使 `<img src="">`，或返回一个 data-URI 占位小图，或在 prompt 里要求 figure 包裹以便整体降级）。**实施时定**：最简稳妥=返回空字符串 + 日志（前端 img 加载失败显示 alt）。
- **副作用核查**：确认没有别处依赖 `_replace` 产整 `<img>`（grep 调用点；目前仅 HTML 路径用）。

### 方案 B（备选）：prompt 改成裸占位符，substitution 仍产整 `<img>`
prompt 改成让 LLM 写**裸** `{{img:X}}`（不包 `<img src>`），如 `<figure>{{img:X}}<figcaption>...</figcaption></figure>`。
- 缺点：LLM 习惯写 `<img src="...">`，裸占位符易被 LLM 自作主张包回 img（deepseek 正面提示也难 100% 约束）→ 脆弱。**不推荐**。

## 改动文件
- 方案 A：`seal_handoff_tools.py` `_resolve_html_report_image_placeholders._replace`（line 458 + 缺图 stub 分支 446/456）；report_writer.py prompt **不动**（或仅明确"占位符放在 src 属性里"以坐实契约）。
- 新增/补测：`test_seal_html_report.py`（见下）。

## TDD（关键——这正是 #234 漏测的）
- **新断言**：`<img src="{{img:X}}">` 经 `_resolve_html_report_image_placeholders` + `sanitize_report_html` 后 == `<img src="data:image/png;base64,...">`（合法、无嵌套、`/` 完整、`data:image/png;base64,` 子串存在）。← 这条若当初存在,#234 不会合入带病版本。
- 缺图：`<img src="{{img:missing}}">` → 降级为不破坏 HTML 的形态（空 src + alt 或占位图），且整篇 HTML 仍能被 HTMLParser 无错解析。
- 全链：mock chart_files + 真 png → seal → 读回 report.html，断言 `<img` 数 == 占位符数、无 `&lt;img`、无 `data:image png`（被拆的特征）。
- 消毒不回归：`<script>` 仍被剥、`data:` 图 URI 保留。

## 验收
- 重跑 dogfood（或对该 thread 重 seal report）→ `report.html` 内 `data:image/png;base64,` 子串数 == 内联图数、无 `&lt;img src=`、前端 ReportCard 内联图可见。
- 下载该 html 离线打开图仍在（base64 自包含——#234 原目标达成）。
- 后端 artifact/seal 测试全绿；裸导入两入口 0 退出。

## 教训（写入修复 commit）
确定性图文产物的 spec **必须有"产物可被目标渲染器正确解析"的端到端断言**，不能只断言"占位符被替换了"（#234 大概只测了后者）。守 `feedback_etho2_spec_misdiagnosed...` 同族：验收看真产物能不能用,不看中间步骤完成。
