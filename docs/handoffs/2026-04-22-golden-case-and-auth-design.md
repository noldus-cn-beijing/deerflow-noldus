# 2026-04-22 Golden-Case 基建 + 本地部署授权方案 — 交接文档

> **给下一个 AI Agent：** 你无法访问本次会话上下文。这份文档让你快速理解我们在 2026-04-22 做了什么，以及接下来怎么继续推进。
>
> **读取顺序**：本文档 → [CLAUDE.md](../../CLAUDE.md) → [roadmap.md](../roadmap.md) → 上一份交接 [2026-04-21-subtask-visibility-done.md](2026-04-21-subtask-visibility-done.md)

---

## 1. 会话概览

**用户**: Qiuyang（Noldus 软件开发工程师）

**本次会话主题**:
1. 回顾 4/21 handoff，确认当前状态和下一阶段工作方向
2. 回答用户四个具体问题：行为学协作边界、产品资料收集、M0.1 余项、fine-tune 启动准备
3. **落地 golden-case 基建**（SCHEMA + TEMPLATE + case-001 + 校验脚本）
4. 修复前端 tsconfig.json 的 `baseUrl` 废弃警告
5. **设计本地部署授权方案**（为未来商业化准备）

**会话成果**：全部工程落地 + 一份新设计文档，0 未决问题。

---

## 2. 本次会话完成的工作 ✅

### 2.1 Golden-Case 协作基建（核心产出）

为 v0.1 的 Layer B 判断质量验收做准备。行为学同事需要按此结构标注 16 个范式的 golden-case。

**新增文件**：

| 文件 | 用途 |
|---|---|
| [golden-cases/SCHEMA.md](../../golden-cases/SCHEMA.md) | 字段字典 + 填写指引，16 个范式枚举、6 种 finding type、severity 分级 |
| [golden-cases/TEMPLATE/metadata.yaml](../../golden-cases/TEMPLATE/metadata.yaml) | 空白模板 — case 身份信息 |
| [golden-cases/TEMPLATE/expected-analysis.yaml](../../golden-cases/TEMPLATE/expected-analysis.yaml) | 空白模板 — 专家期望的分析结论（★ 机器读，跑 assertion） |
| [golden-cases/TEMPLATE/notes.md](../../golden-cases/TEMPLATE/notes.md) | 空白模板 — 专家思维过程记录（半结构化） |
| [golden-cases/case-001-shoaling-baseline/](../../golden-cases/case-001-shoaling-baseline/) | 第一个参考样本，工程侧字段填好，行为学待补 TODO |
| [scripts/validate_golden_case.py](../../scripts/validate_golden_case.py) | Schema 校验脚本，`python3 scripts/validate_golden_case.py` |

**case-001 细节**:
- 5 个受试者 raw data 从 `demo-data/DemoData/斑马鱼鱼群行为/` 拷贝
- 数值字段从 fix5 E2E thread `6f046cc7-775a-4eb9-9027-2022e50781ca` 的 `handoff_code_executor.json` 精确提取，±5% 区间
- 6 处 `TODO(行为学同事)` 标记，等专家 review 补 reasoning/severity/文献
- 校验通过：0 errors, 0 warnings

**关键数据点**（供验证）:
- Subject 3 `mean_nnd = 70.02 mm`（离群个体）
- 排除 Subject 3 后 treatment 组均值 37.23 mm，与 control 37.97 mm 接近
- control n=2（样本量不足，Mann-Whitney U, p=0.8/1.0）

### 2.2 前端 tsconfig.json 修复

- **问题**：IDE 报 `baseUrl` 在 TypeScript 7.0 将废弃
- **修复**：删除 `baseUrl: "."` 字段，保留 `paths: {"@/*": ["./src/*"]}` — `paths` 已经隐含了 `baseUrl` 的语义
- **验证**：`pnpm check` 通过
- **不走 `ignoreDeprecations` 路线的原因**：IDE TS 版本要求 `"6.0"`，但项目 tsc 5.9.3 只接受 `"5.0"`，版本对齐困难

### 2.3 本地部署授权方案设计

新增 [docs/plans/2026-04-22-local-deploy-auth-and-access.md](../plans/2026-04-22-local-deploy-auth-and-access.md)。

**核心设计**:
- **授权层**：RSA 签名 License 文件 + Docker 宿主机指纹绑定 + 到期日检查
- **网络层**：实验室局域网，所有终端浏览器直接访问 `http://服务器IP:2026`
- **隔离层**：浏览器自报名（localStorage），Thread/Memory 按 user_id 隔离 — **不搞登录系统**

**实施阶段**:
- v0.1（9 月）：不做
- v0.1+ 首批客户部署：实装（~300 行代码）
- 商业化：扩展 Web 管理后台

**拒绝的方案**（文档 §5 有对比）:
- better-auth 登录 — 本地部署场景下体验差
- 硬件加密狗 — 和"新时代 AI 产品"定位冲突
- Keygen EE — 架构过重，客户数量有限时自建更轻

### 2.4 用户决策与问答（关键上下文）

用户明确决策:
- **行为学判断设计文档已存在**（[2026-04-21-behavioral-reasoning-design.md](../plans/2026-04-21-behavioral-reasoning-design.md) 1079 行），不需要重新 brainstorming
- **case-001 工程侧先做，再给同事 review** — 专家时间成本降一半
- **本地部署不做加密狗** — Web 多终端定位，用户体验优先
- **v0.1 不做授权** — 内部验证阶段，9 月硬指标是 16 范式 + 微调

---

## 3. 仓库当前状态

**分支**: `feature/etho-skills`（当前工作分支）

**测试基线**（上一轮 fix5 验证）:
- backend: 1660 passed / 14 skipped
- ethoinsight: 131 passed / 3 skipped
- frontend: `pnpm check` 全绿

**最近 tag**: `v0.1` (cd2d6aba, 2026-04-21) — shoaling E2E 闭环

**未追踪文件**（本次会话新增）:
- `golden-cases/`（整个目录）
- `scripts/validate_golden_case.py`
- `docs/handoffs/2026-04-22-golden-case-and-auth-design.md`（本文档）
- `docs/plans/2026-04-22-local-deploy-auth-and-access.md`

**已追踪但有改动**:
- `packages/agent/frontend/tsconfig.json`（去掉 `baseUrl`）

---

## 4. 下一阶段优先级（按依赖顺序）

### P0 — 本周就能启动的（不阻塞外部）

1. **发邮件给行为学同事**（启动 golden-case 标注流程）
   - 附件：`golden-cases/SCHEMA.md` + `golden-cases/TEMPLATE/` + `golden-cases/case-001-shoaling-baseline/`
   - 请求：review case-001 结构合不合理、补齐 TODO 字段
   - 消息模板在 [finetuning-strategy-update.md §3.1 第 154-177 行](../plans/2026-04-21-finetuning-strategy-update.md)

2. **发邮件给产品/技术支持团队**（启动资料收集）
   - 需要：EthoVision XT Reference Manual + 16 个范式 demo 导出 + 技术支持 FAQ + Application Notes
   - 详见 [fine-tuning-data-checklist.md A 类](../plans/2026-04-15-fine-tuning-data-checklist.md)
   - 这是 Phase 1 微调的前置，越早启动越好（roadmap 已识别为头号风险）

3. **M0.1 余项**（3-4 天填充工作）
   - 不支持范式的降级路径 E2E（lead prompt 加降级分支 + 测试，~1-2 天）
   - `read_file` UTF-16 fallback（sandbox/tools.py 加编码 fallback chain，~0.5 天）
   - 429 重试策略优化（`llm_error_handling_middleware.py`，~1 天）

### P1 — 依赖外部反馈的

4. **行为学同事返回 case-001 review** → 定型 schema → 启动批量标注（5-8 月分三批）
5. **产品资料到位** → 开始写 Phase 1 数据采集脚本（`scripts/generate_stats_qa.py` 等）

### P2 — 后续（不在当前会话范围）

6. **Phase 0 M0.2-M0.4**：16 范式批量补全（参考 [behavioral-reasoning-design.md §7](../plans/2026-04-21-behavioral-reasoning-design.md)）
   - 先抽 `templates/_base.py` 基类
   - 第一批：EPM / 旷场 / FST / MWM / shoaling（已完成）
   - 每范式边际成本 3-5 天
7. **v0.1+**：实施本地部署授权方案（~300 行代码，见 [2026-04-22-local-deploy-auth-and-access.md](../plans/2026-04-22-local-deploy-auth-and-access.md)）

---

## 5. 关键上下文（下一个 Agent 必读）

### 5.1 项目定位

- **EthoInsight** — 面向行为学研究员的 AI 分析助手
- **v0.1 硬指标**：2026 年 9 月，16 个范式端到端分析 + 微调 Qwen3-8B 上线
- **当前状态**：shoaling 范式 E2E 闭环，框架通用能力验证通过；15 个范式待补
- **完整架构**：[CLAUDE.md](../../CLAUDE.md) + [roadmap.md](../roadmap.md)

### 5.2 关键设计文档（按阅读优先级）

1. [behavioral-reasoning-design.md](../plans/2026-04-21-behavioral-reasoning-design.md) — **最重要**，行为学判断能力的完整设计（1079 行，L1-L4 分层、Layer A/B 模型、quality-reviewer、异常诊断、验收体系）
2. [finetuning-strategy-update.md](../plans/2026-04-21-finetuning-strategy-update.md) — 微调执行计划（分步、蒸馏、DPO 推迟到 v0.1 后）
3. [fine-tuning-data-checklist.md](../plans/2026-04-15-fine-tuning-data-checklist.md) — 数据采集清单（A-H 八类来源）
4. [2026-04-22-local-deploy-auth-and-access.md](../plans/2026-04-22-local-deploy-auth-and-access.md) — 本次新增，部署授权方案

### 5.3 用户工作风格偏好（重要！）

- **实用工程化优先**，不做过度设计
- **正面提示**（GLM 原则）：不用"禁止 X""不要 X"，会反向激活
- **中文 commit message**，简洁描述改动意图
- **TDD 强制**：每个新功能/bug 修复都带单测
- **不主动 push**：本次会话未 push，继续保持

### 5.4 golden-case 协作的核心思路

**不让行为学同事从零写 YAML**，而是：

1. 工程师先填数值字段（从 handoff JSON 精确提取）
2. 工程师填 `TODO(行为学同事)` 标记需要专家判断的位置
3. 同事基于"已填一半的范本"补齐，专家时间成本降一半

这个流程对 case-001 验证有效，后续 15 个范式照此流程推进。

---

## 6. 可能误导后续 Agent 的点

1. **golden-case 当前是工程侧初稿，不是最终版** — case-001 里的 reasoning 字段很多是 TODO，等同事 review 才能定型。不要把它当成"标准答案"用于任何 assertion
2. **本地部署授权方案 v0.1 不实施** — 不要误以为需要现在写代码。只是设计文档归档
3. **上一个 handoff 提到的"Extracted Context 英文 dump"问题已修复** — fix5 的 C4/C5 通过正面 prompt 引导解决，不要回到 heading 白名单方案
4. **DeerFlow 受保护文件不要乱改** — `lead_agent/prompt.py`、`subagents/builtins/__init__.py` 等有 Noldus 定制，动前先看 [packages/agent/backend/CLAUDE.md](../../packages/agent/backend/CLAUDE.md)
5. **`demo-data/DemoData/` 下是 16 个范式的演示数据**，但只有 `斑马鱼鱼群行为` 目录在 `shoaling.py` 支持。其他范式的 parse 层可能需要扩展

---

## 7. 推荐的接手第一步

**情况 A：用户继续推进 golden-case 标注**
1. 确认 case-001 是否已发给行为学同事（问用户）
2. 如果同事返回了 review，合入修改并跑 `python3 scripts/validate_golden_case.py`
3. 开始 case-002（EPM）模板化，参考 case-001 流程

**情况 B：用户想做 M0.1 余项**
1. 查 [sandbox/tools.py](../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) 看 `read_file` 实现
2. 查 [llm_error_handling_middleware.py](../../packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py) 看 429 重试
3. 查 [lead_agent/prompt.py](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 加降级分支
4. TDD 流程：先写失败测试，再实现

**情况 C：用户想启动 Phase 1 微调数据采集**
1. 读 [finetuning-strategy-update.md §3.2 Step 2](../plans/2026-04-21-finetuning-strategy-update.md) — 工程线
2. 写 `scripts/generate_stats_qa.py`（解析 `ethoinsight/statistics.py` 决策树 → QA）
3. 写 `scripts/generate_skill_qa.py`（解析 skill 文档 → QA）
4. 所有训练数据必须带 `<think>` CoT traces（[§2.3](../plans/2026-04-21-finetuning-strategy-update.md)）

**情况 D：用户问"接下来做什么"**
→ 按 §4 优先级列表给建议，不要替用户决定，让用户选方向

---

## 8. 未解决 / 需用户决策的事项

全部已消化，无阻塞性未决问题。以下是**用户需要在外部行动的事**（不是 Agent 能代劳的）：

1. ✉️ 发邮件给行为学同事（case-001 协作启动）
2. ✉️ 发邮件给产品/技术支持团队（资料收集）
3. 👥 决定是否拉 1-2 位同事分担 golden-case 标注（80-100 小时总工作量）

---

## 9. 测试与验证命令速查

```bash
# Golden-case 校验
cd /home/qiuyangwang/noldus-insight
python3 scripts/validate_golden_case.py
python3 scripts/validate_golden_case.py --strict  # warnings 也当 error

# 前端类型检查
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check

# 后端测试
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test

# ethoinsight 库测试
cd /home/qiuyangwang/noldus-insight/packages/ethoinsight
pytest tests/
```

---

## 10. 签名

**会话时间**: 2026-04-22
**模型**: Claude Opus 4.7 (1M context)
**会话时长**: 中等（约 15-20 轮）
**交付物完整性**: ✅ 所有承诺的工作已落地
**需要用户行动**: 发 2 封邮件（见 §8）

下一个 Agent 接手时，如果用户在新会话里说"继续上次的工作"，先读本文档 §4（下一阶段优先级），然后按 §7（接手第一步）的对应情况推进。
