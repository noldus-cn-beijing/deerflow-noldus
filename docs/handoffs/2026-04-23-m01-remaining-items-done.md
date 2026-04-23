# 2026-04-23 M0.1 余项闭环：降级路径 + UTF-16 Fallback — 交接文档

> **给下一个 AI Agent：** 你无法访问本次会话上下文。这份文档让你快速理解我们在 2026-04-23 做了什么，以及接下来怎么继续推进。
>
> **读取顺序**：本文档 → [CLAUDE.md](../../CLAUDE.md) → [roadmap.md](../roadmap.md) → 上一份交接 [2026-04-22-golden-case-and-auth-design.md](2026-04-22-golden-case-and-auth-design.md)

---

## 1. 会话概览

**用户**: Qiuyang（Noldus 软件开发工程师）

**本次会话主题**:
1. 阅读 4/22 交接文档，确认 M0.1 余项内容
2. **实现 `read_file` UTF-16 编码 fallback**
3. **编写降级路径 E2E 测试 + 修正 prompt 范式列表**
4. 用户确认 429 重试已在之前解决

**会话成果**：M0.1 余项全部闭环，0 未决问题。

---

## 2. 本次会话完成的工作 ✅

### 2.1 `read_file` UTF-16 Fallback

**问题**：EthoVision XT 导出的 .txt 文件是 UTF-16 LE 编码（带 BOM `\xff\xfe`），但 `LocalSandbox.read_file()` 硬编码 `encoding="utf-8"`，遇到 UTF-16 文件直接抛 `UnicodeDecodeError`。`parse_trajectories` 工具已处理 UTF-16，但通用的 `read_file` 没有。

**修改文件**：[local_sandbox.py:334-348](../../packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py)

**实现**：
- 先读前 4 字节检测 BOM
- `\xff\xfe` → UTF-16 LE
- `\xfe\xff` → UTF-16 BE
- `\xef\xbb\xbf` → UTF-8-sig（自动去 BOM）
- 无 BOM → 保持原有 UTF-8 行为
- UTF-16 解码后 strip 前导 BOM 字符

**测试**：在 [test_local_sandbox_encoding.py](../../packages/agent/backend/tests/test_local_sandbox_encoding.py) 新增 4 个测试：
- `test_read_file_utf16_le_bom`
- `test_read_file_utf16_be_bom`
- `test_read_file_utf8_bom`
- `test_read_file_utf8_no_bom_unchanged`（确认原有行为不变）

### 2.2 降级路径 E2E 测试

**问题**：当用户上传不支持范式的数据时，`compute_metrics` 工具返回 `status: failed`，lead agent prompt 已有用 `ask_clarification` 征求方向的逻辑（prompt.py line 294-310），但缺乏自动化测试验证。

**新增文件**：[test_paradigm_degradation.py](../../packages/agent/backend/tests/test_paradigm_degradation.py)

两个测试：
- `test_available_paradigms_only_contains_implemented` — 验证 `get_available_paradigms()` 只返回实际有 template 的范式（当前仅 shoaling）
- `test_compute_metrics_rejects_unsupported_paradigm` — 传入不支持的范式名（epm），验证返回 `status: failed` + 可用范式列表

### 2.3 Prompt 范式列表修正

**问题**：[prompt.py:1101-1105](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) "可用范式模板"列了 11 个范式名，但实际只有 shoaling 有 template。LLM 会误判所有范式都能自动分析。

**修改**：缩减为只列 `shoaling (斑马鱼群体行为)`，后续范式补全时逐个加入。

### 2.4 用户确认的信息

- **429 重试策略已在之前解决** — 不需要再做
- **M0.1 余项至此全部闭环**

---

## 3. 仓库当前状态

**分支**: `feature/etho-skills`（当前工作分支）

**测试基线**:
- backend: **1670 passed** / 14 skipped（比上次 1660 多 10 个新测试）
- 前端 / ethoinsight 库未改动，状态与 4/22 一致

**未提交的改动**（均已测试通过，待 commit）:

| 文件 | 状态 | 说明 |
|---|---|---|
| `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py` | 修改 | `read_file` BOM 检测 fallback |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 修改 | 可用范式列表缩减为 shoaling |
| `packages/agent/backend/tests/test_local_sandbox_encoding.py` | 修改 | 新增 4 个 UTF-16/BOM 测试 |
| `packages/agent/backend/tests/test_paradigm_degradation.py` | 新增 | 2 个降级路径测试 |

**上一次 handoff 提到的未追踪文件**（golden-cases/ 等）仍未追踪，本次未涉及。

---

## 4. M0.1 余项闭环状态

| 余项 | 状态 | 备注 |
|---|---|---|
| 不支持范式的降级路径 E2E | ✅ 完成 | 测试 + prompt 修正 |
| `read_file` UTF-16 fallback | ✅ 完成 | BOM 检测 + 4 个测试 |
| 429 重试策略优化 | ✅ 用户确认已解决 | 在更早的会话中完成 |

**roadmap.md 中 M0.1 余项行可以标记为 ✅。**

---

## 5. 下一阶段优先级

参考上一份交接 [2026-04-22 §4](2026-04-22-golden-case-and-auth-design.md)，M0.1 余项已清零，剩余优先级：

### P0 — 不阻塞外部

1. **发邮件给行为学同事**（启动 golden-case 标注流程）— 用户行动项
2. **发邮件给产品/技术支持团队**（启动资料收集）— 用户行动项
   - 需要：EthoVision XT Reference Manual + 13 范式 demo 导出 + 技术支持 FAQ
   - 详见 [fine-tuning-data-checklist.md A 类](../plans/2026-04-15-fine-tuning-data-checklist.md)

### P1 — 工程侧可推进

3. **M0.2-M0.4 范式批量补全**（EPM / OFT / FST / MWM）
   - 参考 [behavioral-reasoning-design.md §7](../plans/2026-04-21-behavioral-reasoning-design.md)
   - 先抽 `templates/_base.py` 基类，再逐个实现
   - 每范式边际成本 3-5 天

### P2 — 依赖外部反馈

4. 行为学同事返回 case-001 review → 定型 schema → 启动批量标注
5. 产品资料到位 → Phase 1 微调数据采集脚本

---

## 6. 关键上下文（下一个 Agent 必读）

与上一份交接一致，不重复。请先读 [2026-04-22 §5](2026-04-22-golden-case-and-auth-design.md)。

补充一点：本次修改了 **受保护文件** `lead_agent/prompt.py`（缩减范式列表），后续如果上游 DeerFlow 同步，`sync-deerflow.sh` 会标记此文件需人工判断。

---

## 7. 可能误导后续 Agent 的点

1. **M0.1 余项已全部完成** — roadmap.md 还标着 ⬜，需要更新
2. **prompt 里现在只列 shoaling** — 后续补全新范式 template 后，记得同步加回 prompt 列表
3. **改动尚未 commit** — 4 个文件改动已测试通过但未提交，用户未要求 commit

---

## 8. 推荐的接手第一步

**情况 A：用户想 commit 本次改动**
```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py \
       packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
       packages/agent/backend/tests/test_local_sandbox_encoding.py \
       packages/agent/backend/tests/test_paradigm_degradation.py
git commit -m "M0.1 余项：read_file UTF-16 fallback + 降级路径测试 + prompt 范式列表修正"
```

**情况 B：用户想推进范式补全（M0.2）**
1. 读 [behavioral-reasoning-design.md](../plans/2026-04-21-behavioral-reasoning-design.md) 了解 Layer A/B 分层
2. 看 [templates/shoaling.py](../../packages/ethoinsight/ethoinsight/templates/shoaling.py) 了解现有 template 结构
3. 设计 `_base.py` 基类，抽取 shoaling 中可复用的逻辑
4. 第一个新范式建议选 EPM（demo 数据在 `demo-data/DemoData/大鼠高架十字迷宫实验/`）

**情况 C：用户问"接下来做什么"**
→ 按 §5 优先级列表给建议

---

## 9. 测试与验证命令速查

```bash
# 全量后端测试
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate && make test

# 仅本次改动相关测试
PYTHONPATH=. uv run pytest tests/test_local_sandbox_encoding.py tests/test_paradigm_degradation.py -v

# Golden-case 校验
cd /home/qiuyangwang/noldus-insight
python3 scripts/validate_golden_case.py
```

---

## 10. 签名

**会话时间**: 2026-04-23
**模型**: Claude Opus 4.6
**会话时长**: 短（约 10 轮）
**交付物完整性**: ✅ 所有承诺的工作已落地并测试通过
**需要用户行动**: commit 改动（可选）；发 2 封邮件（见 4/22 交接 §8）
