# 2026-05-09 MVP 范式领域知识搬入 skill — 实施完成

## TL;DR

按 plan [docs/superpowers/plans/2026-05-09-mvp-paradigm-knowledge-into-skill-plan.md] 完成 Task 1-3：
- 搬入行为学同事 PR `0ec87dc1` 中填写的 11 个 markdown + 1 个新增（tail_suspension.md）
- SKILL.md 加"分析阶段使用本 skill"段，覆盖 set_experiment_paradigm 之后的 read 引导
- 冒烟通过：skill 仍能 load_skills / parse_skill_file 校验通过 / `make test` 无新增 fail

## 改动清单

| Commit | 内容 |
|---|---|
| `96d6e2c4` | 搬入行为学同事 MVP 6 范式领域知识到 skill |
| `60cccd02` | SKILL.md 加分析阶段渐进披露引导 |

## 验证结果

- skill `parse_skill_file` 校验：✅
- `load_skills(enabled_only=True)` 仍包含 ethovision-paradigm-knowledge：✅
- by-experiment 共 21 .md / by-template 共 20 .md
- `make test`：2179 passed / 14 skipped / 5 failed（全部 pre-existing，与 stash 前 baseline 一致）
- 手工 e2e：未做，留待 templates/epm.py 实施时一并验证

## MVP 范围内已就位 / 范围外未做

**已就位**（agent read 后能拿到完整 ground truth）：
- by-experiment: epm / open_field / zero_maze / light_dark_box / forced_swim / tail_suspension
- by-template: PlusMaze / OpenFieldRectangle / OpenFieldCircle / ZeroMaze / PorsoltCylinder

**范围外未做**（同事尚未填写或非 MVP 必需）：
- by-experiment/shoaling.md — 同事未填，shoaling 知识在 golden-cases/case-001
- by-template/AquariumTrack3D.md — 同事未填
- 其他 14 个 by-experiment + 14 个 by-template — 同事后续 PR 异步推进

## 后续 / 不在本次范围

- 写 `templates/epm.py` / `templates/open_field.py` 等 6 个范式分析模板（依赖本批同事知识 + ethoinsight-analysis skill 引导，是地基 plan 的 E2 任务）
- 在 `ethoinsight-analysis` skill 里加交叉引用，让 code-executor 自己 read by-experiment（可选，当前由 lead 拷段进 task() prompt 的方式更稳）
- 同事下一批 PR 进来时，重复本 plan Task 1 流程搬入新填的内容

## 已知遗留

- agent 当前在 task() 派遣时是否真按 SKILL.md 引导拷"必算指标"段进派遣 prompt — 需在 templates/epm.py 实施 + e2e 时观察
- 如果观察到 agent 跳过 read by-experiment 直接派遣，说明 SKILL.md 引导不够强；回到 SKILL.md 继续加强（"必须 read，否则视为分析准备不充分"）
