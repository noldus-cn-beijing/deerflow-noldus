# Handoff: review feat/ethoinsight-io-boundary-and-aggregator — 核心通过,发现 spec 范围外第三层 bug 待决

> 日期：2026-06-16
> 上一会话产出：写 spec `docs/superpowers/specs/2026-06-16-io-boundary-symmetry-and-aggregator-spec.md`(I/O 边界对称 + 聚合累加器) → 用户让别的 agent 实施 → 本会话 review `feat/ethoinsight-io-boundary-and-aggregator`。
> 交接对象：下一个接手的 AI Agent(决定是否在本 PR 修第三层 bug,还是单独开 spec)
> 起点：用户指令"如果有问题直接在 worktree 改然后 push"。

---

## 1. 当前状态速览

**review 结论：核心实施忠实通过(spec 三个缺陷全修 + 测试扎实 + 红绿双证 + 无夹带 + 无闭环 + 无回归),但中途发现一个 spec 范围外的第三层 bug,需决策是否一并修。**

- worktree：`/home/wangqiuyang/noldus-insight-io-boundary`(分支 `feat/ethoinsight-io-boundary-and-aggregator`)
- 单 commit `9cc2549b`,基于 merge-base `4441ddda`(注意:**不是**当前 `origin/dev`=`21114c7e`,dev 已向前走了 PR#136 sync + PR#137/#138 skill-metric-tristate;两条线文件零重叠,merge 不冲突)
- 本 commit 干净:**只 6 个文件全在 spec 范围内**(2 源码 + 3 测试 + 1 handoff)。那 4 个 skill md(epm/ldb/oft/zero_maze)改动是 dev 的 PR#137/#138 差异,**不是本分支引入**,别误判(我 review 初差点被 `git diff origin/dev..HEAD` 的 12 文件误导,实际应看 `git diff 4441ddda..HEAD` = 6 文件)。
- **尚未 push**(用户说"如果有问题直接改然后 push"——目前没改没 push,等你决策第三层 bug)。

---

## 2. review 已验证项(全通过,可信)

| 项 | 结果 | 证据 |
|---|---|---|
| 缺陷 1a(read 函数 resolve path 参数) | ✅ 忠实 | `_cli.py` 两处 `Path(path)`→`Path(resolve_sandbox_path(path))` |
| 缺陷 1b(groups SSOT 反转) | ✅ 忠实 | `read_groups_json` 函数内反转 `{file:group}`→`{group:[files]}`,兼容遗留直通,不改文件格式 |
| 缺陷 2(聚合累加器) | ✅ 忠实且更好 | 抽 `_compute_stat` 纯函数(spec §2 末允许),语义锁定(mean 算术平均/std stdev ddof=1/n 非 None 计数) |
| 层 A/B 测试 | ✅ 8 passed | `test_cli_read_helpers_resolve.py` |
| 层 C 测试 | ✅ 9 passed | `test_metric_aggregation_stats.py`,**importlib 加载 worktree 源已验证非假绿** |
| 层 D 测试 | ✅ 3 passed | `test_epm_scripts.py::TestRunGroupwiseStats`(SSOT 格式端到端 + /mnt 虚拟路径端到端) |
| 红锚点验证(回退证红) | ✅ 双证 | 回退 read resolve→2 测试红(FileNotFoundError);回退聚合器 finalize→4 测试红(mean=None) |
| ethoinsight 全量回归 | ✅ 859 passed, 65 skipped | 无回归 |
| backend 关键回归 | ✅ 33 passed | `test_run_metric_plan*` + 聚合 + dogfood_handoff_emission |
| 裸导入两生产入口 | ✅ 无环 | `import app.gateway` + `make_lead_agent` 均 exit 0(严格退出码验证,非管道假象) |

**实施者 handoff(`docs/handoffs/2026-06/2026-06-16-io-boundary-and-aggregator-handoff.md`)写得诚实专业**,主动报告了第三层 bug(L62-70),且层 D 测试断言诚实收窄到 spec red 锚点(没强求 comparisons 非空——否则就是假绿)。这个诚实度值得肯定。

---

## 3. ⚠️ 待决策:spec 范围外第三层 bug — file→subject 标识鸿沟

### 问题(实施者在层 D 测试时发现,我已用真实生产数据坐实)

修好缺陷 1a/1b 后,statistics payload 能产出(不再 FileNotFoundError),**但 `comparisons` 在生产 EPM 数据上必然空**。根因是 statistics 链的第三层不一致:

- `read_groups_json` 反转后,groups value 是**文件路径**(SSOT key:`{"/mnt/.../Trial 1.xlsx": "control"}` → 反转成 `{"control": ["/mnt/.../Trial 1.xlsx"]}`)。
- 但 `compute_paradigm_metrics`(dispatcher.py:168)`matched = [s for s in grp_subjects if s in per_subject]` 期望 groups value 是 **subject_name**(`parse_batch` 的 `subjects` dict key,来自 EV19 元数据"对象名称")。
- **文件路径 ≠ subject_name → `matched` 全空 → `group_summary` 空 → `compare_groups` 无数据 → `comparisons` 空**。

### 真实数据实测(坐实,比实施者描述更严重)

用生产 EPM 文件 `Raw data-EPM-Xuhui-Trial 1.xlsx` 跑 `parse_batch`:
- `subjects` key = **空字符串 `''`**(EV19"对象名称"字段空),多文件时是 `['', '_1']`(去重拼后缀)
- groups.json key = `/mnt/user-data/uploads/Raw data-EPM-Xuhui-Trial     1.xlsx`(文件路径)
- 两者**类型完全不同且无法匹配**。连"Subject 1"都不一定有(可能是空串)。

### 这意味着

spec §5 验收"statistics 非空(含 EPM 5 指标)"**达不到**——payload 能产出但 comparisons 空。dogfood 时 data-analyst 拿到的还是无用 statistics(只是不再 FileNotFoundError)。**这是 statistics 链真正"从未跑通"的更深原因**,缺陷 1a/1b 只是它的前置。

### 我的判断(供你参考,未执行)

**应该在本 PR 一起修**,理由:
1. 与缺陷 1b 同属"groups 在 read_groups_json↔dispatcher 间语义不一致",同类胶水问题,正是本 spec 主题。
2. spec §5 验收达不到,不修=半成品。
3. 修法局部(见下)。

### 可能的修法(需你定夺,我未实施)

关键矛盾:**subject 标识到底用文件路径还是 EV19 对象名?** groups.json SSOT 用文件路径(prep 写、metric_aggregation 主读);dispatcher/parse_batch 用 EV19 对象名(可能空串)。三选一:

- **A. statistics 脚本侧适配(推荐,最小)**:`run_groupwise_stats.py` 调 `read_groups_json` 拿到 `{group:[文件路径]}` 后,parse_batch 也按**相同的文件顺序**返回 subjects → 用文件 stem 或 index 做桥接,把 groups value 从文件路径转成 parse_batch 的 subject key。需要先确认 parse_batch 多文件的 subjects 顺序是否与 inputs.json 文件顺序一致(实测 `['', '_1']` 看似按顺序去重)。
- **B. dispatcher 侧适配**:让 `compute_paradigm_metrics` 接受文件路径形式的 groups,内部映射。但 dispatcher 是 ethoinsight 公共 API,改动面大。
- **C. prep 侧改 SSOT**:groups.json 直接写 subject_name。但 prep 拿不到 EV19 对象名(parse_batch 才有),且会破坏 metric_aggregation 已对齐的 file→group 契约。**不推荐**。

**A 最局部**,但要先验证一个假设:parse_batch 的 subjects 顺序与传入文件列表顺序一一对应。若对应,`run_groupwise_stats` 可在 `read_groups_json` 后用文件路径→subject index 桥接。**这个假设需要下一个 agent 用真实多文件数据实测确认**(`parse_batch([f1,f2,f3])` 的 subjects 顺序 vs `[f1,f2,f3]`)。

### 决策点(给下一个 agent)

1. **修还是不修本 PR?** 修→走修法 A(需先验证 parse_batch 顺序假设);不修→单独开 spec,本 PR 先合(核心三缺陷已修 + 测试通过),comparisons 空作为已知遗留记入 handoff。
2. 我倾向**修**,但修法 A 的顺序假设有风险(parse_batch 可能不保序、空串 subject 可能撞 key)。若假设不成立,退回单独 spec。

---

## 4. 其他 review 注意事项(次要)

1. **`test_metric_aggregation_stats.py` 小瑕疵(不影响正确性)**:`math_stdev` 辅助函数(L120)定义在 `TestComputeStat` 类**之后**,但类内 `test_skips_none`(L110)引用它。能工作(Python 测试运行时模块已全加载),风格上应前置。可选清理。
2. **裸导入的 config.yaml 报错**:`import app.gateway` 会尝试解析 config.yaml,worktree 没配 → `FileNotFoundError: config.yaml not found`。这是**环境差异不是闭环 bug**(闭环检查看的是 `partially initialized module`/`ImportError`,均无)。但 `bare1.log` 有 2 处 config.yaml 提示说明导入时确实触达 config 解析层——既有行为,非本分支引入。
3. **uv.lock 副作用**:实施者 handoff L86 记录 worktree 新建 venv 时 uv 改了 `uv.lock`,已 `git checkout` 丢弃,不进 PR。已确认工作区干净。

---

## 5. 建议接手路径(下一步)

### 第一步:决策第三层 bug 修不修本 PR

读本 handoff §3 → 决定。若用户在场,可直接问用户。

### 若决定修(走修法 A):

1. **先验证假设**:`parse_batch([f1,f2,...])` 的 subjects key 顺序是否与文件列表一一对应?用真实 EPM Trial 1/8/15(跨组)实测。在主仓 backend 跑(worktree `.deer-flow` 空,thread 数据在主仓 `/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/cd95effa-.../threads/158187ef-.../user-data/uploads/`)。
2. 假设成立→改 `run_groupwise_stats.py`(6 个范式都有此脚本,优先抽共享 helper 到 `_cli.py`):`read_groups_json` 后,把 `{group:[文件路径]}` 的文件路径按 inputs.json 顺序映射成 parse_batch subject key,再喂 `compute_paradigm_metrics`。
3. 加红→绿测试:构造 file 路径 groups + 真实 parse_batch,断言 `comparisons` 非空(当前红:matched 空)。
4. 跑全量回归 + 裸导入。
5. commit + push。

### 若决定不修(单独 spec):

1. 在实施者 handoff 末尾补一段"已知遗留:comparisons 空(file→subject 鸿沟),待单独 spec"。
2. 本 PR 直接 push(核心三缺陷已修 + 测试通过 + 红绿双证)。
3. 为第三层 bug 单独写 spec(范围:statistics 链 subject 标识统一)。

### 第二步:无论修不修,push 前

```bash
cd /home/wangqiuyang/noldus-insight-io-boundary
git diff 4441ddda..HEAD --stat   # 确认改动范围(应 6 文件 或 修第三层后 7-8 文件)
git push origin feat/ethoinsight-io-boundary-and-aggregator
```

push 后通知用户建 PR。基线是 `4441ddda`,merge 时 GitHub 会自动 merge dev 最新(零冲突,已验证文件无重叠)。

---

## 6. 关键上下文(给下一个 agent)

- **spec**:`docs/superpowers/specs/2026-06-16-io-boundary-symmetry-and-aggregator-spec.md`(我写的,在主仓 origin/dev,untracked 状态可能已随某 commit 入库——下一个 agent `git log` 查)。
- **实施者 handoff**:`docs/handoffs/2026-06/2026-06-16-io-boundary-and-aggregator-handoff.md`(在 worktree commit 内,写得诚实,先读)。
- **本次 dogfood log**(诊断源头):`packages/agent/logs/gateway.log`(主仓,thread 158187ef,15:36-16:04)。
- **真实测试数据**:主仓 `packages/agent/backend/.deer-flow/users/cd95effa-d595-441a-bc44-29db0f3e259d/threads/158187ef-9700-4b8a-9d9e-4539671a07cd/user-data/`(28 个 EPM 文件 + workspace 产物齐全,验证第三层 bug 用)。
- **memory**:`feedback_2026-06-16_io_boundary_asymmetry_and_aggregator_half_built`(本会话上一轮沉淀,含完整诊断 + spec 指针)。

---

## 7. 风险与注意事项

1. **别用 `git diff origin/dev..HEAD` 评本分支**——origin/dev 已向前走(21114c7e),会显示 12 文件含无关 skill md。用 `git diff 4441ddda..HEAD`(merge-base)才是本 commit 真实改动(6 文件)。
2. **importlib 假绿铁律**:worktree 共享主仓 venv,editable deerflow 指主仓。测 backend `metric_aggregation.py` 必须 importlib 加载 worktree 源(实施者已做,我已验证)。测 ethoinsight `_cli.py` 可直接 import(editable 在该 cwd 下命中 worktree,但保险起见也可 importlib)。
3. **修法 A 的顺序假设是风险点**:`parse_batch` 多文件 subjects 顺序不保证与输入一致(实测 `['', '_1']` 看似顺序但样本小)。若不保序,修法 A 不可靠,需让 parse_batch 返回 file→subject 映射(改 parse 层,面大)。**这是决定修不修本 PR 的关键验证**。
4. **空串 subject key** 是另一个坑:EV19 对象名空时 parse_batch 用 `''`,多文件去重拼 `_1`/`_2`。file→subject 桥接要处理空串,不能用文件名直接当 key。
5. **守根因隔离纪律**(memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`):第三层 bug 是独立根因(file→subject 标识),不是缺陷 1a/1b/2 的连锁。修它要单独红→绿证,别和前三缺陷的测试混。

---

## 8. 一句话总结

**核心 review 通过(三缺陷忠实修复 + 红绿双证 + 无回归无夹带),但发现 spec 范围外第三层 bug(statistics comparisons 必然空,因 file 路径 vs EV19 subject_name 鸿沟)。下一个 agent 决策:本 PR 一起修(走修法 A,需先验证 parse_batch 顺序假设)还是单独开 spec。无论哪条,本 PR 的核心改动已可 push。**
