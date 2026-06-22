#!/bin/bash
# ============================================================================
# DeerFlow 上游选择性同步工具
# ============================================================================
#
# 用法: ./scripts/sync-deerflow.sh [--dry-run] [--auto-apply]
#
#   --dry-run      只生成报告，不修改任何文件
#   --auto-apply   自动合入安全文件（默认需要确认）
#
# 工作原理:
#   1. fetch deerflow 上游最新代码
#   2. 找出上次同步点到现在上游改了哪些文件
#   3. 区分"安全文件"（你没改过）和"受保护文件"（你改过）
#   4. 安全文件可以直接 checkout 上游版本
#   5. 受保护文件生成 diff 报告，由你逐个决定
# ============================================================================

set -euo pipefail

# ---- 配置 ----

REMOTE="deerflow"
BRANCH="main"
SUBTREE_PREFIX="packages/agent"
# 上游 harness 路径（相对于上游仓库根目录）
UPSTREAM_HARNESS="backend/packages/harness/deerflow"
# 本地 harness 路径（相对于 noldus-insight 根目录）
LOCAL_HARNESS="${SUBTREE_PREFIX}/${UPSTREAM_HARNESS}"

# 同步基准状态文件 — 显式记录最近一次 sync 到的上游 commit
# 5-25 教训：subtree squash 不一定有（5-21 sync 也没做 squash），LAST_SYNC_COMMIT
# 仅依赖 git log --grep 会失效；用 .deerflow-sync-state 显式覆盖。
SYNC_STATE_FILE=".deerflow-sync-state"

REPORT_DIR="/tmp/deerflow-sync-report"

# 受保护文件列表（相对于 UPSTREAM_HARNESS）
# 这些是你修改过的上游文件，同步时需要人工判断
#
# ⚠️ 添加规则（5-25 sync 教训）：
#   1. 注册类文件（含 BUILTIN_TOOLS / BUILTIN_SUBAGENTS / __all__ 集合字面量 /
#      聚合 import 块）一旦含本地添加项必须保护。哪怕脚本判断"本地未改"也不能让
#      它整文件覆盖，否则会洗掉注册项 — 5-21 PR #23 翻车根因。
#   2. 飞轮 / 训练数据相关 schema 文件（feedback/sql.py 含 verdict 三分类 +
#      revised_text + message_id 四元组主键）— 上游模型在升级时不会带这些字段。
#   3. 任何文件 grep set_experiment_paradigm | identify_ev19 | prep_metric_plan |
#      shared_path | /mnt/shared | ethoinsight | extra_env | ArchivingSummarization |
#      ThinkTag | TrainingData | GateEnforcement | HandoffIsolation | Ev19Template |
#      verdict | revised_text | message_id 命中 → 必须保护。
PROTECTED_FILES=(
    # 高侵入 - Noldus 核心业务逻辑
    "agents/lead_agent/prompt.py"
    "subagents/builtins/__init__.py"
    # 中侵入 - 通用增强 + Noldus 功能
    "agents/middlewares/llm_error_handling_middleware.py"
    "mcp/tools.py"
    "sandbox/tools.py"
    "sandbox/local/local_sandbox.py"
    "agents/lead_agent/agent.py"
    "tools/builtins/task_tool.py"
    "subagents/executor.py"
    "config/paths.py"
    # 低侵入 - 1-3 行改动
    "sandbox/sandbox.py"
    "agents/thread_state.py"
    "agents/middlewares/thread_data_middleware.py"
    # 注册类文件（5-25 PR #36 / 5-21 PR #23 教训）— 含 BUILTIN_TOOLS / __all__ / BUILTIN_SUBAGENTS
    "tools/tools.py"
    "tools/builtins/__init__.py"
    "agents/__init__.py"
    "agents/factory.py"
    "subagents/registry.py"
    # 飞轮 / 训练数据 schema（5-25 PR #36 教训）— 含 verdict 三分类 + revised_text + message_id
    "persistence/feedback/sql.py"
    # Loop detection 中文 + ethoinsight 提示 + tool freq 3/5 阈值（5-25 PR #35）
    "agents/middlewares/loop_detection_middleware.py"
    # Setup agent tool — Noldus 定制（5-06 教训：上游脚本误标为安全文件）
    "tools/builtins/setup_agent_tool.py"
    # Guardrail middleware — Noldus 加了 name kwarg 解决 langchain unique-name 限制
    "guardrails/middleware.py"
    # 6-09 教训（sync-21 review）：以下文件不在旧清单但有 Noldus 定制，被 full-follow 洗掉过，补入保护
    # memory/prompt.py — 2026-05-13 topOfMind/history 隔离（_format_memory 砍会话级字段防文件幻觉）
    "agents/memory/prompt.py"
    # app_config.py — handoff_strict_mode 字段（experiment_context 依赖 + /tmp/disable_strict_handoff 开关）
    "config/app_config.py"
    # 6-22 教训（sync e418d729 review）：engine.py 不在旧清单但有 Noldus idempotent guard
    # （if _engine is not None: return）—— get_local_provider 每请求调 init_engine_from_config，
    # 无 guard 会重跑 os.makedirs 触发 langgraph blockbuster 500。全量合入上游版会洗掉 guard。
    "persistence/engine.py"
)

# ---- 颜色 ----

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---- 参数解析 ----

DRY_RUN=false
AUTO_APPLY=false

for arg in "$@"; do
    case $arg in
        --dry-run)    DRY_RUN=true ;;
        --auto-apply) AUTO_APPLY=true ;;
        -h|--help)
            echo "用法: $0 [--dry-run] [--auto-apply]"
            echo "  --dry-run      只生成报告，不修改文件"
            echo "  --auto-apply   自动合入安全文件（不需要确认）"
            exit 0
            ;;
        *)
            echo "未知参数: $arg"
            exit 1
            ;;
    esac
done

# ---- 辅助函数 ----

is_protected() {
    local file="$1"
    for pf in "${PROTECTED_FILES[@]}"; do
        if [[ "$file" == "$pf" ]]; then
            return 0
        fi
    done
    return 1
}

# ---- 检查前置条件 ----

echo -e "${BOLD}=== DeerFlow 上游选择性同步 ===${NC}"
echo ""

# 确认在项目根目录（兼容 worktree — .git 可以是文件指向真实仓库）
if [[ ! -d "$LOCAL_HARNESS" ]] || { [[ ! -d ".git" ]] && [[ ! -f ".git" ]]; }; then
    echo -e "${RED}错误: 请在 noldus-insight 项目根目录或 worktree 中运行此脚本${NC}"
    exit 1
fi

# 检查工作区是否干净（忽略未追踪文件）
if [[ -n "$(git diff --name-only HEAD)" ]] || [[ -n "$(git diff --cached --name-only)" ]]; then
    echo -e "${RED}错误: 工作区有未提交的改动，请先 commit 或 stash${NC}"
    git status -s
    exit 1
fi

# ---- Step 1: Fetch 上游 ----

echo -e "${BLUE}[1/5] Fetching ${REMOTE}/${BRANCH}...${NC}"
git fetch "$REMOTE" 2>/dev/null

UPSTREAM_HEAD=$(git rev-parse "${REMOTE}/${BRANCH}")
echo "  上游最新: ${UPSTREAM_HEAD:0:7}"

# ---- Step 2: 找上次同步点 ----

echo -e "${BLUE}[2/5] 查找上次同步点...${NC}"

LAST_SYNC_COMMIT=""

# 优先级 1: DEERFLOW_LAST_SYNC 环境变量（手动一次性覆盖）
if [[ -n "${DEERFLOW_LAST_SYNC:-}" ]]; then
    LAST_SYNC_COMMIT="$DEERFLOW_LAST_SYNC"
    echo -e "  ${CYAN}使用环境变量 DEERFLOW_LAST_SYNC=${LAST_SYNC_COMMIT:0:7}${NC}"
# 优先级 2: .deerflow-sync-state 文件（持久化记录）
elif [[ -f "$SYNC_STATE_FILE" ]]; then
    LAST_SYNC_COMMIT=$(grep -E '^last_sync_commit:' "$SYNC_STATE_FILE" 2>/dev/null | head -1 | awk '{print $2}')
    if [[ -n "$LAST_SYNC_COMMIT" ]]; then
        echo "  ${SYNC_STATE_FILE} 记录: ${LAST_SYNC_COMMIT:0:7}"
    fi
fi

# 优先级 3: subtree squash commit message (fallback)
if [[ -z "$LAST_SYNC_COMMIT" ]]; then
    LAST_SYNC_MSG=$(git log --oneline --all --grep="Squashed '${SUBTREE_PREFIX}/' changes from" -1 2>/dev/null || true)

    if [[ -z "$LAST_SYNC_MSG" ]]; then
        echo -e "${YELLOW}  未找到 subtree squash 记录 + 无 ${SYNC_STATE_FILE}，将对比上游所有文件${NC}"
    else
        # 提取 "from XXXXX..YYYYY" 中的 YYYYY（上次同步到的上游 commit）
        LAST_SYNC_TO=$(echo "$LAST_SYNC_MSG" | grep -oP '\.\.([a-f0-9]+)' | sed 's/\.\.//')
        if [[ -n "$LAST_SYNC_TO" ]]; then
            echo "  subtree squash 提取: ${LAST_SYNC_TO}"
            LAST_SYNC_COMMIT="$LAST_SYNC_TO"
        else
            echo -e "${YELLOW}  无法解析 subtree squash，将对比所有文件${NC}"
        fi
    fi
fi

# ---- Step 3: 获取上游改动文件列表 ----

echo -e "${BLUE}[3/5] 分析上游改动...${NC}"

if [[ -n "$LAST_SYNC_COMMIT" ]]; then
    # 检查 commit 是否存在于 deerflow remote
    if git cat-file -e "${LAST_SYNC_COMMIT}" 2>/dev/null; then
        CHANGED_FILES=$(git diff --name-only "${LAST_SYNC_COMMIT}" "${REMOTE}/${BRANCH}" -- "${UPSTREAM_HARNESS}/" 2>/dev/null | sed "s|^${UPSTREAM_HARNESS}/||")
        NEW_COMMITS=$(git log --oneline "${LAST_SYNC_COMMIT}..${REMOTE}/${BRANCH}" -- "${UPSTREAM_HARNESS}/" 2>/dev/null | wc -l)
        echo "  自上次同步以来: ${NEW_COMMITS} 个 commits"
    else
        echo -e "${YELLOW}  同步点 ${LAST_SYNC_COMMIT} 不在 ${REMOTE} 中，改为全量对比${NC}"
        CHANGED_FILES=$(git ls-tree -r --name-only "${REMOTE}/${BRANCH}" -- "${UPSTREAM_HARNESS}/" 2>/dev/null | sed "s|^${UPSTREAM_HARNESS}/||")
    fi
else
    CHANGED_FILES=$(git ls-tree -r --name-only "${REMOTE}/${BRANCH}" -- "${UPSTREAM_HARNESS}/" 2>/dev/null | sed "s|^${UPSTREAM_HARNESS}/||")
fi

if [[ -z "$CHANGED_FILES" ]]; then
    echo -e "${GREEN}  上游无新改动，已是最新！${NC}"
    exit 0
fi

TOTAL_FILES=$(echo "$CHANGED_FILES" | wc -l)
echo "  上游改动文件: ${TOTAL_FILES} 个"

# ---- Step 4: 分类文件 ----

echo -e "${BLUE}[4/5] 分类文件...${NC}"

SAFE_FILES=()
PROTECTED_CHANGED=()
NEW_FILES=()

while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    local_path="${LOCAL_HARNESS}/${file}"

    if [[ ! -f "$local_path" ]]; then
        # 本地不存在 → 上游新增的文件
        NEW_FILES+=("$file")
    elif is_protected "$file"; then
        PROTECTED_CHANGED+=("$file")
    else
        SAFE_FILES+=("$file")
    fi
done <<< "$CHANGED_FILES"

echo ""
echo -e "  ${GREEN}安全文件（可自动合入）: ${#SAFE_FILES[@]}${NC}"
echo -e "  ${GREEN}上游新增文件: ${#NEW_FILES[@]}${NC}"
echo -e "  ${YELLOW}受保护文件（需人工判断）: ${#PROTECTED_CHANGED[@]}${NC}"

# ---- Step 5: 生成报告 + 操作 ----

echo -e "${BLUE}[5/5] 处理文件...${NC}"
echo ""

# 清理报告目录
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

# --- 处理上游新增文件 ---
if [[ ${#NEW_FILES[@]} -gt 0 ]]; then
    echo -e "${CYAN}--- 上游新增文件 (${#NEW_FILES[@]}) ---${NC}"
    for file in "${NEW_FILES[@]}"; do
        echo "  + $file"
    done

    if [[ "$DRY_RUN" == false ]]; then
        if [[ "$AUTO_APPLY" == true ]]; then
            apply_new="y"
        else
            echo ""
            read -rp "是否合入所有新增文件？[Y/n] " apply_new
            apply_new=${apply_new:-y}
        fi

        if [[ "$apply_new" =~ ^[Yy]$ ]]; then
            for file in "${NEW_FILES[@]}"; do
                local_path="${LOCAL_HARNESS}/${file}"
                mkdir -p "$(dirname "$local_path")"
                git show "${REMOTE}/${BRANCH}:${UPSTREAM_HARNESS}/${file}" > "$local_path"
            done
            echo -e "  ${GREEN}已合入 ${#NEW_FILES[@]} 个新增文件${NC}"
        fi
    fi
    echo ""
fi

# --- 处理安全文件 ---
if [[ ${#SAFE_FILES[@]} -gt 0 ]]; then
    echo -e "${CYAN}--- 安全文件 (${#SAFE_FILES[@]}) ---${NC}"
    for file in "${SAFE_FILES[@]}"; do
        echo "  ~ $file"
    done

    if [[ "$DRY_RUN" == false ]]; then
        if [[ "$AUTO_APPLY" == true ]]; then
            apply_safe="y"
        else
            echo ""
            read -rp "是否合入所有安全文件？[Y/n] " apply_safe
            apply_safe=${apply_safe:-y}
        fi

        if [[ "$apply_safe" =~ ^[Yy]$ ]]; then
            for file in "${SAFE_FILES[@]}"; do
                local_path="${LOCAL_HARNESS}/${file}"
                git show "${REMOTE}/${BRANCH}:${UPSTREAM_HARNESS}/${file}" > "$local_path"
            done
            echo -e "  ${GREEN}已合入 ${#SAFE_FILES[@]} 个安全文件${NC}"
        fi
    fi
    echo ""
fi

# --- 处理受保护文件 ---
if [[ ${#PROTECTED_CHANGED[@]} -gt 0 ]]; then
    echo -e "${YELLOW}--- 受保护文件 (${#PROTECTED_CHANGED[@]}) ---${NC}"
    echo -e "${YELLOW}以下文件包含你的定制改动，上游也有新改动，需要逐个判断:${NC}"
    echo ""

    for file in "${PROTECTED_CHANGED[@]}"; do
        local_path="${LOCAL_HARNESS}/${file}"
        report_file="${REPORT_DIR}/${file//\//_}.diff"

        # 获取上游对该文件的最近 commit 信息
        if [[ -n "$LAST_SYNC_COMMIT" ]] && git cat-file -e "${LAST_SYNC_COMMIT}" 2>/dev/null; then
            upstream_log=$(git log --oneline "${LAST_SYNC_COMMIT}..${REMOTE}/${BRANCH}" -- "${UPSTREAM_HARNESS}/${file}" 2>/dev/null)
        else
            upstream_log=$(git log --oneline -3 "${REMOTE}/${BRANCH}" -- "${UPSTREAM_HARNESS}/${file}" 2>/dev/null)
        fi

        # 生成 diff: 上游新版 vs 你的版本
        diff -u <(git show "${REMOTE}/${BRANCH}:${UPSTREAM_HARNESS}/${file}" 2>/dev/null) "$local_path" > "$report_file" 2>/dev/null || true

        diff_lines=$(wc -l < "$report_file")

        echo -e "  ${BOLD}$file${NC} (diff: ${diff_lines} 行)"
        if [[ -n "$upstream_log" ]]; then
            echo "    上游改动:"
            echo "$upstream_log" | sed 's/^/      /'
        fi
        echo "    报告: $report_file"
        echo ""
    done

    echo -e "${YELLOW}查看 diff 报告: ls ${REPORT_DIR}/${NC}"
    echo -e "${YELLOW}对每个文件，你可以:${NC}"
    echo "  a) 保留你的版本（不做任何操作）"
    echo "  b) 手动编辑合入上游的部分改动"
    echo "  c) 接受上游版本: git show ${REMOTE}/${BRANCH}:${UPSTREAM_HARNESS}/<file> > ${LOCAL_HARNESS}/<file>"
fi

# ---- 总结 ----

echo ""
echo -e "${BOLD}=== 同步摘要 ===${NC}"
echo "  上游: ${REMOTE}/${BRANCH} @ ${UPSTREAM_HEAD:0:7}"
[[ -n "${LAST_SYNC_COMMIT:-}" ]] && echo "  同步基准: ${LAST_SYNC_COMMIT:0:7}"
echo "  新增文件: ${#NEW_FILES[@]}"
echo "  安全合入: ${#SAFE_FILES[@]}"
echo "  需人工判断: ${#PROTECTED_CHANGED[@]}"

if [[ ${#PROTECTED_CHANGED[@]} -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}处理完受保护文件后，记得:${NC}"
    echo "  1. cd packages/agent/backend && make test"
    echo "  2. 更新 ${SYNC_STATE_FILE}: echo 'last_sync_commit: ${UPSTREAM_HEAD}' > ${SYNC_STATE_FILE}"
    echo "  3. git add -A && git commit -m 'sync deerflow upstream to ${UPSTREAM_HEAD:0:7}'"
else
    echo ""
    echo -e "${GREEN}提示: 合入后更新 ${SYNC_STATE_FILE} 推进同步基准:${NC}"
    echo "  echo 'last_sync_commit: ${UPSTREAM_HEAD}' > ${SYNC_STATE_FILE}"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo -e "${CYAN}(dry-run 模式，未修改任何文件)${NC}"
fi
