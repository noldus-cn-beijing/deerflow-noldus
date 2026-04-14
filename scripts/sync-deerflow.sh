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

REPORT_DIR="/tmp/deerflow-sync-report"

# 受保护文件列表（相对于 UPSTREAM_HARNESS）
# 这些是你修改过的上游文件，同步时需要人工判断
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

# 确认在项目根目录
if [[ ! -d "$LOCAL_HARNESS" ]] || [[ ! -d ".git" ]]; then
    echo -e "${RED}错误: 请在 noldus-insight 项目根目录运行此脚本${NC}"
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

# 从 git log 中找最近的 squash subtree commit
LAST_SYNC_MSG=$(git log --oneline --all --grep="Squashed '${SUBTREE_PREFIX}/' changes from" -1 2>/dev/null || true)

if [[ -z "$LAST_SYNC_MSG" ]]; then
    echo -e "${YELLOW}  未找到 subtree squash 记录，将对比上游所有文件${NC}"
    LAST_SYNC_COMMIT=""
else
    # 提取 "from XXXXX..YYYYY" 中的 YYYYY（上次同步到的上游 commit）
    LAST_SYNC_TO=$(echo "$LAST_SYNC_MSG" | grep -oP '\.\.([a-f0-9]+)' | sed 's/\.\.//')
    if [[ -n "$LAST_SYNC_TO" ]]; then
        echo "  上次同步到: ${LAST_SYNC_TO}"
        LAST_SYNC_COMMIT="$LAST_SYNC_TO"
    else
        echo -e "${YELLOW}  无法解析同步点，将对比所有文件${NC}"
        LAST_SYNC_COMMIT=""
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
    echo "  2. git add -A && git commit -m 'sync deerflow upstream to ${UPSTREAM_HEAD:0:7}'"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo -e "${CYAN}(dry-run 模式，未修改任何文件)${NC}"
fi
