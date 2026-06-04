#!/bin/bash
# 推送今日 noldus-insight 改动到企业微信群
# 用法: ./scripts/send-daily-digest.sh [天数]

set -euo pipefail
cd "$(dirname "$0")/.."

source .env.wecom 2>/dev/null || { echo "❌ 缺少 .env.wecom 配置文件"; exit 1; }
[ -z "${WECOM_BOT_WEBHOOK:-}" ] && { echo "❌ WECOM_BOT_WEBHOOK 未配置"; exit 1; }

DAYS="${1:-1}"
SINCE=$(date -d "$DAYS days ago 00:00:00" +"%Y-%m-%dT%H:%M:%S+08:00" 2>/dev/null || date -v-${DAYS}d +"%Y-%m-%dT00:00:00+08:00")
TODAY=$(date +"%m月%d日")

# 收集提交
COMMITS=$(git log --since="$SINCE" --pretty=format:"%h | %an | %s" --branches='main,dev' --all 2>/dev/null | head -50)

if [ -z "$COMMITS" ]; then
  echo "今日无新提交，跳过。"
  exit 0
fi

COMMIT_COUNT=$(echo "$COMMITS" | wc -l)

# 如果配了 NewAPI，用 AI 总结；否则直接发简单版本
if [ -n "${NEWAPI_BASE_URL:-}" ] && [ -n "${NEWAPI_API_KEY:-}" ]; then
  SUMMARY=$(curl -s --max-time 90 "$NEWAPI_BASE_URL/v1/chat/completions" \
    -H "Authorization: Bearer $NEWAPI_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg commits "$COMMITS" '{
      model: "deepseek-v4-pro",
      temperature: 0.3,
      messages: [{
        role: "system",
        content: "你是 EthoInsight 项目的日报助手，面向行为学研究员的 AI 分析工具。\n\n根据 git log 写一份简短的日报，发给非技术团队（产品 / 支持 / 管理层）。\n\n规则：\n1. 用一句话概括今天的进展\n2. 🆕 新功能（用非技术语言描述，说\"能做什么\"而不是\"改了什么代码\"）\n3. 🐛 已修复（说\"之前XX问题，现在好了\"）\n4. 🔧 后台维护（一句话带过）\n5. 没有的类别就跳过\n\n控制在200字以内，语气轻松自然。"
      }, {
        role: "user",
        content: $commits
      }]
    }')")
  CONTENT=$(echo "$SUMMARY" | jq -r '.choices[0].message.content // ""')
else
  # 无 NewAPI 时的简单格式
  CONTENT="今日共 ${COMMIT_COUNT} 个提交：\n\n$(echo "$COMMITS" | awk '{print "- "$0}')"
fi

# 推送到企业微信
RESP=$(curl -s -X POST "$WECOM_BOT_WEBHOOK" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg content "## EthoInsight 每日更新 (${TODAY})\n${CONTENT}\n\n---\n📊 今日共 ${COMMIT_COUNT} 个提交 | 🤖 自动生成" '{
    msgtype: "markdown",
    markdown: { content: $content }
  }')")

ERRCODE=$(echo "$RESP" | jq -r '.errcode')
if [ "$ERRCODE" = "0" ]; then
  echo "✅ 日报已推送到企业微信群（${COMMIT_COUNT} 个提交）"
else
  echo "❌ 推送失败: $RESP"
fi
