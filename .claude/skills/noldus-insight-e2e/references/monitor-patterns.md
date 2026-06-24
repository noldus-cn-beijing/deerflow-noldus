# gateway.log Monitor Patterns

While `run-e2e.cjs` runs in the background (~up to 40min), the SKILL.md runbook
arms a **Monitor** against the gateway log for live milestone/failure visibility,
paired with a **bounded Bash waiter** for the single completion notification.

## Why two tools

- The **Monitor** (`tail -F gateway.log | grep`) streams one notification per
  milestone/failure line as it happens — operator visibility during the long
  drive. It is advisory; if it gets noisy/auto-stopped, nothing breaks.
- The **bounded Bash waiter** (`until [ -f $E2E_OUT/.terminal ]; do sleep 10; done`,
  `run_in_background`) fires exactly ONE completion notification when
  `run-e2e.cjs` writes the `.terminal` marker (at terminal break, AND in its
  `finally` on any exit — including fatal). This is the authoritative done
  signal. Per Monitor-tool guidance: an unbounded `tail -F` alone would linger
  after success; pairing it with a bounded waiter that exits on the marker
  avoids that.

## grep alternation (must cover success AND failure)

Per monitoring hygiene: silence is not success. The alternation covers terminal
states, not just the happy path.

```
tail -n0 -F "$GW" | grep -E --line-buffered \
  'handoff_[a-z_]+\.json|sealed_by|seal_[a-z_]+_handoff|fill_[a-z_]+|finalize|experiment_summary fact written|report\.md|captured AI message|started background task|completed async execution|reached max_turns|Tool execution failed|Traceback|ValueError|FileNotFoundError|RuntimeError|Killed|RecursionError|TimeoutError|clarification'
```

- **Success signatures**: `handoff_*.json` (each subagent handoff landing),
  `sealed_by`, `seal_*_handoff`, `fill_*`, `finalize`,
  `experiment_summary fact written`, `report.md`, `completed async execution`.
- **Progress signatures**: `captured AI message` (subagent turn tick),
  `started background task` (subagent dispatch), `clarification` (HITL round —
  visibility without spamming).
- **Failure signatures** (a silent crashloop must emit): `Tool execution failed`,
  `Traceback`, `ValueError`, `FileNotFoundError` (the fill-race signature),
  `RuntimeError`, `Killed`, `RecursionError`, `TimeoutError`, `reached max_turns`.

## Subagent internals caveat (memory: diagnose via replay, not lead-thread dump)

Subagent internal turns (tool_calls, thinking, invalid_tool_calls) do NOT enter
the checkpoint or this gateway log in full detail — `SubagentExecutor` runs with
`checkpointer=False`. The gateway log shows `SubagentExecutor initialized`,
`captured AI message #N`, `completed async execution`, and
`tool_error_handling_middleware` errors. To get the per-turn tool_call sequence
you must independently replay the subagent (手段 c), not just tail the lead
thread. The Monitor here is for coarse milestone/failure detection, not deep
subagent forensics.

## Variables the SKILL.md runbook fills in

- `GW` = `/home/wangqiuyang/noldus-insight/packages/agent/logs/gateway.log`
- `$E2E_OUT/.terminal` = done marker written by `run-e2e.cjs`
- The runbook also greps a bounded `gateway-excerpt.log` (between run start/end
  timestamps) into `$E2E_OUT` as the durable record — the persistent Monitor is
  for live visibility, the excerpt is for the archive.
