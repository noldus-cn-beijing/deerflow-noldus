# TODOS

## Pre-Phase 1

### noldus-kb quality ownership
- **What**: Assign responsibility for noldus-kb knowledge base curation. Define a quality check process for retrieval relevance.
- **Why**: Phase 2 failure classification rubric requires knowing whether noldus-kb returned correct information. If retrieval quality is poor, model reasoning failures get misattributed.
- **Pros**: Clean Phase 2 evaluation, better debugging signal.
- **Cons**: Adds ongoing maintenance burden.
- **Context**: noldus-kb currently has 6200+ papers/manuals. No defined process for adding new papers, removing stale entries, or auditing retrieval quality. The Phase 2 rubric distinguishes "model reasoning failure" from "knowledge gap" — this distinction only works if noldus-kb retrieval is reliable. Without quality ownership, all failures default to "model problem."
- **Depends on**: noldus-kb being operational (it is). No blockers.
