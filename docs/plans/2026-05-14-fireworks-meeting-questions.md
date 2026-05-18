# Fireworks.ai Meeting: Questions for Qwen3-30B-A3B MoE Fine-Tuning Support

> **Date**: 2026-05-14
> **Purpose**: Confirm Fireworks.ai's full support for Qwen3-30B-A3B MoE fine-tuning, as part of EthoInsight's platform selection
> **Background**: [2026-05-13 Base Model Decision Memo](2026-05-13-base-model-decision-memo.md)

---

## 0. Our Use Case (30-Second Overview)

- **Product**: EthoInsight — an AI analysis assistant for behavioral research labs, shipped as on-premise software
- **Hardware**: Customer workstations with NVIDIA RTX 5070 Ti 16GB or higher (5090 / PRO 4500 32GB TBD)
- **Target Model**: Leaning toward Qwen3-30B-A3B-Instruct-2507 (MoE: 30B total / 3B active per token), running NVFP4 quantized locally
- **Fine-Tuning Path**: SFT first (LoRA), CPT deferred, DPO post-v0.1
- **Timeline**: v0.1 planned for September 2026; training data flywheel and golden-cases accumulating now

---

## 1. Core Question: MoE Fine-Tuning Support

### 1.1 Does Fireworks support SFT / LoRA for Qwen3-30B-A3B?

| What we've seen | What we need confirmed |
|---|---|
| One doc page states "Qwen 3 non-MoE models" | Is 30B-A3B MoE supported or not? |
| Another source mentions training on "Qwen3-30B MoE with 128 experts" | Is this live or on the roadmap? If roadmap, ETA? |

### 1.2 MoE LoRA Technical Details

- Which layers do you target with LoRA — the router, the experts, or both? What's your recommended strategy?
- Do you support QLoRA (4-bit quantized LoRA) for 30B-A3B?
- Does the platform handle load balancing / expert collapse automatically during MoE training?
- Typical runtime and cost for a single LoRA SFT run (~1K–3K samples)?

### 1.3 Cross-Check

- You support fine-tuning for DeepSeek V3/R1 (also MoE). What makes Qwen3-30B-A3B's MoE architecture different such that support lags — or is it purely a scheduling issue?
- If Qwen3-30B-A3B is not yet supported but Qwen3.5/3.6 35B-A3B is, which would you recommend?

---

## 2. Model Weight Export & Local Deployment

This is a **dealbreaker** for our on-premise delivery model:

- After LoRA fine-tuning, **can we export the full merged weights** (LoRA merged into base model) to bundle into our product?
- What export formats are supported? GGUF / AWQ / NVFP4 / GPTQ? We specifically need NVFP4 for RTX 5090 local inference.
- Are there additional fees or restrictions on weight export?
- If you only support inference via your API post-fine-tuning — **that's a non-starter for us. Please be upfront.**

---

## 3. Fine-Tuning on Top of Instruct Versions

- We plan to LoRA-SFT on top of `Qwen3-30B-A3B-Instruct-2507` (the RLHF'd instruct checkpoint), not the base model. Is this supported?
- Is there any known degradation risk when stacking LoRA on an already-instruct-tuned model (e.g., instruct capabilities getting washed out)? Any recommended practices?

---

## 4. CPT (Continued Pre-Training) Support

- We may do CPT later to inject behavioral science domain knowledge. Does Fireworks support CPT?
- MoE CPT support status?
- Can CPT + SFT be chained on the same platform (no export/re-import between stages)?

---

## 5. Training Data Requirements

- Minimum number of SFT samples required to launch a training job? Any hard floor?
- Supported data formats? (ShareGPT / Alpaca / OpenAI chat format?)
- Is multi-turn conversation data supported? (Our agent involves multi-turn interactions)
- Is tool-calling format training data supported?

---

## 6. Platform Comparison (We Are Also Evaluating Volcengine)

| Dimension | What we need to clarify |
|---|---|
| **Qwen3-30B-A3B MoE SFT support** | Which platform supports it first? Which is more mature? |
| **Weight export** | Do both platforms support it? Format differences? |
| **China-based access & latency** | Fireworks performance from within China? |
| **Pricing** | Cost range for a single 30B LoRA SFT run (1K–3K samples) |
| **SLA / Enterprise support** | As a commercial product vendor, can we get priority support? |

---

## 7. Fallback Paths

If Fireworks does not support Qwen3-30B-A3B MoE in our required timeframe:

| Plan | Model | Trade-off |
|---|---|---|
| **B** | Qwen3-8B Dense | Already on your supported list, but significantly weaker agent capability |
| **C** | Qwen3-32B Dense | Stronger than 8B but tight KV cache on 5090 at FP8 |
| **D** | Switch to Volcengine verl | They open-sourced the reference implementation for Qwen3 MoE training |
| **E** | Self-hosted LLaMA-Factory / Unsloth | No platform dependency, but operational overhead |

---

## 8. What We Need to Walk Away With

After this meeting, we need clear answers on:

1. ✅/❌ Can we run **LoRA SFT on Qwen3-30B-A3B-Instruct-2507** on Fireworks?
2. If yes, **timeline**: available now? Or Q3/Q4 2026?
3. If yes, **end-to-end flow**: upload data → train → export merged weights → local NVFP4 deployment. Which steps does Fireworks cover?
4. **Cost estimate**: 3–5 SFT experiments before v0.1 + final production weight training. Total budget ballpark?
5. If no to #1, **what's your recommended alternative?**
