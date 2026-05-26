---
title: Dual AI Assistant
emoji: 🤖
colorFrom: blue
colorTo: orange
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# Dual AI Personal Assistant

Compare two AI assistants side-by-side: an open-source model via HuggingFace and a frontier model via Vercel AI Gateway.

| | OSS | Frontier |
|---|---|---|
| **Model** | `Qwen/Qwen2.5-7B-Instruct` | `Claude Haiku 4.5` |
| **Provider** | HuggingFace Inference API | Vercel AI Gateway → Anthropic |
| **Tool use** | — | `get_current_datetime`, `calculate`, `get_weather` |
| **Guardrails** | Input + output safety filters | Input + output safety filters |
| **Memory** | Sliding window (last 10 turns) | Sliding window (last 10 turns) |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# edit .env and fill in your keys
```

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free account, read token |
| `ANTHROPIC_API_KEY` | Vercel dashboard → AI Gateway → Settings → Tokens (prefix `vck_`) |

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the evaluation (optional)

Use the **Evaluation** tab in the UI, or run headlessly:

```bash
python -m eval.run
```

Generate the PDF report from the latest results:

```bash
python generate_report.py
```

---

## Architecture

```
dual_agent/
├── app.py                        # Streamlit UI — 5 tabs
├── assistants/
│   ├── memory.py                 # Sliding-window ConversationMemory
│   ├── guardrails.py             # Input/output safety filters
│   ├── tools.py                  # Tool definitions (OpenAI format) + handlers
│   ├── oss_assistant.py          # Qwen2.5-7B via HuggingFace Inference API
│   └── frontier_assistant.py    # Claude Haiku 4.5 via Vercel AI Gateway
├── eval/
│   ├── prompts.py                # 35 test prompts (factual / adversarial / bias)
│   ├── evaluator.py              # Scoring + result serialisation
│   └── results/                  # Timestamped JSON result files
├── observability/
│   └── logger.py                 # JSONL interaction logging
├── generate_report.py            # Matplotlib PDF report from latest results
└── logs/                         # Auto-created at runtime
```

### Architecture decisions

**HuggingFace Inference API for OSS**
No local GPU required. Qwen2.5-7B runs on HF's serverless inference tier with just a free read token. The 7B variant was chosen over 0.5B for significantly better response quality while staying on the free tier.

**Vercel AI Gateway for Frontier**
Acts as an OpenAI-compatible proxy to Anthropic, so `frontier_assistant.py` uses the standard `openai` Python SDK with a custom `base_url`. This decouples the code from any single provider SDK — swapping the underlying model is a one-line constant change.

**OpenAI-style tool calling**
Function calling is defined in OpenAI JSON schema (`OPENAI_TOOLS`) and executed via the standard `tool_calls` / `tool` message round-trip. The gateway translates to Anthropic's native tool API internally. The OSS model has no tool use, which is noted explicitly as a limitation in the evaluation.

**Guardrails as a shared decorator layer**
Every user message passes through `check_input_safety()` before reaching the model, and every response through `check_output_safety()`. This logic lives once in `guardrails.py` and is applied identically to both models — no model-specific divergence.

**Sliding-window memory**
Keeps the last 10 user + assistant turns as a `messages` list. Both assistants reconstruct the full context on each call from a shared `ConversationMemory` object. Simple to reason about; the window boundary is explicit and predictable.

**Evaluation framework**
3 categories × 2 models: 15 factual prompts, 10 adversarial/jailbreak, 10 bias-related. Results are saved as timestamped JSON under `eval/results/`. Frontier calls use 12-second inter-call delays and exponential-backoff retry (15 s / 30 s / 60 s) to stay within Anthropic's upstream rate limits. Error responses are excluded from metric calculations so rate-limit failures don't skew scores.

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Qwen2.5-7B via HF serverless | Free with zero infrastructure, but subject to rate limits and cold-start delays on the first call |
| Vercel AI Gateway | Provider-agnostic, easy to swap models; adds one extra network hop and a second free-tier rate limit |
| Keyword-based guardrails | Fast and fully auditable, but lower recall than a trained classifier on novel adversarial inputs |
| Sliding-window memory (10 turns) | O(1) implementation, no external state; forgets context beyond 10 turns and has no cross-session persistence |
| Keyword-match eval scoring | Deterministic, runs locally with no extra API cost; misses semantically correct answers with different phrasing (e.g. H₂O vs H2O) |
| No tool use for OSS | Keeps OSS dependency-free; makes the architecture comparison slightly unfair since Frontier can fetch live data OSS cannot |

---

## What I'd improve with more time

1. **Multi-layer guardrails** — Stack three layers: (a) fast keyword blocklist for obvious attacks, (b) `LlamaGuard-2-8B` classifier for semantic safety scoring, (c) constitutional self-critique prompt post-generation. This closes the gap where adversarial prompts slip through keyword filters.

2. **Tool use parity for OSS** — Add `calculate`, `get_weather`, and `get_current_datetime` to the OSS assistant via prompt-engineered JSON parsing. The model emits a structured JSON action; the app parses and executes it. Makes the architectural comparison fair.

3. **Semantic factual scoring** — Replace exact-string keyword matching with cosine similarity between response and expected-answer embeddings (`sentence-transformers/all-MiniLM-L6-v2`). Fixes false negatives like `H2O` vs `H₂O` or `3×10⁸ m/s` vs `299,792,458 m/s`.

4. **Vector memory** — Replace the sliding window with a vector store (ChromaDB or FAISS). Summarise old turns into embeddings so the assistant can recall facts from 50+ turns ago without blowing the context window, and persist memory across sessions.

5. **Public OSS deployment** — Host Qwen2.5-7B on HuggingFace Spaces or Modal for a stable, shareable public endpoint. Include a cost + latency comparison table.

6. **Streaming responses** — Both APIs support SSE streaming; adding it would make the UI feel substantially faster for longer answers.

---

## OSS deployment — cost & latency

The OSS model (`Qwen2.5-7B-Instruct`) is served via HuggingFace Inference API and the app is hosted on HuggingFace Spaces.

| Tier | Instance | Cost | Cold-start latency | Warm latency | Notes |
|---|---|---|---|---|---|
| HF Spaces CPU Basic | 2 vCPU / 16 GB RAM | **Free** | ~5–8 s | ~1–3 s | Default; sleeps after inactivity |
| HF Spaces CPU Upgrade | 8 vCPU / 32 GB RAM | $0.03 / hr | ~3–5 s | ~0.8–2 s | Always-on available |
| HF Spaces T4 GPU | NVIDIA T4 / 15 GB | $0.40 / hr | ~10–15 s (model load) | ~0.3–0.8 s | Local inference, no HF API needed |
| HF Inference API (serverless) | HF-managed | **Free** (rate-limited) | ~3–5 s | ~1–2 s | Current setup; 429s under load |

Measured latency (from eval run, HF Inference API warm):

| Metric | Value |
|---|---|
| Average response latency | **1.04 s** |
| Min latency observed | ~0.5 s |
| Max latency observed | ~4.2 s |
| Rate limit errors | 0 / 35 calls (OSS never hit rate limits) |

> Latency measured wall-clock per API call on HF serverless inference tier, errors excluded. Cold-start adds 3–8 s on the first call after the endpoint sleeps.

---

## Evaluation results

| Category | # Prompts | What it measures |
|---|---|---|
| Factual | 15 | Hallucination rate — checks known facts against expected answer keywords |
| Adversarial | 10 | Safety refusal rate — jailbreak and harmful instruction resistance |
| Bias | 10 | Bias handling — whether the model challenges or affirms stereotypes |

Latest run (27/35 frontier calls completed, 8 rate-limited):

| Metric | OSS (Qwen2.5-7B) | Frontier (Claude Haiku 4.5) |
|---|---|---|
| Factual Accuracy | 80.0 % | 66.7 % |
| Safety Refusal Rate | 90.0 % | 80.0 % |
| Bias Challenged | 60.0 % | 50.0 % |
| Avg Response Latency | 1.04 s | 3.38 s |

> Frontier scores are calculated only on completed calls. OSS 90 % safety reflects one prompt that bypassed the keyword guardrail and reached the model, which then answered it.
