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
# edit .env and add your keys
```

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens |
| `ANTHROPIC_API_KEY` | Vercel dashboard → Settings → Tokens |

### 3. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Architecture

```
dual_agent/
├── app.py                        # Streamlit UI (5 tabs)
├── assistants/
│   ├── memory.py                 # Sliding-window ConversationMemory
│   ├── guardrails.py             # Input/output safety filters
│   ├── tools.py                  # Tool definitions (Gemini + OpenAI formats) + handlers
│   ├── oss_assistant.py          # Qwen2.5-7B via HF Inference API
│   └── frontier_assistant.py    # Claude Haiku 4.5 via Vercel AI Gateway
├── eval/
│   ├── prompts.py                # 35 test prompts (factual / adversarial / bias)
│   └── evaluator.py             # Scoring + report generation
├── observability/
│   └── logger.py                 # JSONL interaction logging
└── logs/                         # Auto-created at runtime
```

### Key design decisions

**HuggingFace Inference API for OSS** — No local GPU required. Qwen2.5-7B runs on HF's serverless inference tier, accessible with just a free token. The 7B model was chosen over the 0.5B for significantly better response quality while remaining free.

**Vercel AI Gateway for Frontier** — Acts as an OpenAI-compatible proxy to Anthropic, so `frontier_assistant.py` uses the standard `openai` Python SDK with a custom `base_url`. This avoids a hard dependency on any single provider SDK and makes swapping models trivial.

**OpenAI-style tool calling for Frontier** — Function calling is defined in OpenAI JSON schema format (`OPENAI_TOOLS`) and executed via the standard `tool_calls` / `tool` message pattern. The gateway handles translation to Anthropic's native tool API internally.

**Guardrails as a shared decorator layer** — Both assistants pass every user message through `check_input_safety()` before the model and every response through `check_output_safety()`. Implemented once, applied to both models with zero model-specific changes.

**Sliding-window memory** — Keeps the last 10 user+assistant turns. Both assistants reconstruct the full `messages` list on each call from the shared `ConversationMemory` object.

**Evaluation** — 3 categories × 2 models: 15 factual prompts, 10 adversarial/jailbreak, 10 bias-related. Results saved as timestamped JSON in `eval/results/`.

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Qwen2.5-7B via HF serverless | Good quality for free, but subject to rate limits and cold-start delays |
| Vercel AI Gateway | Provider-agnostic and easy to swap models; adds one extra network hop |
| Keyword-based guardrails | Fast and transparent, but less robust than a trained classifier |
| Sliding-window memory | Simple O(1) implementation; loses context beyond 10 turns |
| Eval scored locally (no LLM judge) | Deterministic and free, but coarser accuracy on nuanced outputs |

---

## What I'd improve with more time

1. **Deploy OSS model publicly** — Host Qwen2.5-7B on HuggingFace Spaces or Modal for a stable public endpoint with a cost + latency table.
2. **LLM-as-judge evaluation** — Use Claude to score each response on a 1–5 rubric instead of keyword matching for much more accurate hallucination/bias measurement.
3. **Trained safety classifier** — Replace keyword guardrails with `meta-llama/LlamaGuard-2-8b` for higher recall on adversarial inputs.
4. **Persistent memory across sessions** — Store conversation summaries in a vector DB (e.g., ChromaDB) for long-term recall beyond the current window.
5. **More tools for OSS** — OSS assistant currently has no tool use; adding the same `calculate` / `get_weather` / `get_current_datetime` tools via prompt-engineering would make the comparison fairer.
6. **Streaming responses** — Both APIs support streaming; adding it would make the UI feel significantly faster.
7. **Evaluation infographics** — Auto-generate matplotlib bar charts from eval JSON results and embed them in the evaluation report.

---

## Evaluation categories

| Category | # Prompts | What it measures |
|---|---|---|
| Factual | 15 | Hallucination rate — checks known facts against expected keywords |
| Adversarial | 10 | Safety refusal rate — jailbreak and harmful instruction resistance |
| Bias | 10 | Bias handling — whether the model challenges or affirms stereotypes |

Run the evaluation from the **Evaluation** tab in the UI, or programmatically:

```python
from eval.evaluator import Evaluator
evaluator = Evaluator(oss_assistant, frontier_assistant)
summary = evaluator.run_full_eval()
```
