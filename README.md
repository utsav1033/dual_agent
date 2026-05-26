# Dual AI Personal Assistant

Compare two AI assistants side-by-side:

| | OSS | Frontier |
|---|---|---|
| **Model** | `Qwen/Qwen2.5-0.5B-Instruct` | `Gemini 2.0 Flash` |
| **Provider** | HuggingFace Inference API | Google AI Studio |
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
# then edit .env and add your keys
```

| Variable | Where to get it |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |

### 3. Run the app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Architecture

```
dual_agent/
├── app.py                        # Gradio UI (5 tabs)
├── assistants/
│   ├── memory.py                 # Sliding-window ConversationMemory
│   ├── guardrails.py             # Input/output safety filters
│   ├── tools.py                  # Tool definitions + handlers (Gemini)
│   ├── oss_assistant.py          # Qwen2.5 via HF Inference API
│   └── frontier_assistant.py    # Gemini 2.0 Flash with function calling
├── eval/
│   ├── prompts.py                # 35 test prompts (factual/adversarial/bias)
│   └── evaluator.py             # Scoring + report generation
├── observability/
│   └── logger.py                 # JSONL interaction logging
└── logs/                         # Auto-created at runtime
```

### Key design decisions

**HuggingFace Inference API for OSS** — No local GPU required. The 0.5B model runs on HF's free serverless inference tier, making it accessible with just a token.

**Gemini function calling** — The Gemini SDK's native tool-use API is used instead of prompt-engineering hacks, giving reliable structured tool invocation.

**Guardrails as a decorator layer** — Both assistants pass every user message through `check_input_safety()` before it reaches the model, and every model response through `check_output_safety()`. This works for both models with zero model-specific changes.

**Sliding-window memory** — Keeps the last 10 user+assistant pairs. The OSS assistant reconstructs the full messages list on each call; Gemini uses its native `ChatSession` which maintains history internally.

**Evaluation** — 3 categories × 2 models, run sequentially with 0.5 s rate-limit pauses. Results saved as timestamped JSON in `eval/results/`.

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| Qwen2.5-0.5B (tiny model) | Fast and free, but lower quality than larger OSS models |
| HF serverless inference | No GPU cost, but subject to rate limits and cold-start delays |
| Keyword-based guardrails | Fast and transparent, but less robust than a trained classifier |
| Sliding-window memory | Simple O(1) implementation, but loses context beyond 10 turns |
| Eval scored locally (no LLM judge) | Deterministic and free, but coarser accuracy on nuanced outputs |

---

## What I'd improve with more time

1. **Larger OSS model** — Deploy Qwen2.5-7B or Llama-3.2-3B on Modal/RunPod for far better quality.
2. **LLM-as-judge evaluation** — Use Gemini to score each response on a 1–5 rubric instead of keyword matching.
3. **Trained safety classifier** — Replace keyword guardrails with `meta-llama/LlamaGuard-2-8b` for higher recall.
4. **Persistent memory across sessions** — Store conversation summaries in a vector DB (e.g., ChromaDB) for long-term recall.
5. **More tools** — Web search, calendar access, code execution sandbox.
6. **Evaluation infographics** — Auto-generate matplotlib bar charts from eval JSON and embed in the report.
7. **Streaming responses** — Both APIs support streaming; adding it would make the UI feel much faster.

---

## Deployment (HuggingFace Spaces)

To deploy the OSS model publicly:

1. Create a new Space at https://huggingface.co/spaces (select Gradio SDK).
2. Push this repo as-is.
3. Add `HF_TOKEN` and `GEMINI_API_KEY` as Space secrets.
4. The `app.py` entry point is already Spaces-compatible.

The OSS model uses HF's serverless inference so no GPU tier is needed on the Space itself.
