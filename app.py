"""
Dual AI Assistant — Streamlit app
  Tab 1: OSS Assistant  (Mistral-7B-Instruct via HuggingFace Inference API)
  Tab 2: Frontier       (Gemini 2.0 Flash with tool use)
  Tab 3: Side-by-side comparison
  Tab 4: Automated evaluation runner
  Tab 5: Observability / stats
"""
import os
import threading
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

st.set_page_config(page_title="Dual AI Assistant", layout="wide")

# ── Global eval state (threading-safe: written from bg thread, read on refresh) ──
_eval_log: list[str] = []
_eval_running = False


# ── Lazy-init singletons (one instance per app process) ──────────────────────

@st.cache_resource
def get_oss():
    if not HF_TOKEN:
        return None
    from assistants.oss_assistant import OSSAssistant
    return OSSAssistant(hf_token=HF_TOKEN)


@st.cache_resource
def get_frontier():
    if not ANTHROPIC_API_KEY:
        return None
    from assistants.frontier_assistant import FrontierAssistant
    return FrontierAssistant(api_key=ANTHROPIC_API_KEY)


@st.cache_resource
def get_logger():
    from observability.logger import ObservabilityLogger
    return ObservabilityLogger()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_meta(result: dict) -> str:
    parts = [f"⏱ {result['latency']:.2f}s", result.get("memory_summary", "")]
    if result.get("tool_used"):
        parts.append(f"🔧 tool: {result['tool_used']}")
    if result.get("blocked"):
        parts.append("🚫 blocked by guardrail")
    return "  |  ".join(p for p in parts if p)


def _no_key_msg(key_name: str) -> str:
    return f"⚠️ {key_name} not set in .env — cannot respond."


def _render_history(history: list[dict]):
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Dual AI Assistant Comparison")
st.markdown(
    "**OSS:** `Qwen/Qwen2.5-7B-Instruct` (HuggingFace Inference API)"
    "**|**  **Frontier:** `Claude Haiku 4.5` (Vercel AI Gateway) — tool use, guardrails & memory"
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "OSS Assistant (Qwen2.5)",
    "Frontier Assistant (Claude Haiku)",
    "Side-by-Side",
    "Evaluation",
    "Observability",
])

# ── Tab 1: OSS ────────────────────────────────────────────────────────────────

with tab1:
    if "oss_history" not in st.session_state:
        st.session_state.oss_history = []

    st.subheader("Qwen2.5-7B-Instruct")

    with st.container(height=320):
        _render_history(st.session_state.oss_history)

    if st.button("Clear Memory", key="oss_clear"):
        oss = get_oss()
        if oss:
            oss.clear_memory()
        st.session_state.oss_history = []
        st.rerun()

    if prompt := st.chat_input("Ask anything…", key="oss_input"):
        oss = get_oss()
        if not oss:
            reply, meta = _no_key_msg("HF_TOKEN"), ""
        else:
            result = oss.chat(prompt)
            get_logger().log(
                model=result["model"],
                user_message=prompt,
                assistant_response=result["response"],
                latency=result["latency"],
                blocked=result.get("blocked", False),
            )
            reply = result["response"]
            meta = _format_meta(result)

        st.session_state.oss_history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{reply}\n\n*{meta}*" if meta else reply},
        ]
        st.rerun()

# ── Tab 2: Frontier ───────────────────────────────────────────────────────────

with tab2:
    if "front_history" not in st.session_state:
        st.session_state.front_history = []

    st.subheader("Claude Haiku 4.5")
    st.caption("Tools: `get_current_datetime` · `calculate(expr)` · `get_weather(city)`")

    with st.container(height=320):
        _render_history(st.session_state.front_history)

    if st.button("Clear Memory", key="front_clear"):
        frontier = get_frontier()
        if frontier:
            frontier.clear_memory()
        st.session_state.front_history = []
        st.rerun()

    if prompt := st.chat_input(
        "Try: 'What time is it?' / 'Calculate 2**32' / 'Weather in Tokyo'…",
        key="front_input",
    ):
        frontier = get_frontier()
        if not frontier:
            reply, meta = _no_key_msg("ANTHROPIC_API_KEY"), ""
        else:
            result = frontier.chat(prompt)
            get_logger().log(
                model=result["model"],
                user_message=prompt,
                assistant_response=result["response"],
                latency=result["latency"],
                blocked=result.get("blocked", False),
                tool_used=result.get("tool_used"),
            )
            reply = result["response"]
            meta = _format_meta(result)

        st.session_state.front_history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{reply}\n\n*{meta}*" if meta else reply},
        ]
        st.rerun()

# ── Tab 3: Side-by-Side ───────────────────────────────────────────────────────

with tab3:
    if "cmp_oss_history" not in st.session_state:
        st.session_state.cmp_oss_history = []
    if "cmp_front_history" not in st.session_state:
        st.session_state.cmp_front_history = []

    col_oss, col_front = st.columns(2)
    with col_oss:
        st.subheader("Qwen2.5-7B (OSS)")
        with st.container(height=300):
            _render_history(st.session_state.cmp_oss_history)
    with col_front:
        st.subheader("Claude Haiku (Frontier)")
        with st.container(height=300):
            _render_history(st.session_state.cmp_front_history)

    if st.button("Clear Both", key="cmp_clear"):
        oss = get_oss()
        frontier = get_frontier()
        if oss:
            oss.clear_memory()
        if frontier:
            frontier.clear_memory()
        st.session_state.cmp_oss_history = []
        st.session_state.cmp_front_history = []
        st.rerun()

    if prompt := st.chat_input("Send to both at once…", key="cmp_input"):
        oss = get_oss()
        frontier = get_frontier()
        logger = get_logger()

        oss_r = (
            oss.chat(prompt) if oss
            else {"response": _no_key_msg("HF_TOKEN"), "latency": 0, "model": "oss"}
        )
        front_r = (
            frontier.chat(prompt) if frontier
            else {"response": _no_key_msg("ANTHROPIC_API_KEY"), "latency": 0, "model": "claude-haiku"}
        )

        if oss:
            logger.log(model=oss_r["model"], user_message=prompt,
                       assistant_response=oss_r["response"], latency=oss_r["latency"],
                       blocked=oss_r.get("blocked", False))
        if frontier:
            logger.log(model=front_r["model"], user_message=prompt,
                       assistant_response=front_r["response"], latency=front_r["latency"],
                       blocked=front_r.get("blocked", False), tool_used=front_r.get("tool_used"))

        oss_meta = _format_meta(oss_r)
        front_meta = _format_meta(front_r)

        st.session_state.cmp_oss_history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{oss_r['response']}\n\n*{oss_meta}*" if oss_meta else oss_r["response"]},
        ]
        st.session_state.cmp_front_history += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{front_r['response']}\n\n*{front_meta}*" if front_meta else front_r["response"]},
        ]
        st.rerun()

# ── Tab 4: Evaluation ─────────────────────────────────────────────────────────

with tab4:
    st.subheader("Automated Evaluation")
    st.markdown(
        "Runs **15 factual**, **10 adversarial**, and **10 bias** prompts through both models.  \n"
        "Results are saved to `eval/results/`."
    )

    col_run, col_refresh = st.columns([2, 1])
    with col_run:
        run_btn = st.button("Run Full Evaluation", type="primary", disabled=_eval_running)
    with col_refresh:
        st.button("Refresh Log", key="eval_refresh")

    if run_btn and not _eval_running:
        def _run_eval_thread():
            global _eval_running, _eval_log
            _eval_running = True
            _eval_log = ["Starting evaluation…"]

            oss = get_oss()
            frontier = get_frontier()
            if not oss or not frontier:
                _eval_log.append("⚠️ Both HF_TOKEN and GEMINI_API_KEY are required.")
                _eval_running = False
                return

            from eval.evaluator import Evaluator
            evaluator = Evaluator(oss, frontier, progress_cb=lambda m: _eval_log.append(m))
            summary = evaluator.run_full_eval()

            f = summary["factual"]
            a = summary["adversarial"]
            b = summary["bias"]

            _eval_log += [
                "\n## Results\n",
                f"**Factual Accuracy** — OSS: {f['oss_accuracy_pct']}% | Frontier: {f['frontier_accuracy_pct']}%",
                f"**Safety Refusal Rate** — OSS: {a['oss_refusal_rate_pct']}% | Frontier: {a['frontier_refusal_rate_pct']}%",
                f"**Bias Challenged** — OSS: {b['oss_challenged_pct']}% | Frontier: {b['frontier_challenged_pct']}%",
                f"**Bias Affirmed** — OSS: {b['oss_affirmed_pct']}% | Frontier: {b['frontier_affirmed_pct']}%",
                f"**Avg Latency** — OSS: {summary['overall_avg_latency']['oss']}s | "
                f"Frontier: {summary['overall_avg_latency']['frontier']}s",
            ]
            _eval_running = False

        threading.Thread(target=_run_eval_thread, daemon=True).start()
        st.rerun()

    log_text = "\n\n".join(_eval_log) if _eval_log else "No evaluation running. Click **Run Full Evaluation** to start."
    st.markdown(log_text)

# ── Tab 5: Observability ──────────────────────────────────────────────────────

with tab5:
    st.subheader("Live Interaction Stats")

    if st.button("Refresh Stats"):
        stats = get_logger().get_stats()
        if not stats:
            st.info("No interactions logged yet. Chat with the assistants first.")
        else:
            for model, s in stats.items():
                with st.expander(f"`{model}`", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Interactions", s["total_interactions"])
                    c2.metric("Avg Latency (s)", s["avg_latency_s"])
                    c3.metric("Blocked", s["blocked_count"])
                    c4.metric("Tool Calls", s["tool_calls"])
