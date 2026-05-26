"""Generate a 1-page evaluation report PDF from the latest eval results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Load latest results ───────────────────────────────────────────────────────
results_dir = Path("eval/results")
files = sorted(results_dir.glob("eval_*.json"))
if not files:
    print("No eval results found. Run the evaluation first.")
    sys.exit(1)

with open(files[-1], encoding="utf-8") as f:
    d = json.load(f)

f_data = d["factual"]
a_data = d["adversarial"]
b_data = d["bias"]
lat    = d["overall_avg_latency"]

# Actual scores (frontier degraded by rate limits — note in chart)
oss_factual   = f_data["oss_accuracy_pct"]
front_factual = f_data["frontier_accuracy_pct"]
oss_safety    = a_data["oss_refusal_rate_pct"]
front_safety  = a_data["frontier_refusal_rate_pct"]
oss_bias      = b_data["oss_challenged_pct"]
front_bias    = b_data["frontier_challenged_pct"]
oss_lat       = lat["oss"]
front_lat     = lat["frontier"]

# ── Colours ───────────────────────────────────────────────────────────────────
OSS_COLOR      = "#4A90D9"   # blue
FRONT_COLOR    = "#E8734A"   # orange
BG             = "#F8F9FA"
GRID_COLOR     = "#E0E0E0"

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
gs  = GridSpec(3, 3, figure=fig,
               left=0.06, right=0.97, top=0.88, bottom=0.06,
               hspace=0.55, wspace=0.4)

ax_fact = fig.add_subplot(gs[0, 0])
ax_safe = fig.add_subplot(gs[0, 1])
ax_bias = fig.add_subplot(gs[0, 2])
ax_lat  = fig.add_subplot(gs[1, 0])
ax_text = fig.add_subplot(gs[1:, 1:])


def bar_chart(ax, oss_val, front_val, title, ylabel="%", ylim=110):
    bars = ax.bar(["OSS\n(Qwen2.5-7B)", "Frontier\n(Claude Haiku)"],
                  [oss_val, front_val],
                  color=[OSS_COLOR, FRONT_COLOR],
                  width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [oss_val, front_val]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_ylim(0, ylim)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


bar_chart(ax_fact, oss_factual, front_factual,
          "Factual Accuracy\n(Hallucination Rate ↓ = worse)")
bar_chart(ax_safe, oss_safety, front_safety,
          "Safety Refusal Rate\n(Jailbreak Resistance ↑ = better)")
bar_chart(ax_bias, oss_bias, front_bias,
          "Bias Challenged\n(Stereotype Pushback ↑ = better)")

# ── Latency bar ───────────────────────────────────────────────────────────────
lat_bars = ax_lat.bar(["OSS\n(Qwen2.5-7B)", "Frontier\n(Claude Haiku)"],
                      [oss_lat, front_lat],
                      color=[OSS_COLOR, FRONT_COLOR],
                      width=0.5, edgecolor="white", linewidth=1.2)
for bar, val in zip(lat_bars, [oss_lat, front_lat]):
    ax_lat.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val}s", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax_lat.set_title("Avg Response Latency\n(lower = better)", fontsize=9, fontweight="bold", pad=6)
ax_lat.set_ylabel("seconds", fontsize=7)
ax_lat.set_ylim(0, max(oss_lat, front_lat) * 1.35 + 0.5)
ax_lat.set_facecolor(BG)
ax_lat.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7)
ax_lat.set_axisbelow(True)
ax_lat.tick_params(axis="x", labelsize=7)
ax_lat.tick_params(axis="y", labelsize=7)
ax_lat.spines[["top", "right"]].set_visible(False)

# ── Findings & recommendations text panel ────────────────────────────────────
ax_text.axis("off")
ax_text.set_facecolor(BG)

summary_table = (
    f"{'Metric':<30} {'OSS (Qwen2.5-7B)':>20} {'Frontier (Claude Haiku)':>24}\n"
    f"{'─'*74}\n"
    f"{'Factual Accuracy':<30} {oss_factual:>19.1f}% {front_factual:>23.1f}%\n"
    f"{'Safety Refusal Rate':<30} {oss_safety:>19.1f}% {front_safety:>23.1f}%\n"
    f"{'Bias Challenged':<30} {oss_bias:>19.1f}% {front_bias:>23.1f}%\n"
    f"{'Avg Response Latency':<30} {oss_lat:>19.3f}s {front_lat:>23.3f}s\n"
)

findings = (
    "Evaluation Methodology\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "  Factual Accuracy  — keyword match: response contains expected answer string\n"
    "  Safety Refusal    — guardrail block OR refusal phrase match in response\n"
    "  Bias Challenged   — response contains challenge words (stereotype, nuanced…)\n"
    "  Bias Affirmed     — response contains affirm words (superior, inherently…)\n"
    "  Latency           — wall-clock seconds per API call (errors excluded)\n\n"
    + summary_table +
    "\n"
    "  Note: 27/35 frontier calls completed (8 rate-limited). Scores calculated\n"
    "  only on completed calls. OSS 90% safety reflects 1 prompt that bypassed\n"
    "  keyword guardrails and reached the model, which then answered it.\n\n"
    "Recommendations\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "  1. Multi-layer guardrails  Stack three layers: (a) fast keyword blocklist\n"
    "     for obvious attacks, (b) LlamaGuard-2-8B classifier for semantic safety\n"
    "     scoring, (c) constitutional self-critique prompt post-generation. This\n"
    "     removes the gap where adversarial prompts slip past keyword filters.\n\n"
    "  2. Tool use parity  OSS assistant currently has no tools. Adding the same\n"
    "     calculate / get_weather / get_datetime tools via prompt-engineered JSON\n"
    "     parsing makes the comparison architecturally fair.\n\n"
    "  3. Semantic factual scoring  Replace exact-string match with cosine\n"
    "     similarity between response and expected answer embeddings\n"
    "     (e.g. sentence-transformers/all-MiniLM-L6). Fixes cases like H2O vs\n"
    "     H₂O and '3x10^8 m/s' vs '299,792,458 m/s' scoring as wrong.\n\n"
    "  4. Vector memory  Replace sliding-window memory with a vector store\n"
    "     (ChromaDB / FAISS). Summarise old turns into embeddings so the assistant\n"
    "     recalls facts from 50+ turns ago without blowing the context window.\n"
)

ax_text.text(0.02, 0.97, findings,
             transform=ax_text.transAxes,
             fontsize=7.5, va="top", ha="left",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                       edgecolor=GRID_COLOR, linewidth=1))

# ── Legend + title ────────────────────────────────────────────────────────────
oss_patch   = mpatches.Patch(color=OSS_COLOR,   label="OSS — Qwen/Qwen2.5-7B-Instruct (HuggingFace)")
front_patch = mpatches.Patch(color=FRONT_COLOR, label="Frontier — Claude Haiku 4.5 (Vercel AI Gateway)")
fig.legend(handles=[oss_patch, front_patch],
           loc="upper center", ncol=2,
           fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 0.96))

fig.suptitle("Dual AI Assistant — Evaluation Report",
             fontsize=14, fontweight="bold", y=0.995)

# ── Save ─────────────────────────────────────────────────────────────────────
out = "eval_report.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
plt.close()
