"""
app.py — Medical Diagnosis Assistant (Gradio UI)

Run:
    pip install gradio scikit-learn pandas numpy
    python app.py
"""

import gradio as gr
from ml_model import MedicalDiagnosisModel

# ── Bootstrap model (loads once at startup) ────────────────────────────────────
model = MedicalDiagnosisModel()
model.train()          # reads Training.csv & Diseases_Symptoms.csv


# ── Chat logic ─────────────────────────────────────────────────────────────────
GREETING = (
    "👋 Hello! I'm **MedAI**, an educational symptom-checker powered by "
    "a Random Forest classifier.\n\n"
    "Type your symptoms separated by commas — e.g. `fever, headache, nausea` — "
    "and I'll suggest the most likely conditions.\n\n"
    "⚕️ *Remember: this tool is for educational use only. Always consult a "
    "licensed medical professional for real diagnosis and treatment.*"
)

def _format_confidence_bar(probability: float, confidence: str) -> str:
    filled  = round(probability / 10)
    bar     = "█" * filled + "░" * (10 - filled)
    emoji   = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}[confidence]
    return f"{emoji} `{bar}` **{probability}%** confidence"


def _build_result_markdown(result: dict) -> str:
    lines = []

    # ── Matched / unrecognised symptoms ───────────────────────────────────────
    if result["matched_symptoms"]:
        syms = ", ".join(f"`{s}`" for s in result["matched_symptoms"])
        lines.append(f"✅ **Recognised symptoms:** {syms}")

    if result["fuzzy_suggestions"]:
        for raw, cands in result["fuzzy_suggestions"].items():
            lines.append(
                f"🔍 *'{raw}'* — did you mean: "
                + ", ".join(f"`{c}`" for c in cands) + "?"
            )

    if result["unrecognised"]:
        unk = ", ".join(f"`{u}`" for u in result["unrecognised"])
        lines.append(f"⚠️ **Not recognised:** {unk} — check spelling or try synonyms.")

    lines.append("")  # blank line

    # ── Top-3 predictions ─────────────────────────────────────────────────────
    if not result["top3"]:
        lines.append("❌ No significant prediction could be made. "
                     "Try adding more specific symptoms.")
        return "\n".join(lines)

    uncertainty = result["uncertainty"]
    unc_emoji   = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}[uncertainty]
    lines.append(f"### 🩺 Top Predictions  ·  Uncertainty: {unc_emoji} {uncertainty}")
    lines.append("")

    for i, pred in enumerate(result["top3"], 1):
        bar = _format_confidence_bar(pred["probability"], pred["confidence"])
        lines += [
            f"#### {i}. {pred['disease']}",
            bar,
            f"> 💊 **Preliminary Treatment:** {pred['treatment']}",
            "",
        ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(
        f"---\n"
        f"🤖 *Model CV accuracy: {result['model_cv_accuracy']}%* · "
        f"⚕️ *This is NOT a medical diagnosis. Consult a doctor.*"
    )
    return "\n".join(lines)


def respond(message: str, history: list) -> tuple:
    """Gradio chat callback."""
    message = message.strip()
    if not message:
        return history, ""

    # Parse comma-separated symptoms
    raw_symptoms = [s.strip() for s in message.replace(";", ",").split(",") if s.strip()]

    if not raw_symptoms:
        reply = "Please enter at least one symptom, e.g. `fever, cough`."
    else:
        result = model.predict(raw_symptoms)
        reply  = _build_result_markdown(result)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def autocomplete_symptoms(partial: str) -> gr.update:
    suggestions = model.autocomplete(partial, max_results=10)
    choices = [s.replace("_", " ") for s in suggestions]
    return gr.update(choices=choices, visible=bool(choices))


def insert_suggestion(selected: str, current: str) -> tuple:
    """Append the selected suggestion to the current input."""
    parts   = [p.strip() for p in current.split(",") if p.strip()]
    keyword = selected.replace(" ", "_")
    if keyword not in parts:
        parts.append(keyword)
    new_val = ", ".join(parts)
    return new_val, gr.update(visible=False)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;1,9..144,300&display=swap');

/* ── Root palette ──────────────────────────────────────────────────────── */
:root {
  --bg-page      : #080d18;
  --bg-panel     : #0e1525;
  --bg-card      : #141d2f;
  --bg-input     : #1a253a;
  --border       : #1f2f4a;
  --accent       : #3eefd0;
  --accent-dim   : #1f8c7a;
  --accent2      : #7b8cde;
  --danger       : #f4636a;
  --warn         : #f5a623;
  --text-primary : #d4e0ef;
  --text-muted   : #6a7d96;
  --radius       : 14px;
  --radius-sm    : 8px;
}

/* ── Page ──────────────────────────────────────────────────────────────── */
body, .gradio-container {
  background : var(--bg-page) !important;
  font-family: 'DM Sans', sans-serif !important;
  color      : var(--text-primary) !important;
}

/* ── Header ────────────────────────────────────────────────────────────── */
#header {
  text-align : center;
  padding    : 36px 16px 8px;
}
#header .title {
  font-family : 'Fraunces', serif;
  font-size   : 2.6rem;
  font-weight : 300;
  letter-spacing: -0.5px;
  color       : var(--accent);
  line-height : 1.1;
  margin      : 0;
}
#header .subtitle {
  font-size  : 0.92rem;
  color      : var(--text-muted);
  margin-top : 6px;
}

/* ── Disclaimer banner ─────────────────────────────────────────────────── */
#disclaimer {
  background : linear-gradient(135deg, rgba(244,99,106,.12), rgba(245,166,35,.08));
  border     : 1px solid rgba(244,99,106,.35);
  border-radius: var(--radius);
  padding    : 12px 20px;
  font-size  : 0.83rem;
  color      : #f4b8bb;
  text-align : center;
  margin     : 0 0 6px;
}

/* ── Chat window ───────────────────────────────────────────────────────── */
#chatbox {
  background    : var(--bg-panel) !important;
  border        : 1px solid var(--border) !important;
  border-radius : var(--radius) !important;
  padding       : 10px !important;
  min-height    : 480px !important;
}

/* User bubble */
#chatbox .message.user {
  background    : linear-gradient(135deg, #1e3a5f, #1a2e4f) !important;
  border        : 1px solid #274870 !important;
  border-radius : 16px 4px 16px 16px !important;
  color         : #c7d9f0 !important;
  font-size     : 0.94rem !important;
  padding       : 12px 16px !important;
}

/* Bot bubble */
#chatbox .message.bot {
  background    : var(--bg-card) !important;
  border        : 1px solid var(--border) !important;
  border-radius : 4px 16px 16px 16px !important;
  color         : var(--text-primary) !important;
  font-size     : 0.92rem !important;
  padding       : 14px 18px !important;
  line-height   : 1.65 !important;
}

/* Markdown inside bot bubble */
#chatbox .message.bot h3 {
  font-family : 'Fraunces', serif;
  font-weight : 300;
  font-size   : 1.05rem;
  color       : var(--accent);
  margin      : 14px 0 6px;
}
#chatbox .message.bot h4 {
  font-size  : 0.94rem;
  font-weight: 600;
  color      : var(--accent2);
  margin     : 10px 0 4px;
}
#chatbox .message.bot code {
  font-family : 'DM Mono', monospace;
  background  : rgba(62,239,208,.08);
  color       : var(--accent);
  padding     : 1px 6px;
  border-radius: 4px;
  font-size   : 0.88em;
}
#chatbox .message.bot blockquote {
  border-left : 3px solid var(--accent-dim);
  margin      : 6px 0;
  padding     : 6px 12px;
  color       : #9bb3cc;
  font-size   : 0.88rem;
}
#chatbox .message.bot hr {
  border-color: var(--border);
  margin      : 12px 0 8px;
}

/* ── Input row ─────────────────────────────────────────────────────────── */
#input-row {
  display    : flex;
  gap        : 10px;
  margin-top : 10px;
}

#symptom-input textarea {
  background    : var(--bg-input) !important;
  border        : 1px solid var(--border) !important;
  border-radius : var(--radius-sm) !important;
  color         : var(--text-primary) !important;
  font-family   : 'DM Mono', monospace !important;
  font-size     : 0.91rem !important;
  padding       : 12px 16px !important;
  resize        : none !important;
  transition    : border-color .2s;
}
#symptom-input textarea:focus {
  border-color : var(--accent) !important;
  outline      : none !important;
}
#symptom-input textarea::placeholder { color: var(--text-muted) !important; }

/* ── Buttons ───────────────────────────────────────────────────────────── */
#send-btn {
  background    : linear-gradient(135deg, var(--accent-dim), var(--accent)) !important;
  border        : none !important;
  border-radius : var(--radius-sm) !important;
  color         : #04120e !important;
  font-weight   : 600 !important;
  font-size     : 0.9rem !important;
  padding       : 0 26px !important;
  cursor        : pointer !important;
  transition    : opacity .2s, transform .15s;
  white-space   : nowrap;
}
#send-btn:hover   { opacity:.88; transform: translateY(-1px); }
#send-btn:active  { transform: translateY(0); }

#clear-btn {
  background   : transparent !important;
  border       : 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color        : var(--text-muted) !important;
  font-size    : 0.88rem !important;
  padding      : 0 18px !important;
  cursor       : pointer !important;
  transition   : border-color .2s, color .2s;
}
#clear-btn:hover { border-color: var(--danger) !important; color: var(--danger) !important; }

/* ── Autocomplete dropdown ─────────────────────────────────────────────── */
#autocomplete-box {
  background    : var(--bg-card) !important;
  border        : 1px solid var(--border) !important;
  border-radius : var(--radius-sm) !important;
  padding       : 8px !important;
}
#autocomplete-box label { color: var(--text-muted) !important; font-size:.8rem !important; }
#autocomplete-box .wrap { gap: 6px !important; }
#autocomplete-box button {
  background    : var(--bg-input) !important;
  border        : 1px solid var(--border) !important;
  border-radius : 20px !important;
  color         : var(--text-primary) !important;
  font-size     : 0.82rem !important;
  padding       : 4px 12px !important;
  font-family   : 'DM Mono', monospace !important;
  transition    : border-color .15s, background .15s;
}
#autocomplete-box button:hover {
  background   : rgba(62,239,208,.12) !important;
  border-color : var(--accent) !important;
  color        : var(--accent) !important;
}

/* ── Stats bar ─────────────────────────────────────────────────────────── */
#stats-bar {
  background : var(--bg-card);
  border     : 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding    : 10px 18px;
  display    : flex;
  gap        : 24px;
  font-size  : 0.8rem;
  color      : var(--text-muted);
  flex-wrap  : wrap;
}
#stats-bar span { color: var(--accent); font-family: 'DM Mono', monospace; }

/* ── Examples ──────────────────────────────────────────────────────────── */
#examples-box { margin-top: 6px; }
#examples-box .label { color: var(--text-muted) !important; font-size:.78rem !important; }
#examples-box button {
  background   : transparent !important;
  border       : 1px dashed var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color        : var(--text-muted) !important;
  font-size    : 0.8rem !important;
  padding      : 4px 14px !important;
  transition   : all .15s;
}
#examples-box button:hover {
  border-color : var(--accent2) !important;
  color        : var(--accent2) !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
"""


# ── Gradio UI ──────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    n_symptoms  = len(model.symptoms)
    n_diseases  = len(model.diseases)
    cv_accuracy = model.cv_score

    with gr.Blocks(title="MedAI — Symptom Checker") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="header">
          <p class="title">⚕ MedAI</p>
          <p class="subtitle">AI-Powered Symptom Checker · Random Forest Classifier</p>
        </div>
        """)

        # ── Disclaimer ───────────────────────────────────────────────────────
        gr.HTML("""
        <div id="disclaimer">
          🚨 <strong>MEDICAL DISCLAIMER</strong> — This tool is strictly for
          <em>educational and informational purposes only</em>. It does <strong>NOT</strong>
          constitute medical advice, diagnosis, or treatment. Always consult a
          licensed healthcare professional for any medical concern.
        </div>
        """)

        # ── Stats bar ────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="stats-bar">
          <div>Symptoms catalogued: <span>{n_symptoms}</span></div>
          <div>Diseases covered: <span>{n_diseases}</span></div>
          <div>Model CV accuracy: <span>{cv_accuracy*100:.1f}%</span></div>
          <div>Algorithm: <span>Random Forest (200 trees)</span></div>
        </div>
        """)

        # ── Chat window ──────────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            value=[{"role": "assistant", "content": GREETING}],
            elem_id="chatbox",
            label="",
            show_label=False,
            height=500,
        )

        # ── Autocomplete dropdown (hidden until typing) ──────────────────────
        autocomplete_box = gr.CheckboxGroup(
            choices=[],
            label="💡 Suggestions — click to add",
            elem_id="autocomplete-box",
            visible=False,
            interactive=True,
        )

        # ── Input row ────────────────────────────────────────────────────────
        with gr.Row(elem_id="input-row"):
            symptom_input = gr.Textbox(
                placeholder="e.g. fever, headache, fatigue, nausea ...",
                lines=2,
                max_lines=4,
                show_label=False,
                elem_id="symptom-input",
                scale=5,
            )
            with gr.Column(scale=1, min_width=120):
                send_btn  = gr.Button("Analyse →", elem_id="send-btn",  variant="primary")
                clear_btn = gr.Button("Clear",     elem_id="clear-btn")

        # ── Quick examples ───────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["fever, headache, chills, sweating"],
                ["cough, shortness of breath, chest pain"],
                ["itching, skin rash, nodal skin eruptions"],
                ["fatigue, weight loss, night sweats, loss of appetite"],
                ["joint pain, muscle weakness, stiff neck"],
            ],
            inputs=symptom_input,
            label="🧪 Quick examples",
            elem_id="examples-box",
        )

        # ── Event wiring ─────────────────────────────────────────────────────
        # Live autocomplete as user types
        symptom_input.change(
            fn=autocomplete_symptoms,
            inputs=symptom_input,
            outputs=autocomplete_box,
        )

        # Click a suggestion chip → append to input
        autocomplete_box.select(
            fn=insert_suggestion,
            inputs=[autocomplete_box, symptom_input],
            outputs=[symptom_input, autocomplete_box],
        )

        # Send on button click
        send_btn.click(
            fn=respond,
            inputs=[symptom_input, chatbot],
            outputs=[chatbot, symptom_input],
        )

        # Send on Enter (Shift+Enter for newline)
        symptom_input.submit(
            fn=respond,
            inputs=[symptom_input, chatbot],
            outputs=[chatbot, symptom_input],
        )

        # Clear chat
        clear_btn.click(
            fn=lambda: ([{"role": "assistant", "content": GREETING}], ""),
            inputs=None,
            outputs=[chatbot, symptom_input],
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        share       = False,
        show_error  = True,
        css         = CSS,
    )
