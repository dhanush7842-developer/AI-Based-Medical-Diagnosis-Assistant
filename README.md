# ⚕ MedAI — AI-Based Medical Diagnosis Assistant

> **Educational use only. Not a substitute for professional medical advice.**

---

## Project structure

```
medical_chatbot/
├── app.py               ← Gradio UI  (run this)
├── ml_model.py          ← Random Forest + fuzzy matching engine
├── data_processor.py    ← CSV loading, cleaning, bug-fixes
├── requirements.txt
├── Training.csv         ← your dataset (place here)
└── Diseases_Symptoms.csv← treatment info (place here)
```

---

## Quick start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your CSVs in the same folder, then run
python app.py
# → opens at http://localhost:7860
```

---

## Bugs fixed

| # | Problem | Fix |
|---|---------|-----|
| 1 | `UnicodeDecodeError` on CSV load | Tries utf-8 → latin-1 → cp1252 → iso-8859-1 automatically |
| 2 | NaN in symptom columns crashes RF | `pd.to_numeric(..., errors='coerce').fillna(0)` |
| 3 | Diseases with 1–2 samples skew model | Filtered at `MIN_SAMPLES_PER_DISEASE = 5` |
| 4 | All-zero feature columns | Detected and dropped before training |
| 5 | Column name mismatches (spaces, caps) | Normalised to `lower_snake_case` |
| 6 | Target column name variation | Heuristic search for 'prognosis'/'disease' |
| 7 | Treatment CSV column variance | Keyword-based column detection |

---

## Features

- 🔍 **Fuzzy symptom matching** — handles typos & partial input via `difflib`
- 💡 **Live autocomplete** — chips appear as you type
- 📊 **Top-3 predictions** with probability bars and confidence level
- ⚠️ **Uncertainty indicator** — flags when predictions are low-confidence
- 💊 **Treatment suggestions** pulled from `Diseases_Symptoms.csv`
- 🛡️ **Medical disclaimer** — prominently displayed, can't be missed
- 🤖 **5-fold CV accuracy** shown in the stats bar so users know model quality

---

## Note on "antigravity"

`import antigravity` is a classic Python Easter egg (opens the xkcd comic).  
This project uses **Gradio** instead — the industry-standard Python library for
building interactive ML web UIs without JavaScript. The interface runs entirely
in the browser and is fully self-hosted.
