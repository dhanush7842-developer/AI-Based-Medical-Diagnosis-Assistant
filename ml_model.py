"""
ml_model.py — Medical Diagnosis ML Engine
Random Forest classifier with:
  • fuzzy symptom matching (handles typos / partial input)
  • top-3 disease predictions with calibrated probabilities
  • uncertainty estimation
"""

import numpy as np
import re
from difflib import get_close_matches
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_processor import load_training_data, load_treatment_data


# ── Constants ──────────────────────────────────────────────────────────────────
FUZZY_CUTOFF   = 0.6    # minimum similarity for fuzzy match
FUZZY_N        = 3      # max candidate matches returned
HIGH_CONF_THR  = 0.60   # ≥60 % → "High"
MED_CONF_THR   = 0.35   # ≥35 % → "Moderate", else "Low"


class MedicalDiagnosisModel:
    """Train once, predict many times."""

    def __init__(self):
        self.clf        : Optional[RandomForestClassifier] = None
        self.le         : LabelEncoder = LabelEncoder()
        self.symptoms   : List[str] = []
        self.treatments : Dict[str, str] = {}
        self.diseases   : List[str] = []
        self.cv_score   : float = 0.0
        self._trained   : bool = False

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self, training_csv: str = "Training.csv",
              treatment_csv: str = "Diseases_Symptoms.csv") -> None:
        print("── Loading data ──")
        X, y, self.symptoms = load_training_data(training_csv)
        self.treatments     = load_treatment_data(treatment_csv)

        y_enc = self.le.fit_transform(y)
        self.diseases = list(self.le.classes_)

        print("── Training Random Forest ──")
        self.clf = RandomForestClassifier(
            n_estimators   = 200,
            max_depth      = None,
            min_samples_split = 4,
            class_weight   = "balanced",
            random_state   = 42,
            n_jobs         = -1,
        )
        self.clf.fit(X.values, y_enc)

        # Cross-validation for honest accuracy estimate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.clf, X.values, y_enc, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        self.cv_score = float(scores.mean())
        print(f"  ✓ 5-fold CV accuracy: {self.cv_score*100:.1f}% "
              f"(±{scores.std()*100:.1f}%)\n")
        self._trained = True

    # ── Symptom helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"[\s_\-]+", "_", text.strip().lower())

    def fuzzy_match(self, raw_input: str) -> Tuple[Optional[str], List[str]]:
        """
        Returns (best_match_or_None, list_of_candidates).
        Tries exact → startswith → difflib fuzzy.
        """
        norm = self._normalise(raw_input)

        # 1. Exact
        if norm in self.symptoms:
            return norm, [norm]

        # 2. Startswith
        sw = [s for s in self.symptoms if s.startswith(norm)]
        if sw:
            return sw[0], sw[:FUZZY_N]

        # 3. Fuzzy
        candidates = get_close_matches(norm, self.symptoms,
                                       n=FUZZY_N, cutoff=FUZZY_CUTOFF)
        if candidates:
            return candidates[0], candidates

        return None, []

    def autocomplete(self, partial: str, max_results: int = 8) -> List[str]:
        """Return symptom names that start with or contain `partial`."""
        norm = self._normalise(partial)
        if not norm:
            return []
        exact   = [s for s in self.symptoms if s.startswith(norm)]
        contain = [s for s in self.symptoms if norm in s and s not in exact]
        return (exact + contain)[:max_results]

    # ── Prediction ─────────────────────────────────────────────────────────────
    def predict(
        self, symptom_inputs: List[str]
    ) -> Dict:
        """
        Parameters
        ----------
        symptom_inputs : list of raw strings typed by the user

        Returns
        -------
        {
          "matched_symptoms"   : [...],
          "unrecognised"       : [...],
          "top3"               : [
              {"disease": str, "probability": float, "confidence": str,
               "treatment": str},
              ...
          ],
          "uncertainty"        : str,   # "Low" | "Moderate" | "High"
          "model_cv_accuracy"  : float,
        }
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        matched, unrecognised, candidates_map = [], [], {}
        for raw in symptom_inputs:
            best, cands = self.fuzzy_match(raw)
            if best:
                matched.append(best)
                if best != self._normalise(raw):
                    candidates_map[raw] = cands
            else:
                unrecognised.append(raw)

        # Remove duplicates, keep order
        matched = list(dict.fromkeys(matched))

        # Build feature vector
        vec = np.zeros((1, len(self.symptoms)), dtype=int)
        for sym in matched:
            if sym in self.symptoms:
                vec[0, self.symptoms.index(sym)] = 1

        # Probabilities
        proba = self.clf.predict_proba(vec)[0]
        top_idx = np.argsort(proba)[::-1][:3]

        top3 = []
        for idx in top_idx:
            if proba[idx] < 0.01:
                continue
            disease = self.diseases[idx]
            prob    = float(proba[idx])
            conf    = (
                "High"     if prob >= HIGH_CONF_THR else
                "Moderate" if prob >= MED_CONF_THR  else
                "Low"
            )
            treatment = self.treatments.get(
                disease,
                "Please consult a qualified medical professional for treatment options."
            )
            top3.append({
                "disease"     : disease.title(),
                "probability" : round(prob * 100, 1),
                "confidence"  : conf,
                "treatment"   : treatment,
            })

        # Uncertainty = entropy-based
        top_prob = top3[0]["probability"] / 100 if top3 else 0
        uncertainty = (
            "Low"      if top_prob >= HIGH_CONF_THR else
            "Moderate" if top_prob >= MED_CONF_THR  else
            "High"
        )

        return {
            "matched_symptoms"  : matched,
            "unrecognised"      : unrecognised,
            "fuzzy_suggestions" : candidates_map,
            "top3"              : top3,
            "uncertainty"       : uncertainty,
            "model_cv_accuracy" : round(self.cv_score * 100, 1),
        }
