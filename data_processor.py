"""
data_processor.py — Medical ChatBot Data Pipeline
Handles: encoding issues, missing values, label alignment,
         insufficient-data filtering, and symptom list extraction.
"""

import pandas as pd
import numpy as np
import re
import os
from typing import Tuple, List, Dict

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_SAMPLES_PER_DISEASE = 5       # diseases with fewer rows are dropped
TRAINING_CSV   = "Training.csv"
TREATMENT_CSV  = "Diseases_Symptoms.csv"
ENCODINGS      = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_csv_safe(filepath: str) -> pd.DataFrame:
    """Try multiple encodings; raise a clear error if all fail."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[DataProcessor] '{filepath}' not found. "
            "Place it in the same directory as this script."
        )
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            print(f"  ✓ Loaded '{filepath}' with encoding={enc!r}  "
                  f"({len(df)} rows × {len(df.columns)} cols)")
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(
        f"[DataProcessor] Could not decode '{filepath}' with any of {ENCODINGS}."
    )


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, replace spaces/dashes with underscores."""
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[\s\-]+", "_", regex=True)
          .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def _clean_string_column(series: pd.Series) -> pd.Series:
    """Strip whitespace, lowercase, normalise internal spaces."""
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .str.replace(r"\s+", " ", regex=True)
    )


# ── Main loader ────────────────────────────────────────────────────────────────
def load_training_data(
    filepath: str = TRAINING_CSV,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Returns
    -------
    X          : binary symptom DataFrame  (rows = samples)
    y          : disease label Series
    symptoms   : sorted list of all symptom column names
    """
    df = _load_csv_safe(filepath)
    df = _clean_column_names(df)

    # ── Locate the target column ───────────────────────────────────────────────
    if "prognosis" not in df.columns:
        candidates = [c for c in df.columns if "prognos" in c or "disease" in c]
        if not candidates:
            raise KeyError(
                "[DataProcessor] 'prognosis' column not found. "
                f"Available columns: {list(df.columns)}"
            )
        df = df.rename(columns={candidates[0]: "prognosis"})
        print(f"  ⚠ Renamed '{candidates[0]}' → 'prognosis'")

    # ── Clean target ──────────────────────────────────────────────────────────
    df["prognosis"] = _clean_string_column(df["prognosis"])
    df = df[df["prognosis"].notna() & (df["prognosis"] != "nan")]

    # ── Symptom columns ───────────────────────────────────────────────────────
    symptom_cols = [c for c in df.columns if c != "prognosis"]

    # Convert to numeric (0/1); non-numeric → NaN → 0
    for col in symptom_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Drop columns that are entirely zero (useless features)
    all_zero = [c for c in symptom_cols if df[c].sum() == 0]
    if all_zero:
        print(f"  ⚠ Dropping {len(all_zero)} all-zero symptom columns: {all_zero[:5]}…")
        df.drop(columns=all_zero, inplace=True)
        symptom_cols = [c for c in symptom_cols if c not in all_zero]

    # ── Filter diseases with too few samples ──────────────────────────────────
    counts = df["prognosis"].value_counts()
    rare   = counts[counts < MIN_SAMPLES_PER_DISEASE].index.tolist()
    if rare:
        print(f"  ⚠ Removing {len(rare)} disease(s) with < {MIN_SAMPLES_PER_DISEASE} "
              f"samples: {rare}")
        df = df[~df["prognosis"].isin(rare)]

    X = df[symptom_cols]
    y = df["prognosis"]

    print(f"  ✓ Training data ready: {X.shape[0]} samples, "
          f"{X.shape[1]} symptoms, {y.nunique()} diseases\n")
    return X, y, sorted(symptom_cols)


def load_treatment_data(filepath: str = TREATMENT_CSV) -> Dict[str, str]:
    """
    Returns dict  {disease_name: treatment_text}.
    Gracefully handles missing files, bad encodings, varied column names.
    """
    if not os.path.exists(filepath):
        print(f"  ⚠ '{filepath}' not found — treatment info will be unavailable.")
        return {}

    df = _load_csv_safe(filepath)
    df = _clean_column_names(df)

    # Detect disease + treatment columns heuristically
    disease_col   = _find_column(df, ["disease", "condition", "prognosis", "name"])
    treatment_col = _find_column(df, ["treatment", "cure", "remedy", "management"])

    if disease_col is None or treatment_col is None:
        print(f"  ⚠ Could not identify disease/treatment columns in {filepath}. "
              f"Found: {list(df.columns)}")
        return {}

    df[disease_col]   = _clean_string_column(df[disease_col])
    df[treatment_col] = df[treatment_col].fillna("No treatment info available.").astype(str).str.strip()

    mapping = dict(zip(df[disease_col], df[treatment_col]))
    print(f"  ✓ Treatment data: {len(mapping)} entries\n")
    return mapping


def _find_column(df: pd.DataFrame, keywords: List[str]):
    """Return first column name that contains any keyword."""
    for kw in keywords:
        for col in df.columns:
            if kw in col:
                return col
    return None
