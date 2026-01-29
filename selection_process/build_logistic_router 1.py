# build_logistic_router.py
# -----------------------------------------------------------
# Trains a logistic classifier WITHOUT using gender (male/female),
# builds a callable router function that classifies a single input
# into 0 (Upskilling) or 1 (Employment). No files are saved.
# -----------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -----------------------------
# Public API
# -----------------------------
def build_logistic_router(
    data_path: str | Path,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    # classification threshold for routing (prob >= threshold -> Employment=1)
    threshold: float = 0.50,
    # set class_weight="balanced" if you expect imbalance
    class_weight=None
):
    """
    Returns:
        route_kid(raw: dict, threshold: float|None=None) -> dict
            Raw keys are the pipeline feature names (see REQUIRED_KEYS below).
            Output: {
                'prob_employment': float,
                'pred_label': int,   # 1=Employment, 0=Upskilling
                'route': 'Employment' | 'Upskilling'
            }
        clf: sklearn Pipeline (preprocessor + logistic model)
        metrics: dict of train/test metrics (for quick verification)
    """

    data_path = Path(data_path)
    df = pd.read_excel(data_path, engine="openpyxl")

    # -----------------------------
    # Columns used by the model (GENDER EXCLUDED)
    # -----------------------------
    cat_cols = ["location", "family", "schooltype", "highestedu"]      # <- gender removed
    num_cols = ["markspct", "yearssinceedu", "quiz", "age"]
    bin_cols = ["disability", "bpl", "techathome", "comfortapps", "extracurr"]
    label_col = "label"

    # If label missing but computed_score exists, build label (>= 40th percentile)
    if label_col not in df.columns:
        assert "computed_score" in df.columns, (
            "Missing 'label'. Provide 'computed_score' so we can create label "
            "as 1 if score >= 40th percentile, else 0."
        )
        thr40 = np.percentile(df["computed_score"].values, 40)
        df[label_col] = (df["computed_score"] >= thr40).astype(int)

    # Ensure required columns exist (ignore gender even if present)
    REQUIRED = set(cat_cols + num_cols + bin_cols + [label_col])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Keep only the columns we need (and drop any stray columns like 'gender')
    df = df[list(REQUIRED)].copy()

    # -----------------------------
    # Train / Test split (stratified)
    # -----------------------------
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # -----------------------------
    # Preprocess & Model
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
        ],
        remainder="drop",
    )

    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight=class_weight
    )

    clf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", logreg),
    ])

    # Fit
    clf.fit(X_train, y_train)

    # -----------------------------
    # Quick metrics (no saving)
    # -----------------------------
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    proba_train = clf.predict_proba(X_train)[:, 1]
    pred_train = (proba_train >= 0.5).astype(int)

    metrics = {
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "accuracy_train": float((pred_train == y_train).mean()),
        "accuracy_test": float(accuracy_score(y_test, pred_test)),
        "precision_test": float(precision_score(y_test, pred_test)),
        "recall_test": float(recall_score(y_test, pred_test)),
        "f1_test": float(f1_score(y_test, pred_test)),
        "roc_auc_test": float(roc_auc_score(y_test, proba_test)),
    }

    # -----------------------------
    # Router callable
    # -----------------------------
    REQUIRED_KEYS = cat_cols + num_cols + bin_cols

    def _as_dataframe(raw: dict) -> pd.DataFrame:
        """Validate and convert a single raw dict to a one-row DataFrame."""
        # Drop gender if user sends it by mistake
        raw_ = {k: v for k, v in raw.items() if k != "gender"}

        # Ensure all required keys exist
        missing_keys = [k for k in REQUIRED_KEYS if k not in raw_]
        if missing_keys:
            raise ValueError(f"Missing keys in input: {missing_keys}")

        # Keep only required keys in correct order
        row = {k: raw_[k] for k in REQUIRED_KEYS}
        return pd.DataFrame([row], columns=REQUIRED_KEYS)

    def route_kid(raw: dict, threshold_override: float | None = None) -> dict:
        """
        Decide whether a kid should go to:
          - 0: 'Upskilling'
          - 1: 'Employment'

        Inputs:
            raw: dict with keys:
                location, family, schooltype, highestedu,
                markspct, yearssinceedu, quiz, age,
                disability, bpl, techathome, comfortapps, extracurr
            threshold_override: optional float to override default threshold

        Returns:
            dict with:
                prob_employment (float), pred_label (0/1), route (str)
        """
        thr = threshold if threshold_override is None else float(threshold_override)

        df_one = _as_dataframe(raw)
        p = float(clf.predict_proba(df_one)[:, 1][0])
        yhat = int(p >= thr)
        routing = "Employment" if yhat == 1 else "Upskilling"

        return {
            "prob_employment": p,
            "pred_label": yhat,
            "route": routing
        }

    return route_kid, clf, metrics


# -----------------------------
# Example usage (uncomment to run interactively)
# -----------------------------
# route_kid, clf, metrics = build_logistic_router(
#     data_path=r"C:\Users\VV\Downloads\synthetic_equalweights_quiz50pct.xlsx",
#     threshold=0.50
# )
# print("Metrics:", metrics)
#
# example = {
#     "location":"rural",
#     "family":"single parent",
#     "schooltype":"govt",
#     "highestedu":"10",
#     "markspct":62.5,
#     "yearssinceedu":3,
#     "quiz":70,
#     "age":21,
#     "disability":0,
#     "bpl":1,
#     "techathome":1,
#     "comfortapps":1,
#     "extracurr":1
# }
# print(route_kid(example))         # -> {'prob_employment': ..., 'pred_label': 0/1, 'route': 'Upskilling'/'Employment'}
# print(route_kid(example, threshold_override=0.60))  # custom threshold
