# compare_models_single_metric.py
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import roc_auc_score

# Plotly
import plotly.graph_objects as go

DATA_FILE = "synthetic_equalweights_quiz50pct.xlsx"

# -------------------------------
# Load & preprocess
# -------------------------------
def load_data():
    df = pd.read_excel(DATA_FILE, engine="openpyxl")

    cat_cols = ["location", "family", "schooltype", "highestedu"]
    num_cols = ["markspct", "yearssinceedu", "quiz", "age"]
    bin_cols = ["disability", "bpl", "techathome", "comfortapps", "extracurr"]
    label_col = "label"

    # Build label if missing
    if label_col not in df.columns:
        thr40 = np.percentile(df["computed_score"], 40)
        df[label_col] = (df["computed_score"] >= thr40).astype(int)

    X = df[cat_cols + num_cols + bin_cols]
    y = df[label_col].astype(int)
    return X, y, cat_cols, num_cols, bin_cols

def build_preprocessor(cat_cols, num_cols, bin_cols):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
            ("bin", "passthrough", bin_cols),
        ]
    )

# -------------------------------
# Main comparison
# -------------------------------
if __name__ == "__main__":
    # Choose ONE metric:
    METRIC_NAME = "AUC"     # options: "AUC" (recommended), “Accuracy”, “F1”
    print(f"\nComparing models using SINGLE metric: {METRIC_NAME}\n")

    X, y, cat_cols, num_cols, bin_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = build_preprocessor(cat_cols, num_cols, bin_cols)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    scores = {}

    for name, model in models.items():
        clf = Pipeline([("pre", pre), ("model", model)])
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)

        scores[name] = auc  # single metric stored
        print(f"{name}: AUC = {auc:.4f}")

    # ---------------------------
    # Select BEST model
    # ---------------------------
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]

    print("\n======================================")
    print(f"BEST MODEL SELECTED: {best_model}")
    print(f"BEST {METRIC_NAME}: {best_score:.4f}")
    print("======================================\n")

    # ---------------------------
    # Plot single-metric bar chart
    # ---------------------------
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        marker_color="royalblue",
    ))

    fig.update_layout(
        title=f"Model Comparison (Metric = {METRIC_NAME})",
        xaxis_title="Model",
        yaxis_title=METRIC_NAME,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.show()