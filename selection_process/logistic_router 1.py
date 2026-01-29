# logistic_router.py
import joblib
import pandas as pd

MODEL_FILE = "model.joblib"

REQUIRED_KEYS = [
    "location","family","schooltype","highestedu",
    "markspct","yearssinceedu","quiz","age",
    "disability","bpl","techathome","comfortapps","extracurr"
]

model = joblib.load(MODEL_FILE)

def route_kid(raw: dict, threshold_override: float | None = None, base_threshold=0.50):
    thr = threshold_override if threshold_override is not None else base_threshold

    # drop gender if provided
    raw = {k: v for k, v in raw.items() if k != "gender"}

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    df = pd.DataFrame([raw], columns=REQUIRED_KEYS)
    p = float(model.predict_proba(df)[:, 1][0])
    label = int(p >= thr)
    route = "Employment" if label == 1 else "Upskilling"

    return {"prob_employment": p, "pred_label": label, "route": route}