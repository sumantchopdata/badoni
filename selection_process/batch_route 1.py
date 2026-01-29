# batch_route.py
import pandas as pd
from logistic_router import route_kid, REQUIRED_KEYS

def main(input_file, output_prefix="kids", threshold_override=None):
    df = pd.read_excel(input_file, engine="openpyxl")

    # validate columns
    missing = [c for c in REQUIRED_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file missing columns: {missing}")

    results = []
    for _, row in df.iterrows():
        ex = {k: row[k] for k in REQUIRED_KEYS}
        results.append(route_kid(ex, threshold_override))

    df["prob_employment"] = [r["prob_employment"] for r in results]
    df["pred_label"] = [r["pred_label"] for r in results]
    df["route"] = [r["route"] for r in results]

    emp = df[df["route"] == "Employment"]
    up  = df[df["route"] == "Upskilling"]

    emp.to_excel(f"{output_prefix}_employment.xlsx", index=False)
    up.to_excel(f"{output_prefix}_upskilling.xlsx", index=False)

    print("Created:")
    print(f"- {output_prefix}_employment.xlsx")
    print(f"- {output_prefix}_upskilling.xlsx")

if __name__ == "__main__":
    main("test2.xlsx")   # CHANGE THIS