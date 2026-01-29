# train_once.py
from build_logistic_router import build_logistic_router
import inspect, argparse, joblib

TRAIN_FILE = "synthetic_equalweights_quiz50pct.xlsx"
MODEL_FILE = "model.joblib"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logistic",
                        choices=["logistic", "random_forest", "svm", "gradient_boosting"])
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--cv", type=int, default=None)
    args = parser.parse_args()

    sig = inspect.signature(build_logistic_router)
    params = sig.parameters

    if "model_name" in params and "cv_folds" in params:
        # New multi-model API
        route_kid, clf, metrics = build_logistic_router(
            TRAIN_FILE,
            model_name=args.model,
            threshold=args.threshold,
            cv_folds=args.cv
        )
    else:
        # Old API fallback (logistic only)
        print("[WARN] Multi-model parameters not supported by current build_logistic_router. "
              "Training logistic regression only.")
        route_kid, clf, metrics = build_logistic_router(
            TRAIN_FILE,
            threshold=args.threshold
        )

    joblib.dump(clf, MODEL_FILE)
    print("\nModel saved to", MODEL_FILE)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
