import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (classification_report, confusion_matrix, accuracy_score)

FEATURES = [
    "attendance_pct", "gpa", "backlogs", "family_income_lpa",
    "participation_score", "study_hours_day",
    "distance_km", "has_scholarship",
]
TARGET   = "dropout_risk"
CLASS_ORDER = ["Low", "Medium", "High"]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df  = pd.read_csv(csv_path)
    X   = df[FEATURES]
    le  = LabelEncoder()
    le.fit(CLASS_ORDER)
    y   = le.transform(df[TARGET])
    return X, y, le


def build_models() -> dict:
    """Return a dict of {name: pipeline}."""
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, random_state=42,
                                              class_weight="balanced")),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", probability=True,
                           class_weight="balanced", random_state=42)),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=7)),
        ]),
    }


# ── training ─────────────────────────────────────────────────────────────────

def train_and_evaluate(csv_path: str = "data/dropout_data.csv",
                        model_dir: str = "models") -> dict:
    """
    Train all models, pick the best one, save it and return a results dict.
    """
    os.makedirs(model_dir, exist_ok=True)

    X, y, le = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models  = build_models()
    results = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
        results[name] = {
            "pipeline":    pipe,
            "accuracy":    round(acc * 100, 2),
            "cv_accuracy": round(cv * 100, 2),
            "report":      classification_report(y_test, y_pred,
                               target_names=CLASS_ORDER, output_dict=True),
            "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    # ── pick best by CV accuracy ──────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["cv_accuracy"])
    best_pipe = results[best_name]["pipeline"]

    # save best model + label encoder + feature list
    joblib.dump({
        "pipeline": best_pipe,
        "encoder":  le,
        "features": FEATURES,
        "classes":  CLASS_ORDER,
        "best_model_name": best_name,
    }, os.path.join(model_dir, "dropout_model.pkl"))

    print(f"\n[DROPOUT] Best model → {best_name}  "
          f"(CV acc: {results[best_name]['cv_accuracy']}%)")

    # drop pipeline objects before returning (not JSON-serialisable)
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "pipeline"}
        for k, v in results.items()
    }
    summary["best_model"] = best_name
    return summary


# ── inference ─────────────────────────────────────────────────────────────────

def predict(input_dict: dict, model_dir: str = "models") -> dict:
    """
    input_dict keys must match FEATURES.
    Returns: { risk_label, probabilities, confidence }
    """
    bundle = joblib.load(os.path.join(model_dir, "dropout_model.pkl"))
    pipe   = bundle["pipeline"]
    le     = bundle["encoder"]

    X   = pd.DataFrame([input_dict])[FEATURES]
    idx = pipe.predict(X)[0]
    proba = pipe.predict_proba(X)[0]

    label = le.inverse_transform([idx])[0]
    prob_dict = {c: round(float(p)*100, 1) for c, p in zip(CLASS_ORDER, proba)}
    confidence = round(float(proba.max())*100, 1)

    return {
        "risk_label":    label,
        "probabilities": prob_dict,
        "confidence":    confidence,
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    for model, info in results.items():
        if model == "best_model":
            continue
        print(f"  {model:<22} acc={info['accuracy']}%  cv={info['cv_accuracy']}%")
    print(f"\n  ★ Best → {results['best_model']}")
                            