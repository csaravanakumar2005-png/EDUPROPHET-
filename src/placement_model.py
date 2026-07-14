import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble         import RandomForestRegressor
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, OrdinalEncoder
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      accuracy_score, mean_absolute_error, r2_score)

FEATURES = [
    "cgpa", "technical_score", "communication_score", "aptitude_score",
    "internships", "projects_count", "certifications", "hackathons",
    "github_repos", "leadership_roles", "mock_interview_score",
]
TARGET_CLASS = "placed"
TARGET_REG   = "expected_package_lpa"

PLACEMENT_LABELS = {0: "Not Ready", 1: "Placement Ready"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X  = df[FEATURES]
    y  = df[TARGET_CLASS]

    # For regression — only placed students
    placed_df = df[df[TARGET_CLASS] == 1]
    X_reg = placed_df[FEATURES]
    y_reg = placed_df[TARGET_REG]

    return X, y, X_reg, y_reg


def build_classifiers() -> dict:
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

def train_and_evaluate(csv_path: str = "data/placement_data.csv",
                        model_dir: str = "models") -> dict:
    os.makedirs(model_dir, exist_ok=True)

    X, y, X_reg, y_reg = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Classification ────────────────────────────────────────────────────
    clfs    = build_classifiers()
    results = {}

    for name, pipe in clfs.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()
        results[name] = {
            "pipeline":    pipe,
            "accuracy":    round(acc * 100, 2),
            "cv_accuracy": round(cv * 100, 2),
            "report":      classification_report(y_test, y_pred,
                               target_names=list(PLACEMENT_LABELS.values()),
                               output_dict=True),
            "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    best_name = max(results, key=lambda k: results[k]["cv_accuracy"])
    best_clf  = results[best_name]["pipeline"]

    # ── Regression (package prediction) ──────────────────────────────────
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )
    reg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=200, random_state=42)),
    ])
    reg_pipe.fit(X_r_train, y_r_train)
    y_r_pred = reg_pipe.predict(X_r_test)
    reg_mae  = round(mean_absolute_error(y_r_test, y_r_pred), 2)
    reg_r2   = round(r2_score(y_r_test, y_r_pred), 4)

    # ── save ──────────────────────────────────────────────────────────────
    joblib.dump({
        "classifier":      best_clf,
        "regressor":       reg_pipe,
        "features":        FEATURES,
        "labels":          PLACEMENT_LABELS,
        "best_model_name": best_name,
        "reg_mae":         reg_mae,
        "reg_r2":          reg_r2,
    }, os.path.join(model_dir, "placement_model.pkl"))

    print(f"\n[PLACEMENT] Best classifier → {best_name}  "
          f"(CV acc: {results[best_name]['cv_accuracy']}%)")
    print(f"[PLACEMENT] Package regressor → MAE={reg_mae} LPA  R²={reg_r2}")

    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "pipeline"}
        for k, v in results.items()
    }
    summary["best_model"] = best_name
    summary["regressor"]  = {"mae": reg_mae, "r2": reg_r2}
    return summary


# ── inference ─────────────────────────────────────────────────────────────────

def predict(input_dict: dict, model_dir: str = "models") -> dict:
    """
    Returns:
      { placement_label, placed (0/1), probability, confidence,
        expected_package_lpa (if placed) }
    """
    bundle = joblib.load(os.path.join(model_dir, "placement_model.pkl"))
    clf    = bundle["classifier"]
    reg    = bundle["regressor"]
    labels = bundle["labels"]

    X     = pd.DataFrame([input_dict])[FEATURES]
    cls   = int(clf.predict(X)[0])
    proba = clf.predict_proba(X)[0]
    conf  = round(float(proba.max()) * 100, 1)

    result = {
        "placed":          cls,
        "placement_label": labels[cls],
        "probability":     {labels[k]: round(float(p)*100, 1) for k, p in enumerate(proba)},
        "confidence":      conf,
    }

    if cls == 1:
        pkg = reg.predict(X)[0]
        result["expected_package_lpa"] = round(float(pkg), 2)

    return result


if __name__ == "__main__":
    results = train_and_evaluate()
    for model, info in results.items():
        if model in ("best_model", "regressor"):
            continue
        print(f"  {model:<22} acc={info['accuracy']}%  cv={info['cv_accuracy']}%")
    print(f"\n  ★ Best classifier → {results['best_model']}")
    print(f"  ★ Regressor MAE   → {results['regressor']['mae']} LPA")