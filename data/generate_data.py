import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 1000  # number of student records

# ─────────────────────────────────────────────
#  DROPOUT RISK DATASET
# ─────────────────────────────────────────────
def generate_dropout_data(n=N):
    data = {
        "student_id":        [f"STU{str(i).zfill(4)}" for i in range(1, n+1)],
        "attendance_pct":    np.clip(np.random.normal(72, 18, n), 20, 100).round(1),
        "gpa":               np.clip(np.random.normal(6.8, 1.5, n), 3.0, 10.0).round(2),
        "backlogs":          np.random.choice([0,1,2,3,4,5,6], n, p=[0.35,0.25,0.18,0.10,0.06,0.04,0.02]),
        "family_income_lpa": np.clip(np.random.exponential(4.5, n), 0.5, 30).round(2),
        "participation_score": np.random.randint(0, 101, n),
        "study_hours_day":   np.clip(np.random.normal(4.5, 2.0, n), 0.5, 12).round(1),
        "distance_km":       np.clip(np.random.exponential(15, n), 0, 80).round(1),
        "has_scholarship":   np.random.choice([0, 1], n, p=[0.65, 0.35]),
    }

    df = pd.DataFrame(data)

    # Deterministic risk scoring
    score = (
        - 0.40 * df["attendance_pct"]
        - 3.5  * df["gpa"]
        + 4.0  * df["backlogs"]
        - 0.20 * df["family_income_lpa"]
        - 0.12 * df["participation_score"]
        - 1.80 * df["study_hours_day"]
        + 0.10 * df["distance_km"]
        - 5.0  * df["has_scholarship"]
        + np.random.normal(0, 4, n)
    )

    # Normalise to [0,1]
    score_norm = (score - score.min()) / (score.max() - score.min())

    def label(s):
        if s < 0.35:  return "Low"
        if s < 0.65:  return "Medium"
        return "High"

    df["dropout_risk"] = score_norm.apply(label)
    return df


# ─────────────────────────────────────────────
#  PLACEMENT READINESS DATASET
# ─────────────────────────────────────────────
def generate_placement_data(n=N):
    data = {
        "student_id":           [f"STU{str(i).zfill(4)}" for i in range(1, n+1)],
        "cgpa":                 np.clip(np.random.normal(7.4, 1.2, n), 4.0, 10.0).round(2),
        "technical_score":      np.random.randint(30, 101, n),   # out of 100
        "communication_score":  np.random.randint(30, 101, n),
        "aptitude_score":       np.random.randint(30, 101, n),
        "internships":          np.random.choice([0,1,2,3], n, p=[0.40,0.35,0.18,0.07]),
        "projects_count":       np.random.choice([0,1,2,3,4,5], n, p=[0.10,0.20,0.30,0.22,0.12,0.06]),
        "certifications":       np.random.choice([0,1,2,3,4], n, p=[0.20,0.30,0.28,0.14,0.08]),
        "hackathons":           np.random.choice([0,1,2,3], n, p=[0.45,0.30,0.17,0.08]),
        "github_repos":         np.random.randint(0, 41, n),
        "leadership_roles":     np.random.choice([0,1,2], n, p=[0.55,0.35,0.10]),
        "mock_interview_score": np.random.randint(30, 101, n),
        "branch":               np.random.choice(
                                    ["CSE","ECE","ME","CE","IT","EEE"],
                                    n, p=[0.28,0.20,0.15,0.12,0.15,0.10]),
    }

    df = pd.DataFrame(data)

    # Placement likelihood score
    score = (
          3.5  * df["cgpa"]
        + 0.30 * df["technical_score"]
        + 0.20 * df["communication_score"]
        + 0.20 * df["aptitude_score"]
        + 8.0  * df["internships"]
        + 3.0  * df["projects_count"]
        + 2.5  * df["certifications"]
        + 2.0  * df["hackathons"]
        + 0.30 * df["github_repos"]
        + 3.0  * df["leadership_roles"]
        + 0.25 * df["mock_interview_score"]
        + np.random.normal(0, 5, n)
    )

    score_norm = (score - score.min()) / (score.max() - score.min())
    df["placed"] = (score_norm > 0.45).astype(int)

    # Estimated package (LPA) only for placed students
    pkg = np.where(
        df["placed"] == 1,
        np.clip(3.0 + score_norm * 18 + np.random.normal(0, 1.5, n), 3.0, 45.0),
        0.0
    ).round(2)
    df["expected_package_lpa"] = pkg

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    dropout_df   = generate_dropout_data()
    placement_df = generate_placement_data()

    dropout_df.to_csv("data/dropout_data.csv",   index=False)
    placement_df.to_csv("data/placement_data.csv", index=False)

    print(f"✅ dropout_data.csv   → {len(dropout_df)} rows")
    print(f"   Distribution: {dropout_df['dropout_risk'].value_counts().to_dict()}")
    print(f"✅ placement_data.csv → {len(placement_df)} rows")
    print(f"   Placed: {placement_df['placed'].sum()} | Not placed: {(placement_df['placed']==0).sum()}")