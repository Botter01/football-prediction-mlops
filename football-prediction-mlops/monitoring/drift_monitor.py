import pandas as pd
import psycopg2
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "drift_monitoring",
    "user": "drift_user",
    "password": "drift_pass"
}

def get_current_traffic(baseline, drifted=False):
    if drifted:
        sample = baseline.sample(n=200, replace=True).reset_index(drop=True)
        sample["home_form"] = (sample["home_form"] + 1.0).clip(0, 3)
        sample["away_form"] = (sample["away_form"] - 0.8).clip(0, 3)
        sample["home_goal_diff_trend"] = sample["home_goal_diff_trend"] + 1.5
        return sample
    return baseline.sample(n=200, replace=True).reset_index(drop=True)

def compute_drift(baseline, current):
    report = Report([DataDriftPreset()])
    result = report.run(current, baseline)
    result_dict = result.dict()

    metrics = result_dict["metrics"]
    dataset_drift_share = None
    per_feature = {}

    for m in metrics:
            metric_name = m.get("metric_name", "")

            if "DriftedColumnsCount" in metric_name:
                dataset_drift_share = m["value"]["share"]

            elif "ValueDrift" in metric_name:
                col_name = m["config"].get("column", "unknown")
                drift_score = m["value"]
                threshold = m["config"].get("threshold", 0.1)
                drift_detected = drift_score > threshold

                per_feature[col_name] = {
                    "drift_score": float(drift_score),
                    "drift_detected": bool(drift_detected)
                }

    return dataset_drift_share, per_feature

def save_to_postgres(dataset_drift_share, per_feature):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    checked_at = datetime.now()

    for feature_name, info in per_feature.items():
        cur.execute(
            """
            INSERT INTO drift_metrics (checked_at, feature_name, drift_score, drift_detected, dataset_drift_share)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (checked_at, feature_name, info["drift_score"], info["drift_detected"], dataset_drift_share)
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"{len(per_feature)} feature-metrika elmentve: {checked_at}")

if __name__ == "__main__":
    baseline_df = pd.read_csv("premier_league_features.csv")
    feature_cols = ["home_form", "away_form", "home_goal_diff_trend", "away_goal_diff_trend", "h2h_home_points"]
    baseline = baseline_df[feature_cols]

    import random
    current = get_current_traffic(baseline, drifted=random.choice([True, False]))

    dataset_drift_share, per_feature = compute_drift(baseline, current)
    save_to_postgres(dataset_drift_share, per_feature)