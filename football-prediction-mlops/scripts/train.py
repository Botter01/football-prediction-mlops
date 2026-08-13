import argparse
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baseline_accuracy", type=float, default=0.50)
    args = parser.parse_args()

    mlflow.autolog()

    df = pd.read_csv(args.data_path)
    feature_cols = ["home_form", "away_form", "home_goal_diff_trend", "away_goal_diff_trend", "h2h_home_points"]

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", accuracy)

    print(f"New model accuracy: {accuracy:.4f}")
    print(f"Baseline accuracy: {args.baseline_accuracy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, "model.pkl"))

    result = {
        "accuracy": accuracy,
        "baseline_accuracy": args.baseline_accuracy,
        "passed_gate": accuracy >= args.baseline_accuracy
    }
    with open(os.path.join(args.output_dir, "validation_result.json"), "w") as f:
        json.dump(result, f)

    if not result["passed_gate"]:
        raise ValueError(
            f"Model did not pass validation gate: {accuracy:.4f} < baseline {args.baseline_accuracy:.4f}"
        )

if __name__ == "__main__":
    main()