import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_features
from anomaly_model import UAVAnomalyDetector
from tuning import tune_isolation_forest
from preprocessing import inject_synthetic_anomalies


def run_baseline_detection(df):
    print("\n=== Baseline Detection ===")

    detector = UAVAnomalyDetector(
        n_estimators=200,
        contamination=0.02
    )

    detector.fit(df)
    scores, preds = detector.predict()

    anomalies = (preds == -1)

    print("Total samples:", len(df))
    print("Detected anomalies:", anomalies.sum())

    return scores, preds


def run_sensitivity_analysis(df):
    print("\n=== Sensitivity Analysis ===")

    for contamination in [0.01, 0.02, 0.05]:
        detector = UAVAnomalyDetector(
            n_estimators=200,
            contamination=contamination
        )

        detector.fit(df)
        _, preds = detector.predict()

        print(
            f"Contamination={contamination} → "
            f"Anomalies={np.sum(preds == -1)}"
        )


def run_tuning(df):
    print("\n=== Hyperparameter Tuning (Synthetic Injection) ===")

    df_aug, labels = inject_synthetic_anomalies(df)
    best_params, best_score = tune_isolation_forest(df_aug, labels)

    print("Best Parameters:", best_params)
    print("Best F1 Score:", best_score)


def main():
    df = load_features()

    scores, preds = run_baseline_detection(df)

    run_sensitivity_analysis(df)

    run_tuning(df)

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
