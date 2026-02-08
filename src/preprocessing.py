import numpy as np
import pandas as pd

def inject_synthetic_anomalies(df, anomaly_fraction=0.02):
    df_copy = df.copy()
    n_samples=len(df_copy)
    n_anomalies=int(n_samples*anomaly_fraction)
    anomaly_indices = np.random.choice(
        n_samples,
        n_anomalies,
        replace=False
    )
    df_copy.loc[anomaly_indices, "roll_rate"] *= 20
    df_copy.loc[anomaly_indices, "pitch_rate"] *= 20
    df_copy.loc[anomaly_indices, "yaw_rate"] *= 20

    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1

    return df_copy, labels

def load_features():
    att = pd.read_csv("csv/sample_vehicle_attitude_0.csv")
    ctrl = pd.read_csv("csv/sample_control_state_0.csv")

    # Merge on nearest timestamp
    df = pd.merge_asof(
        att.sort_values("timestamp"),
        ctrl.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )

    features = df[
        [
            "rollspeed",
            "pitchspeed",
            "yawspeed",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",
            "horz_acc_mag",
            "x_acc",
            "y_acc",
            "z_acc"
        ]
    ].dropna()

    return features
