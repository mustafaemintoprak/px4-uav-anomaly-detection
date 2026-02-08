# PX4 UAV Telemetry Anomaly Detection

Unsupervised anomaly detection framework built on real PX4 UAV flight
controller telemetry logs.

------------------------------------------------------------------------

## 🚁 Project Overview

This project analyzes real PX4 `.ulg` flight logs and builds a
structured anomaly detection pipeline for UAV flight dynamics.

The system:

-   Parses real flight controller telemetry
-   Extracts dynamic stability features
-   Synchronizes multi-rate time-series signals
-   Applies Isolation Forest for unsupervised anomaly detection
-   Performs sensitivity analysis
-   Supports synthetic anomaly injection for validation

The objective is to detect statistically rare or abnormal flight
behavior in real UAV telemetry data.

------------------------------------------------------------------------

## 🧠 Engineering Motivation

UAV flight controllers continuously generate high-frequency telemetry
data including:

-   Angular rates
-   Orientation (quaternions)
-   Linear acceleration
-   Velocity
-   Actuator outputs

Detecting abnormal patterns in these signals is critical for:

-   Fault detection
-   Instability detection
-   Actuator saturation monitoring
-   Sensor anomaly detection
-   Predictive maintenance research

This project demonstrates a systems-oriented anomaly detection workflow
on real flight data.

------------------------------------------------------------------------

## 🏗 Architecture

    PX4 ULog (.ulg)
            ↓
    CSV Conversion (pyulog)
            ↓
    Feature Extraction & Timestamp Synchronization
            ↓
    StandardScaler (Normalization)
            ↓
    Isolation Forest
            ↓
    Anomaly Detection
            ↓
    Sensitivity & Hyperparameter Analysis

------------------------------------------------------------------------

## 📊 Feature Space

The anomaly detection model operates on dynamic flight features
extracted from:

### vehicle_attitude

-   rollspeed
-   pitchspeed
-   yawspeed

### control_state

-   roll_rate
-   pitch_rate
-   yaw_rate
-   x_acc
-   y_acc
-   z_acc
-   horizontal acceleration magnitude

These features represent UAV dynamic stability and motion behavior.

------------------------------------------------------------------------

## 🔬 Methodology

### 1️⃣ Unsupervised Anomaly Detection

Isolation Forest is used to identify statistically rare telemetry states
without labeled failures.

Example configuration:

``` python
IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)
```

-   `contamination` defines expected anomaly percentage\
-   The model outputs anomaly scores and binary predictions

------------------------------------------------------------------------

### 2️⃣ Sensitivity Analysis

Anomaly counts are evaluated across multiple contamination levels:

    0.005
    0.01
    0.02
    0.05

This helps evaluate detection robustness and model stability.

------------------------------------------------------------------------

### 3️⃣ Synthetic Fault Injection (Validation)

Since the dataset does not contain labeled failures:

-   Artificial anomalies are injected into angular rate signals
-   Labels are generated for injected samples
-   GridSearchCV is used with F1-score for hyperparameter tuning

This enables objective evaluation despite absence of real fault labels.

------------------------------------------------------------------------

## 📈 Example Results

-   6461 telemetry samples
-   10 flight dynamic features
-   \~130 anomalies detected at 2% contamination
-   F1 ≈ 0.57 during synthetic validation

------------------------------------------------------------------------

## 🚀 How to Run

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the full pipeline:

``` bash
python src/main.py
```

------------------------------------------------------------------------

## 📦 Dependencies

-   numpy
-   pandas
-   scikit-learn
-   matplotlib
-   pyulog

------------------------------------------------------------------------

## ⚠️ Notes

-   The dataset contains no true failure labels.
-   Hyperparameter tuning relies on synthetic anomaly injection.
-   Isolation Forest detects statistical rarity, not confirmed faults.

------------------------------------------------------------------------

## 🎯 Future Work

-   Physically realistic fault simulation (oscillation, drift, actuator
    bias)
-   Time-window based anomaly persistence detection
-   Real-time streaming implementation
-   Comparison with other anomaly detection algorithms (LOF, One-Class
    SVM, Autoencoders)

------------------------------------------------------------------------

## 📜 License

MIT License
