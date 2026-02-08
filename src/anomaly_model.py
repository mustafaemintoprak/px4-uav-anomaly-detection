from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )

    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    return scores, predictions