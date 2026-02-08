import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class UAVAnomalyDetector:
    def __init__(
        self,
        n_estimators=200,
        contamination=0.02,
        random_state=42
    ):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, df):
        self.X_scaled = self.scaler.fit_transform(df)
        self.model.fit(self.X_scaled)

    def predict(self):
        scores = self.model.decision_function(self.X_scaled)
        preds = self.model.predict(self.X_scaled)
        return scores, preds
