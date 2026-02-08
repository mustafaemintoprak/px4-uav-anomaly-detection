from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
import numpy as np


def tune_isolation_forest(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "contamination": [0.01, 0.02, 0.05],
        "max_samples": [256, 512]
    }
    def anomaly_scorer(estimator, X_val, y_val):
        preds = estimator.fit(X_val).predict(X_val)
        preds_binary = np.where(preds == -1, 1, 0)
        return f1_score(y_val, preds_binary)

    grid = GridSearchCV(
        IsolationForest(random_state=42),
        param_grid,
        scoring=anomaly_scorer,
        cv=3
    )

    grid.fit(X_scaled, y)

    return grid.best_params_, grid.best_score_