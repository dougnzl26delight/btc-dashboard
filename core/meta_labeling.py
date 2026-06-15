"""Meta-labeling. López de Prado AFML (2018) Ch. 3.6.

Two-step structure:
  Primary:    raw signal generates entry/exit calls (direction + magnitude)
  Secondary:  ML classifier predicts P(success | features) at each event
  Action:     only act on primary if secondary confidence > threshold

The secondary classifier's role is FILTERING the primary, not adding direction.
This often yields meaningful Sharpe lift on weak signals by pruning the trades
that the primary system misclassifies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from core import exits


def build_meta_dataset(
    prices: pd.Series,
    primary_signal: pd.Series,
    feature_df: pd.DataFrame,
    horizon: int = 30,
    pt_sigma: float = 2.0,
    sl_sigma: float = 2.0,
    vol_window: int = 30,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate (X, y) for the secondary classifier.

    Events fire on direction-changes of the primary signal: each time the
    signal crosses from <= threshold to > threshold (long entry) or
    from >= -threshold to < -threshold (short entry).

    X = features at event times. y = 1 if profit barrier hit, else 0.
    """
    sign = pd.Series(0.0, index=primary_signal.index)
    sign[primary_signal > threshold] = 1.0
    sign[primary_signal < -threshold] = -1.0
    prev = sign.shift(1).fillna(0.0)
    events = pd.Series(0.0, index=primary_signal.index)
    events[(sign == 1.0) & (prev != 1.0)] = 1.0
    events[(sign == -1.0) & (prev != -1.0)] = -1.0

    barriers = exits.triple_barrier(
        prices,
        events,
        horizon_days=horizon,
        pt_sigma=pt_sigma,
        sl_sigma=sl_sigma,
        vol_window=vol_window,
    )
    if barriers.empty:
        return pd.DataFrame(), pd.Series(dtype=int)

    barriers["label"] = (barriers["return"] > 0).astype(int)
    # Note: barriers timestamps may have ms precision while feature_df is ns;
    # use set intersection to avoid pandas precision-mismatch reindex pitfalls.
    event_ts_set = set(barriers["event_ts"])
    common_idx = [ts for ts in feature_df.index if ts in event_ts_set]
    if not common_idx:
        return pd.DataFrame(), pd.Series(dtype=int)
    X = feature_df.loc[common_idx].dropna()
    barriers_keyed = barriers.set_index("event_ts")
    y = barriers_keyed.loc[X.index, "label"]
    return X, y


def train_meta_classifier(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> tuple[RandomForestClassifier | None, dict]:
    """Train a RandomForest meta-classifier with time-series CV.

    Returns (fitted_model, diagnostics_dict).
    """
    if len(X) < 50:
        return None, {"error": "insufficient data", "n_obs": len(X)}

    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 20)))
    accuracies: list[float] = []
    base_clf = RandomForestClassifier(
        n_estimators=200, max_depth=4, random_state=42, n_jobs=-1
    )

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if y_tr.nunique() < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=4, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        accuracies.append(float(clf.score(X_te, y_te)))

    final = RandomForestClassifier(
        n_estimators=200, max_depth=4, random_state=42, n_jobs=-1
    )
    final.fit(X, y)

    return final, {
        "n_obs": int(len(X)),
        "base_rate": float(y.mean()),
        "n_features": int(X.shape[1]),
        "cv_accuracies": accuracies,
        "mean_cv_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "feature_importance": dict(zip(X.columns, final.feature_importances_)),
    }


def filter_signal(
    primary_signal: pd.Series,
    feature_df: pd.DataFrame,
    classifier: RandomForestClassifier,
    confidence_threshold: float = 0.55,
) -> pd.Series:
    """Apply meta-classifier at runtime to filter the primary signal.

    Returns a series same shape as primary_signal, zeroed where the secondary
    classifier's confidence < confidence_threshold.
    """
    aligned = feature_df.reindex(primary_signal.index).dropna()
    if aligned.empty:
        return pd.Series(0.0, index=primary_signal.index)

    proba = classifier.predict_proba(aligned)[:, 1]
    confident_mask = proba > confidence_threshold

    out = pd.Series(0.0, index=primary_signal.index)
    confident_idx = aligned.index[confident_mask]
    out.loc[confident_idx] = primary_signal.loc[confident_idx].values
    return out
