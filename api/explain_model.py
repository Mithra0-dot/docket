"""Compute SHAP explanations and persist explainable escalation decisions."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psycopg
import shap
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from train_model import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = "/app/models/fraud_model.joblib"
RISK_ODDS_MARGIN = 4.0


def logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def find_decision_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def classify(
    probability: float,
    shap_row: np.ndarray,
    decision_threshold: float,
    shap_conflict_threshold: float,
) -> tuple[str, bool]:
    boundary_logit = logit(decision_threshold)
    score_logit = logit(float(np.clip(probability, 1e-7, 1 - 1e-7)))
    near_boundary = abs(score_logit - boundary_logit) < np.log(RISK_ODDS_MARGIN)
    dominant_shap = shap_row[np.argsort(np.abs(shap_row))[-5:]]
    positive_signal = bool(np.any(dominant_shap > 0))
    negative_signal = bool(np.any(dominant_shap < 0))
    conflict = positive_signal and negative_signal
    if near_boundary or conflict:
        return "ambiguous", conflict
    return ("auto-block" if probability >= decision_threshold else "auto-clear"), conflict


def explain(database_url: str, model_path: Path, random_state: int) -> dict[str, object]:
    query = "SELECT transaction_id, {} , class FROM transactions ORDER BY transaction_id".format(
        ", ".join(FEATURE_COLUMNS)
    )
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query).fetchall()
    if not rows:
        raise RuntimeError("transactions table is empty")

    transaction_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    features = np.asarray([row[1:-1] for row in rows], dtype=np.float32)
    labels = np.asarray([row[-1] for row in rows], dtype=np.int8)
    _, test_indexes = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=random_state, stratify=labels
    )
    model = joblib.load(model_path)
    test_probabilities = model.predict_proba(features[test_indexes])[:, 1]
    decision_threshold = find_decision_threshold(labels[test_indexes], test_probabilities)

    explainer = shap.TreeExplainer(model)
    test_shap = np.asarray(explainer.shap_values(features[test_indexes], check_additivity=False))
    if test_shap.ndim == 3:
        test_shap = test_shap[:, :, 1]
    shap_conflict_threshold = float(np.quantile(np.abs(test_shap), 0.75))

    all_probabilities = model.predict_proba(features)[:, 1]
    all_shap = np.asarray(explainer.shap_values(features, check_additivity=False))
    if all_shap.ndim == 3:
        all_shap = all_shap[:, :, 1]
    decisions = [
        classify(probability, shap_row, decision_threshold, shap_conflict_threshold)
        for probability, shap_row in zip(all_probabilities, all_shap)
    ]
    scored_at = datetime.now(timezone.utc)
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS shap_values (
                transaction_id BIGINT PRIMARY KEY REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                contributions JSONB NOT NULL,
                tier TEXT NOT NULL CHECK (tier IN ('auto-clear', 'ambiguous', 'auto-block')),
                conflict BOOLEAN NOT NULL,
                explained_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        with connection.cursor() as cursor:
            cursor.executemany(
                     """INSERT INTO shap_values (transaction_id, contributions, tier, conflict, explained_at)
                         VALUES (%s, %s::jsonb, %s, %s, %s)
                         ON CONFLICT (transaction_id) DO UPDATE SET contributions = EXCLUDED.contributions,
                   tier = EXCLUDED.tier, conflict = EXCLUDED.conflict, explained_at = EXCLUDED.explained_at""",
                (
                    (int(transaction_id), json.dumps(dict(zip(FEATURE_COLUMNS, shap_row.tolist()))), tier, conflict, scored_at)
                    for transaction_id, shap_row, (tier, conflict) in zip(transaction_ids, all_shap, decisions)
                ),
            )
        connection.commit()

    tier_counts = {tier: sum(decision[0] == tier for decision in decisions) for tier in ("auto-clear", "ambiguous", "auto-block")}
    return {
        "decision_threshold": decision_threshold,
        "lower_comfort_boundary": 1 / (1 + np.exp(-(logit(decision_threshold) - np.log(RISK_ODDS_MARGIN)))),
        "upper_comfort_boundary": 1 / (1 + np.exp(-(logit(decision_threshold) + np.log(RISK_ODDS_MARGIN)))),
        "shap_conflict_threshold": shap_conflict_threshold,
        "test_f1_at_decision_threshold": float(
            f1_score(labels[test_indexes], test_probabilities >= decision_threshold)
        ),
        "tier_counts": tier_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    result = explain(
        os.getenv("DATABASE_URL", "postgresql://docket:docket@localhost:5432/docket"),
        Path(args.model_path),
        args.random_state,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()