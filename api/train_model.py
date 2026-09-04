"""Train the fraud model, report held-out metrics, and persist scores."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psycopg
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURE_COLUMNS = ["time", *[f"v{i}" for i in range(1, 29)], "amount"]
DEFAULT_MODEL_PATH = "/app/models/fraud_model.joblib"


def train(database_url: str, model_path: Path, test_size: float, random_state: int) -> dict[str, float | int]:
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
    train_indexes, test_indexes = train_test_split(
        np.arange(len(labels)), test_size=test_size, random_state=random_state, stratify=labels
    )
    negative_count = int(np.sum(labels[train_indexes] == 0))
    positive_count = int(np.sum(labels[train_indexes] == 1))
    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=negative_count / positive_count,
        random_state=random_state,
        n_jobs=2,
    )
    model.fit(features[train_indexes], labels[train_indexes])
    test_probabilities = model.predict_proba(features[test_indexes])[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(np.int8)
    metrics = {
        "test_rows": int(len(test_indexes)),
        "precision": float(precision_score(labels[test_indexes], test_predictions, zero_division=0)),
        "recall": float(recall_score(labels[test_indexes], test_predictions, zero_division=0)),
        "f1": float(f1_score(labels[test_indexes], test_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels[test_indexes], test_probabilities)),
        "average_precision": float(average_precision_score(labels[test_indexes], test_probabilities)),
        "scale_pos_weight": float(negative_count / positive_count),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS risk_scores (
                transaction_id BIGINT PRIMARY KEY REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
                scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        probabilities = model.predict_proba(features)[:, 1]
        scored_at = datetime.now(timezone.utc)
        with connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO risk_scores (transaction_id, score, scored_at)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (transaction_id) DO UPDATE SET score = EXCLUDED.score, scored_at = EXCLUDED.scored_at""",
                zip(transaction_ids.tolist(), probabilities.tolist(), [scored_at] * len(transaction_ids)),
            )
        connection.commit()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    metrics = train(
        os.getenv("DATABASE_URL", "postgresql://docket:docket@localhost:5432/docket"),
        Path(args.model_path),
        args.test_size,
        args.random_state,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()