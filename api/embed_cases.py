"""Persist the real transaction feature vectors for pgvector retrieval."""

import argparse
import os
from datetime import datetime, timezone

import psycopg

from train_model import FEATURE_COLUMNS


def vector_literal(values: tuple[float, ...]) -> str:
    return "[{}]".format(",".join(str(float(value)) for value in values))


def embed(database_url: str) -> int:
    query = "SELECT transaction_id, {} FROM transactions ORDER BY transaction_id".format(
        ", ".join(FEATURE_COLUMNS)
    )
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query).fetchall()
        if not rows:
            raise RuntimeError("transactions table is empty")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS case_embeddings (
                transaction_id BIGINT PRIMARY KEY REFERENCES transactions(transaction_id) ON DELETE CASCADE,
                embedding VECTOR(30) NOT NULL,
                embedded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS case_embeddings_embedding_idx
               ON case_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"""
        )
        embedded_at = datetime.now(timezone.utc)
        with connection.cursor() as cursor:
            cursor.executemany(
                """INSERT INTO case_embeddings (transaction_id, embedding, embedded_at)
                   VALUES (%s, %s::vector, %s)
                   ON CONFLICT (transaction_id) DO UPDATE SET
                       embedding = EXCLUDED.embedding, embedded_at = EXCLUDED.embedded_at""",
                (
                    (row[0], vector_literal(tuple(row[1:])), embedded_at)
                    for row in rows
                ),
            )
        connection.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://docket:docket@localhost:5432/docket"
    )
    print(f"Embedded {embed(database_url):,} transactions")


if __name__ == "__main__":
    main()