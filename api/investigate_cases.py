"""Generate and persist local-LLM investigation memos for ambiguous cases."""

import argparse
import json
import os

import psycopg

from main import call_investigation_memo, investigation_context, save_memo, GROUNDING_SYSTEM


def investigate_all(
    database_url: str,
    limit: int | None = None,
    transaction_id: int | None = None,
) -> int:
    with psycopg.connect(database_url) as connection:
        query = """SELECT transaction_id FROM shap_values
                   WHERE tier = 'ambiguous' ORDER BY transaction_id"""
        if transaction_id is not None:
            query = """SELECT transaction_id FROM shap_values
                       WHERE tier = 'ambiguous' AND transaction_id = %s"""
            rows = connection.execute(query, (transaction_id,)).fetchall()
        elif limit is not None:
            query += " LIMIT %s"
            rows = connection.execute(query, (limit,)).fetchall()
        else:
            rows = connection.execute(query).fetchall()

    for index, (transaction_id,) in enumerate(rows, start=1):
        context, _, existing_memo, _ = investigation_context(transaction_id)
        if existing_memo:
            continue
        memo, model_name = call_investigation_memo(
            GROUNDING_SYSTEM,
            """Investigate this ambiguous transaction. Produce a short memo with exactly these
sections: Suspicious signals, What precedents suggest, Recommendation. The recommendation
must be only further block, clear, or escalate. Ground every statement in the supplied data.

Evidence:
""" + json.dumps(context, sort_keys=True),
        )
        save_memo(transaction_id, memo, model_name)
        print(f"Investigated {transaction_id} ({index}/{len(rows)})")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Process at most this many cases")
    selection.add_argument(
        "--transaction-id",
        type=int,
        help="Process this specific ambiguous transaction",
    )
    args = parser.parse_args()
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://docket:docket@localhost:5432/docket"
    )
    print(
        f"Processed {investigate_all(database_url, args.limit, args.transaction_id):,} ambiguous cases"
    )


if __name__ == "__main__":
    main()