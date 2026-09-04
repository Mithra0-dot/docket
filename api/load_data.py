"""Load the Kaggle credit-card fraud CSV into Postgres without fabricating rows."""

import argparse
import csv
import os
from pathlib import Path

import psycopg

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
DB_COLUMNS = [column.lower() for column in FEATURE_COLUMNS]


def find_csv(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        return path

    download_error: Exception | None = None
    try:
        import kagglehub

        dataset_dir = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
        candidates = list(dataset_dir.rglob("creditcard.csv"))
        if candidates:
            return candidates[0]
    except Exception as error:
        download_error = error

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    local_csv = data_dir / "creditcard.csv"
    if local_csv.is_file():
        return local_csv

    if download_error:
        raise RuntimeError(
            "Kaggle download failed. Authenticate with `kagglehub.login()` or configure "
            "KAGGLE_USERNAME/KAGGLE_KEY`, then rerun. Alternatively place the official "
            "creditcard.csv at data/creditcard.csv and rerun. No sample data was created."
        ) from download_error

    raise FileNotFoundError(
        "Kaggle returned no creditcard.csv. Place the official file at "
        "data/creditcard.csv or pass --csv PATH."
    )


def load(csv_path: Path, database_url: str) -> int:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != FEATURE_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns. Expected {FEATURE_COLUMNS}, got {reader.fieldnames}"
            )

        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("TRUNCATE transactions RESTART IDENTITY")
                with cursor.copy(
                    "COPY transactions (time, v1, v2, v3, v4, v5, v6, v7, v8, v9, "
                    "v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, "
                    "v22, v23, v24, v25, v26, v27, v28, amount, class) FROM STDIN"
                ) as copy:
                    count = 0
                    for row in reader:
                        copy.write_row(tuple(row[column] for column in FEATURE_COLUMNS))
                        count += 1
            connection.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to the official creditcard.csv")
    args = parser.parse_args()
    database_url = os.getenv(
        "DATABASE_URL", "postgresql://docket:docket@localhost:5432/docket"
    )
    csv_path = find_csv(args.csv)
    print(f"Loaded {load(csv_path, database_url):,} rows from {csv_path}")


if __name__ == "__main__":
    main()
