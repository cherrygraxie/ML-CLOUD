
"""
HPC Dataset Execution and Duplicate Investigation
=================================================

Script:
    CLOUD AI\analyze_execution_and_duplicates.py

Dataset:
    CLOUD AI\data\processed\eagle_hpc_ml_v3.parquet

Outputs:
    CLOUD AI\data\validation\execution_status_analysis.csv
    CLOUD AI\data\validation\zero_resource_analysis.csv
    CLOUD AI\data\validation\duplicate_pattern_analysis.csv
    CLOUD AI\data\validation\negative_runtime_rows.csv
    CLOUD AI\data\validation\execution_duplicate_investigation.json

Designed for large Parquet datasets using batch processing.
"""

from pathlib import Path
from collections import Counter
import hashlib
import json
import sqlite3
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

# This script is directly inside the CLOUD AI project folder.
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eagle_hpc_ml_v3.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"

BATCH_SIZE = 100_000
MAX_EXPORTED_PATTERNS = 100_000

EXECUTION_REPORT = OUTPUT_DIR / "execution_status_analysis.csv"
ZERO_REPORT = OUTPUT_DIR / "zero_resource_analysis.csv"
DUPLICATE_REPORT = OUTPUT_DIR / "duplicate_pattern_analysis.csv"
NEGATIVE_REPORT = OUTPUT_DIR / "negative_runtime_rows.csv"
JSON_REPORT = OUTPUT_DIR / "execution_duplicate_investigation.json"

# Temporary database used to combine workload-pattern counts
# across batches. It is removed and rebuilt on every run.
PATTERN_DB = OUTPUT_DIR / "pattern_counts_temp.sqlite3"


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

EXECUTION_COLUMNS = [
    "state",
    "executed_job",
]

RESOURCE_COLUMNS = [
    "nodes_used",
    "processors_used",
    "wallclock_used_seconds",
]

RUNTIME_COLUMN = "wallclock_used_seconds"

# These columns describe requested resources and job configuration.
# They are used to identify repeated workload patterns, NOT to
# declare that records are definitely duplicate job submissions.
PATTERN_CANDIDATES = [
    "nodes_req",
    "processors_req",
    "memory_req_gb",
    "memory_req_scope",
    "memory_req_total_gb",
    "wallclock_req_seconds",
    "gpus_requested",
    "partition",
    "qos",
    "python_job",
    "reframe_job",
    "job_type_hash",
    "day",
    "day_of_week",
    "hour",
    "minute",
    "submit_year",
    "submit_month",
    "state",
    "executed_job",
]


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def print_section(title):
    """Print a readable section heading."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def json_safe(value):
    """Convert pandas/numpy values to JSON-compatible values."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if value is pd.NA:
        return None

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def canonical_pattern(values):
    """Serialize a workload pattern consistently for hashing."""
    normalized = [json_safe(value) for value in values]

    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_csv(records, output_path, columns=None):
    """Write records to CSV, including headers for empty reports."""
    dataframe = pd.DataFrame(records, columns=columns)
    dataframe.to_csv(output_path, index=False)


# ============================================================
# 4. SQLITE PATTERN STORE
# ============================================================

def initialize_pattern_database(connection):
    """Create the persistent aggregation table."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS workload_patterns (
            pattern_hash TEXT PRIMARY KEY,
            pattern_json TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL
        )
    """)
    connection.commit()


def aggregate_patterns(connection, dataframe, pattern_columns):
    """
    Aggregate workload patterns within a batch and upsert them
    into SQLite so patterns spanning multiple batches are combined.
    """
    if not pattern_columns or dataframe.empty:
        return

    batch_counts = (
        dataframe[pattern_columns]
        .value_counts(dropna=False)
        .rename("occurrence_count")
        .reset_index()
    )

    rows_to_insert = []

    for row in batch_counts.itertuples(index=False, name=None):
        values = row[:-1]
        count = int(row[-1])

        pattern_json = canonical_pattern(values)
        pattern_hash = hashlib.sha256(
            pattern_json.encode("utf-8")
        ).hexdigest()

        rows_to_insert.append(
            (pattern_hash, pattern_json, count)
        )

    connection.executemany(
        """
        INSERT INTO workload_patterns
            (pattern_hash, pattern_json, occurrence_count)
        VALUES (?, ?, ?)
        ON CONFLICT(pattern_hash)
        DO UPDATE SET occurrence_count =
            occurrence_count + excluded.occurrence_count
        """,
        rows_to_insert,
    )


def export_pattern_report(connection, pattern_columns, output_path):
    """
    Export repeated patterns ordered by frequency.

    The CSV contains a JSON representation of the pattern because
    this is more robust than dynamically creating dozens of columns.
    """
    query = """
        SELECT pattern_json, occurrence_count
        FROM workload_patterns
        WHERE occurrence_count > 1
        ORDER BY occurrence_count DESC, pattern_hash ASC
        LIMIT ?
    """

    rows = connection.execute(
        query,
        (MAX_EXPORTED_PATTERNS,),
    ).fetchall()

    exported = []

    for pattern_json, count in rows:
        exported.append({
            "pattern": pattern_json,
            "occurrence_count": int(count),
            "pattern_columns": ", ".join(pattern_columns),
        })

    write_csv(
        exported,
        output_path,
        columns=[
            "pattern",
            "occurrence_count",
            "pattern_columns",
        ],
    )

    total_repeated_patterns = connection.execute(
        """
        SELECT COUNT(*)
        FROM workload_patterns
        WHERE occurrence_count > 1
        """
    ).fetchone()[0]

    total_excess_occurrences = connection.execute(
        """
        SELECT COALESCE(SUM(occurrence_count - 1), 0)
        FROM workload_patterns
        WHERE occurrence_count > 1
        """
    ).fetchone()[0]

    return {
        "unique_workload_patterns": int(
            connection.execute(
                "SELECT COUNT(*) FROM workload_patterns"
            ).fetchone()[0]
        ),
        "patterns_occurring_more_than_once": int(
            total_repeated_patterns
        ),
        "excess_occurrences_across_repeated_patterns": int(
            total_excess_occurrences
        ),
        "patterns_exported_to_csv": len(exported),
        "maximum_patterns_exported": MAX_EXPORTED_PATTERNS,
    }


# ============================================================
# 5. MAIN ANALYSIS
# ============================================================

def main():
    print_section("HPC EXECUTION AND DUPLICATE INVESTIGATION")

    # --------------------------------------------------------
    # Validate paths before beginning the analysis.
    # --------------------------------------------------------
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_PATH}\n\n"
            "Expected location:\n"
            r"C:\Users\imkir\OneDrive\Desktop\CLOUD AI"
            r"\data\processed\eagle_hpc_ml_v3.parquet"
            "\n\nCheck that the dataset exists and the filename "
            "is exactly eagle_hpc_ml_v3.parquet."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open the Parquet metadata without loading all records.
    parquet_file = pq.ParquetFile(DATASET_PATH)
    total_rows_metadata = parquet_file.metadata.num_rows
    available_columns = parquet_file.schema_arrow.names

    print(f"\nRows reported by Parquet: {total_rows_metadata:,}")
    print(f"Number of columns        : {len(available_columns)}")

    # Confirm required columns.
    required_columns = [
        "nodes_used",
        "processors_used",
        "wallclock_used_seconds",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing_required:
        raise ValueError(
            "Required columns are missing from the dataset: "
            + ", ".join(missing_required)
        )

    execution_columns = [
        column
        for column in EXECUTION_COLUMNS
        if column in available_columns
    ]

    resource_columns = [
        column
        for column in RESOURCE_COLUMNS
        if column in available_columns
    ]

    pattern_columns = [
        column
        for column in PATTERN_CANDIDATES
        if column in available_columns
    ]

    print(f"Execution columns: {execution_columns}")
    print(f"Resource columns : {resource_columns}")
    print(f"Pattern columns  : {pattern_columns}")

    if not pattern_columns:
        raise ValueError(
            "No workload-pattern columns were found. "
            "Check PATTERN_CANDIDATES against the dataset schema."
        )

    # --------------------------------------------------------
    # Initialize counters.
    # --------------------------------------------------------
    total_rows_processed = 0

    execution_counter = {
        column: Counter()
        for column in execution_columns
    }

    resource_counter = {
        column: {
            "missing": 0,
            "zero": 0,
            "negative": 0,
            "positive": 0,
            "valid": 0,
        }
        for column in resource_columns
    }

    # Additional useful counters for nodes_used.
    nodes_used_counter = {
        "missing": 0,
        "zero": 0,
        "negative": 0,
        "positive": 0,
    }

    negative_rows_written = 0
    first_negative_batch = True

    # Start each run with a clean temporary pattern database.
    if PATTERN_DB.exists():
        PATTERN_DB.unlink()

    connection = sqlite3.connect(str(PATTERN_DB))
    initialize_pattern_database(connection)

    try:
        # ----------------------------------------------------
        # Process Parquet records in batches.
        # ----------------------------------------------------
        print_section("PROCESSING DATASET IN BATCHES")

        for batch_number, record_batch in enumerate(
            parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=available_columns,
            ),
            start=1,
        ):
            dataframe = record_batch.to_pandas()
            batch_rows = len(dataframe)
            total_rows_processed += batch_rows

            # -----------------------------------------------
            # A. Execution status distributions
            # -----------------------------------------------
            for column in execution_columns:
                values = (
                    dataframe[column]
                    .astype("string")
                    .fillna("<MISSING>")
                )

                counts = values.value_counts(dropna=False)

                for value, count in counts.items():
                    execution_counter[column][str(value)] += int(count)

            # -----------------------------------------------
            # B. Missing, zero, negative and positive resources
            # -----------------------------------------------
            for column in resource_columns:
                numeric_values = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

                missing_mask = numeric_values.isna()
                zero_mask = numeric_values.eq(0).fillna(False)
                negative_mask = numeric_values.lt(0).fillna(False)
                positive_mask = numeric_values.gt(0).fillna(False)

                resource_counter[column]["missing"] += int(
                    missing_mask.sum()
                )
                resource_counter[column]["zero"] += int(
                    zero_mask.sum()
                )
                resource_counter[column]["negative"] += int(
                    negative_mask.sum()
                )
                resource_counter[column]["positive"] += int(
                    positive_mask.sum()
                )
                resource_counter[column]["valid"] += int(
                    numeric_values.notna().sum()
                )

                if column == "nodes_used":
                    nodes_used_counter["missing"] += int(
                        missing_mask.sum()
                    )
                    nodes_used_counter["zero"] += int(
                        zero_mask.sum()
                    )
                    nodes_used_counter["negative"] += int(
                        negative_mask.sum()
                    )
                    nodes_used_counter["positive"] += int(
                        positive_mask.sum()
                    )

            # -----------------------------------------------
            # C. Save negative runtime records incrementally
            # -----------------------------------------------
            runtime_values = pd.to_numeric(
                dataframe[RUNTIME_COLUMN],
                errors="coerce",
            )

            negative_mask = runtime_values.lt(0).fillna(False)

            if negative_mask.any():
                negative_rows = dataframe.loc[
                    negative_mask
                ].copy()

                negative_rows.to_csv(
                    NEGATIVE_REPORT,
                    mode="w" if first_negative_batch else "a",
                    header=first_negative_batch,
                    index=False,
                )

                negative_rows_written += len(negative_rows)
                first_negative_batch = False

            # -----------------------------------------------
            # D. Aggregate repeated workload patterns
            # -----------------------------------------------
            aggregate_patterns(
                connection,
                dataframe,
                pattern_columns,
            )

            # Commit each batch to avoid holding all updates
            # in an uncommitted transaction.
            connection.commit()

            print(
                f"Batch {batch_number:>4}: "
                f"{total_rows_processed:,} / "
                f"{total_rows_metadata:,} rows processed"
            )

            # Release batch objects before reading the next batch.
            del dataframe
            del record_batch

    finally:
        connection.commit()

    try:
        # ----------------------------------------------------
        # 6. Build execution status report.
        # ----------------------------------------------------
        execution_records = []

        for column, counts in execution_counter.items():
            for value, count in sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                execution_records.append({
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "percentage_of_dataset": (
                        round(
                            count / total_rows_processed * 100,
                            6,
                        )
                        if total_rows_processed
                        else 0.0
                    ),
                })

        write_csv(
            execution_records,
            EXECUTION_REPORT,
            columns=[
                "column",
                "value",
                "count",
                "percentage_of_dataset",
            ],
        )

        # ----------------------------------------------------
        # 7. Build zero/missing/negative resource report.
        # ----------------------------------------------------
        resource_records = []

        for column, counts in resource_counter.items():
            for category in [
                "missing",
                "zero",
                "negative",
                "positive",
                "valid",
            ]:
                count = counts[category]

                resource_records.append({
                    "resource_column": column,
                    "category": category,
                    "count": int(count),
                    "percentage_of_dataset": (
                        round(
                            count / total_rows_processed * 100,
                            6,
                        )
                        if total_rows_processed
                        else 0.0
                    ),
                })

        write_csv(
            resource_records,
            ZERO_REPORT,
            columns=[
                "resource_column",
                "category",
                "count",
                "percentage_of_dataset",
            ],
        )

        # If no negative runtime records exist, still create a
        # header-only CSV so the report is always present.
        if first_negative_batch:
            empty_negative_df = pd.DataFrame(
                columns=available_columns
            )
            empty_negative_df.to_csv(
                NEGATIVE_REPORT,
                index=False,
            )

        # ----------------------------------------------------
        # 8. Export cross-batch workload-pattern statistics.
        # ----------------------------------------------------
        duplicate_summary = export_pattern_report(
            connection,
            pattern_columns,
            DUPLICATE_REPORT,
        )

        # ----------------------------------------------------
        # 9. Construct final JSON summary.
        # ----------------------------------------------------
        nodes_used_numeric = None

        # Re-read only nodes_used to calculate its descriptive
        # statistics without retaining the entire dataset.
        nodes_used_values = []

        for record_batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["nodes_used"],
        ):
            series = pd.to_numeric(
                record_batch.column("nodes_used").to_pandas(),
                errors="coerce",
            )

            valid_values = series.dropna()

            if not valid_values.empty:
                nodes_used_values.append(
                    valid_values.describe()[
                        ["mean", "std", "min", "max"]
                    ].to_dict()
                )

        # Combine batch-level statistics using streaming formulas.
        # This avoids keeping all individual values in memory.
        count = 0
        mean = 0.0
        m2 = 0.0
        minimum = None
        maximum = None

        for record_batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["nodes_used"],
        ):
            values = pd.to_numeric(
                record_batch.column("nodes_used").to_pandas(),
                errors="coerce",
            ).dropna().to_numpy(dtype=np.float64)

            if values.size == 0:
                continue

            batch_count = int(values.size)
            batch_mean = float(values.mean())
            batch_m2 = float(
                np.sum((values - batch_mean) ** 2)
            )

            batch_minimum = float(values.min())
            batch_maximum = float(values.max())

            if minimum is None or batch_minimum < minimum:
                minimum = batch_minimum

            if maximum is None or batch_maximum > maximum:
                maximum = batch_maximum

            if count == 0:
                count = batch_count
                mean = batch_mean
                m2 = batch_m2
            else:
                delta = batch_mean - mean
                combined_count = count + batch_count

                m2 += (
                    batch_m2
                    + delta * delta * count * batch_count
                    / combined_count
                )

                mean += delta * batch_count / combined_count
                count = combined_count

        standard_deviation = (
            float(np.sqrt(m2 / (count - 1)))
            if count > 1
            else None
        )

        nodes_used_numeric = {
            "valid_count": int(count),
            "mean": float(mean) if count else None,
            "standard_deviation": standard_deviation,
            "minimum": minimum,
            "maximum": maximum,
        }

        # ----------------------------------------------------
        # 10. Save final investigation report.
        # ----------------------------------------------------
        report = {
            "dataset_path": str(DATASET_PATH),
            "project_root": str(PROJECT_ROOT),
            "rows_reported_by_parquet": int(total_rows_metadata),
            "rows_processed": int(total_rows_processed),
            "row_count_matches_parquet_metadata": (
                total_rows_processed == total_rows_metadata
            ),
            "batch_size": BATCH_SIZE,
            "columns_available": available_columns,
            "execution_columns_analyzed": execution_columns,
            "resource_columns_analyzed": resource_columns,
            "workload_pattern_columns": pattern_columns,
            "execution_status_counts": {
                column: dict(counts)
                for column, counts in execution_counter.items()
            },
            "resource_quality": resource_counter,
            "nodes_used_quality": nodes_used_counter,
            "nodes_used_statistics": nodes_used_numeric,
            "negative_runtime_rows_written": int(
                negative_rows_written
            ),
            "workload_pattern_analysis": duplicate_summary,
            "reports": {
                "execution_status": str(EXECUTION_REPORT),
                "zero_resource_analysis": str(ZERO_REPORT),
                "workload_patterns": str(DUPLICATE_REPORT),
                "negative_runtime_rows": str(NEGATIVE_REPORT),
                "summary_json": str(JSON_REPORT),
            },
            "interpretation_notes": [
                (
                    "A zero resource value is counted separately "
                    "from a missing value."
                ),
                (
                    "Repeated workload patterns are not necessarily "
                    "duplicate job submissions."
                ),
                (
                    "No dataset rows were deleted or modified."
                ),
                (
                    "Negative runtime values require investigation "
                    "before training a runtime prediction model."
                ),
                (
                    "This script does not determine whether a job "
                    "was successfully executed; inspect the meaning "
                    "of the state and executed_job fields."
                ),
            ],
        }

        with open(JSON_REPORT, "w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

    finally:
        connection.close()

        # Remove the temporary SQLite database after successful
        # report generation. If the run fails earlier, it remains
        # available for troubleshooting.
        if JSON_REPORT.exists() and PATTERN_DB.exists():
            PATTERN_DB.unlink(missing_ok=True)

    # --------------------------------------------------------
    # 11. Display final summary.
    # --------------------------------------------------------
    print_section("ANALYSIS COMPLETE")

    print(f"Rows processed: {total_rows_processed:,}")
    print(
        "Matches Parquet metadata:",
        total_rows_processed == total_rows_metadata,
    )

    print("\nResource quality:")
    for column, counts in resource_counter.items():
        print(
            f"  {column}: "
            f"missing={counts['missing']:,}, "
            f"zero={counts['zero']:,}, "
            f"negative={counts['negative']:,}, "
            f"positive={counts['positive']:,}"
        )

    print(f"\nNegative runtime rows: {negative_rows_written:,}")

    print("\nWorkload-pattern summary:")
    for key, value in duplicate_summary.items():
        print(f"  {key}: {value:,}")

    print("\nGenerated reports:")
    for output_path in [
        EXECUTION_REPORT,
        ZERO_REPORT,
        DUPLICATE_REPORT,
        NEGATIVE_REPORT,
        JSON_REPORT,
    ]:
        print(f"  {output_path}")

    print("\nImportant: repeated workload patterns are not proof of")
    print("duplicate jobs. Do not delete rows based on this report alone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise