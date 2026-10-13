# ============================================================
# investigate_dataset.py
# HPC Eagle Dataset - Deep Investigation
#
# Purpose:
#   1. Investigate missing values
#   2. Investigate duplicate rows
#   3. Analyze requested vs actual resources
#   4. Analyze candidate ML targets
#   5. Recommend suitable prediction targets
#
# Dataset:
#   data/processed/eagle_hpc_ml_v3.parquet
#
# Outputs:
#   data/validation/dataset_investigation_report.json
#   data/validation/missing_value_analysis.csv
#   data/validation/target_analysis.csv
#   data/validation/resource_consistency_analysis.csv
#   data/validation/duplicate_hashes.sqlite
#
# ============================================================

from pathlib import Path
import hashlib
import json
import math
import sqlite3
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as patypes


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eagle_hpc_ml_v3.parquet"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "validation"
)

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    VALIDATION_DIR
    / "dataset_investigation_report.json"
)

MISSING_REPORT_PATH = (
    VALIDATION_DIR
    / "missing_value_analysis.csv"
)

TARGET_REPORT_PATH = (
    VALIDATION_DIR
    / "target_analysis.csv"
)

RESOURCE_REPORT_PATH = (
    VALIDATION_DIR
    / "resource_consistency_analysis.csv"
)

DUPLICATE_DB_PATH = (
    VALIDATION_DIR
    / "duplicate_hashes.sqlite"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 100_000


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

REQUEST_COLUMNS = [
    "nodes_req",
    "processors_req",
    "memory_req_gb",
    "memory_req_total_gb",
    "wallclock_req_seconds",
    "gpus_requested",
]

ACTUAL_COLUMNS = [
    "nodes_used",
    "processors_used",
    "wallclock_used_seconds",
]

TARGET_COLUMNS = [
    "nodes_used",
    "processors_used",
    "wallclock_used_seconds",
]

CATEGORICAL_COLUMNS = [
    "memory_req_scope",
    "partition",
    "qos",
    "python_job",
    "reframe_job",
    "job_type_hash",
    "state",
]

TEMPORAL_COLUMNS = [
    "day",
    "day_of_week",
    "hour",
    "minute",
    "submit_year",
    "submit_month",
]

DERIVED_COLUMNS = [
    "processors_req_per_node",
    "memory_req_per_node",
    "estimated_nodes_from_processors",
]

EXPECTED_COLUMNS = [
    "nodes_req",
    "processors_req",
    "memory_req_gb",
    "memory_req_scope",
    "memory_req_missing",
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
    "nodes_used",
    "processors_used",
    "wallclock_used_seconds",
    "processors_req_per_node",
    "memory_req_per_node",
    "estimated_nodes_from_processors",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_missing(value):
    """
    Safely determine whether a scalar value is missing.
    """

    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, (bool, np.bool_)):
            return bool(result)

        return False

    except Exception:
        return False


def safe_float(value):
    """
    Convert a value to float safely.
    """

    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def safe_int(value, default=0):
    """
    Convert a value to integer safely.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_json_value(value):
    """
    Convert NumPy/Pandas values into JSON-compatible values.
    """

    if isinstance(value, dict):
        return {
            str(k): safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            safe_json_value(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            safe_json_value(v)
            for v in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if math.isfinite(value):
            return value

        return None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        if math.isfinite(value):
            return value

        return None

    return value


# ============================================================
# MAIN INVESTIGATOR CLASS
# ============================================================

class DatasetInvestigator:

    def __init__(self):

        print("=" * 80)
        print("HPC EAGLE DATASET INVESTIGATION")
        print("=" * 80)

        print()
        print(f"Dataset: {DATASET_PATH}")
        print(f"Dataset exists: {DATASET_PATH.exists()}")

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{DATASET_PATH}"
            )

        # ----------------------------------------------------
        # Open Parquet
        # ----------------------------------------------------

        self.parquet_file = pq.ParquetFile(DATASET_PATH)

        self.schema = self.parquet_file.schema_arrow

        self.columns = self.schema.names

        self.total_rows = self.parquet_file.metadata.num_rows

        self.total_row_groups = (
            self.parquet_file.num_row_groups
        )

        print(f"Rows: {self.total_rows:,}")
        print(f"Columns: {len(self.columns)}")
        print(f"Row groups: {self.total_row_groups}")

        # ----------------------------------------------------
        # Validate expected columns
        # ----------------------------------------------------

        self.missing_expected_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in self.columns
        ]

        if self.missing_expected_columns:

            print()
            print("WARNING: Expected columns missing:")

            for column in self.missing_expected_columns:
                print(f"  - {column}")

        # ----------------------------------------------------
        # Detect numeric columns
        # ----------------------------------------------------

        self.numeric_columns = []

        for field in self.schema:

            if (
                patypes.is_integer(field.type)
                or patypes.is_floating(field.type)
                or patypes.is_decimal(field.type)
            ):
                self.numeric_columns.append(field.name)

        print()
        print("Numeric columns detected:")

        for column in self.numeric_columns:
            print(f"  - {column}")

        # ----------------------------------------------------
        # Statistics containers
        # ----------------------------------------------------

        self.numeric_stats = {}

        self.initialize_numeric_statistics()

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        self.missing_counts = {
            column: 0
            for column in self.columns
        }

        # ----------------------------------------------------
        # Target statistics
        # ----------------------------------------------------

        self.target_stats = {}

        self.initialize_target_statistics()

        # ----------------------------------------------------
        # Resource consistency statistics
        # ----------------------------------------------------

        self.resource_checks = {
            "nodes": {
                "both_present": 0,
                "actual_gt_requested": 0,
                "actual_lt_requested": 0,
                "actual_eq_requested": 0,
                "invalid": 0,
            },
            "processors": {
                "both_present": 0,
                "actual_gt_requested": 0,
                "actual_lt_requested": 0,
                "actual_eq_requested": 0,
                "invalid": 0,
            },
            "wallclock": {
                "both_present": 0,
                "actual_gt_requested": 0,
                "actual_lt_requested": 0,
                "actual_eq_requested": 0,
                "invalid": 0,
            },
        }

        # ----------------------------------------------------
        # Duplicate database
        # ----------------------------------------------------

        self.duplicate_connection = None

        self.initialize_duplicate_database()

        # ----------------------------------------------------
        # Counters
        # ----------------------------------------------------

        self.rows_processed = 0

        self.batches_processed = 0

        self.start_time = None

    # ========================================================
    # INITIALIZE NUMERIC STATISTICS
    # ========================================================

    def initialize_numeric_statistics(self):

        for column in self.numeric_columns:

            self.numeric_stats[column] = {
                "count": 0,
                "missing": 0,
                "zero": 0,
                "negative": 0,
                "positive": 0,
                "sum": 0.0,
                "sum_squared": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }

    # ========================================================
    # INITIALIZE TARGET STATISTICS
    # ========================================================

    def initialize_target_statistics(self):

        for column in TARGET_COLUMNS:

            self.target_stats[column] = {
                "count": 0,
                "missing": 0,
                "zero": 0,
                "negative": 0,
                "positive": 0,
                "sum": 0.0,
                "sum_squared": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }

    # ========================================================
    # INITIALIZE SQLITE DUPLICATE DATABASE
    # ========================================================

    def initialize_duplicate_database(self):

        print()
        print("Initializing duplicate investigation database...")

        # Remove stale database from a previous run.
        if DUPLICATE_DB_PATH.exists():

            try:
                DUPLICATE_DB_PATH.unlink()

            except PermissionError:

                raise PermissionError(
                    f"Cannot remove existing database:\n"
                    f"{DUPLICATE_DB_PATH}\n\n"
                    f"Make sure no previous process is using it."
                )

        self.duplicate_connection = sqlite3.connect(
            DUPLICATE_DB_PATH
        )

        cursor = self.duplicate_connection.cursor()

        # WITHOUT ROWID reduces storage overhead.
        cursor.execute(
            """
            CREATE TABLE row_hashes (
                hash BLOB PRIMARY KEY
            ) WITHOUT ROWID
            """
        )

        self.duplicate_connection.commit()

        print(
            f"Duplicate database created:\n"
            f"{DUPLICATE_DB_PATH}"
        )

    # ========================================================
    # CREATE ROW HASH
    # ========================================================

    def create_row_hash(self, row):

        """
        Create a stable 128-bit MD5 hash from the complete row.

        The hash is used only for duplicate investigation.
        No rows are deleted or modified.
        """

        parts = []

        for value in row:

            if is_missing(value):

                parts.append("<NA>")

            else:

                # Include type information to reduce accidental
                # collisions between representations.
                parts.append(
                    f"{type(value).__name__}:{repr(value)}"
                )

        canonical = "\x1f".join(parts)

        return hashlib.md5(
            canonical.encode("utf-8")
        ).digest()

    # ========================================================
    # PROCESS DUPLICATE HASHES
    # ========================================================

    def process_duplicate_hashes(self, dataframe):

        """
        Insert row hashes into SQLite.

        INSERT OR IGNORE means:
            - first occurrence is stored
            - repeated occurrence is ignored

        We calculate duplicate count at the end as:

            total rows - unique hashes
        """

        rows = dataframe.itertuples(
            index=False,
            name=None
        )

        hashes = []

        for row in rows:

            row_hash = self.create_row_hash(row)

            hashes.append((row_hash,))

        if hashes:

            cursor = self.duplicate_connection.cursor()

            cursor.executemany(
                """
                INSERT OR IGNORE INTO row_hashes(hash)
                VALUES (?)
                """,
                hashes
            )

        # Commit once per batch.
        self.duplicate_connection.commit()

    # ========================================================
    # UPDATE NUMERIC STATISTICS
    # ========================================================

    def update_numeric_statistics(
        self,
        dataframe
    ):

        for column in self.numeric_columns:

            if column not in dataframe.columns:
                continue

            series = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

            missing_mask = series.isna()

            missing_count = int(
                missing_mask.sum()
            )

            self.missing_counts[column] += missing_count

            stats = self.numeric_stats[column]

            stats["missing"] += missing_count

            valid = series[
                ~missing_mask
            ]

            if len(valid) == 0:
                continue

            values = valid.to_numpy(
                dtype=np.float64
            )

            finite_mask = np.isfinite(values)

            values = values[finite_mask]

            if len(values) == 0:
                continue

            stats["count"] += len(values)

            stats["zero"] += int(
                np.count_nonzero(values == 0)
            )

            stats["negative"] += int(
                np.count_nonzero(values < 0)
            )

            stats["positive"] += int(
                np.count_nonzero(values > 0)
            )

            stats["sum"] += float(
                np.sum(values)
            )

            stats["sum_squared"] += float(
                np.sum(values ** 2)
            )

            current_min = float(
                np.min(values)
            )

            current_max = float(
                np.max(values)
            )

            if current_min < stats["min"]:
                stats["min"] = current_min

            if current_max > stats["max"]:
                stats["max"] = current_max

    # ========================================================
    # UPDATE TARGET STATISTICS
    # ========================================================

    def update_target_statistics(
        self,
        dataframe
    ):

        for column in TARGET_COLUMNS:

            if column not in dataframe.columns:
                continue

            series = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

            missing_mask = series.isna()

            stats = self.target_stats[column]

            missing_count = int(
                missing_mask.sum()
            )

            stats["missing"] += missing_count

            valid = series[
                ~missing_mask
            ]

            if len(valid) == 0:
                continue

            values = valid.to_numpy(
                dtype=np.float64
            )

            finite_mask = np.isfinite(values)

            values = values[finite_mask]

            if len(values) == 0:
                continue

            stats["count"] += len(values)

            stats["zero"] += int(
                np.count_nonzero(values == 0)
            )

            stats["negative"] += int(
                np.count_nonzero(values < 0)
            )

            stats["positive"] += int(
                np.count_nonzero(values > 0)
            )

            stats["sum"] += float(
                np.sum(values)
            )

            stats["sum_squared"] += float(
                np.sum(values ** 2)
            )

            current_min = float(
                np.min(values)
            )

            current_max = float(
                np.max(values)
            )

            if current_min < stats["min"]:
                stats["min"] = current_min

            if current_max > stats["max"]:
                stats["max"] = current_max

    # ========================================================
    # RESOURCE CONSISTENCY CHECKS
    # ========================================================

    def process_resource_checks(
        self,
        dataframe
    ):

        comparisons = [
            (
                "nodes",
                "nodes_req",
                "nodes_used",
            ),
            (
                "processors",
                "processors_req",
                "processors_used",
            ),
            (
                "wallclock",
                "wallclock_req_seconds",
                "wallclock_used_seconds",
            ),
        ]

        for resource, requested_col, actual_col in comparisons:

            if (
                requested_col not in dataframe.columns
                or actual_col not in dataframe.columns
            ):
                continue

            requested = pd.to_numeric(
                dataframe[requested_col],
                errors="coerce"
            )

            actual = pd.to_numeric(
                dataframe[actual_col],
                errors="coerce"
            )

            both_present = (
                requested.notna()
                & actual.notna()
            )

            req_values = requested[both_present].to_numpy(
                dtype=np.float64
            )

            actual_values = actual[both_present].to_numpy(
                dtype=np.float64
            )

            finite_mask = (
                np.isfinite(req_values)
                & np.isfinite(actual_values)
            )

            req_values = req_values[finite_mask]
            actual_values = actual_values[finite_mask]

            self.resource_checks[resource]["both_present"] += (
                len(req_values)
            )

            if len(req_values) == 0:
                continue

            self.resource_checks[resource]["actual_gt_requested"] += int(
                np.count_nonzero(
                    actual_values > req_values
                )
            )

            self.resource_checks[resource]["actual_lt_requested"] += int(
                np.count_nonzero(
                    actual_values < req_values
                )
            )

            self.resource_checks[resource]["actual_eq_requested"] += int(
                np.count_nonzero(
                    actual_values == req_values
                )
            )

            self.resource_checks[resource]["invalid"] += int(
                np.count_nonzero(
                    (req_values < 0)
                    | (actual_values < 0)
                )
            )

    # ========================================================
    # PROCESS ONE BATCH
    # ========================================================

    def process_batch(
        self,
        dataframe
    ):

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        for column in self.columns:

            if column not in dataframe.columns:
                continue

            if column in self.numeric_columns:

                # Numeric missing values are already counted
                # during numeric statistics.
                continue

            missing = int(
                dataframe[column].isna().sum()
            )

            self.missing_counts[column] += missing

        # ----------------------------------------------------
        # Numeric statistics
        # ----------------------------------------------------

        self.update_numeric_statistics(
            dataframe
        )

        # ----------------------------------------------------
        # Target statistics
        # ----------------------------------------------------

        self.update_target_statistics(
            dataframe
        )

        # ----------------------------------------------------
        # Resource consistency
        # ----------------------------------------------------

        self.process_resource_checks(
            dataframe
        )

        # ----------------------------------------------------
        # Duplicate investigation
        # ----------------------------------------------------

        self.process_duplicate_hashes(
            dataframe
        )

    # ========================================================
    # FINALIZE NUMERIC STATISTICS
    # ========================================================

    def finalize_statistics(
        self,
        statistics
    ):

        final = {}

        for column, stats in statistics.items():

            count = stats["count"]

            if count > 0:

                mean = (
                    stats["sum"]
                    / count
                )

                variance = (
                    stats["sum_squared"]
                    / count
                    - mean ** 2
                )

                # Floating-point rounding can produce tiny
                # negative variance.
                variance = max(
                    0.0,
                    variance
                )

                std = math.sqrt(
                    variance
                )

                minimum = stats["min"]

                maximum = stats["max"]

            else:

                mean = None
                std = None
                minimum = None
                maximum = None

            final[column] = {
                "count": int(count),
                "missing": int(stats["missing"]),
                "zero": int(stats["zero"]),
                "negative": int(stats["negative"]),
                "positive": int(stats["positive"]),
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
            }

        return final

    # ========================================================
    # CREATE MISSING VALUE REPORT
    # ========================================================

    def create_missing_report(self):

        rows = []

        for column in self.columns:

            missing = int(
                self.missing_counts.get(
                    column,
                    0
                )
            )

            percentage = (
                missing
                / self.total_rows
                * 100
                if self.total_rows > 0
                else 0
            )

            rows.append(
                {
                    "column": column,
                    "missing_count": missing,
                    "missing_percentage": percentage,
                    "non_missing_count": (
                        self.total_rows
                        - missing
                    ),
                }
            )

        report = pd.DataFrame(rows)

        report = report.sort_values(
            "missing_percentage",
            ascending=False
        )

        report.to_csv(
            MISSING_REPORT_PATH,
            index=False
        )

        print()
        print(
            f"Missing-value report written to:\n"
            f"{MISSING_REPORT_PATH}"
        )

        return report

    # ========================================================
    # CREATE TARGET REPORT
    # ========================================================

    def create_target_report(self):

        final_stats = self.finalize_statistics(
            self.target_stats
        )

        rows = []

        for column, stats in final_stats.items():

            rows.append(
                {
                    "target": column,
                    "valid_count": stats["count"],
                    "missing_count": stats["missing"],
                    "zero_count": stats["zero"],
                    "negative_count": stats["negative"],
                    "positive_count": stats["positive"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )

        report = pd.DataFrame(rows)

        report.to_csv(
            TARGET_REPORT_PATH,
            index=False
        )

        print()
        print(
            f"Target analysis written to:\n"
            f"{TARGET_REPORT_PATH}"
        )

        return report

    # ========================================================
    # CREATE RESOURCE CONSISTENCY REPORT
    # ========================================================

    def create_resource_report(self):

        rows = []

        for resource, stats in self.resource_checks.items():

            total = stats["both_present"]

            if total > 0:

                greater_pct = (
                    stats["actual_gt_requested"]
                    / total
                    * 100
                )

                less_pct = (
                    stats["actual_lt_requested"]
                    / total
                    * 100
                )

                equal_pct = (
                    stats["actual_eq_requested"]
                    / total
                    * 100
                )

            else:

                greater_pct = 0
                less_pct = 0
                equal_pct = 0

            rows.append(
                {
                    "resource": resource,
                    "rows_with_request_and_actual": total,
                    "actual_gt_requested": stats[
                        "actual_gt_requested"
                    ],
                    "actual_lt_requested": stats[
                        "actual_lt_requested"
                    ],
                    "actual_eq_requested": stats[
                        "actual_eq_requested"
                    ],
                    "invalid_rows": stats[
                        "invalid"
                    ],
                    "actual_gt_requested_pct": greater_pct,
                    "actual_lt_requested_pct": less_pct,
                    "actual_eq_requested_pct": equal_pct,
                }
            )

        report = pd.DataFrame(rows)

        report.to_csv(
            RESOURCE_REPORT_PATH,
            index=False
        )

        print()
        print(
            f"Resource consistency report written to:\n"
            f"{RESOURCE_REPORT_PATH}"
        )

        return report

    # ========================================================
    # DUPLICATE SUMMARY
    # ========================================================

    def get_duplicate_summary(self):

        cursor = self.duplicate_connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM row_hashes
            """
        )

        unique_rows = int(
            cursor.fetchone()[0]
        )

        duplicate_rows = (
            self.total_rows
            - unique_rows
        )

        duplicate_percentage = (
            duplicate_rows
            / self.total_rows
            * 100
            if self.total_rows > 0
            else 0
        )

        return {
            "total_rows": self.total_rows,
            "unique_rows": unique_rows,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
        }

    # ========================================================
    # TARGET RECOMMENDATION
    # ========================================================

    def create_target_recommendation(
        self,
        final_target_stats
    ):

        recommendations = []

        for target in TARGET_COLUMNS:

            stats = final_target_stats.get(
                target,
                {}
            )

            count = stats.get(
                "count",
                0
            )

            missing = stats.get(
                "missing",
                0
            )

            negative = stats.get(
                "negative",
                0
            )

            if self.total_rows > 0:

                coverage = (
                    count
                    / self.total_rows
                    * 100
                )

                missing_pct = (
                    missing
                    / self.total_rows
                    * 100
                )

            else:

                coverage = 0
                missing_pct = 0

            # -----------------------------------------------
            # Basic suitability
            # -----------------------------------------------

            if count == 0:

                recommendation = "REJECT"

                reason = (
                    "No valid observations."
                )

            elif negative > 0:

                recommendation = "INVESTIGATE"

                reason = (
                    "Contains negative values. "
                    "Check data semantics before modeling."
                )

            elif coverage < 80:

                recommendation = "INVESTIGATE"

                reason = (
                    "Less than 80% valid coverage."
                )

            elif missing_pct > 20:

                recommendation = "INVESTIGATE"

                reason = (
                    "More than 20% missing values."
                )

            else:

                recommendation = "GOOD CANDIDATE"

                reason = (
                    "Good valid coverage and suitable "
                    "numeric target structure."
                )

            recommendations.append(
                {
                    "target": target,
                    "valid_percentage": coverage,
                    "missing_percentage": missing_pct,
                    "recommendation": recommendation,
                    "reason": reason,
                }
            )

        return recommendations

    # ========================================================
    # CREATE COMPLETE REPORT
    # ========================================================

    def create_report(
        self,
        missing_report,
        target_report,
        resource_report,
        duplicate_summary,
        final_numeric_stats,
        final_target_stats,
        target_recommendations
    ):

        total_missing_cells = int(
            sum(
                self.missing_counts.values()
            )
        )

        report = {

            "dataset": {

                "path": str(DATASET_PATH),

                "rows": self.total_rows,

                "columns": len(self.columns),

                "row_groups": self.total_row_groups,

                "column_names": self.columns,

                "file_size_bytes": (
                    DATASET_PATH.stat().st_size
                ),
            },

            "schema": {

                "numeric_columns": self.numeric_columns,

                "categorical_columns": [
                    column
                    for column in CATEGORICAL_COLUMNS
                    if column in self.columns
                ],

                "temporal_columns": [
                    column
                    for column in TEMPORAL_COLUMNS
                    if column in self.columns
                ],

                "derived_columns": [
                    column
                    for column in DERIVED_COLUMNS
                    if column in self.columns
                ],
            },

            "missing_values": {

                "total_missing_cells": (
                    total_missing_cells
                ),

                "columns_with_missing_values": [
                    column
                    for column in self.columns
                    if self.missing_counts.get(
                        column,
                        0
                    ) > 0
                ],
            },

            "duplicates": duplicate_summary,

            "numeric_statistics": final_numeric_stats,

            "target_statistics": final_target_stats,

            "resource_consistency": self.resource_checks,

            "target_recommendations": (
                target_recommendations
            ),

            "ml_role_recommendation": {

                "prediction_targets": [
                    "nodes_used",
                    "processors_used",
                    "wallclock_used_seconds",
                ],

                "requested_features": [
                    column
                    for column in REQUEST_COLUMNS
                    if column in self.columns
                ],

                "categorical_features": [
                    column
                    for column in CATEGORICAL_COLUMNS
                    if column in self.columns
                ],

                "temporal_features": [
                    column
                    for column in TEMPORAL_COLUMNS
                    if column in self.columns
                ],

                "derived_features": [
                    column
                    for column in DERIVED_COLUMNS
                    if column in self.columns
                ],

                "possible_post_execution_columns": [
                    "state",
                    "executed_job",
                    "nodes_used",
                    "processors_used",
                    "wallclock_used_seconds",
                ],

                "warning": (
                    "Actual execution columns should not be "
                    "used as pre-execution prediction inputs "
                    "for the corresponding target. Columns "
                    "such as state and executed_job must be "
                    "verified for availability at submission "
                    "time before using them as predictors."
                ),
            },

            "investigation_status": {

                "dataset_read": True,

                "missing_values_analyzed": True,

                "duplicates_analyzed": True,

                "target_statistics_analyzed": True,

                "resource_consistency_analyzed": True,

                "rows_modified": False,

                "rows_deleted": False,
            },
        }

        report = safe_json_value(report)

        with open(
            REPORT_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

        print()
        print(
            f"Investigation report written to:\n"
            f"{REPORT_PATH}"
        )

        return report

    # ========================================================
    # PROCESS ENTIRE DATASET
    # ========================================================

    def process_dataset(self):

        print()
        print("=" * 80)
        print("PROCESSING DATASET")
        print("=" * 80)

        print(
            f"Batch size: {BATCH_SIZE:,}"
        )

        self.start_time = time.time()

        # ----------------------------------------------------
        # Iterate over Parquet batches
        # ----------------------------------------------------

        for batch in self.parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            dataframe = batch.to_pandas()

            self.process_batch(
                dataframe
            )

            batch_rows = len(dataframe)

            self.rows_processed += batch_rows

            self.batches_processed += 1

            # ------------------------------------------------
            # Progress information
            # ------------------------------------------------

            elapsed = (
                time.time()
                - self.start_time
            )

            if elapsed > 0:

                rows_per_second = (
                    self.rows_processed
                    / elapsed
                )

            else:

                rows_per_second = 0

            progress = (
                self.rows_processed
                / self.total_rows
                * 100
                if self.total_rows > 0
                else 0
            )

            print(
                f"\r"
                f"Progress: {progress:6.2f}% | "
                f"Rows: {self.rows_processed:,}/"
                f"{self.total_rows:,} | "
                f"Batch: {self.batches_processed:,} | "
                f"Speed: {rows_per_second:,.0f} rows/s",
                end="",
                flush=True
            )

        print()
        print()

        elapsed = (
            time.time()
            - self.start_time
        )

        print(
            f"Processing complete in "
            f"{elapsed:.2f} seconds."
        )

        print(
            f"Rows processed: "
            f"{self.rows_processed:,}"
        )

        print(
            f"Batches processed: "
            f"{self.batches_processed:,}"
        )

    # ========================================================
    # RUN COMPLETE INVESTIGATION
    # ========================================================

    def run(self):

        try:

            # ------------------------------------------------
            # Process dataset
            # ------------------------------------------------

            self.process_dataset()

            # ------------------------------------------------
            # Finalize statistics
            # ------------------------------------------------

            final_numeric_stats = (
                self.finalize_statistics(
                    self.numeric_stats
                )
            )

            final_target_stats = (
                self.finalize_statistics(
                    self.target_stats
                )
            )

            # ------------------------------------------------
            # Reports
            # ------------------------------------------------

            missing_report = (
                self.create_missing_report()
            )

            target_report = (
                self.create_target_report()
            )

            resource_report = (
                self.create_resource_report()
            )

            # ------------------------------------------------
            # Duplicate summary
            # ------------------------------------------------

            duplicate_summary = (
                self.get_duplicate_summary()
            )

            # ------------------------------------------------
            # Target recommendation
            # ------------------------------------------------

            target_recommendations = (
                self.create_target_recommendation(
                    final_target_stats
                )
            )

            # ------------------------------------------------
            # Complete JSON report
            # ------------------------------------------------

            self.create_report(
                missing_report,
                target_report,
                resource_report,
                duplicate_summary,
                final_numeric_stats,
                final_target_stats,
                target_recommendations
            )

            # ------------------------------------------------
            # Console summary
            # ------------------------------------------------

            self.print_summary(
                duplicate_summary,
                final_target_stats,
                target_recommendations
            )

        finally:

            self.close()

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    def print_summary(
        self,
        duplicate_summary,
        final_target_stats,
        target_recommendations
    ):

        print()
        print("=" * 80)
        print("INVESTIGATION SUMMARY")
        print("=" * 80)

        print()
        print("DATASET")
        print("-" * 80)

        print(
            f"Rows:              {self.total_rows:,}"
        )

        print(
            f"Rows processed:    {self.rows_processed:,}"
        )

        print(
            f"Columns:           {len(self.columns)}"
        )

        print(
            f"Row groups:        {self.total_row_groups}"
        )

        print()
        print("MISSING VALUES")
        print("-" * 80)

        total_missing = sum(
            self.missing_counts.values()
        )

        print(
            f"Total missing cells: "
            f"{total_missing:,}"
        )

        print()

        for column in self.columns:

            missing = self.missing_counts.get(
                column,
                0
            )

            if missing > 0:

                percentage = (
                    missing
                    / self.total_rows
                    * 100
                    if self.total_rows > 0
                    else 0
                )

                print(
                    f"{column:35s}"
                    f"{missing:12,}"
                    f" ({percentage:6.2f}%)"
                )

        print()
        print("DUPLICATES")
        print("-" * 80)

        print(
            f"Total rows:        "
            f"{duplicate_summary['total_rows']:,}"
        )

        print(
            f"Unique rows:       "
            f"{duplicate_summary['unique_rows']:,}"
        )

        print(
            f"Duplicate rows:    "
            f"{duplicate_summary['duplicate_rows']:,}"
        )

        print(
            f"Duplicate %:       "
            f"{duplicate_summary['duplicate_percentage']:.4f}%"
        )

        print()
        print("TARGET ANALYSIS")
        print("-" * 80)

        for target, stats in final_target_stats.items():

            print()
            print(f"Target: {target}")

            print(
                f"  Valid:       "
                f"{stats['count']:,}"
            )

            print(
                f"  Missing:     "
                f"{stats['missing']:,}"
            )

            print(
                f"  Zero:        "
                f"{stats['zero']:,}"
            )

            print(
                f"  Negative:    "
                f"{stats['negative']:,}"
            )

            print(
                f"  Mean:        "
                f"{stats['mean']}"
            )

            print(
                f"  Std:         "
                f"{stats['std']}"
            )

            print(
                f"  Min:         "
                f"{stats['min']}"
            )

            print(
                f"  Max:         "
                f"{stats['max']}"
            )

        print()
        print("TARGET RECOMMENDATIONS")
        print("-" * 80)

        for item in target_recommendations:

            print(
                f"{item['target']:30s}"
                f" -> "
                f"{item['recommendation']}"
            )

            print(
                f"   Valid: "
                f"{item['valid_percentage']:.2f}%"
            )

            print(
                f"   Missing: "
                f"{item['missing_percentage']:.2f}%"
            )

            print(
                f"   Reason: "
                f"{item['reason']}"
            )

        print()
        print("=" * 80)
        print("INVESTIGATION COMPLETE")
        print("=" * 80)

        print()
        print("Generated files:")

        print(
            f"  1. {REPORT_PATH}"
        )

        print(
            f"  2. {MISSING_REPORT_PATH}"
        )

        print(
            f"  3. {TARGET_REPORT_PATH}"
        )

        print(
            f"  4. {RESOURCE_REPORT_PATH}"
        )

        print(
            f"  5. {DUPLICATE_DB_PATH}"
        )

        print()
        print(
            "IMPORTANT: No dataset rows were modified or deleted."
        )

    # ========================================================
    # CLOSE RESOURCES
    # ========================================================

    def close(self):

        if self.duplicate_connection is not None:

            try:
                self.duplicate_connection.commit()

            except Exception:
                pass

            try:
                self.duplicate_connection.close()

            except Exception:
                pass

            self.duplicate_connection = None


# ============================================================
# MAIN
# ============================================================

def main():

    investigator = DatasetInvestigator()

    investigator.run()


if __name__ == "__main__":
    main()