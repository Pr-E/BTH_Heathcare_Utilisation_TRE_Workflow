"""Stage 01: schema-aware ingestion of approved TRE extracts.

This stage intentionally performs *no analytical cleaning or derivation*.
Its responsibilities are limited to:

1. reading each approved source file;
2. applying configured source-to-canonical column mappings;
3. validating the canonical schema;
4. preserving a stable column order; and
5. writing aggregate QA summaries.

Keeping ingestion separate from cleaning makes source issues visible and makes
future TRE extract refreshes easier to audit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_pipeline_config, source_dir, output_dir
from .mapping import apply_column_mapping
from .schemas import TABLE_SCHEMAS
from .qa import basic_table_summary, print_table_summary, save_records
from bth_analysis.audit import (
    dataframe_preview,
    metric,
    save_stage_summary,
    section,
    stage_footer,
    stage_header,
)


def _expected_schema(table_key: str, table_cfg: dict[str, Any]) -> list[str]:
    """Return the table's canonical expected column order."""
    configured = table_cfg.get("columns")
    if configured:
        return list(configured)
    if table_key not in TABLE_SCHEMAS:
        raise KeyError(f"No canonical schema registered for table {table_key!r}.")
    return list(TABLE_SCHEMAS[table_key])


def validate_schema(
    df: pd.DataFrame,
    table_key: str,
    table_cfg: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate required columns after source-to-canonical mapping.

    Missing required columns always stop the workflow.  Extra columns can be
    either blocked (strict mode) or retained/ignored according to configuration.
    """
    expected = _expected_schema(table_key, table_cfg)
    actual = list(df.columns)

    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    order_matches = actual == expected

    if missing:
        raise ValueError(f"{table_key}: missing required canonical columns: {missing}")

    if strict and extra:
        raise ValueError(
            f"{table_key}: unexpected columns under strict schema mode: {extra}. "
            "Either update the approved schema/mapping or set strict_schema=false "
            "for a documented exploratory run."
        )

    return {"missing": missing, "extra": extra, "order_matches": order_matches}


def align_columns(
    df: pd.DataFrame,
    table_key: str,
    table_cfg: dict[str, Any],
    *,
    preserve_extra: bool = False,
) -> pd.DataFrame:
    """Place canonical fields in a deterministic order."""
    expected = _expected_schema(table_key, table_cfg)
    if preserve_extra:
        extra = [c for c in df.columns if c not in expected]
        return df[expected + extra].copy()
    return df[expected].copy()


def _read_source(path: Path, table_cfg: dict[str, Any], low_memory: bool) -> pd.DataFrame:
    """Read CSV or Parquet according to the table configuration."""
    file_format = str(table_cfg.get("format", path.suffix.lstrip(".") or "csv")).lower()
    if file_format in {"csv", "txt"}:
        csv_cfg = table_cfg.get("csv", {}) or {}
        return pd.read_csv(
            path,
            low_memory=low_memory,
            encoding=csv_cfg.get("encoding", "utf-8"),
            sep=csv_cfg.get("sep", ","),
        )
    if file_format in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported source format {file_format!r} for {path.name}")


def _write_ingested(df: pd.DataFrame, path: Path) -> None:
    """Write a canonical CSV consumed by all downstream stages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_ingestion(config_path: str | Path = "config/pipeline_tre.yaml") -> list[dict[str, Any]]:
    """Ingest all configured TRE tables into the canonical layer."""
    config = load_pipeline_config(config_path)
    src_dir = source_dir(config)
    out_dir = output_dir(config, "ingested_dir")
    qa_dir = output_dir(config, "qa_dir")

    strict_schema = bool(config.get("ingestion", {}).get("strict_schema", True))
    preserve_extra = bool(config.get("ingestion", {}).get("preserve_extra_columns", False))
    low_memory = bool(config.get("ingestion", {}).get("csv", {}).get("low_memory", False))
    overwrite = bool(config.get("outputs", {}).get("overwrite", True))

    stage_header(
        "01",
        "TRE SOURCE INGESTION + SCHEMA MAPPING",
        purpose=(
            "Read the six approved BTH extracts, apply configured source-to-canonical column names, "
            "validate the required schema and write a deterministic canonical copy. "
            "No cleaning, imputation or analytical derivation is performed here."
        ),
        inputs=[src_dir],
        outputs=[out_dir, qa_dir],
    )

    if not src_dir.exists():
        raise FileNotFoundError(f"TRE source directory not found: {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    qa_records: list[dict[str, Any]] = []

    for table_key, table_cfg in config["tables"].items():
        filename = table_cfg["filename"]
        source_path = src_dir / filename
        # Downstream stages use the configured filename regardless of source format.
        # For Parquet sources, write a CSV with the same stem for deterministic reuse.
        target_name = table_cfg.get("canonical_filename", f"{Path(filename).stem}.csv")
        target_path = out_dir / target_name

        if not source_path.exists():
            raise FileNotFoundError(f"{table_key}: TRE source file not found: {source_path}")
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"{table_key}: target exists and overwrite=False: {target_path}"
            )

        raw = _read_source(source_path, table_cfg, low_memory=low_memory)
        mapped = apply_column_mapping(raw, table_cfg)
        schema = validate_schema(mapped, table_key, table_cfg, strict=strict_schema)
        aligned = align_columns(
            mapped,
            table_key,
            table_cfg,
            preserve_extra=preserve_extra,
        )

        summary = basic_table_summary(aligned, table_key, table_cfg)
        summary.update({
            "source_filename": filename,
            "canonical_filename": target_name,
            "schema_missing_columns": len(schema["missing"]),
            "schema_extra_columns": len(schema["extra"]),
            "source_order_matched_after_mapping": bool(schema["order_matches"]),
            "status": "PASS",
        })
        qa_records.append(summary)

        _write_ingested(aligned, target_path)
        print_table_summary(summary, prefix="PASS")
        print(f"  mapped source -> canonical schema; output={target_name}")
        print(
            f"  schema required-missing={summary['schema_missing_columns']}; "
            f"extra-retained/reported={summary['schema_extra_columns']}; "
            f"canonical order matched={summary['source_order_matched_after_mapping']}"
        )

    save_records(qa_records, qa_dir / "01_ingestion_summary.csv")

    qa_df = pd.DataFrame(qa_records)
    total_rows = int(qa_df["rows"].sum()) if not qa_df.empty else 0
    schema_missing_total = int(qa_df["schema_missing_columns"].sum()) if not qa_df.empty else 0
    schema_extra_total = int(qa_df["schema_extra_columns"].sum()) if not qa_df.empty else 0

    section("STAGE 01 KEY FINDINGS")
    metric("tables successfully ingested", f"{len(qa_records)}/{len(config['tables'])}")
    metric("source rows read across all extracts", f"{total_rows:,}")
    metric("required canonical columns missing", schema_missing_total)
    metric("extra source columns reported", schema_extra_total)
    dataframe_preview(
        qa_df,
        columns=[
            "table", "rows", "columns", "unique_patients", "missing_patient_ids",
            "schema_missing_columns", "schema_extra_columns", "status",
        ],
        max_rows=12,
    )

    audit_dir = config["_project_root"] / "outputs" / "audit"
    summary_path = save_stage_summary(
        audit_dir,
        stage_key="ingestion",
        stage_code="01",
        title="TRE source ingestion + schema mapping",
        status="PASS",
        key_findings={
            "tables_ingested": len(qa_records),
            "source_rows_across_extracts": total_rows,
            "required_columns_missing_total": schema_missing_total,
            "extra_columns_reported_total": schema_extra_total,
        },
        qa_files=[qa_dir / "01_ingestion_summary.csv"],
        warnings=[
            "Stage 01 preserves source content; cleaning/missingness treatment starts in Stage 02.",
            "Extra columns are retained because strict_schema=false; review unexpected fields in the QA summary."
        ],
        config_path=config_path,
    )
    stage_footer(
        stage_key="ingestion",
        audit_dir=audit_dir,
        summary_path=summary_path,
        qa_files=[qa_dir / "01_ingestion_summary.csv"],
    )
    return qa_records
