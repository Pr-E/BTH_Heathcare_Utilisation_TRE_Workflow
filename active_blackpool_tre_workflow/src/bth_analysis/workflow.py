"""Shared configuration, path and reproducibility helpers for the TRE workflow.

The TRE package intentionally keeps configuration outside analysis code.  The
same Python functions can therefore run against refreshed approved BTH extracts
without hard-coded analyst paths.  This module reads configuration metadata only;
it does not read patient-level source or analytical data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import subprocess
from typing import Any

import yaml


def load_workflow_config(
    path: str | Path = "config/workflow_tre.yaml",
) -> dict[str, Any]:
    """Load analytical workflow YAML and attach portable project metadata."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Workflow configuration not found: {path}")

    # safe_load is appropriate for analyst-maintained configuration because it
    # does not execute arbitrary Python constructors embedded in YAML.
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Attach runtime-only metadata used by every stage for portable paths.
    cfg["_config_path"] = path
    cfg["_project_root"] = path.parent.parent.resolve()
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    """Resolve absolute paths or project-relative configured paths."""
    path = Path(value)
    if not path.is_absolute():
        path = cfg["_project_root"] / path
    return path.resolve()


def output_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a named analytical/audit output directory from workflow config."""
    if key not in cfg.get("outputs", {}):
        raise KeyError(f"Output key {key!r} is not configured.")
    return resolve_path(cfg, cfg["outputs"][key])


def config_sha256(path: str | Path) -> str:
    """Return a content digest so every run can be tied to exact configuration."""
    path = Path(path).resolve()
    return sha256(path.read_bytes()).hexdigest()


def _safe_git_commit(project_root: Path) -> str | None:
    """Return current Git commit when available without making Git mandatory."""
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return value or None
    except Exception:
        # Some TRE environments do not include Git or may ingest an archive
        # without repository metadata; reproducibility still relies on config
        # hashes and package/version information in that case.
        return None




def _dependency_versions() -> dict[str, str | None]:
    """Return installed package versions for reproducibility without importing data."""
    distributions = [
        "bth-active-blackpool-tre",
        "pandas",
        "numpy",
        "PyYAML",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "joblib",
    ]
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_run_manifest(
    cfg: dict[str, Any],
    *,
    pipeline_config: str | Path | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Create non-patient reproducibility metadata for the current run."""
    workflow_path = Path(cfg["_config_path"])
    manifest: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "workflow_config": str(workflow_path),
        "workflow_config_sha256": config_sha256(workflow_path),
        "project_root": str(cfg["_project_root"]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _safe_git_commit(Path(cfg["_project_root"])),
        "dependency_versions": _dependency_versions(),
    }

    # Source-pipeline configuration is hashed separately because schema/mapping
    # changes may occur independently of the analytical design configuration.
    if pipeline_config is not None:
        pipeline_path = Path(pipeline_config).resolve()
        manifest["pipeline_config"] = str(pipeline_path)
        if pipeline_path.exists():
            manifest["pipeline_config_sha256"] = config_sha256(pipeline_path)
    return manifest


def write_run_manifest(
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    filename: str = "run_manifest.json",
) -> Path:
    """Persist one JSON run manifest in the configured TRE audit directory."""
    audit_dir = output_path(cfg, "audit_dir")
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / filename
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
