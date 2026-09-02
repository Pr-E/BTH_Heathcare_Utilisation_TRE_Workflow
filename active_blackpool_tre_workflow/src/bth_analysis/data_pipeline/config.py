"""Configuration/path helpers for real-source pipeline stages inside the TRE.

Paths are configured in YAML so the workflow can run in an approved TRE
workspace without changing analytical code. The module also enforces TRE source
mode before source files are accessed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(
    config_path: str | Path = "config/pipeline_tre.yaml",
) -> dict[str, Any]:
    """Load source-pipeline YAML and attach the resolved repository root."""
    # Resolve once so every subsequent relative path has one unambiguous base.
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline configuration not found: {config_path}")

    # safe_load prevents arbitrary Python object construction from YAML.
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    # Private runtime metadata is attached after parsing; it is not persisted
    # back to the analyst-maintained YAML.
    config["_config_path"] = config_path
    config["_project_root"] = config_path.parent.parent.resolve()
    return config


def resolve_from_project(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve an absolute path or a project-relative configured path."""
    path = Path(value)
    if not path.is_absolute():
        path = config["_project_root"] / path
    return path.resolve()


def source_dir(config: dict[str, Any]) -> Path:
    """Return the approved TRE source directory and reject any non-TRE source mode."""
    mode = str(config.get("data_source", {}).get("mode", "tre")).strip().lower()
    if mode != "tre":
        raise ValueError(
            f"TRE workflow requires data_source.mode='tre'; received {mode!r}."
        )

    raw = config.get("data_source", {}).get("tre_dir")
    if not raw:
        raise ValueError(
            "data_source.tre_dir must point to the approved TRE source folder."
        )
    return resolve_from_project(config, raw)


def output_dir(config: dict[str, Any], key: str) -> Path:
    """Resolve one named source-pipeline output directory from configuration."""
    if key not in config.get("outputs", {}):
        raise KeyError(f"Pipeline output key {key!r} is not configured.")
    return resolve_from_project(config, config["outputs"][key])
