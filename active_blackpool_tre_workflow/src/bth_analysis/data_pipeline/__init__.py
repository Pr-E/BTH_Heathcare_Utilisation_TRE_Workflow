"""Public entry points for deterministic real-source data pipeline stages."""

# Re-export the first three source-data stages for convenient programmatic use.
# Later stages are imported directly by the orchestration layer to keep this
# namespace small and explicit.
from .ingestion import run_ingestion
from .cleaning import run_cleaning
from .preprocessing import run_preprocessing

__all__ = ["run_ingestion", "run_cleaning", "run_preprocessing"]
