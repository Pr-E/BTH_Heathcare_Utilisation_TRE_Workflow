"""Run Stage: Descriptive/EDA.

This wrapper intentionally contains no analytical logic.  It only makes the
project ``src`` package importable when running from a checked-out TRE folder
and then calls the reviewed stage function using the version-controlled YAML
configuration.  Keeping logic out of wrapper scripts prevents hidden differences
between notebook/manual runs and the production pipeline.
"""

# ``Path`` resolves the repository root relative to this script rather than an
# analyst-specific absolute Windows/Linux location.
from pathlib import Path

# ``sys.path`` is modified only so the package can run before/without an editable
# installation.  In the TRE we still recommend ``python -m pip install -e .``.
import sys

# The project root is one directory above ``scripts``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All importable package code lives under ``src``.
SRC_DIR = PROJECT_ROOT / "src"

# Insert ``src`` at the front of the module search path only if it is not already
# present; this avoids duplicate entries during repeated interactive runs.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the single reviewed stage entry point.
from bth_analysis.analysis.descriptive import run_descriptive


# Execute only when this file is run as a script, not when imported by tests.
if __name__ == "__main__":
    # All parameters are supplied by version-controlled configuration files so a
    # rerun can be reproduced from the repository plus the approved TRE extracts.
    run_descriptive('config/workflow_tre.yaml')
