# Package validation

The analytical chain runs from Stage 00 preflight through Stage 09 clustering,
followed by Stage 10 release-output pre-screen.

Validation completed before packaging:

```text
Python compilation: PASS
Toy-data test suite: 8/8 PASS
Orchestrator imports: PASS
Configured TRE source path: /project/readonly
Patient/event reference-cohort mappings: explicit and validated
```

Validation commands:

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

The toy-data tests contain no real patient data.
