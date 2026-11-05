# AppWorld Beaker notes

Setup, datasets, and objectives are in the [main README](../README.md#beaker-integrations). Internals:

- **Smoke:** strict smoke checks structure and labeled data only; it does not run the agent.
- **Isolation:** each case runs in a fresh subprocess (`appworld_subprocess.py`, `appworld_case_worker.py`) because AppWorld mutates process-global state. Variants in a scenario run sequentially; cases run in parallel. Cancellation kills the worker process group; failures keep a traceback artifact.
- **Setup:** `appworld_setup.py` reuses local data or downloads the pinned assets, verifying checksums.
- **Tracing:** Beaker traces candidate execution, not scoring, and provider errors surface in evaluation runs. Worker traces are imported through the Beaker SDK's capture-adoption interfaces, so after an SDK upgrade recheck trace import as well as results.
