# Import the recipe before any test module imports tau2: it sets TAU2_DATA_DIR,
# which tau2 reads once at import time.
import tau2_bench  # noqa: F401
