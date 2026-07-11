import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests run in dev mode with an explicit throwaway vault key unless a test
# overrides these to exercise fail-closed behavior.
os.environ.setdefault("WTRMLN_DEV_MODE", "1")
