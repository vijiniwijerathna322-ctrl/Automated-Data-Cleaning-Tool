"""Streamlit Community Cloud entry point.

The application lives in ``src/datacleaner``; this thin launcher exists so hosts that expect
``app.py`` at the repository root keep working. With no ``DATACLEANER_API_URL`` set, the UI
runs the API inside its own process (see ``datacleaner.ui.embedded``).
"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

# Works even if the package was not pip-installed.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_path(str(SRC / "datacleaner" / "ui" / "app.py"), run_name="__main__")
