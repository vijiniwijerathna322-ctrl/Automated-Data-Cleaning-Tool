"""Launch the UI: ``python -m datacleaner.ui`` or the ``datacleaner-ui`` command.

Extra arguments are passed to Streamlit, e.g. ``datacleaner-ui --server.port 8501``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(cli.main())


if __name__ == "__main__":
    main()
