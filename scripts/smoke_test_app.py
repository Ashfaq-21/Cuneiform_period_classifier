"""Headless smoke test of the Streamlit app against the real model and cache.

Loads app/app.py, selects a held-out (test split) tablet, clicks Analyze, and
prints what the page shows. Requires a trained model and cached test tablets.

    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_app.py [TABLET_ID]
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def main() -> int:
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    if app.exception:
        print("app raised:", app.exception)
        return 1
    if app.error:
        print("app shows error:", [e.value for e in app.error])
        return 1
    select = app.selectbox[0]
    if len(sys.argv) > 1:
        select.select(sys.argv[1])
    print("selected tablet:", select.value, f"(of {len(select.options)} held-out tablets)")
    app.button(key="analyze_sample").click().run()
    if app.exception:
        print("app raised:", app.exception)
        return 1
    for metric in app.metric:
        print(f"{metric.label}: {metric.value}")
    print("result banner:", [s.value for s in app.success] + [e.value for e in app.error])
    return 0 if len(app.metric) >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
