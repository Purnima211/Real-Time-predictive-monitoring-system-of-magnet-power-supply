"""
app.py
------
Entry point for the Industrial Magnet Power Supply Monitoring System.

Run with:
    python app.py
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui import MainWindow
from utils import CONFIG, DARK_STYLESHEET, LOGGER


def main() -> int:
    LOGGER.info("Starting %s", CONFIG.APP_TITLE)

    app = QApplication(sys.argv)
    app.setApplicationName(CONFIG.APP_TITLE)
    app.setOrganizationName(CONFIG.ORG_NAME)
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
