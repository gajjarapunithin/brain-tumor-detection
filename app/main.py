"""Launch the NeuroScan AI desktop application."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow, configure_logging
from app.services import APP_ICON


def main() -> int:
    configure_logging()
    application = QApplication(sys.argv)
    application.setApplicationName("NeuroScan AI")
    application.setOrganizationName("NeuroScan Research")
    application.setWindowIcon(QIcon(str(APP_ICON)))
    style_path = Path(__file__).parent / "styles/theme.qss"
    application.setStyleSheet(style_path.read_text(encoding="utf-8"))
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
