"""Qt startup and threading tests (optional dependency: requirements-ui.txt)."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.worker import InferenceWorker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_window_starts_without_fake_result_and_worker_keeps_event_loop_responsive(qapp):
    window = MainWindow()
    assert window.pages.currentIndex() == 0
    assert not window.run_button.isEnabled()
    assert window.results_panel.prediction.text() == "—"
    window.close()

    worker = InferenceWorker("unused.png")
    worker.service.run = lambda _path: (time.sleep(0.12), {"ok": True})[1]
    loop = QEventLoop()
    timer_fired = []
    worker.completed.connect(lambda result: loop.quit())
    worker.start()
    QTimer.singleShot(20, lambda: timer_fired.append(True))
    QTimer.singleShot(2000, loop.quit)
    loop.exec()
    worker.wait()
    assert timer_fired
