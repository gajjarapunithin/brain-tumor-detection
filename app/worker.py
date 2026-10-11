"""Qt worker that keeps model inference off the GUI thread."""

from PySide6.QtCore import QThread, Signal

from app.services import InferenceService


class InferenceWorker(QThread):
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, image_path: str, parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.service = InferenceService()

    def run(self) -> None:
        try:
            self.completed.emit(self.service.run(self.image_path))
        except Exception as exc:
            self.failed.emit(str(exc))
