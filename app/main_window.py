"""NeuroScan AI desktop workspace."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

from app.services import (
    ANALYSIS_ROOT, APP_ICON, CLASSIFICATION_CHECKPOINT, HISTORY_PATH, PROJECT_ROOT,
    SEGMENTATION_CHECKPOINT, HistoryService, InferenceError, export_artifact,
    validate_image,
)
from app.widgets import MRIWorkspace, ResultsPanel
from app.worker import InferenceWorker

LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuroScan AI · MRI Research Workstation")
        self.setWindowIcon(QIcon(str(APP_ICON)))
        self.resize(1480, 930)
        self.setMinimumSize(1120, 720)
        self.setAcceptDrops(True)
        self.history = HistoryService()
        self.selected_image: Path | None = None
        self.dimensions: tuple[int, int] | None = None
        self.current_result: dict | None = None
        self.worker: InferenceWorker | None = None
        self._build_ui()
        self._refresh_history()
        self._update_model_status()

    def _build_ui(self) -> None:
        shell = QWidget()
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.setCentralWidget(shell)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(205)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(15, 22, 15, 18)
        brand_row = QWidget()
        brand_layout = QHBoxLayout(brand_row)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(9)
        brand_icon = QLabel()
        brand_icon.setPixmap(QPixmap(str(APP_ICON)).scaled(
            40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        brand_layout.addWidget(brand_icon)
        brand = QLabel("NeuroScan AI")
        brand.setObjectName("brand")
        brand.setWordWrap(True)
        brand_layout.addWidget(brand, 1)
        side_layout.addWidget(brand_row)
        subtitle = QLabel("MRI RESEARCH WORKSTATION")
        subtitle.setObjectName("mutedLabel")
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(32)
        self.nav_buttons: list[QPushButton] = []
        for index, title in enumerate(("Overview", "Analyze MRI", "Analysis History", "About & Methodology")):
            button = QPushButton(title)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, page=index: self._show_page(page))
            side_layout.addWidget(button)
            self.nav_buttons.append(button)
        side_layout.addStretch(1)
        self.backend_status = QLabel("●  Checking model files…")
        self.backend_status.setWordWrap(True)
        self.backend_status.setObjectName("mutedLabel")
        side_layout.addWidget(self.backend_status)
        side_layout.addWidget(QLabel("Research build · 1.0"))
        shell_layout.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(22, 18, 22, 18)
        content_layout.setSpacing(14)
        header = QFrame()
        header.setObjectName("headerPanel")
        header_layout = QHBoxLayout(header)
        self.page_heading = QLabel("Overview")
        self.page_heading.setObjectName("sectionTitle")
        self.device_label = QLabel("Device · Auto-detected at analysis")
        self.device_label.setObjectName("mutedLabel")
        new_analysis = QPushButton("＋  New Analysis")
        new_analysis.setObjectName("primaryButton")
        new_analysis.clicked.connect(lambda: self._show_page(1))
        header_layout.addWidget(self.page_heading, 1)
        header_layout.addWidget(self.device_label)
        header_layout.addWidget(new_analysis)
        content_layout.addWidget(header)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._overview_page())
        self.pages.addWidget(self._analysis_page())
        self.pages.addWidget(self._history_page())
        self.pages.addWidget(self._about_page())
        content_layout.addWidget(self.pages, 1)
        shell_layout.addWidget(content, 1)
        self._show_page(0)

    def _overview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        hero = QFrame()
        hero.setObjectName("panel")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(32, 30, 32, 30)
        eyebrow = QLabel("NEUROSCAN AI  /  MRI WORKSPACE")
        eyebrow.setObjectName("eyebrow")
        hero_layout.addWidget(eyebrow)
        title = QLabel("Brain MRI analysis,\nmade inspectable.")
        title.setStyleSheet("font-size: 30pt; font-weight: 700; color: #f3f1e9;")
        hero_layout.addWidget(title)
        description = QLabel(
            "Run the trained U-Net localization model and full-image ResNet50 classifier. "
            "Review the original scan, predicted mask, overlay, and model scores together."
        )
        description.setObjectName("mutedLabel")
        description.setWordWrap(True)
        hero_layout.addWidget(description)
        choose = QPushButton("Choose MRI Image")
        choose.setObjectName("primaryButton")
        choose.setMaximumWidth(230)
        choose.clicked.connect(self._choose_image)
        hero_layout.addSpacing(12)
        hero_layout.addWidget(choose)
        hint = QLabel("Or drag a supported JPG, PNG, BMP, or TIFF image into this window.")
        hint.setObjectName("mutedLabel")
        hero_layout.addWidget(hint)
        layout.addWidget(hero, 1)
        cards = QHBoxLayout()
        for title_text, body in (
            ("Segmentation", "U-Net estimates tumor-region localization."),
            ("Classification", "ResNet50 scores four trained image classes."),
            ("Review", "Inspect image, mask, overlay, and saved result files."),
        ):
            card = QFrame()
            card.setObjectName("panel")
            box = QVBoxLayout(card)
            heading = QLabel(title_text)
            heading.setObjectName("sectionTitle")
            copy = QLabel(body)
            copy.setObjectName("mutedLabel")
            copy.setWordWrap(True)
            box.addWidget(heading)
            box.addWidget(copy)
            cards.addWidget(card, 1)
        layout.addLayout(cards)
        layout.addStretch(1)
        return page

    def _analysis_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.choose_button = QPushButton("Choose MRI")
        self.choose_button.clicked.connect(self._choose_image)
        self.run_button = QPushButton("▶  Run Analysis")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_analysis)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._clear_current)
        toolbar.addWidget(self.choose_button)
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addStretch(1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.analysis_status = QLabel("Choose an MRI image to begin.")
        self.analysis_status.setObjectName("mutedLabel")
        toolbar.addWidget(self.analysis_status, 1)
        toolbar.addWidget(self.progress)
        layout.addLayout(toolbar)

        body = QHBoxLayout()
        self.workspace = MRIWorkspace()
        image_panel = QFrame()
        image_panel.setObjectName("imagePanel")
        image_layout = QVBoxLayout(image_panel)
        image_layout.addWidget(self.workspace)
        body.addWidget(image_panel, 7)
        self.results_panel = ResultsPanel()
        self.results_panel.setMinimumWidth(305)
        self.results_panel.setMaximumWidth(390)
        body.addWidget(self.results_panel, 3)
        layout.addLayout(body, 1)

        exports = QHBoxLayout()
        self.export_overlay = QPushButton("Save Overlay")
        self.export_mask = QPushButton("Save Mask")
        self.export_json = QPushButton("Export JSON")
        self.open_folder = QPushButton("Open Results Folder")
        self.copy_summary = QPushButton("Copy Summary")
        for button, callback in (
            (self.export_overlay, lambda: self._export("overlay")),
            (self.export_mask, lambda: self._export("predicted_mask")),
            (self.export_json, lambda: self._export("result_json")),
            (self.open_folder, self._open_output_folder),
            (self.copy_summary, self._copy_summary),
        ):
            button.clicked.connect(callback)
            exports.addWidget(button)
        for button in (self.export_overlay, self.export_mask, self.export_json, self.open_folder, self.copy_summary):
            button.setEnabled(False)
        layout.addLayout(exports)
        return page

    def _history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel("Completed analyses are stored locally. Select a record to reopen its saved images.")
        intro.setObjectName("mutedLabel")
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.history_list = QListWidget()
        self.history_list.currentRowChanged.connect(self._select_history)
        row.addWidget(self.history_list, 1)
        self.history_detail = QLabel("No analysis selected")
        self.history_detail.setWordWrap(True)
        self.history_detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_detail.setObjectName("panel")
        self.history_detail.setMinimumWidth(270)
        row.addWidget(self.history_detail, 1)
        layout.addLayout(row, 1)
        controls = QHBoxLayout()
        open_selected = QPushButton("Open Selected Analysis")
        open_selected.clicked.connect(self._open_selected_history)
        clear = QPushButton("Clear History…")
        clear.clicked.connect(self._clear_history)
        controls.addWidget(open_selected)
        controls.addWidget(clear)
        controls.addStretch(1)
        layout.addLayout(controls)
        return page

    def _about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        panel = QFrame()
        panel.setObjectName("panel")
        box = QVBoxLayout(panel)
        title = QLabel("About & Methodology")
        title.setObjectName("sectionTitle")
        text = QLabel(
            "NeuroScan AI is a local research-use desktop interface for the project’s trained BRISC2025 models.\n\n"
            "• U-Net produces a binary localization estimate. Its training annotations cover tumor-bearing classes; "
            "a mask alone does not establish tumor presence.\n"
            "• ResNet50 classifies the full MRI image as glioma, meningioma, pituitary, or no tumor.\n"
            "• Scores are model softmax outputs, not clinically calibrated probabilities.\n"
            "• Inference uses the existing CLI and hardware service. Images and results are not transmitted.\n\n"
            "Research-use prototype. AI predictions may be incorrect and are not a medical diagnosis. "
            "Interpret results with qualified medical professionals."
        )
        text.setWordWrap(True)
        text.setObjectName("mutedLabel")
        box.addWidget(title)
        box.addWidget(text)
        box.addStretch(1)
        layout.addWidget(panel, 1)
        return page

    def _show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.page_heading.setText(("Overview", "Analyze MRI", "Analysis History", "About & Methodology")[index])
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        if index == 2:
            self._refresh_history()

    def _update_model_status(self) -> None:
        missing = [name for name, path in (
            ("U-Net", SEGMENTATION_CHECKPOINT), ("ResNet50", CLASSIFICATION_CHECKPOINT),
        ) if not path.is_file()]
        if missing:
            self.backend_status.setText("●  Model checkpoint missing\n" + ", ".join(missing))
            self.backend_status.setStyleSheet("color: #ff8282")
        else:
            self.backend_status.setText("●  Checkpoints available\nDevice checked per run")
            self.backend_status.setStyleSheet("color: #70dca4")

    def _choose_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Choose MRI image", str(self.selected_image.parent if self.selected_image else PROJECT_ROOT),
            "MRI images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if file_name:
            self._set_image(Path(file_name))

    def _set_image(self, path: Path) -> None:
        try:
            dimensions = validate_image(path)
        except InferenceError as exc:
            QMessageBox.warning(self, "Image not supported", str(exc))
            return
        self.selected_image = path.resolve()
        self.dimensions = dimensions
        self.current_result = None
        self.workspace.set_source(self.selected_image, dimensions)
        self.workspace.mask.set_placeholder("Run analysis to generate the predicted mask")
        self.workspace.overlay.set_placeholder("Run analysis to generate the overlay")
        self.workspace.compare_mask.set_placeholder("Mask not generated")
        self.workspace.compare_overlay.set_placeholder("Overlay not generated")
        self.results_panel.reset()
        self.analysis_status.setText("Image ready for analysis.")
        self.run_button.setEnabled(True)
        self._set_export_enabled(False)
        self._show_page(1)

    def _run_analysis(self) -> None:
        if not self.selected_image:
            return
        self.run_button.setEnabled(False)
        self.choose_button.setEnabled(False)
        self.progress.show()
        self.analysis_status.setText("Analyzing MRI with the trained models…")
        self.results_panel.status.setText("Preparing models and analyzing image…")
        self.worker = InferenceWorker(str(self.selected_image), self)
        self.worker.completed.connect(self._analysis_succeeded)
        self.worker.failed.connect(self._analysis_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _analysis_succeeded(self, payload: dict) -> None:
        self.current_result = payload
        outputs = payload["outputs"]
        self.workspace.set_results(outputs["predicted_mask"], outputs["overlay"])
        self.results_panel.show_result(payload)
        self.analysis_status.setText("Analysis completed. Results saved locally.")
        self.device_label.setText(f"Device · {payload.get('device', 'unknown').upper()}")
        try:
            self.history.record(payload)
            self._refresh_history()
        except OSError:
            LOGGER.exception("Could not persist completed inference in local history")
            QMessageBox.warning(self, "History unavailable", "Analysis completed, but the local history record could not be saved.")
        self._set_export_enabled(True)

    def _analysis_failed(self, message: str) -> None:
        LOGGER.error("Inference error: %s", message)
        self.analysis_status.setText("Analysis failed. Your selected image is still available to retry.")
        self.results_panel.status.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis failed", message)

    def _worker_finished(self) -> None:
        self.progress.hide()
        self.run_button.setEnabled(self.selected_image is not None)
        self.choose_button.setEnabled(True)
        self.worker = None

    def _clear_current(self) -> None:
        self.selected_image = None
        self.dimensions = None
        self.current_result = None
        self.workspace.filename.setText("No image selected")
        for canvas in (self.workspace.original, self.workspace.mask, self.workspace.overlay,
                       self.workspace.compare_original, self.workspace.compare_mask,
                       self.workspace.compare_overlay):
            canvas.set_placeholder("Select an MRI image to begin")
        self.results_panel.reset()
        self.analysis_status.setText("Choose an MRI image to begin.")
        self.run_button.setEnabled(False)
        self._set_export_enabled(False)

    def _set_export_enabled(self, enabled: bool) -> None:
        for button in (self.export_overlay, self.export_mask, self.export_json, self.open_folder, self.copy_summary):
            button.setEnabled(enabled)

    def _export(self, key: str) -> None:
        if not self.current_result:
            return
        source = Path(self.current_result["outputs"][key])
        default_name = source.name
        destination, _ = QFileDialog.getSaveFileName(self, "Save analysis artifact", default_name)
        if not destination:
            return
        try:
            saved = export_artifact(source, destination)
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Export complete", f"Saved file:\n{saved}")

    def _open_output_folder(self) -> None:
        if self.current_result:
            folder = Path(self.current_result["outputs"]["result_json"]).parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _copy_summary(self) -> None:
        if not self.current_result:
            return
        result = self.current_result
        classification = result["classification"]
        summary = (
            f"Prediction: {classification['predicted_class'].replace('_', ' ').title()} "
            f"(model score {classification['scores'][classification['predicted_class']]:.1%})\n"
            f"Device: {result.get('device', 'unknown')}\n"
            f"Segmentation foreground: {result['segmentation']['mask_pixel_fraction']:.2%} of pixels"
        )
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(summary)
        self.analysis_status.setText("Prediction summary copied to clipboard.")

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        for entry in self.history.list_entries():
            display = f"{entry.get('filename', 'MRI')}   ·   {entry.get('predicted_class', 'unknown').replace('_', ' ').title()}\n{entry.get('timestamp', '')}  |  {entry.get('device', 'unknown')}"
            self.history_list.addItem(QListWidgetItem(display))

    def _select_history(self, row: int) -> None:
        entries = self.history.list_entries()
        if row < 0 or row >= len(entries):
            self.history_detail.setText("No analysis selected")
            return
        entry = entries[row]
        output_paths = entry.get("outputs", {})
        available = [f"{name}: {'available' if Path(path).is_file() else 'file missing'}"
                     for name, path in output_paths.items()]
        self.history_detail.setText(
            f"{entry.get('filename')}\n\nPrediction: {entry.get('predicted_class')}\n"
            f"Date: {entry.get('timestamp')}\nDevice: {entry.get('device')}\n\n" + "\n".join(available)
        )

    def _open_selected_history(self) -> None:
        row = self.history_list.currentRow()
        entries = self.history.list_entries()
        if row < 0 or row >= len(entries):
            return
        record = entries[row]
        source = Path(record.get("input_image", ""))
        try:
            if source.is_file():
                self._set_image(source)
            else:
                self.selected_image = None
                self.current_result = None
                self._show_page(1)
                self.analysis_status.setText("Original MRI file is no longer available; saved outputs may still be viewed.")
            mask_path = record.get("outputs", {}).get("predicted_mask")
            overlay_path = record.get("outputs", {}).get("overlay")
            if mask_path and Path(mask_path).is_file() and overlay_path and Path(overlay_path).is_file():
                self.workspace.mask.set_image(mask_path)
                self.workspace.overlay.set_image(overlay_path)
                self.workspace.compare_mask.set_image(mask_path)
                self.workspace.compare_overlay.set_image(overlay_path)
            if source.is_file() and mask_path and Path(mask_path).is_file():
                self.workspace.set_results(mask_path, overlay_path)
            json_path = record.get("outputs", {}).get("result_json")
            if json_path and Path(json_path).is_file():
                import json
                self.current_result = json.loads(Path(json_path).read_text(encoding="utf-8"))
                self.results_panel.show_result(self.current_result)
                self._set_export_enabled(True)
                self._show_page(1)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Could not reopen analysis", str(exc))

    def _clear_history(self) -> None:
        answer = QMessageBox.question(self, "Clear local history", "Remove all recent-analysis records? Generated images will not be deleted.")
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.history.clear()
                self._refresh_history()
                self.history_detail.setText("Analysis history is empty.")
            except OSError as exc:
                QMessageBox.critical(self, "Could not clear history", str(exc))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self._set_image(Path(url.toLocalFile()))
                event.acceptProposedAction()
                return


def configure_logging() -> None:
    log_path = PROJECT_ROOT / "results/ui_analyses/app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        encoding="utf-8",
    )
