"""Reusable MRI viewer and result presentation widgets."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QScrollArea, QTabWidget,
    QVBoxLayout, QWidget,
)

from app.services import CLASS_NAMES


class ImageCanvas(QWidget):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        title_label = QLabel(title)
        title_label.setObjectName("mutedLabel")
        layout.addWidget(title_label)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image = QLabel("Select an MRI image to begin")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(160, 160)
        self.scroll.setWidget(self.image)
        layout.addWidget(self.scroll, 1)
        self.pixmap = QPixmap()
        self.zoom = 1.0

    def set_image(self, path: str | Path, empty_text: str = "") -> None:
        self.pixmap = QPixmap(str(path))
        self.zoom = 1.0
        if self.pixmap.isNull():
            self.image.setText(empty_text or "Image unavailable")
            self.image.setPixmap(QPixmap())
        else:
            self.image.setText("")
            self._render()

    def set_placeholder(self, text: str) -> None:
        self.pixmap = QPixmap()
        self.image.setPixmap(QPixmap())
        self.image.setText(text)

    def change_zoom(self, factor: float | None) -> None:
        if factor is None:
            self.zoom = 1.0
        else:
            self.zoom = min(4.0, max(0.25, self.zoom * factor))
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if not self.pixmap.isNull():
            self._render()

    def _render(self) -> None:
        if self.pixmap.isNull():
            return
        size = self.pixmap.size() * self.zoom
        if self.zoom == 1.0:
            size.scale(self.scroll.viewport().size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(self.pixmap.scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        ))


class MRIWorkspace(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.filename = QLabel("No image selected")
        self.filename.setObjectName("mutedLabel")
        self.zoom_out = self._button("−")
        self.zoom_in = self._button("+")
        self.fit = self._button("Fit")
        controls.addWidget(self.filename, 1)
        controls.addWidget(self.zoom_out)
        controls.addWidget(self.zoom_in)
        controls.addWidget(self.fit)
        layout.addLayout(controls)

        self.tabs = QTabWidget()
        self.original = ImageCanvas("ORIGINAL MRI")
        self.mask = ImageCanvas("PREDICTED LOCALIZATION MASK")
        self.overlay = ImageCanvas("MRI + PREDICTED MASK")
        self.comparison = QWidget()
        comparison_layout = QHBoxLayout(self.comparison)
        self.compare_original = ImageCanvas("ORIGINAL")
        self.compare_mask = ImageCanvas("MASK")
        self.compare_overlay = ImageCanvas("OVERLAY")
        comparison_layout.addWidget(self.compare_original)
        comparison_layout.addWidget(self.compare_mask)
        comparison_layout.addWidget(self.compare_overlay)
        self.tabs.addTab(self.original, "Original")
        self.tabs.addTab(self.mask, "Mask")
        self.tabs.addTab(self.overlay, "Overlay")
        self.tabs.addTab(self.comparison, "Compare")
        layout.addWidget(self.tabs, 1)
        self._canvases = (self.original, self.mask, self.overlay,
                          self.compare_original, self.compare_mask, self.compare_overlay)

    @staticmethod
    def _button(label: str):
        from PySide6.QtWidgets import QPushButton
        return QPushButton(label)

    def set_source(self, path: str | Path, dimensions: tuple[int, int]) -> None:
        self.filename.setText(f"{Path(path).name}   ·   {dimensions[0]} × {dimensions[1]} px")
        self.original.set_image(path)
        self.compare_original.set_image(path)

    def set_results(self, mask: str | Path, overlay: str | Path) -> None:
        self.mask.set_image(mask)
        self.overlay.set_image(overlay)
        self.compare_mask.set_image(mask)
        self.compare_overlay.set_image(overlay)

    def set_zoom(self, factor: float | None) -> None:
        for canvas in self._canvases:
            canvas.change_zoom(factor)


class ResultsPanel(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        title = QLabel("Analysis result")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.status = QLabel("Select an MRI to begin analysis.")
        self.status.setWordWrap(True)
        self.status.setObjectName("mutedLabel")
        layout.addWidget(self.status)
        self.prediction = QLabel("—")
        self.prediction.setObjectName("predictionCard")
        self.prediction.setWordWrap(True)
        layout.addWidget(self.prediction)
        self.score_title = QLabel("CLASS SCORES · NOT CLINICALLY CALIBRATED")
        self.score_title.setObjectName("eyebrow")
        layout.addWidget(self.score_title)
        self.bars: dict[str, QProgressBar] = {}
        self.score_labels: dict[str, QLabel] = {}
        for name in CLASS_NAMES:
            row = QHBoxLayout()
            label = QLabel(name.replace("_", " ").title())
            label.setMinimumWidth(95)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            score = QLabel("—")
            score.setMinimumWidth(48)
            score.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label)
            row.addWidget(bar, 1)
            row.addWidget(score)
            layout.addLayout(row)
            self.bars[name] = bar
            self.score_labels[name] = score
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)
        self.mask_info = QLabel("Segmentation localization: not available")
        self.mask_info.setWordWrap(True)
        layout.addWidget(self.mask_info)
        self.device_info = QLabel("Device: selected when analysis runs")
        self.device_info.setWordWrap(True)
        self.device_info.setObjectName("mutedLabel")
        layout.addWidget(self.device_info)
        layout.addStretch(1)
        self.disclaimer = QLabel(
            "Research-use prototype. Predictions may be incorrect and are not a medical diagnosis. "
            "Interpret results with a qualified medical professional. The mask is a localization estimate."
        )
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setObjectName("disclaimer")
        layout.addWidget(self.disclaimer)

    def show_result(self, payload: dict) -> None:
        classification = payload["classification"]
        predicted = classification["predicted_class"]
        score = float(classification["scores"][predicted])
        self.status.setText("Analysis completed")
        self.prediction.setText(f"{predicted.replace('_', ' ').title()}\nModel score  {score:.1%}")
        for name, bar in self.bars.items():
            value = float(classification["scores"][name])
            bar.setValue(round(value * 1000))
            self.score_labels[name].setText(f"{value:.1%}")
        fraction = float(payload["segmentation"]["mask_pixel_fraction"])
        self.mask_info.setText(
            f"Tumor localization mask available\nForeground: {fraction:.2%} of image pixels "
            "(not a tumor probability)."
        )
        message = payload.get("device_message", f"Inference executed on {payload.get('device', 'unknown')}.")
        if payload.get("device_fallback"):
            message = f"Fallback: {message}"
        self.device_info.setText(f"Device: {payload.get('device', 'unknown').upper()}\n{message}")

    def reset(self) -> None:
        self.status.setText("Select an MRI to begin analysis.")
        self.prediction.setText("—")
        for name in CLASS_NAMES:
            self.bars[name].setValue(0)
            self.score_labels[name].setText("—")
        self.mask_info.setText("Segmentation localization: not available")
        self.device_info.setText("Device: selected when analysis runs")
