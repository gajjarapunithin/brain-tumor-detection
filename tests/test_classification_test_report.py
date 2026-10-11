import pytest

from src.data.dataset import CLASS_NAMES
from scripts.evaluate_classification_test import build_official_test_report


def test_official_test_report_uses_class_order_and_supports():
    report = build_official_test_report(
        [0, 0, 1, 2, 3], [0, 1, 1, 2, 0], class_names=CLASS_NAMES
    )
    assert report["class_names"] == list(CLASS_NAMES)
    assert list(report["per_class"]) == list(CLASS_NAMES)
    assert [report["per_class"][name]["support"] for name in CLASS_NAMES] == [2, 1, 1, 1]
    assert report["confusion_matrix"] == [
        [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]
    ]
    assert report["row_normalized_confusion_matrix"][0] == [0.5, 0.5, 0.0, 0.0]
    assert report["row_normalized_confusion_matrix"][3] == [1.0, 0.0, 0.0, 0.0]


def test_official_test_report_uses_zero_for_absent_or_unpredicted_classes():
    report = build_official_test_report([0, 0], [0, 0], class_names=CLASS_NAMES)
    assert report["per_class"]["glioma"]["recall"] == 1.0
    assert report["per_class"]["meningioma"]["support"] == 0
    assert report["row_normalized_confusion_matrix"][1] == [0.0, 0.0, 0.0, 0.0]
    assert report["per_class"]["meningioma"]["precision"] == 0.0


@pytest.mark.parametrize("targets,predictions", [([], []), ([0], [4]), ([0], [0, 1])])
def test_official_test_report_rejects_invalid_rows(targets, predictions):
    with pytest.raises(ValueError):
        build_official_test_report(targets, predictions, class_names=CLASS_NAMES)
