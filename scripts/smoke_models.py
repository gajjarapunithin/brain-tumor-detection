"""Run small real-data model, loss, optimizer, checkpoint, and inference checks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BRISCClassificationDataset, BRISCSegmentationDataset
from src.evaluation.metrics import classification_metrics, segmentation_metrics
from src.models.classifier import TumorClassifier
from src.models.pipeline import TumorInferencePipeline
from src.models.unet import UNet
from src.training.losses import DiceBCELoss, classification_loss


def _changed(before: torch.Tensor, after: torch.Tensor) -> bool:
    return not torch.equal(before, after)


def run_smoke(output_dir: Path) -> dict[str, object]:
    torch.manual_seed(123)
    device = "cpu"
    image_size = 64
    classification = BRISCClassificationDataset(
        Config().brisc_dir / "classification_task", "train", image_size
    )
    segmentation = BRISCSegmentationDataset(
        Config().brisc_dir / "segmentation_task", "train", image_size
    )
    class_images, class_labels = next(iter(DataLoader(classification, batch_size=2, shuffle=False)))
    seg_images, seg_masks = next(iter(DataLoader(segmentation, batch_size=2, shuffle=False)))
    assert class_images.shape == (2, 1, image_size, image_size)
    assert class_labels.shape == (2,)
    assert seg_images.shape == seg_masks.shape == (2, 1, image_size, image_size)
    assert set(torch.unique(seg_masks).tolist()).issubset({0.0, 1.0})

    segmenter = UNet(in_channels=1, out_channels=1, features=(8, 16, 32)).to(device)
    classifier = TumorClassifier(num_classes=4, freeze_backbone=True, pretrained=False).to(device)
    seg_loss_fn = DiceBCELoss()
    cls_loss_fn = classification_loss()

    seg_before = next(segmenter.parameters()).detach().clone()
    seg_optimizer = torch.optim.Adam(segmenter.parameters(), lr=1e-3)
    seg_loss = seg_loss_fn(segmenter(seg_images), seg_masks)
    seg_optimizer.zero_grad()
    seg_loss.backward()
    seg_optimizer.step()
    assert _changed(seg_before, next(segmenter.parameters()).detach())

    head_parameter = classifier.backbone.fc.weight
    cls_before = head_parameter.detach().clone()
    cls_optimizer = torch.optim.SGD([head_parameter, classifier.backbone.fc.bias], lr=1e-3)
    cls_loss = cls_loss_fn(classifier(class_images), class_labels)
    cls_optimizer.zero_grad()
    cls_loss.backward()
    cls_optimizer.step()
    assert _changed(cls_before, head_parameter.detach())

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "smoke.pt"
    torch.save({"segmentation": segmenter.state_dict(), "classification": classifier.state_dict()}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded_segmenter = UNet(in_channels=1, out_channels=1, features=(8, 16, 32)).to(device)
    reloaded_classifier = TumorClassifier(num_classes=4, pretrained=False).to(device)
    reloaded_segmenter.load_state_dict(payload["segmentation"])
    reloaded_classifier.load_state_dict(payload["classification"])
    pipeline = TumorInferencePipeline(reloaded_segmenter, reloaded_classifier, device=device)
    prediction = pipeline.predict_tensor(seg_images[:1], use_roi=False)
    assert prediction["mask"].shape == (1, 1, image_size, image_size)
    assert prediction["class_probability"].shape == (1, 4)

    segmentation_score = segmentation_metrics(
        prediction["mask"][0, 0].numpy(), seg_masks[0, 0].numpy()
    )
    classification_score = classification_metrics(
        class_labels.numpy(), classifier(class_images).argmax(1).detach().numpy(), labels=[0, 1, 2, 3]
    )
    return {
        "classification_batch": list(class_images.shape),
        "segmentation_batch": list(seg_images.shape),
        "segmentation_loss": float(seg_loss.detach()),
        "classification_loss": float(cls_loss.detach()),
        "segmentation_metrics": segmentation_score,
        "classification_metrics": {key: value.tolist() if hasattr(value, "tolist") else value
                                    for key, value in classification_score.items()},
        "checkpoint": str(checkpoint),
        "inference_class_index": int(prediction["class_index"].item()),
    }


if __name__ == "__main__":
    print(run_smoke(Path("experiments/smoke")))
