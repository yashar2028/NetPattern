import json
from collections import defaultdict

import numpy as np
import pytest
import torch
from PIL import Image

from netpattern_engine.data import readers, sample_files
from netpattern_engine.data.dataset import ImageDataset, load_image
from netpattern_engine.data.index import build_index
from netpattern_engine.data.splits import split_samples
from netpattern_engine.data.transforms import ModelInput, build_transforms
from netpattern_engine.spec.data import IntensitySpec, SplitSpec
from netpattern_engine.spec.pipeline import _DATASET_ADAPTER
from netpattern_engine.spec.task import TaskSpec
from netpattern_engine.spec.transforms import TransformsSpec

pytest.importorskip("nibabel")
pytest.importorskip("pydicom")

CLASSIFICATION = TaskSpec(type="classification.single_label")


def _data(**raw):
    return _DATASET_ADAPTER.validate_python(raw)


def _groups_by_split(index, splits):
    seen = defaultdict(set)
    for split, members in splits.items():
        for i in members:
            seen[index.samples[i].group].add(split)
    return seen


def test_png_image_folder_with_stratified_split(tmp_path):
    root = sample_files.write_png_classification(tmp_path / "png", per_class=10)
    data = _data(format="image_folder", root=root)

    index = build_index(data, CLASSIFICATION)
    splits = split_samples(index.samples, data.split, CLASSIFICATION.type)

    assert index.classes == ["class_0", "class_1"]
    assert len(index.samples) == 20
    assert {len(splits["train"]), len(splits["val"]), len(splits["test"])} <= set(range(1, 20))
    for split in ("val", "test"):
        assert {index.samples[i].label for i in splits[split]} == {0, 1}
    assert split_samples(index.samples, data.split, CLASSIFICATION.type) == splits  # deterministic


def test_nifti_volumes_expand_to_slices_and_never_cross_splits(tmp_path):
    paths = sample_files.write_nifti_dataset(tmp_path / "nifti", volumes=12, size=32, depth=8)
    data = _data(format="image_folder", root=paths["classification"])

    index = build_index(data, CLASSIFICATION)
    splits = split_samples(index.samples, data.split, CLASSIFICATION.type)

    assert len(index.samples) == 12 * 8
    assert len({s.group for s in index.samples}) == 12
    assert all(len(found) == 1 for found in _groups_by_split(index, splits).values())
    image = load_image(index.samples[3], data, CLASSIFICATION.type)
    assert image.shape == (1, 32, 32) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0


def test_two_and_a_half_d_stacks_neighbouring_slices(tmp_path):
    paths = sample_files.write_nifti_dataset(tmp_path / "nifti", volumes=2, size=32, depth=6)
    data = _data(format="image_folder", root=paths["classification"], slices={"context": 1})

    index = build_index(data, CLASSIFICATION)
    image = load_image(index.samples[0], data, CLASSIFICATION.type)

    assert image.shape == (3, 32, 32)
    assert np.array_equal(image[0], image[1])  # first slice: the missing neighbour repeats the edge


def test_nifti_segmentation_masks_align_and_empty_slices_can_be_skipped(tmp_path):
    paths = sample_files.write_nifti_dataset(tmp_path / "nifti", volumes=4, size=32, depth=8)
    task = TaskSpec(type="segmentation.semantic")
    every = build_index(_data(format="segmentation_folders", root=paths["segmentation"]), task)
    data = _data(
        format="segmentation_folders", root=paths["segmentation"], slices={"skip_empty_masks": True}
    )
    lesion_only = build_index(data, task)

    assert every.classes == ["background", "lesion"]
    assert 0 < len(lesion_only.samples) < len(every.samples)
    transform = build_transforms(
        TransformsSpec(preset="basic"),
        task.type,
        ModelInput(3, (32, 32), 32, (0.5,) * 3, (0.5,) * 3),
        train=False,
    )
    image, mask = ImageDataset(lesion_only.samples, data, task.type, 3, 2, transform)[0]
    assert image.shape == (3, 32, 32) and mask.shape == (32, 32)
    assert mask.dtype == torch.int64 and set(mask.unique().tolist()) <= {0, 1} and mask.max() == 1


def test_dicom_series_grouped_by_patient_and_windowed(tmp_path):
    root = sample_files.write_dicom_series(tmp_path / "dicom", patients=6, slices=4, size=32)
    data = _data(format="image_folder", root=root)

    index = build_index(data, CLASSIFICATION)
    splits = split_samples(index.samples, data.split, CLASSIFICATION.type)

    assert len(index.samples) == 24
    assert {s.group for s in index.samples} == {f"patient-{i:03d}" for i in range(6)}
    assert all(len(found) == 1 for found in _groups_by_split(index, splits).values())
    image = load_image(index.samples[0], data, CLASSIFICATION.type)
    assert image.shape == (1, 32, 32) and 0.0 <= image.min() <= image.max() <= 1.0


def test_intensity_window_maps_raw_values():
    raw = np.array([-1000.0, 0.0, 40.0, 80.0, 3000.0], dtype=np.float32)
    low, high = readers.intensity_bounds(
        raw, IntensitySpec(mode="window", window_center=40, window_width=80)
    )

    mapped = readers.apply_bounds(raw, low, high)

    assert mapped.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_csv_manifests_for_multi_label_and_regression(tmp_path):
    root = tmp_path / "csv"
    root.mkdir()
    rows = ["path,label,score,patient"]
    for i in range(6):
        Image.fromarray(np.full((8, 8), i * 30, dtype=np.uint8)).save(root / f"{i}.png")
        rows.append(f"{i}.png,{'cat|dog' if i % 2 else 'cat'},{i * 1.5},p{i // 2}")
    (root / "manifest.csv").write_text("\n".join(rows))

    multi = build_index(
        _data(format="csv", root=str(root), group_column="patient"),
        TaskSpec(type="classification.multi_label"),
    )
    regression = build_index(
        _data(format="csv", root=str(root), target_columns=["score"]), TaskSpec(type="regression")
    )

    assert multi.classes == ["cat", "dog"]
    assert multi.samples[1].label == (0, 1) and multi.samples[1].group == "p0"
    assert regression.targets == ["score"] and regression.samples[2].label == (3.0,)


def _write_detection_images(root, count=3):
    (root / "images").mkdir(parents=True)
    for i in range(count):
        Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(root / "images" / f"{i}.png")


def test_coco_voc_and_yolo_boxes(tmp_path):
    detection = TaskSpec(type="detection.bbox")

    coco = tmp_path / "coco"
    _write_detection_images(coco)
    (coco / "annotations.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": f"{i}.png", "width": 60, "height": 40} for i in range(3)
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 0,
                        "bbox": [10, 5, 20, 10],
                        "category_id": 7,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 7, "name": "lesion"}],
            }
        )
    )
    coco_index = build_index(_data(format="coco", root=str(coco)), detection)
    assert coco_index.classes == ["lesion"]
    assert coco_index.samples[0].boxes == ((10, 5, 30, 15),) and coco_index.samples[
        0
    ].box_labels == (1,)

    voc = tmp_path / "voc"
    (voc / "JPEGImages").mkdir(parents=True)
    (voc / "Annotations").mkdir()
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(voc / "JPEGImages" / "a.png")
    (voc / "Annotations" / "a.xml").write_text(
        "<annotation><filename>a.png</filename><object><name>car</name>"
        "<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>20</ymax></bndbox>"
        "</object></annotation>"
    )
    voc_index = build_index(_data(format="voc", root=str(voc)), detection)
    assert voc_index.classes == ["car"] and voc_index.samples[0].boxes == ((1.0, 2.0, 30.0, 20.0),)

    yolo = tmp_path / "yolo"
    _write_detection_images(yolo, count=1)
    (yolo / "labels").mkdir()
    (yolo / "labels" / "0.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    (yolo / "classes.txt").write_text("person\n")
    yolo_index = build_index(_data(format="yolo", root=str(yolo)), detection)
    assert yolo_index.classes == ["person"]
    assert yolo_index.samples[0].boxes == ((15.0, 10.0, 45.0, 30.0),)


def test_detection_transforms_move_boxes_with_the_image():
    data = _data(format="synthetic", num_samples=4, image_size=32)
    task = TaskSpec(type="detection.bbox")
    index = build_index(data, task)
    flip = TransformsSpec(train=[{"op": "RandomHorizontalFlip", "params": {"p": 1.0}}])
    transform = build_transforms(
        flip, task.type, ModelInput(3, (32, 32), 32, (0.5,) * 3, (0.5,) * 3), train=True
    )

    _, target = ImageDataset(index.samples, data, task.type, 3, 4, transform)[0]

    x0, _, x1, _ = index.samples[0].boxes[0]
    assert target["boxes"][0, 0].item() == pytest.approx(32 - x1)
    assert target["boxes"][0, 2].item() == pytest.approx(32 - x0)


def test_predefined_split_requires_values_on_every_sample(tmp_path):
    root = sample_files.write_png_classification(tmp_path / "png", per_class=2)
    index = build_index(_data(format="image_folder", root=root), CLASSIFICATION)

    with pytest.raises(Exception, match="predefined split"):
        split_samples(index.samples, SplitSpec(strategy="predefined"), CLASSIFICATION.type)
