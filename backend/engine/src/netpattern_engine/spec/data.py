"""Dataset specs: where the images are, how they are read, and how they are split."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from netpattern_engine.spec.base import StrictModel


class IntensitySpec(StrictModel):
    """How raw pixel values become floats in [0, 1].

    Medical images are never squeezed into 8 bits: windowing and scaling happen in
    float32, and the choice is recorded in the spec.
    """

    # auto: 8/16-bit images are divided by their max value; DICOM uses the file's own
    # window if present; NIfTI and other float data use the percentiles below.
    mode: Literal["auto", "window", "percentile", "minmax"] = "auto"
    window_center: float | None = None
    window_width: float | None = Field(default=None, gt=0)
    percentiles: tuple[float, float] = (0.5, 99.5)

    @model_validator(mode="after")
    def _window_needs_values(self) -> "IntensitySpec":
        if self.mode == "window" and (self.window_center is None or self.window_width is None):
            raise ValueError("mode 'window' needs window_center and window_width")
        low, high = self.percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        return self


class SliceSpec(StrictModel):
    """How volumes (NIfTI, multi-frame DICOM) become 2D samples. There are no 3D CNNs."""

    # Voxel axis to slice along after reorienting to RAS (2 = axial).
    axis: Literal[0, 1, 2] = 2
    start: int | None = Field(default=None, ge=0)
    stop: int | None = Field(default=None, ge=1)
    step: int = Field(default=1, ge=1)
    # 2.5D: neighbouring slices on each side stacked as channels (1 -> 3 channels).
    context: int = Field(default=0, ge=0, le=3)
    # Segmentation: skip slices whose mask is entirely background.
    skip_empty_masks: bool = False


class SplitSpec(StrictModel):
    """Train/val/test split. Samples of the same group (volume, patient) stay together."""

    strategy: Literal["random", "stratified", "predefined"] = "stratified"
    train: float = Field(default=0.7, ge=0, le=1)
    val: float = Field(default=0.15, ge=0, le=1)
    test: float = Field(default=0.15, ge=0, le=1)
    seed: int = 42

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "SplitSpec":
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train + val + test must be 1.0, got {total:g}")
        if self.train <= 0:
            raise ValueError("train fraction must be greater than 0")
        return self


class _DatasetBase(StrictModel):
    intensity: IntensitySpec = IntensitySpec()
    slices: SliceSpec = SliceSpec()
    split: SplitSpec = SplitSpec()
    # Use at most this many samples (after slicing): handy for quick experiments.
    limit: int | None = Field(default=None, ge=1)


class _FileDataset(_DatasetBase):
    # Directory inside the sandbox, e.g. /data/datasets/oxford-pets/classification.
    root: str = Field(min_length=1)


class ImageFolderDataset(_FileDataset):
    """root/<class name>/<image files>: single-label classification."""

    format: Literal["image_folder"]


class CsvDataset(_FileDataset):
    """A CSV manifest with one row per image (classification, multi-label, regression)."""

    format: Literal["csv"]
    manifest: str = "manifest.csv"
    path_column: str = "path"
    label_column: str = "label"
    # Multi-label rows list their labels in one cell, separated by this character.
    label_separator: str = "|"
    # Regression: one or more numeric columns.
    target_columns: list[str] | None = Field(default=None, min_length=1)
    # Rows with the same group value (e.g. patient id) always land in the same split.
    group_column: str | None = None
    # Optional column with train/val/test for the "predefined" split strategy.
    split_column: str | None = None
    # Optional column selecting one slice of a volume per row.
    slice_column: str | None = None


class SegmentationFoldersDataset(_FileDataset):
    """root/images/* and root/masks/* matched by file name; masks are PNG or NIfTI."""

    format: Literal["segmentation_folders"]
    images_dir: str = "images"
    masks_dir: str = "masks"
    # Remap mask values to class indices, e.g. {"1": 1, "2": 0, "3": 255} for
    # Oxford-IIIT Pet trimaps (pet, background, border -> ignored).
    mask_mapping: dict[int, int] | None = None


class CocoDataset(_FileDataset):
    """COCO JSON boxes (detection)."""

    format: Literal["coco"]
    annotations: str = "annotations.json"
    images_dir: str = "images"


class VocDataset(_FileDataset):
    """Pascal VOC XML boxes (detection)."""

    format: Literal["voc"]
    annotations_dir: str = "Annotations"
    images_dir: str = "JPEGImages"


class YoloDataset(_FileDataset):
    """YOLO txt boxes (detection): images/, labels/ and classes.txt."""

    format: Literal["yolo"]
    images_dir: str = "images"
    labels_dir: str = "labels"


class SyntheticDataset(_DatasetBase):
    """Generated in memory: learnable toy data for any task (tests, sanity runs)."""

    format: Literal["synthetic"]
    num_samples: int = Field(default=64, ge=4, le=100_000)
    image_size: int = Field(default=64, ge=8, le=1024)
    channels: Literal[1, 3] = 3
    num_classes: int = Field(default=3, ge=2, le=50)
    seed: int = 0


DatasetSpec = Annotated[
    ImageFolderDataset
    | CsvDataset
    | SegmentationFoldersDataset
    | CocoDataset
    | VocDataset
    | YoloDataset
    | SyntheticDataset,
    Field(discriminator="format"),
]

DATASET_FORMATS: dict[str, tuple[str, ...]] = {
    "image_folder": ("classification.single_label",),
    "csv": ("classification.single_label", "classification.multi_label", "regression"),
    "segmentation_folders": ("segmentation.semantic",),
    "coco": ("detection.bbox",),
    "voc": ("detection.bbox",),
    "yolo": ("detection.bbox",),
    "synthetic": (
        "classification.single_label",
        "classification.multi_label",
        "regression",
        "segmentation.semantic",
        "detection.bbox",
    ),
}
