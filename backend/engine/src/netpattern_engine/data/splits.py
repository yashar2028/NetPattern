"""Train/val/test splits that keep every group (volume, patient) in a single split."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from netpattern_engine.data.index import Sample
from netpattern_engine.errors import EngineError
from netpattern_engine.spec.data import SplitSpec

SPLITS = ("train", "val", "test")


def split_samples(samples: list[Sample], spec: SplitSpec, task_type: str) -> dict[str, list[int]]:
    """Indices of the samples in each split."""
    if spec.strategy == "predefined":
        return _predefined(samples)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample.group or sample.path].append(i)

    stratify = spec.strategy == "stratified" and task_type == "classification.single_label"
    if stratify:
        buckets: dict[object, list[str]] = defaultdict(list)
        for name, members in groups.items():
            label = Counter(samples[i].label for i in members).most_common(1)[0][0]
            buckets[label].append(name)
        bucket_list = [
            sorted(names) for _, names in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        ]
    else:
        bucket_list = [sorted(groups)]

    rng = random.Random(spec.seed)
    assigned: dict[str, list[int]] = {split: [] for split in SPLITS}
    for names in bucket_list:
        rng.shuffle(names)
        for split, chosen in zip(SPLITS, _allocate(names, spec)):
            for name in chosen:
                assigned[split].extend(groups[name])

    for split in SPLITS:
        assigned[split].sort()
    return assigned


def _allocate(names: list[str], spec: SplitSpec) -> tuple[list[str], list[str], list[str]]:
    total = len(names)
    n_val = round(total * spec.val)
    n_test = round(total * spec.test)
    # Small buckets still contribute to val/test when those splits are requested.
    if spec.val > 0 and n_val == 0 and total >= 3:
        n_val = 1
    if spec.test > 0 and n_test == 0 and total >= 3:
        n_test = 1
    n_train = max(total - n_val - n_test, 1 if total else 0)
    n_val = min(n_val, total - n_train)
    n_test = total - n_train - n_val
    return names[:n_train], names[n_train : n_train + n_val], names[n_train + n_val :]


def _predefined(samples: list[Sample]) -> dict[str, list[int]]:
    assigned: dict[str, list[int]] = {split: [] for split in SPLITS}
    missing = 0
    for i, sample in enumerate(samples):
        if sample.split in assigned:
            assigned[sample.split].append(i)
        else:
            missing += 1
    if missing:
        raise EngineError(
            f"{missing} sample(s) have no train/val/test value; the predefined split needs a "
            "split column on every row"
        )
    if not assigned["train"]:
        raise EngineError("the predefined split has no training samples")
    return assigned
