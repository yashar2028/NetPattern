"""Fetch real test datasets into the datasets volume (PLAN §12.4).

    python -m app.datasets.samples oxford-pets [--breeds Abyssinian,Bengal,...] [--per-breed 40]
    python -m app.datasets.samples synthetic-nifti

Oxford-IIIT Pet covers three tasks from one download:
  classification  manifest.csv (breed per image)
  segmentation    annotations/trimaps (1 pet, 2 background, 3 border -> mask_mapping)
  detection       annotations/xmls (pet head boxes, Pascal VOC format)
The image archive is in random order, so streaming stops as soon as every chosen
breed has enough images (a few hundred MB instead of 790 MB).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import tarfile
import urllib.request
from pathlib import Path

from app.services import storage

PETS_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
PETS_ANNOTATIONS = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
DEFAULT_BREEDS = ("Abyssinian", "Bengal", "Siamese", "beagle", "pug", "boxer")


def _breed(file_name: str) -> str:
    return Path(file_name).stem.rsplit("_", 1)[0]


def fetch_oxford_pets(breeds: tuple[str, ...], per_breed: int, target: Path) -> dict[str, int]:
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    counts = {breed: len(list(images_dir.glob(f"{breed}_*.jpg"))) for breed in breeds}
    needed = {breed for breed, count in counts.items() if count < per_breed}

    if needed:
        print(f"Streaming {PETS_IMAGES} until each breed has {per_breed} images …", flush=True)
        with (
            urllib.request.urlopen(PETS_IMAGES, timeout=120) as response,
            tarfile.open(fileobj=response, mode="r|gz") as archive,
        ):
            for member in archive:
                name = Path(member.name).name
                breed = _breed(name)
                if not member.isfile() or breed not in needed or not name.endswith(".jpg"):
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                data = source.read()
                if not data.startswith(b"\xff\xd8"):  # skip the few files that are not JPEGs
                    continue
                (images_dir / name).write_bytes(data)
                counts[breed] += 1
                if counts[breed] >= per_breed:
                    needed.discard(breed)
                    print(f"  {breed}: {counts[breed]} images", flush=True)
                if not needed:
                    break

    stems = {path.stem for path in images_dir.glob("*.jpg") if _breed(path.name) in breeds}
    trimaps, xmls = target / "annotations" / "trimaps", target / "annotations" / "xmls"
    trimaps.mkdir(parents=True, exist_ok=True)
    xmls.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {PETS_ANNOTATIONS} …", flush=True)
    with (
        urllib.request.urlopen(PETS_ANNOTATIONS, timeout=120) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for member in archive:
            path = Path(member.name)
            if not member.isfile() or path.stem not in stems or path.name.startswith("._"):
                continue
            if path.parent.name == "trimaps" and path.suffix == ".png":
                destination = trimaps / path.name
            elif path.parent.name == "xmls" and path.suffix == ".xml":
                destination = xmls / path.name
            else:
                continue
            source = archive.extractfile(member)
            if source is not None:
                destination.write_bytes(source.read())

    with (target / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label"])
        for stem in sorted(stems):
            writer.writerow([f"images/{stem}.jpg", _breed(stem)])
    return {
        "images": len(stems),
        "trimaps": len(list(trimaps.glob("*.png"))),
        "head_boxes": len(list(xmls.glob("*.xml"))),
    }


async def make_synthetic_nifti(out: str, volumes: int) -> int:
    """Real NIfTI files, written by the engine inside a sandbox (it has nibabel)."""
    from app.db.session import AsyncSessionLocal
    from app.workers import queue
    from app.workers.cli import follow

    async with AsyncSessionLocal() as db:
        job = await queue.enqueue(
            db,
            "make_sample_dataset",
            {"options": {"type": "nifti", "out": out, "volumes": volumes}},
        )
    print(f"submitted {job.id}; the worker writes the files", flush=True)
    finished = await follow(job.id)
    return 0 if finished is not None and finished.status.value == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.datasets.samples",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pets = commands.add_parser(
        "oxford-pets", help="Oxford-IIIT Pet subset: classification, segmentation, detection"
    )
    pets.add_argument("--breeds", default=",".join(DEFAULT_BREEDS))
    pets.add_argument("--per-breed", type=int, default=40)
    nifti = commands.add_parser(
        "synthetic-nifti", help="synthetic NIfTI volumes (classification + segmentation)"
    )
    nifti.add_argument("--volumes", type=int, default=16)
    args = parser.parse_args(argv)

    if args.command == "oxford-pets":
        target = storage.datasets_root() / "oxford-pets"
        breeds = tuple(b.strip() for b in args.breeds.split(",") if b.strip())
        summary = fetch_oxford_pets(breeds, args.per_breed, target)
        print(f"Oxford-IIIT Pet subset in {target}: {summary}")
        return 0
    return asyncio.run(make_synthetic_nifti("synthetic-nifti", args.volumes))


if __name__ == "__main__":
    sys.exit(main())
