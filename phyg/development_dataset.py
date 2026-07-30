"""Strict paired LOLv2-Synthetic Development datasets; no Test/Eval paths."""

import hashlib
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def load_frozen_manifest(path):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    sections = {"train": [], "validation": []}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line in ("[train]", "[validation]"):
            current = line[1:-1]
        elif line and not line.startswith("#"):
            if current is None:
                raise ValueError("manifest entry before section")
            sections[current].append(line)
    canonical = "\n".join(
        ["[train]", *sections["train"], "[validation]", *sections["validation"]]
    )
    return sections, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_pairs(root, sections, expected_hash, actual_hash):
    root = Path(root).resolve()
    if root.name != "Train" or root.parent.name != "Synthetic":
        raise ValueError("data_root must be LOLv2/Synthetic/Train; Test/Eval is forbidden")
    low_names = sorted(path.name for path in (root / "Low").iterdir() if path.is_file())
    normal_names = sorted(path.name for path in (root / "Normal").iterdir() if path.is_file())
    if low_names != normal_names:
        raise ValueError("sorted Low/Normal filename sets differ")
    train, validation = sections["train"], sections["validation"]
    if len(train) != 810 or len(validation) != 90:
        raise ValueError("frozen split must be exactly 810/90")
    if set(train) & set(validation) or set(train) | set(validation) != set(low_names):
        raise ValueError("train/validation are not an exact disjoint partition")
    if actual_hash != expected_hash:
        raise ValueError(f"split SHA-256 mismatch: {actual_hash}")


class DevelopmentPairDataset(Dataset):
    def __init__(self, root, filenames, crop_size=None):
        self.root = Path(root)
        self.filenames = list(filenames)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        low = Image.open(self.root / "Low" / name).convert("RGB")
        gt = Image.open(self.root / "Normal" / name).convert("RGB")
        if low.size != gt.size:
            raise ValueError(f"geometry mismatch: {name}")
        if self.crop_size is not None:
            width, height = low.size
            size = self.crop_size
            if width < size or height < size:
                raise ValueError(f"image smaller than crop: {name}")
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            low, gt = TF.crop(low, top, left, size, size), TF.crop(gt, top, left, size, size)
            if random.random() < 0.5:
                low, gt = TF.hflip(low), TF.hflip(gt)
            if random.random() < 0.5:
                low, gt = TF.vflip(low), TF.vflip(gt)
        return TF.to_tensor(low), TF.to_tensor(gt), name
