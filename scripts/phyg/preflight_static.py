#!/usr/bin/env python3
"""Standard-library-only audit for hosts without PyTorch/PyYAML."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("C:/Users/hxy/Desktop/PhyG-PEFT")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    tag = subprocess.check_output(
        ["git", "rev-list", "-n", "1", "retinexformer-official-frozen"],
        cwd=ROOT, text=True,
    ).strip()
    author = subprocess.check_output(
        ["git", "rev-parse", "author-baseline"], cwd=ROOT, text=True
    ).strip()
    frozen = "e8086f9d6e4badf0eebf48825264a785d3aa541e"
    require(tag == frozen and author == frozen, "frozen refs changed")
    forbidden = subprocess.check_output([
        "git", "diff", "--name-only", "retinexformer-official-frozen", "--",
        "basicsr/models/archs/RetinexFormer_arch.py",
        "Enhancement/test_from_dataset.py", "basicsr/test.py",
    ], cwd=ROOT, text=True).strip()
    require(not forbidden, "official network/Test entry changed")
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "basicsr/data"], cwd=ROOT, text=True
    ).splitlines())
    local = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "basicsr/data").rglob("*") if path.is_file()
    }
    require(tracked == local and len(local) == 25, "repository incomplete")
    pairs = [
        (SOURCE / "phyg_peft/physical_degradation.py",
         ROOT / "phyg/physical_degradation.py"),
        (SOURCE / "phyg_peft/dataset_wrappers.py",
         ROOT / "phyg/development_dataset.py"),
        (SOURCE / "phyg_peft/metrics.py", ROOT / "phyg/metrics.py"),
        (SOURCE / "protocols/lolv2_synthetic_dev_split_seed20260726.txt",
         ROOT / "configs/phyg/protocols/lolv2_synthetic_dev_split_seed20260726.txt"),
    ]
    for source, target in pairs:
        require(source.read_bytes() == target.read_bytes(),
                f"exact copy differs: {target}")
    manifest = pairs[-1][1].read_text(encoding="utf-8").splitlines()
    train_index = manifest.index("[train]")
    validation_index = manifest.index("[validation]")
    train = manifest[train_index + 1:validation_index]
    validation = manifest[validation_index + 1:]
    require(len(train) == 810 and len(validation) == 90, "split count")
    require(not set(train) & set(validation), "split overlap")
    canonical = "\n".join(["[train]", *train, "[validation]", *validation])
    require(hashlib.sha256(canonical.encode()).hexdigest() ==
            "e3e73abce5405d7fa4cbb7d1f16470dc924abf97c77208ba049da1613d0a944d",
            "canonical split hash")
    require(sha256(pairs[-1][1]) ==
            "f80b8d5cebd3b8f6529b0670e91037320bf939dba6f16d8a5402861465f543d6",
            "raw split hash")
    base = (ROOT / "configs/phyg/base_seed1234.yml").read_text(encoding="utf-8")
    frozen_lines = (
        "seed: 1234", "batch_size: 8", "crop_size: 128", "epochs: 60",
        "steps_per_epoch: 101", "total_steps: 6060", "drop_last: true",
        "num_workers: 0", "lr: 1.0e-5", "T_max: 6060",
        "eta_min: 1.0e-7", "max_norm: 0.01",
        "gt_mean_correction: false",
        "initialization: pretrained_weights/LOL_v2_synthetic.pth",
    )
    for line in frozen_lines:
        require(line in base, f"base protocol missing {line}")
    arms = list((ROOT / "configs/phyg").glob("*_seed1234.yml"))
    require(len(arms) == 7, "expected base plus six train-arm configs")
    print("PASS: frozen refs and forbidden-file integrity")
    print("PASS: 25 official basicsr/data files are tracked")
    print("PASS: four exact source copies")
    print("PASS: frozen split 810/90, disjoint, raw/canonical hashes")
    print("PASS: shared base budget/init literals and six arm configs")
    print("official_test_participation=NONE")


if __name__ == "__main__":
    main()
