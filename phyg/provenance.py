"""Hashing, configuration and Git provenance helpers."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sha256_manifest(path):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(None, 1)
        records[relative.strip()] = digest.lower()
    return records


def verify_sha256_manifest(root, manifest):
    root = Path(root)
    records = load_sha256_manifest(manifest)
    for relative, expected in records.items():
        target = root / relative
        if not target.is_file():
            raise FileNotFoundError(f"SHA-256 target missing: {relative}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {relative}: {actual} != {expected}"
            )
    return records


def canonical_config_bytes(config):
    return json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def config_sha256(config):
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_name = config.pop("base", None)
    if base_name:
        base_path = (path.parent / base_name).resolve()
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        config = deep_merge(base, config)
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = config_sha256(config)
    return config


def git_commit(root="."):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def assert_development_root(path):
    path = Path(path).resolve()
    lowered = {part.lower() for part in path.parts}
    if "test" in lowered or "eval" in lowered:
        raise ValueError("Test/Eval path is forbidden")
    if path.name != "Train" or path.parent.name != "Synthetic":
        raise ValueError("data root must be LOLv2/Synthetic/Train")
    return path
