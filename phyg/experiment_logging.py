"""Crash-safe experiment metrics logging without touching experiment RNG."""

import json
import os
from pathlib import Path


EPOCH_FIELDS = {
    "epoch", "global_step", "mean_train_loss", "learning_rate",
    "elapsed_seconds",
}
SUMMARY_FIELDS = {
    "config_sha256", "split_raw_sha256", "split_canonical_sha256",
    "initialization_checkpoint_sha256", "git_commit", "best_epoch",
    "best_step", "best_psnr", "best_ssim", "best_lpips",
    "current_epoch", "global_step", "completed",
    "official_test_participation",
}


class ExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.logged_epochs = self._read_logged_epochs()

    def _read_logged_epochs(self):
        epochs = set()
        if not self.metrics_path.exists():
            return epochs
        with self.metrics_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = EPOCH_FIELDS - record.keys()
                if missing:
                    raise ValueError(
                        f"metrics line {line_number} missing {sorted(missing)}"
                    )
                epoch = int(record["epoch"])
                if epoch in epochs:
                    raise ValueError(f"duplicate epoch {epoch} in metrics.jsonl")
                epochs.add(epoch)
        return epochs

    def append_epoch(self, record):
        missing = EPOCH_FIELDS - record.keys()
        if missing:
            raise ValueError(f"epoch record missing {sorted(missing)}")
        epoch = int(record["epoch"])
        if epoch in self.logged_epochs:
            return False
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.logged_epochs.add(epoch)
        return True

    def update_summary(self, summary):
        missing = SUMMARY_FIELDS - summary.keys()
        if missing:
            raise ValueError(f"summary missing {sorted(missing)}")
        temporary = self.summary_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.summary_path)


def replay_statistics(replay_type, replay, global_step):
    if replay_type == "physical":
        return {
            "physical_sample_counts": {
                key: int(replay.counts[key])
                for key in (
                    "identity", "bright_clean", "dark_noisy",
                    "warm_camera", "cool_camera",
                )
            }
        }
    if replay_type == "gamma":
        return {
            "gamma_call_count": int(global_step),
            "last_gamma": replay.last_gamma,
        }
    return {}


def make_summary(context, completed):
    best = context.get("best_metric") or {}
    return {
        "config_sha256": context["config_sha256"],
        "split_raw_sha256": context["split_raw_sha256"],
        "split_canonical_sha256": context["split_canonical_sha256"],
        "initialization_checkpoint_sha256":
            context["initialization_checkpoint_sha256"],
        "git_commit": context["git_commit"],
        "best_epoch": context.get("best_epoch"),
        "best_step": context.get("best_step"),
        "best_psnr": best.get("psnr"),
        "best_ssim": best.get("ssim"),
        "best_lpips": best.get("lpips"),
        "current_epoch": int(context["epoch"]) - 1,
        "global_step": int(context["global_step"]),
        "completed": bool(completed),
        "official_test_participation": "NONE",
    }


def print_epoch_row(record):
    validation = ""
    if "psnr" in record:
        validation = (
            f" psnr={record['psnr']:.4f} ssim={record['ssim']:.4f}"
            f" lpips={record['lpips']:.4f} best={int(record['is_best'])}"
        )
    replay = ""
    if "gamma_call_count" in record:
        replay = (
            f" gamma_n={record['gamma_call_count']}"
            f" gamma={record['last_gamma']:.2f}"
        )
    elif "physical_sample_counts" in record:
        counts = record["physical_sample_counts"]
        replay = " phys=" + "/".join(
            str(counts[key]) for key in (
                "identity", "bright_clean", "dark_noisy",
                "warm_camera", "cool_camera",
            )
        )
    print(
        f"epoch={record['epoch']:02d} step={record['global_step']:04d}"
        f" loss={record['mean_train_loss']:.6f}"
        f" lr={record['learning_rate']:.3e}"
        f" sec={record['elapsed_seconds']:.1f}{validation}{replay}",
        flush=True,
    )
