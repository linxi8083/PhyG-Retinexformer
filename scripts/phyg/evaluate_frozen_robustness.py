#!/usr/bin/env python3
"""Evaluate six frozen Retinexformer arms on Development robustness protocols."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.models.archs import define_network
from phyg.checkpoint import trusted_torch_load
from phyg.development_dataset import DevelopmentPairDataset
from phyg.metrics import AuthorValidationMetrics
from phyg.provenance import sha256_file
from phyg.robustness_evaluation import (
    MODEL_NAMES,
    checkpoint_audit,
    load_robustness_config,
    load_validation,
    protocol_sha256,
    reject_forbidden_path,
)
from phyg.robustness_protocol import (
    PROTOCOLS,
    SOURCE_EVALUATOR_SHA256,
    SOURCE_PHYSICAL_SHA256,
    apply_frozen_degradation,
    preset_seed,
)

METRICS = ("psnr", "ssim", "lpips")


def mean_sd(values):
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def load_models(config, checkpoints, device):
    models = {}
    for name in MODEL_NAMES:
        model = define_network(dict(config["network"])).to(device)
        state = trusted_torch_load(checkpoints[name]["path"], device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        models[name] = model
    return models


def write_csvs(output, protocol, results):
    paths = []
    for model_name in MODEL_NAMES:
        for preset_name in PROTOCOLS[protocol]:
            path = output / protocol / "per_image" / (
                f"{model_name}__{preset_name}.csv"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                fields = ["image", *METRICS,
                          "delta_psnr_vs_identity", "delta_ssim_vs_identity",
                          "delta_lpips_vs_identity"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row, baseline in zip(
                    results[(model_name, preset_name)],
                    results[("identity", preset_name)],
                ):
                    writer.writerow({
                        **row,
                        **{f"delta_{metric}_vs_identity":
                           row[metric] - baseline[metric] for metric in METRICS},
                    })
            paths.append(path)
        degraded = [name for name in PROTOCOLS[protocol] if name != "identity"]
        average_path = output / protocol / "per_image" / (
            f"{model_name}__four_degradation_average_delta.csv"
        )
        with average_path.open("w", newline="", encoding="utf-8") as stream:
            fields = ["image", "delta_psnr_vs_identity",
                      "delta_ssim_vs_identity", "delta_lpips_vs_identity"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, identity_row in enumerate(results[("identity", degraded[0])]):
                writer.writerow({
                    "image": identity_row["image"],
                    **{
                        f"delta_{metric}_vs_identity": statistics.mean([
                            results[(model_name, preset)][index][metric]
                            - results[("identity", preset)][index][metric]
                            for preset in degraded
                        ])
                        for metric in METRICS
                    },
                })
        paths.append(average_path)
    return paths


def make_report(protocol, config, checkpoints, results, count):
    lines = [
        f"# Retinexformer Frozen Robustness: {protocol.title()}", "",
        f"- Development Validation images: `{count}`",
        f"- Seed: `{config['seed']}`",
        f"- Protocol SHA-256: `{protocol_sha256()}`",
        "- Metrics: 8-bit RGB PSNR; RGB SSIM 11x11 sigma=1.5; AlexNet LPIPS",
        "- GT mean correction: `false`",
        "- official_test_participation: `NONE`", "",
        "## Absolute metrics", "",
        "| Model | Degradation | PSNR mean±sample SD | SSIM mean±sample SD | LPIPS mean±sample SD |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        for preset_name in PROTOCOLS[protocol]:
            rows = results[(model_name, preset_name)]
            formatted = []
            for metric in METRICS:
                avg, sd = mean_sd([row[metric] for row in rows])
                formatted.append(f"{avg:.6f}±{sd:.6f}")
            lines.append(
                f"| {model_name} | {preset_name} | " + " | ".join(formatted) + " |"
            )
    lines += ["", "## Paired difference versus Identity arm", "",
              "| Model | Degradation | ΔPSNR mean±sample SD | ΔSSIM mean±sample SD | ΔLPIPS mean±sample SD |",
              "|---|---|---:|---:|---:|"]
    for model_name in MODEL_NAMES:
        for preset_name in PROTOCOLS[protocol]:
            rows = results[(model_name, preset_name)]
            base = results[("identity", preset_name)]
            deltas = []
            for metric in METRICS:
                paired = [row[metric] - base[index][metric]
                          for index, row in enumerate(rows)]
                avg, sd = mean_sd(paired)
                deltas.append(f"{avg:+.6f}±{sd:.6f}")
            lines.append(
                f"| {model_name} | {preset_name} | "
                + " | ".join(deltas) + " |"
            )
    degraded = [name for name in PROTOCOLS[protocol] if name != "identity"]
    lines += ["", "## Four-degradation paired average versus Identity arm", "",
              "| Model | ΔPSNR mean±sample SD | ΔSSIM mean±sample SD | ΔLPIPS mean±sample SD |",
              "|---|---:|---:|---:|"]
    for model_name in MODEL_NAMES:
        values = {metric: [] for metric in METRICS}
        for index in range(count):
            for metric in METRICS:
                values[metric].append(statistics.mean([
                    results[(model_name, preset)][index][metric]
                    - results[("identity", preset)][index][metric]
                    for preset in degraded
                ]))
        formatted = []
        for metric in METRICS:
            avg, sd = mean_sd(values[metric])
            formatted.append(f"{avg:+.6f}±{sd:.6f}")
        lines.append(f"| {model_name} | " + " | ".join(formatted) + " |")
    lines += ["", "No final candidate is selected by this report.", "",
              "## Frozen checkpoints", ""]
    for name in MODEL_NAMES:
        lines.append(f"- `{name}`: `{checkpoints[name]['sha256']}`")
    return "\n".join(lines) + "\n"


def artifact_manifest(output, config, checkpoints, artifacts):
    records = {
        "config": {"path": config["_path"], "sha256": config["_sha256"]},
        "checkpoints": checkpoints,
        "source_sha256": {
            "physical_degradation.py": SOURCE_PHYSICAL_SHA256,
            "evaluate_physical_robustness.py": SOURCE_EVALUATOR_SHA256,
        },
        "protocol_sha256": protocol_sha256(),
        "artifacts": {str(path.relative_to(output)): sha256_file(path)
                      for path in artifacts},
        "official_test_participation": "NONE",
    }
    path = output / "artifact_sha256.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phyg/robustness_seed1234.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol", choices=("seen", "unseen", "both"),
                        default="both")
    parser.add_argument("--output", default="outputs/frozen_robustness_seed1234")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    output = reject_forbidden_path(args.output)
    config = load_robustness_config(args.config)
    root, filenames, _, _ = load_validation(config)
    checkpoints = checkpoint_audit(config, strict_models=True)
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        filenames = filenames[:args.max_images]
    dataset = DevelopmentPairDataset(root, filenames, crop_size=None)
    device = torch.device(args.device)
    models = load_models(config, checkpoints, device)
    metrics = AuthorValidationMetrics(device)
    protocols = ("seen", "unseen") if args.protocol == "both" else (args.protocol,)
    artifacts = []
    for protocol in protocols:
        results = defaultdict(list)
        with torch.inference_mode():
            for image_index in range(len(dataset)):
                low, gt, filename = dataset[image_index]
                low, gt = low.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
                for preset_name, parameters in PROTOCOLS[protocol].items():
                    generator = torch.Generator(device=device)
                    generator.manual_seed(preset_seed(
                        config["seed"], image_index, protocol, preset_name
                    ))
                    degraded = apply_frozen_degradation(
                        low, generator=generator, **parameters
                    )
                    for model_name, model in models.items():
                        output_image = model(degraded).clamp(0, 1)
                        values = metrics.evaluate_tensor_pair(output_image, gt)
                        results[(model_name, preset_name)].append(
                            {"image": filename, **values}
                        )
        artifacts += write_csvs(output, protocol, results)
        report = output / f"{protocol}_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(make_report(
            protocol, config, checkpoints, results, len(dataset)
        ), encoding="utf-8")
        artifacts.append(report)
    artifact_manifest(output, config, checkpoints, artifacts)
    print(json.dumps({
        "status": "PASS", "protocols": list(protocols),
        "images": len(dataset), "models": list(MODEL_NAMES),
        "output": str(output), "final_candidate_selected": False,
        "official_test_participation": "NONE",
    }, indent=2))


if __name__ == "__main__":
    main()
