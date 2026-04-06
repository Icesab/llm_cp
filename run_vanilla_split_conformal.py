from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVAL_TASK_ORDER = [
    "computer_security",
    "high_school_computer_science",
    "college_computer_science",
    "machine_learning",
    "formal_logic",
    "high_school_biology",
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
    "college_chemistry",
    "marketing",
    "public_relations",
    "management",
    "business_ethics",
    "professional_accounting",
]

MARKERS = ["o", "^", "p", "d", "*", "o", "^", "p", "d", "*", "s", "o", "^", "p", "d", "*"]
GROUP_COLOR = {
    "computer_security": "C0",
    "high_school_computer_science": "C0",
    "college_computer_science": "C0",
    "machine_learning": "C0",
    "formal_logic": "C0",
    "high_school_biology": "C2",
    "anatomy": "C2",
    "clinical_knowledge": "C2",
    "college_medicine": "C2",
    "professional_medicine": "C2",
    "college_chemistry": "C2",
    "marketing": "C1",
    "public_relations": "C1",
    "management": "C1",
    "business_ethics": "C1",
    "professional_accounting": "C1",
}
DEFAULT_INPUT_DIR = Path("outputs/mixtral_vanilla")
DEFAULT_OUTPUT_DIR = Path("outputs/mixtral_vanilla_eval")


def display_name(subject_id: str) -> str:
    return subject_id.replace("_", " ")


def parse_subjects(subjects_arg: str | None) -> List[str]:
    if not subjects_arg:
        return list(EVAL_TASK_ORDER)

    requested = [item.strip() for item in subjects_arg.split(",") if item.strip()]
    invalid = [item for item in requested if item not in EVAL_TASK_ORDER]
    if invalid:
        valid = ", ".join(EVAL_TASK_ORDER)
        raise ValueError(f"Invalid subjects: {', '.join(invalid)}. Valid subjects are: {valid}")
    return [subject for subject in EVAL_TASK_ORDER if subject in requested]


def safe_mean(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def safe_std(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.std(values))


def calibrate_lac(scores, targets, alpha: float = 0.1, return_dist: bool = False):
    scores = torch.tensor(scores, dtype=torch.float)
    targets = torch.tensor(targets)
    assert scores.size(0) == targets.size(0)
    assert targets.size(0)
    n = torch.tensor(targets.size(0))
    assert n

    score_dist = torch.take_along_dim(1 - scores, targets.unsqueeze(1), 1).flatten()
    quantile_level = torch.ceil((n + 1) * (1 - alpha)) / n
    assert 0 <= quantile_level <= 1, f"{alpha=} {n=} {quantile_level=}"
    qhat = torch.quantile(score_dist, quantile_level, interpolation="higher")
    return (qhat, score_dist) if return_dist else qhat


def inference_lac(scores, qhat, allow_empty_sets: bool = False):
    scores = torch.tensor(scores, dtype=torch.float)
    n = scores.size(0)

    elements_mask = scores >= (1 - qhat)
    if not allow_empty_sets:
        elements_mask[torch.arange(n), scores.argmax(1)] = True
    return elements_mask


def get_coverage(psets, targets, precision=None) -> float:
    psets = torch.as_tensor(psets).clone()
    targets = torch.as_tensor(targets).clone()
    n = psets.shape[0]
    if n == 0:
        return float("nan")
    coverage = psets[torch.arange(n), targets].float().mean().item()
    if precision is not None:
        coverage = round(coverage, precision)
    return coverage


def get_size(psets, precision=1) -> float:
    psets = torch.as_tensor(psets).clone()
    if psets.shape[0] == 0:
        return float("nan")
    size = psets.sum(1).float().mean().item()
    if precision is not None:
        size = round(size, precision)
    return size


def get_accuracy(scores, targets) -> float:
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    n = len(scores)
    if n == 0:
        return float("nan")
    correct = (scores.argmax(1) == targets).sum()
    return float(correct / n)


def load_datasets(input_dir: Path, subject_ids: List[str]):
    datasets = {}
    for subject_id in subject_ids:
        scores_path = input_dir / f"{subject_id}_scores.npy"
        targets_path = input_dir / f"{subject_id}_targets.npy"
        if not scores_path.exists() or not targets_path.exists():
            raise FileNotFoundError(f"Missing score files for {subject_id} in {input_dir}")

        scores = np.load(scores_path)
        targets = np.load(targets_path)
        if scores.ndim != 3:
            raise ValueError(f"{scores_path} must have shape [num_prompts, num_questions, 4], got {scores.shape}")
        if scores.shape[2] != 4:
            raise ValueError(f"{scores_path} last dimension must be 4, got {scores.shape}")
        if targets.ndim != 1:
            raise ValueError(f"{targets_path} must have shape [num_questions], got {targets.shape}")
        if scores.shape[1] != targets.shape[0]:
            raise ValueError(
                f"Question count mismatch for {subject_id}: scores={scores.shape}, targets={targets.shape}"
            )

        datasets[display_name(subject_id)] = {
            "subject_id": subject_id,
            "scores": scores,
            "targets": targets,
        }
    return datasets


def compute_group_boundaries(subject_ids: List[str]) -> List[int]:
    boundaries = []
    previous_color = None
    for index, subject_id in enumerate(subject_ids):
        current_color = GROUP_COLOR[subject_id]
        if previous_color is not None and current_color != previous_color:
            boundaries.append(index)
        previous_color = current_color
    return boundaries


def run_trials(datasets, alpha: float, num_trials: int, seed: int):
    all_scores = defaultdict(list)
    all_psets = defaultdict(list)
    all_targets = defaultdict(list)
    other_all_coverage = defaultdict(dict)
    other_all_size = defaultdict(dict)

    for trial_idx in range(num_trials):
        rng = np.random.default_rng(seed + trial_idx)
        for name, results in datasets.items():
            scores = results["scores"].mean(0)
            targets = results["targets"]

            index = rng.permutation(len(scores))
            n = len(scores) // 2
            cal_scores = scores[index][:n]
            val_scores = scores[index][n:]
            cal_targets = targets[index][:n]
            val_targets = targets[index][n:]

            qhat = calibrate_lac(cal_scores, cal_targets, alpha=alpha)
            psets = inference_lac(val_scores, qhat)

            all_psets[name].append(psets)
            all_scores[name].append(val_scores)
            all_targets[name].append(val_targets)

            for other_name, other_results in datasets.items():
                other_scores = other_results["scores"].mean(0)
                other_targets = other_results["targets"]
                if other_scores.shape[0] == 0:
                    continue

                other_psets = inference_lac(other_scores, qhat)
                # Intentional notebook compatibility: the first encounter only
                # initializes storage and does not append the current trial.
                if other_name not in other_all_coverage[name]:
                    other_all_coverage[name][other_name] = []
                    other_all_size[name][other_name] = []
                else:
                    other_all_coverage[name][other_name].append(get_coverage(other_psets, other_targets))
                    other_all_size[name][other_name].append(get_size(other_psets))

    return {
        "all_scores": all_scores,
        "all_psets": all_psets,
        "all_targets": all_targets,
        "other_all_coverage": other_all_coverage,
        "other_all_size": other_all_size,
    }


def summarize_cross_subject(datasets, other_all_coverage, other_all_size):
    names = list(datasets.keys())
    mean_coverage = np.full((len(names), len(names)), np.nan, dtype=float)
    std_coverage = np.full((len(names), len(names)), np.nan, dtype=float)
    mean_size = np.full((len(names), len(names)), np.nan, dtype=float)
    std_size = np.full((len(names), len(names)), np.nan, dtype=float)

    for i, calibrated_name in enumerate(names):
        results = other_all_coverage[calibrated_name]
        size_results = other_all_size[calibrated_name]
        for j, evaluated_name in enumerate(names):
            coverage_values = results.get(evaluated_name, [])
            size_values = size_results.get(evaluated_name, [])
            mean_coverage[i, j] = safe_mean(coverage_values)
            std_coverage[i, j] = safe_std(coverage_values)
            mean_size[i, j] = safe_mean(size_values)
            std_size[i, j] = safe_std(size_values)

    return {
        "names": names,
        "mean_coverage": mean_coverage,
        "std_coverage": std_coverage,
        "mean_size": mean_size,
        "std_size": std_size,
    }


def compute_stratified_metrics(datasets, all_psets, all_scores, all_targets):
    acc_results = {}
    cov_results = {}
    naive_results = {}

    for dataset_name in datasets.keys():
        stratified_coverage = defaultdict(list)
        stratified_accuracy = defaultdict(list)
        naive_coverage = defaultdict(list)

        for k in range(1, 5):
            for trial_idx, psets in enumerate(all_psets[dataset_name]):
                mask = psets.sum(1) == k
                masked_psets = psets[mask]
                masked_scores = all_scores[dataset_name][trial_idx][mask]
                masked_targets = all_targets[dataset_name][trial_idx][mask]
                stratified_coverage[k].append(get_coverage(masked_psets, masked_targets))
                stratified_accuracy[k].append(get_accuracy(masked_scores, masked_targets))

            for trial_idx, scores in enumerate(all_scores[dataset_name]):
                sorted_scores = np.flip(scores.argsort(), 1)[:, :k]
                psets = np.eye(4, dtype=int)[sorted_scores].sum(1)
                naive_coverage[k].append(get_coverage(psets.copy(), all_targets[dataset_name][trial_idx]))

        acc_results[dataset_name] = dict(stratified_accuracy)
        cov_results[dataset_name] = dict(stratified_coverage)
        naive_results[dataset_name] = dict(naive_coverage)

    return {
        "acc_results": acc_results,
        "cov_results": cov_results,
        "naive_results": naive_results,
    }


def write_metric_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "subject_name", "mean", "std", "num_trials"])
        writer.writeheader()
        writer.writerows(rows)


def plot_coverage_bars(
    path: Path,
    subject_ids: List[str],
    subject_names: List[str],
    values: List[float],
    alpha: float,
) -> None:
    fontsize = 32
    plt.figure(figsize=(8, 12))
    bars = plt.barh(subject_names, values)
    for bar, subject_id in zip(bars, subject_ids):
        bar.set_color(GROUP_COLOR[subject_id])
    plt.axvline(1 - alpha, ls="--", lw=6, c="r")
    plt.xlim(0.5, 1)
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.gca().set_xticklabels([f"{x:.0%}" for x in plt.gca().get_xticks()])
    plt.xlabel("Coverage", fontsize=fontsize)
    plt.grid(ls="-", axis="x")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_size_bars(path: Path, subject_ids: List[str], subject_names: List[str], values: List[float]) -> None:
    fontsize = 32
    plt.figure(figsize=(8, 12))
    bars = plt.barh(subject_names, values)
    for bar, subject_id in zip(bars, subject_ids):
        bar.set_color(GROUP_COLOR[subject_id])
    plt.xlim(1, 4)
    plt.xticks([1, 2, 3, 4], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Conformal prediction set size", fontsize=fontsize)
    plt.grid(ls="-", axis="x")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_heatmap(path: Path, names: List[str], subject_ids: List[str], values: np.ndarray, alpha: float) -> None:
    fontsize = 24
    plt.figure(figsize=(12, 12))
    plt.grid(False)
    plt.imshow(values - (1 - alpha), cmap=plt.get_cmap("RdYlGn"))
    plt.xticks(np.arange(len(names)), names, fontsize=fontsize - 4, rotation=50, ha="right", va="top")
    plt.yticks(np.arange(len(names)), names, fontsize=fontsize - 4)
    plt.xlabel("Evaluated on", fontsize=fontsize + 8)
    plt.ylabel("Calibrated on", fontsize=fontsize + 8)

    for boundary in compute_group_boundaries(subject_ids):
        plt.axhline(boundary - 0.5, ls="-", lw=4, c="k")
        plt.axvline(boundary - 0.5, ls="-", lw=4, c="k")

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_yticklabels([f"{x:.1%}" for x in cbar.ax.get_yticks()])
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_line_metric(path: Path, names: List[str], subject_ids: List[str], metric_results, ylabel: str) -> None:
    fontsize = 24
    plt.figure(figsize=(14, 8))

    for index, name in enumerate(names):
        subject_id = subject_ids[index]
        marker = MARKERS[EVAL_TASK_ORDER.index(subject_id)]
        color = GROUP_COLOR[subject_id]
        values = [safe_mean(metric_results[name][k]) for k in sorted(metric_results[name])]
        plt.plot(
            sorted(metric_results[name]),
            values,
            label=name,
            ls="--",
            marker=marker,
            markersize=12 if ylabel == "Coverage" else 10,
            c=color,
            markeredgecolor="k",
        )

    plt.legend(fontsize=fontsize - 4, bbox_to_anchor=(1.1, 1.50), ncol=3)
    plt.xticks([1, 2, 3, 4], fontsize=fontsize)
    if ylabel == "Coverage":
        plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=fontsize)
    else:
        plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=fontsize)
    plt.gca().set_yticklabels([f"{x:.0%}" for x in plt.gca().get_yticks()])
    plt.xlabel(
        "Naive prediction set size" if "naive" in path.name else "Conformal prediction set size",
        fontsize=fontsize + 8,
    )
    plt.ylabel(ylabel, fontsize=fontsize + 8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the vanilla split conformal evaluation from ConformalLLM as a headless script."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--subjects", default=None, help="Comma-separated subset of underscore task ids.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_trials <= 0:
        parser.error("--num-trials must be a positive integer.")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1.")

    try:
        subject_ids = parse_subjects(args.subjects)
    except ValueError as exc:
        parser.error(str(exc))
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(input_dir, subject_ids)
    trial_outputs = run_trials(datasets, alpha=args.alpha, num_trials=args.num_trials, seed=args.seed)
    cross_subject = summarize_cross_subject(
        datasets,
        trial_outputs["other_all_coverage"],
        trial_outputs["other_all_size"],
    )
    derived_metrics = compute_stratified_metrics(
        datasets,
        trial_outputs["all_psets"],
        trial_outputs["all_scores"],
        trial_outputs["all_targets"],
    )

    names = cross_subject["names"]
    ordered_subject_ids = [datasets[name]["subject_id"] for name in names]

    coverage_rows = []
    size_rows = []
    per_subject_summary = {}
    for index, name in enumerate(names):
        subject_id = datasets[name]["subject_id"]
        coverage_values = trial_outputs["other_all_coverage"][name].get(name, [])
        size_values = trial_outputs["other_all_size"][name].get(name, [])
        coverage_mean = safe_mean(coverage_values)
        coverage_std = safe_std(coverage_values)
        size_mean = safe_mean(size_values)
        size_std = safe_std(size_values)

        coverage_rows.append(
            {
                "subject_id": subject_id,
                "subject_name": name,
                "mean": coverage_mean,
                "std": coverage_std,
                "num_trials": len(coverage_values),
            }
        )
        size_rows.append(
            {
                "subject_id": subject_id,
                "subject_name": name,
                "mean": size_mean,
                "std": size_std,
                "num_trials": len(size_values),
            }
        )
        per_subject_summary[subject_id] = {
            "subject_name": name,
            "coverage_mean": coverage_mean,
            "coverage_std": coverage_std,
            "set_size_mean": size_mean,
            "set_size_std": size_std,
            "record_count": len(coverage_values),
        }

    write_metric_csv(output_dir / "coverage_by_subject.csv", coverage_rows)
    write_metric_csv(output_dir / "set_size_by_subject.csv", size_rows)

    plot_coverage_bars(
        output_dir / "coverage.png",
        ordered_subject_ids,
        names,
        [cross_subject["mean_coverage"][i, i] for i in range(len(names))],
        args.alpha,
    )
    plot_size_bars(
        output_dir / "set_size.png",
        ordered_subject_ids,
        names,
        [cross_subject["mean_size"][i, i] for i in range(len(names))],
    )
    plot_heatmap(
        output_dir / "calibration_heatmap.png",
        names,
        ordered_subject_ids,
        cross_subject["mean_coverage"],
        args.alpha,
    )
    plot_line_metric(
        output_dir / "selective_accuracy.png",
        names,
        ordered_subject_ids,
        derived_metrics["acc_results"],
        ylabel="Top-1 accuracy",
    )
    plot_line_metric(
        output_dir / "stratified_coverage.png",
        names,
        ordered_subject_ids,
        derived_metrics["cov_results"],
        ylabel="Coverage",
    )
    plot_line_metric(
        output_dir / "naive_topk_coverage.png",
        names,
        ordered_subject_ids,
        derived_metrics["naive_results"],
        ylabel="Coverage",
    )

    summary_path = output_dir / f"summary_alpha_{args.alpha}.json"
    summary = {
        "alpha": args.alpha,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "subjects": [
            {"subject_id": datasets[name]["subject_id"], "subject_name": name}
            for name in names
        ],
        "per_subject": per_subject_summary,
        "mean_coverage_matrix": cross_subject["mean_coverage"].tolist(),
        "mean_size_matrix": cross_subject["mean_size"].tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
