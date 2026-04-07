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
DEFAULT_INPUT_DIR = Path("outputs/mixtral_scores_with_gates")
DEFAULT_OUTPUT_DIR = Path("outputs/mixtral_weighted_eval")


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


def kl_divergence_np(p, q, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def randomize_gate(rng: np.random.Generator, gate: np.ndarray, tau: int) -> np.ndarray:
    gate = np.asarray(gate, dtype=np.float64)
    gate = np.clip(gate, 1e-12, None)
    gate = gate / gate.sum()
    sampled = rng.multinomial(tau, gate)
    return sampled.astype(np.float64) / float(tau)


def divergence_weight(divergence_value: float, tau: int) -> float:
    return float(np.exp(np.clip(-tau * divergence_value, a_min=-700.0, a_max=0.0)))


def weighted_qhat_lac(
    cal_scores: np.ndarray,
    cal_targets: np.ndarray,
    cal_gates: np.ndarray,
    test_gate: np.ndarray,
    alpha: float,
    tau: int,
    rng: np.random.Generator,
):
    cal_lac = 1.0 - cal_scores[np.arange(len(cal_scores)), cal_targets]
    randomized_gate = randomize_gate(rng, test_gate, tau=tau)

    cal_weights = np.array(
        [divergence_weight(kl_divergence_np(randomized_gate, gate), tau=tau) for gate in cal_gates],
        dtype=np.float64,
    )
    test_weight = divergence_weight(kl_divergence_np(randomized_gate, test_gate), tau=tau)

    total_weight = float(cal_weights.sum() + test_weight)
    if total_weight <= 0:
        raise RuntimeError("Weighted conformal encountered non-positive total weight.")

    normalized_cal_weights = cal_weights / total_weight
    normalized_test_weight = test_weight / total_weight

    cal_weight_sum = float(cal_weights.sum())
    if cal_weight_sum > 0:
        normalized_cal_only = cal_weights / cal_weight_sum
        effective_calibration_size = float(1.0 / np.sum(normalized_cal_only**2))
    else:
        effective_calibration_size = 0.0

    sort_index = np.argsort(cal_lac)
    sorted_scores = cal_lac[sort_index]
    sorted_weights = normalized_cal_weights[sort_index]
    finite_cdf = np.cumsum(sorted_weights)
    threshold = 1.0 - alpha

    if len(finite_cdf) == 0 or finite_cdf[-1] < threshold:
        qhat = float("inf")
        inf_qhat = True
    else:
        qhat = float(sorted_scores[np.searchsorted(finite_cdf, threshold, side="left")])
        inf_qhat = False

    return qhat, {
        "inf_qhat": inf_qhat,
        "effective_calibration_size": effective_calibration_size,
        "test_weight": normalized_test_weight,
    }


def inference_lac_weighted(score_row: np.ndarray, qhat: float, allow_empty_sets: bool = False) -> np.ndarray:
    score_row = np.asarray(score_row, dtype=np.float32)
    prediction_set = score_row >= (1.0 - qhat)
    if not allow_empty_sets:
        prediction_set[np.argmax(score_row)] = True
    return prediction_set


def load_datasets(input_dir: Path, subject_ids: List[str]):
    datasets = {}
    for subject_id in subject_ids:
        scores_path = input_dir / f"{subject_id}_scores.npy"
        targets_path = input_dir / f"{subject_id}_targets.npy"
        gates_path = input_dir / f"{subject_id}_gates.npy"

        if not scores_path.exists() or not targets_path.exists() or not gates_path.exists():
            raise FileNotFoundError(f"Missing scores/targets/gates files for {subject_id} in {input_dir}")

        scores = np.load(scores_path)
        targets = np.load(targets_path)
        gates = np.load(gates_path)

        if scores.ndim != 3 or scores.shape[2] != 4:
            raise ValueError(f"{scores_path} must have shape [num_prompts, num_questions, 4], got {scores.shape}")
        if targets.ndim != 1:
            raise ValueError(f"{targets_path} must have shape [num_questions], got {targets.shape}")
        if gates.ndim != 3 or gates.shape[2] != 8:
            raise ValueError(f"{gates_path} must have shape [num_prompts, num_questions, 8], got {gates.shape}")
        if scores.shape[0] != gates.shape[0] or scores.shape[1] != gates.shape[1]:
            raise ValueError(
                f"Prompt/question shape mismatch for {subject_id}: scores={scores.shape}, gates={gates.shape}"
            )
        if scores.shape[1] != targets.shape[0]:
            raise ValueError(
                f"Question count mismatch for {subject_id}: scores={scores.shape}, targets={targets.shape}"
            )
        if not np.isfinite(scores).all():
            raise ValueError(f"{scores_path} contains non-finite values.")
        if not np.isfinite(gates).all():
            raise ValueError(f"{gates_path} contains non-finite values.")

        scores_mean = scores.mean(axis=0).astype(np.float32)
        gates_mean = gates.mean(axis=0).astype(np.float32)
        gates_mean = gates_mean / np.clip(gates_mean.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)

        datasets[display_name(subject_id)] = {
            "subject_id": subject_id,
            "scores": scores_mean,
            "targets": targets.astype(np.int64),
            "gates": gates_mean.astype(np.float32),
        }
    return datasets


def run_trials(datasets, alpha: float, num_trials: int, tau: int, seed: int):
    all_scores = defaultdict(list)
    all_psets = defaultdict(list)
    all_targets = defaultdict(list)
    coverage_by_subject = defaultdict(list)
    size_by_subject = defaultdict(list)
    inf_qhat_rate_by_subject = defaultdict(list)
    effective_calibration_size_by_subject = defaultdict(list)

    for trial_idx in range(num_trials):
        rng = np.random.default_rng(seed + trial_idx)
        for name, results in datasets.items():
            scores = results["scores"]
            targets = results["targets"]
            gates = results["gates"]

            index = rng.permutation(len(scores))
            n = len(scores) // 2
            cal_scores = scores[index][:n]
            val_scores = scores[index][n:]
            cal_targets = targets[index][:n]
            val_targets = targets[index][n:]
            cal_gates = gates[index][:n]
            val_gates = gates[index][n:]

            psets = []
            inf_qhat_count = 0
            effective_sizes = []
            for score_row, gate_row in zip(val_scores, val_gates):
                qhat, diagnostics = weighted_qhat_lac(
                    cal_scores=cal_scores,
                    cal_targets=cal_targets,
                    cal_gates=cal_gates,
                    test_gate=gate_row,
                    alpha=alpha,
                    tau=tau,
                    rng=rng,
                )
                psets.append(inference_lac_weighted(score_row, qhat, allow_empty_sets=False))
                inf_qhat_count += int(diagnostics["inf_qhat"])
                effective_sizes.append(diagnostics["effective_calibration_size"])

            psets = np.asarray(psets, dtype=bool)

            all_psets[name].append(psets)
            all_scores[name].append(val_scores)
            all_targets[name].append(val_targets)
            coverage_by_subject[name].append(get_coverage(psets, val_targets))
            size_by_subject[name].append(get_size(psets))
            inf_qhat_rate_by_subject[name].append(float(inf_qhat_count / max(len(val_scores), 1)))
            effective_calibration_size_by_subject[name].append(safe_mean(effective_sizes))

    return {
        "all_scores": all_scores,
        "all_psets": all_psets,
        "all_targets": all_targets,
        "coverage_by_subject": coverage_by_subject,
        "size_by_subject": size_by_subject,
        "inf_qhat_rate_by_subject": inf_qhat_rate_by_subject,
        "effective_calibration_size_by_subject": effective_calibration_size_by_subject,
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
        description="Run same-subject KL-weighted conformal evaluation using prompt-averaged scores and MoE gates."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--tau", type=int, default=20)
    parser.add_argument("--subjects", default=None, help="Comma-separated subset of underscore task ids.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_trials <= 0:
        parser.error("--num-trials must be a positive integer.")
    if args.tau <= 0:
        parser.error("--tau must be a positive integer.")
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
    trial_outputs = run_trials(
        datasets=datasets,
        alpha=args.alpha,
        num_trials=args.num_trials,
        tau=args.tau,
        seed=args.seed,
    )
    derived_metrics = compute_stratified_metrics(
        datasets,
        trial_outputs["all_psets"],
        trial_outputs["all_scores"],
        trial_outputs["all_targets"],
    )

    names = list(datasets.keys())
    ordered_subject_ids = [datasets[name]["subject_id"] for name in names]
    coverage_rows = []
    size_rows = []
    per_subject_summary = {}
    overall_inf_qhat_values = []
    overall_effective_size_values = []

    for name in names:
        subject_id = datasets[name]["subject_id"]
        coverage_values = trial_outputs["coverage_by_subject"][name]
        size_values = trial_outputs["size_by_subject"][name]
        inf_qhat_values = trial_outputs["inf_qhat_rate_by_subject"][name]
        effective_size_values = trial_outputs["effective_calibration_size_by_subject"][name]

        coverage_rows.append(
            {
                "subject_id": subject_id,
                "subject_name": name,
                "mean": safe_mean(coverage_values),
                "std": safe_std(coverage_values),
                "num_trials": len(coverage_values),
            }
        )
        size_rows.append(
            {
                "subject_id": subject_id,
                "subject_name": name,
                "mean": safe_mean(size_values),
                "std": safe_std(size_values),
                "num_trials": len(size_values),
            }
        )
        per_subject_summary[subject_id] = {
            "subject_name": name,
            "coverage_mean": safe_mean(coverage_values),
            "coverage_std": safe_std(coverage_values),
            "set_size_mean": safe_mean(size_values),
            "set_size_std": safe_std(size_values),
            "inf_qhat_rate": safe_mean(inf_qhat_values),
            "mean_effective_calibration_size": safe_mean(effective_size_values),
            "record_count": len(coverage_values),
        }
        overall_inf_qhat_values.extend(inf_qhat_values)
        overall_effective_size_values.extend(effective_size_values)

    write_metric_csv(output_dir / "coverage_by_subject.csv", coverage_rows)
    write_metric_csv(output_dir / "set_size_by_subject.csv", size_rows)

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

    summary_path = output_dir / f"summary_alpha_{args.alpha}_tau_{args.tau}.json"
    summary = {
        "alpha": args.alpha,
        "tau": args.tau,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "subjects": [
            {"subject_id": datasets[name]["subject_id"], "subject_name": name}
            for name in names
        ],
        "per_subject": per_subject_summary,
        "inf_qhat_rate": safe_mean(overall_inf_qhat_values),
        "mean_effective_calibration_size": safe_mean(overall_effective_size_values),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
