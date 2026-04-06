from __future__ import annotations

import argparse
import inspect
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import prompt_questions as p
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


TASK_LIST = [
    "college_computer_science",
    "formal_logic",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "clinical_knowledge",
    "high_school_biology",
    "anatomy",
    "college_chemistry",
    "college_medicine",
    "professional_medicine",
    "business_ethics",
    "professional_accounting",
    "public_relations",
    "management",
    "marketing",
]

PROMPT_LIST = [
    p.prompt_q_list_college_cs,
    p.prompt_q_list_formal_logic,
    p.prompt_q_list_high_school_cs,
    p.prompt_q_list_computer_security,
    p.prompt_q_list_machine_learning,
    p.prompt_q_list_clinical_knowledge,
    p.prompt_q_list_high_school_bio,
    p.prompt_q_list_anatomy,
    p.promtp_q_list_college_chemistry,
    p.prompt_q_list_college_medicine,
    p.prompt_q_list_professional_medicine,
    p.prompt_q_list_business_ethics,
    p.prompt_q_list_professional_accounting,
    p.prompt_q_list_pr,
    p.prompt_q_list_management,
    p.prompt_q_list_marketing,
]

PROMPTS_BY_SUBJECT = dict(zip(TASK_LIST, PROMPT_LIST))
ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
OPTION_ORDER = ("A", "B", "C", "D")
DEFAULT_MODEL_ID = "mistralai/Mixtral-8x7B-v0.1"
DEFAULT_OUTPUT_DIR = Path("outputs/mixtral_vanilla")
DEFAULT_TOKEN_LIMIT = 1500
DEFAULT_GATE_POOLING = "last2_mean_over_valid_prefix_tokens"


def modify_task_data(task_data, token_limit: int, max_size_prompt_len: int):
    """
    Keep the repo's original character-length filtering behavior unchanged.
    """
    new_task_data = {
        "train": defaultdict(list),
        "validation": defaultdict(list),
        "test": defaultdict(list),
    }
    for split in new_task_data.keys():
        for i in range(len(task_data[split])):
            q = task_data[split]["input"][i]
            a = task_data[split]["A"][i]
            b = task_data[split]["B"][i]
            c = task_data[split]["C"][i]
            d = task_data[split]["D"][i]
            target = task_data[split]["target"][i]
            if len(q) + max(map(len, [a, b, c, d])) + max_size_prompt_len < token_limit:
                new_task_data[split]["input"].append(q)
                new_task_data[split]["A"].append(a)
                new_task_data[split]["B"].append(b)
                new_task_data[split]["C"].append(c)
                new_task_data[split]["D"].append(d)
                new_task_data[split]["target"].append(target)
    return new_task_data


def get_prompt(task_data, task: str, question_num: int = 0, prompt_q: str | None = None) -> str:
    if prompt_q is None:
        prompt_set = "test"
        if question_num > len(task_data["test"]["input"]) - 1:
            print("prompt question id exceeds the length of test set")
            print("selecting last question of the test set")
            question_num = len(task_data["test"]["input"]) - 1
        prompt_add = f"This is a question from {task.replace('_', ' ')}.\n"
        prompt_add += f"{task_data[prompt_set]['input'][question_num]}\n"
        for letter in OPTION_ORDER:
            prompt_add += "    " + letter + ". " + task_data[prompt_set][letter][question_num] + "\n"
        prompt_add += f"The correct answer is option: {task_data[prompt_set]['target'][question_num]}\n"
    else:
        prompt_add = f"This is a question from {task.replace('_', ' ')}."
        prompt_add += prompt_q
        prompt_add += "\n"
    prompt_add += f"You are the world's best expert in {task.replace('_', ' ')}. "
    prompt_add += "Reason step-by-step and answer the following question. "
    return prompt_add


def build_question_records(
    task_data,
    prompt_add: str,
    prompt_q_id: int | None = None,
    max_questions: int | None = None,
) -> List[Dict[str, object]]:
    """
    Return prompt prefix and option-specific strings separately so future MoE
    gating code can reuse the label-independent prefix text.
    """
    question_records: List[Dict[str, object]] = []
    splits = ["train", "validation", "test"]
    if prompt_q_id is not None:
        print(f"Excluding test set question no {prompt_q_id} from dataset")

    for split in splits:
        start = 1 if split == "train" else 0
        for i in range(start, len(task_data[split]["input"])):
            if split == "test" and prompt_q_id is not None and i == prompt_q_id:
                continue

            prefix_text = prompt_add + task_data[split]["input"][i] + "\n"
            for letter in OPTION_ORDER:
                prefix_text += f"({letter}) {task_data[split][letter][i]} "
            prefix_text += "\nThe correct answer is option: "

            question_records.append(
                {
                    "prefix_text": prefix_text,
                    "option_texts": {letter: prefix_text + letter for letter in OPTION_ORDER},
                    "target": task_data[split]["target"][i],
                }
            )

            if max_questions is not None and len(question_records) >= max_questions:
                return question_records

    return question_records


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp_logits
    return probabilities


def extract_answer(batch):
    probabilities = softmax(np.array([answer[-1] for answer in batch]))
    output_with_probabilities = [(batch[i][0], probabilities[i]) for i in range(len(batch))]
    return output_with_probabilities


def accuracy(predicted_probs, correct_answers: Sequence[str]) -> float:
    total_count = len(correct_answers)
    assert len(correct_answers) == len(predicted_probs)
    correct_count = 0

    for i in range(total_count):
        max_prob_answer = max(predicted_probs[i], key=lambda x: x[1])[0].strip()
        if correct_answers[i] == max_prob_answer:
            correct_count += 1.0

    return correct_count / total_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_data(subject_name: str):
    return load_dataset("lukaemon/mmlu", subject_name, trust_remote_code=True)


def load_model_and_tokenizer(model_id: str, max_memory_per_gpu_gib: int):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for mixtral_gpt_scores.py. Activate the server GPU environment "
            "and rerun after confirming torch.cuda.is_available() is True."
        )
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA devices detected. This script requires at least one visible GPU.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_memory = {i: f"{max_memory_per_gpu_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    return next(model.parameters()).device


def validate_option_suffix_single_token(tokenizer, prefix_text: str) -> None:
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    for option in OPTION_ORDER:
        option_ids = tokenizer(prefix_text + option, add_special_tokens=False).input_ids
        if len(option_ids) != len(prefix_ids) + 1:
            raise RuntimeError(
                "Expected each option suffix to add exactly one token. "
                f"Prefix {prefix_text!r} with option {option!r} produced {len(option_ids) - len(prefix_ids)} tokens."
            )


def score_options(model, tokenizer, option_texts: Dict[str, str], return_router: bool = False):
    device = get_input_device(model)
    batch = []

    with torch.no_grad():
        for option in OPTION_ORDER:
            text = option_texts[option]
            encoded = tokenizer(text, return_tensors="pt")
            if encoded["input_ids"].shape[1] < 2:
                raise ValueError(f"Expected at least two tokens for option {option}, got: {text!r}")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, use_cache=False)
            logits = outputs.logits.detach().float().cpu()[:, -2:-1, :]
            option_ids = encoded["input_ids"].detach().cpu()[:, -1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            option_log_prob = torch.gather(log_probs, 2, option_ids[:, :, None]).squeeze(-1)
            batch.append((option, option_log_prob.item()))

            del outputs, encoded, logits, option_ids, log_probs, option_log_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    probs = extract_answer(batch)
    extra = {}
    if return_router:
        extra["router_logits"] = None
        extra["pooled_gate"] = None
    return probs, extra


def model_supports_kwarg(model, kwarg_name: str) -> bool:
    return kwarg_name in inspect.signature(model.forward).parameters


def build_router_forward_kwargs(model) -> Dict[str, object]:
    if not model_supports_kwarg(model, "output_router_logits"):
        raise RuntimeError(
            "The current transformers/Mixtral build does not expose output_router_logits on model.forward()."
        )

    kwargs: Dict[str, object] = {
        "output_router_logits": True,
        "use_cache": False,
        "return_dict": True,
    }
    if model_supports_kwarg(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1
    return kwargs


def ensure_prob_simplex(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    sums = x.sum(dim=-1, keepdim=True)
    looks_like_prob = bool(torch.all(x >= -1e-6)) and bool(
        torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)
    )
    if not looks_like_prob:
        x = torch.softmax(x, dim=-1)
    x = torch.clamp(x, min=0.0)
    x = x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return x


def get_valid_prefix_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, bos_token_id: int | None) -> torch.Tensor:
    token_ids = input_ids[0].detach().cpu()
    valid_mask = attention_mask[0].detach().cpu().bool()
    if bos_token_id is None:
        return valid_mask

    mask_without_bos = valid_mask & token_ids.ne(bos_token_id)
    if mask_without_bos.any():
        return mask_without_bos
    return valid_mask


def reshape_router_layer(router_layer: torch.Tensor, seq_len: int) -> torch.Tensor:
    layer = router_layer.detach().float().cpu()
    if layer.ndim == 3:
        if layer.shape[0] != 1:
            raise RuntimeError(f"Expected batch size 1 for router layer, got shape {tuple(layer.shape)}")
        layer = layer[0]
    elif layer.ndim != 2:
        raise RuntimeError(f"Unsupported router layer shape: {tuple(layer.shape)}")

    if layer.shape[0] != seq_len:
        raise RuntimeError(
            "Router token dimension does not match the prefix token length. "
            f"router_layer.shape={tuple(layer.shape)}, seq_len={seq_len}"
        )
    return layer


def router_logits_match_sequence(router_logits, seq_len: int) -> bool:
    if router_logits is None or len(router_logits) < 2:
        return False
    for router_layer in router_logits[-2:]:
        layer = router_layer.detach()
        if layer.ndim == 3:
            if layer.shape[0] != 1 or layer.shape[1] != seq_len:
                return False
        elif layer.ndim == 2:
            if layer.shape[0] != seq_len:
                return False
        else:
            return False
    return True


def extract_pooled_gate(model, tokenizer, prefix_text: str) -> np.ndarray:
    device = get_input_device(model)
    encoded = tokenizer(prefix_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    forward_kwargs = build_router_forward_kwargs(model)

    with torch.no_grad():
        outputs = model(**encoded, **forward_kwargs)

    router_logits = getattr(outputs, "router_logits", None)
    seq_len = int(encoded["input_ids"].shape[1])
    if "logits_to_keep" in forward_kwargs and not router_logits_match_sequence(router_logits, seq_len=seq_len):
        del outputs
        fallback_kwargs = dict(forward_kwargs)
        fallback_kwargs.pop("logits_to_keep", None)
        with torch.no_grad():
            outputs = model(**encoded, **fallback_kwargs)
        router_logits = getattr(outputs, "router_logits", None)

    if router_logits is None:
        raise RuntimeError("Model output does not contain router_logits even though output_router_logits=True.")
    if len(router_logits) < 2:
        raise RuntimeError(f"Expected at least 2 router layers, got {len(router_logits)}")

    valid_mask = get_valid_prefix_mask(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        bos_token_id=tokenizer.bos_token_id,
    )

    pooled_layers = []
    for router_layer in router_logits[-2:]:
        layer = reshape_router_layer(router_layer, seq_len=seq_len)
        layer = ensure_prob_simplex(layer)
        layer = layer[valid_mask]
        if layer.shape[0] == 0:
            raise RuntimeError("No valid prefix tokens remained after attention-mask/BOS filtering.")
        pooled_layers.append(layer)

    router_probs = torch.stack(pooled_layers, dim=0)
    pooled = router_probs.mean(dim=(0, 1))
    pooled = pooled / pooled.sum().clamp_min(1e-12)

    del outputs, encoded, router_logits, pooled_layers, router_probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pooled.detach().cpu().numpy().astype(np.float32)


def parse_subjects(subjects_arg: str | None) -> List[str]:
    if not subjects_arg:
        return list(TASK_LIST)

    requested = [item.strip() for item in subjects_arg.split(",") if item.strip()]
    invalid = [item for item in requested if item not in TASK_LIST]
    if invalid:
        valid = ", ".join(TASK_LIST)
        raise ValueError(f"Invalid subjects: {', '.join(invalid)}. Valid subjects are: {valid}")
    return [subject for subject in TASK_LIST if subject in requested]


def initialize_manifest(
    manifest_path: Path,
    model_id: str,
    num_prompts: int,
    token_limit: int,
    seed: int,
    max_questions: int | None,
    save_gates: bool,
    gate_pooling: str | None,
    max_memory_per_gpu_gib: int,
):
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    manifest["model_id"] = model_id
    manifest["task_list"] = list(TASK_LIST)
    manifest["num_prompts"] = num_prompts
    manifest["token_limit"] = token_limit
    manifest["seed"] = seed
    manifest["max_questions"] = max_questions
    manifest["save_gates"] = save_gates
    manifest["gate_pooling"] = gate_pooling
    manifest["max_memory_per_gpu_gib"] = max_memory_per_gpu_gib
    manifest.setdefault("subjects", {})
    return manifest


def write_manifest(manifest_path: Path, manifest) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def accuracy_pickle_path(output_dir: Path, num_prompts: int) -> Path:
    if num_prompts == 10:
        return output_dir / "accuracy_gpt_prompts_10.pkl"
    return output_dir / f"accuracy_gpt_prompts_{num_prompts}.pkl"


def skip_context_matches(manifest: dict, num_prompts: int, max_questions: int | None, save_gates: bool) -> bool:
    if not manifest:
        return False
    return (
        manifest.get("num_prompts") == num_prompts
        and manifest.get("max_questions") == max_questions
        and manifest.get("save_gates") == save_gates
    )


def should_skip_subject(
    output_dir: Path,
    subject_name: str,
    num_prompts: int,
    save_gates: bool,
    manifest: dict,
    max_questions: int | None,
) -> bool:
    if not skip_context_matches(manifest, num_prompts=num_prompts, max_questions=max_questions, save_gates=save_gates):
        return False

    scores_path = output_dir / f"{subject_name}_scores.npy"
    targets_path = output_dir / f"{subject_name}_targets.npy"
    gates_path = output_dir / f"{subject_name}_gates.npy"

    if not scores_path.exists() or not targets_path.exists():
        return False
    if save_gates and not gates_path.exists():
        return False

    try:
        scores = np.load(scores_path)
        targets = np.load(targets_path)
        gates = np.load(gates_path) if save_gates else None
    except Exception:
        return False

    if scores.ndim != 3 or targets.ndim != 1:
        return False
    if scores.shape[0] != num_prompts or scores.shape[1] != targets.shape[0]:
        return False
    if save_gates:
        if gates.ndim != 3:
            return False
        if gates.shape[0] != num_prompts or gates.shape[1] != targets.shape[0] or gates.shape[2] != 8:
            return False

    return True


def run_subject(
    subject_name: str,
    prompts: Sequence[str],
    model,
    tokenizer,
    output_dir: Path,
    token_limit: int,
    max_questions: int | None,
    accuracy_dict,
    save_gates: bool,
):
    task_data = load_subject_data(subject_name)
    max_size_prompt = int(np.max(np.array([len(text) for text in prompts])))
    task_data_modified = modify_task_data(
        task_data,
        token_limit=token_limit,
        max_size_prompt_len=max_size_prompt,
    )

    prediction_lists = []
    gate_lists = []
    acc_list = []
    solution_answers = None
    question_count = None

    print(f"Generating GPT-prompt predictions for {subject_name}")
    for prompt_index, prompt_q in enumerate(prompts):
        prompt_add = get_prompt(task_data, task=subject_name, prompt_q=prompt_q)
        if prompt_index % 5 == 0:
            print(prompt_add)

        question_records = build_question_records(
            task_data_modified,
            prompt_add=prompt_add,
            max_questions=max_questions,
        )
        if not question_records:
            raise RuntimeError(f"No questions available for subject {subject_name} after filtering.")

        prompt_predictions = []
        prompt_gates = []
        prompt_targets = []
        for question_idx, record in enumerate(question_records, start=1):
            validate_option_suffix_single_token(tokenizer, record["prefix_text"])
            probs, _ = score_options(model, tokenizer, record["option_texts"], return_router=False)
            prompt_predictions.append(probs)
            if save_gates:
                prompt_gates.append(extract_pooled_gate(model, tokenizer, record["prefix_text"]))
            prompt_targets.append(record["target"])

            if question_idx % 25 == 0 or question_idx == len(question_records):
                print(
                    f"[{subject_name}] prompt {prompt_index + 1}/{len(prompts)} "
                    f"question {question_idx}/{len(question_records)}"
                )

        if solution_answers is None:
            solution_answers = prompt_targets
            question_count = len(prompt_targets)
        elif prompt_targets != solution_answers:
            raise RuntimeError(f"Target mismatch across prompts for subject {subject_name}.")

        acc = round(accuracy(prompt_predictions, prompt_targets), 3)
        acc_list.append(acc)
        prediction_lists.append(prompt_predictions)
        if save_gates:
            gate_lists.append(prompt_gates)
        print(f"Accuracy on {subject_name} for prompt {prompt_index} is {acc:.3f}")

    scores = np.array(
        [[[answer[1] for answer in question] for question in predictions] for predictions in prediction_lists],
        dtype=np.float32,
    )
    targets = np.array([ANSWER_MAP[answer] for answer in solution_answers], dtype=np.int64)

    scores_path = output_dir / f"{subject_name}_scores.npy"
    targets_path = output_dir / f"{subject_name}_targets.npy"
    np.save(scores_path, scores)
    np.save(targets_path, targets)

    gate_shape = None
    if save_gates:
        gates = np.array(gate_lists, dtype=np.float32)
        gates_path = output_dir / f"{subject_name}_gates.npy"
        np.save(gates_path, gates)
        gate_shape = list(gates.shape)

    accuracy_dict[subject_name] = acc_list
    with accuracy_pickle_path(output_dir, len(prompts)).open("wb") as handle:
        pickle.dump(accuracy_dict, handle)

    return {
        "question_count": question_count,
        "score_shape": list(scores.shape),
        "target_shape": list(targets.shape),
        "gate_shape": gate_shape,
        "accuracy_mean": float(np.mean(np.array(acc_list))),
        "accuracy_by_prompt": acc_list,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Mixtral GPT-prompt multiple-choice scores for ConformalLLM.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subjects", default=None, help="Comma-separated subset of underscore task ids.")
    parser.add_argument("--num-prompts", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-gates", action="store_true")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_questions is not None and args.max_questions <= 0:
        parser.error("--max-questions must be a positive integer.")
    if args.max_memory_per_gpu_gib <= 0:
        parser.error("--max-memory-per-gpu-gib must be a positive integer.")

    try:
        subjects = parse_subjects(args.subjects)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = initialize_manifest(
        manifest_path,
        model_id=args.model_id,
        num_prompts=args.num_prompts,
        token_limit=DEFAULT_TOKEN_LIMIT,
        seed=args.seed,
        max_questions=args.max_questions,
        save_gates=args.save_gates,
        gate_pooling=DEFAULT_GATE_POOLING if args.save_gates else None,
        max_memory_per_gpu_gib=args.max_memory_per_gpu_gib,
    )

    accuracy_dict = {}
    accuracy_path = accuracy_pickle_path(output_dir, args.num_prompts)
    if accuracy_path.exists():
        try:
            with accuracy_path.open("rb") as handle:
                accuracy_dict = pickle.load(handle)
        except Exception:
            accuracy_dict = {}

    for subject_name in subjects:
        subject_prompts = PROMPTS_BY_SUBJECT[subject_name][: args.num_prompts]

        if args.skip_existing and should_skip_subject(
            output_dir=output_dir,
            subject_name=subject_name,
            num_prompts=args.num_prompts,
            save_gates=args.save_gates,
            manifest=manifest,
            max_questions=args.max_questions,
        ):
            print(f"Skipping {subject_name}: matching outputs already exist.")
            scores = np.load(output_dir / f"{subject_name}_scores.npy")
            targets = np.load(output_dir / f"{subject_name}_targets.npy")
            gate_shape = None
            if args.save_gates:
                gates = np.load(output_dir / f"{subject_name}_gates.npy")
                gate_shape = list(gates.shape)
            manifest["subjects"][subject_name] = {
                "question_count": int(targets.shape[0]),
                "score_shape": list(scores.shape),
                "target_shape": list(targets.shape),
                "gate_shape": gate_shape,
            }
            write_manifest(manifest_path, manifest)
            continue

        subject_metadata = run_subject(
            subject_name=subject_name,
            prompts=subject_prompts,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            token_limit=DEFAULT_TOKEN_LIMIT,
            max_questions=args.max_questions,
            accuracy_dict=accuracy_dict,
            save_gates=args.save_gates,
        )
        manifest["subjects"][subject_name] = subject_metadata
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
