from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


BASE_MODEL = "google/gemma-2-2b"
MODEL_IDS = {
    "base": BASE_MODEL,
    "math_ft": "MergeBench/gemma-2-2b_math",
    "coding_ft": "MergeBench/gemma-2-2b_coding",
}
MODEL_TARGETS = ("merged", *MODEL_IDS.keys())
LLM_TASKS = ("gsm8k", "math", "humaneval", "mbpp")
DIAGNOSTIC_TASKS = ("diagnostics",)


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _limit_items(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None or limit < 0:
        return items
    return items[:limit]


def _task_limit(config: dict[str, Any], task: str) -> int | None:
    task_specific = config.get(f"{task}_limit")
    if task_specific is not None and int(task_specific) >= 0:
        return int(task_specific)
    code_limit = config.get("code_limit")
    if task in {"humaneval", "mbpp"} and code_limit is not None and int(code_limit) >= 0:
        return int(code_limit)
    generic = config.get("limit")
    if generic is None:
        return None
    generic = int(generic)
    return None if generic < 0 else generic


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN in the environment or create the Modal secret "
            "`HF_TOKEN` with an HF_TOKEN key."
        )
    return token


def run_synthetic(
    output_dir: Path,
    n: int = 500,
    epochs: int = 3,
    seq_len: int = 16,
    eval_n: int = 100,
    max_eval_len: int = 31,
    seed: int = 0,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import GPT2Config, GPT2LMHeadModel

    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)

    def encode(s: str):
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)

    def decode(t) -> str:
        return "".join(itos[int(i)] for i in t)

    def make_example() -> str:
        return "".join(random.choice(chars) for _ in range(seq_len))

    def build_dataset(transform_fn: Callable[[str], str]):
        xs, ys = [], []
        for _ in range(n):
            s = make_example()
            xs.append(encode(s))
            ys.append(encode(transform_fn(s)))
        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    def make_model():
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=32,
            n_embd=64,
            n_layer=2,
            n_head=2,
        )
        return GPT2LMHeadModel(config).to(device)

    def train(model, x, y) -> list[float]:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        model.train()
        for _ in range(epochs):
            total = 0.0
            for i in range(len(x)):
                inp = x[i].unsqueeze(0)
                target = y[i].unsqueeze(0)
                out = model(inp).logits
                loss = loss_fn(out.view(-1, vocab_size), target.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            losses.append(total / len(x))
        return losses

    def run(model, s: str) -> str:
        model.eval()
        with torch.no_grad():
            inp = encode(s).unsqueeze(0).to(device)
            out = model(inp).logits.argmax(-1)
        return decode(out[0].cpu())

    def combined_run(m1, m2, m3, s: str) -> str:
        m1.eval()
        m2.eval()
        m3.eval()
        inp = encode(s).unsqueeze(0).to(device)
        with torch.no_grad():
            logp1 = F.log_softmax(m1(inp).logits, dim=-1)
            logp2 = F.log_softmax(m2(inp).logits, dim=-1)
            logp3 = F.log_softmax(m3(inp).logits, dim=-1)
            out_tokens = (logp1 + logp2 - logp3).argmax(dim=-1)
        return decode(out_tokens[0].cpu())

    transforms = {
        "A_to_K": lambda s: s.replace("A", "K"),
        "M_to_B": lambda s: s.replace("M", "B"),
        "identity": lambda s: s,
    }
    datasets = {name: build_dataset(fn) for name, fn in transforms.items()}
    models = {name: make_model() for name in transforms}
    losses = {
        name: train(models[name], datasets[name][0], datasets[name][1])
        for name in transforms
    }

    tests = ["AMMA", "HELLO", "MAP", "GAMMA", "AAA MMM", "TEST AM"]
    rows = []
    for test in tests:
        rows.append(
            {
                "input": test,
                "A_to_K": run(models["A_to_K"], test),
                "M_to_B": run(models["M_to_B"], test),
                "identity": run(models["identity"], test),
                "combined": combined_run(
                    models["A_to_K"],
                    models["M_to_B"],
                    models["identity"],
                    test,
                ),
            }
        )

    _jsonl_write(output_dir / "synthetic_predictions.jsonl", rows)
    length_rows = []
    for length in range(1, max_eval_len + 1):
        totals = {
            "A_to_K_exact": 0,
            "M_to_B_exact": 0,
            "identity_exact": 0,
            "combined_exact": 0,
        }
        for _ in range(eval_n):
            s = "".join(random.choice(chars) for _ in range(length))
            a_to_k = s.replace("A", "K")
            m_to_b = s.replace("M", "B")
            identity_s = s
            combined = a_to_k.replace("M", "B")
            totals["A_to_K_exact"] += int(run(models["A_to_K"], s) == a_to_k)
            totals["M_to_B_exact"] += int(run(models["M_to_B"], s) == m_to_b)
            totals["identity_exact"] += int(run(models["identity"], s) == identity_s)
            totals["combined_exact"] += int(
                combined_run(
                    models["A_to_K"],
                    models["M_to_B"],
                    models["identity"],
                    s,
                )
                == combined
            )
        length_rows.append(
            {
                "length": length,
                "n": eval_n,
                **{key: value / eval_n for key, value in totals.items()},
            }
        )

    length_eval_path = output_dir / "synthetic_length_eval.jsonl"
    _jsonl_write(length_eval_path, length_rows)
    return {
        "task": "synthetic",
        "dataset_size": n,
        "epochs": epochs,
        "device": str(device),
        "losses": losses,
        "predictions_file": str(output_dir / "synthetic_predictions.jsonl"),
        "length_eval_file": str(length_eval_path),
        "length_eval": length_rows,
    }


def _torch_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _model_device(model):
    return model.get_input_embeddings().weight.device


def load_generator(target: str, max_new_tokens: int, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _hf_token()
    dtype = _torch_dtype(dtype_name)
    device_map = "auto" if torch.cuda.is_available() else None

    def load_one(model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "token": token,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map is None:
            model = model.to("cpu")
        model.eval()
        return tokenizer, model

    @torch.inference_mode()
    def single_model_generate(tokenizer, model, prompt: str) -> str:
        dev = _model_device(model)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = output[0, enc["input_ids"].shape[1] :]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def merged_generate(tokenizer, models, prompt: str) -> str:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True)
        states: dict[str, dict[str, Any]] = {}
        model_devices = {name: _model_device(model) for name, model in models.items()}

        for name, model in models.items():
            dev = model_devices[name]
            out = model(
                input_ids=enc["input_ids"].to(dev),
                attention_mask=enc["attention_mask"].to(dev),
                use_cache=True,
            )
            states[name] = {
                "pkv": out.past_key_values,
                "attn": enc["attention_mask"].to(dev),
                "logits": out.logits[:, -1, :].cpu(),
            }

        generated_ids: list[int] = []
        eos_id = tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            logp_base = torch.log_softmax(states["base"]["logits"], dim=-1)
            logp_math = torch.log_softmax(states["math_ft"]["logits"], dim=-1)
            logp_code = torch.log_softmax(states["coding_ft"]["logits"], dim=-1)
            merged_logp = logp_math + logp_code - logp_base.clamp(min=-1e4)
            next_token = torch.argmax(merged_logp, dim=-1, keepdim=True)

            token_id = int(next_token.item())
            if eos_id is not None and token_id == eos_id:
                break
            generated_ids.append(token_id)

            for name, model in models.items():
                dev = model_devices[name]
                states[name]["attn"] = torch.cat(
                    [
                        states[name]["attn"],
                        torch.ones(
                            (1, 1),
                            device=dev,
                            dtype=states[name]["attn"].dtype,
                        ),
                    ],
                    dim=1,
                )
                out = model(
                    input_ids=next_token.to(dev),
                    attention_mask=states[name]["attn"],
                    past_key_values=states[name]["pkv"],
                    use_cache=True,
                )
                states[name]["pkv"] = out.past_key_values
                states[name]["logits"] = out.logits[:, -1, :].cpu()

        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    if target == "merged":
        tokenizer, base = load_one(MODEL_IDS["base"])
        _, math_ft = load_one(MODEL_IDS["math_ft"])
        _, coding_ft = load_one(MODEL_IDS["coding_ft"])
        models = {"base": base, "math_ft": math_ft, "coding_ft": coding_ft}
        return lambda prompt: merged_generate(tokenizer, models, prompt)

    tokenizer, model = load_one(MODEL_IDS[target])
    return lambda prompt: single_model_generate(tokenizer, model, prompt)


def extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    return match.group(1).strip() if match else text.strip()


def normalize_math_answer(ans: str) -> str:
    ans = ans.replace(" ", "").strip().lower()
    try:
        return str(float(ans))
    except ValueError:
        return ans


def score_math(pred: str, ref: str) -> bool:
    return normalize_math_answer(extract_boxed(pred)) == normalize_math_answer(extract_boxed(ref))


def normalize_gsm8k_answer(ans: str) -> str:
    ans = ans.strip().lower().replace(",", "")
    ans = ans.replace("$", "")
    ans = re.sub(r"\s+", "", ans)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", ans)
    if match:
        num = match.group(0)
        try:
            return str(float(num)) if "." in num else str(int(num))
        except ValueError:
            return num
    return ans


def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return normalize_gsm8k_answer(answer)


def extract_gsm8k_pred(text: str) -> str:
    if "####" in text:
        return normalize_gsm8k_answer(text.split("####")[-1])
    lowered = text.lower()
    patterns = [
        r"(?:the answer is|final answer is|answer:)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"([-+]?\d[\d,]*(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return normalize_gsm8k_answer(match.group(1))
    return normalize_gsm8k_answer(text)


def build_math_prompt(problem: str) -> str:
    few_shot = """Solve the following math problem step by step. Put your final answer in \\boxed{}.
Problem: What is $2^{10}$?
Solution: $2^{10} = 1024$. The answer is $\\boxed{1024}$.
Problem: Simplify $\\frac{x^2 - 1}{x - 1}$.
Solution: $\\frac{x^2-1}{x-1} = \\frac{(x+1)(x-1)}{x-1} = x+1$. The answer is $\\boxed{x+1}$.
"""
    return few_shot + f"Problem: {problem}\nSolution:"


def build_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math word problem step by step. "
        "End your response with '#### <final numeric answer>'.\n"
        f"Question: {question}\nAnswer:"
    )


def load_all_models(dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _hf_token()
    dtype = _torch_dtype(dtype_name)
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models = {}
    for name, model_id in MODEL_IDS.items():
        kwargs: dict[str, Any] = {
            "token": token,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device_map is None:
            model = model.to("cpu")
        model.eval()
        models[name] = model
    return tokenizer, models


def _diagnostic_prompt_sets(limit: int, seed: int) -> dict[str, list[str]]:
    from datasets import concatenate_datasets, load_dataset
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    rng = random.Random(seed)

    gsm8k_ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed)
    gsm8k = [
        build_gsm8k_prompt(ex["question"])
        for ex in gsm8k_ds.select(range(min(limit, len(gsm8k_ds))))
    ]

    try:
        math_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "all", split="test")
    except Exception:
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        math_ds = concatenate_datasets(
            [load_dataset("DigitalLearningGmbH/MATH-lighteval", subj, split="test") for subj in subjects]
        )
    math_ds = math_ds.shuffle(seed=seed)
    math_prompts = [
        build_math_prompt(ex["problem"])
        for ex in math_ds.select(range(min(limit, len(math_ds))))
    ]

    humaneval_items = list(get_human_eval_plus().items())
    mbpp_items = list(get_mbpp_plus().items())
    rng.shuffle(humaneval_items)
    rng.shuffle(mbpp_items)

    generic = [
        "Write a concise project update for a team that missed its Friday deadline.",
        "Explain why keeping a changelog helps maintainers review software changes.",
        "List three practical tradeoffs when choosing between latency and throughput.",
        "Draft a short note asking a collaborator to clarify an ambiguous requirement.",
        "Summarize the main risks of relying on a single benchmark for model evaluation.",
        "Describe how to debug a flaky integration test in a continuous integration system.",
        "Give a neutral explanation of why a product launch might be postponed.",
        "Outline a small experiment that would test whether a cache improves performance.",
    ][:limit]

    return {
        "gsm8k": gsm8k,
        "math": math_prompts,
        "humaneval": [problem["prompt"] for _, problem in humaneval_items[:limit]],
        "mbpp": [problem["prompt"] for _, problem in mbpp_items[:limit]],
        "generic": generic,
    }


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    total_weight = sum(int(row["n_tokens"]) for row in rows)
    if total_weight == 0:
        return None
    return sum(float(row[key]) * int(row["n_tokens"]) for row in rows) / total_weight


def run_factorization_diagnostics(
    output_dir: Path,
    limit: int = 4,
    max_seq_tokens: int = 128,
    dtype_name: str = "auto",
    seed: int = 0,
    max_examples: int = 24,
) -> dict[str, Any]:
    import torch

    tokenizer, models = load_all_models(dtype_name)
    prompt_sets = _diagnostic_prompt_sets(limit=limit, seed=seed)
    thresholds = (0.01, 0.1, 1.0)
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for domain, prompts in prompt_sets.items():
        for prompt_idx, prompt in enumerate(prompts):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_tokens,
            )
            if enc["input_ids"].shape[1] < 2:
                continue

            logps: dict[str, torch.Tensor] = {}
            top1: dict[str, torch.Tensor] = {}
            for name, model in models.items():
                dev = _model_device(model)
                with torch.inference_mode():
                    out = model(
                        input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev),
                    )
                logits = out.logits[:, :-1, :].float().cpu()
                logps[name] = torch.log_softmax(logits, dim=-1)[0]
                top1[name] = logps[name].argmax(dim=-1)
                del out, logits

            base_logp = logps["base"]
            math_logp = logps["math_ft"]
            code_logp = logps["coding_ft"]

            math_prob = math_logp.exp()
            code_prob = code_logp.exp()
            kl_math = (math_prob * (math_logp - base_logp)).sum(dim=-1)
            kl_code = (code_prob * (code_logp - base_logp)).sum(dim=-1)
            del math_prob, code_prob

            merged_logp = math_logp + code_logp - base_logp.clamp(min=-1e4)
            merged_logp = merged_logp - torch.logsumexp(merged_logp, dim=-1, keepdim=True)
            merged_top1 = merged_logp.argmax(dim=-1)

            n_tokens = int(kl_math.numel())
            row: dict[str, Any] = {
                "domain": domain,
                "prompt_idx": prompt_idx,
                "n_tokens": n_tokens,
                "mean_kl_math_to_base": float(kl_math.mean().item()),
                "mean_kl_code_to_base": float(kl_code.mean().item()),
                "median_kl_math_to_base": float(kl_math.median().item()),
                "median_kl_code_to_base": float(kl_code.median().item()),
                "math_base_top1_disagree_fraction": float((top1["math_ft"] != top1["base"]).float().mean().item()),
                "code_base_top1_disagree_fraction": float((top1["coding_ft"] != top1["base"]).float().mean().item()),
                "math_code_top1_disagree_fraction": float((top1["math_ft"] != top1["coding_ft"]).float().mean().item()),
                "merged_new_top1_fraction": float(
                    (
                        (merged_top1 != top1["base"])
                        & (merged_top1 != top1["math_ft"])
                        & (merged_top1 != top1["coding_ft"])
                    )
                    .float()
                    .mean()
                    .item()
                ),
            }

            for threshold in thresholds:
                suffix = str(threshold).replace(".", "_")
                math_active = kl_math > threshold
                code_active = kl_code > threshold
                row[f"both_active_fraction_kl_gt_{suffix}"] = float(
                    (math_active & code_active).float().mean().item()
                )
                row[f"only_math_active_fraction_kl_gt_{suffix}"] = float(
                    (math_active & ~code_active).float().mean().item()
                )
                row[f"only_code_active_fraction_kl_gt_{suffix}"] = float(
                    (~math_active & code_active).float().mean().item()
                )
                row[f"neither_active_fraction_kl_gt_{suffix}"] = float(
                    (~math_active & ~code_active).float().mean().item()
                )

            rows.append(row)

            new_top1 = (
                (merged_top1 != top1["base"])
                & (merged_top1 != top1["math_ft"])
                & (merged_top1 != top1["coding_ft"])
            )
            input_ids = enc["input_ids"][0].cpu()
            for pos_tensor in torch.nonzero(new_top1, as_tuple=False).flatten():
                if len(examples) >= max_examples:
                    break
                pos = int(pos_tensor.item())
                context_start = max(0, pos - 12)
                examples.append(
                    {
                        "domain": domain,
                        "prompt_idx": prompt_idx,
                        "position": pos,
                        "context_tail": tokenizer.decode(input_ids[context_start : pos + 1]),
                        "base_top1": tokenizer.decode([int(top1["base"][pos].item())]),
                        "math_top1": tokenizer.decode([int(top1["math_ft"][pos].item())]),
                        "code_top1": tokenizer.decode([int(top1["coding_ft"][pos].item())]),
                        "merged_top1": tokenizer.decode([int(merged_top1[pos].item())]),
                        "kl_math_to_base": float(kl_math[pos].item()),
                        "kl_code_to_base": float(kl_code[pos].item()),
                    }
                )

    domain_summary = []
    metric_keys = [
        "mean_kl_math_to_base",
        "mean_kl_code_to_base",
        "math_base_top1_disagree_fraction",
        "code_base_top1_disagree_fraction",
        "math_code_top1_disagree_fraction",
        "merged_new_top1_fraction",
        "both_active_fraction_kl_gt_0_01",
        "both_active_fraction_kl_gt_0_1",
        "both_active_fraction_kl_gt_1_0",
    ]
    for domain in sorted({row["domain"] for row in rows}):
        domain_rows = [row for row in rows if row["domain"] == domain]
        summary_row: dict[str, Any] = {
            "domain": domain,
            "n_prompts": len(domain_rows),
            "n_tokens": sum(int(row["n_tokens"]) for row in domain_rows),
        }
        for key in metric_keys:
            summary_row[key] = _weighted_mean(domain_rows, key)
        domain_summary.append(summary_row)

    prompt_path = output_dir / "factorization_diagnostic_prompts.jsonl"
    summary_path = output_dir / "factorization_diagnostic_by_domain.jsonl"
    examples_path = output_dir / "factorization_diagnostic_examples.jsonl"
    _jsonl_write(prompt_path, rows)
    _jsonl_write(summary_path, domain_summary)
    _jsonl_write(examples_path, examples)
    return {
        "task": "diagnostics",
        "limit_per_domain": limit,
        "max_seq_tokens": max_seq_tokens,
        "prompt_metrics_file": str(prompt_path),
        "domain_summary_file": str(summary_path),
        "examples_file": str(examples_path),
        "domain_summary": domain_summary,
        "n_new_top1_examples_saved": len(examples),
    }


def evaluate_math(
    generator: Callable[[str], str],
    output_dir: Path,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    from datasets import concatenate_datasets, load_dataset
    from tqdm import tqdm

    try:
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "all", split="test")
    except Exception:
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        ds = concatenate_datasets(
            [load_dataset("DigitalLearningGmbH/MATH-lighteval", subj, split="test") for subj in subjects]
        )
    ds = ds.shuffle(seed=seed)
    total_limit = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    correct = 0
    for idx, ex in tqdm(
        enumerate(ds.select(range(total_limit))),
        total=total_limit,
        desc="MATH",
        unit="problem",
    ):
        pred = generator(build_math_prompt(ex["problem"]))
        ok = score_math(pred, ex["solution"])
        correct += int(ok)
        rows.append(
            {
                "idx": idx,
                "level": ex.get("level", ""),
                "type": ex.get("type", ""),
                "problem": ex["problem"],
                "prediction": pred,
                "gold": ex["solution"],
                "correct": ok,
            }
        )
    out_file = output_dir / "math_answers.jsonl"
    _jsonl_write(out_file, rows)
    return {
        "task": "math",
        "variant": "two_shot_notebook_prompt",
        "correct": correct,
        "total": total_limit,
        "accuracy": correct / total_limit if total_limit else None,
        "answers_file": str(out_file),
    }


def evaluate_gsm8k(
    generator: Callable[[str], str],
    output_dir: Path,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed)
    total_limit = len(ds) if limit is None else min(limit, len(ds))
    rows = []
    correct = 0
    for idx, ex in tqdm(
        enumerate(ds.select(range(total_limit))),
        total=total_limit,
        desc="GSM8K",
        unit="problem",
    ):
        pred = generator(build_gsm8k_prompt(ex["question"]))
        pred_answer = extract_gsm8k_pred(pred)
        gold_answer = extract_gsm8k_gold(ex["answer"])
        ok = pred_answer == gold_answer
        correct += int(ok)
        rows.append(
            {
                "idx": idx,
                "question": ex["question"],
                "prediction": pred,
                "gold": ex["answer"],
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "correct": ok,
            }
        )
    out_file = output_dir / "gsm8k_answers.jsonl"
    _jsonl_write(out_file, rows)
    return {
        "task": "gsm8k",
        "variant": "direct_generation_prompt_not_lm_eval_8shot",
        "correct": correct,
        "total": total_limit,
        "accuracy": correct / total_limit if total_limit else None,
        "answers_file": str(out_file),
    }


def generate_evalplus_samples(
    task: str,
    generator: Callable[[str], str],
    output_dir: Path,
    limit: int | None,
    run_evalplus: bool,
) -> dict[str, Any]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from tqdm import tqdm

    getters = {
        "humaneval": get_human_eval_plus,
        "mbpp": get_mbpp_plus,
    }
    problems = getters[task]()
    items = _limit_items(list(problems.items()), limit)
    rows = []
    failed = 0
    for task_id, problem in tqdm(items, desc=task, unit="problem"):
        try:
            completion = generator(problem["prompt"])
        except Exception as exc:
            failed += 1
            completion = ""
            print(f"{task_id} failed: {exc}", file=sys.stderr)
        rows.append({"task_id": task_id, "completion": completion})

    samples_path = output_dir / f"samples_{task}.jsonl"
    _jsonl_write(samples_path, rows)
    result: dict[str, Any] = {
        "task": task,
        "total": len(items),
        "failed_generation": failed,
        "samples_file": str(samples_path),
    }

    if run_evalplus and rows:
        if len(items) != len(problems):
            result.update(
                {
                    "evalplus_skipped": True,
                    "evalplus_skip_reason": (
                        "EvalPlus requires samples for every problem; "
                        f"generated {len(items)} of {len(problems)} due to the configured limit."
                    ),
                }
            )
            return result

        cmd = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            "--dataset",
            task,
            "--samples",
            str(samples_path),
        ]
        proc = subprocess.run(cmd, cwd=output_dir, text=True, capture_output=True, check=False)
        stdout_path = output_dir / f"evalplus_{task}.stdout.txt"
        stderr_path = output_dir / f"evalplus_{task}.stderr.txt"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        result.update(
            {
                "evalplus_returncode": proc.returncode,
                "evalplus_stdout_file": str(stdout_path),
                "evalplus_stderr_file": str(stderr_path),
            }
        )
    return result


def _wandb_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for task in summary.get("tasks", []):
        task_name = task.get("task", "unknown")
        for key in ("accuracy", "correct", "total", "failed_generation", "evalplus_returncode"):
            value = task.get(key)
            if isinstance(value, (int, float)) and value is not None:
                metrics[f"{task_name}/{key}"] = float(value)

        if task_name == "synthetic":
            length_eval = task.get("length_eval", [])
            for key in ("A_to_K_exact", "M_to_B_exact", "identity_exact", "combined_exact"):
                values = [
                    row.get(key)
                    for row in length_eval
                    if isinstance(row.get(key), (int, float))
                ]
                if values:
                    metrics[f"synthetic/min_{key}"] = float(min(values))
                    metrics[f"synthetic/mean_{key}"] = float(sum(values) / len(values))

        if task_name == "diagnostics":
            for row in task.get("domain_summary", []):
                domain = row.get("domain", "unknown")
                for key, value in row.items():
                    if key in {"domain", "n_prompts"}:
                        continue
                    if isinstance(value, (int, float)) and value is not None:
                        metrics[f"diagnostics/{domain}/{key}"] = float(value)

    elapsed = summary.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        metrics["run/elapsed_seconds"] = float(elapsed)
    return metrics


def _wandb_task_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in summary.get("tasks", []):
        rows.append(
            {
                "task": task.get("task"),
                "total": task.get("total"),
                "correct": task.get("correct"),
                "accuracy": task.get("accuracy"),
                "failed_generation": task.get("failed_generation"),
                "evalplus_skipped": task.get("evalplus_skipped", False),
                "evalplus_returncode": task.get("evalplus_returncode"),
                "notes": task.get("evalplus_skip_reason") or task.get("variant", ""),
            }
        )
    return rows


def _wandb_table_from_rows(wandb_module, rows: list[dict[str, Any]], columns: list[str]):
    table = wandb_module.Table(columns=columns)
    for row in rows:
        table.add_data(*(row.get(column) for column in columns))
    return table


def _start_wandb(config: dict[str, Any], output_dir: Path):
    if not config.get("use_wandb"):
        return None

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "use_wandb is true, but WANDB_API_KEY is not set. "
            "Create the Modal secret `WANDB_API_KEY` or export WANDB_API_KEY."
        )

    import wandb

    project = config.get("wandb_project") or "logit-composition-replication"
    entity = config.get("wandb_entity") or None
    run_name = config.get("job_name") or output_dir.name
    wandb_dir = Path(os.environ.get("WANDB_DIR", "/tmp/wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        dir=str(wandb_dir),
    )
    return run


def _finish_wandb(run, summary: dict[str, Any], output_dir: Path) -> None:
    if run is None:
        return

    import wandb

    metrics = _wandb_metrics(summary)
    if metrics:
        wandb.log(metrics)

    task_columns = [
        "task",
        "total",
        "correct",
        "accuracy",
        "failed_generation",
        "evalplus_skipped",
        "evalplus_returncode",
        "notes",
    ]
    task_rows = _wandb_task_rows(summary)
    if task_rows:
        task_table = _wandb_table_from_rows(wandb, task_rows, task_columns)
        accuracy_rows = [
            row
            for row in task_rows
            if isinstance(row.get("accuracy"), (int, float))
        ]
        payload: dict[str, Any] = {"tables/task_summary": task_table}
        if accuracy_rows:
            accuracy_table = _wandb_table_from_rows(
                wandb,
                accuracy_rows,
                ["task", "accuracy"],
            )
            payload["charts/accuracy_by_task"] = wandb.plot.bar(
                accuracy_table,
                "task",
                "accuracy",
                title="Accuracy by task",
            )
        wandb.log(payload)

    run.summary.update(
        {
            "output_dir": str(output_dir),
            "elapsed_seconds": summary.get("elapsed_seconds"),
            "tasks": [task.get("task") for task in summary.get("tasks", [])],
            "task_summary": task_rows,
        }
    )

    artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run.name or output_dir.name).strip("-")
    artifact = wandb.Artifact(
        name=f"{artifact_name}-outputs",
        type="replication-results",
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    wandb.finish()


def run_from_config(config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_config(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(config["seed"]))
    wandb_run = _start_wandb(config, output_dir)

    started = time.time()
    summary: dict[str, Any] = {
        "config": config,
        "started_unix": started,
        "tasks": [],
    }

    if config["include_synthetic"]:
        summary["tasks"].append(
            run_synthetic(
                output_dir=output_dir,
                n=int(config["synthetic_n"]),
                epochs=int(config["synthetic_epochs"]),
                seq_len=int(config["synthetic_seq_len"]),
                eval_n=int(config["synthetic_eval_n"]),
                max_eval_len=int(config["synthetic_max_eval_len"]),
                seed=int(config["seed"]),
            )
        )

    if "diagnostics" in config["tasks"]:
        diagnostic_limit = _task_limit(config, "diagnostics")
        if diagnostic_limit is None:
            diagnostic_limit = 8
        summary["tasks"].append(
            run_factorization_diagnostics(
                output_dir=output_dir,
                limit=int(diagnostic_limit),
                max_seq_tokens=int(config["diagnostic_max_seq_tokens"]),
                dtype_name=config["dtype"],
                seed=int(config["seed"]),
            )
        )

    tasks = [task for task in config["tasks"] if task in LLM_TASKS]
    if tasks:
        generator = load_generator(
            target=config["target"],
            max_new_tokens=int(config["max_new_tokens"]),
            dtype_name=config["dtype"],
        )
        if "gsm8k" in tasks:
            summary["tasks"].append(
                evaluate_gsm8k(
                    generator,
                    output_dir,
                    limit=_task_limit(config, "gsm8k"),
                    seed=int(config["seed"]),
                )
            )
        if "math" in tasks:
            summary["tasks"].append(
                evaluate_math(
                    generator,
                    output_dir,
                    limit=_task_limit(config, "math"),
                    seed=int(config["seed"]),
                )
            )
        for code_task in ("humaneval", "mbpp"):
            if code_task in tasks:
                summary["tasks"].append(
                    generate_evalplus_samples(
                        code_task,
                        generator,
                        output_dir,
                        limit=_task_limit(config, code_task),
                        run_evalplus=bool(config["run_evalplus"]),
                    )
                )

    summary["finished_unix"] = time.time()
    summary["elapsed_seconds"] = summary["finished_unix"] - started
    _write_json(output_dir / "summary.json", summary)
    _finish_wandb(wandb_run, summary, output_dir)
    return summary


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "target": "merged",
        "tasks": list(LLM_TASKS),
        "limit": 3,
        "gsm8k_limit": None,
        "math_limit": None,
        "code_limit": None,
        "max_new_tokens": 256,
        "dtype": "auto",
        "run_evalplus": False,
        "seed": 0,
        "include_synthetic": False,
        "synthetic_n": 500,
        "synthetic_epochs": 3,
        "synthetic_seq_len": 16,
        "synthetic_eval_n": 100,
        "synthetic_max_eval_len": 31,
        "diagnostic_max_seq_tokens": 128,
        "use_wandb": False,
        "wandb_project": "logit-composition-replication",
        "wandb_entity": "",
        "output_dir": "results/partial",
    }
    normalized = {**defaults, **config}
    if isinstance(normalized["tasks"], str):
        normalized["tasks"] = [
            task.strip().lower()
            for task in normalized["tasks"].split(",")
            if task.strip()
        ]
    target = normalized["target"]
    if target not in MODEL_TARGETS:
        raise ValueError(f"Unknown target {target!r}; expected one of {MODEL_TARGETS}")
    unknown_tasks = sorted(set(normalized["tasks"]) - set((*LLM_TASKS, *DIAGNOSTIC_TASKS, "synthetic")))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if "synthetic" in normalized["tasks"]:
        normalized["include_synthetic"] = True
        normalized["tasks"] = [task for task in normalized["tasks"] if task != "synthetic"]
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded partial replication.")
    parser.add_argument("--target", choices=MODEL_TARGETS, default="merged")
    parser.add_argument("--tasks", default="gsm8k,math,humaneval,mbpp")
    parser.add_argument("--limit", type=int, default=3, help="Generic per-task limit. Use -1 for full.")
    parser.add_argument("--gsm8k-limit", type=int, default=None)
    parser.add_argument("--math-limit", type=int, default=None)
    parser.add_argument("--code-limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--run-evalplus", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=500)
    parser.add_argument("--synthetic-epochs", type=int, default=3)
    parser.add_argument("--synthetic-seq-len", type=int, default=16)
    parser.add_argument("--synthetic-eval-n", type=int, default=100)
    parser.add_argument("--synthetic-max-eval-len", type=int, default=31)
    parser.add_argument("--diagnostic-max-seq-tokens", type=int, default=128)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="logit-composition-replication")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--output-dir", default="results/partial")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_from_config(vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
