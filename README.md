# Supplementary Code for Compositional Generalization in Autoregressive Models via Logit Composition

This repository contains the notebook used for the synthetic letter-replacement experiment and the logit-composition LLM evaluations.

## Environment

- Python 3.12 was used for the submitted runs.
- Install the notebook dependencies with:

```bash
pip install torch transformers datasets lm_eval evalplus pandas tqdm
```

- The LLM evaluations require Hugging Face access to `google/gemma-2-2b` and the MergeBench checkpoints. Set the token outside the notebook:

```bash
export HF_TOKEN=<your-huggingface-token>
```

The notebook intentionally does not contain tokens, author names, local paths, or institution-specific paths.

## Models

- Base: `google/gemma-2-2b`
- Math expert: `MergeBench/gemma-2-2b_math`
- Coding expert: `MergeBench/gemma-2-2b_coding`

The merged decoder uses greedy decoding with `MAX_NEW = 512`, `SAMPLE_MERGED = False`, `torch.manual_seed(0)`, float16 on CUDA, and `device_map="auto"` when CUDA is available.

## Benchmarks

- GSM8K: `lm_eval.simple_evaluate`, task `gsm8k`, 8-shot, batch size 1, flexible exact-match score reported.
- MATH: `DigitalLearningGmbH/MATH-lighteval`, all test subjects, two-shot prompt in the notebook, boxed-answer exact match after light normalization.
- HumanEval+: `evalplus`, dataset `humaneval`, pass@1.
- MBPP+: `evalplus`, dataset `mbpp`, pass@1.

## Partial Modal Replication

This repo also includes a bounded Modal runner for GPU-backed sanity checks before spending credits on a full replication. It runs the logit-composed model by default and writes all artifacts to a persistent Modal Volume named `logit-composition-replication`.

Install the local Modal client:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-modal.txt
python -m modal setup
```

Provide Hugging Face access to `google/gemma-2-2b` and the MergeBench checkpoints. Either export `HF_TOKEN` before running Modal, or create a Modal Secret named `HF_TOKEN` with an `HF_TOKEN` key:

```bash
modal secret create HF_TOKEN HF_TOKEN="$HF_TOKEN"
```

Set `MODAL_HF_SECRET_NAME=<secret-name>` if you want the runner to use a different Modal Secret name.

To log to Weights & Biases, create or reuse a Modal Secret named `WANDB_API_KEY` with a `WANDB_API_KEY` key:

```bash
modal secret create WANDB_API_KEY WANDB_API_KEY="$WANDB_API_KEY"
```

Set `MODAL_WANDB_SECRET_NAME=<secret-name>` if you want the runner to use a different Modal Secret name.

Run a small GPU partial replication:

```bash
modal run modal_app.py --gpu L40S --target merged --limit 3 --max-new-tokens 256
```

For a cheaper smoke run:

```bash
modal run modal_app.py --gpu A10G --target merged --tasks gsm8k,math --limit 1 --max-new-tokens 96
```

Useful options:

- `--target merged|base|math_ft|coding_ft`
- `--tasks gsm8k,math,humaneval,mbpp,diagnostics`
- `--limit N`, or task-specific `--gsm8k-limit`, `--math-limit`, `--code-limit`
- `--run-evalplus` to run EvalPlus after generating HumanEval+/MBPP+ samples. EvalPlus scoring is skipped for partial code samples because EvalPlus expects one sample for every problem in the benchmark.
- `--use-wandb --wandb-project PROJECT [--wandb-entity ENTITY]` to log scalar metrics, result tables, charts, and an output artifact to W&B.
- `--job-name NAME` to choose the result folder under `/results/NAME` in the Modal Volume

The GSM8K partial runner uses the lightweight direct-generation prompt from `Baselines_Code.ipynb`, not the full `lm_eval` 8-shot harness used for the reported GSM8K table.

Run the factorization diagnostic used for review evidence:

```bash
modal run modal_app.py --gpu L40S --tasks diagnostics --limit 4 --diagnostic-max-seq-tokens 128 --job-name review-evidence
```

Add `--include-synthetic --synthetic-n 2000 --synthetic-epochs 5 --synthetic-eval-n 200` to also check the toy task at held-out lengths 1-31.

## Reported Results

| Benchmark | Base | Coding expert | Math expert | Logit composition |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 3.2 | 3.2 | 50.2 | 41.2 |
| MATH | 16.3 | 16.7 | 25.5 | 24.2 |
| HumanEval+ | 3.7 | 24.4 | 13.4 | 30.5 |
| MBPP+ | 33.9 | 38.1 | 32.8 | 39.4 |

The notebook writes generated samples for EvalPlus to `samples_humaneval.jsonl` and `samples_mbpp.jsonl`; these files are not included in the supplement.
