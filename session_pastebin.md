# Session Summary: Modal Partial Replication + W&B + Review Evidence

Repo:

`/Users/sheikheddy/Projects/Compositional-Generalization-in-Autoregressive-Models-via-Logit-Composition`

Sensitive values intentionally omitted. Secret names are included, but no HF or W&B token values are included.

## Initial Goal

Set up and run partial replication experiments for the paper/repo on Modal using cloud GPU credits, connect Weights & Biases, and gather empirical evidence for or against a draft review of the paper.

## Files Added Or Modified

- `.gitignore`
  - Ignores `.venv/`, `results/`, `wandb/`, Python caches, and generated sample/result files.
- `requirements-modal.txt`
  - Adds local Modal client dependency: `modal>=1.0`.
- `modal_app.py`
  - Modal app `logit-composition-partial-replication`.
  - Uses Modal Volume `logit-composition-replication`.
  - Uses Modal secrets named `HF_TOKEN` and `WANDB_API_KEY`.
  - Remote image installs torch, transformers, accelerate, datasets, evalplus, pandas, tqdm, sentencepiece, hf-transfer, wandb.
  - Supports GPU selection, bounded task limits, optional EvalPlus, optional synthetic run, and W&B logging.
- `src/logit_composition/__init__.py`
- `src/logit_composition/replication.py`
  - Extracted notebook logic into scriptable code.
  - Supports targets: `merged`, `base`, `math_ft`, `coding_ft`.
  - Supports tasks: `gsm8k`, `math`, `humaneval`, `mbpp`, `diagnostics`.
  - Adds synthetic held-out-length evaluation for lengths 1-31.
  - Adds factorization diagnostics: per-token expert-vs-base KL, top-1 disagreement, both-experts-active fractions, merged-new-top1 fraction.
  - Adds W&B scalar metrics, task summary tables, accuracy chart table, and output artifact upload.
  - Skips EvalPlus scoring for partial code samples because EvalPlus requires every benchmark problem to be present.
- `README.md`
  - Added Modal setup/run instructions.
  - Added W&B setup/run instructions.
  - Documented diagnostics command and partial EvalPlus caveat.

## Modal And W&B Setup

Modal profile was already authenticated as `sheikheddy`.

Available Modal secrets verified by name:

- `HF_TOKEN`
- `WANDB_API_KEY`

Local venv:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-modal.txt
.venv/bin/python -m pip install 'wandb>=0.18,<1'
```

## Review Evidence Run

Command:

```bash
.venv/bin/python -m modal run modal_app.py \
  --gpu L40S \
  --target merged \
  --tasks diagnostics \
  --limit 4 \
  --diagnostic-max-seq-tokens 128 \
  --include-synthetic \
  --synthetic-n 2000 \
  --synthetic-epochs 5 \
  --synthetic-eval-n 200 \
  --job-name review-evidence-20260708
```

Modal app:

`ap-1BUsB4XtY88XE83Iz9erPf`

Local artifacts:

`results/review-evidence-20260708/`

Main results:

- Synthetic toy task trained at length 16 and evaluated on lengths 1-31.
- 200 random strings per length.
- Exact accuracy was 100% for:
  - `A_to_K`
  - `M_to_B`
  - `identity`
  - combined `A_to_K + M_to_B`
- This pushes back on the review question asking whether exactness persists at unseen lengths 17-31, at least for the repo implementation and sampled evaluation.

LLM factorization diagnostic, 4 prompts per domain, max 128 tokens:

| Domain | Tokens | KL math->base | KL code->base | Both active KL>0.01 | Both active KL>0.1 | Merged top-1 new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generic | 51 | 0.149 | 0.245 | 98.0% | 39.2% | 2.0% |
| GSM8K | 333 | 0.164 | 0.163 | 88.0% | 36.0% | 11.4% |
| HumanEval | 508 | 0.112 | 0.124 | 87.0% | 29.1% | 7.9% |
| MATH | 508 | 0.211 | 0.145 | 68.5% | 24.4% | 3.9% |
| MBPP | 197 | 0.126 | 0.187 | 92.9% | 32.0% | 7.1% |

Interpretation:

- Supports the review criticism that the LLM experiments are disconnected from the exact factorization theory.
- Both experts diverge from the base on most tokens under a small KL threshold.
- Both experts are simultaneously active on a substantial fraction of tokens under KL>0.1, suggesting routing ambiguity.

## W&B-Connected Partial Replication

First W&B run:

- Job name: `partial-wandb-20260708`
- W&B run: `https://wandb.ai/cvpr-flux-sae/logit-composition-replication/runs/irhuchti`
- Outcome: completed, but EvalPlus returned code 1 because partial samples do not include every benchmark problem. This was a runner logic issue, not a generation failure.
- Superseded by clean run below.

Patch after first W&B run:

- If `--run-evalplus` is set but code samples are partial, skip EvalPlus scoring and record:
  - `evalplus_skipped: true`
  - reason: EvalPlus requires samples for every problem.

Clean W&B run command:

```bash
.venv/bin/python -m modal run modal_app.py \
  --gpu L40S \
  --target merged \
  --tasks gsm8k,math,humaneval,mbpp \
  --limit 3 \
  --max-new-tokens 256 \
  --run-evalplus \
  --use-wandb \
  --wandb-project logit-composition-replication \
  --job-name partial-wandb-20260708-clean
```

Clean W&B run:

`https://wandb.ai/cvpr-flux-sae/logit-composition-replication/runs/hndovb1p`

Modal app:

`ap-4FYvxuLydjMYfSo4jd2UPC`

Local artifacts:

`results/partial-wandb-20260708-clean/`

Clean run results:

- Elapsed remote runtime: 109.57s.
- GSM8K: 0/3, accuracy 0.0.
- MATH: 2/3, accuracy 0.6667.
- HumanEval+: generated 3/3, no generation failures.
- MBPP+: generated 3/3, no generation failures.
- EvalPlus scoring skipped for partial HumanEval+ because generated 3 of 164.
- EvalPlus scoring skipped for partial MBPP+ because generated 3 of 378.

Downloaded local files:

- `results/partial-wandb-20260708-clean/summary.json`
- `results/partial-wandb-20260708-clean/gsm8k_answers.jsonl`
- `results/partial-wandb-20260708-clean/math_answers.jsonl`
- `results/partial-wandb-20260708-clean/samples_humaneval.jsonl`
- `results/partial-wandb-20260708-clean/samples_mbpp.jsonl`

## W&B Empty UI Issue

User reported that the W&B run looked empty.

Checked via W&B API:

- Run state: finished.
- Summary metrics existed.
- One history row existed.
- Output artifact existed.

Likely issue:

- Original W&B logging only had one scalar row and no explicit visible tables/charts, making the UI look empty or unhelpful.

Fix:

- Patched `replication.py` to log:
  - scalar metrics,
  - `tables/task_summary`,
  - `charts/accuracy_by_task_table`,
  - output artifact.

Backfilled existing clean run without GPU:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path
import wandb
from logit_composition.replication import _finish_wandb

output_dir = Path('results/partial-wandb-20260708-clean')
summary = json.loads((output_dir / 'summary.json').read_text())
run = wandb.init(
    project='logit-composition-replication',
    entity='cvpr-flux-sae',
    id='hndovb1p',
    resume='allow',
    name='partial-wandb-20260708-clean',
)
_finish_wandb(run, summary, output_dir)
print('backfilled', run.url)
PY
```

Post-backfill W&B API verification:

- Run finished: true.
- Has `tables/task_summary`: true.
- Has `charts/accuracy_by_task_table`: true.
- Artifact count: 3.
- History rows: 3.

## Paper PDF Quick Check

PDF:

`/Users/sheikheddy/Downloads/18929_Compositional_Generaliza.pdf`

PDF metadata:

- 24 pages.
- Text extractable with `pdftotext`.

Confirmed paper-side review points from extracted text:

- Main toy example uses A->B and C->D.
- Appendix C toy implementation uses A->K and M->B.
- The paper explicitly states that the theory does not prove arbitrary pretrained/fine-tuned Transformers length-generalize, nor that approximate factorization persists outside training lengths.
- Limitations say factorization assumptions may not fully hold in practice, empirical evaluation uses one base model, and lacks strong model-merging comparisons.
- Checklist says no confidence intervals, bootstrap estimates, or repeated-run error bars.
- DExperts is cited.
- I did not find GeDi, proxy-tuning, or Dekoninck/model arithmetic by name in the extracted text.
- The paper references task arithmetic and several model-merging works, including TIES and model soups.

## Important Caveats

- The partial GSM8K runner uses a lightweight direct-generation prompt from the baseline script, not the full `lm_eval` 8-shot GSM8K protocol used in the reported table.
- Full EvalPlus scoring requires generating all HumanEval+ and MBPP+ samples.
- The diagnostics are small-sample mechanism evidence, not a replacement for a full benchmark replication.
- All secret values were intentionally omitted from this summary.

