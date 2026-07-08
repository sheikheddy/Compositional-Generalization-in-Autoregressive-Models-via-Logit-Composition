from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import modal


APP_NAME = "logit-composition-partial-replication"
VOLUME_NAME = "logit-composition-replication"
DEFAULT_HF_SECRET_NAME = "HF_TOKEN"
DEFAULT_WANDB_SECRET_NAME = "WANDB_API_KEY"
REMOTE_SRC = "/root/src"
REMOTE_VOLUME = "/vol"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

if os.environ.get("HF_TOKEN"):
    hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})
else:
    hf_secret = modal.Secret.from_name(
        os.environ.get("MODAL_HF_SECRET_NAME", DEFAULT_HF_SECRET_NAME)
    )

if os.environ.get("WANDB_API_KEY"):
    wandb_secret = modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]})
else:
    wandb_secret = modal.Secret.from_name(
        os.environ.get("MODAL_WANDB_SECRET_NAME", DEFAULT_WANDB_SECRET_NAME)
    )

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch<3",
        "transformers>=4.46,<5",
        "accelerate>=1.0,<2",
        "datasets>=3,<5",
        "evalplus>=0.3,<1",
        "pandas>=2.2,<3",
        "tqdm>=4.66,<5",
        "sentencepiece>=0.2,<0.3",
        "protobuf>=4,<7",
        "hf-transfer>=0.1.9,<1",
        "wandb>=0.18,<1",
    )
    .env(
        {
            "PYTHONPATH": REMOTE_SRC,
            "HF_HOME": f"{REMOTE_VOLUME}/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src", REMOTE_SRC)
)


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={REMOTE_VOLUME: volume},
    timeout=6 * 60 * 60,
)
def run_partial_replication(config: dict) -> dict:
    import sys

    sys.path.insert(0, REMOTE_SRC)

    from logit_composition.replication import run_from_config

    config = dict(config)
    if not config.get("job_name"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        config["job_name"] = f"{config.get('target', 'merged')}-{stamp}"
    config["output_dir"] = f"{REMOTE_VOLUME}/results/{config['job_name']}"

    result = run_from_config(config)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    target: str = "merged",
    tasks: str = "gsm8k,math,humaneval,mbpp",
    limit: int = 3,
    gsm8k_limit: int = -1,
    math_limit: int = -1,
    code_limit: int = -1,
    max_new_tokens: int = 256,
    gpu: str = "L40S",
    dtype: str = "auto",
    run_evalplus: bool = False,
    include_synthetic: bool = False,
    synthetic_n: int = 500,
    synthetic_epochs: int = 3,
    synthetic_eval_n: int = 100,
    synthetic_max_eval_len: int = 31,
    diagnostic_max_seq_tokens: int = 128,
    use_wandb: bool = True,
    wandb_project: str = "logit-composition-replication",
    wandb_entity: str = "",
    job_name: str = "",
) -> None:
    config = {
        "target": target,
        "tasks": tasks,
        "limit": limit,
        "gsm8k_limit": None if gsm8k_limit < 0 else gsm8k_limit,
        "math_limit": None if math_limit < 0 else math_limit,
        "code_limit": None if code_limit < 0 else code_limit,
        "max_new_tokens": max_new_tokens,
        "dtype": dtype,
        "run_evalplus": run_evalplus,
        "include_synthetic": include_synthetic,
        "synthetic_n": synthetic_n,
        "synthetic_epochs": synthetic_epochs,
        "synthetic_eval_n": synthetic_eval_n,
        "synthetic_max_eval_len": synthetic_max_eval_len,
        "diagnostic_max_seq_tokens": diagnostic_max_seq_tokens,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "job_name": job_name,
    }
    result = run_partial_replication.with_options(gpu=gpu).remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
