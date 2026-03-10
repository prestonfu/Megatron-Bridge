import argparse
import glob
import logging
import os
import time

logging.getLogger("nemo_evaluator").setLevel(logging.ERROR)

import yaml
from tqdm import tqdm

from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EvaluationConfig,
    EvaluationTarget,
)
from nemo_evaluator.api import check_endpoint, evaluate

import wandb

TASKS = [
    "adlr_mmlu_pro_5_shot_base",
    "adlr_mmlu",
    "adlr_agieval_en_cot",
    "adlr_humaneval_greedy",
    "adlr_mbpp_sanitized_3_shot_greedy",
    "adlr_gsm8k_cot_8_shot",
    "adlr_minerva_math_nemo_4_shot",
    "adlr_math_500_4_shot_sampled",
    "adlr_arc_challenge_llama_25_shot",
    "hellaswag",
    "openbookqa",
    "piqa",
    "adlr_race",
    "adlr_winogrande_5_shot",
    "adlr_global_mmlu_lite_5_shot",
    "adlr_mgsm_native_cot_8_shot",
]

ENDPOINT_URL = "http://0.0.0.0:8000/v1/completions/"
MODEL_ID = "megatron_model"


def count_result_files(task_output_dir):
    """Count completed sample result files as a proxy for progress within a task."""
    patterns = [
        os.path.join(task_output_dir, "**", "*.json"),
        os.path.join(task_output_dir, "**", "*.jsonl"),
    ]
    count = 0
    for pattern in patterns:
        count += len(glob.glob(pattern, recursive=True))
    return count


def log_task_to_wandb(wandb_run, task, task_output_dir):
    """Log results from a task's results.yml to wandb if available."""
    results_path = os.path.join(task_output_dir, "results", "results.yml")
    if not os.path.exists(results_path):
        # Try finding any results.yml under the task dir
        found = glob.glob(os.path.join(task_output_dir, "**", "results.yml"), recursive=True)
        if not found:
            return
        results_path = found[0]

    with open(results_path) as f:
        results = yaml.safe_load(f)

    artifact = wandb.Artifact(name=f"eval_{task}", type="evaluation_results")
    artifact.add_file(local_path=results_path, name="results.yml")
    wandb_run.log_artifact(artifact)

    if not isinstance(results, dict) or "results" not in results:
        return

    log_dict = {}
    for category in ["tasks", "groups"]:
        category_results = results["results"].get(category, {})
        for name, result in category_results.items():
            for metric_name, metric_result in result.get("metrics", {}).items():
                scores = metric_result.get("scores", {}).get(metric_name, {})
                if "value" in scores:
                    log_dict[f"{task}/{category.rstrip('s')}/{name}/{metric_name}/value"] = scores["value"]
                if "stats" in scores and "stderr" in scores["stats"]:
                    log_dict[f"{task}/{category.rstrip('s')}/{name}/{metric_name}/stderr"] = scores["stats"]["stderr"]

    if log_dict:
        wandb_run.log(log_dict)


def main():
    parser = argparse.ArgumentParser(description="Run eval suite against a locally deployed Megatron model")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--parallelism", type=int, default=4,
                        help="Number of concurrent requests to the model server")
    parser.add_argument("--request_timeout", type=int, default=10 * 60 * 60)
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="Cap samples per task; omit for full eval")
    parser.add_argument("--wandb_project", type=str, default="megatron-eval")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    server_ready = check_endpoint(
        endpoint_url=ENDPOINT_URL,
        endpoint_type="completions",
        model_name=MODEL_ID,
    )
    if not server_ready:
        raise RuntimeError("Server is not ready. Check deploy logs.")

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        resume="allow",
    )

    api_endpoint = ApiEndpoint(url=ENDPOINT_URL, type="completions", model_id=MODEL_ID)
    target_cfg = EvaluationTarget(api_endpoint=api_endpoint)

    task_bar = tqdm(TASKS, desc="Tasks", unit="task")
    results = {}

    for task in task_bar:
        task_bar.set_description(f"Task: {task}")
        task_output_dir = os.path.join(args.output_dir, task)

        eval_params = ConfigParams(
            parallelism=args.parallelism,
            request_timeout=args.request_timeout,
            limit_samples=args.limit_samples,
        )
        eval_cfg = EvaluationConfig(
            type=task,
            params=eval_params,
            output_dir=task_output_dir,
        )

        # Track intra-task progress by polling result files in a background thread
        import threading
        stop_poll = threading.Event()
        sample_bar = tqdm(desc=f"  {task} samples", unit="sample", leave=False, position=1)

        def poll_progress():
            prev = 0
            while not stop_poll.is_set():
                current = count_result_files(task_output_dir)
                if current > prev:
                    sample_bar.update(current - prev)
                    prev = current
                time.sleep(2)

        poll_thread = threading.Thread(target=poll_progress, daemon=True)
        poll_thread.start()

        try:
            result = evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)
            results[task] = result
            task_bar.write(f"✓ {task}: {result}")
            if wandb_run:
                log_task_to_wandb(wandb_run, task, task_output_dir)
        except Exception as e:
            results[task] = {"error": str(e)}
            task_bar.write(f"✗ {task} FAILED: {e}")
        finally:
            stop_poll.set()
            poll_thread.join(timeout=3)
            sample_bar.close()

    task_bar.write(f"\n{'='*60}\nAll tasks completed\n{'='*60}")
    for task, result in results.items():
        task_bar.write(f"  {task}: {result}")

    if wandb_run:
        wandb_run.finish()



if __name__ == "__main__":
    main()
