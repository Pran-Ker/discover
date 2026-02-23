import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from tinker_cookbook.utils import logtree

from tasks.mle_bench.prompt import create_prompt
from tasks.mle_bench.task import MleBenchTask
from tinker_cookbook.recipes.ttt.state import MleBenchState
from tinker_cookbook.recipes.ttt.env_ttt import BaseTTTEnv
from tinker_cookbook.recipes.ttt.dataset_builder import DatasetConfig


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 2,
        "rew_type": "linear",
        "fail_score": 0.0,
        "eval_timeout": 300,
        "worst_perf_log": -10000,
        "n_item": 200,
    }
}


def verify_mle_bench(
    generation: str,
    step: int,
    num_cpus_per_task: int = 2,
    eval_timeout: int = 300,
    log_path: str = "",
    competition_id: str = "spaceship-titanic",
    **kwargs,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"MLE-bench grading: competition={competition_id}")

    task = MleBenchTask(config_ns, log_path, competition_id=competition_id)
    out = task.compute_score(generation, step=step)

    # Extract grading details from the stored report (set by get_reward)
    report = task._last_report
    raw_score = None
    medal = "none"
    is_lower_better = False

    if report is not None and report.valid_submission and report.score is not None:
        raw_score = report.score
        is_lower_better = report.is_lower_better
        if report.gold_medal:
            medal = "gold"
        elif report.silver_medal:
            medal = "silver"
        elif report.bronze_medal:
            medal = "bronze"
        elif report.above_median:
            medal = "above_median"
        else:
            medal = "below_median"
    elif out["correctness"] > 0 and report is not None and not report.valid_submission:
        # verify() passed (non-empty DataFrame) but grading found it invalid
        out["correctness"] = 0.0
        out["score"] = 0.0

    # Log full verifier results
    if report is not None:
        logtree.log_text(
            f"MLE-bench result: raw_score={raw_score}, medal={medal}, "
            f"reward={out['score']}, correctness={out['correctness']}, "
            f"is_lower_better={is_lower_better}, valid={report.valid_submission}, "
            f"thresholds(gold={report.gold_threshold}, silver={report.silver_threshold}, "
            f"bronze={report.bronze_threshold}, median={report.median_threshold})"
        )
    else:
        logtree.log_text(
            f"MLE-bench result: no report (code failed), "
            f"msg={out['msg']}, correctness={out['correctness']}"
        )

    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": out["performance"],  # from _transform_reward, like CP/denoising
        "raw_score": raw_score,
        "medal": medal,
        "is_lower_better": is_lower_better,
        "stdout": out.get("stdout", ""),
    }


class MleBenchEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # problem_idx is the competition_id (e.g. "spaceship-titanic")
        self.competition_id = self.problem_idx

    def _get_improvement_prompt(self, state: MleBenchState) -> str:
        """Build contextual improvement prompt with value context."""
        prompt = create_prompt(
            self.competition_id,
            budget_s=self.budget_s or 300,
            num_cpus=self.num_cpus_per_task,
        )

        if state.code and state.code.strip():
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()

            value_ctx = ""
            if state.raw_score is not None:
                value_ctx = f"\nPrevious score: {state.raw_score}"
                value_ctx += f"\nMedal achieved: {state.medal}"
            if state.parent_values and state.value is not None:
                if state.is_lower_better:
                    value_ctx += f"\nScore before/after (lower is better): {-state.parent_values[0]:.4f} -> {-state.value:.4f}"
                else:
                    value_ctx += f"\nScore before/after: {state.parent_values[0]:.4f} -> {state.value:.4f}"

            if state.observation and state.observation.strip():
                stdout = state.observation.strip()
                if len(stdout) > 500:
                    stdout = "...(truncated)\n" + stdout[-500:]
                value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

            code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively improving your solution.{value_ctx}

Reason about how you could improve the predictions. Consider:
- Better feature engineering
- Different model architectures
- Hyperparameter tuning
- Ensemble methods
- Better data preprocessing
Unless you make a meaningful improvement, you will not be rewarded.
"""
        else:
            code_section = """
This is your first attempt. Write code to produce the best predictions you can.
Start by examining the data to understand the problem, then build a model.
"""

        return f"{prompt}\n{code_section}"

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int = 2,
        eval_timeout: int = 300,
        log_path: str = "",
        competition_id: str = "spaceship-titanic",
        **kwargs,
    ) -> dict[str, Any]:
        return verify_mle_bench(
            generation, step, num_cpus_per_task, eval_timeout, log_path, competition_id
        )

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "competition_id": self.competition_id,
        }

    def _get_timeout_response(self) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": None,
            "medal": "none",
            "is_lower_better": False,
            "stdout": "",
        }

    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
            "raw_score": None,
            "medal": "none",
            "is_lower_better": False,
            "stdout": "",
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        if correctness > 0:
            return outs.get("score", 0.0)
        return 0.0

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> MleBenchState:
        performance = outs.get("performance")
        if performance is None:
            return None
        parent_state = self.initial_state
        parent_values = (
            [parent_state.value] + parent_state.parent_values
            if parent_state.value is not None
            else []
        )
        return MleBenchState(
            timestep=step_idx,
            code=parsed_code,
            value=performance,
            raw_score=outs.get("raw_score"),
            is_lower_better=outs.get("is_lower_better", False),
            medal=outs.get("medal", "none"),
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        return {
            "format": correct_format,
            "score": outs.get("score", 0.0),
            "correctness": outs.get("correctness", 0.0),
            "correct": outs.get("correctness", 0.0),
            "performance": outs.get("performance", 0.0),
            "raw_score": outs.get("raw_score"),
            "medal": outs.get("medal", "none"),
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message["content"],
            "ref": outs.get("msg", ""),
        }
