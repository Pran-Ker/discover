import json
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ttt_discover import (
    DiscoverConfig,
    Environment,
    SandboxRewardEvaluator,
    State,
    discover,
)
from ttt_discover.tinker_utils.dataset_builder import VerifyResult
from ttt_discover.tinker_utils.state import to_json_serializable

TRAIN_CSV = "/Users/pran-ker/Developer/Hexo/sia/tasks/lawbench/data/public/train.csv"
CLASSES_JSON = "/Users/pran-ker/Developer/Hexo/sia/tasks/lawbench/data/public/classes.json"

with open(CLASSES_JSON, encoding="utf-8") as _f:
    ALL_CLASSES: list[str] = list(json.load(_f))


# ── State ─────────────────────────────────────────────────────────────────────

class LawbenchState(State):
    accuracy: float

    def __init__(
        self,
        timestep: int,
        construction: list[Any],
        code: str,
        value: float = None,
        parent_values: list[float] = None,
        parents: list[dict] = None,
        id: str = None,
        observation: str = "",
        accuracy: float = None,
    ):
        super().__init__(timestep, construction, code, value, parent_values, parents, id, observation)
        self.accuracy = accuracy

    def to_dict(self) -> dict:
        return {
            "type": "LawbenchState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LawbenchState":
        return cls(
            timestep=d["timestep"],
            construction=d["construction"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
            accuracy=d.get("accuracy"),
        )


# ── Program builder ───────────────────────────────────────────────────────────

def _build_eval_program(
    fn_code: str,
    train_path: str,
    test_path: str,
    sub_path: str,
    gt_path: str,
    classes_path: str,
) -> str:
    header = """\
import pandas as pd
import numpy as np
import json
"""
    wrapper = f"""\

def run_lawbench():
    _train_path = {train_path!r}
    _test_path = {test_path!r}
    _sub_path = {sub_path!r}
    _gt_path = {gt_path!r}
    _classes_path = {classes_path!r}
    classify(_train_path, _test_path, _sub_path, _classes_path)
    submission = pd.read_csv(_sub_path)
    gt = pd.read_csv(_gt_path)
    merged = submission.merge(gt, on="id")
    correct = (merged["label_x"] == merged["label_y"]).sum()
    return float(correct) / len(merged)
"""
    return header + "\n\n" + fn_code + "\n" + wrapper


# ── Reward evaluator ──────────────────────────────────────────────────────────

class LawbenchRewardEvaluator(SandboxRewardEvaluator):

    def get_program_entrypoint(self) -> str:
        return "run_lawbench"

    def get_reward(self, code: str, state: LawbenchState) -> dict:
        extracted = self._extract_code(code)
        if extracted is None:
            return self._get_failure_entry("Cannot extract Python code from model response")

        _EVAL_SEEDS = [42, 137, 271]
        tmp_dir = tempfile.mkdtemp(prefix="lawbench_eval_")
        try:
            df = pd.read_csv(TRAIN_CSV)
            fold_scores = []

            for seed in _EVAL_SEEDS:
                train_df, val_df = train_test_split(
                    df, test_size=0.2, random_state=seed, stratify=df["label"]
                )

                train_path = os.path.join(tmp_dir, f"train_{seed}.csv")
                test_path = os.path.join(tmp_dir, f"test_{seed}.csv")
                gt_path = os.path.join(tmp_dir, f"val_gt_{seed}.csv")
                sub_path = os.path.join(tmp_dir, f"submission_{seed}.csv")

                train_df.to_csv(train_path, index=False)
                val_df[["id", "text"]].to_csv(test_path, index=False)
                val_df[["id", "label"]].to_csv(gt_path, index=False)

                full_program = _build_eval_program(
                    extracted, train_path, test_path, sub_path, gt_path, CLASSES_JSON
                )

                try:
                    output = self.run_eval_code(full_program)
                except Exception as e:
                    return self._get_failure_entry(f"Evaluation failed (seed={seed}): {e}")

                if not isinstance(output, (int, float)) or not (0.0 <= float(output) <= 1.0):
                    return self._get_failure_entry(
                        f"run_lawbench() must return accuracy in [0, 1], got: {output!r} (seed={seed})"
                    )

                fold_scores.append(float(output))

            accuracy = float(np.mean(fold_scores))
            return {
                "reward": accuracy,
                "msg": "",
                "correctness": 1.0,
                "raw_score": accuracy,
                "result_construction": [],
                "stdout": getattr(self, "_last_stdout", ""),
                "metrics": {"accuracy": accuracy, "fold_scores": fold_scores},
            }

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── System prompt ─────────────────────────────────────────────────────────────

_CLASSES_STR = json.dumps(ALL_CLASSES, ensure_ascii=False)

SYSTEM_PROMPT = f"""\
You are a machine learning engineer solving a Chinese legal text classification task.

## Task
LawBench: given a Chinese criminal case description (事实, ~400 chars), predict one of 191 criminal charge labels (罪名).

## Dataset
- ~5,332 training samples (columns: id, label, text)
- ~913 test samples (columns: id, text — no labels)
- 191 classes (Chinese criminal charges), imbalanced

## Valid labels (must predict only from this list)
{_CLASSES_STR}

## Your job
Write a Python function with this exact signature:

```python
def classify(train_path: str, test_path: str, submission_path: str, classes_path: str) -> None:
    ...
```

The function must:
1. Read training data from `train_path` (columns: id, label, text)
2. Read valid classes from `classes_path` (JSON list of 191 strings) — only predict labels from this list
3. Fit a classifier on the training samples
4. Read test data from `test_path` (columns: id, text)
5. Write `submission.csv` to `submission_path` with columns: id, label

## Available libraries
pandas, numpy, scikit-learn
Do NOT use external APIs or packages that require network access.

## Output format
Respond with a single ```python ... ``` block containing only the `classify` function and any helpers it needs.
"""


# ── Environment ───────────────────────────────────────────────────────────────

class LawbenchEnv(Environment):
    reward_function = LawbenchRewardEvaluator
    state_type = LawbenchState

    @classmethod
    def create_initial_state(cls, problem_type: str = "lawbench") -> LawbenchState:
        return LawbenchState(
            timestep=-1,
            construction=[],
            code="",
            value=None,
            accuracy=None,
        )

    def get_question(self) -> str:
        state = self.initial_state
        has_code = state.code and state.code.strip()

        prompt = SYSTEM_PROMPT

        if has_code:
            clean_code = state.code.strip()
            for marker in ("```python", "```"):
                if clean_code.startswith(marker):
                    clean_code = clean_code[len(marker):].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()

            accuracy_line = ""
            if state.accuracy is not None:
                accuracy_line = f"\nCurrent validation accuracy: {state.accuracy:.4f} (higher is better)"

            prompt += f"""
## Previous implementation
```python
{clean_code}
```
{accuracy_line}

Reason about how to improve this approach, then write an improved `classify` function.
"""
        else:
            prompt += "\nWrite your `classify` function."

        return prompt

    def is_maximize(self) -> bool:
        return True

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: VerifyResult,
    ) -> LawbenchState:
        accuracy = None
        if outs.metrics:
            accuracy = outs.metrics.get("accuracy")

        return self.state_type(
            timestep=step_idx,
            construction=outs.result_construction,
            code=parsed_code,
            value=outs.raw_score if self.is_maximize() else -outs.raw_score,
            observation=outs.stdout,
            accuracy=accuracy,
        )

    def _build_metrics(
        self,
        outs: VerifyResult,
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        correctness = outs.correctness
        return {
            "format": correct_format,
            "reward": outs.reward,
            "correctness": correctness,
            "raw_score": outs.raw_score if correctness > 0 else None,
            "initial_raw_score": self.initial_state.value,
            "msg": outs.msg,
            "prompt": self.get_question(),
            "response": message["content"],
            "parsed_code": parsed_code,
            "accuracy": outs.metrics.get("accuracy") if outs.metrics else None,
        }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = DiscoverConfig(
        env_type=LawbenchEnv,
        model_name="openai/gpt-oss-120b",
        lora_rank=32,
        group_size=32,
        groups_per_batch=4,
        learning_rate=4e-5,
        num_epochs=5,
        temperature=1.0,
        kl_penalty_coef=0.1,
        phase1_max_tokens=16000,
        problem_type="lawbench",
        experiment_name="lawbench-ttt",
        wandb_project="discover-ttt-lawbench",
        num_cpus_per_task=1,
        eval_timeout=180,
    )
    discover(config)
