import json
import re
from pathlib import Path
from typing import Final

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import (
    Generate,
    TaskState,
    generate,
    solver,
    system_message,
    user_message,
)

from preprocess import find_outcome_edits, preprocess_raw_data

PROMPTS_DIR: Final = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_FILE: Final = PROMPTS_DIR / "system_prompt.txt"
FINAL_ANSWER_PROMPT_FILE: Final = PROMPTS_DIR / "final_answer_prompt.txt"
USER_PROMPT_TEMPLATE_FILE: Final = PROMPTS_DIR / "user_template.txt"

SCHEMA_CATEGORIES: Final = (
    "rewording/rephrasing",
    "elaboration",
    "modification",
    "reordering",
    "addition",
    "removal",
)


def outcome_edits_dataset(raw_data_dir: str, raw_labels_file: str | None = None):
    """Create a dataset of primary outcome edits for model evaluation.

    Args:
        raw_data_dir: Path to the raw trial data.
        raw_labels_file: *Not yet implemented* A file containing true labels for the data. Defaults to None.

    Returns:
        The dataset for evaluation.
    """

    if raw_labels_file is not None:
        raise NotImplementedError("evals with labeled data not yet implemented")

    with open(USER_PROMPT_TEMPLATE_FILE) as f:
        user_prompt_template = f.read()

    latest, summaries, originals = preprocess_raw_data(raw_data_dir)
    outcome_edits = find_outcome_edits(latest, summaries, originals)
    rows = outcome_edits.rows(named=True)

    samples: list[Sample] = []
    for row in rows:
        samples.append(
            Sample(
                input=user_prompt_template.format(
                    primary_outcomes_before=json.dumps(
                        row["primary_outcomes_before"], indent=2
                    ),
                    primary_outcomes_after=json.dumps(
                        row["primary_outcomes_after"], indent=2
                    ),
                ),
                metadata=row,
            )
        )

    return MemoryDataset(
        location=raw_data_dir,
        samples=samples,
    )


@solver
def parse_model_response():
    # pattern to match a comma-separated array of integers enclosed in square brackets
    int_arr_pattern = re.compile(r"\[[0-9]+(,\ ?[0-9]+)*\]")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        response = state.messages[-1].text
        m = int_arr_pattern.search(response)
        result: list[int] = sorted(
            set(
                []
                if m is None
                else [int(x.strip()) for x in m.group(0).strip("[]").split(",")]
            )
        )
        state.metadata["model_prediction"] = {
            "categories": [
                SCHEMA_CATEGORIES[i - 1]
                for i in result
                if 0 < i <= len(SCHEMA_CATEGORIES)
            ],
            "explanation": next(
                m for m in state.messages if m.role == "assistant"
            ).text,
        }
        return state

    return solve


@task
def outcome_switching_task(raw_data_dir: str, raw_labels_file: str | None = None):
    with open(SYSTEM_PROMPT_FILE) as f:
        system_prompt = f.read()

    with open(FINAL_ANSWER_PROMPT_FILE) as f:
        final_answer_prompt = f.read()

    return Task(
        dataset=outcome_edits_dataset(raw_data_dir, raw_labels_file),
        solver=[
            system_message("{system_prompt}", system_prompt=system_prompt),
            generate(),
            user_message("{fa_prompt}", fa_prompt=final_answer_prompt),
            generate(),
            parse_model_response(),
        ],
        scorer=None,
    )
