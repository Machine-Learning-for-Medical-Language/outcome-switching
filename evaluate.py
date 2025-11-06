import json
from pathlib import Path
from typing import Final

import polars as pl
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, TaskState, generate, solver, system_message

from preprocess import Preprocessor

PROMPTS_DIR: Final = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_FILE: Final = PROMPTS_DIR / "system_prompt.txt"
USER_PROMPT_TEMPLATE_FILE: Final = PROMPTS_DIR / "user_template.txt"

SCHEMA_CATEGORIES: Final = (
    "time frame change",
    "addition",
    "removal",
    "other",
)


def outcome_edits_dataset(raw_data_file: str):
    with open(USER_PROMPT_TEMPLATE_FILE) as f:
        user_prompt_template = f.read()

    preprocessor = Preprocessor(raw_data_file)
    preprocessor.apply_filters_for_eval()
    dataset_df = (
        preprocessor.df.select(
            "nct_id",
            "version",
            "labels",
            pl.col("data")
            .struct.field("study")
            .struct.field("protocolSection")
            .struct.field("outcomesModule")
            .struct.field("primaryOutcomes"),
        )
        .explode("labels")
        .pivot("labels", index="nct_id")
        .select(
            "nct_id",
            version_before="version_prospective",
            version_after="version_latest",
            primary_outcomes_before="primaryOutcomes_prospective",
            primary_outcomes_after="primaryOutcomes_latest",
        )
    )

    samples: list[Sample] = []
    for row in dataset_df.rows(named=True):
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
        location=raw_data_file,
        samples=samples,
    )


@solver
def parse_model_response():
    """A solver to extract predictions and explanation from a model response."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_response = state.output.completion

        explanation: str = ""
        predicted_categories: list[str] = []

        for part in model_response.split("### "):
            if part.startswith("ANSWER"):
                answer = part.removeprefix("ANSWER").strip()
                predicted_categories = [
                    category
                    for category in SCHEMA_CATEGORIES
                    if category in [c.strip() for c in answer.lower().split(",")]
                ]
            if part.startswith("EXPLANATION"):
                explanation = part.removeprefix("EXPLANATION").strip()

        state.metadata["model_prediction"] = {
            "categories": predicted_categories,
            "explanation": explanation,
        }

        return state

    return solve


@task
def outcome_switching_task(raw_data_file: str):
    with open(SYSTEM_PROMPT_FILE) as f:
        system_prompt = f.read()

    return Task(
        dataset=outcome_edits_dataset(raw_data_file),
        solver=[
            # workaround for weird formatting behavior with JSON examples in prompt
            system_message("{system_prompt}", system_prompt=system_prompt),
            generate(),
            parse_model_response(),
        ],
        scorer=None,
    )
