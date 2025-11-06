import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from inspect_ai.log import read_eval_log, read_eval_log_samples
from rich.console import Console
from rich.progress import track

MESH_FILE: Final = Path(__file__).parent / "data" / "mesh" / "d2025.bin"

console = Console()


def str_to_date(expr: pl.Expr):
    """
    Some of the clinicaltrials.gov fields representing dates can be formatted as either
    `YYYY-MM-DD` or just `YYYY-MM`. This function returns an Expr that assumes the day `01`
    when it's missing, and converts to a polars Date column.
    """

    return (
        expr.str.pad_end(8, "-").str.pad_end(9, "0").str.pad_end(10, "1").str.to_date()
    )


def load_model_predictions(
    eval_log_file: str | os.PathLike,
    cache_dir: str | os.PathLike | None = None,
):
    def iter_samples():
        iterator = read_eval_log_samples(
            log_file=eval_log_file, all_samples_required=True
        )

        total_samples = read_eval_log(
            eval_log_file, header_only=True
        ).eval.dataset.samples

        if total_samples > 500:
            iterator = track(
                iterator,
                total=total_samples,
                description="Reading eval log samples...",
                console=console,
            )
        for sample in iterator:
            yield {
                "nct_id": sample.metadata["nct_id"],
                "version_before": sample.metadata["version_before"],
                "version_after": sample.metadata["version_after"],
                "model_prediction": sample.metadata["model_prediction"],
            }

    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{Path(eval_log_file).stem}.parquet"
        if cache_file.exists():
            return pl.read_parquet(cache_file)
        else:
            df = pl.DataFrame(iter_samples())
            df.write_parquet(cache_file)
            return df
    else:
        return pl.DataFrame(iter_samples())


def simulate_gold_labels(
    predictions: list[bool],
    eval_tp: int,
    eval_tn: int,
    eval_fp: int,
    eval_fn: int,
    n_simulations=1000,
    random_state=None,
):
    """
    Given a list of model predictions on target data and the model's performance
    statistics on an evaluation set, sample plausible gold labels for the target data.

    This function assumes that the evaluation data and the new data are drawn from the
    same distribution.

    Args:
        predictions: A binary list of the model's predictions on new data.
        eval_tp: The number of true positive labels by the model on the evaluation data.
        eval_tn: The number of true negative labels by the model on the evaluation data.
        eval_fp: The number of false positive labels by the model on the evaluation data.
        eval_fn: The number of false negative labels by the model on the evaluation data.
        n_simulations: How many lists of sampled gold labels to return. Defaults to 20000.
        random_state: Starting state for the random number generator. Defaults to None.

    Returns:
        A boolean ndarray with shape `(n_simulations, len(predictions))`
    """

    rng = np.random.default_rng(random_state)

    n_eval = eval_tp + eval_tn + eval_fp + eval_fn

    # There is uncertainty in these variables, so we could
    # instead sample different values for each iteration.
    model_tpr = eval_tp / (eval_tp + eval_fn)  # P(pred=1|gold=1)
    model_fnr = eval_fn / (eval_tp + eval_fn)  # P(pred=0|gold=1)
    p_gold_pos = (eval_tp + eval_fn) / n_eval  # P(gold=1)
    p_pred_pos = (eval_tp + eval_fp) / n_eval  # P(pred=1)
    p_pred_neg = (eval_tn + eval_fn) / n_eval  # P(pred=0)

    preds_arr = np.array(predictions, dtype=bool)

    simulations: list[list[bool]] = []
    for _ in range(n_simulations):
        p_gold_given_pred = np.where(
            preds_arr,
            # If model predicted True for this sample:
            # P(gold=1|pred=1) = P(pred=1|gold=1) * P(gold=1) / P(pred=1)
            model_tpr * p_gold_pos / p_pred_pos,
            # If model predicted False for this sample:
            # P(gold=1|pred=0) = P(pred=0|gold=1) * P(gold=1) / P(pred=0)
            model_fnr * p_gold_pos / p_pred_neg,
        )

        simulations.append(rng.random(len(preds_arr)) < p_gold_given_pred)

    return np.array(simulations)


def parse_mesh_hierarchy() -> pl.DataFrame:
    """Parses the MeSH hierarchy so we can map MeSH IDs to therapeutic areas.

    Returns:
        A dataframe of records with MeSH ID, term name, and hierarchy level (distance from root).
    """
    tree_map: defaultdict[str, list[str]] = defaultdict(list)
    name_map: dict[str, str] = {}

    with open(MESH_FILE) as f:
        content = f.read()

    name_pattern = re.compile(r"MH = (.+)")
    id_pattern = re.compile(r"UI = (D\d+)")
    tree_ids_pattern = re.compile(r"MN = ([A-Z0-9.]+)")

    for record in content.strip().split("*NEWRECORD"):
        if record.strip() == "":
            continue

        name_match = name_pattern.search(record)
        id_match = id_pattern.search(record)

        if name_match is None or id_match is None:
            raise ValueError(f"error reading {MESH_FILE!s}")

        name = name_match.group(1)
        mesh_id = id_match.group(1)
        tree_ids = tree_ids_pattern.findall(record)

        tree_map[mesh_id] = tree_ids
        for tree_id in tree_ids:
            name_map[tree_id] = name

    data = {"mesh_id": [], "term_name": [], "level": []}
    for mesh_id, tree_ids in tree_map.items():
        for tree_id in tree_ids:
            if tree_id.startswith("C"):  # conditions only
                for level in range(len(tree_id.split("."))):
                    data["mesh_id"].append(mesh_id)
                    data["term_name"].append(
                        name_map[".".join(tree_id.split(".")[: level + 1])]
                    )
                    data["level"].append(level)

    return pl.DataFrame(data).unique()
