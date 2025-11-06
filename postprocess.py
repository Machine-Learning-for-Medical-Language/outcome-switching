import os
from typing import Literal

import polars as pl

from preprocess import Preprocessor
from utils import console, load_model_predictions, simulate_gold_labels


class Postprocessor:
    def __init__(
        self,
        trials_data_path: str | os.PathLike,
        model_eval_log: str | os.PathLike,
        cache_dir: str | os.PathLike,
    ):
        with console.status("Filtering trials..."):
            preprocessor = Preprocessor(trials_data_path)
            preprocessor.apply_inclusion_filters()

        model_predictions = load_model_predictions(model_eval_log, cache_dir=cache_dir)

        latest = (
            preprocessor.latest_versions()
            .rename({"version": "version_after"})
            .join(
                preprocessor.prospective_versions().select(
                    "nct_id", version_before="version"
                ),
                on="nct_id",
            )
        )

        self.df = latest.join(model_predictions, on="nct_id", how="left").select(
            "nct_id",
            "data",
            "retrieved",
            "version_before",
            "version_after",
            "model_prediction",
        )

    def get_model_binary_labels(self):
        categories_list_field = (
            pl.col("model_prediction").struct.field("categories").list
        )
        return self.df.select(
            "nct_id",
            "version_before",
            "version_after",
            time_frame_change=categories_list_field.contains(
                pl.lit("time frame change")
            ),
            addition=categories_list_field.contains(pl.lit("addition")),
            removal=categories_list_field.contains(pl.lit("removal")),
        )

    def simulate_gold_labels(
        self,
        category: Literal["time frame change", "addition", "removal"],
        eval_tp: int,
        eval_tn: int,
        eval_fp: int,
        eval_fn: int,
        n_simulations: int = 1000,
    ):
        model_labels = self.df.filter(pl.col("model_prediction").is_not_null()).select(
            "nct_id",
            "version_before",
            "version_after",
            model_label=pl.col("model_prediction")
            .struct.field("categories")
            .list.contains(pl.lit(category)),
        )

        simulations = simulate_gold_labels(
            predictions=model_labels["model_label"],
            eval_tp=eval_tp,
            eval_tn=eval_tn,
            eval_fp=eval_fp,
            eval_fn=eval_fn,
            n_simulations=n_simulations,
        )

        simulations_df = model_labels.with_columns(
            gold_simulations=simulations.transpose()
        )
        return (
            self.df.select("nct_id", "version_before", "version_after")
            .join(
                simulations_df,
                on=["nct_id", "version_before", "version_after"],
                how="left",
            )
            .with_columns(pl.col("gold_simulations").fill_null([False] * n_simulations))
        )
