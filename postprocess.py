import hashlib
import os
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from preprocess import Preprocessor
from utils import DerivedFields, EvalResult, console, load_model_predictions


class Postprocessor:
    def __init__(
        self,
        trials_data_path: str | os.PathLike,
        model_eval_log: str | os.PathLike,
        cache_dir: str | os.PathLike,
    ):
        def prepare_df():
            with console.status("Filtering trials..."):
                preprocessor = Preprocessor(trials_data_path)
                preprocessor.apply_inclusion_filters()

            model_predictions = load_model_predictions(
                model_eval_log, cache_dir=cache_dir
            )

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

            df = (
                latest.join(model_predictions, on="nct_id", how="left")
                .select(
                    "nct_id",
                    "data",
                    "retrieved",
                    "version_before",
                    "version_after",
                    "model_prediction",
                )
                .with_columns(
                    DerivedFields.start_date,
                    DerivedFields.primary_intervention_type,
                    DerivedFields.therapeutic_areas,
                    DerivedFields.enrollment_count,
                    DerivedFields.lead_sponsor,
                    DerivedFields.design_allocation,
                    addition=pl.col("model_prediction")
                    .struct.field("categories")
                    .list.contains("addition"),
                    removal=pl.col("model_prediction")
                    .struct.field("categories")
                    .list.contains("removal"),
                    tf_change=pl.col("model_prediction")
                    .struct.field("categories")
                    .list.contains("time frame change"),
                )
                .with_columns(
                    any_change=pl.col("addition").or_(
                        pl.col("removal"), pl.col("tf_change")
                    ),
                )
            )

            infrequent_therapeutic_areas = (
                df.select(
                    pl.col("therapeutic_areas")
                    .explode()
                    .value_counts(sort=True)
                    .struct.unnest()
                )
                .drop_nulls()
                .filter(pl.col("count") < (len(df) * 0.05))["therapeutic_areas"]
            )

            uninformative_therapeutic_areas = (
                "Animal Diseases",
                "Chemically-Induced Disorders",
                "Pathological Conditions, Signs and Symptoms",
            )

            df = df.with_columns(
                relevant_therapeutic_areas=pl.col("therapeutic_areas").list.eval(
                    pl.element().filter(
                        (
                            pl.element().is_in(infrequent_therapeutic_areas)
                            | pl.element().is_in(uninformative_therapeutic_areas)
                        ).not_()
                    )
                )
            ).with_columns(
                relevant_therapeutic_areas=pl.when(
                    pl.col("relevant_therapeutic_areas").eq([])
                )
                .then(pl.lit(["Other"]))
                .otherwise(pl.col("relevant_therapeutic_areas"))
            )

            return df

        if cache_dir is not None:
            cache_file = (
                Path(cache_dir)
                / f"postprocess-{hashlib.sha256((str(trials_data_path) + str(model_eval_log)).encode()).hexdigest()}.parquet"
            )
            if cache_file.exists():
                self.df = pl.read_parquet(cache_file)
            else:
                df = prepare_df()
                df.write_parquet(cache_file)
                self.df = df
        else:
            self.df = prepare_df()

    def cohort_summary_table(self):
        table_1_data = self.df.select(
            start_year=pl.col("start_date")
            .dt.year()
            .cut(
                range(2016, 2024, 2),
                labels=[
                    "2015–2016",
                    "2017–2018",
                    "2019–2020",
                    "2021–2022",
                    "2023–2024",
                ],
            ),
            primary_intervention_type="primary_intervention_type",
            therapeutic_areas="relevant_therapeutic_areas",
            industry=pl.when(pl.col("lead_sponsor") == pl.lit("INDUSTRY"))
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No")),
            randomized=pl.when(
                pl.col("design_allocation").fill_null("NA") == pl.lit("RANDOMIZED")
            )
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No")),
            participants_ord=pl.col("enrollment_count").cut(
                [20, 50, 100, 500],
                labels=[
                    "A. 0–19 participants",
                    "B. 20–49 participants",
                    "C. 50–99 participants",
                    "D. 100–499 participants",
                    "E. ≥500 participants",
                ],
                left_closed=True,
            ),
        )

        N = len(self.df)
        fmt_count_expr = pl.format(
            f"{{}}/{N:,} ({{}}%)",
            pl.col("count").map_elements(lambda x: f"{x:,}"),
            (pl.col("count") * 100 / N).round(2),
        ).alias("count")

        start_years = (
            table_1_data["start_year"]
            .value_counts()
            .select(label=pl.col("start_year").cast(pl.String), count=fmt_count_expr)
            .sort("label")
        )
        primary_intervention_types = (
            table_1_data["primary_intervention_type"]
            .value_counts()
            .sort("count", descending=True)
            .select(
                label=pl.col("primary_intervention_type")
                .str.replace_all("_", " ")
                .str.to_titlecase(),
                count=fmt_count_expr,
            )
        )
        therapeutic_areas = (
            table_1_data["therapeutic_areas"]
            .explode()
            .value_counts()
            .sort(
                pl.col("therapeutic_areas") != "Other", pl.col("count"), descending=True
            )
            .select(label="therapeutic_areas", count=fmt_count_expr)
        )
        industry = (
            table_1_data["industry"]
            .value_counts()
            .select(label="industry", count=fmt_count_expr)
            .sort("label", descending=True)
        )
        randomized = (
            table_1_data["randomized"]
            .value_counts()
            .select(label="randomized", count=fmt_count_expr)
            .sort("label", descending=True)
        )
        participants = (
            table_1_data["participants_ord"]
            .value_counts()
            .sort("participants_ord")
            .select(
                label=pl.col("participants_ord").cat.slice(3).cast(pl.String),
                count=fmt_count_expr,
            )
        )

        table_1 = pl.DataFrame([{"label": "", "count": "Total Trials"}])
        for header, subtable in (
            ("Start Year", start_years),
            ("Primary Intervention Type", primary_intervention_types),
            ("Therapeutic Area", therapeutic_areas),
            ("Any Industry Funding", industry),
            ("Randomized Study Design", randomized),
            ("Enrolled Participants", participants),
        ):
            table_1 = table_1.vstack(pl.DataFrame([{"label": header, "count": ""}]))
            table_1 = table_1.vstack(subtable)

        return table_1

    def estimate_prevalences(
        self,
        eval_results: dict[Literal["addition", "removal", "tf_change"], EvalResult],
        n_simulations: int = 1000,
        random_state=None,
    ):
        rng = np.random.default_rng(random_state)
        result = []

        mask = self.df["addition"].to_numpy() != None  # noqa: E711

        def compute_estimate(category_title: str, p_gold_given_pred: np.ndarray):
            simulations = np.array(
                [
                    np.where(
                        mask,
                        rng.random(len(p_gold_given_pred)) < p_gold_given_pred,
                        False,
                    )
                    for _ in range(n_simulations)
                ]
            )
            prevalences = simulations.sum(axis=1)
            result.append(
                {
                    "category": category_title,
                    "estimate": np.median(prevalences),
                    "ci": np.percentile(prevalences, (2.5, 97.5)),
                }
            )

        for category, eval_result in eval_results.items():
            compute_estimate(
                category,
                np.where(
                    self.df[category].to_numpy().astype(bool),
                    eval_result.p_gold_given_pred(True, True),
                    eval_result.p_gold_given_pred(True, False),
                ),
            )

        compute_estimate(
            "any_change",
            self.df.select("addition", "removal", "tf_change")
            .map_rows(
                lambda row: EvalResult.p_at_least_one_gold(
                    [eval_results[cat] for cat in ["addition", "removal", "tf_change"]],
                    row,
                )
            )["map"]
            .to_numpy(),
        )

        return pl.DataFrame(result).with_columns(
            disp=pl.format(
                "{}% [{}%—{}%]",
                (pl.col("estimate") * 100 / len(self.df)).round(2),
                (pl.col("ci").list[0] * 100 / len(self.df)).round(2),
                (pl.col("ci").list[1] * 100 / len(self.df)).round(2),
            )
        )

    def multivariate_analysis_table(
        self,
        eval_results: dict[Literal["addition", "removal", "tf_change"], EvalResult],
    ):
        def get_any_change_prob(struct):
            return EvalResult.p_at_least_one_gold(
                [eval_results[cat] for cat in ["addition", "removal", "tf_change"]],
                [struct[cat] for cat in ["addition", "removal", "tf_change"]],
            )

        return (
            self.df.with_columns(
                [
                    pl.when(pl.col(cat).is_null())
                    .then(1)
                    .otherwise(
                        pl.when(pl.col(cat))
                        .then(eval_results[cat].p_gold_given_pred(True, True))
                        .otherwise(eval_results[cat].p_gold_given_pred(False, False))
                    )
                    .alias(f"{cat}_weight")
                    for cat in ["addition", "removal", "tf_change"]
                ]
            )
            .with_columns(
                any_change_prob=pl.when(pl.col("any_change").is_null())
                .then(0)
                .otherwise(
                    pl.struct("addition", "removal", "tf_change").map_elements(
                        get_any_change_prob,
                        skip_nulls=True,
                        return_dtype=float,
                    )
                ),
            )
            .with_columns(
                any_change_weight=pl.when(pl.col("any_change"))
                .then(pl.col("any_change_prob"))
                .otherwise(pl.lit(1).sub(pl.col("any_change_prob")))
            )
            .drop("any_change_prob")
            .with_columns(
                pl.col("addition").fill_null(False),
                pl.col("removal").fill_null(False),
                pl.col("tf_change").fill_null(False),
                pl.col("any_change").fill_null(False),
                start_year=DerivedFields.start_date.dt.year(),
                industry=DerivedFields.lead_sponsor.eq("INDUSTRY"),
                randomized=DerivedFields.design_allocation.fill_null("NA").eq(
                    "RANDOMIZED"
                ),
                therapeutic_areas="relevant_therapeutic_areas",
            )
            .drop(
                "data",
                "retrieved",
                "model_prediction",
                "start_date",
                "lead_sponsor",
                "design_allocation",
                "relevant_therapeutic_areas",
            )
        )
