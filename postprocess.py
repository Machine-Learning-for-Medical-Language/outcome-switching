import hashlib
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
import polars.selectors as cs

from preprocess import Preprocessor
from utils import DerivedFields, EvalResult, console, load_model_predictions


class Postprocessor:
    def __init__(
        self,
        trials_data_path: str | Path,
        model_eval_log: str | Path,
        cache_dir: str | Path | None,
        eval_results: dict[Literal["addition", "removal", "tf_change"], EvalResult],
        n_simulations: int = 1000,
        random_state=None,
    ):
        def prepare_df() -> pl.DataFrame:
            with console.status("Filtering trials..."):
                preprocessor = Preprocessor(trials_data_path)
                preprocessor.apply_inclusion_filters()

            model_predictions = load_model_predictions(
                model_eval_log, cache_dir=cache_dir
            )
            import typing

            latest = typing.cast(
                pl.DataFrame,
                preprocessor.latest_versions()
                .rename({"version": "version_after"})
                .join(
                    preprocessor.prospective_versions().select(
                        "nct_id", version_before="version"
                    ),
                    on="nct_id",
                ),
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

            infrequent_primary_intervention_types = (
                df.select(
                    pl.col("primary_intervention_type")
                    .value_counts(sort=True)
                    .struct.unnest()
                )
                .drop_nulls()
                .filter(pl.col("count") < (len(df) * 0.05))["primary_intervention_type"]
            )

            df = df.with_columns(
                primary_intervention_type=pl.when(
                    pl.col("primary_intervention_type").is_in(
                        infrequent_primary_intervention_types
                    )
                )
                .then(pl.lit("Other"))
                .otherwise("primary_intervention_type")
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
                relevant_therapeutic_areas=pl.when(pl.col("therapeutic_areas").eq([]))
                .then(pl.lit(["Unknown"]))
                .otherwise(
                    pl.col("therapeutic_areas").list.eval(
                        pl.element().filter(
                            (
                                pl.element().is_in(infrequent_therapeutic_areas)
                                | pl.element().is_in(uninformative_therapeutic_areas)
                            ).not_()
                        )
                    )
                )
            ).with_columns(
                relevant_therapeutic_areas=pl.when(
                    pl.col("relevant_therapeutic_areas").eq([])
                )
                .then(pl.lit(["Other"]))
                .otherwise(pl.col("relevant_therapeutic_areas"))
            )

            df = self._simulate_true_labels(
                df, eval_results, n_simulations, random_state
            )

            return df

        if cache_dir is not None:
            cache_file = (
                Path(cache_dir)
                / f"postprocess-{hashlib.sha256((str(trials_data_path) + str(model_eval_log)).encode()).hexdigest()}.parquet"
            )
            if cache_file.exists():
                self.df = cast(pl.DataFrame, pl.read_parquet(cache_file))
            else:
                df = prepare_df()
                df.write_parquet(cache_file)
                self.df = df
        else:
            self.df = prepare_df()

    def cohort_summary_table(self):
        table_1_data = self.df.select(
            "nct_id",
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
            )
            .cast(pl.String),
            primary_intervention_type=pl.col("primary_intervention_type")
            .str.replace_all("_", " ")
            .str.to_titlecase(),
            therapeutic_areas="relevant_therapeutic_areas",
            industry=pl.when(pl.col("lead_sponsor") == pl.lit("INDUSTRY"))
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No")),
            randomized=pl.when(
                pl.col("design_allocation").fill_null("NA") == pl.lit("RANDOMIZED")
            )
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No")),
            participants_ord=pl.col("enrollment_count")
            .cut(
                [20, 50, 100, 500],
                labels=[
                    "A. 0–19 participants",
                    "B. 20–49 participants",
                    "C. 50–99 participants",
                    "D. 100–499 participants",
                    "E. ≥500 participants",
                ],
                left_closed=True,
            )
            .cast(pl.String),
        )

        N = len(self.df)
        fmt_count_expr = pl.format(
            "{} ({})",
            pl.col("count").map_elements(lambda x: f"{x}"),
            (pl.col("count") / N).map_elements(lambda x: f"{x * 100:.2f}"),
        ).alias("count")

        def estimate_modifications(df: pl.DataFrame, col: str) -> pl.DataFrame:
            if col == "therapeutic_areas":
                t1 = table_1_data.explode("therapeutic_areas")
            else:
                t1 = table_1_data
            new_col = {"label": [], "modifications": []}
            for val in df["label"].unique():
                trial_ids = t1.filter(pl.col(col).eq(val)).select("nct_id")
                modifications = (
                    Postprocessor.estimate_change_prevalences(
                        self.df.join(trial_ids, on="nct_id")
                    )
                    .filter(pl.col("change_type").eq(pl.lit("any_change")))
                    .rows(named=True)[0]
                )
                new_col["label"].append(val)
                new_col["modifications"].append(modifications)
            return df.join(pl.DataFrame(new_col), on="label").with_columns(
                modifications=pl.format(
                    "{} ({}) [{}–{}]",
                    pl.col("modifications").struct.field("estimate").cast(pl.Int32),
                    # pl.col("modifications")
                    # .struct.field("ci")
                    # .struct.field("lower")
                    # .cast(pl.Int32),
                    # pl.col("modifications")
                    # .struct.field("ci")
                    # .struct.field("upper")
                    # .cast(pl.Int32),
                    pl.col("modifications").struct.field("pct").round(2),
                    pl.col("modifications")
                    .struct.field("pct_ci")
                    .struct.field("lower")
                    .round(2),
                    pl.col("modifications")
                    .struct.field("pct_ci")
                    .struct.field("upper")
                    .round(2),
                )
            )

        start_years = (
            table_1_data["start_year"]
            .value_counts()
            .select(
                label=pl.col("start_year"),
                count=fmt_count_expr,
            )
            .pipe(estimate_modifications, "start_year")
            .sort("label")
        )
        primary_intervention_types = (
            table_1_data["primary_intervention_type"]
            .value_counts()
            .select(
                label=pl.col("primary_intervention_type"),
                count=fmt_count_expr,
                count_num="count",
            )
            .pipe(estimate_modifications, "primary_intervention_type")
            .sort("count_num", descending=True)
            .drop("count_num")
        )
        therapeutic_areas = (
            table_1_data["therapeutic_areas"]
            .explode()
            .value_counts()
            .select(label="therapeutic_areas", count=fmt_count_expr, count_num="count")
            .pipe(estimate_modifications, "therapeutic_areas")
            .sort(
                pl.col("label") != "Unknown",
                pl.col("label") != "Other",
                pl.col("label"),
                descending=[True, True, False],
            )
            .drop("count_num")
        )
        industry = (
            table_1_data["industry"]
            .value_counts()
            .select(
                label="industry",
                count=fmt_count_expr,
            )
            .pipe(estimate_modifications, "industry")
            .sort("label", descending=True)
        )
        randomized = (
            table_1_data["randomized"]
            .value_counts()
            .select(
                label="randomized",
                count=fmt_count_expr,
            )
            .pipe(estimate_modifications, "randomized")
            .sort("label", descending=True)
        )
        participants = (
            table_1_data["participants_ord"]
            .value_counts()
            .select(
                label=pl.col("participants_ord"),
                count=fmt_count_expr,
            )
            .pipe(estimate_modifications, "participants_ord")
            .sort("label")
            .with_columns(pl.col("label").str.slice(3))
        )

        table_1 = pl.DataFrame(
            [
                {
                    "label": "",
                    "count": f"Total trials, N (%) (N={N})",
                    "modifications": "Trials with primary outcome modification, N (row %) [95% CI]",
                }
            ]
        )
        for header, subtable in (
            ("Start Year", start_years),
            ("Primary Intervention Type", primary_intervention_types),
            ("Any Industry Funding", industry),
            ("Randomized Study Design", randomized),
            ("Enrolled Participants", participants),
            ("Therapeutic Area", therapeutic_areas),
        ):
            table_1 = table_1.vstack(
                pl.DataFrame([{"label": header, "count": "", "modifications": ""}])
            )
            table_1 = table_1.vstack(subtable)

        return table_1

    @staticmethod
    def _simulate_true_labels(
        df: pl.DataFrame,
        eval_results: dict[Literal["addition", "removal", "tf_change"], EvalResult],
        n_simulations: int = 1000,
        random_state=None,
    ) -> pl.DataFrame:
        rng = np.random.default_rng(random_state)
        result = df

        mask = df["addition"].to_numpy() != None  # noqa: E711

        def simulate(category_title: str, p_gold_given_pred: np.ndarray) -> pl.Series:
            simulations = np.array(
                [
                    np.where(
                        mask,
                        rng.random(len(p_gold_given_pred)) < p_gold_given_pred,
                        False,
                    )
                    for _ in range(n_simulations)
                ]
            ).T  # transpose to (trials, simulations)
            return pl.Series(
                name=f"{category_title}_simulations",
                values=simulations,
                dtype=pl.Array(pl.Boolean, n_simulations),
            )

        for category, eval_result in eval_results.items():
            result = result.with_columns(
                simulate(
                    category,
                    np.where(
                        df[category].to_numpy().astype(bool),
                        eval_result.p_gold_given_pred(True, True),
                        eval_result.p_gold_given_pred(True, False),
                    ),
                )
            )

        result = result.with_columns(
            simulate(
                "any_change",
                df.select("addition", "removal", "tf_change")
                .map_rows(
                    lambda row: EvalResult.p_at_least_one_gold(
                        [
                            eval_results[cat]
                            for cat in ("addition", "removal", "tf_change")
                        ],
                        row,
                    )
                )["map"]
                .to_numpy(),
            )
        )

        return result

    @staticmethod
    def estimate_change_prevalences(filtered_df: pl.DataFrame) -> pl.DataFrame:
        def vertical_sum(series: pl.Series):
            arr_len = series.arr.len().item(0)
            return pl.concat_arr(series.arr.get(i).sum() for i in range(arr_len))

        return (
            filtered_df.select(cs.ends_with("_simulations"))
            .map_columns(cs.all(), vertical_sum)
            .head(1)
            .unpivot(variable_name="change_type", value_name="simulations")
            .with_columns(
                pl.col("change_type").str.strip_suffix("_simulations"),
                estimate=pl.col("simulations").arr.median(),
                ci=pl.struct(
                    lower=pl.col("simulations")
                    .arr.to_list()
                    .list.eval(pl.element().quantile(0.025))
                    .list.item(),
                    upper=pl.col("simulations")
                    .arr.to_list()
                    .list.eval(pl.element().quantile(0.975))
                    .list.item(),
                ),
            )
            .with_columns(
                pct=pl.col("estimate") * 100 / pl.lit(len(filtered_df)),
                pct_ci=pl.struct(
                    lower=pl.col("ci").struct.field("lower")
                    * 100
                    / pl.lit(len(filtered_df)),
                    upper=pl.col("ci").struct.field("upper")
                    * 100
                    / pl.lit(len(filtered_df)),
                ),
            )
        )

    def multivariate_analysis_table(
        self,
        eval_results: dict[Literal["addition", "removal", "tf_change"], EvalResult],
    ):
        def get_any_change_prob(struct):
            return EvalResult.p_at_least_one_gold(
                [eval_results[cat] for cat in ("addition", "removal", "tf_change")],
                [struct[cat] for cat in ("addition", "removal", "tf_change")],
            )

        return (
            self.df.drop(cs.ends_with("_simulations"))
            .with_columns(
                [
                    pl.when(pl.col(cat).is_null())
                    .then(1)
                    .otherwise(
                        pl.when(pl.col(cat))
                        .then(eval_results[cat].p_gold_given_pred(True, True))
                        .otherwise(eval_results[cat].p_gold_given_pred(False, False))
                    )
                    .alias(f"{cat}_weight")
                    for cat in ("addition", "removal", "tf_change")
                ]
            )
            .with_columns(
                any_change_prob=pl.when(pl.col("any_change").is_null())
                .then(0)
                .otherwise(
                    pl.struct("addition", "removal", "tf_change").map_elements(
                        get_any_change_prob,
                        skip_nulls=True,
                        return_dtype=pl.Float32,
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
