from datetime import date

import polars as pl

from preprocess import find_outcome_edits, load_model_predictions, preprocess_raw_data


class Analysis:
    def __init__(self, raw_data_dir: str, model_predictions_file: str | None = None):
        self.latest, self.summaries, self.originals = preprocess_raw_data(raw_data_dir)
        self.edits = find_outcome_edits(self.latest, self.summaries, self.originals)
        self.predictions = None
        if model_predictions_file is not None:
            self.predictions = load_model_predictions(model_predictions_file)
            self.latest = self.latest.join(
                self.predictions.select(
                    "nct_id", "model_explanation", "model_predictions"
                ),
                on="nct_id",
                how="left",
            )

    def _get_stats(self, *dfs: pl.DataFrame):
        stats = {
            "total_trials": [df.shape[0] for df in dfs],
            "with_textual_edits": [
                df.join(self.edits, on="nct_id").shape[0] for df in dfs
            ],
        }
        if self.predictions is not None:
            stats["with_significant_changes"] = [
                df.select(
                    pl.any_horizontal(
                        pl.col("model_predictions").struct.field(
                            "modification", "addition", "removal"
                        )
                    )
                )
                .sum()
                .item()
                for df in dfs
            ]
        return pl.DataFrame(stats)

    def overall(self):
        return self._get_stats(self.latest)

    def by_start_year(self):
        years = [date(y, 1, 1) for y in [2008, 2011, 2014, 2017, 2020, 2023]]
        stats = self._get_stats(
            *[
                self.latest.filter(
                    pl.col("start_date") >= start,
                    pl.col("start_date") < end,
                )
                for start, end in zip(years, years[1:])
            ]
        )
        return stats.insert_column(
            0,
            pl.Series(
                "start_year",
                [
                    f"{start.year}–{end.year - 1}"
                    for start, end in zip(years, years[1:])
                ],
            ),
        )

    def by_therapeutic_area(self):
        therapeutic_areas_sorted = (
            self.latest["therapeutic_areas"]
            .list.explode()
            .value_counts(sort=True)["therapeutic_areas"]
            .to_list()
        )
        # move "Other" to the end
        if "Other" in therapeutic_areas_sorted:
            therapeutic_areas_sorted.remove("Other")
            therapeutic_areas_sorted.append("Other")
        stats = self._get_stats(
            *[
                self.latest.filter(pl.col("therapeutic_areas").list.contains(pl.lit(t)))
                for t in therapeutic_areas_sorted
            ]
        )
        return stats.insert_column(
            0, pl.Series("therapeutic_area", therapeutic_areas_sorted)
        )

    def by_industry_funding(self):
        ind_expr = pl.col("lead_sponsor").eq("INDUSTRY")
        stats = self._get_stats(
            self.latest.filter(ind_expr), self.latest.filter(ind_expr.not_())
        )
        return stats.insert_column(0, pl.Series("any_industry_funding", ["yes", "no"]))

    def by_randomization(self):
        rand_expr = pl.col("design_allocation").eq("RANDOMIZED")
        stats = self._get_stats(
            self.latest.filter(rand_expr), self.latest.filter(rand_expr.not_())
        )
        return stats.insert_column(
            0, pl.Series("randomized_study_design", ["yes", "no"])
        )

    def by_enrollment(self):
        enrollment_cutoffs = [0, 20, 50, 100, 500] + [
            self.latest["enrollment_count"].max() + 1,
        ]
        stats = self._get_stats(
            *[
                self.latest.filter(
                    pl.col("enrollment_count") >= lo, pl.col("enrollment_count") < hi
                )
                for lo, hi in zip(enrollment_cutoffs, enrollment_cutoffs[1:])
            ]
        )
        return stats.insert_column(
            0,
            pl.Series(
                "trial_enrollment",
                [
                    f"{lo}–{hi - 1} participants"
                    for lo, hi in zip(enrollment_cutoffs, enrollment_cutoffs[1:])
                ],
            ),
        )
