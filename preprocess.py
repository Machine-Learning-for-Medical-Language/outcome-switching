import os
from contextlib import contextmanager
from dataclasses import dataclass

import polars as pl

from utils import DerivedFields


def _count_trials(df: pl.DataFrame) -> int:
    return len(df["nct_id"].unique())


@dataclass
class Filter:
    desc: str
    before: int
    after: int

    def __str__(self):
        return (
            f"{self.before} → {self.after} (-{self.before - self.after}): {self.desc}"
        )


class Preprocessor:
    def __init__(self, raw_data_file: str | os.PathLike):
        self.raw_df: pl.DataFrame = pl.read_parquet(raw_data_file)
        self.df: pl.DataFrame = self.raw_df
        self.filters: list[Filter] = []

    def reset(self):
        self.df = self.raw_df
        self.filters = []

    @contextmanager
    def _filtering(self, desc: str):
        before = _count_trials(self.df)
        yield
        after = _count_trials(self.df)
        self.filters.append(Filter(desc=desc, before=before, after=after))

    def summary(self):
        return "\n".join(
            [f"{_count_trials(self.raw_df)}: all downloaded trials"]
            + [f"  {f}" for f in self.filters]
        )

    def latest_versions(self):
        return self.df.filter(pl.col("labels").list.contains("latest"))

    def prospective_versions(self):
        return self.df.filter(pl.col("labels").list.contains("prospective"))

    def version_map(self):
        """
        Construct a DataFrame mapping trial IDs to the version numbers of the trial's
        prospective and latest versions.

        Returns:
            A DataFrame with "nct_id", "prospective", and "latest" columns, where the latter
            two are version numbers.
        """

        return (
            self.df.select("nct_id", "version", "labels")
            .explode("labels")
            .pivot("labels", index="nct_id")
        )

    def filter_prospective(self):
        """Remove all trials that were not prospectively registered."""

        prospective_trial_ids = (
            self.df.lazy()
            .filter(pl.col("labels").list.contains("prospective"))
            .select("nct_id")
        )

        with self._filtering("remove non-prospectively registered trials"):
            self.df = self.df.lazy().join(prospective_trial_ids, on="nct_id").collect()

    def filter_intervention_type(self):
        """
        Remove trials where the primary intervention type is "BEHAVIORAL",
        "DIETARY_SUPPLEMENT", or "RADIATION". Also remove trials without a primary intervention type.
        The primary intervention type is the first intervention type listed that is
        not "PROCEDURE" or "OTHER".

        The purpose of this filtering is to remove trials where the primary intervention is not a
        regulated product.
        """

        relevant_intervention_trial_ids = (
            self.df.lazy()
            .filter(pl.col("labels").list.contains("latest"))
            .select("nct_id", DerivedFields.primary_intervention_type)
            .filter(
                pl.col("primary_intervention_type")
                .is_in(["BEHAVIORAL", "DIETARY_SUPPLEMENT", "RADIATION", None])
                .not_()
            )
            .select("nct_id")
        )

        with self._filtering(
            "remove trials without a relevant primary intervention type"
        ):
            self.df = (
                self.df.lazy()
                .join(relevant_intervention_trial_ids, on="nct_id")
                .collect()
            )

    def filter_too_many_primary_outcomes(self):
        """
        Remove trials where either the prospective or the latest version
        has more than 5 primary outcomes.
        """

        trials_with_le_five_outcomes = (
            (
                self.df.lazy()
                .group_by("nct_id")
                .agg(
                    n_primary_outcomes=pl.col("data")
                    .struct.field("study")
                    .struct.field("protocolSection")
                    .struct.field("outcomesModule")
                    .struct.field("primaryOutcomes")
                    .list.len()
                    .max()
                )
            )
            .filter(pl.col("n_primary_outcomes").le(5))
            .select("nct_id")
        )

        with self._filtering("remove trials with more than 5 primary outcomes"):
            self.df = (
                self.df.lazy().join(trials_with_le_five_outcomes, on="nct_id").collect()
            )

    def filter_primary_outcome_edits(self):
        """
        Remove trials where the primary outcomes are the same between the prospective
        and latest versions of the trial.
        """
        trials_with_primary_outcome_edits = (
            self.df.lazy()
            .select(
                "nct_id",
                "labels",
                pl.col("data")
                .struct.field("study")
                .struct.field("protocolSection")
                .struct.field("outcomesModule")
                .struct.field("primaryOutcomes"),
            )
            .explode("labels")
            .group_by("nct_id")
            .n_unique()
            .filter(pl.col("primaryOutcomes").eq(2))
            .select("nct_id")
        )

        with self._filtering(
            "remove trials without textual edits to the primary outcomes"
        ):
            self.df = (
                self.df.lazy()
                .join(trials_with_primary_outcome_edits, on="nct_id")
                .collect()
            )

    def apply_inclusion_filters(self):
        self.filter_prospective()
        self.filter_too_many_primary_outcomes()
        self.filter_intervention_type()

    def apply_filters_for_eval(self):
        self.apply_inclusion_filters()
        self.filter_primary_outcome_edits()
