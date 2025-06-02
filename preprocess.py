"""
Preprocess raw data from clinicaltrials.gov and Inspect log files.
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

import polars as pl
from inspect_ai.log import read_eval_log_samples

MESH_FILE: Final = Path(__file__).parent / "d2025.bin"

# Utils


def str_col_to_date(col_name):
    """Some of the clinicaltrials.gov fields representing dates can be formatted as either
    `YYYY-MM-DD` or just `YYYY-MM`. This function returns an Expr that assumes the day `01`
    when it's missing, and converts to a polars Date column.

    Args:
        col_name: The name of the column to convert.

    Returns:
        An expression converting the String column to a Date column.
    """

    return (
        pl.col(col_name)
        .str.pad_end(8, "-")
        .str.pad_end(9, "0")
        .str.pad_end(10, "1")
        .str.to_date()
    )


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


# -- Preprocessing raw data -- #


def preprocess_latest_versions(raw: pl.DataFrame):
    """Preprocess raw trial data from clinicaltrials.gov of the latest versions of trials
    for analysis.


    Args:
        raw: The raw data to preprocess. The raw data should be a dataframe generated using polars' normalize_json()
        on a list of JSON records downloaded directly from clinicaltrials.gov.

    Raises:
        ValueError: If any trial records don't have an actual start date.

    Returns:
        The preprocessed dataframe of trials.
    """
    no_actual_start_date = raw.filter(
        pl.col("protocolSection.statusModule.startDateStruct.type")
        .is_null()
        .or_(pl.col("protocolSection.statusModule.startDateStruct.type").ne("ACTUAL"))
    )

    if no_actual_start_date.shape[0] > 0:
        raise ValueError(
            f"{no_actual_start_date.shape[0]} trials don't have an actual start date!"
        )

    raw_lazy = raw.lazy()

    # Get relevant columns from raw data
    df = raw_lazy.select(
        nct_id="protocolSection.identificationModule.nctId",
        primary_outcomes="protocolSection.outcomesModule.primaryOutcomes",
        start_date=str_col_to_date("protocolSection.statusModule.startDateStruct.date"),
        first_submit_date=str_col_to_date(
            "protocolSection.statusModule.studyFirstSubmitDate"
        ),
        last_update_submit_date=str_col_to_date(
            "protocolSection.statusModule.lastUpdateSubmitDate"
        ),
        phases="protocolSection.designModule.phases",
        lead_sponsor="protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
        conditions="protocolSection.conditionsModule.conditions",
        condition_meshes="derivedSection.conditionBrowseModule.meshes",
        design_allocation="protocolSection.designModule.designInfo.allocation",
        enrollment_count="protocolSection.designModule.enrollmentInfo.count",
        enrollment_type="protocolSection.designModule.enrollmentInfo.type",
        status="protocolSection.statusModule.overallStatus",
        is_fda_regulated_drug="protocolSection.oversightModule.isFdaRegulatedDrug",
        is_fda_regulated_device="protocolSection.oversightModule.isFdaRegulatedDevice",
    )

    # Derived columns
    df = df.with_columns(
        submit_delay=pl.col("first_submit_date").sub(pl.col("start_date")),
        submitted_late=pl.col("first_submit_date")
        .sub(pl.col("start_date"))
        .gt(pl.duration(days=21)),
    )

    # Remove trials registered late
    df = df.filter(pl.col("submitted_late").eq(False))
    df = df.drop("submitted_late")

    # Get per-trial intervention types
    interventions = (
        raw_lazy.select(
            nct_id="protocolSection.identificationModule.nctId",
            interventions="protocolSection.armsInterventionsModule.interventions",
        )
        .explode("interventions")
        .select("nct_id", pl.col("interventions").struct.field("type"))
        .group_by("nct_id")
        .agg(intervention_types=pl.col("type").unique().sort())
    )
    df = df.join(interventions, on="nct_id")

    # Ignore PROCEDURE and OTHER intervention types
    df = df.with_columns(
        intervention_types_filtered=pl.col("intervention_types").list.eval(
            pl.element().filter(pl.element().is_in(["PROCEDURE", "OTHER"]).not_())
        )
    )

    # Remove trials with intervention types starting with BEHAVIORAL or DIETARY_SUPPLEMENT
    df = df.filter(
        pl.col("intervention_types_filtered")
        .list.first()
        .is_in(["BEHAVIORAL", "DIETARY_SUPPLEMENT"])
        .not_()
    )

    # Intervention category is the first InterventionType in the remaining list
    df = df.with_columns(
        intervention_category=pl.col("intervention_types_filtered").list.first()
    ).drop("intervention_types_filtered")

    # Get therapeutic areas from condition mesh IDs
    df = df.with_columns(pl.col("condition_meshes").fill_null([]))
    mesh_hierarchy = parse_mesh_hierarchy().lazy().filter(pl.col("level") == 0)

    ignore_categories = (
        "Animal Diseases",
        "Chemically-Induced Disorders",
        "Stomatognathic Diseases",
        "Pathological Conditions, Signs and Symptoms",
    )

    mesh_mapping = (
        df.explode("condition_meshes")
        .select("nct_id", mesh_id=pl.col("condition_meshes").struct.field("id"))
        .join(mesh_hierarchy, on="mesh_id")
        .select("nct_id", pl.col("term_name").alias("therapeutic_areas"))
        .unique()
        .filter(pl.col("therapeutic_areas").is_in(ignore_categories).not_())
    )

    # therapeutic areas in less than 10% of trials will be lumped into "Other" category
    frequency_cutoff = df.collect().select(pl.len() * 0.10).item()

    infrequent_categories = (
        mesh_mapping.group_by("therapeutic_areas")
        .len()
        .filter(pl.col("len") < frequency_cutoff)
        .select("therapeutic_areas")
    )

    mesh_mapping = (
        mesh_mapping.join(infrequent_categories, on="therapeutic_areas", how="anti")
        .join(df.select("nct_id"), on="nct_id", how="right")
        .fill_null("Other")
        .group_by("nct_id")
        .agg("therapeutic_areas")
    )

    df = df.join(mesh_mapping, on="nct_id")

    # Remove nulls
    def without_nulls(df: pl.DataFrame, allowed_null_cols: list[str]):
        df = df.with_columns(
            pl.col("primary_outcomes").fill_null([]),
            pl.col("design_allocation").fill_null("NA"),
            # assume it's an estimated value instead of an actual value
            pl.col("enrollment_type").fill_null("ESTIMATED"),
            pl.col("is_fda_regulated_drug").fill_null(False),
            pl.col("is_fda_regulated_device").fill_null(False),
            # pl.col("therapeutic_area").fill_null("Other"),
        )

        for col in df.columns:
            nulls = df.select(col).null_count().item()
            if nulls > 0 and col not in allowed_null_cols:
                raise ValueError(f"Column '{col}' has {nulls} null values!")

        return df

    df = without_nulls(df.collect(), allowed_null_cols=[])
    return df


def preprocess_version_summaries(raw: pl.DataFrame):
    """Preprocess raw data of trial history summaries from clinicaltrials.gov.

    Args:
        raw: A dataframe of raw trial history summary data generated with polars'
        json_normalize() method.

    Returns:
        The preprocessed dataframe of history summaries.
    """
    return (
        raw.explode("changes")
        .unnest("changes")
        .select(
            "nct_id",
            "version",
            version_status="status",
            version_date=str_col_to_date("date"),
            edited_module_labels="moduleLabels",
            last_primary_outcome_update_version="lastUpdateVersions.primaryOutcomes",
        )
    )


def preprocess_original_versions(raw: pl.DataFrame):
    """Preprocess raw data of trial original versions from clinicaltrials.gov.

    Args:
        raw: A dataframe of raw trial original version data generated with polars'
        json_normalize() method.

    Returns:
        The preprocessed dataframe of original versions.
    """
    return raw.select(
        nct_id="study.protocolSection.identificationModule.nctId",
        version="studyVersion",
        primary_outcomes="study.protocolSection.outcomesModule.primaryOutcomes",
    )


def preprocess_raw_data(raw_data_dir: str):
    """Preprocess latest version data, version summary data, and original version data
    from a directory of raw data.

    Args:
        raw_data_dir: A path to a directory containing three raw data parquet files:
        `latest_versions.parquet`, `version_summaries.parquet`, and `original_versions.parquet`.

    Returns:
        A tuple of three dataframes: `(latest_versions, version_summaries, original_versions)`.
    """
    return (
        preprocess_latest_versions(
            pl.read_parquet(os.path.join(raw_data_dir, "latest_versions.parquet"))
        ),
        preprocess_version_summaries(
            pl.read_parquet(os.path.join(raw_data_dir, "version_summaries.parquet"))
        ),
        preprocess_original_versions(
            pl.read_parquet(os.path.join(raw_data_dir, "original_versions.parquet"))
        ),
    )


# -- Find outcome edits for evaluation -- #


def find_outcome_edits(
    latest_versions: pl.DataFrame,
    version_summaries: pl.DataFrame,
    original_versions: pl.DataFrame,
):
    """Finds edits made to trials' primary outcomes, comparing the original versions to the latest versions. Version summary data is required to track version numbers (useful for generating clinicaltrials.gov URLs).

    Args:
        latest_versions: The preprocessed latest version data.
        version_summaries: The preprocessed version summary data.
        original_versions: The preprocessed original version data.

    Returns:
        A dataframe of edits made to trials' primary outcomes.
    """
    latest_versions_lazy = latest_versions.lazy()
    version_summaries_lazy = version_summaries.lazy()
    original_versions_lazy = original_versions.lazy()

    # get NCT ID, primary outcome list, and submission date for latest version
    after_data = latest_versions_lazy.select(
        "nct_id", "primary_outcomes", version_date="last_update_submit_date"
    )

    # join with summaries (on version_date) to get the version number for the latest version
    after_data = after_data.join(
        version_summaries_lazy.select("nct_id", "version_date", "version"),
        on=["nct_id", "version_date"],
    )

    # in case there are multiple versions with the same date, ensure we only take the most recent one
    after_data = after_data.group_by("nct_id").agg(
        pl.all().sort_by("version", descending=True).first()
    )

    # now we know the "after" version number and primary outcome list
    after_data = after_data.select(
        "nct_id",
        primary_outcomes_after="primary_outcomes",
        version_after="version",
    )

    # get version number for "before" data
    before_data = original_versions_lazy.join(
        version_summaries_lazy, on=["nct_id", "version"]
    )

    # rename columns with "before" suffix
    before_data = before_data.select(
        "nct_id",
        primary_outcomes_before="primary_outcomes",
        version_before="version",
    )

    # join before and after data
    result = after_data.join(before_data, on="nct_id")

    # ensure primary outcomes are different between the two versions
    result = result.filter(
        pl.col("primary_outcomes_before").ne(pl.col("primary_outcomes_after"))
    )

    # execute optimized query
    return result.collect()


# -- Load model predictions -- #


def load_model_predictions(eval_log_file: str):
    """Load model predictions from an evaluation log into a dataframe for analysis.

    Args:
        eval_log_file: Path to the evaluation log file.

    Returns:
        A dataframe containing model predictions and explanations from the evaluation.
    """
    nct_id: list[str] = []
    version_before: list[int] = []
    version_after: list[int] = []
    model_explanation: list[str] = []
    model_predictions: list[tuple[bool, bool, bool, bool, bool]] = []

    for sample in read_eval_log_samples(eval_log_file):
        nct_id.append(sample.metadata["nct_id"])
        version_before.append(sample.metadata["version_before"])
        version_after.append(sample.metadata["version_after"])
        model_explanation.append(sample.metadata["model_prediction"]["explanation"])

        predicted_categories = sample.metadata["model_prediction"]["categories"]
        model_predictions.append(
            tuple(
                category in predicted_categories
                for category in (
                    "rewording/rephrasing",
                    "elaboration",
                    "modification",
                    "addition",
                    "removal",
                )
            )
        )

    return pl.DataFrame(
        data={
            "nct_id": nct_id,
            "version_before": version_before,
            "version_after": version_after,
            "model_explanation": model_explanation,
            "model_predictions": model_predictions,
        },
        schema=pl.Schema(
            {
                "nct_id": pl.String(),
                "version_before": pl.UInt16(),
                "version_after": pl.UInt16(),
                "model_explanation": pl.String(),
                "model_predictions": pl.Struct(
                    {
                        "rewording/rephrasing": pl.Boolean,
                        "elaboration": pl.Boolean,
                        "modification": pl.Boolean,
                        "addition": pl.Boolean,
                        "removal": pl.Boolean,
                    }
                ),
            }
        ),
    )
