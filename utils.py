import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Final, NamedTuple, Self

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
    eval_log_file: str | Path,
    cache_dir: str | Path | None = None,
):
    def iter_samples():
        iterator = read_eval_log_samples(
            log_file=eval_log_file, all_samples_required=True
        )

        total_samples = read_eval_log(
            eval_log_file, header_only=True
        ).eval.dataset.samples
        if total_samples is None:
            raise ValueError("no samples found in eval log file")
        elif total_samples > 500:
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


class EvalResult(NamedTuple):
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def tpr(self):
        return self.tp / (self.tp + self.fn)

    @property
    def fnr(self):
        return self.fn / (self.tp + self.fn)

    @property
    def N(self):
        return self.tp + self.tn + self.fp + self.fn

    def p_gold(self, gold: bool):
        if gold:
            return (self.tp + self.fn) / self.N
        else:
            return (self.fp + self.tn) / self.N

    def p_pred(self, pred: bool):
        if pred:
            return (self.tp + self.fp) / self.N
        else:
            return (self.tn + self.fn) / self.N

    def p_gold_given_pred(self, gold: bool, pred: bool):
        if pred:
            # P(gold=1|pred=1) = P(pred=1|gold=1) * P(gold=1) / P(pred=1)
            p = self.tpr * self.p_gold(True) / self.p_pred(True)
        else:
            # P(gold=1|pred=0) = P(pred=0|gold=1) * P(gold=1) / P(pred=0)
            p = self.fnr * self.p_gold(True) / self.p_pred(False)

        if not gold:
            p = 1 - p

        return p

    @classmethod
    def p_at_least_one_gold(
        cls, eval_results: list[Self], predicted_labels: list[bool]
    ):
        """Calculate the probability that the gold label is `True` for at least one category.

        Args:
            eval_results: A list of `EvalResult` objects, one for each category.
            predicted_labels: A list of predicted labels, one for each category.

        Returns:
            The probability that the gold label for at least one of the categories is `True`.
        """
        # use p_gold_given_pred to calculate the probability that
        # NONE of the gold labels are True, then subtract from 1
        p_no_gold = 1.0
        for eval_result, pred in zip(eval_results, predicted_labels, strict=True):
            p_no_gold *= eval_result.p_gold_given_pred(gold=False, pred=pred)
        return 1 - p_no_gold


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

    data = {"mesh_id": [], "term_name": [], "level": [], "tree_id": []}
    for mesh_id, tree_ids in tree_map.items():
        for tree_id in tree_ids:
            if (
                tree_id.startswith("C")  # diseases
                or tree_id.startswith("F")  # psychiatry and psychology
            ):
                for level in range(len(tree_id.split("."))):
                    data["mesh_id"].append(mesh_id)
                    data["term_name"].append(
                        name_map[".".join(tree_id.split(".")[: level + 1])]
                    )
                    data["level"].append(level)
                    data["tree_id"].append(tree_id)

    return pl.DataFrame(data).unique()


def _mesh_ids_to_therapeutic_areas(
    mesh_ids_col: pl.Expr,
):
    base_mesh_terms = (
        parse_mesh_hierarchy()
        .filter(pl.col("level") == 0)
        .select("mesh_id", "term_name")
        .rows()
    )
    mapping: dict[str, list[str]] = {}
    for mesh_id, term_name in base_mesh_terms:
        if mesh_id not in mapping:
            mapping[mesh_id] = []
        mapping[mesh_id].append(term_name)

    return (
        mesh_ids_col.list.filter(pl.element().is_in(mapping))
        .list.eval(
            pl.element()
            .replace_strict(mapping, return_dtype=pl.List(pl.String))
            .flatten()
        )
        .list.unique()
    )


class DerivedFields:
    # protocolSection.statusModule.startDateStruct.date
    start_date: ClassVar[pl.Expr] = (
        pl.col("data")
        .struct.field("study")
        .struct.field("protocolSection")
        .struct.field("statusModule")
        .struct.field("startDateStruct")
        .struct.field("date")
        .pipe(str_to_date)
        .alias("start_date")
    )

    # Extract the first listed intervention type that is not "PROCEDURE" or "OTHER".
    primary_intervention_type: ClassVar[pl.Expr] = (
        pl.col("data")
        .struct.field("study")
        .struct.field("protocolSection")
        .struct.field("armsInterventionsModule")
        .struct.field("interventions")
        .list.eval(pl.element().struct.field("type"))
        .list.eval(
            pl.element().filter(pl.element().is_in(["PROCEDURE", "OTHER"]).not_())
        )
        .list.first()
        .alias("primary_intervention_type")
    )

    # List of MeSH IDs for all condition MeSH terms associated with the trial.
    therapeutic_areas: ClassVar[pl.Expr] = (
        pl.col("data")
        .struct.field("study")
        .struct.field("derivedSection")
        .struct.field("conditionBrowseModule")
        .struct.field("meshes")
        .fill_null([])
        .list.eval(pl.element().struct.field("id"))
        .pipe(_mesh_ids_to_therapeutic_areas)
        .alias("therapeutic_areas")
    )

    # e.g., whether the trial is randomized
    design_allocation: ClassVar[pl.Expr] = (
        pl.col("data")
        .struct.field("study")
        .struct.field("protocolSection")
        .struct.field("designModule")
        .struct.field("designInfo")
        .struct.field("allocation")
        .alias("design_allocation")
    )

    lead_sponsor: ClassVar[pl.Expr] = (
        pl.col("data")
        .struct.field("study")
        .struct.field("protocolSection")
        .struct.field("sponsorCollaboratorsModule")
        .struct.field("leadSponsor")
        .struct.field("class")
        .alias("lead_sponsor")
    )

    enrollment_count: ClassVar[pl.Expr] = (
        pl.col("data")
        .struct.field("study")
        .struct.field("protocolSection")
        .struct.field("designModule")
        .struct.field("enrollmentInfo")
        .struct.field("count")
        .alias("enrollment_count")
    )
