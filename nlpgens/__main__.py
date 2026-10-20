"""Command-line interface: ``python -m nlpgens {merge,summary} RUN_DIR [RUN_DIR ...]``."""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from nlpgens.reader import load_run
from nlpgena.annotate.taxonomy import ALL_FINAL_LABELS

# The ten yes/no classifiers counted in Appendix A of the paper. All but about_outcome only run on outcome sentences.
REPORTED_CLASSIFIERS = [
    "about_outcome",
    "contains_embedded_sentence",
    "is_reporting_clause",
    "is_generalised",
    "contains_quantification_in_main_clause",
    "ref_study_population",
    "ref_study",
    "is_present_tense_main_verb",
    "contains_hedging",
    "contains_vague_descriptor",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m nlpgens",
        description="Merge the annotation passes of NLPGens / NLPGenA run directories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="Write one merged table (CSV, or Parquet if --out ends in .parquet).")
    merge.add_argument("run_dirs", nargs="+", type=Path, help="One or more run directories.")
    merge.add_argument("--out", required=True, type=Path, help="Output file.")
    merge.add_argument("--with-raw-responses", action="store_true", help="Include the LLM reasoning for each label.")
    merge.add_argument("--with-logprobs", action="store_true", help="Include answer-token logprobs.")
    merge.add_argument("--with-section-content", action="store_true", help="Include the full section text per row.")
    merge.add_argument("--paper-names", action="store_true", help="Use the column names from the paper.")

    summary = subparsers.add_parser("summary", help="Print sentence counts and the final-label distribution.")
    summary.add_argument("run_dirs", nargs="+", type=Path, help="One or more run directories.")
    return parser.parse_args(argv)


def summarise(df: pd.DataFrame) -> str:
    group = df["venue"].fillna(df["run"]).rename("venue")
    per_venue = pd.DataFrame(
        {
            "papers": df.groupby(group)["paper_path"].nunique(),
            "sentences": df.groupby(group).size(),
            "about_outcome": df["about_outcome"].eq(True).groupby(group).sum(),
            "fully_labelled": df["final_label"].notna().groupby(group).sum(),
        }
    )
    per_venue.loc["Total"] = per_venue.sum()

    counts = df["final_label"].value_counts().reindex(ALL_FINAL_LABELS, fill_value=0)
    counts.loc["Total"] = counts.sum()
    labels = pd.DataFrame({"n": counts, "%": (100 * counts / counts["Total"]).round(1)})

    # Counted as in Appendix A: about_outcome for every sentence, the other nine classifiers for outcome sentences.
    outcome = df["about_outcome"].eq(True).fillna(False).astype(bool)
    downstream = REPORTED_CLASSIFIERS[1:]
    missing = df["about_outcome"].isna() | (outcome & df[downstream].isna().any(axis=1))
    tied = [
        "about_outcome" in cols or (is_outcome and not cols.isdisjoint(downstream))
        for cols, is_outcome in zip(df["tie_broken"].str.split(";").apply(set), outcome)
    ]
    lines = [
        per_venue.to_string(),
        "",
        labels.to_string(),
        "",
        f"sentences with a missing label: {int(missing.sum())}",
        f"sentences with a tie-broken label: {sum(tied)}",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "merge":
        load_kwargs = dict(
            paper_names=args.paper_names,
            with_raw_responses=args.with_raw_responses,
            with_logprobs=args.with_logprobs,
            with_section_content=args.with_section_content,
        )
    else:
        load_kwargs = {}
    df = pd.concat([load_run(run_dir, **load_kwargs) for run_dir in args.run_dirs], ignore_index=True)

    if args.command == "summary":
        print(summarise(df))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
