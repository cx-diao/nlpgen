"""Merge repeated annotation passes by majority vote and recompute final labels.

This follows the procedure in Appendix A of the paper: each classifier label is
majority-voted across passes (pass 1 breaks ties), and the final taxonomy label
is then recomputed from the voted labels with the decision tree (Algorithm 1).
"""

from collections import Counter
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nlpgena.annotate.taxonomy import FINAL_LABEL_INPUT_COLUMNS, get_final_label
from nlpgena.annotate.utils import ANNOTATION_COLUMNS

# `notes` and `get_target_sent` are free text and are taken from pass 1;
# `final_label` is recomputed from the voted labels.
VOTE_COLUMNS = [c for c in ANNOTATION_COLUMNS if c not in ("notes", "get_target_sent", "final_label")]
COMPANION_SUFFIXES = ("_raw_response", "_logprobs")


def canon_label(v):
    # Normalise a cell to a comparable vote (None = missing); handle bool and string "true"/"false".
    if pd.isna(v):
        return None
    # Specifically catch both Python bools and NumPy bools, casting them to a native Python bool
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        lo = v.strip().lower()
        if lo == "true":
            return True
        if lo == "false":
            return False
    return v


def majority_vote(pass_values: Sequence[Any]) -> Tuple[Any, bool]:
    """Return the majority label across passes and whether a tie had to be broken.

    Missing values do not vote. Ties go to pass 1's label, or to the first
    available vote when pass 1 has none. Returns (None, False) when every pass
    is missing.
    """
    votes = [canon_label(v) for v in pass_values]
    available = [v for v in votes if v is not None]
    if not available:
        return None, False

    counts = Counter(available)
    top = counts.most_common(1)[0][1]
    tied = [v for v, c in counts.items() if c == top]
    if len(tied) == 1:
        return tied[0], False
    first_pass = votes[0]
    return (first_pass if first_pass is not None else available[0]), True


def majority_vote_merge_passes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-pass DataFrames of one paper into a single DataFrame.

    Starts from a copy of pass 1 and replaces every column in VOTE_COLUMNS with
    its row-wise majority. Each label's companion columns ({col}_raw_response,
    {col}_logprobs) are taken from the first pass that agrees with the merged
    label, so the stored reasoning explains the merged label. A ``tie_broken``
    column lists (";"-separated) the classifiers whose vote was tied.
    """
    assert dfs, "No passes to merge"
    num_rows = len(dfs[0])
    for i, df in enumerate(dfs):
        assert len(df) == num_rows, f"Pass {i + 1} has {len(df)} rows, expected {num_rows}"

    merged = dfs[0].copy()
    tied_columns: List[List[str]] = [[] for _ in range(num_rows)]

    for col in VOTE_COLUMNS:
        per_pass = [df[col].tolist() for df in dfs]
        merged_values: List[Optional[Any]] = []
        source_pass: List[int] = []
        for row_idx in range(num_rows):
            row_values = [values[row_idx] for values in per_pass]
            value, was_tied = majority_vote(row_values)
            merged_values.append(value)
            if was_tied:
                tied_columns[row_idx].append(col)
            if value is None:
                source_pass.append(0)
            else:
                source_pass.append(next(p for p, v in enumerate(row_values) if canon_label(v) == value))

        merged[col] = _as_boolean_if_possible(merged_values)

        for suffix in COMPANION_SUFFIXES:
            companion = f"{col}{suffix}"
            if companion in merged.columns:
                per_pass_companion = [df[companion].tolist() for df in dfs]
                merged[companion] = [per_pass_companion[p][row_idx] for row_idx, p in enumerate(source_pass)]

    merged["tie_broken"] = [";".join(cols) for cols in tied_columns]
    return merged


def final_label_from_labels(labels: dict) -> Optional[str]:
    """Final label for one merged row, or None if it is not about outcomes or a label is missing."""
    if canon_label(labels.get("about_outcome")) is not True:
        return None
    votes = {c: canon_label(labels.get(c)) for c in FINAL_LABEL_INPUT_COLUMNS}
    if any(v is None for v in votes.values()):
        return None
    return get_final_label({c: {"label": v} for c, v in votes.items()})


def recompute_final_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` whose final_label is recomputed from its classifier labels."""
    df = df.copy()
    columns = ("about_outcome",) + FINAL_LABEL_INPUT_COLUMNS
    rows = zip(*(df[c].tolist() for c in columns))
    df["final_label"] = [final_label_from_labels(dict(zip(columns, values))) for values in rows]
    return df


def _as_boolean_if_possible(values: List[Any]):
    if all(v is None or isinstance(v, bool) for v in values):
        return pd.array(values, dtype="boolean")
    return values
