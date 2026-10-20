import pandas as pd

from nlpgens.merge import (
    VOTE_COLUMNS,
    final_label_from_labels,
    majority_vote,
    majority_vote_merge_passes,
    recompute_final_labels,
)
from nlpgena.annotate.taxonomy import FINAL_LABEL_INPUT_COLUMNS


def test_majority_vote():
    assert majority_vote([True, True, False]) == (True, False)
    assert majority_vote(["True", "false", "FALSE"]) == (False, False)
    # Ties go to pass 1, or to the first available vote when pass 1 is missing.
    assert majority_vote([True, False, None]) == (True, True)
    assert majority_vote([None, False, True]) == (False, True)
    assert majority_vote([None, None, float("nan")]) == (None, False)


def _pass(about_outcome, is_generalised, raw_response, target):
    # Every voted column must be present, as in real pass files; the ones not under test stay empty.
    columns = {c: [None, None] for c in VOTE_COLUMNS}
    columns.update(
        {
            "sentence": ["s0", "s1"],
            "about_outcome": about_outcome,
            "is_generalised": is_generalised,
            "is_generalised_raw_response": raw_response,
            "get_target_sent": target,
        }
    )
    return pd.DataFrame(columns)


def test_merge_passes_votes_labels_and_realigns_reasoning():
    passes = [
        _pass([True, False], [False, None], ["p1-r0", "p1-r1"], ["t1", "t1"]),
        _pass([True, False], [True, True], ["p2-r0", "p2-r1"], ["t2", "t2"]),
        _pass([True, True], [True, False], ["p3-r0", "p3-r1"], ["t3", "t3"]),
    ]
    merged = majority_vote_merge_passes(passes)

    assert merged["about_outcome"].tolist() == [True, False]
    assert merged["is_generalised"].tolist() == [True, True]
    # The reasoning comes from the first pass that agrees with the merged label.
    assert merged["is_generalised_raw_response"].tolist() == ["p2-r0", "p2-r1"]
    # Row 1: is_generalised is 1-1 between passes 2 and 3 (pass 1 missing), so pass 2 wins the tie.
    assert merged["tie_broken"].tolist() == ["", "is_generalised"]
    # Free text is taken from pass 1.
    assert merged["get_target_sent"].tolist() == ["t1", "t1"]


def test_final_label_needs_about_outcome_and_all_inputs():
    labels = {c: False for c in FINAL_LABEL_INPUT_COLUMNS}
    assert final_label_from_labels({**labels, "about_outcome": True}) == "non-gen"
    assert final_label_from_labels({**labels, "about_outcome": False}) is None
    assert final_label_from_labels({**labels, "about_outcome": True, "ref_study": None}) is None

    df = pd.DataFrame([{**labels, "about_outcome": True, "is_generalised": True, "final_label": "stale"}])
    assert recompute_final_labels(df)["final_label"].tolist() == ["gen/bare/non-generic"]
