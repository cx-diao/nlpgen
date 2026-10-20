"""NLPGenX taxonomy: merge the classifier labels of one sentence into a final label.

This module has no LLM dependencies so that the NLPGens reader can recompute
final labels without installing the annotation runtime.
"""

from typing import Any, Dict

# Final labels, named as in the paper (Figure 1, Table 3).
ALL_FINAL_LABELS = [
    "non-gen",
    "gen/framed/non-generic",
    "gen/framed/generic",
    "gen/bare/non-generic",
    "gen/bare/generic",
]

# Classifier columns that feed the decision tree (Algorithm 1 in the paper).
FINAL_LABEL_INPUT_COLUMNS = (
    "is_generalised",
    "contains_quantification_in_main_clause",
    "ref_study_population",
    "ref_study",
    "is_present_tense_main_verb",
    "is_reporting_clause",
    "contains_embedded_sentence",
)

# Column names used in the code and in the NLPGens CSVs -> names used in the paper.
# Columns not listed here (about_outcome, is_generalised, ref_study, ...) have the same name in both.
# is_opinion and has_citation are annotated but not reported in the paper.
CODE_TO_PAPER_COLUMN = {
    "contains_quantification_in_main_clause": "quantified",
    "is_present_tense_main_verb": "is_present_tense",
    "contains_embedded_sentence": "sentence_embedded",
    "contains_vague_descriptor": "is_vague",
}


def get_final_label(buffer: Dict[str, Any]) -> str:
    """Return the final taxonomy label for one about_outcome sentence.

    Args:
        buffer: Maps each name in FINAL_LABEL_INPUT_COLUMNS to a dict with a boolean ``label``.
    """

    def _label(key: str) -> bool:
        return bool(buffer[key]["label"])

    if not _label("is_generalised"):
        final_label = "non-gen"
    elif _label("ref_study_population"):
        final_label = "gen/framed/non-generic"
    elif _label("ref_study"):
        # Framed claims are generic only if the frame can be removed (reporting clause with an
        # embedded sentence) and what remains is present tense without quantification.
        is_generic = (
            _label("is_present_tense_main_verb")
            and _label("is_reporting_clause")
            and _label("contains_embedded_sentence")
            and not _label("contains_quantification_in_main_clause")
        )
        final_label = "gen/framed/generic" if is_generic else "gen/framed/non-generic"
    else:
        is_generic = _label("is_present_tense_main_verb") and not _label("contains_quantification_in_main_clause")
        final_label = "gen/bare/generic" if is_generic else "gen/bare/non-generic"

    assert final_label in ALL_FINAL_LABELS, f"Final label {final_label} is not in {ALL_FINAL_LABELS}"
    return final_label
