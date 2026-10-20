import itertools

from substantiator.annotate.taxonomy import ALL_FINAL_LABELS, FINAL_LABEL_INPUT_COLUMNS, get_final_label


def algorithm_1(is_generalised, ref_study_population, ref_study, is_present_tense, is_reporting_clause,
                sentence_embedded, quantified):
    """Algorithm 1 of the paper, transcribed line by line."""
    if not is_generalised:
        return "non-gen"
    if ref_study_population:
        return "gen/framed/non-generic"
    if ref_study:
        if not is_present_tense:
            return "gen/framed/non-generic"
        if not is_reporting_clause:
            return "gen/framed/non-generic"
        if not sentence_embedded:
            return "gen/framed/non-generic"
        if quantified:
            return "gen/framed/non-generic"
        return "gen/framed/generic"
    if not is_present_tense:
        return "gen/bare/non-generic"
    if quantified:
        return "gen/bare/non-generic"
    return "gen/bare/generic"


def test_final_label_matches_algorithm_1_for_every_combination():
    seen = set()
    for labels in itertools.product([False, True], repeat=len(FINAL_LABEL_INPUT_COLUMNS)):
        values = dict(zip(FINAL_LABEL_INPUT_COLUMNS, labels))
        expected = algorithm_1(
            is_generalised=values["is_generalised"],
            ref_study_population=values["ref_study_population"],
            ref_study=values["ref_study"],
            is_present_tense=values["is_present_tense_main_verb"],
            is_reporting_clause=values["is_reporting_clause"],
            sentence_embedded=values["contains_embedded_sentence"],
            quantified=values["contains_quantification_in_main_clause"],
        )
        assert get_final_label({c: {"label": v} for c, v in values.items()}) == expected, values
        seen.add(expected)
    assert seen == set(ALL_FINAL_LABELS)
