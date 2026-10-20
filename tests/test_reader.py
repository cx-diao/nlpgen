from pathlib import Path

from nlpgens import ALL_FINAL_LABELS, IDENTIFIER_COLUMNS, NLPEER_SUBSETS, find_pass_files, load_nlpgens
from nlpgens.reader import paper_identifiers, to_paper_names

SAMPLE_OUTPUT = Path(__file__).resolve().parent.parent / "output"


def test_load_sample_runs():
    df = load_nlpgens(SAMPLE_OUTPUT, progress=False)

    assert list(df.columns[: len(IDENTIFIER_COLUMNS)]) == IDENTIFIER_COLUMNS
    assert set(df["venue"]) == {venue for venue, _ in NLPEER_SUBSETS.values()}
    assert df["source_licence"].notna().all()
    assert df["paper_path"].nunique() == 5
    assert not any(c.endswith("_raw_response") or c.endswith("_logprobs") for c in df.columns)

    outcome = df["about_outcome"].eq(True)
    assert set(df.loc[outcome, "final_label"]) <= set(ALL_FINAL_LABELS)
    assert df.loc[~outcome, "final_label"].isna().all()


def test_load_with_raw_responses_and_paper_names():
    run = SAMPLE_OUTPUT / "260430_1814_COLING2020_v120b_t0p3_rerun"
    df = load_nlpgens(SAMPLE_OUTPUT, runs=[run.name], with_raw_responses=True, paper_names=True, progress=False)
    assert {"quantified", "quantified_raw_response", "is_present_tense", "sentence_embedded", "is_vague"} <= set(
        df.columns
    )
    assert "contains_quantification_in_main_clause" not in df.columns


def test_paper_identifiers():
    ids = paper_identifiers(Path("nlpeer_v0.1/COLING2020/data/52/v1"))
    assert ids["subset"] == "COLING2020"
    assert ids["venue"] == "COLING '20"
    assert ids["paper_id"] == "52"
    assert ids["version"] == "v1"
    assert ids["source_licence"] == "CC BY-NC-SA 4.0"

    no_data_level = paper_identifiers(Path("nlpeer_new/ARR-EMNLP-24-v1.1/00a5158786a2/v1"))
    assert no_data_level["venue"] == "EMNLP '24"
    assert no_data_level["paper_id"] == "00a5158786a2"

    own = paper_identifiers(Path("my_papers/paper_a"))
    assert own["paper_id"] == "paper_a"
    assert own["subset"] is None and own["version"] is None


def test_find_pass_files_uses_latest_complete_run(tmp_path):
    for name in [
        "m_1_of_2_20260101_000000_annotation.csv",
        "m_2_of_2_20260101_000000_annotation.csv",
        "m_2_of_1_20260301_000000_annotation.csv",  # incomplete: pass 1 missing
        "m_1_of_2_20260201_000000_annotation.csv",
        "m_2_of_2_20260201_000000_annotation.csv",
    ]:
        (tmp_path / name).write_text("sentence\n")
    files = find_pass_files(tmp_path)
    assert [f.name for f in files] == [
        "m_1_of_2_20260201_000000_annotation.csv",
        "m_2_of_2_20260201_000000_annotation.csv",
    ]


def test_to_paper_names_keeps_unmapped_columns():
    import pandas as pd

    df = pd.DataFrame(columns=["about_outcome", "contains_vague_descriptor", "contains_vague_descriptor_logprobs"])
    assert list(to_paper_names(df).columns) == ["about_outcome", "is_vague", "is_vague_logprobs"]
