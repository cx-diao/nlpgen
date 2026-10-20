"""Classifier configuration and CSV helpers used by annotate_at_scale/annotate_paper_batch.py."""

from pathlib import Path

import pandas as pd

from substantiator.annotate.classifier import *
from substantiator.annotate.paper_level_annotate import *
from substantiator.annotate.sent_level_annotate import *
from substantiator.annotate.utils import (
    ANNOTATION_COLUMNS,
    LOGPROB_COLUMNS,
    PROMPT_TEMPLATE_PATH,
    RAW_RESPONSE_COLUMNS,
)


# NOTE: annotators are executed in the order they appear in the config dict.
# Make sure to define dependencies before dependents.
NAME2CLASSIFIER_TEMPLATE_PATH_NAME = {
    # "what_is_outcome": {
    #     "class": WhatIsOutcomeAnnotator,
    #     "template_path": PROMPT_TEMPLATE_PATH / "what_is_outcome.yaml",
    #     "template_name": "latest",
    #     "scope": "paper",
    #     "is_auxiliary": True,
    # },
    "about_outcome": {
        "class": DiscourseRoleClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "discourse_role.yaml",
        "template_name": "latest",
    },
    "contains_embedded_sentence": {
        "class": EmbeddedSentenceClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "embedded_sentence.yaml",
        "template_name": "latest",
    },
    "is_reporting_clause": {
        "class": IsReportingClauseClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "is_reporting_clause.yaml",
        "template_name": "latest",
    },
    "get_target_sent": {
        "class": GetTargetSentAnnotator,
        "template_path": PROMPT_TEMPLATE_PATH / "get_target_sent.yaml",
        "template_name": "latest",
    },
    "is_generalised": {
        "class": GeneralityClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "generality.yaml",
        "template_name": "latest",
    },
    "ref_study": {
        "class": ParticularStudyClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "particular_study.yaml",
        "template_name": "latest",
    },
    "ref_study_population": {
        "class": SamplePopulationClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "sample_population.yaml",
        "template_name": "latest",
    },
    "is_present_tense_main_verb": {
        "class": IsPresentTenseClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "tense.yaml",
        "template_name": "latest",
    },
    "contains_quantification_in_main_clause": {
        "class": QuantificationClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "quantification.yaml",
        "template_name": "latest",
    },
    "contains_hedging": {
        "class": HedgingClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "hedging.yaml",
        "template_name": "latest",
    },
    "contains_vague_descriptor": {
        "class": VagueDescriptorClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "vague_descriptor.yaml",
        "template_name": "latest",
    },
    "is_opinion": {
        "class": OpinionClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "opinion.yaml",
        "template_name": "latest",
    },
    "has_citation": {
        "class": CitationClassifier,
        "template_path": PROMPT_TEMPLATE_PATH / "citation.yaml",
        "template_name": "latest",
    },
}


def load_data(csv_path: Path):
    """
    Load and parse the CSV file containing paper sentences.

    Args:
        csv_path: Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame with proper column names and structure
    """
    assert csv_path.exists(), f"Input file does not exist: {csv_path}"
    assert csv_path.is_file(), f"Input path is not a file: {csv_path}"

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Filter out rows with missing sentences (NaN values)
        df_clean = df[df["sentence"].notna() & (df["sentence"].str.strip() != "")]

        print(f"Clean data shape (non-empty sentences): {df_clean.shape}")

        return df_clean

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def initialise_annotation_columns(df):
    """
    Ensure all annotation columns exist and are initialised to None.

    Args:
        df: DataFrame containing the annotation data

    Returns:
        pd.DataFrame: DataFrame with all annotation columns initialised
    """
    # Create a copy of the dataframe
    df_modified = df.copy()

    # Initialise all annotation columns to None (object dtype) if they don't exist
    for column in ANNOTATION_COLUMNS + RAW_RESPONSE_COLUMNS + LOGPROB_COLUMNS:
        if column not in df_modified.columns:
            df_modified[column] = pd.array([None] * len(df_modified), dtype="object")
            print(f"Created column {column} and set to None")

    return df_modified


def save_data(df, output_path):
    """
    Save the modified DataFrame to a CSV file.

    Args:
        df: DataFrame to save
        output_path: Path where to save the CSV file
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

