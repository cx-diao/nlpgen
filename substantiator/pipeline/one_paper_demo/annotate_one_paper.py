import argparse
import asyncio
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rag.llm.ollama import OllamaModel

from substantiator.annotate.orchestrator import GeneralisationAnnotator
from substantiator.annotate.classifier import *
from substantiator.annotate.paper_level_annotate import *
from substantiator.annotate.sent_level_annotate import *
from substantiator.annotate.utils import (
    ANNOTATION_COLUMNS,
    LOGPROB_COLUMNS,
    PROMPT_TEMPLATE_PATH,
    RAW_RESPONSE_COLUMNS,
)
from substantiator.utils.general import get_timestamp


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


def get_sentence_info(row):
    """
    Warning: deprecated function, use annotate_paper_batch.py instead.
    Extract sentence information including role and section content.

    Args:
        row: A pandas Series representing a row from the dataset

    Returns:
        dict: Dictionary containing sentence role and section content information
    """
    return {
        "section": row["section"],
        "section_content": row["section_content"],
        "two_sent_context": row["two_sent_context"],
        "sentence": row["sentence"],
        "sentence_role": {
            "about_outcome": row["about_outcome"],
            "contains_embedded_sentence": row["contains_embedded_sentence"],
            "is_reporting_clause": row["is_reporting_clause"],
            "get_target_sent": row["get_target_sent"],
            "is_generalised": row["is_generalised"],
            "contains_quantification_in_main_clause": row["contains_quantification_in_main_clause"],
            "ref_study_population": row["ref_study_population"],
            "ref_study": row["ref_study"],
            "is_present_tense_main_verb": row["is_present_tense_main_verb"],
        },
        "final_label": row["final_label"],
        "notes": row["notes"] if pd.notna(row["notes"]) else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate paper sentences with generalisation classifiers")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file (from produce_annotation_form.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory for annotated CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-r1:32b",
        help="Model name to use for annotation (default: deepseek-r1:32b)",
    )
    args = parser.parse_args()

    # Load the original data
    data = load_data(args.input)
    assert data is not None, "Failed to load data"

    print(f"\nProcessing {len(data)} sentences...")
    modified_data = initialise_annotation_columns(data)

    llm = OllamaModel(model_name=args.model)
    annotator = GeneralisationAnnotator(llm, NAME2CLASSIFIER_TEMPLATE_PATH_NAME, debug=False)

    for idx, (_, row) in tqdm(enumerate(modified_data.iterrows()), total=len(modified_data)):
        sentence_info = get_sentence_info(row)

        # Build sentence placeholder
        sentence_placeholder = {
            "claim": sentence_info["sentence"],
            "section": sentence_info["section"],
            "section_content": sentence_info["section_content"],
            "two_sent_context": sentence_info["two_sent_context"],
        }

        # Build paper context (empty for now, can be extended with abstract, etc.)
        paper_context = {}

        annotation_result = asyncio.run(annotator.annotate_one(sentence_placeholder, paper_context=paper_context))
        for role, value in annotation_result.items():
            modified_data.loc[idx, role] = value["label"]
            modified_data.loc[idx, f"{role}_raw_response"] = value["raw_response"]

    # Create output filename
    model_name_safe = llm.model_name.replace(":", "_")
    output_filename = f"annotated_{get_timestamp()}_{model_name_safe}.csv"
    output_path = args.output / output_filename

    # Save the modified data
    save_data(modified_data, output_path)


if __name__ == "__main__":
    main()
