#!/usr/bin/env python3
"""
Auto-check script to verify all sentences from the Introduction section
in the original ITG JSON appear in the output CSV file.
"""
import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


def normalise_whitespace(text: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison by removing all punctuation and extra spaces."""
    text = normalise_whitespace(text)
    # Remove all punctuation for comparison
    text = re.sub(r"[,;:.!?()]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def extract_introduction_text_from_itg(itg_path: Path) -> str:
    """Extract all text from the Introduction section."""
    with itg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    in_introduction = False
    introduction_chunks = []

    for node in nodes:
        ntype = node.get("ntype", "")
        content = node.get("content", "")

        # Check if we're entering the Introduction section
        if ntype == "heading" and "introduction" in content.lower():
            in_introduction = True
            continue

        # Check if we're leaving Introduction (next heading)
        if in_introduction and ntype == "heading":
            break

        # Collect content from Introduction
        if in_introduction and ntype in {"paragraph", "list_item", "formula"}:
            cleaned = normalise_whitespace(content)
            if cleaned:
                introduction_chunks.append(cleaned)

    return " ".join(introduction_chunks)


def extract_introduction_sentences_from_csv(csv_path: Path) -> List[str]:
    """Extract all sentence entries from Introduction section in CSV."""
    sentences = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["section"] == "Introduction":
                sentences.append(row["sentence"])
    return sentences


def extract_introduction_sentences_from_rows(
    rows: Iterable[Sequence[str]],
) -> List[str]:
    """Extract Introduction sentences from in-memory rows."""
    sentences: List[str] = []
    for row in rows:
        if not row:
            continue
        section = row[0]
        if section == "Introduction":
            # Expect tuples like (section, section_content, sentence)
            sentence = row[2] if len(row) > 2 else ""
            if sentence:
                sentences.append(sentence)
    return sentences


def extract_key_phrases(text: str) -> Set[str]:
    """Extract key 5-word phrases from text for coverage checking."""
    # Normalize first, then split into words
    normalized = normalize_for_comparison(text)
    words = normalized.split()
    phrases = set()
    # Create 5-word sliding windows
    for i in range(len(words) - 4):
        phrase = " ".join(words[i : i + 5])
        phrases.add(phrase)
    return phrases


def check_coverage(original_text: str, csv_sentences: List[str]) -> tuple[bool, List[str]]:
    """
    Check if all key phrases from original appear in CSV output.
    Returns: (passes, missing_phrases)
    """
    # Extract key phrases from both
    original_phrases = extract_key_phrases(original_text)
    csv_text = " ".join(csv_sentences)
    csv_phrases = extract_key_phrases(csv_text)

    # Find missing phrases
    missing = original_phrases - csv_phrases

    passes = len(missing) == 0
    return passes, sorted(missing)


@dataclass
class CoverageResult:
    passes: bool
    missing_phrases: List[str]
    original_phrases: int
    csv_phrases: int

    @property
    def missing_count(self) -> int:
        return len(self.missing_phrases)


def run_coverage_check_from_sentences(
    itg_path: Path,
    csv_sentences: List[str],
    *,
    original_text: Optional[str] = None,
) -> CoverageResult:
    """Run coverage check using pre-collected Introduction sentences."""
    if original_text is None:
        original_text = extract_introduction_text_from_itg(itg_path)
    passes, missing = check_coverage(original_text, csv_sentences)
    original_phrase_count = len(extract_key_phrases(original_text))
    csv_phrase_count = len(extract_key_phrases(" ".join(csv_sentences)))
    return CoverageResult(
        passes=passes,
        missing_phrases=missing,
        original_phrases=original_phrase_count,
        csv_phrases=csv_phrase_count,
    )


def run_coverage_check_from_csv(itg_path: Path, csv_path: Path) -> CoverageResult:
    """Convenience wrapper to run coverage check by reading a CSV file."""
    csv_sentences = extract_introduction_sentences_from_csv(csv_path)
    return run_coverage_check_from_sentences(itg_path, csv_sentences)


def main():
    parser = argparse.ArgumentParser(
        description="Check that Introduction section sentences are fully covered in output CSV"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input paper.itg.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file to check",
    )
    args = parser.parse_args()

    # Extract data
    print(f"Reading original ITG file: {args.input}")
    original_text = extract_introduction_text_from_itg(args.input)
    print(f"Original Introduction text length: {len(original_text)} chars\n")

    print(f"Reading CSV output: {args.output}")
    csv_sentences = extract_introduction_sentences_from_csv(args.output)
    print(f"CSV has {len(csv_sentences)} Introduction sentences\n")

    # Check coverage
    result = run_coverage_check_from_sentences(
        args.input, csv_sentences, original_text=original_text
    )

    if result.passes:
        print("✓ PASS: All Introduction content phrases are covered in the output!")
    else:
        print(f"✗ FAIL: {result.missing_count} key phrases are missing from output\n")
        print("Missing phrases (sample):")
        for i, phrase in enumerate(result.missing_phrases[:10], 1):
            print(f"{i}. {phrase}")
        if result.missing_count > 10:
            print(f"... and {result.missing_count - 10} more")

    return 0 if result.passes else 1


if __name__ == "__main__":
    exit(main())
