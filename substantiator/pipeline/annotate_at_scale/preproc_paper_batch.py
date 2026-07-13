#!/usr/bin/env python3
"""
Batch preprocessing utility for ITG paper files.

For each provided paper, this script generates annotation CSV files using the
one-paper pipeline and validates coverage of the Introduction section.
Supply a text file listing absolute or relative paper.itg.json paths. Outputs
are written alongside the list file's directory, preserving the input directory
structure. Only successful coverage checks write annotation CSVs; failures are
captured in the top-level log files. Paths located under ``data/<collection>/...``
omit the leading ``data`` component in the generated output layout while retaining
the collection directory (e.g. ``data/foo/...`` -> ``foo/...``).
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set

from loguru import logger
from tqdm.auto import tqdm

from substantiator.pipeline.one_paper_demo.produce_annotation_form import (
    AnnotationResult,
    SECTION_KEYWORDS,
    build_annotation_result,
    write_annotation_csvs,
)
from substantiator.pipeline.one_paper_demo.check_introduction_coverage import (
    CoverageResult,
    extract_introduction_sentences_from_rows,
    run_coverage_check_from_sentences,
)


LOG_HEADERS = [
    "input_path",
    "self_check_status",
    "num_sections",
    "num_all_sentences",
    "num_selected_sentences",
    "output_all_csv",
    "output_selected_csv",
    "missing_phrase_count",
    "missing_phrase_sample",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a batch of paper.itg.json files listed in a text file: "
            "generate annotation CSVs, run coverage checks, and emit logs."
        )
    )
    parser.add_argument(
        "--input-path-list-txt",
        type=Path,
        required=True,
        help="Path to a text file containing newline-delimited paper.itg.json paths.",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        help="Optional override for section keywords used to select sentences.",
    )
    return parser.parse_args()


def normalize_base_name(path: Path) -> str:
    name = path.name
    if name.endswith(".itg.json"):
        return name[: -len(".itg.json")]
    return path.stem


def build_default_record(input_path: Path) -> Dict[str, object]:
    record: Dict[str, object] = {key: "" for key in LOG_HEADERS}
    record["input_path"] = str(input_path)
    record["num_sections"] = 0
    record["num_all_sentences"] = 0
    record["num_selected_sentences"] = 0
    record["missing_phrase_count"] = 0
    return record


def apply_annotation_stats(record: Dict[str, object], annotation: AnnotationResult) -> None:
    record["num_sections"] = annotation.num_sections
    record["num_all_sentences"] = annotation.num_all_sentences
    record["num_selected_sentences"] = annotation.num_selected_sentences


def apply_coverage_stats(record: Dict[str, object], coverage: CoverageResult) -> None:
    record["missing_phrase_count"] = coverage.missing_count
    record["missing_phrase_sample"] = "; ".join(coverage.missing_phrases[:5])


def read_input_paths(list_path: Path) -> List[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Input list file does not exist: {list_path}")
    paths: List[Path] = []
    skipped_count = 0
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned:
                continue
            p = Path(cleaned).expanduser().resolve()
            if any(part.startswith("._") for part in p.parts):
                skipped_count += 1
                continue
            paths.append(p)
    if skipped_count:
        logger.info("Skipped {} macOS resource fork (._) paths", skipped_count)
    if not paths:
        raise ValueError(f"No input paths found in {list_path}")
    return paths


def read_processed_paths(log_csv: Path) -> Set[Path]:
    """
    Read the existing log file and return a set of already processed paths.
    """
    processed_paths: Set[Path] = set()
    if not log_csv.exists():
        return processed_paths
    
    with log_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            input_path_str = row.get("input_path", "").strip()
            if input_path_str:
                processed_paths.add(Path(input_path_str).resolve())
    
    return processed_paths


def append_log_entry(log_csv: Path, entry: Dict[str, object], write_header: bool = False) -> None:
    """
    Append a single log entry to the CSV file.
    """
    mode = "w" if write_header else "a"
    with log_csv.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def derive_output_subdir(input_path: Path) -> Path:
    """
    Determine the subdirectory for outputs by taking the portion of the path
    after the leading ``data/`` component (if present). Falls back to the input's
    parent name when no meaningful suffix exists.
    """
    parts = list(input_path.parts)
    data_idx = None
    for idx, part in enumerate(parts):
        if part == "data":
            data_idx = idx
            break

    if data_idx is None:
        return Path(input_path.parent.name)

    start = data_idx + 1
    sub_parts = parts[start:-1]  # drop filename
    if not sub_parts:
        return Path(input_path.parent.name)
    return Path(*sub_parts)


def main() -> int:
    args = parse_args()

    input_paths = read_input_paths(args.input_path_list_txt)
    list_path = args.input_path_list_txt.resolve()
    output_root = list_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    log_csv = output_root / "preproc_log.csv"

    # Sanity check: read already processed paths if log exists
    processed_paths = read_processed_paths(log_csv)
    log_exists = log_csv.exists()
    
    if processed_paths:
        logger.info(
            f"Found existing log file with {len(processed_paths)} processed paper(s). "
            f"Resuming from where we left off..."
        )
    
    # Filter out already processed paths
    remaining_paths = [p for p in input_paths if p.resolve() not in processed_paths]
    
    if not remaining_paths:
        logger.info(
            f"All {len(input_paths)} paper(s) have already been processed. "
            f"Nothing to do."
        )
        return 0
    
    logger.info(
        f"Processing {len(remaining_paths)} remaining paper(s) "
        f"out of {len(input_paths)} total."
    )

    section_keywords: List[str] = (
        [kw for kw in args.keywords if kw] if args.keywords else SECTION_KEYWORDS
    )

    # Write header if this is a new log file
    if not log_exists:
        append_log_entry(log_csv, {}, write_header=True)

    path_iter = tqdm(remaining_paths, desc="Processing papers", unit="file")
    processed_count = 0

    for input_path in path_iter:
        record = build_default_record(input_path)
        try:
            annotation = build_annotation_result(
                input_path, section_keywords=section_keywords
            )
            apply_annotation_stats(record, annotation)

            introduction_sentences = extract_introduction_sentences_from_rows(
                annotation.all_rows
            )
            coverage = run_coverage_check_from_sentences(
                input_path, introduction_sentences
            )
            apply_coverage_stats(record, coverage)

            relative_dir = derive_output_subdir(input_path)
            base_name = normalize_base_name(input_path)
            all_csv = output_root / relative_dir / f"{base_name}_all_sentences.csv"
            selected_csv = (
                output_root / relative_dir / f"{base_name}_selected_sentences.csv"
            )

            if coverage.passes:
                write_annotation_csvs(
                    annotation, output_all=all_csv, output_selected=selected_csv
                )
                record["self_check_status"] = "pass"
                record["output_all_csv"] = str(all_csv)
                record["output_selected_csv"] = str(selected_csv)
            else:
                record["self_check_status"] = "fail"
                record["output_all_csv"] = ""
                record["output_selected_csv"] = ""
        except Exception as exc:
            record["self_check_status"] = "error"
            record["error"] = str(exc)
            record["output_all_csv"] = ""
            record["output_selected_csv"] = ""
        
        # Write to log immediately after finishing one paper
        append_log_entry(log_csv, record, write_header=False)
        processed_count += 1

    logger.info(
        f"Completed batch preprocessing for {processed_count} file(s). "
        f"Total processed (including previous runs): {len(processed_paths) + processed_count}. "
        f"Output base directory: {output_root}. "
        f"Log written to: {log_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
