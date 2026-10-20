import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import spacy

# Global variable: list of keywords to filter sections by
# Sentences from sections containing any of these keywords (case-insensitive)
# will be included in the selected output file
SECTION_KEYWORDS = [
    "abstract",
    "result",
    "introduction",
    "conclusion",
    "discussion",
    "ablation",
]


@dataclass
class AnnotationResult:
    sections: List[Tuple[str, str]]
    all_rows: List[Tuple[str, str, str, str]]
    selected_rows: List[Tuple[str, str, str, str]]

    @property
    def num_sections(self) -> int:
        return len(self.sections)

    @property
    def num_all_sentences(self) -> int:
        return len(self.all_rows)

    @property
    def num_selected_sentences(self) -> int:
        return len(self.selected_rows)


def load_itg_json(itg_path: Path) -> dict:
    assert itg_path.exists(), f"Input file does not exist: {itg_path}"
    assert itg_path.is_file(), f"Input path is not a file: {itg_path}"
    with itg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "Top-level JSON must be an object."
    assert "nodes" in data, "JSON must contain 'nodes'."
    assert isinstance(data["nodes"], list), "'nodes' must be a list."
    return data


def normalise_whitespace(text: str) -> str:
    # Collapse whitespace and trim
    return re.sub(r"\s+", " ", text).strip()


def determine_heading_level(node: dict) -> int:
    """Infer heading depth based on structured numbering in metadata."""
    explicit_level = node.get("level")
    if isinstance(explicit_level, int) and explicit_level > 0:
        return explicit_level

    section_id = None
    meta = node.get("meta")
    if isinstance(meta, dict):
        section_id = meta.get("section")
    elif isinstance(meta, (str, int, float)):
        section_id = meta

    parts: List[str] = []
    if isinstance(section_id, str):
        parts = [segment for segment in re.split(r"[^\w]+", section_id) if segment]
    elif section_id is not None:
        section_str = str(section_id)
        parts = [segment for segment in re.split(r"[^\w]+", section_str) if segment]

    if parts:
        return len(parts)

    return 1


def extract_sections(
    nodes: Iterable[dict],
    *,
    include_ntypes: Optional[set] = None,
) -> List[Tuple[str, str]]:
    # Only include proper content types by default
    if include_ntypes is None:
        include_ntypes = {"paragraph", "list_item", "formula"}

    sections: List[Tuple[str, str]] = []
    current_section_name: Optional[str] = None
    current_accumulated_chunks: List[str] = []
    heading_stack: List[Tuple[int, str]] = []

    def flush_current_section() -> None:
        nonlocal current_section_name, current_accumulated_chunks
        if current_section_name is not None:
            full_content = normalise_whitespace(" ".join(current_accumulated_chunks))
            sections.append((current_section_name, full_content))
        current_accumulated_chunks = []

    for node in nodes:
        assert isinstance(node, dict), "Each node must be an object."
        assert "ntype" in node, "Node missing 'ntype'."
        assert "content" in node, "Node missing 'content'."
        ntype = node["ntype"]
        content = node["content"]
        assert isinstance(ntype, str), "'ntype' must be a string."
        assert isinstance(content, str), "'content' must be a string."

        # Start a new section when we encounter a heading-like node
        if ntype in {"heading", "abstract"}:
            flush_current_section()

            heading_title = normalise_whitespace(content)
            heading_level = determine_heading_level(node) if ntype == "heading" else 1

            if heading_level <= 0:
                heading_level = 1

            if ntype == "abstract":
                heading_stack.clear()

            heading_stack = [(level, name) for (level, name) in heading_stack if level < heading_level]

            if heading_stack:
                combined_name = f"{heading_stack[-1][1]} > {heading_title}"
            else:
                combined_name = heading_title

            heading_stack.append((heading_level, combined_name))
            current_section_name = combined_name
            continue

        # Skip everything until the first heading appears
        if current_section_name is None:
            continue

        # Only accumulate content for explicitly included types
        if ntype in include_ntypes:
            cleaned = normalise_whitespace(content)
            if cleaned:
                # Special handling for Conclusion: only take the first paragraph that looks like actual conclusion
                current_accumulated_chunks.append(cleaned)

    # Flush the final section
    if current_section_name is not None:
        flush_current_section()

    return sections


def build_sentenciser():
    # Prefer full English model for better sentence boundary detection
    if spacy is None:
        raise ValueError("spacy is not installed")

    for model in [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "xx_sent_ud_sm",
    ]:
        nlp = spacy.load(model)
        return nlp

        # Note: don't use fallback, it will cause different results without notifying the user

        # Fallback to a blank pipeline with simple sentencizer
        # nlp = spacy.blank("en")
        # if "sentencizer" not in nlp.pipe_names:
        #     nlp.add_pipe("sentencizer")
        # return nlp


def split_sentences(text: str, *, nlp=None) -> List[str]:
    """Split raw section text into complete, readable sentences."""

    def split_on_semicolons_outside_citations(raw: str) -> List[str]:
        segments: List[str] = []
        current: List[str] = []
        depth = 0
        bracket_pairs = {"(": ")", "[": "]", "{": "}"}
        closing = set(bracket_pairs.values())

        for ch in raw:
            if ch in bracket_pairs:
                depth += 1
                current.append(ch)
            elif ch in closing:
                if depth > 0:
                    depth -= 1
                current.append(ch)
            elif ch == ";" and depth == 0:
                current.append(ch)
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
            else:
                current.append(ch)

        remaining = "".join(current).strip()
        if remaining:
            segments.append(remaining)
        return segments

    def is_complete(sent: str) -> bool:
        stripped = sent.rstrip()
        # Allow for trailing quotes/parentheses after punctuation
        while stripped and stripped[-1] in "\"')]" and stripped[:-1]:
            stripped = stripped[:-1]
        return bool(stripped) and stripped[-1] in ".!?;"

    def finalise_sentence(sent: str) -> str:
        stripped = sent.rstrip()
        suffix = ""
        while stripped and stripped[-1] in "\"')]" and stripped[:-1]:
            suffix = stripped[-1] + suffix
            stripped = stripped[:-1]
        if stripped.endswith(";"):
            stripped = stripped[:-1].rstrip() + "."
        return normalise_whitespace(stripped + suffix)

    def merge_incomplete(chunks: List[str]) -> List[str]:
        merged: List[str] = []
        buffer: List[str] = []
        for chunk in chunks:
            buffer.append(chunk)
            candidate = " ".join(buffer).strip()
            if is_complete(candidate):
                merged.append(finalise_sentence(candidate))
                buffer = []
        if buffer:
            merged.append(finalise_sentence(" ".join(buffer).strip()))
        return merged

    if not text.strip():
        return []

    semicolon_chunks = split_on_semicolons_outside_citations(text)

    parts: List[str] = []
    if nlp is not None:
        for chunk in semicolon_chunks:
            doc = nlp(chunk)
            parts.extend(normalise_whitespace(sent.text) for sent in doc.sents if sent.text.strip())
    else:
        for chunk in semicolon_chunks:
            pieces = re.split(r"(?<=[.!?])\s+", chunk)
            parts.extend(normalise_whitespace(p) for p in pieces if p.strip())

    return merge_incomplete(parts)


def write_csv(rows: Iterable[Tuple[str, str, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "section_content", "two_sent_context", "sentence"])
        for section, section_content, sentence, two_sent_context in rows:
            writer.writerow([section, section_content, two_sent_context, sentence])


def generate_rows(sections: List[Tuple[str, str]]) -> Iterable[Tuple[str, str, str, str]]:
    nlp = build_sentenciser()
    for section_name, section_content in sections:
        # It is acceptable that a section has empty content; in that case, yield no rows
        if not section_content:
            continue
        sentences = split_sentences(section_content, nlp=nlp)
        if not sentences:
            continue
        cleaned_section_content = " ".join(sentences)
        for i, sent in enumerate(sentences):
            # Build two_sent_context: up to two preceding sentences + current sentence
            context_parts = []
            if i >= 2:
                context_parts.append(sentences[i - 2])
            if i >= 1:
                context_parts.append(sentences[i - 1])
            # context_parts.append(sent)
            two_sent_context = " ".join(context_parts)
            yield (section_name, cleaned_section_content, sent, two_sent_context)


def filter_rows_by_section_keywords(
    rows: List[Tuple[str, str, str, str]], keywords: List[str]
) -> List[Tuple[str, str, str, str]]:
    """Filter rows to only include sentences from sections whose names contain any of the keywords."""
    if not keywords:
        return rows

    keywords_lower = [kw.lower() for kw in keywords]
    filtered_rows = []

    for section_name, section_content, sentence, two_sent_context in rows:
        section_name_lower = section_name.lower()
        if any(keyword in section_name_lower for keyword in keywords_lower):
            filtered_rows.append((section_name, section_content, sentence, two_sent_context))

    return filtered_rows


def build_annotation_result(itg_path: Path, *, section_keywords: Optional[List[str]] = None) -> AnnotationResult:
    """Build structured annotation data for an ITG paper."""
    data = load_itg_json(itg_path)
    nodes = data["nodes"]
    sections = extract_sections(nodes)
    rows = list(generate_rows(sections))
    keywords = section_keywords if section_keywords is not None else SECTION_KEYWORDS
    selected_rows = filter_rows_by_section_keywords(rows, keywords)
    return AnnotationResult(sections=sections, all_rows=rows, selected_rows=selected_rows)


def write_annotation_csvs(annotation: AnnotationResult, *, output_all: Path, output_selected: Path) -> None:
    """Write annotation CSV files for all and keyword-selected sentences."""
    write_csv(annotation.all_rows, output_all)
    write_csv(annotation.selected_rows, output_selected)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse an ITG JSON paper, group content by headings, and output a CSV with"
            " columns: section, section_content, sentence."
        )
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
        help="Path to output CSV file (all sentences)",
    )
    parser.add_argument(
        "--output-selected",
        type=Path,
        required=True,
        help="Path to output CSV file (filtered sentences from sections with keywords)",
    )
    args = parser.parse_args()

    annotation = build_annotation_result(args.input)
    write_annotation_csvs(annotation, output_all=args.output, output_selected=args.output_selected)

    print(
        f"Completed. Sections: {annotation.num_sections:,}; "
        f"All sentences: {annotation.num_all_sentences:,}; "
        f"Selected sentences: {annotation.num_selected_sentences:,}.\n"
        f"Full CSV written to: {args.output}\n"
        f"Selected CSV written to: {args.output_selected}"
    )


if __name__ == "__main__":
    main()
