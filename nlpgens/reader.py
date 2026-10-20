"""Load NLPGens (or any NLPGenA output) from annotation run directories.

A run directory mirrors the input dataset layout and holds one CSV per
annotation pass for every paper version, named
``<model>_<k>_of_<N>_<timestamp>_annotation.csv``. The loaders below find
these files, merge the passes by majority vote, recompute the final labels
and add identifier columns.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from nlpgens.merge import COMPANION_SUFFIXES, majority_vote_merge_passes, recompute_final_labels
from substantiator.annotate.taxonomy import CODE_TO_PAPER_COLUMN

PASS_FILE_PATTERN = re.compile(
    r"^(?P<model>.+)_(?P<pass_idx>\d+)_of_(?P<num_passes>\d+)_(?P<timestamp>\d{8}_\d{6})_annotation\.csv$"
)
VERSION_PATTERN = re.compile(r"^v\d+$")

# NLPeer subset -> (venue name used in the paper, subset-level licence of the source papers; Appendix J).
NLPEER_SUBSETS = {
    "PeerRead-ACL2017": ("ACL '17", "CC BY 4.0"),
    "COLING2020": ("COLING '20", "CC BY-NC-SA 4.0"),
    "ARR-22": ("ARR '22", "CC BY-NC-SA 4.0"),
    "EMNLP23": ("EMNLP '23", "CC BY-NC 4.0"),
    "ARR-EMNLP-24-v1.1": ("EMNLP '24", "CC BY-NC 4.0"),
}

IDENTIFIER_COLUMNS = ["run", "paper_path", "subset", "venue", "paper_id", "version", "row_idx", "source_licence"]


def find_pass_files(paper_dir: Path) -> List[Path]:
    """Return the pass CSVs of one paper version, ordered by pass index.

    If the directory holds several complete runs, the most recent one is used.
    """
    runs: Dict[tuple, Dict[int, Path]] = defaultdict(dict)
    for path in Path(paper_dir).glob("*_annotation.csv"):
        match = PASS_FILE_PATTERN.match(path.name)
        if match:
            key = (match["model"], match["timestamp"], int(match["num_passes"]))
            runs[key][int(match["pass_idx"])] = path
    complete = {key: passes for key, passes in runs.items() if sorted(passes) == list(range(1, key[2] + 1))}
    if not complete:
        raise FileNotFoundError(f"No complete set of annotation passes in {paper_dir}")

    latest = max(complete, key=lambda key: key[1])
    if len(complete) > 1:
        logger.warning("{} holds {} complete runs; using the latest ({})", paper_dir, len(complete), latest[1])
    return [complete[latest][i] for i in sorted(complete[latest])]


def find_paper_dirs(run_dir: Path) -> List[Path]:
    """Return every directory under ``run_dir`` that holds annotation pass CSVs."""
    paper_dirs = {path.parent for path in Path(run_dir).rglob("*_annotation.csv") if PASS_FILE_PATTERN.match(path.name)}
    return sorted(paper_dirs)


def load_paper(
    paper_dir: Path,
    *,
    with_raw_responses: bool = False,
    with_logprobs: bool = False,
    with_section_content: bool = False,
) -> pd.DataFrame:
    """Load and merge the annotation passes of one paper version.

    Raw LLM responses, logprobs and full section text make up most of the file
    size, so they are only read when requested.
    """

    def _keep(column: str) -> bool:
        if column.endswith("_raw_response"):
            return with_raw_responses
        if column.endswith("_logprobs"):
            return with_logprobs
        if column == "section_content":
            return with_section_content
        return True

    passes = [pd.read_csv(path, usecols=_keep) for path in find_pass_files(paper_dir)]
    return recompute_final_labels(majority_vote_merge_passes(passes))


def load_run(
    run_dir: Path,
    *,
    paper_names: bool = False,
    progress: bool = True,
    **load_kwargs,
) -> pd.DataFrame:
    """Load every paper of one annotation run into a single DataFrame.

    Args:
        run_dir: An NLPGens run directory, or the output directory of your own NLPGenA run.
        paper_names: Rename columns to the names used in the paper (see CODE_TO_PAPER_COLUMN).
        progress: Show a progress bar over papers.
        **load_kwargs: Passed to :func:`load_paper` (with_raw_responses, with_logprobs, with_section_content).
    """
    run_dir = Path(run_dir)
    paper_dirs = find_paper_dirs(run_dir)
    if not paper_dirs:
        raise FileNotFoundError(f"No annotation pass CSVs found under {run_dir}")

    frames = []
    for paper_dir in tqdm(paper_dirs, desc=f"Loading {run_dir.name}", unit="paper", disable=not progress):
        df = load_paper(paper_dir, **load_kwargs)
        identifiers = paper_identifiers(paper_dir.relative_to(run_dir))
        identifiers["run"] = run_dir.name
        for position, column in enumerate(IDENTIFIER_COLUMNS):
            if column == "row_idx":
                df.insert(position, column, range(len(df)))
            else:
                df.insert(position, column, identifiers[column])
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    return to_paper_names(merged) if paper_names else merged


def load_nlpgens(root: Path, *, runs: Optional[Sequence[str]] = None, **load_kwargs) -> pd.DataFrame:
    """Load all NLPGens runs found directly under ``root`` (the unpacked dataset).

    Args:
        root: Directory that contains the run directories.
        runs: Optional subset of run directory names to load.
        **load_kwargs: Passed to :func:`load_run`.
    """
    root = Path(root)
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (runs is None or d.name in runs))
    run_dirs = [d for d in run_dirs if find_paper_dirs(d)]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with annotation pass CSVs under {root}")
    return pd.concat([load_run(d, **load_kwargs) for d in run_dirs], ignore_index=True)


def paper_identifiers(paper_path: Path) -> Dict[str, Optional[str]]:
    """Derive subset, paper id and version from a path such as ``nlpeer_v0.1/COLING2020/data/52/v1``."""
    parts = Path(paper_path).parts
    version = parts[-1] if parts and VERSION_PATTERN.match(parts[-1]) else None
    paper_id_index = len(parts) - 2 if version else len(parts) - 1
    paper_id = parts[paper_id_index] if paper_id_index >= 0 else None

    # A known NLPeer subset name anywhere in the path, else the directory just above "data".
    subset = next((part for part in parts[:paper_id_index] if part in NLPEER_SUBSETS), None)
    if subset is None and "data" in parts[:paper_id_index]:
        data_index = parts.index("data")
        subset = parts[data_index - 1] if data_index > 0 else None
    venue, licence = NLPEER_SUBSETS.get(subset, (None, None))
    return {
        "paper_path": Path(paper_path).as_posix(),
        "subset": subset,
        "venue": venue,
        "paper_id": paper_id,
        "version": version,
        "source_licence": licence,
    }


def to_paper_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename classifier columns (and their companion columns) to the names used in the paper."""
    mapping = {}
    for code_name, paper_name in CODE_TO_PAPER_COLUMN.items():
        mapping[code_name] = paper_name
        for suffix in COMPANION_SUFFIXES:
            mapping[f"{code_name}{suffix}"] = f"{paper_name}{suffix}"
    return df.rename(columns=mapping)
