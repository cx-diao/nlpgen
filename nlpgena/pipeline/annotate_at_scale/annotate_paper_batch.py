#!/usr/bin/env python3
"""
Annotate selected sentence CSVs produced by ``preproc_paper_batch.py``.

This batch script reads a preprocessing ``preproc_log.csv`` file, locates the
``output_selected_csv`` entries, and runs the generalisation annotator on each
CSV similarly to the one-paper demo script. Results are written next to the
source CSV as ``<model>_<timestamp>_annotation.csv`` using a per-run
timestamp, and a companion ``<model>_<timestamp>_annotation_log.csv`` tracks
progress so interrupted runs can resume without repeating completed work.

Two LLM backends are available:

* ``--backend vllm`` (default): the configuration used for the paper. Runs
  ``openai/gpt-oss-*`` in-process with vLLM on CUDA GPUs (see requirements-vllm.txt).
* ``--backend openai``: any OpenAI-compatible Chat Completions endpoint,
  configured with ``--model`` and ``--base-url``. A proof of concept; it does
  not reproduce the paper's annotations. It does not request logprobs, so the
  ``*_logprobs`` columns of its output are empty.
"""
import argparse
import asyncio
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm
from loguru import logger

from nlpgena.annotate.orchestrator import GeneralisationAnnotator
from nlpgena.pipeline.one_paper_demo.annotate_one_paper import (
    NAME2CLASSIFIER_TEMPLATE_PATH_NAME,
    initialise_annotation_columns,
    load_data,
    save_data,
)
from nlpgena.utils.general import get_timestamp

try:
    from rag.llm.vllm import VLLMModel
except ImportError as exc:  # vLLM is only needed for --backend vllm (requirements-vllm.txt)
    VLLMModel = None
    VLLM_IMPORT_ERROR = exc

DEFAULT_VLLM_MODEL = "openai/gpt-oss-120b"


ANNOTATION_LOG_HEADERS = [
    "input_path",
    "output_selected_csv",
    "status",
    "annotated_csv",
    "model",
    "num_sentences",
    "started_at",
    "finished_at",
    "error",
]

ROW_KEY_SEPARATOR = "||"
LOG_FILENAME_PATTERN = re.compile(r"^(?P<model>.+)_(?P<timestamp>\d{8}_\d{6})_annotation_log\.csv$")


def sanitize_model_name(model: str) -> str:
    """Replace filesystem-unfriendly characters with underscores."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def extract_timestamp_from_log(filename: str) -> Optional[str]:
    match = LOG_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return match.group("timestamp")


def make_row_key(model: str, selected_csv: str) -> str:
    """Combine model and selected CSV path into a unique state key."""
    return f"{model}{ROW_KEY_SEPARATOR}{selected_csv}"


def parse_annotated_csv_field(value: str) -> List[Path]:
    """Deserialize the annotated_csv log field into a list of paths."""
    if not value:
        return []
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [Path(entry) for entry in parsed if entry]
        except json.JSONDecodeError:
            pass
    return [Path(value)]


@dataclass(frozen=True)
class PreprocRecord:
    input_path: Path
    selected_csv: Path
    self_check_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate the selected sentence CSVs listed in a preprocessing log. "
            "The script resumes from the latest per-model annotation log if present."
        )
    )
    parser.add_argument(
        "--log-csv",
        required=True,
        type=Path,
        help="Path to the preprocessing preproc_log.csv produced by preproc_paper_batch.py.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "openai"],
        default="vllm",
        help="LLM backend: in-process vLLM as in the paper, or an OpenAI-compatible endpoint (default: vllm).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            f"Model name. Defaults to {DEFAULT_VLLM_MODEL} for --backend vllm; "
            "required for --backend openai (the name the endpoint expects)."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "--backend openai only: endpoint URL, e.g. http://localhost:11434/v1 for Ollama. "
            "Defaults to OPENAI_BASE_URL, or the OpenAI API. The API key is read from OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=_optional_float,
        default=0.0,
        help="--backend openai only: sampling temperature, or 'none' to omit it (default: 0).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="--backend openai only: reasoning effort for reasoning models (not sent by default).",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=8,
        help="--backend openai only: maximum number of requests in flight (default: 8).",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=3,
        help="Number of repeated passes per sentence (default: 3, as in the paper).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output while annotating (default: disabled).",
    )
    parser.add_argument(
        "--use-intermediate-log",
        type=Path,
        default=None,
        help="Path to an existing annotation_log.csv to resume from its intermediate outputs.",
    )
    parser.add_argument(
        "--max-concurrent-papers",
        type=int,
        default=8,
        help="Maximum number of papers to annotate concurrently (default: 8).",
    )
    args = parser.parse_args()
    if args.model is None:
        if args.backend == "openai":
            parser.error("--model is required with --backend openai")
        args.model = DEFAULT_VLLM_MODEL
    return args


def _optional_float(value: str) -> Optional[float]:
    return None if value.lower() == "none" else float(value)


def build_llm(args: argparse.Namespace):
    """Instantiate the LLM backend selected on the command line."""
    if args.backend == "vllm":
        if VLLMModel is None:
            raise ImportError(
                "--backend vllm needs the packages in requirements-vllm.txt "
                f"(importing rag.llm.vllm failed: {VLLM_IMPORT_ERROR})"
            )
        return VLLMModel(model_name=args.model)

    from rag.llm.openai_compat import OpenAICompatModel

    return OpenAICompatModel(
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        max_concurrency=args.max_concurrent_requests,
    )


def read_preproc_log(log_path: Path) -> List[PreprocRecord]:
    if not log_path.exists():
        raise FileNotFoundError(f"Preprocessing log not found: {log_path}")
    records: List[PreprocRecord] = []
    preproc_errors: List[str] = []
    with log_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "output_selected_csv" not in reader.fieldnames or "self_check_status" not in reader.fieldnames:
            raise ValueError(f"Unexpected log format in {log_path}")
        for row in reader:
            selected = row.get("output_selected_csv", "").strip()
            if not selected:
                if (row.get("error") or "").strip():
                    preproc_errors.append(f"{row.get('input_path', '')}: {row['error'].strip()}")
                continue
            input_path = row.get("input_path", "").strip()
            records.append(
                PreprocRecord(
                    input_path=Path(input_path) if input_path else Path(""),
                    selected_csv=Path(selected),
                    self_check_status=row.get("self_check_status", ""),
                )
            )
    if not records:
        details = "; preprocessing errors: " + " | ".join(preproc_errors[:3]) if preproc_errors else ""
        raise ValueError(f"No rows with output_selected_csv found in {log_path}{details}")
    return records


def read_annotation_log(annotation_log_path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Load an existing annotation log if present.

    Returns:
        A mapping keyed by a (model, output_selected_csv) combination string
        to the stored row dict, and the order of rows as they appeared in the
        log.
    """
    if not annotation_log_path.exists():
        return {}, []

    state: Dict[str, Dict[str, str]] = {}
    order: List[str] = []

    with annotation_log_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected_csv = (row.get("output_selected_csv") or "").strip()
            if not selected_csv:
                continue
            model = (row.get("model") or "").strip()
            row["output_selected_csv"] = selected_csv
            row["model"] = model
            key = make_row_key(model, selected_csv)
            state[key] = row
            order.append(key)
    return state, order


def write_annotation_log(annotation_log_path: Path, order: Iterable[str], state: Dict[str, Dict[str, str]]) -> None:
    annotation_log_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with annotation_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_LOG_HEADERS)
        writer.writeheader()
        for key in order:
            row = state.get(key)
            if not row:
                continue
            writer.writerow({header: row.get(header, "") for header in ANNOTATION_LOG_HEADERS})
            seen.add(key)
        # Persist any additional rows not covered by the supplied order.
        for key, row in state.items():
            if key in seen:
                continue
            writer.writerow({header: row.get(header, "") for header in ANNOTATION_LOG_HEADERS})


def collect_model_entry_keys(state: Dict[str, Dict[str, str]], model: str) -> Tuple[List[str], bool]:
    """Return keys for the requested model and whether any are unfinished."""
    keys: List[str] = []
    has_unfinished = False
    for key, row in state.items():
        row_model = row.get("model", "").strip()
        if row_model != model:
            continue
        keys.append(key)
        if row.get("status") != "completed":
            has_unfinished = True
    return keys, has_unfinished


def find_unfinished_annotation_log(
    log_dir: Path, safe_model: str, model: str
) -> Optional[Tuple[Path, str, Dict[str, Dict[str, str]], List[str]]]:
    """
    Locate the most recent annotation log for ``model`` that still has unfinished entries.
    """
    logger.info("Searching for existing annotation logs for model {}", model)
    search_paths = sorted(log_dir.glob(f"{safe_model}_*_annotation_log.csv"), reverse=True)
    if not search_paths:
        logger.info("No prior annotation logs found in {}", log_dir)
        return None

    for path in search_paths:
        extracted = extract_timestamp_from_log(path.name)
        if not extracted:
            logger.debug("Ignoring non-matching log file {}", path.name)
            continue

        logger.info("Checking annotation log {}", path)
        state, order = read_annotation_log(path)
        model_keys, has_unfinished_for_model = collect_model_entry_keys(state, model)

        if not model_keys:
            logger.info("Log {} has no entries for model {}", path, model)
            continue
        if not has_unfinished_for_model:
            logger.info("Log {} has no unfinished entries for model {}", path, model)
            continue

        logger.warning(
            "Resuming unfinished annotations for model {} using log {} (timestamp {})",
            model,
            path,
            extracted,
        )
        return path, extracted, state, order

    logger.info("No unfinished annotation logs detected for model {}", model)
    return None


def should_skip(record: PreprocRecord, row_state: Dict[str, str]) -> Tuple[bool, str]:
    """
    Determine if the record should be skipped and return a human-readable reason.
    """
    if record.self_check_status.lower() != "pass":
        return True, f"preprocessing self_check_status is '{record.self_check_status}'"
    if not row_state:
        return False, ""
    if row_state.get("status") != "completed":
        return False, ""
    annotated_paths = parse_annotated_csv_field(row_state.get("annotated_csv", ""))
    if annotated_paths and all(path.exists() for path in annotated_paths):
        return True, "annotation already completed and outputs exist"
    return False, ""


async def annotate_csv(
    annotator: GeneralisationAnnotator,
    selected_csv: Path,
    model_name_safe: str,
    run_timestamp: str,
    num_passes: int,
    debug: bool = False,
) -> Tuple[List[Path], int, List[Tuple[int, str]]]:
    data = load_data(selected_csv)
    if data is None:
        raise RuntimeError(f"Failed to load data from {selected_csv}")
    base_prepared = initialise_annotation_columns(data)

    # Always clear final_label so it is recomputed from scratch rather than
    # inherited from an intermediate result CSV.
    base_prepared["final_label"] = pd.array([None] * len(base_prepared), dtype="object")

    # Filter out classifiers that already have an annotation in this CSV
    active_config = {}
    for name, config in NAME2CLASSIFIER_TEMPLATE_PATH_NAME.items():
        if name in base_prepared.columns and base_prepared[name].notna().any() and (base_prepared[name] != "").any():
            logger.info("Skipping annotator '{}' as column is already populated.", name)
            continue
        active_config[name] = config

    annotator.update_classifiers(active_config)

    logprob_cols = [c for c in base_prepared.columns if c.endswith("_logprobs")]
    annotator.validate_logprob_columns(logprob_cols)
    total_sentences = len(base_prepared)
    progress = tqdm(
        total=total_sentences,
        desc=f"Annotating {selected_csv.stem or selected_csv.name}",
        unit="sentence",
        leave=False,
    )

    # Build sentence placeholders with unified interface
    sentence_placeholders = []
    for _, row in base_prepared.iterrows():
        placeholder = {
            "claim": str(row["sentence"]) if "sentence" in row else "",
            "section": str(row["section"]) if "section" in row else "",
            "section_content": str(row["section_content"]) if "section_content" in row else "",
            "two_sent_context": str(row["two_sent_context"]) if "two_sent_context" in row else "",
        }
        for col in base_prepared.columns:
            if col not in placeholder and col != "final_label":
                val = row[col]
                if pd.notna(val):
                    placeholder[col] = val

        # Reconstruct downstream dependencies for classifiers that were skipped
        for name in NAME2CLASSIFIER_TEMPLATE_PATH_NAME:
            if name not in active_config:
                label_val = row.get(name)
                placeholder[f"{name}_label"] = label_val if pd.notna(label_val) else None

                raw_res = row.get(f"{name}_raw_response")
                placeholder[f"{name}_raw_response"] = raw_res if pd.notna(raw_res) else None

                logprobs = row.get(f"{name}_logprobs")
                placeholder[f"{name}_logprobs"] = logprobs if pd.notna(logprobs) else None

        sentence_placeholders.append(placeholder)

    # Build paper context (can be extended with abstract, title, etc. if available in CSV)
    paper_context = {}
    first_row = base_prepared.iloc[0]
    paper_context["abstract"] = str(first_row["section_content"]) if "section_content" in first_row else ""

    async def _run_one_sentence(idx, placeholder):
        try:
            # Phase 1: run about_outcome for all passes concurrently
            phase1_tasks = [
                annotator.run_about_outcome_phase(placeholder, paper_context)
                for _ in range(num_passes)
            ]
            phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

            _sent_errors: List[Tuple[int, str]] = []
            phase1_ok: List[Tuple[int, dict, dict, dict, list]] = []
            for pass_idx, result in enumerate(phase1_results):
                if isinstance(result, BaseException):
                    logger.error(
                        "sentence {:>3d} pass {} phase1 FAILED: {}: {}",
                        idx, pass_idx, type(result).__name__, result,
                    )
                    _sent_errors.append((idx, f"pass {pass_idx} phase1: {type(result).__name__}: {result}"))
                else:
                    ao_result, buffer, uph, errs = result
                    phase1_ok.append((pass_idx, ao_result, buffer, uph, errs))

            # Cross-pass gate: if ANY pass has about_outcome=True, force
            # downstream annotation for ALL passes.
            any_about_true = any(
                ao_result.get("label")
                for _, ao_result, _, _, _ in phase1_ok
            )

            # Phase 2: run downstream classifiers for all successful Phase 1 passes
            phase2_pass_indices = [pass_idx for pass_idx, _, _, _, _ in phase1_ok]
            phase2_coros = [
                annotator.run_downstream_phase(
                    buffer, uph, errs, force_downstream=any_about_true
                )
                for _, _, buffer, uph, errs in phase1_ok
            ]
            phase2_results = await asyncio.gather(*phase2_coros, return_exceptions=True)

            # Reassemble into pass-ordered results (None for Phase 1 failures)
            cleaned_passes = [None] * num_passes
            for pass_idx, result in zip(phase2_pass_indices, phase2_results):
                if isinstance(result, BaseException):
                    logger.error(
                        "sentence {:>3d} pass {} phase2 FAILED: {}: {}",
                        idx, pass_idx, type(result).__name__, result,
                    )
                    _sent_errors.append((idx, f"pass {pass_idx} phase2: {type(result).__name__}: {result}"))
                else:
                    errs = result.pop("_errors", [])
                    for clf_name, msg in errs:
                        _sent_errors.append((idx, f"pass {pass_idx} classifier {clf_name}: {msg}"))
                    cleaned_passes[pass_idx] = result

            progress.update(1)
            return cleaned_passes, _sent_errors
        except Exception as exc:
            logger.error(
                "sentence {:>3d}/{} FAILED unexpectedly: {}: {}",
                idx, total_sentences, type(exc).__name__, exc,
            )
            progress.update(1)
            return None, [(idx, f"{type(exc).__name__}: {exc}")]

    raw_results = await asyncio.gather(
        *(_run_one_sentence(i, p) for i, p in enumerate(sentence_placeholders))
    )

    all_results = []
    all_errors: List[Tuple[int, str]] = []
    for passes, sent_errors in raw_results:
        all_results.append(passes)
        all_errors.extend(sent_errors)
    # all_results[sentence_idx][pass_idx] -> annotation dict  (or None for failures)

    # If about_outcome ran but no sentence got a label in any pass, the run is broken (e.g. a
    # wrong API key or a setting the model rejects). Fail the paper instead of writing empty
    # outputs, so that it is logged as an error and a rerun annotates it again.
    if "about_outcome" in active_config and total_sentences > 0:
        any_labelled = any(
            annotation is not None and (annotation.get("about_outcome") or {}).get("label") is not None
            for sent_passes in all_results
            if sent_passes is not None
            for annotation in sent_passes
        )
        if not any_labelled:
            progress.close()
            first_error = all_errors[0][1] if all_errors else "none recorded"
            raise RuntimeError(
                "No sentence received an about_outcome label: every LLM call failed or gave no Yes/No answer. "
                f"First error: {first_error}"
            )

    # Transpose from sentence-major to pass-major for DataFrame output
    annotations_by_pass: List[List[Optional[Dict[str, Any]]]] = []
    for pass_idx in range(num_passes):
        pass_annotations: List[Optional[Dict[str, Any]]] = []
        for sent_idx in range(total_sentences):
            sent_passes = all_results[sent_idx]
            if sent_passes is None or sent_passes[pass_idx] is None:
                pass_annotations.append(None)
            else:
                pass_annotations.append(sent_passes[pass_idx])
        annotations_by_pass.append(pass_annotations)

    # Find a non-None annotation to discover column schema
    _schema_annotation = None
    for _ann_list in annotations_by_pass:
        for _ann in _ann_list:
            if _ann is not None:
                _schema_annotation = _ann
                break
        if _schema_annotation is not None:
            break

    # Resolve any pending logprob futures before writing to DataFrames.
    # Logprob extraction runs in a background pool so the classifier chain
    # isn't blocked; we collect the results now that all sentences are done.
    _pending_futures = []
    _future_locations = []  # (container_dict, key, sent_idx, pass_idx, role)
    for sent_idx, sent_passes in enumerate(all_results):
        if sent_passes is None:
            continue
        for p_idx, ann in enumerate(sent_passes):
            if ann is None:
                continue
            for role, value in ann.items():
                if value and asyncio.isfuture(value.get("logprobs")):
                    _pending_futures.append(value["logprobs"])
                    _future_locations.append((value, "logprobs", sent_idx, p_idx, role))
    if _pending_futures:
        resolved = await asyncio.gather(*_pending_futures, return_exceptions=True)
        for (container, key, s_idx, p_idx, role), result in zip(_future_locations, resolved):
            if isinstance(result, BaseException):
                all_errors.append((s_idx, f"pass {p_idx} logprob {role}: {result}"))
                container[key] = None
            else:
                container[key] = result

    per_pass_outputs: List[Tuple[int, Any]] = []
    for pass_idx, annotations in enumerate(annotations_by_pass, start=1):
        pass_df = base_prepared.copy()
        if _schema_annotation is not None:
            for role in _schema_annotation:
                for suffix in ("", "_raw_response", "_logprobs"):
                    col = f"{role}{suffix}"
                    if col not in pass_df.columns or pass_df[col].dtype != object:
                        pass_df[col] = pd.array([None] * len(pass_df), dtype="object")
        for idx, annotation_result in enumerate(annotations):
            if annotation_result is None:
                continue
            for role, value in annotation_result.items():
                value = value or {}
                pass_df.loc[idx, role] = value.get("label")
                pass_df.loc[idx, f"{role}_raw_response"] = value.get("raw_response")
                logprobs = value.get("logprobs")
                pass_df.loc[idx, f"{role}_logprobs"] = json.dumps(logprobs) if logprobs is not None else None
        per_pass_outputs.append((pass_idx, pass_df))

    progress.close()

    output_paths: List[Path] = []
    for pass_idx, dataframe in per_pass_outputs:
        output_name = f"{model_name_safe}_{pass_idx}_of_{num_passes}_{run_timestamp}_annotation.csv"
        output_path = selected_csv.parent / output_name
        save_data(dataframe, output_path)
        output_paths.append(output_path)

    return output_paths, len(base_prepared), all_errors


async def _annotate_one_paper(
    record: PreprocRecord,
    *,
    llm,
    model: str,
    safe_model: str,
    run_timestamp: str,
    num_passes: int,
    debug: bool,
    intermediate_by_csv: Dict[str, Dict[str, str]],
    annotation_log_path: Path,
    state: Dict[str, Dict[str, str]],
    order: List[str],
    log_lock: asyncio.Lock,
    paper_progress: tqdm,
) -> None:
    """Annotate a single paper CSV. Designed to run concurrently via asyncio.gather."""
    selected_key = str(record.selected_csv)
    row_key = make_row_key(model, selected_key)

    async with log_lock:
        if row_key in state:
            row_state = state[row_key]
        else:
            row_state = {
                "input_path": str(record.input_path) if record.input_path else "",
                "output_selected_csv": selected_key,
                "status": "pending",
                "annotated_csv": "",
                "model": model,
                "num_sentences": "",
                "started_at": "",
                "finished_at": "",
                "error": "",
            }
            state[row_key] = row_state

        row_state["output_selected_csv"] = selected_key

        if row_key not in order:
            order.append(row_key)

    row_state = state[row_key]
    skip, reason = should_skip(record, row_state)
    if skip:
        logger.info("Skipping {} for model {}: {}", selected_key, model, reason)
        paper_progress.update(1)
        return

    logger.info("Annotating {} for model {}", selected_key, model)

    target_csv = record.selected_csv
    if intermediate_by_csv and selected_key in intermediate_by_csv:
        inter_row = intermediate_by_csv[selected_key]
        annotated_paths = parse_annotated_csv_field(inter_row.get("annotated_csv", ""))
        if annotated_paths and annotated_paths[-1].exists():
            target_csv = annotated_paths[-1]
            logger.warning("Resuming from intermediate output {} for {}", target_csv, selected_key)
        else:
            logger.warning("Intermediate output missing for {}. Starting from scratch.", selected_key)

    async with log_lock:
        row_state["status"] = "in_progress"
        if not row_state.get("started_at"):
            row_state["started_at"] = get_timestamp()
        row_state["finished_at"] = ""
        row_state["error"] = ""
        row_state["model"] = model
        write_annotation_log(annotation_log_path, order, state)

    paper_annotator = GeneralisationAnnotator(llm, NAME2CLASSIFIER_TEMPLATE_PATH_NAME, debug=debug)

    try:
        output_paths, num_sentences, sentence_errors = await annotate_csv(
            paper_annotator, target_csv, safe_model, run_timestamp, num_passes, debug,
        )
    except Exception as exc:
        async with log_lock:
            row_state["status"] = "error"
            row_state["finished_at"] = get_timestamp()
            row_state["error"] = str(exc)
            write_annotation_log(annotation_log_path, order, state)
        logger.exception("Annotation failed for {}", selected_key)
        paper_progress.update(1)
        return

    error_summary = ""
    if sentence_errors:
        error_lines = [f"sent {idx}: {msg}" for idx, msg in sentence_errors[:20]]
        error_summary = f"{len(sentence_errors)} error(s): " + "; ".join(error_lines)
        if len(sentence_errors) > 20:
            error_summary += f" ... and {len(sentence_errors) - 20} more"

    async with log_lock:
        row_state["status"] = "completed"
        row_state["annotated_csv"] = json.dumps([str(path) for path in output_paths])
        row_state["num_sentences"] = str(num_sentences)
        row_state["finished_at"] = get_timestamp()
        if error_summary:
            existing = row_state.get("error", "")
            row_state["error"] = (existing + "\n" + error_summary).strip() if existing else error_summary
        write_annotation_log(annotation_log_path, order, state)
    logger.success(
        "Completed annotation for {} with {} sentences; outputs: {}",
        selected_key, num_sentences, ", ".join(map(str, output_paths)),
    )
    paper_progress.update(1)


async def main() -> int:
    args = parse_args()
    if args.num_passes <= 0:
        raise ValueError("--num-passes must be a positive integer")
    log_csv_path = args.log_csv.resolve()
    log_dir = log_csv_path.parent
    safe_model = sanitize_model_name(args.model)

    logger.info("Reading preprocessing log from {}", log_csv_path)
    records = read_preproc_log(log_csv_path)
    logger.info("Loaded {} records with selected sentences", len(records))

    intermediate_by_csv: Dict[str, Dict[str, str]] = {}
    if args.use_intermediate_log:
        logger.info("Reading intermediate annotation log from {}", args.use_intermediate_log)
        inter_state, _ = read_annotation_log(args.use_intermediate_log)
        for val in inter_state.values():
            if val.get("status") == "completed":
                key_csv = val.get("output_selected_csv", "").strip()
                if key_csv:
                    intermediate_by_csv[key_csv] = val

    found_log = find_unfinished_annotation_log(log_dir, safe_model, args.model)
    if found_log:
        annotation_log_path, run_timestamp, state, order = found_log
    else:
        run_timestamp = get_timestamp()
        annotation_log_path = log_dir / f"{safe_model}_{run_timestamp}_annotation_log.csv"
        state: Dict[str, Dict[str, str]] = {}
        order: List[str] = []
        logger.info(
            "Starting fresh annotation run for model {} with log {}",
            args.model,
            annotation_log_path,
        )

    llm = build_llm(args)
    log_lock = asyncio.Lock()
    paper_progress = tqdm(total=len(records), desc="Annotating CSVs", unit="file")

    shared_kwargs = dict(
        llm=llm,
        model=args.model,
        safe_model=safe_model,
        run_timestamp=run_timestamp,
        num_passes=args.num_passes,
        debug=args.debug,
        intermediate_by_csv=intermediate_by_csv,
        annotation_log_path=annotation_log_path,
        state=state,
        order=order,
        log_lock=log_lock,
        paper_progress=paper_progress,
    )

    semaphore = asyncio.Semaphore(args.max_concurrent_papers)
    logger.info("Concurrency limit: {} papers at a time", args.max_concurrent_papers)

    async def _throttled(record: PreprocRecord) -> None:
        async with semaphore:
            await _annotate_one_paper(record, **shared_kwargs)

    try:
        await asyncio.gather(*(_throttled(record) for record in records))
    finally:
        paper_progress.close()
        llm.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
