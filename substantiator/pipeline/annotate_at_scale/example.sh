#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root so all module and data paths resolve consistently.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHONBIN="${PYTHONBIN:-python3}"
SOURCE_PAPER_DIR="data/nlpeer_v0.1/COLING2020/data/52"
INPUT_PATH_LIST_TXT="output/reviewer_demo/path.txt"
PREPROC_LOG="$(dirname "${INPUT_PATH_LIST_TXT}")/preproc_log.csv"

bash substantiator/pipeline/annotate_at_scale/produce_path_txt.sh \
  "${SOURCE_PAPER_DIR}" \
  "${INPUT_PATH_LIST_TXT}"

$PYTHONBIN -m substantiator.pipeline.annotate_at_scale.preproc_paper_batch \
  --input-path-list-txt "${INPUT_PATH_LIST_TXT}"

$PYTHONBIN -m substantiator.pipeline.annotate_at_scale.annotate_paper_batch \
  --log-csv "${PREPROC_LOG}" \
  --model "openai/gpt-oss-20b" \
  --num-passes 1 \
  --max-concurrent-papers 1
