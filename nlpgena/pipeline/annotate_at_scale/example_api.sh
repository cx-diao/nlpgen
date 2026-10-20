#!/usr/bin/env bash
# Proof-of-concept run on one sample paper through an OpenAI-compatible API.
# This does NOT reproduce the paper's annotations; see the README.
# This backend does not request logprobs: the *_logprobs output columns stay empty.
#
# Required:  MODEL     model name as the endpoint expects it (e.g. gpt-oss:20b on Ollama)
# Optional:  BASE_URL  endpoint URL (e.g. http://localhost:11434/v1); defaults to the OpenAI API
#            OPENAI_API_KEY  API key, if the endpoint needs one
#            OUTPUT_DIR      run directory for all outputs, absolute or relative to the
#                            repository root (default: output/example_api)
#            NUM_PASSES      number of passes (default: 1)
#            TEMPERATURE     sampling temperature (default: 0); "none" leaves it out, for
#                            models that only accept their default temperature
set -euo pipefail

: "${MODEL:?Set MODEL to the model name, e.g. MODEL=gpt-oss:20b}"

# Run from the repository root so all module and data paths resolve consistently.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHONBIN="${PYTHONBIN:-python3}"
SOURCE_PAPER_DIR="data/nlpeer_v0.1/COLING2020/data/52"
OUTPUT_DIR="${OUTPUT_DIR:-output/example_api}"
INPUT_PATH_LIST_TXT="${OUTPUT_DIR}/path.txt"
PREPROC_LOG="${OUTPUT_DIR}/preproc_log.csv"

BACKEND_ARGS=(--backend openai --model "${MODEL}")
if [[ -n "${BASE_URL:-}" ]]; then
  BACKEND_ARGS+=(--base-url "${BASE_URL}")
fi
if [[ -n "${TEMPERATURE:-}" ]]; then
  BACKEND_ARGS+=(--temperature "${TEMPERATURE}")
fi

bash nlpgena/pipeline/annotate_at_scale/produce_path_txt.sh \
  "${SOURCE_PAPER_DIR}" \
  "${INPUT_PATH_LIST_TXT}"

$PYTHONBIN -m nlpgena.pipeline.annotate_at_scale.preproc_paper_batch \
  --input-path-list-txt "${INPUT_PATH_LIST_TXT}"

$PYTHONBIN -m nlpgena.pipeline.annotate_at_scale.annotate_paper_batch \
  --log-csv "${PREPROC_LOG}" \
  "${BACKEND_ARGS[@]}" \
  --num-passes "${NUM_PASSES:-1}" \
  --max-concurrent-papers 1
