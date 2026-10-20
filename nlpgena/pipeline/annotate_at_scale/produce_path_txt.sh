#!/usr/bin/env bash

# Generate a list of all .itg.json files located under version directories,
# preferring the lowest version number available for each paper.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <input_dir> <output_txt> [seed]" >&2
  exit 1
fi

input_dir=$1
output_file=$2
seed=${3:-42}

if [[ ! -d $input_dir ]]; then
  echo "Input directory '$input_dir' does not exist." >&2
  exit 1
fi

output_dir=$(dirname "$output_file")
mkdir -p "$output_dir"

find "$input_dir" -type f -name '*.itg.json' | awk -F/ '{
  n = NF
  ver = $(n-1)
  gsub(/^v/, "", ver)
  ver_num = ver + 0

  key = ""
  for (i = 1; i <= n-2; i++) key = key (i > 1 ? "/" : "") $i

  if (!(key in best) || ver_num < best[key]) {
    best[key] = ver_num
    path[key] = $0
  }
}
END {
  for (k in path) print path[k]
}' | awk -v seed="$seed" 'BEGIN{srand(seed)} {print rand(), $0}' \
   | sort -n | cut -d' ' -f2- > "$output_file"

echo "Wrote paths to $output_file (seed=$seed)"