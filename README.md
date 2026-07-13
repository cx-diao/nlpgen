# Substantiator annotation pipeline

This repository is a reviewer-facing snapshot of the Substantiator annotation
pipeline. It is intended primarily for inspecting the pipeline's functions,
code organization, sample inputs, and sample outputs.

## Reviewer note

The annotation runtime was developed and tested on CUDA-enabled, multi-GPU
Kubernetes worker nodes. It depends on a compatible NVIDIA driver, CUDA
toolchain, PyTorch/vLLM stack, model weights, and sufficient GPU memory. It may
not run directly on a local workstation or a differently configured compute
node.

`substantiator/pipeline/annotate_at_scale/example.sh` is therefore a reference
entry point rather than a guarantee of a portable local execution environment.
The preprocessing code and checked-in samples can still be reviewed without
running the GPU annotation stage.

## Pipeline functions

The batch pipeline has three main stages:

1. `produce_path_txt.sh` searches a dataset directory for `*.itg.json` papers,
   groups versioned copies of each paper, and selects the lowest numbered
   version.
2. `preproc_paper_batch.py` extracts sections and sentences from each
   `paper.itg.json`, checks Introduction coverage, and writes both complete and
   section-selected sentence tables. Selected sections include Abstract,
   Introduction, Results, Discussion, Conclusion, and Ablation sections.
3. `annotate_paper_batch.py` applies the prompt-based sentence classifiers,
   repeats annotation passes when requested, and records resumable progress.
   The annotations cover discourse role, embedded and reporting clauses,
   generality, study and population references, tense, quantification, hedging,
   vague descriptions, opinion, and citation information.

## Code structure

```text
substantiator/
  pipeline/annotate_at_scale/
    example.sh                 Reference end-to-end invocation
    produce_path_txt.sh        Version-aware ITG path collection
    preproc_paper_batch.py     Batch preprocessing and coverage log
    annotate_paper_batch.py    Concurrent, resumable annotation
  pipeline/one_paper_demo/
    produce_annotation_form.py Section extraction and sentence selection
    check_introduction_coverage.py
    annotate_one_paper.py      Classifier and prompt configuration
  annotate/
    classifier/                Sentence classifier implementations
    prompt_template/           YAML prompt templates
    orchestrator.py            Annotation phase orchestration
rag/
  llm/vllm.py                  vLLM model adapter
data/                          Five sample paper inputs
output/                        Five sample annotation runs
```

## Sample input data

`data/` contains five papers from different review datasets:

- ARR-EMNLP-24: *Instance-Level Dynamic LoRAs Composition for Cross-Task
  Generalization*
- EMNLP23: *Impact of Co-occurrence on Factual Knowledge of Large Language
  Models*
- ARR-22: *What kind of company do words keep?*
- COLING2020: *Learning to Decouple Relations: Few-Shot Relation
  Classification with Entity-Guided Attention and Confusion-Aware Training*
- PeerRead ACL2017: *Time Expression Analysis and Recognition Using Syntactic
  Token Types and General Heuristic Rules*

The datasets use versioned paper directories with this general structure:

```text
data/<dataset path>/<paper id>/
  meta.json
  v1/
    paper.itg.json
    meta.json
    paper.pdf
    paper.tei
    reviews.json
  v2/
    ...
```

The exact companion files vary by dataset. Only `paper.itg.json` is consumed by
this pipeline; PDFs, metadata, reviews, TEI files, and other files are retained
to show the surrounding dataset structure. An ITG file is a JSON object with a
`nodes` list whose entries represent headings, abstracts, paragraphs, list
items, formulas, and related document elements.

When multiple versions exist, the path collection stage selects the lowest
numbered version. EMNLP23 contains only a usable `v2` ITG sample, while the
other samples include `v1`.

## Sample output data

`output/` contains **five top-level directories representing five different
annotation runs**, not five individual output files:

- `merged_260503_1717_ARR-EMNLP-24-v1.1_v120b_t0p3`
- `merged_260503_1730_EMNLP23_v120b_t0p3`
- `260416_0229_ARR-22_v120b_t0p3_accelerate`
- `260417_2133_PeerRead-ACL2017_v120b_t0p3_accelerate`
- `260430_1814_COLING2020_v120b_t0p3_rerun`

A complete run has the following per-paper structure:

```text
output/<run>/<dataset-relative paper path>/<version>/
  paper_all_sentences.csv
  paper_selected_sentences.csv
  openai_gpt-oss-120b_1_of_3_<timestamp>_annotation.csv
  openai_gpt-oss-120b_2_of_3_<timestamp>_annotation.csv
  openai_gpt-oss-120b_3_of_3_<timestamp>_annotation.csv
```

The three annotation files are repeated passes from the same run. ARR-22,
PeerRead ACL2017, and COLING2020 retain both preprocessing CSVs and all three
annotation CSVs. The two `merged_*` directories retain only their three
annotation-pass CSVs. Historical bookkeeping files (`path.txt`,
`preproc_log.csv`, and annotation logs) are not included.

These checked-in annotations were produced with `openai/gpt-oss-120b` using
three passes. They are provided to demonstrate the annotation schema and
expected directory layout.

## Reference invocation

For a compatible Kubernetes GPU environment, the minimal reference command is:

```bash
bash substantiator/pipeline/annotate_at_scale/example.sh
```

It uses the included COLING2020 paper `52`, `openai/gpt-oss-20b`, and one
annotation pass. Runtime dependencies are listed in `requirements.txt` and
`rag/requirements.txt`; spaCy's `en_core_web_sm` model must also be installed.
