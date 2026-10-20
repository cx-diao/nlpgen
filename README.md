# NLPGen: Mapping Generalisation in NLP Research

<p align="center">
  <img src="assets/nlpgen.png" alt="NLPGenX taxonomy" width="700"><br>
  <sub>Cover design inspired by <a href="https://gustavocilleruelo.com/">Gustavo Cilleruelo Calderón</a>.</sub>
</p>

Code and data for the EMNLP 2026 paper
**[How broad is that claim? Mapping Generalisation in NLP Research](https://arxiv.org/abs/2609.14770)**.

- **NLPGenX** is a taxonomy that labels a sentence about a study's outcomes by how far it generalises.
- **NLPGenA** is an LLM-based framework that assigns these labels automatically. This repository contains its code.
- **NLPGens** is a dataset of 180K sentences from 4.5K NLP papers, labelled with NLPGenA. [Request access here](https://forms.cloud.microsoft/e/a0fAPrq4zj).

## Repository layout

```text
nlpgena/
  pipeline/annotate_at_scale/
    produce_path_txt.sh          1. collect paper.itg.json paths (lowest version of each paper)
    preproc_paper_batch.py       2. extract sentences from the selected sections, check coverage
    annotate_paper_batch.py      3. run the classifiers (resumable)
    example.sh                   example run with the paper's vLLM backend (not tested with this release)
    example_api.sh               example run through an OpenAI-compatible API
  annotate/
    classifier/                  one class per annotation task
    prompt_template/             the prompts (Appendix L of the paper)
    orchestrator.py              runs the classifiers for one sentence
    taxonomy.py                  final-label decision tree (Algorithm 1)
rag/llm/                         LLM backends (see "LLM backends" below)
  openai_compat.py               OpenAI-compatible backend (tested)
  vllm.py                        vLLM backend, the paper's configuration (not tested with this release)
nlpgens/                         tools to read NLPGens and NLPGenA output
data/                            one sample NLPeer paper per venue (see data/README.md)
output/                          the NLPGens annotations of those sample papers
tests/
```

## Installation

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) (as
recommended for vLLM) and Python 3.10. Clone the repository and run every
command from its root.

```bash
uv venv --python 3.10
source .venv/bin/activate

# Core: preprocessing, the OpenAI-compatible backend and the NLPGens tools (no GPU needed)
uv pip install -r requirements.txt

# Or the paper's vLLM configuration (not tested with this release), on a machine
# with NVIDIA GPUs (includes the core packages)
uv pip install -r requirements-vllm.txt --torch-backend=auto
```

`--torch-backend=auto` picks the PyTorch build that matches your CUDA driver.
Both sets include spaCy's `en_core_web_sm` model, so there is nothing else to
download. To check the installation, run `uv pip install pytest && pytest`; the
tests need neither a GPU nor an API key.

## NLPGenA

### How it works

The pipeline runs in three stages:

1. `produce_path_txt.sh <input dir> <run dir>/path.txt` finds every
   `paper.itg.json` under the input directory and keeps the lowest-numbered
   version of each paper.
2. `preproc_paper_batch.py` splits each paper into sentences (spaCy) and keeps
   the sections whose heading contains *abstract*, *introduction*, *result*,
   *discussion*, *conclusion* or *ablation*. It also checks that the extracted
   sentences cover the whole Introduction; papers that fail are left out. The
   output mirrors the input layout under the run directory (a leading `data/` is
   dropped).
3. `annotate_paper_batch.py` runs the classifiers on every selected sentence and
   writes one CSV per pass next to that paper's sentence CSV. If interrupted,
   rerun the same command and it resumes.

Each sentence first goes through gating classifier `about_outcome`. Only sentences about the
study's outcomes reach the other classifiers. The final label is then computed
from their yes/no answers (Algorithm 1 in the paper,
[`taxonomy.py`](nlpgena/annotate/taxonomy.py)). Some classifiers have
different names in the code than in the paper:

| Name in the code and CSVs | Name in the paper | Section |
|---|---|---|
| `about_outcome` | `about_outcome` | 4.1 |
| `is_generalised` | `is_generalised` | 4.2 |
| `ref_study` | `ref_study` | 4.3 |
| `ref_study_population` | `ref_study_population` | 4.3 |
| `is_present_tense_main_verb` | `is_present_tense` | 4.4 |
| `contains_quantification_in_main_clause` | `quantified` | 4.4 |
| `is_reporting_clause`, `contains_embedded_sentence`, `get_target_sent` | `removable_framing` (`is_reporting_clause`, `sentence_embedded`, `get_target_sent`) | 4.4, App. D |
| `contains_hedging` | `contains_hedging` | 4.5 |
| `contains_vague_descriptor` | `is_vague` | 4.5 |
| `is_opinion`, `has_citation` | not reported | — |

`is_opinion` and `has_citation` also run, but they are not part of the taxonomy
and were not reported or validated in the paper.

### LLM backends

All LLM calls go through one of the two backends in [`rag/llm/`](rag/llm).
**We have only tested the OpenAI-compatible backend with the code in this
repository.**

| Module | How it is used | Answer-token logprobs | Tested with this release |
|---|---|---|---|
| `openai_compat.py` | `annotate_paper_batch.py --backend openai`: the OpenAI API or any OpenAI-compatible server (hosted providers, Ollama, `vllm serve`) | No; the `*_logprobs` columns are empty | Yes |
| `vllm.py` | `annotate_paper_batch.py --backend vllm`: the configuration that produced NLPGens | Yes, top 20 | No; it may be out of date |

### Quickstart: an OpenAI-compatible API

> [!WARNING]
> This path is for trying NLPGenA on your own papers. It does **not** reproduce
> the paper's annotations, which were produced with the
> [vLLM configuration](#the-papers-configuration-vllm) below.

The `openai` backend works with any server that implements the OpenAI Chat
Completions API: the OpenAI API, hosted model providers, `vllm serve` or
[Ollama](https://ollama.com). You choose the model and, for servers other than
the OpenAI API, the base URL:

```bash
# The OpenAI API; the key is read from OPENAI_API_KEY
export OPENAI_API_KEY=...
MODEL=<model name> OUTPUT_DIR=output/example_openai \
  bash nlpgena/pipeline/annotate_at_scale/example_api.sh

# Any other OpenAI-compatible endpoint, e.g. Ollama on your own machine
# (gpt-oss:20b is the smaller sibling of the model used in the paper)
ollama pull gpt-oss:20b
MODEL=gpt-oss:20b BASE_URL=http://localhost:11434/v1 OUTPUT_DIR=output/example_ollama \
  bash nlpgena/pipeline/annotate_at_scale/example_api.sh
```

This annotates the sample COLING '20 paper (83 selected sentences) with one
pass. The script reads these variables:

| Variable | Meaning |
|---|---|
| `MODEL` | model name, as the endpoint expects it (required) |
| `BASE_URL` | endpoint URL; leave it out for the OpenAI API |
| `OUTPUT_DIR` | folder for all outputs of the run, absolute or relative to the repository root (default `output/example_api`). Use a new folder for each model: a rerun in the same folder resumes the earlier run, and the `nlpgens` tools read only the most recent run in a folder |
| `NUM_PASSES` | number of passes (default 1; the paper used 3) |
| `TEMPERATURE` | sampling temperature (default 0). Set it to `none` to leave it out, for models that only accept their default temperature. |

Check the result with `python -m nlpgens summary <OUTPUT_DIR>`. The script calls
`annotate_paper_batch.py` with `--backend openai`. The options that only apply
to this backend are `--base-url`, `--temperature` (default 0; `none` omits it,
for models that reject the parameter), `--reasoning-effort` and
`--max-concurrent-requests` (default 8).

This backend does not request logprobs, so every `<classifier>_logprobs` column
of its output is empty. The labels and raw responses are not affected.

Each pass makes one call per sentence, plus about eleven more for each sentence
about outcomes (roughly half of them): the sample paper takes about 460 calls
per pass. Keep this in mind with paid APIs.

### The paper's configuration (vLLM)

> [!CAUTION]
> We have not tested the vLLM backend with the code in this repository, and it
> may be out of date. We describe it here because it is the configuration that
> produced NLPGens.

NLPGens was produced with `openai/gpt-oss-120b`, served in-process with vLLM
0.10.2 on NVIDIA A100, H100 and H200 GPUs (80 GB) in a Kubernetes cluster. The
vLLM backend in [`rag/llm/vllm.py`](rag/llm/vllm.py) uses these settings:

| Setting | Value |
|---|---|
| Prompt format | Harmony: system message with reasoning effort `high`; developer message "Always follow the output format requested by the user."; user message = the filled prompt template |
| Sampling | temperature 0, top_p 0.9, top_k 40, repetition penalty 1.1, seed 0, max_tokens 16384 |
| Logprobs | top 20 at the answer token (stored in the `*_logprobs` columns) |
| Parallelism | tensor parallel over all visible GPUs; batch limits chosen by GPU type |
| Passes | 3 (`--num-passes 3`). If `about_outcome` is yes in any pass, all passes run the downstream classifiers |
| Merging | majority vote per classifier, pass 1 breaks ties, final label recomputed (`python -m nlpgens merge`) |

Even at temperature 0, vLLM's output is not fully deterministic, so a rerun will
not match NLPGens exactly. As a rough guide, one pass over a 100-sentence paper
took about 10 minutes on 4×H200.

With the vLLM environment from [Installation](#installation):

```bash
# One NLPeer subset, e.g. COLING 2020
bash nlpgena/pipeline/annotate_at_scale/produce_path_txt.sh \
  /path/to/nlpeer_v0.1/COLING2020 output/coling2020/path.txt
python -m nlpgena.pipeline.annotate_at_scale.preproc_paper_batch \
  --input-path-list-txt output/coling2020/path.txt
python -m nlpgena.pipeline.annotate_at_scale.annotate_paper_batch \
  --log-csv output/coling2020/preproc_log.csv \
  --backend vllm --model openai/gpt-oss-120b --num-passes 3
python -m nlpgens merge output/coling2020 --out output/coling2020/merged.csv
```

`example.sh` runs the same pipeline on the sample COLING '20 paper with
`openai/gpt-oss-20b` and one pass, which fits on a single GPU.

### Annotating your own papers

**If your papers are ITG files** (the format NLPeer uses), put each one at
`<dir>/<paper id>/v1/paper.itg.json` and run the three stages on `<dir>`, as above.

**Otherwise, write the sentences yourself.** Make one CSV per paper with these
columns:

| Column | Content |
|---|---|
| `sentence` | the sentence to annotate (required) |
| `two_sent_context` | up to two preceding sentences followed by the sentence itself, joined with spaces (required; used by `about_outcome`, `is_generalised` and `ref_study_population`) |
| `section` | the section heading (optional; kept in the output) |

Then write a `preproc_log.csv` that lists the CSVs (paths are absolute, or
relative to the repository root, where you run the commands):

```csv
input_path,self_check_status,output_selected_csv
,pass,my_run/paper_a/sentences.csv
,pass,my_run/paper_b/sentences.csv
```

and run the annotation stage and the merge:

```bash
python -m nlpgena.pipeline.annotate_at_scale.annotate_paper_batch \
  --log-csv my_run/preproc_log.csv --backend openai --model <model> --base-url <url> --num-passes 3
python -m nlpgens merge my_run --out my_run/merged.csv
```

The pass CSVs are written next to each input CSV and have the same columns as NLPGens.

## NLPGens

### Access and licence

NLPGens is available on request: **[fill in this form](https://forms.cloud.microsoft/e/a0fAPrq4zj)**.

NLPGens is released under **CC BY-NC-SA 4.0**, the most restrictive licence among
its source subsets. It is built from five NLP subsets of
NLPeer (Dycke et al., 2023). Its
sentences come from the papers in those subsets, and their source licences
still apply (see [`data/README.md`](data/README.md)).

### Contents

NLPGens consists of five annotation runs, one per NLPeer subset. About 21 GB in total.

| Run directory | Venue | Papers | Section sentences | `about_outcome` sentences | Fully labelled |
|---|---|---:|---:|---:|---:|
| `260417_2133_PeerRead-ACL2017_v120b_t0p3_accelerate` | ACL '17 | 133 | 8,915 | 4,055 | 4,055 |
| `260430_1814_COLING2020_v120b_t0p3_rerun` | COLING '20 | 89 | 6,253 | 2,929 | 2,929 |
| `260416_0229_ARR-22_v120b_t0p3_accelerate` | ARR '22 | 473 | 42,451 | 18,609 | 18,607 |
| `merged_260503_1730_EMNLP23_v120b_t0p3` | EMNLP '23 | 1,996 | 133,008 | 67,602 | 67,602 |
| `merged_260503_1717_ARR-EMNLP-24-v1.1_v120b_t0p3` | EMNLP '24 | 1,792 | 204,999 | 87,241 | 87,233 |
| **Total** | | **4,483** | **395,626** | **180,436** | **180,426** |

*Section sentences* are all sentences from the selected sections; NLPGens keeps
them all. *Fully labelled* sentences are about outcomes and have a final label.
The two `merged_*` runs combine a first run with a rerun of part of its papers.

### Layout

```text
<run_dir>/
  path.txt                                  input papers of the run
  preproc_log.csv                           preprocessing result per paper
  openai_gpt-oss-120b_<timestamp>_annotation_log.csv
  nlpeer_v0.1/<subset>/data/<paper id>/<version>/      (nlpeer_new/... for EMNLP '23 and '24)
    paper_all_sentences.csv                 every sentence of the paper    (not in merged_* runs)
    paper_selected_sentences.csv            sentences from the selected sections (not in merged_* runs)
    openai_gpt-oss-120b_1_of_3_<timestamp>_annotation.csv
    openai_gpt-oss-120b_2_of_3_<timestamp>_annotation.csv
    openai_gpt-oss-120b_3_of_3_<timestamp>_annotation.csv
```

The three `_annotation.csv` files are the three passes over the same sentences,
one row per sentence. The logs record absolute paths on our cluster; the
`nlpgens` tools do not need them. [`output/`](output) has the files of one paper
from each run for demostration.

### Columns of a pass CSV

| Column | Content |
|---|---|
| `section`, `sentence` | section heading and sentence |
| `two_sent_context` | up to two preceding sentences and the sentence |
| `section_content` | full text of the section (repeated on every row) |
| one column per classifier (see the [classifier table](#how-it-works)) | `True`, `False`, or empty when the classifier did not run |
| `get_target_sent` | the sentence with its study framing removed; equal to `sentence` when there is no removable frame |
| `final_label` | **outdated, do not use**: computed with an earlier version of the label scheme. Use the `nlpgens` tools, which recompute it |
| `<classifier>_raw_response` | the model's response for that classifier; for yes/no classifiers, its stated reasoning followed by `Answer: Yes/No` |
| `<classifier>_logprobs` | JSON list of the 20 most likely tokens at the answer position, with logprobs. Only the vLLM backend fills it; it is empty with `--backend openai` |
| `notes`, `final_label_raw_response`, `final_label_logprobs` | unused (empty) |

The classifiers after `about_outcome` only run when **at least one** pass labels the
sentence as about outcomes; otherwise their columns are empty.

### Reading NLPGens

```python
from nlpgens import load_nlpgens

df = load_nlpgens("path/to/NLPGens")          # every section sentence, passes merged by majority vote
labelled = df[df["final_label"].notna()]       # the 180,426 fully labelled sentences
labelled.groupby("venue")["final_label"].value_counts(normalize=True)
```

`load_nlpgens` returns one row per sentence with the voted classifier labels,
the recomputed `final_label` and these extra columns:

| Column | Content |
|---|---|
| `run`, `paper_path` | run directory and the paper's path inside it |
| `subset`, `venue`, `paper_id`, `version` | where the paper comes from in NLPeer |
| `row_idx` | row of the sentence in the pass CSVs |
| `source_licence` | licence of the NLPeer subset the sentence comes from |
| `tie_broken` | classifiers whose vote was tied and decided by pass 1 (`;`-separated) |

Options: `runs=[...]` loads only some runs; `with_raw_responses=True`,
`with_logprobs=True` and `with_section_content=True` add the large text columns
(they are skipped by default); `paper_names=True` uses the paper's column names.
`load_run(run_dir)` and `load_paper(paper_dir)` load a single run or paper, and
work the same on the output of your own NLPGenA runs. Loading the full dataset
without the large columns takes about two minutes.

From the command line:

```bash
python -m nlpgens summary path/to/NLPGens/*/          # counts per venue and label distribution
python -m nlpgens merge path/to/NLPGens/*/ --out nlpgens.csv   # one merged table (.parquet needs pyarrow)
```

Running `summary` on the full dataset reproduces the counts in the paper (Tables 1 and 3, Appendix A):

```text
            papers  sentences  about_outcome  fully_labelled
venue
ACL '17        133       8915           4055            4055
ARR '22        473      42451          18609           18607
COLING '20      89       6253           2929            2929
EMNLP '23     1996     133008          67602           67602
EMNLP '24     1792     204999          87241           87233
Total         4483     395626         180436          180426

                             n      %
final_label
non-gen                  57050   31.6
gen/framed/non-generic   98449   54.6
gen/framed/generic        2119    1.2
gen/bare/non-generic      6956    3.9
gen/bare/generic         15852    8.8
Total                   180426  100.0

sentences with a missing label: 11
sentences with a tie-broken label: 94
```

### Notes

- **Reliability.** Table 3 of the paper estimates how likely each final label is
  to be correct, from the human verification of each classifier: `non-gen` 1.00,
  `gen/framed/non-generic` 0.95, `gen/bare/generic` 0.94, `gen/bare/non-generic`
  0.92, `gen/framed/generic` 0.75. You can use these to weight or filter labels.
- **Missing and tied labels.** After merging, 13 of the 2,019,550 labels of the
  ten reported yes/no classifiers are still missing, in 11 sentences from 7
  papers: one sentence has no `about_outcome` label, and the other 10 are about
  outcomes but have no final label. In 94 sentences a label was decided by the
  pass-1 tie-break; the classifiers concerned are listed in `tie_broken`.
- **Annotated version.** Each paper was annotated in its lowest-numbered version
  with an ITG file, usually the reviewed submission rather than the camera-ready
  paper.
- **Prompts.** The prompts in this repository match Appendix L of the paper. Note that `is_reporting_clause.yaml` was corrected **after** NLPGens was
  annotated: in the *MedBERT … achieves* example, the reasoning now reads
  "achieves" *is not* a reporting verb, consistent with its answer (No). To reproduce the reported results in the paper, revert this correction to use the original prompt. We expect the effect of this error on the results to be minimal.

## Citation

```bibtex
@inproceedings{diao2026broad,
  title     = {How broad is that claim? Mapping Generalisation in {NLP} Research},
  author    = {Diao, Chenxin and Stepanova, Nataliya and Allaway, Emily},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.14770},
}
```

## Licence

- Code: [Apache-2.0](LICENSE).
- NLPGens and the sample annotations in [`output/`](output): CC BY-NC-SA 4.0.
- Sample papers in [`data/`](data): their original NLPeer licences (see [`data/README.md`](data/README.md)).
