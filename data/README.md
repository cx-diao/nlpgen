# Sample NLPeer papers

This folder holds one paper from each of the five NLPeer subsets used to build
NLPGens. The papers are here so you can see how NLPeer is organised and run
NLPGenA on real input. They are not the full corpus.

The papers come from **NLPeer** (Dycke et al., 2023). To get
the full corpus, request it through the official TUdatalib deposits:
[v1](https://tudatalib.ulb.tu-darmstadt.de/items/eeb96857-15ea-4075-9857-2f98a9728145)
and [v2](https://tudatalib.ulb.tu-darmstadt.de/items/d4a4061b-e4e3-4b1e-a90d-d48a3d69e3c0).
NLPeer is intended for the computational study of peer review; please follow its
terms of use.

| Venue in the paper | Path | Paper (camera-ready title) |
|---|---|---|
| ACL '17 | `nlpeer_v0.1/PeerRead-ACL2017/data/12` | Time Expression Analysis and Recognition Using Syntactic Token Types and General Heuristic Rules |
| COLING '20 | `nlpeer_v0.1/COLING2020/data/52` | Learning to Decouple Relations: Few-Shot Relation Classification with Entity-Guided Attention and Confusion-Aware Training |
| ARR '22 | `nlpeer_v0.1/ARR-22/data/00aff1ff…` | What company do words keep? Revisiting the distributional semantics of J.R. Firth & Zellig Harris |
| EMNLP '23 | `nlpeer_new/EMNLP23/EMNLP23/data/2` | Impact of Co-occurrence on Factual Knowledge of Large Language Models |
| EMNLP '24 | `nlpeer_new/ARR-EMNLP-24-v1.1/00a51587…` | Instance-Level Dynamic LoRAs Composition for Cross-Task Generalization |

## Layout

Each paper is a directory with one subdirectory per paper version:

```text
<subset>/data/<paper id>/
  meta.json              paper-level metadata (venue, origin dataset, ...)
  v1/                    earliest version (usually the reviewed submission)
    meta.json            version metadata: title, abstract, licence, ...
    paper.itg.json       the paper as an intertextual graph (ITG)  <- the only file NLPGenA reads
    paper.pdf            PDF of this version (some subsets also have clean.pdf)
    paper.tei            GROBID TEI XML
    reviews.json         peer reviews of this version
  v2/                    later version (usually camera-ready)
    ...
```

The exact files vary by subset: COLING '20 also has `paper.tex`, ACL '17 has
`paper.peerread.json`, EMNLP '23 has `paper.docling.json` and keeps its reviews in
`v1/` without a parsed paper, and EMNLP '24 adds `metareview.json` and
`comments.json`. The two newer subsets are also nested slightly differently, as
the paths in the table show. NLPGenA does not use any review files.

An ITG file is a JSON object with a `nodes` list whose entries (`ntype`: `title`,
`heading`, `abstract`, `paragraph`, `list_item`, `formula`, ...) follow the paper's
reading order. NLPGenA annotates the lowest-numbered version that has a
`paper.itg.json`: `v1` for all samples except EMNLP '23, where it is `v2`.

## Licences

| Venue | Subset licence (NLPeer; Appendix J of our paper) | `license` field in `v1/meta.json` | `license` field in `v2/meta.json` |
|---|---|---|---|
| ACL '17 | CC BY 4.0 | — | CC BY 4.0 |
| COLING '20 | CC BY-NC-SA 4.0 | — | CC BY 4.0 |
| ARR '22 | CC BY-NC-SA 4.0 | ACL copyright notice | CC BY 4.0 |
| EMNLP '23 | CC BY-NC 4.0 | ACL copyright notice | ACL copyright notice |
| EMNLP '24 | CC BY-NC 4.0 | ACL copyright notice | CC BY 4.0 |

The files are redistributed unchanged, under their original terms, with
attribution to NLPeer and to the authors of each paper. The Apache-2.0 licence of
this repository's code does not apply to them.
