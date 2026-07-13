from pathlib import Path
import pickle
import yaml
from abc import abstractmethod
from typing import Iterable

import spacy
import zss
from datasketch import MinHash, MinHashLSH
from loguru import logger

from rag.search.core import BaseSearch


def pos_ngrams(doc, n):
    """Return list of overlapping POS n-grams with BOS/EOS markers."""
    tags = ["<BOS>"] + [tok.pos_ for tok in doc] + ["<EOS>"]
    return ["_".join(tags[i : i + n]) for i in range(len(tags) - n + 1)]


def to_tree(tok):
    return zss.Node(tok.dep_, [to_tree(c) for c in tok.children])


def doc_tree(text, nlp):
    doc = nlp(text)
    return zss.Node("DOC", [to_tree(sent.root) for sent in doc.sents])


class SyntaticSearch(BaseSearch):
    def __init__(
        self,
        config_path: Path | str = "rag/search/config/syntatic.yaml",
        force_reindex: bool = True,
        *,
        corpus_iterable: Iterable[str] = None,
        debug=False,
    ):
        super().__init__(config_path, force_reindex, corpus_iterable=corpus_iterable, debug=debug)
        self._load_index()

    def _load_config(self, config_path: Path | str):
        self.config = {
            "index_path": None,
            "lsh_threshold": None,
            "ngram": None,
            "spacy_model": None,
            "num_perm": None,
            "progress_every": None,
        }
        with Path(config_path).open() as f:
            yaml_config = yaml.safe_load(f)
        self.config.update(yaml_config)
        assert all(v is not None for v in self.config.values()), "Missing configuration parameter values."

        self.nlp = spacy.load(self.config["spacy_model"])
        self.lsh = None
        self.trees = None
        self.texts = None

    def _load_index(self):
        if self.config["index_path"].exists():
            with self.config["index_path"].open("rb") as f:
                self.lsh, self.trees, self.texts, self.config["ngram"] = pickle.load(f)
        else:
            self.lsh = None
            self.trees = None
            self.texts = None

    def build_index_from_iterable(self, corpus: Iterable[str]):
        lsh = MinHashLSH(threshold=self.config["lsh_threshold"], num_perm=self.config["num_perm"])
        trees, texts = [], []
        for idx, sentence in enumerate(corpus):
            doc = self.nlp(sentence)
            mh = MinHash(num_perm=self.config["num_perm"])
            for gram in pos_ngrams(doc, self.config["ngram"]):
                mh.update(gram.encode())
            lsh.insert(idx, mh)
            trees.append(doc_tree(sentence, self.nlp))
            texts.append(sentence)
            if idx and idx % self.config["progress_every"] == 0:
                logger.info(f"processed {idx:,} sentences")
        with open(self.config["index_path"], "wb") as f:
            pickle.dump((lsh, trees, texts, self.config["ngram"]), f)
        logger.info(f"Indexed {len(texts):,} sentences.")
        logger.info(f"Index is saved to {self.config['index_path']}")
        self.lsh = lsh
        self.trees = trees
        self.texts = texts

    @staticmethod
    def structural_similarity(t1, t2):
        dist = zss.simple_distance(t1, t2)
        return 1 / (1 + dist)

    def retrieve(self, query: str, topk: int = 5, debug: bool = False):
        assert self.lsh is not None and self.trees is not None and self.texts is not None, "Index not loaded."
        q_mh = MinHash(num_perm=self.config["num_perm"])
        for gram in pos_ngrams(self.nlp(query), self.config["ngram"]):
            q_mh.update(gram.encode())
        cand_ids = self.lsh.query(q_mh)
        q_tree = doc_tree(query, self.nlp)
        scored = [(self.structural_similarity(q_tree, self.trees[i]), self.texts[i]) for i in cand_ids]
        scored.sort(reverse=True, key=lambda x: x[0])
        if debug or self.debug:
            logger.info(f"retrieved examples: {scored[:topk]}")
        return scored[:topk]
