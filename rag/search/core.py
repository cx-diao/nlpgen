from abc import ABC, abstractmethod
from pathlib import Path
import yaml
from typing import Any, Iterable


class BaseSearch(ABC):
    def __init__(
        self,
        config_path: Path | str,
        force_reindex: bool,
        *,
        corpus_iterable: Iterable[str] = None,
        debug: bool = False,
    ):
        self.debug = debug
        self._load_config(config_path)

        assert "index_path" in self.config, "index_path must be in config."
        self.config["index_path"] = Path(self.config["index_path"])
        if force_reindex or not self.config["index_path"].exists():
            assert corpus_iterable is not None, "corpus_iterable must be provided to create index."
            self.build_index_from_iterable(corpus_iterable)

    @abstractmethod
    def _load_config(self, config_path: Path | str) -> dict:
        pass

    @abstractmethod
    def build_index_from_iterable(self, corpus: Iterable):
        pass

    @abstractmethod
    def retrieve(self, query: str, **kwargs):
        pass
