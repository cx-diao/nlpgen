import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH

_BRACKET_RE = re.compile(r"[()\[\]]")


class CitationClassifier(BaseAnnotator):
    """
    Citation classifier that determines whether a sentence contains an
    explicit citation marker (e.g. author-year or numeric references).

    Before querying the LLM, a fast heuristic check is applied: if the
    sentence contains no round or square bracket character (``(``, ``)``,
    ``[``, ``]``), it is impossible for a standard citation marker to be
    present, so the classifier short-circuits to False.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        prompt_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        logprob_pattern: Optional[re.Pattern] = ANSWER_PATTERN,
        debug: bool = False,
    ):
        if prompt_builder is None:
            logger.warning(
                "No prompt_builder provided, using default 'latest' template from citation.yaml"
            )
            prompt_builder = prompt_builder_from_yaml(
                PROMPT_TEMPLATE_PATH / "citation.yaml", "latest"
            )

        super().__init__(
            llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug
        )

    @staticmethod
    def _has_bracket(text: str) -> bool:
        """Return True if *text* contains at least one bracket: ( ) [ ]."""
        return bool(_BRACKET_RE.search(text))

    @staticmethod
    def _no_citation_result() -> Dict[str, Any]:
        return {
            "label": False,
            "raw_response": "Skipped: no bracket character",
            "logprobs": None,
        }

    async def annotate(self, placeholder2value: Dict[str, Any]) -> Dict[str, Any]:
        assert isinstance(placeholder2value, dict), "Input must be a dictionary of placeholders."
        claim = placeholder2value["claim"]

        if not self._has_bracket(claim):
            if self.debug:
                logger.info(f"No bracket in sentence — short-circuiting to False: {claim}")
            return self._no_citation_result()

        return await super().annotate(placeholder2value)

    async def annotate_batch(
        self, placeholder2value_list: List[Dict[str, Any]], *, batch_size: int = 4, debug: bool = False
    ) -> List[Dict[str, Any]]:
        assert isinstance(placeholder2value_list, list), "Input must be a list of dictionaries."

        results: List[Optional[Dict[str, Any]]] = [None] * len(placeholder2value_list)
        to_send: List[tuple[int, Dict[str, Any]]] = []

        for idx, p2v in enumerate(placeholder2value_list):
            assert isinstance(p2v, dict), "Each item must be a dictionary."
            if not self._has_bracket(p2v["claim"]):
                results[idx] = self._no_citation_result()
            else:
                to_send.append((idx, p2v))

        if to_send:
            llm_results = await super().annotate_batch(
                [p2v for _, p2v in to_send], batch_size=batch_size, debug=debug
            )
            for (orig_idx, _), llm_result in zip(to_send, llm_results):
                results[orig_idx] = llm_result

        return results
