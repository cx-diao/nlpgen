import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class GeneralityClassifier(BaseAnnotator):
    """
    Generality Classifier that determines whether a claim is generalised or not generalised.

    This class handles binary generality classification using an LLM:
    1. Takes a claim text (with context)
    2. Uses a prompt template to classify if it's generalised or not generalised
    3. Returns the classification result
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
            logger.warning("No prompt_builder provided, using default 'latest' template from generality.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "generality.yaml", "latest")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)

    def _parse_outcome_from_response(self, response: str) -> Dict[str, Any]:
        raw_response = (response or "").strip()

        match = self.logprob_pattern.search(raw_response)

        if match:
            answer_text = raw_response[match.end() :].strip().lower()
        else:
            logger.warning(
                f"Could not locate an 'Answer:' marker in the LLM response. "
                f"Falling back to keyword search. Response: {raw_response}"
            )
            answer_text = raw_response.lower()

        if answer_text.startswith("not generalised") or answer_text.startswith("not generalized"):
            label = False
        elif answer_text.startswith("generalised") or answer_text.startswith("generalized"):
            label = True
        elif "not generalised" in answer_text or "not generalized" in answer_text:
            label = False
        elif "generalised" in answer_text or "generalized" in answer_text:
            label = True
        elif "not" in answer_text:
            label = False
            logger.warning(
                f"Inferred 'not generalised' from presence of 'not' in response: {raw_response}"
            )
        else:
            logger.warning(f"Unclear answer in LLM response. Defaulting to None. Response: {raw_response}")
            label = None

        return {
            "label": label,
            "raw_response": raw_response,
        }
