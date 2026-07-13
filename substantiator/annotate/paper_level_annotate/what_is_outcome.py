import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from substantiator.annotate.core import BaseAnnotator
from substantiator.annotate.utils import PROMPT_TEMPLATE_PATH


class WhatIsOutcomeAnnotator(BaseAnnotator):
    """
    What Is Outcome Annotator that determines the outcome of the study.

    This class handles outcome classification using an LLM:
    1. Takes the abstract of the paper
    2. Returns the outcome description
    """

    def __init__(
        self,
        llm: LLM,
        *,
        prompt_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        logprob_pattern: Optional[re.Pattern] = None,
        debug: bool = False,
    ):
        if prompt_builder is None:
            logger.warning("No prompt_builder provided, using default 'v1' template from what_is_outcome.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "what_is_outcome.yaml", "v1")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)

    def _parse_outcome_from_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the annotation from the LLM response.

        Args:
            response: The raw response from the LLM

        Returns:
            Dictionary containing the annotation result with appropriate keys
        """
        raw_response = (response or "").strip()