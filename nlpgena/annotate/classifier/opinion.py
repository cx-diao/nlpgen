import re
from typing import Any, Callable, Dict, Optional

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class OpinionClassifier(BaseAnnotator):
    """
    Opinion classifier that determines whether a sentence expresses an opinion
    rather than a factual claim.
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
                "No prompt_builder provided, using default 'latest' template from opinion.yaml"
            )
            prompt_builder = prompt_builder_from_yaml(
                PROMPT_TEMPLATE_PATH / "opinion.yaml", "latest"
            )

        super().__init__(
            llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug
        )

