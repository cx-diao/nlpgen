import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import PROMPT_TEMPLATE_PATH, ANSWER_PATTERN


class HedgingClassifier(BaseAnnotator):
    """
    Hedging Classifier that determines whether a claim contains explicit hedging.

    This class handles binary hedging classification using an LLM:
    1. Takes a claim text
    2. Uses the `hedging.yaml` prompt template to classify explicit hedging
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
            logger.warning("No prompt_builder provided, using default 'latest' template from hedging.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "hedging.yaml", "latest")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)

