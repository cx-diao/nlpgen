import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class QuantificationClassifier(BaseAnnotator):
    """
    Quantification Classifier that determines whether a claim includes non-numeric quantifiers.

    This class handles binary quantification classification using an LLM:
    1. Takes a claim text (via ``get_target_sent_label`` placeholder)
    2. Uses a prompt template to classify if it includes non-numeric quantifiers
    3. Returns the classification result with reasoning
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
            logger.warning("No prompt_builder provided, using default 'latest' template from quantification.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "quantification.yaml", "latest")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)
