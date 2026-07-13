import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from substantiator.annotate.core import BaseAnnotator
from substantiator.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class IsReportingClauseClassifier(BaseAnnotator):
    """
    Is Reporting Clause Classifier that determines whether the sentence contains a reporting verb.

    This class handles binary classification using an LLM:
    1. Takes a claim text
    2. Uses a prompt template to classify if the sentence contains a reporting verb
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
            logger.warning("No prompt_builder provided, using default 'latest' template from is_reporting_clause.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "is_reporting_clause.yaml", "latest")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)
