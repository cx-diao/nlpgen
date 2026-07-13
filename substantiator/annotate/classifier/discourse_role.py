import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from substantiator.annotate.core import BaseAnnotator
from substantiator.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class DiscourseRoleClassifier(BaseAnnotator):
    """
    Discourse Role Classifier that determines whether a claim is about OUTCOME or not.

    This class handles binary discourse role classification using an LLM:
    1. Takes a claim text and surrounding section content
    2. Uses a prompt template to classify if it's about OUTCOME
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
            logger.warning("No prompt_builder provided, using default 'latest' template from discourse_role.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "discourse_role.yaml", "latest")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)
