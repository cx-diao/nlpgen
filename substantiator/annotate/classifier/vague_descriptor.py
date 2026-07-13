import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from substantiator.annotate.core import BaseAnnotator
from substantiator.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class VagueDescriptorClassifier(BaseAnnotator):
    """
    Vague Descriptor Classifier that determines whether a claim includes
    an under-specified adjective or adverb.

    This class handles binary classification using an LLM:
    1. Takes a claim text
    2. Uses the `vague_descriptor.yaml` prompt template to classify
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
            logger.warning(
                "No prompt_builder provided, using default 'latest' template from vague_descriptor.yaml"
            )
            prompt_builder = prompt_builder_from_yaml(
                PROMPT_TEMPLATE_PATH / "vague_descriptor.yaml", "latest"
            )

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)

