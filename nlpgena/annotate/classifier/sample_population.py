import re
from typing import Dict, Any, Optional, Callable

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import PROMPT_TEMPLATE_PATH, ANSWER_PATTERN


class SamplePopulationClassifier(BaseAnnotator):
    """
    Sample Population Classifier that determines whether a claim references the sample population in the study.

    This class handles binary sample population classification using an LLM:
    1. Takes a claim text
    2. Uses a prompt template to classify if it references the sample population in the study
    3. Returns the classification result
    """

    def __init__(
        self,
        llm: LLM,
        *,
        ask_sample_population_str: bool = False,
        prompt_builder: Optional[Callable[[Dict[str, Any]], str]] = None,
        logprob_pattern: Optional[re.Pattern] = ANSWER_PATTERN,
        debug: bool = False,
    ):
        if prompt_builder is None:
            logger.warning("No prompt_builder provided, using default 'v1' template from sample_population.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "sample_population.yaml", "v1")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)
