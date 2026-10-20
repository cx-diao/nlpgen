import re
from typing import Dict, Any, Callable, Optional, List

from loguru import logger

from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from nlpgena.annotate.core import BaseAnnotator
from nlpgena.annotate.utils import ANSWER_PATTERN, PROMPT_TEMPLATE_PATH


class GetTargetSentAnnotator(BaseAnnotator):
    """
    Target Sentence Annotator that extracts the embedded sentence (complement clause)
    from a reporting clause.

    This class handles annotation using an LLM:
    1. Takes a claim text
    2. Uses a prompt template to extract the reported sentence
    3. Returns the extracted sentence with reasoning
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
            logger.warning("No prompt_builder provided, using default 'latest' template from get_target_sent.yaml")
            prompt_builder = prompt_builder_from_yaml(PROMPT_TEMPLATE_PATH / "get_target_sent.yaml", "latest")

        super().__init__(llm, prompt_builder=prompt_builder, logprob_pattern=logprob_pattern, debug=debug)

    async def annotate(self, placeholder2value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate the input using the LLM, but only if it's a reporting clause with an embedded sentence.
        Otherwise, return the original claim.
        """
        is_reporting = placeholder2value.get("is_reporting_clause_label", False)
        has_embedded = placeholder2value.get("contains_embedded_sentence_label", False)

        if not (is_reporting and has_embedded):
            return {
                "label": placeholder2value.get("claim", ""),
                "raw_response": "",
            }

        return await super().annotate(placeholder2value)

    async def annotate_batch(
        self, placeholder2value_list: List[Dict[str, Any]], *, batch_size: int = 4, debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Annotate a batch, filtering locally any items that don't need the LLM.
        """
        results = [None] * len(placeholder2value_list)
        llm_indices = []
        llm_inputs = []

        for i, p in enumerate(placeholder2value_list):
            is_reporting = p.get("is_reporting_clause_label", False)
            has_embedded = p.get("contains_embedded_sentence_label", False)
            if is_reporting and has_embedded:
                llm_indices.append(i)
                llm_inputs.append(p)
            else:
                results[i] = {
                    "label": p.get("claim", ""),
                    "raw_response": "",
                }

        if llm_inputs:
            llm_results = await super().annotate_batch(llm_inputs, batch_size=batch_size, debug=debug)
            for idx, res in zip(llm_indices, llm_results):
                results[idx] = res

        return results

    def _parse_outcome_from_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the extracted sentence from the LLM response.

        Expected response format::

            Reasoning: <analysis text>
            Answer: <extracted sentence>
        """
        raw_response = (response or "").strip()

        matches = list(ANSWER_PATTERN.finditer(raw_response))

        if matches:
            match = matches[-1]
            target_sent = raw_response[match.end():].strip().lstrip(":.-*_ ").strip()
            reasoning = raw_response[:match.start()].strip()
        else:
            logger.warning(
                f"Could not locate an 'Answer' marker in the LLM response. "
                f"Using last line as target sentence. Response: {raw_response}"
            )
            lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
            target_sent = lines[-1] if lines else ""
            reasoning = "\n".join(lines[:-1]) if len(lines) > 1 else ""

        return {"label": target_sent, "reasoning": reasoning, "raw_response": raw_response}
