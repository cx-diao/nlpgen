import re
from abc import ABC
from typing import Dict, Any, List, Callable, Optional

from loguru import logger

from rag.generate.core import LLMGenerator
from rag.llm.core import LLM
from rag.prompt_builder.core import prompt_builder_from_yaml
from substantiator.annotate.utils import YESNO_LOGPROB_PATTERN


class BaseAnnotator(ABC):
    """
    Base class for all annotators that determines binary annotation using an LLM.

    This class handles the common functionality for binary annotation:
    1. Takes a claim text
    2. Uses a prompt template to annotate
    3. Returns the annotation result

    Subclasses can override _parse_outcome_from_response for custom parsing, but the default
    implementation already handles common yes/no style responses.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        prompt_builder: Callable[[Dict[str, Any]], str],
        logprob_pattern: Optional[re.Pattern] = None,
        debug: bool = False,
    ):
        """
        Initialise the BaseAnnotator.

        Args:
            llm: LLM instance that has an async_chat method.
            prompt_builder: Function that takes a placeholder dictionary and returns the complete prompt.
            logprob_pattern: Optional regex forwarded to the LLM backend so it
                can locate the answer token within the final-channel output.
            debug: Whether to enable debug logging.
        """
        self.llm = llm
        self.debug = debug
        self.prompt_builder = prompt_builder
        self.logprob_pattern = logprob_pattern

        return_logprobs = logprob_pattern is not None
        self.generator = LLMGenerator(
            prompt_builder=self.prompt_builder,
            llm=llm,
            logprob_pattern=self.logprob_pattern,
            return_logprobs=return_logprobs,
            debug=debug,
        )

    def _attach_metadata(self, parsed: Dict[str, Any], generation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure every parsed result carries the original raw response plus any available logprob metadata.
        """
        parsed = parsed or {}
        if "raw_response" not in parsed:
            parsed["raw_response"] = generation.get("content")
        parsed["logprobs"] = generation.get("answer_logprobs")
        return parsed

    async def annotate(self, placeholder2value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Annotate the input using the LLM.

        Args:
            placeholder2value: Dictionary containing placeholders required by the prompt template (e.g., {'claim': ...})

        Returns:
            Dictionary containing the annotation result with keys determined by _parse_outcome_from_response
        """
        assert isinstance(placeholder2value, dict), "Input must be a dictionary of placeholders."
        # Remove the hardcoded "claim" requirement - let the prompt template determine required fields

        # Generate response using LLMGenerator
        generation = await self.generator.generate(placeholder2value)
        content = generation.get("content", "")

        if self.debug:
            logger.info(f"LLM response: {content}")

        # Parse the response using the abstract method
        result = self._parse_outcome_from_response(content)

        return self._attach_metadata(result, generation)

    def _parse_outcome_from_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the annotation from the LLM response.

        Expects the LLM to produce output containing an ``Answer`` marker
        (optionally bold), e.g.::

            Reasoning: <analysis text>
            Answer: Yes

        The last occurrence of the marker is used so that the word "answer"
        appearing earlier in reasoning text does not cause a false match.

        Args:
            response: The raw response from the LLM

        Returns:
            Dictionary with keys ``label`` (bool | None), ``reasoning`` (str), and
            ``raw_response`` (str).  ``label`` is *None* when no yes/no token
            can be extracted at all.
        """
        raw_response = (response or "").strip()

        assert self.logprob_pattern is not None, (
            "logprob_pattern must be set to use the default _parse_outcome_from_response; "
            "override this method if your annotator does not use a logprob pattern."
        )
        matches = list(self.logprob_pattern.finditer(raw_response))

        if matches:
            match = matches[-1]
            answer_text = raw_response[match.end() :].strip().lstrip(":.-*_ ").lower()
        else:
            logger.warning(
                f"Could not locate an 'Answer' marker in the LLM response. "
                f"Falling back to yes/no search. Response: {raw_response}"
            )
            answer_text = raw_response.lower()

        if answer_text.startswith("yes"):
            label = True
        elif answer_text.startswith("no"):
            label = False
        else:
            # Last resort: search for any yes/no token anywhere in the text
            yesno_matches = list(YESNO_LOGPROB_PATTERN.finditer(answer_text))
            if yesno_matches:
                last_yesno = yesno_matches[-1].group(1).lower()
                label = last_yesno == "yes"
                logger.warning(f"Extracted label from last yes/no token ('{last_yesno}') in response: {raw_response}")
            else:
                logger.warning(f"No yes/no token found in LLM response. Defaulting to None. Response: {raw_response}")
                label = None

        return {"label": label, "raw_response": raw_response}

    async def __call__(self, placeholder2value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to call annotate with a dictionary of placeholders.

        Args:
            placeholder2value: Dictionary containing placeholders required by the prompt template

        Returns:
            Dictionary containing the annotation result
        """
        return await self.annotate(placeholder2value)

    async def annotate_claim(self, claim: str) -> Dict[str, Any]:
        """
        Convenience method for the common case where only 'claim' is provided.

        Args:
            claim: The claim text to annotate

        Returns:
            Dictionary containing the annotation result
        """
        return await self.annotate({"claim": claim})

    async def annotate_batch(
        self, placeholder2value_list: List[Dict[str, Any]], *, batch_size: int = 4, debug: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Annotate a batch of claims.

        Args:
            placeholder2value_list: List of dictionaries, each containing placeholders required by the prompt template (e.g., {'claim': ...})

        Returns:
            List of dictionaries, each containing the annotation result
        """
        assert isinstance(placeholder2value_list, list), "Input must be a list of dictionaries."
        for placeholder2value in placeholder2value_list:
            assert isinstance(placeholder2value, dict), "Each item must be a dictionary."
            # Remove the hardcoded "claim" requirement - let the prompt template determine required fields

        # Generate batch responses using LLMGenerator
        generations = await self.generator.generate_batch(placeholder2value_list, batch_size=batch_size, debug=debug)

        results = []
        for generation in generations:
            content = generation.get("content", "")
            result = self._parse_outcome_from_response(content)
            results.append(self._attach_metadata(result, generation))
        return results
