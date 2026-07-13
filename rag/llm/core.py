from abc import ABC, abstractmethod
from typing import Dict, Any, List


class LLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    async def async_chat(self, prompt: str) -> Dict[str, Any]:
        """
        Send a chat message to the LLM and get a response.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Dictionary containing the response, typically with a 'message' key
            containing a 'content' field
        """
        pass

    @abstractmethod
    async def async_chat_batch(self, prompt_list: List[str], *, batch_size: int) -> List[Dict[str, Any]]:
        """
        Send a batch of chat messages to the LLM and get a batch of responses.
        """
        pass


# TODO: define abstract class
