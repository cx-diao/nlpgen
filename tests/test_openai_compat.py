import asyncio
from types import SimpleNamespace

import pytest

from rag.llm.openai_compat import OpenAICompatModel


class FakeCompletions:
    def __init__(self, delay=0.0):
        self.requests = []
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def create(self, **request):
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        message = SimpleNamespace(content="Reasoning: ...\nAnswer: Yes")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _model(completions, **kwargs):
    model = OpenAICompatModel("some-model", base_url="http://localhost:1/v1", **kwargs)
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return model


def test_request_and_response_shape():
    completions = FakeCompletions()
    model = _model(completions, reasoning_effort="high")
    response = asyncio.run(model.async_chat("Is it?", return_logprobs=True, logprob_pattern=None))

    assert response == {"message": {"content": "Reasoning: ...\nAnswer: Yes"}, "answer_logprobs": None}
    request = completions.requests[0]
    assert request["model"] == "some-model"
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["messages"][1]["content"] == "Is it?"
    assert request["temperature"] == 0.0
    assert request["reasoning_effort"] == "high"


def test_temperature_can_be_omitted():
    completions = FakeCompletions()
    asyncio.run(_model(completions, temperature=None).async_chat("Is it?"))
    assert "temperature" not in completions.requests[0]
    assert "reasoning_effort" not in completions.requests[0]


def test_concurrency_is_bounded():
    completions = FakeCompletions(delay=0.01)
    model = _model(completions, max_concurrency=3)
    results = asyncio.run(model.async_chat_batch([f"q{i}" for i in range(10)], batch_size=1))
    assert len(results) == 10
    assert completions.max_in_flight == 3


def test_api_key_required_for_openai_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatModel("some-model")
