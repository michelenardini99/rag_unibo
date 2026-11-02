"""Shared setup for the unit tests.

The tests cover the logic that runs without GPUs, downloaded models or
running services (vLLM, Qdrant), so they can run in CI. Settings are pinned
through environment variables, which take precedence over a local `.env`,
so results don't depend on how a developer's machine is configured.
"""

import os
from types import SimpleNamespace

os.environ["EMBEDDINGS_DEVICE_ID"] = "0"
os.environ["GENERATION_MODEL_VISION"] = "false"
os.environ["PICTURE_DESCRIPTION"] = "false"
os.environ["CRAWL_STRIP_QUERY_STRINGS"] = "true"

import pytest  # noqa: E402


class FakeLLM:
    """Stands in for the OpenAI-compatible vLLM client: returns canned replies
    in order and records the messages it was sent."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        content = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(message=SimpleNamespace(content=content))


@pytest.fixture
def fake_llm():
    return FakeLLM
