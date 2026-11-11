import unittest

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, CompletionUsage
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.responses import Response

from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing

from utils.api_model.model_provider import OpenAIChatCompletionsModelWithRetry


class FakeChatCompletionStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk


class OpenAIChatCompletionsModelWithRetryTests(unittest.IsolatedAsyncioTestCase):
    def build_model(self) -> OpenAIChatCompletionsModelWithRetry:
        client = AsyncOpenAI(api_key="test-key", base_url="https://example.com/v1")
        return OpenAIChatCompletionsModelWithRetry(
            model="test-model",
            openai_client=client,
            retry_times=1,
            retry_delay=0.0,
            debug=False,
        )

    async def test_raw_get_response_keeps_standard_chat_completion_path(self) -> None:
        model = self.build_model()

        chat_completion = ChatCompletion(
            id="chatcmpl_1",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(role="assistant", content="hello"),
                )
            ],
            created=0,
            model="test-model",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

        async def fake_fetch_response(*args, **kwargs):
            return chat_completion

        model._fetch_response = fake_fetch_response  # type: ignore[method-assign]

        response = await model.raw_get_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
        )

        self.assertEqual(response.usage.input_tokens, 3)
        self.assertEqual(response.usage.output_tokens, 2)
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(response.output[0].type, "message")
        self.assertEqual(response.output[0].content[0].text, "hello")

    async def test_raw_get_response_handles_tuple_stream_fallback(self) -> None:
        model = self.build_model()

        initial_response = Response(
            id="resp_1",
            created_at=0.0,
            model="test-model",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )
        stream = FakeChatCompletionStream(
            [
                ChatCompletionChunk(
                    id="chatcmpl_1",
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content="hello streamed", role="assistant"),
                            finish_reason=None,
                            index=0,
                            logprobs=None,
                        )
                    ],
                    created=0,
                    model="test-model",
                    object="chat.completion.chunk",
                    usage=None,
                ),
                ChatCompletionChunk(
                    id="chatcmpl_1",
                    choices=[],
                    created=0,
                    model="test-model",
                    object="chat.completion.chunk",
                    usage=CompletionUsage(prompt_tokens=7, completion_tokens=4, total_tokens=11),
                ),
            ]
        )

        async def fake_fetch_response(*args, **kwargs):
            return initial_response, stream

        model._fetch_response = fake_fetch_response  # type: ignore[method-assign]

        response = await model.raw_get_response(
            system_instructions=None,
            input="hi",
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
        )

        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 4)
        self.assertEqual(response.usage.total_tokens, 11)
        self.assertEqual(response.output[0].type, "message")
        self.assertEqual(response.output[0].content[0].text, "hello streamed")


if __name__ == "__main__":
    unittest.main()
