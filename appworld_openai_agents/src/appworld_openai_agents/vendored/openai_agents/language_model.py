# Vendored from StonyBrookNLP/appworld (https://github.com/StonyBrookNLP/appworld)
# Source path: experiments/code/openai_agents/language_model.py
# Commit: a072b7a86e7c1d5b1d7175659d750ebb9b79f10a
# License: Apache-2.0 (see LICENSE in this recipe folder)
# Modified: rewrote imports as above; added an `api_type` option so `type: openai` can route to the SDK's native `OpenAIResponsesModel` (default) instead of only `OpenAIChatCompletionsModel`; made output-text extraction robust to leading non-message items (e.g. reasoning items emitted by reasoning models on the Responses API).
from typing import Any

from agents.extensions.models.litellm_model import LitellmModel
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI


class LanguageModel:
    def __init__(
        self,
        type: str,
        name: str,
        settings: dict[str, Any] | None = None,
        extras: dict[str, Any] | None = None,
        api_type: str = "responses",
    ):
        self.settings = ModelSettings(**(settings or {}))
        self._model: OpenAIChatCompletionsModel | OpenAIResponsesModel | LitellmModel
        if type == "openai" and api_type == "responses":
            self._model = OpenAIResponsesModel(
                model=name,
                openai_client=AsyncOpenAI(**(extras or {})),
            )
        elif type == "openai":
            self._model = OpenAIChatCompletionsModel(
                model=name,
                openai_client=AsyncOpenAI(**(extras or {})),
            )
        elif type == "litellm":
            self._model = LitellmModel(model=name, **(extras or {}))
        else:
            raise ValueError(f"Unsupported model type: {type}")

    async def generate(self, input: Any) -> dict[str, Any]:
        if isinstance(input, list):
            input_ = []
            system_instructions: str | None = None
            for item in input:
                if isinstance(item, dict) and item.get("role", None) == "system":
                    system_instructions = item.get("content", "")
                else:
                    input_.append(item)
            input = input_
        response = await self._model.get_response(
            system_instructions=system_instructions,
            input=input,
            prompt=None,
            model_settings=self.settings,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
        )
        if not response:
            return {"content": ""}
        for item in response.output:
            if item.type == "message" and item.content and item.content[0].type == "output_text":
                return {"content": item.content[0].text}
        return {"content": ""}
