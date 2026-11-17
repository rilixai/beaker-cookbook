"""The fixed user simulator: upstream's ``UserSimulator`` with a pinned model and tracing.

It never follows ``runtime.model``: a model-comparison run must change the
agent, not the test. Each simulator turn is traced in its own ``tau2.user_sim``
stage so reports can separate simulator spend from agent spend.
"""

from __future__ import annotations

from typing import Any

from beaker.sdk.utils import to_json_safe
from beaker.tracing import current_trace
from tau2.data_model.message import UserMessage
from tau2.user.user_simulator import UserSimulator, UserState, ValidUserInputMessage


# Upstream v2.0.0 benchmark default for the text user simulator.
USER_LLM = "openai/gpt-5.5"
USER_LLM_ARGS: dict[str, Any] = {"reasoning_effort": "low"}
USER_NAME = "beaker_traced_user_simulator"


class TracedUserSimulator(UserSimulator[UserState]):
    def _generate_next_message(self, message: ValidUserInputMessage, state: UserState) -> UserMessage:
        trace = current_trace()
        with trace.stage("tau2.user_sim"):
            with trace.model_call(
                operation="chat",
                provider="openai",
                model=self.llm.removeprefix("openai/"),
                input_messages=to_json_safe(message.model_dump(mode="json", exclude={"raw_data"})),
            ) as call:
                reply: UserMessage = super()._generate_next_message(message, state)
                call.output(to_json_safe(reply.raw_data) if reply.raw_data else reply.content)
                if reply.usage:
                    call.usage(
                        input_tokens=reply.usage.get("prompt_tokens"),
                        output_tokens=reply.usage.get("completion_tokens"),
                    )
                return reply
