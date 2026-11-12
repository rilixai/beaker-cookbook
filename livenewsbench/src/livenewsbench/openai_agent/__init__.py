"""The editable agent for OpenAI's built-in web search.

`answer_question(question, *, model) -> str` is its whole contract.
"""

from .agent import answer_question


__all__ = ["answer_question"]
