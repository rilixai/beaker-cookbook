"""System prompts.

Both constants are **paper text**, quoted from the OfficeQA Pro technical
report (arXiv:2603.08655, Databricks, March 2026), Appendix E.5 "Agent
Prompt". They are cited as such; no software license is claimed for them.
The text was checked against the arXiv PDF (v1) line by line; the PDF
renders the prompt as 18 numbered lines, with blank lines at 2, 5, 8, 15 and
17 as reproduced here.
"""

from __future__ import annotations


# Appendix E.5: "The following prompt is used across all three agent
# baselines (Claude Agent SDK, OpenAI Codex SDK, Gemini CLI) and custom agent
# setup with file system access and web search enabled."
SYSTEM_PROMPT_VERBATIM = """You are an agent that is an expert in answering questions related to the U.S treasury & economy. Given the question, please use the treasury docs -- these are located in the folder ../officeqa_corpus/ (relative to your current working directory), and you can use file system search to look through them, given the tools provided. Note that these documents are very long, so you should not try to read the whole file. You also have access to web search in the case that you need to look something up.
If multiple documents report the same metric, use the most up-to-date revision unless the question specifies an exact date or document. Do not rely on the first matching value you encounter.
You must always provide an answer that is the result of any computation or reasoning to determine the answer to the question.
Use maximum precision in your calculations unless otherwise specified by the question.

REQUIRED FORMAT for completion:
When you have the final answer, keep any final reasoning you used before getting to the answer and then only return the value in the XML tags.

<REASONING>
[final reasoning - including steps & sources used...]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>

Never respond with follow-up questions.

FAILURE CONDITION: If you do not produce a <FINAL_ANSWER> tag, your response will be considered incomplete and you will fail the task."""


_WEB_SEARCH_SENTENCE = " You also have access to web search in the case that you need to look something up."

# Seed variant. Sole deviation from Appendix E.5: the sentence "You also have
# access to web search in the case that you need to look something up." is
# removed, because the seed registers no web_search tool. Everything else is
# byte-identical to SYSTEM_PROMPT_VERBATIM.
# Consequence: §2.2 states 22% of Pro questions require internet search for
# external values, so this configuration has a structural ceiling near 78%.
# The revision-handling sentence ("If multiple documents report the same
# metric, use the most up-to-date revision ...") is kept on purpose: it is in
# the report's default prompt and §5.1 documents that agents still fail at
# revision handling despite it, so it is an open problem, not a missing
# instruction.
SYSTEM_PROMPT_SEED = SYSTEM_PROMPT_VERBATIM.replace(_WEB_SEARCH_SENTENCE, "", 1)
assert SYSTEM_PROMPT_SEED != SYSTEM_PROMPT_VERBATIM
assert SYSTEM_PROMPT_SEED.count("\n") == SYSTEM_PROMPT_VERBATIM.count("\n")


def system_prompt_for(tools: tuple[str, ...] | list[str]) -> str:
    """Verbatim prompt when ``web`` is registered, seed variant otherwise."""
    return SYSTEM_PROMPT_VERBATIM if "web" in tools else SYSTEM_PROMPT_SEED


# Ours, not from the report. The report says the agent receives "a reminder
# message of remaining steps" (§4.1) without giving its wording.
def step_reminder(remaining: int) -> str:
    if remaining <= 1:
        return (
            "Reminder: this is your last step. You must respond now with your final answer in the required "
            "<REASONING>...</REASONING><FINAL_ANSWER>...</FINAL_ANSWER> format. Do not call any more tools."
        )
    return f"Reminder: you have {remaining} steps remaining (each tool call or message uses one step)."


def user_message(payload_json: str) -> str:
    """The first user turn: the leakage-safe sample (uid, question, corpus manifest) as JSON."""
    return payload_json
