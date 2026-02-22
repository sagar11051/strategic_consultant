"""
Signal tools used by supervisors and task agents to communicate control flow.

  Question — ask the user a clarifying question (triggers HITL interrupt)
  Done     — signal that the current task or subtask is complete

Neither tool has side effects; both are handled in make_tool_node() with a
simple acknowledgement response so the LLM receives a valid ToolMessage.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
class Question(BaseModel):
    """
    Ask the user a clarifying question or request their input before proceeding.

    Use this when:
    - You need additional context to complete the research task
    - You found something unexpected and want the user to decide whether to pursue it
    - You are presenting discoveries and want to confirm direction before the report
    - You need to verify an assumption before committing to a report section

    The question will be shown to the user via the human-in-the-loop interface.
    Wait for the user's answer before continuing.
    """

    content: str = Field(description="The question to ask the user")


@tool
class Done(BaseModel):
    """
    Signal that the current task or subtask is complete.

    Use this after:
    - Completing all assigned research tasks for this turn
    - Writing a final report section
    - Finishing a response to a user follow-up

    Do NOT use this if you still have pending tool calls to make.
    """

    done: bool = Field(default=True, description="Always True — signals completion")
    summary: str = Field(
        default="",
        description="Brief summary of what was accomplished in this task",
    )
