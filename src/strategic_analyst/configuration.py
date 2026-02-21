from dataclasses import dataclass, field
from typing import Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class AgentConfiguration:
    """Runtime configuration injected via thread config."""

    user_id: str = "default_user"
    session_id: str = ""
    model_name: str = "claude-sonnet-4-6"
    utility_model_name: str = "claude-haiku-4-5-20251001"
    rag_top_k: int = 8
    max_research_tasks: int = 5
    max_supervisor_retries: int = 2

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "AgentConfiguration":
        configurable = (config or {}).get("configurable", {})
        return cls(
            **{k: v for k, v in configurable.items() if k in cls.__dataclass_fields__}
        )
