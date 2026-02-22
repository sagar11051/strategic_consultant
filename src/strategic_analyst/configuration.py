import os
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class AgentConfiguration:
    """Runtime configuration injected via thread config (RunnableConfig['configurable'])."""

    user_id: str = "default_user"
    session_id: str = ""
    model_name: str = field(
        default_factory=lambda: os.getenv("MAIN_AGENT_MODEL", "Mistral-Nemo-Instruct-2407")
    )
    utility_model_name: str = field(
        default_factory=lambda: os.getenv("UTILITY_MODEL", "Mistral-Nemo-Instruct-2407")
    )
    rag_top_k: int = 10
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
