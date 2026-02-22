"""
test_schemas.py — Unit tests for schema models and reducer functions.
"""

import pytest

from strategic_analyst.schemas import (
    AgentInput,
    ResearchFinding,
    ResearchPlan,
    ResearchTask,
    merge_dicts,
)


def test_merge_dicts_combines_keys():
    a = {"task_1": {"answer": "a"}}
    b = {"task_2": {"answer": "b"}}
    result = merge_dicts(a, b)
    assert "task_1" in result
    assert "task_2" in result


def test_merge_dicts_b_wins_on_conflict():
    a = {"key": "original"}
    b = {"key": "updated"}
    result = merge_dicts(a, b)
    assert result["key"] == "updated"


def test_merge_dicts_empty_inputs():
    assert merge_dicts({}, {}) == {}
    assert merge_dicts({"a": 1}, {}) == {"a": 1}
    assert merge_dicts({}, {"b": 2}) == {"b": 2}


def test_research_task_defaults():
    task = ResearchTask(
        task_id="task_1",
        question="What is the market size?",
        data_sources=["web"],
    )
    assert task.priority == "medium"
    assert task.dependencies == []


def test_research_finding_model():
    finding = ResearchFinding(
        task_id="task_1",
        answer="The market is $5B.",
        evidence=["Source A says $5B"],
        sources=["https://example.com"],
        confidence="high",
    )
    assert finding.task_id == "task_1"
    assert finding.confidence == "high"
    assert finding.gaps == ""


def test_research_plan_model():
    plan = ResearchPlan(
        title="APAC Fintech Analysis",
        objective="Understand the APAC fintech landscape",
        background="Rapid growth in digital payments",
        tasks=[
            ResearchTask(
                task_id="task_1",
                question="Market size?",
                data_sources=["web"],
            )
        ],
        expected_deliverable="Strategic report with recommendations",
    )
    assert len(plan.tasks) == 1
    assert plan.frameworks == []
