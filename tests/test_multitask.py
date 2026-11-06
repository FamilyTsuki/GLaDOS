"""Several jobs at once: long ones go to the background worker, results come back on their own."""

from __future__ import annotations

from datetime import date, timedelta
import json
import queue
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from glados.autonomy.agents.nexus_agent import NexusAgent, NexusConfig
from glados.autonomy.slots import TaskSlotStore
from glados.core.tool_offer import filter_tools_for_message
from glados.tools.agent.base import AgentConfig, _Collect
from glados.tools.agent.definitions import agent_definitions
from glados.tools.agent.worker import DelegateTask


@pytest.mark.parametrize(
    "message",
    ["En attendant, calcule le prix de ton API à l'année.", "Meanwhile, find the most interesting GitHub commit."],
)
def test_juggling_requests_offers_the_background_worker(message: str) -> None:
    offered = {tool["function"]["name"] for tool in filter_tools_for_message(agent_definitions, message)}
    assert "delegate_task" in offered


def test_finished_background_task_is_announced_without_a_review() -> None:
    from glados.autonomy.loop import AGENT_INSTRUCTION
    from tests.test_nexus_core import make_loop

    loop = make_loop()
    loop._central_queue = queue.Queue()
    loop._slot_store.update_slot(
        "task_agent_abc",
        "Agent: find the most interesting GitHub commit",
        "done",
        "Agent task done: find the most interesting GitHub commit",
        report=json.dumps({"task": "commit", "report": "obra/superpowers v7.0.2"}),
        update_priority="important",
    )
    loop._scan_slots()
    assert loop._deliver_direct()
    request = loop._central_queue.get_nowait()
    assert AGENT_INSTRUCTION in request["content"] and "superpowers" in request["content"]
    assert loop._llm_queue.empty()


def test_a_worker_out_of_steps_still_reports_what_it_found(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(
        [
            {"content": "", "tool_calls": [{"id": "1", "function": {"name": "agent_notes", "arguments": "{}"}}]},
            {"content": "Found obra/superpowers v7.0.2; ranking unfinished."},
        ]
    )
    asked: list[list[dict[str, Any]]] = []

    def ask(self: DelegateTask, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        asked.append(tools)
        return next(replies)

    monkeypatch.setattr(DelegateTask, "ask_worker", ask)
    monkeypatch.setattr(DelegateTask, "run_tool", lambda self, name, arguments: "{}")
    worker = DelegateTask(_Collect(), {"agent": AgentConfig(enabled=True), "mcp_manager": MagicMock()})
    status, text, _ = worker.loop("find a commit", threading.Event(), steps=1)
    assert (status, text) == ("partial", "Found obra/superpowers v7.0.2; ranking unfinished.")
    assert any(t["function"]["name"] == "web_search" for t in asked[0]), "the worker can search the web"
    assert asked[-1] == [], "the final report is asked for without tools"


def test_api_cost_is_projected_from_a_month_of_real_spending(tmp_path) -> None:
    from glados.core.settings_files import write_settings

    today = date.today()
    spend = {(today - timedelta(days=n)).isoformat(): 0.4 for n in range(1, 40)} | {today.isoformat(): 0.05}
    path = tmp_path / "nexus.yaml"
    write_settings(path, {"spend": spend})
    agent = NexusAgent(
        NexusConfig(),
        {"url": "http://localhost/api/chat", "model": "m"},
        slot_store=TaskSlotStore(),
        settings_path=path,
    )
    status = agent.status()
    assert status["average_daily_spend_usd"] == 0.4 and status["projected_yearly_spend_usd"] == 146.0
    assert status["yearly_spend_at_full_budget_usd"] == 182.5
    agent.charge("gpt-5-nano", {"prompt_tokens": 1000, "completion_tokens": 0})
    assert len(agent._spend) == 31, "a month of history is kept, older days are dropped"
