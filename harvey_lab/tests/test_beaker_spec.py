"""Hermetic checks for the Harvey LAB Beaker optimization contract."""

from __future__ import annotations

from beaker import RolloutContext

from harvey_lab.beaker_spec import HarveyLabDataLoader, _agent_for_case, build_spec


def _row() -> dict:
    return {
        "id": "contracts/t1",
        "input": {"task_id": "contracts/t1"},
        "expected": {
            "task_fingerprint": "f" * 64,
            "title": "Termination memo",
            "instructions": "Summarize the termination fee.",
            "deliverables": {"memo.md": "Memo"},
            "criteria": [
                {
                    "id": "C1",
                    "title": "States fee",
                    "match_criteria": "Mentions the termination fee.",
                    "deliverables": ["memo.md"],
                }
            ],
        },
        "metadata": {"practice_area": "contracts", "work_type": "analyze"},
        "group_key": "contracts",
    }


def test_beaker_spec_exposes_the_live_prompt_seeds() -> None:
    spec = build_spec(None)  # type: ignore[arg-type]

    assert set(spec.seed_targets.prompts) == {"system_prompt", "task_template"}
    assert "professional legal-work task" in spec.seed_targets.prompts["system_prompt"]
    assert "<execution_context>" in spec.seed_targets.prompts["task_template"]


def test_data_loader_maps_real_task_contract_to_case() -> None:
    loader = HarveyLabDataLoader()
    parsed = loader.parse_row(_row(), context=None)  # type: ignore[arg-type]
    case = next(iter(loader.iter_cases(parsed, context=None)))  # type: ignore[arg-type]

    assert case.case_id == "contracts/t1"
    assert case.ground_truth["criteria"][0]["id"] == "C1"
    assert case.group_key == "contracts"


def test_optimized_prompts_are_bound_to_each_agent_instance(tmp_path) -> None:
    targets = build_spec(None).seed_targets  # type: ignore[arg-type]
    optimized = type(targets)(
        prompts={
            "system_prompt": "OPTIMIZED SYSTEM {{max_turns}} {{finish_tool}} {{abandon_tool}}",
            "task_template": "OPTIMIZED TASK {{title}} {{instructions}} {{deliverables}}",
        }
    )

    agent = _agent_for_case(
        tasks_root=tmp_path,
        targets=optimized,
        runtime=RolloutContext(model=None, user_id="test"),
    )

    assert "OPTIMIZED SYSTEM" in agent.system_prompt
    assert "{{title}}" in agent.task_template
