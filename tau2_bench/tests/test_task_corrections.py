"""Regression checks for the reviewed benchmark corrections; no model calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.multilingual.runtime import pin_language_artifacts, task_context

from tau2_bench import benchmark, data


def retail_tools() -> RetailTools:
    return RetailTools(RetailDB.model_validate_json((data.ensure_data() / "tau2/domains/retail/db.json").read_text()))


def test_multi_product_modification_preserves_each_catalog_variant() -> None:
    tools = retail_tools()
    action = next(
        a for a in benchmark.task("retail", "20").evaluation_criteria.actions if a.name == "modify_pending_order_items"
    )
    result = tools.modify_pending_order_items(**action.arguments)
    expected_ids = action.arguments["new_item_ids"]
    modified = [item for item in result.items if item.item_id in expected_ids]
    assert len(modified) == 4
    for item in modified:
        variant = tools.db.products[item.product_id].variants[item.item_id]
        assert item.price == variant.price
        assert item.options == variant.options
    # Updating the resulting order record must not change the catalog itself.
    modified[0].options["test"] = "order-only"
    assert "test" not in tools.db.products[modified[0].product_id].variants[modified[0].item_id].options


def test_camera_reference_keeps_constraints_when_no_replacement_exists() -> None:
    tools = retail_tools()
    original = tools.db.orders["#W4689314"].items[0]
    variants = tools.db.products[original.product_id].variants.values()
    assert not any(
        v.available
        and v.options["resolution"] == original.options["resolution"]
        and v.options["storage"] == original.options["storage"]
        and int(v.options["zoom"].removesuffix("x")) > int(original.options["zoom"].removesuffix("x"))
        for v in variants
    )
    criteria = benchmark.task("retail", "52").evaluation_criteria
    assert not any(a.name == "exchange_delivered_order_items" for a in criteria.actions)
    assert criteria.nl_assertions


def test_airline_change_of_plans_keeps_insured_economy_reservation() -> None:
    criteria = benchmark.task("airline", "39").evaluation_criteria
    cancellations = {a.arguments["reservation_id"] for a in criteria.actions if a.name == "cancel_reservation"}
    assert cancellations == {"8C8K4E", "LU15PA"}


@pytest.mark.parametrize("task_id", ["4", "20", "52"])
def test_reviewed_retail_reference_writes_execute(task_id: str) -> None:
    tools = retail_tools()
    for action in benchmark.task("retail", task_id).evaluation_criteria.actions:
        # Reference reads include deliberate failed lookups; only writes define the target state.
        if action.name.startswith(("modify_", "return_", "exchange_", "cancel_")):
            getattr(tools, action.name)(**action.arguments)


def test_unflagged_tasks_and_population_are_unchanged() -> None:
    flagged = {"airline": {"39"}, "retail": {"4", "52"}}
    for domain in data.DOMAINS:
        original = pin_language_artifacts(benchmark.run_config(domain), update=False)
        assert set(benchmark.task_ids(domain)) == {t.id for t in original.tasks}
        for task in original.tasks:
            if task.id not in flagged[domain]:
                assert benchmark.task(domain, task.id) == task


def test_language_context_pins_corrected_task() -> None:
    selected = benchmark.task("retail", "52")
    original = next(
        t for t in pin_language_artifacts(benchmark.run_config("retail"), update=False).tasks if t.id == "52"
    )
    assert (
        task_context(benchmark.run_config("retail"), selected).task_sha256
        != task_context(benchmark.run_config("retail"), original).task_sha256
    )


def test_prepare_records_corrections_without_changing_splits(tmp_path: Path) -> None:
    counts = data.prepare(tmp_path)
    assert {split: c["total"] for split, c in counts.items()} == {"train": 77, "validation": 27, "test": 60}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["task_corrections_sha256"]
        == hashlib.sha256(Path(benchmark.__file__).with_name("task_corrections.json").read_bytes()).hexdigest()
    )
