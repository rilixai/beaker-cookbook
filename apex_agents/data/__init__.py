"""Raw APEX-Agents primitives — no GEPA dependency.

* :mod:`.dataset` — HuggingFace loader (``mercor/apex-agents``), record
  + rubric dataclasses, ``Sample`` conversion.
* :mod:`.world_splits` — world-level train/val splitters (and the
  k-fold partitioner) plus the stratified case cap used to hold train
  width constant across train-size points.

A reader writing a new benchmark looks here to see what the underlying
dataset surface looks like, then goes to :mod:`apex_agents.optimization`
to see how it gets wired into GEPA.
"""

from .dataset import (
    APEX_AGENTS_HF_REPO,
    DEFAULT_DOMAIN,
    ApexAgentsRecord,
    RubricCriterion,
    cases_from_records,
    filter_investment_banking,
    load_apex_agents_cases,
    load_apex_agents_records,
    record_to_case,
    world_ids_for_cases,
)
from .world_splits import (
    fixed_val_split,
    stratified_case_cap,
    world_held_out_val_split,
    world_level_folds,
)


__all__ = [
    "APEX_AGENTS_HF_REPO",
    "DEFAULT_DOMAIN",
    "ApexAgentsRecord",
    "RubricCriterion",
    "cases_from_records",
    "filter_investment_banking",
    "fixed_val_split",
    "load_apex_agents_cases",
    "load_apex_agents_records",
    "record_to_case",
    "stratified_case_cap",
    "world_held_out_val_split",
    "world_ids_for_cases",
    "world_level_folds",
]
