"""Raw APEX-Agents dataset primitives.

* :mod:`.dataset` — HuggingFace loader (``mercor/apex-agents``) producing
  plain typed records + rubric dataclasses.
* :mod:`.world_splits` — the world-level cross-world validation split plus
  the stratified record cap used to hold world width constant.
"""

from .dataset import (
    APEX_AGENTS_HF_REPO,
    DEFAULT_DOMAIN,
    ApexAgentsRecord,
    RubricCriterion,
    filter_investment_banking,
    load_apex_agents_records,
    records_from_rows,
    world_ids_for_records,
)
from .world_splits import (
    fixed_val_split,
    stratified_case_cap,
)


__all__ = [
    "APEX_AGENTS_HF_REPO",
    "DEFAULT_DOMAIN",
    "ApexAgentsRecord",
    "RubricCriterion",
    "filter_investment_banking",
    "fixed_val_split",
    "load_apex_agents_records",
    "records_from_rows",
    "stratified_case_cap",
    "world_ids_for_records",
]
