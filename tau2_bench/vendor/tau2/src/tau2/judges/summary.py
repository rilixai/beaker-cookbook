"""Read-only coverage summaries over saved, matching diagnostic verdicts."""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from tau2.data_model.benchmark import BenchmarkFrame, CellFrame
from tau2.data_model.judges import (
    ContractModel,
    DiagnosticId,
    JudgeCatalog,
    PlanAction,
    StoredVerdict,
    SuitePlan,
    SuiteResult,
)
from tau2.judges.suite import JudgeExecutionConfig, judge_saved_run


class DiagnosticSummary(ContractModel):
    """Each measure retains its own unit definition and coverage denominator."""

    measure: Annotated[
        DiagnosticId,
        Field(description="Validated measure reported independently of task success."),
    ]
    applicable_calls: Annotated[
        int,
        Field(
            description="Selected saved calls within the catalog applicability frame."
        ),
    ]
    judged_calls: Annotated[
        int,
        Field(description="Applicable calls with matching successful saved verdicts."),
    ]
    not_judged_calls: Annotated[
        int,
        Field(
            description="Applicable calls with no matching successful/error verdict."
        ),
    ]
    error_calls: Annotated[int, Field(description="Matching saved judge errors.")]
    missing_input_calls: Annotated[
        int, Field(description="Applicable calls missing required audio/reference.")
    ]
    not_applicable_calls: Annotated[
        int, Field(description="Calls outside selected language/participant scope.")
    ]
    pass_units: Annotated[
        int, Field(description="Judged opportunities without a violation.")
    ]
    fail_units: Annotated[
        int, Field(description="Judged opportunities with a violation.")
    ]
    no_opportunity_units: Annotated[
        int, Field(description="Explicit units with no applicable opportunity.")
    ]
    error_units: int = Field(
        default=0, description="Unjudged units excluded from pass fractions."
    )
    partial_calls: int = Field(
        default=0, description="Error calls with some successful units."
    )
    complete: Annotated[
        bool,
        Field(description="Every applicable call has a matching usable saved verdict."),
    ]
    observed_pass_fraction: Annotated[
        float | None,
        Field(
            description="Binary unit pass fraction over observed opportunities; coverage reported separately."
        ),
    ]


def summarize_suite(
    plan: SuitePlan, stored: list[StoredVerdict]
) -> list[DiagnosticSummary]:
    """Summarize recorded diagnostics without model calls or reward mutation."""
    records = {
        (
            row.identity.input.run_id,
            row.identity.input.simulation_id,
            row.identity.input.trial,
            row.identity.measure,
        ): row
        for row in stored
    }
    if len(records) != len(stored):
        raise ValueError("Duplicate stored verdict identities")
    summaries = []
    for measure in dict.fromkeys(item.measure for item in plan.items):
        applicable = judged = not_judged = errors = missing_input = not_applicable = 0
        passed = failed = no_opportunity = error_units = partial = 0
        for item in (item for item in plan.items if item.measure == measure):
            if item.action == PlanAction.NOT_APPLICABLE:
                not_applicable += 1
                continue
            applicable += 1
            if item.action == PlanAction.MISSING_INPUT:
                missing_input += 1
                continue
            row = records.get(
                (
                    item.input.run_id,
                    item.input.simulation_id,
                    item.input.trial,
                    item.measure,
                )
            )
            if row is None or row.identity != item.identity:
                not_judged += 1
            elif row.error is not None:
                errors += 1
            else:
                assert row.verdict is not None
                if row.verdict.unit_errors:
                    errors += 1
                    error_units += len(row.verdict.unit_errors)
                    partial += bool(row.verdict.units)
                else:
                    judged += 1
                for unit in row.verdict.units:
                    if not unit.opportunity:
                        no_opportunity += 1
                    elif unit.violation:
                        failed += 1
                    else:
                        passed += 1
        summaries.append(
            DiagnosticSummary(
                measure=measure,
                applicable_calls=applicable,
                judged_calls=judged,
                not_judged_calls=not_judged,
                error_calls=errors,
                missing_input_calls=missing_input,
                not_applicable_calls=not_applicable,
                pass_units=passed,
                fail_units=failed,
                no_opportunity_units=no_opportunity,
                error_units=error_units,
                partial_calls=partial,
                complete=applicable == judged,
                observed_pass_fraction=passed / (passed + failed)
                if passed + failed
                else None,
            )
        )
    return summaries


class DiagnosticExport(ContractModel):
    """Typed export of selected coverage and saved verdicts; task reward is separate."""

    schema_version: Annotated[
        str,
        Field(
            default="diagnostics-export-v2", description="Structured export version."
        ),
    ]
    plan: Annotated[
        SuitePlan, Field(description="Current missing/stale/applicability states.")
    ]
    summary: Annotated[
        list[DiagnosticSummary],
        Field(description="Independent measure summaries with explicit coverage."),
    ]
    verdicts: Annotated[
        list[StoredVerdict],
        Field(
            description="Selected stored records, including errors and stale provenance."
        ),
    ]


class CellDiagnosticExport(ContractModel):
    """Cell-scoped diagnostic coverage; absent results remain visible."""

    cell: Annotated[CellFrame, Field(description="Frozen language/domain cell.")]
    diagnostics: Annotated[
        DiagnosticExport | None,
        Field(
            description="Independent diagnostics, or null when cell results are absent."
        ),
    ]


class MatrixDiagnosticExport(ContractModel):
    """Diagnostics retain language-specific applicability across a released matrix."""

    frame: Annotated[BenchmarkFrame, Field(description="Recorded published benchmark.")]
    cells: Annotated[
        list[CellDiagnosticExport],
        Field(
            description="Per-cell diagnostic reports, including missing result cells."
        ),
    ]


def export_suite(result: SuiteResult) -> DiagnosticExport:
    """Build independent diagnostic scores from explicit selected coverage."""
    return DiagnosticExport(
        plan=result.plan,
        summary=summarize_suite(result.plan, result.verdicts),
        verdicts=result.verdicts,
    )


def build_export(
    results: Path,
    catalog: JudgeCatalog,
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
    execute: bool = False,
    execution: JudgeExecutionConfig | None = None,
) -> DiagnosticExport | MatrixDiagnosticExport:
    """Export an ordinary run or selected matrix using the same typed workflow."""
    if not (results / "benchmark-frame.json").is_file():
        return export_suite(
            judge_saved_run(
                results,
                catalog,
                only=only,
                trials=trials,
                force=force,
                execute=execute,
                execution=execution,
            )
        )
    from tau2.metrics.benchmark import load_saved_benchmark

    matrix = load_saved_benchmark(results)
    cells = []
    for saved in matrix.cells:
        diagnostics = None
        if saved.results_path is not None:
            diagnostics = export_suite(
                judge_saved_run(
                    saved.results_path.parent,
                    catalog,
                    only=only,
                    trials=trials,
                    force=force,
                    execute=execute,
                    execution=execution,
                )
            )
        cells.append(CellDiagnosticExport(cell=saved.cell, diagnostics=diagnostics))
    return MatrixDiagnosticExport(frame=matrix.submission.frame, cells=cells)
