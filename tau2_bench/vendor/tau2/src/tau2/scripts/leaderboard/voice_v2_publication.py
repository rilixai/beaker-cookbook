"""Package the frozen 2026-10-02 Voice V2 release exports as 2.0 packages.

Maintainer-only. The release calls predate the current matrix layout and some
recorded condition fields, so ``tau2 submit prepare <matrix-root>`` cannot read
them. This adapter reads the saved exports (``trajectories/<lang>/<domain>/``)
and their saved judge verdicts as raw JSON, checks them against the published
frame, and writes the same package a matrix submission produces. It never runs
a simulation or a judge, and it must never qualify an incomplete new run.
"""

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from tau2.data_model.benchmark import (
    BenchmarkFrame,
    BenchmarkSubmission,
    TaskTrial,
    TrialStatus,
)
from tau2.data_model.judges import (
    PlanAction,
    PlannedVerdict,
    StoredVerdict,
    SuitePlan,
)
from tau2.judges.submission import export_diagnostics
from tau2.metrics.benchmark import summarize
from tau2.runner.presets import load_frame
from tau2.scripts.leaderboard.benchmark_package import (
    BenchmarkPackage,
    PackageMetadata,
    SubmitterMetadata,
    load_package,
)

EVIDENCE = "historical-voice-v2-2026-10-02"
CORRECTION_FORMAT = "voice-v2-gemini-retry-correction-v1"


def file_sha256(path: Path) -> str:
    """Hash a saved input without modifying it."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_dir(root: Path, language: str, domain: str) -> Path:
    return root / "trajectories" / language / domain


def _validate_simulation_files(
    root: Path, simulations: list[dict], language: str, domain: str
) -> None:
    """Check the directory payloads against the canonical result index."""
    sim_dir = _cell_dir(root, language, domain) / "simulations"
    indexed_ids = {entry["id"] for entry in simulations}
    file_ids = {path.stem for path in sim_dir.glob("*.json")}
    if indexed_ids != file_ids:
        raise ValueError(f"{language}/{domain}: simulation files differ from index")
    for entry in simulations:
        payload = json.loads((sim_dir / f"{entry['id']}.json").read_text())
        if any(
            payload.get(field) != entry[field]
            for field in ("id", "task_id", "trial", "termination_reason")
        ):
            raise ValueError(
                f"{language}/{domain}: simulation/index identity mismatch: {entry['id']}"
            )
        reward_info = payload.get("reward_info") or {}
        if reward_info.get("reward") != entry["reward"]:
            raise ValueError(
                f"{language}/{domain}: simulation/index reward mismatch: {entry['id']}"
            )


def _validate_cell_frame(
    data: dict,
    simulations: list[dict],
    language: str,
    domain: str,
    official: BenchmarkFrame,
) -> dict[str, str]:
    """Check release metadata against the frame without inventing later fields."""
    info = data["info"]
    frame = info.get("benchmark") or {}
    cell = next(
        item
        for item in official.cells
        if (item.language, item.domain) == (language, domain)
    )
    if frame.get("benchmark_version") != official.benchmark_version:
        raise ValueError(f"{language}/{domain}: trajectory is not voice-v2")
    if frame.get("frame_revision") != official.revision:
        raise ValueError(f"{language}/{domain}: frame revision mismatch")
    if frame.get("binding_revision") != cell.binding_revision:
        raise ValueError(f"{language}/{domain}: frozen binding mismatch")
    if frame.get("canonical_task_ids") != cell.canonical_task_ids:
        raise ValueError(f"{language}/{domain}: canonical task frame mismatch")
    if frame.get("language") != language or frame.get("domain") != domain:
        raise ValueError(f"{language}/{domain}: benchmark frame identity mismatch")
    if info.get("language") != language:
        raise ValueError(f"{language}/{domain}: recorded language mismatch")
    if info["environment_info"]["domain_name"] != domain:
        raise ValueError(f"{language}/{domain}: recorded domain mismatch")
    selected = set(frame["selected_task_ids"])
    mapped = {
        str(task_id): details["canonical_task_id"]
        for task_id, details in info["language_tasks"].items()
    }
    task_ids = [mapped[str(task["id"])] for task in data["tasks"]]
    if (
        frame["selected_task_ids"] != cell.canonical_task_ids
        or len(task_ids) != len(cell.canonical_task_ids)
        or len(set(task_ids)) != len(task_ids)
        or set(task_ids) != selected
    ):
        raise ValueError(f"{language}/{domain}: selected 50-task frame mismatch")
    if frame["trials"] != official.trials or info["num_trials"] != 1:
        raise ValueError(f"{language}/{domain}: V2 publication requires one trial")
    expected = {(task_id, 0) for task_id in selected}
    actual = [(mapped[str(sim["task_id"])], sim["trial"]) for sim in simulations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"{language}/{domain}: missing or duplicate task/trial keys")
    if any(
        sim["reward"] is None and sim["termination_reason"] != "infrastructure_error"
        for sim in simulations
    ):
        raise ValueError(
            f"{language}/{domain}: unscored call lacks infrastructure status"
        )
    return mapped


def _validate_source_correction(root: Path) -> None:
    """A retry-correction manifest must still describe the corrected export."""
    path = root / "correction.json"
    if not path.is_file():
        return
    correction = json.loads(path.read_text())
    if correction.get("format") != CORRECTION_FORMAT:
        raise ValueError("unknown source correction format")
    for item in correction["corrections"]:
        cell = _cell_dir(root, item["language"], item["domain"])
        simulation = cell / "simulations" / f"{item['new_index']['id']}.json"
        if file_sha256(cell / "results.json") != item["corrected_results_sha256"]:
            raise ValueError("corrected result differs from source correction")
        if file_sha256(simulation) != item["corrected_simulation_sha256"]:
            raise ValueError("corrected simulation differs from source correction")


def _load_verdicts(
    judges_roots: list[Path], language: str, simulation_id: str
) -> list[StoredVerdict]:
    """Exactly one judge root holds each scored call's saved verdict file."""
    paths = [
        path
        for root in judges_roots
        if (path := root / language / f"{simulation_id}.json").is_file()
    ]
    if len(paths) != 1:
        raise ValueError(
            f"{language}/{simulation_id}: expected one saved verdict file, "
            f"found {len(paths)}"
        )
    rows = [
        StoredVerdict.model_validate(row) for row in json.loads(paths[0].read_text())
    ]
    measures = [row.identity.measure for row in rows]
    if len(measures) != len(set(measures)):
        raise ValueError(f"{paths[0]}: duplicate judge measure")
    if any(row.identity.input.simulation_id != simulation_id for row in rows):
        raise ValueError(f"{paths[0]}: verdict simulation ID mismatch")
    return rows


def _judge_plan(
    scored: list[str], verdicts: dict[str, list[StoredVerdict]], measures: list[str]
) -> SuitePlan:
    """Every saved measure of the language, for every scored call of the cell.

    A call without a saved verdict for one of its language's measures stays
    explicit as not judged; it is never dropped or counted as a failure.
    """
    items = []
    for simulation_id in scored:
        rows = {row.identity.measure: row for row in verdicts[simulation_id]}
        call_input = next(iter(rows.values())).identity.input
        for measure in measures:
            row = rows.get(measure)
            items.append(
                PlannedVerdict(
                    input=call_input,
                    measure=measure,
                    action=PlanAction.REUSE if row else PlanAction.MISSING,
                    identity=row.identity if row else None,
                    reason="Frozen 2026-10-02 release verdict"
                    if row
                    else "No saved release verdict",
                )
            )
    return SuitePlan(items=items)


def build_historical_package(
    source_root: Path, judges_roots: list[Path], submitter: SubmitterMetadata
) -> BenchmarkPackage:
    """Derive the 2.0 package for one provider's frozen 18-cell export."""
    _validate_source_correction(source_root)
    official = load_frame("voice", num_trials=1)
    trials: list[TaskTrial] = []
    sources: dict[str, str] = {}
    agents: set[tuple[str, str]] = set()
    caller_models: set[str] = set()
    scored_by_cell: dict[tuple[str, str], list[str]] = {}
    verdicts: dict[str, dict[str, list[StoredVerdict]]] = defaultdict(dict)
    for cell in official.cells:
        language, domain = cell.language, cell.domain
        results_path = _cell_dir(source_root, language, domain) / "results.json"
        data = json.loads(results_path.read_text())
        simulations = data["simulation_index"]
        info = data["info"]
        audio_config = info["benchmark"]["conditions"]["audio_native_config"]
        agents.add((audio_config["provider"], audio_config["model"]))
        caller_models.add(info["user_info"]["llm"])
        task_mapping = _validate_cell_frame(
            data, simulations, language, domain, official
        )
        _validate_simulation_files(source_root, simulations, language, domain)
        benchmark = info["benchmark"]
        sources[f"{language}/{domain}"] = file_sha256(results_path)
        trials.extend(
            TaskTrial(
                language=language,
                domain=domain,
                canonical_task_id=task_mapping[str(sim["task_id"])],
                trial=sim["trial"],
                condition_fingerprint=benchmark["condition_fingerprint"],
                binding_revision=benchmark["binding_revision"],
                status=TrialStatus.SCORED
                if sim["reward"] is not None
                else TrialStatus.ERROR,
                reward=sim["reward"],
            )
            for sim in simulations
        )
        scored = [sim["id"] for sim in simulations if sim["reward"] is not None]
        scored_by_cell[(language, domain)] = scored
        for simulation_id in scored:
            verdicts[language][simulation_id] = _load_verdicts(
                judges_roots, language, simulation_id
            )
    if len(agents) != 1 or len(caller_models) != 1:
        raise ValueError("Voice V2 export mixes agent or caller models")
    for language, by_call in verdicts.items():
        saved = {
            path.stem
            for root in judges_roots
            for path in (root / language).glob("*.json")
        }
        if saved != set(by_call):
            raise ValueError(f"{language}: judge IDs differ from scored calls")
    catalogs = {
        row.identity.catalog_revision
        for by_call in verdicts.values()
        for rows in by_call.values()
        for row in rows
    }
    if len(catalogs) != 1:
        raise ValueError(f"Saved verdicts mix judge catalogs: {sorted(catalogs)}")
    cells = []
    for (language, domain), scored in scored_by_cell.items():
        measures = list(
            dict.fromkeys(
                row.identity.measure
                for rows in verdicts[language].values()
                for row in rows
            )
        )
        stored = [row for call in scored for row in verdicts[language][call]]
        cells.append(
            (
                language,
                domain,
                _judge_plan(scored, verdicts[language], measures),
                stored,
            )
        )
    provider, model = agents.pop()
    return BenchmarkPackage(
        metadata=PackageMetadata(
            **submitter.model_dump(),
            agent_provider=provider,
            agent_model=model,
            evidence=EVIDENCE,
        ),
        submission=BenchmarkSubmission(
            frame=official,
            trials=trials,
            summary=summarize(official, trials),
            sources=sources,
        ),
        diagnostics=export_diagnostics(official, catalogs.pop(), cells),
    ).check()


def prepare_historical_package(
    source_root: Path,
    judges_roots: list[Path],
    metadata_file: Path,
    output: Path,
) -> BenchmarkPackage:
    """Write ``<output>/<name>/`` from the saved export; nothing is copied."""
    submitter = SubmitterMetadata.model_validate_json(metadata_file.read_text())
    package = build_historical_package(source_root, judges_roots, submitter)
    destination = output / package.name
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    package.write(destination)
    print(f"Saved {destination}")
    return package


def validate_historical_package(
    directory: Path, source_root: Path, judges_roots: list[Path]
) -> BenchmarkPackage:
    """Recompute the package from the saved export and verdicts."""
    package = load_package(directory)
    if package.metadata.evidence != EVIDENCE:
        raise ValueError(f"{directory} is not a historical Voice V2 package")
    submitter = SubmitterMetadata.model_validate(
        package.metadata.model_dump(include=set(SubmitterMetadata.model_fields))
    )
    expected = build_historical_package(source_root, judges_roots, submitter)
    expected.metadata.trajectories = package.metadata.trajectories
    if expected.submission != package.submission:
        raise ValueError("Submission differs from the saved release export")
    if expected.diagnostics != package.diagnostics:
        raise ValueError("Judge diagnostics differ from the saved release verdicts")
    if expected.metadata != package.metadata:
        raise ValueError("Recorded agent identity differs from the saved export")
    return package
