"""Harvey LAB: a legal knowledge-worker agent + local rubric evaluation.

A file-producing legal agent on Harvey's public Legal Agent Benchmark (LAB).
The agent harness is Artificial Analysis' `Stirrup
<https://github.com/ArtificialAnalysis/Stirrup>`_ framework (the harness AA's
Harvey LAB-AA leaderboard runs on); this package supplies the domain file
tools over a per-task workspace and grades deliverables with LAB's all-pass
rubric methodology (a batched LLM judge with deliverable-scoped context).

Layout:

* ``agent/`` — the Stirrup-driven legal agent (workspace + file tools + prompts).
* ``data/`` — loads LAB task records from a local checkout + the frozen splits.
* ``evaluation/`` — the batched rubric judge + the dataset eval runner.
* ``splits/`` — frozen train/test task-id lists (see ``splits/README.md``).
* ``cli.py`` — run the agent, or run + grade it.
"""
