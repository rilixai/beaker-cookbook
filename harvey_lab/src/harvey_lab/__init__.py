"""Harvey LAB legal-agent benchmark recipe for the rilixai optimizer.

A GEPA-optimized, file-producing legal agent on Harvey's Legal Agent
Benchmark (LAB). The agent harness is Artificial Analysis' `Stirrup
<https://github.com/ArtificialAnalysis/Stirrup>`_ framework (the harness
AA's Harvey LAB-AA leaderboard runs on); the recipe supplies domain file
tools over a per-task workspace and grades deliverables with LAB's
all-pass rubric methodology (a per-criterion LLM judge with
deliverable-scoped context).

Shape B (SDK-only): the recipe depends on the lightweight ``rilixai``
SDK; the GEPA reflect/propose loop runs server-side via ``rilixai run``.
"""
