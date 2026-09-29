"""Wrapper: run automationbench-skills on the 36/18 hillclimb subsets. Usage: hc_run.py run --split {train,test} ..."""
import sys
from pathlib import Path
import automationbench_skills.data as data
import automationbench_skills.data.tasks as tasks
H = Path(__file__).parent
def load_split(split):
    names = [l.strip() for l in (H / f"hc_{split}.txt").read_text().split() if l.strip()]
    by = {s.task_name: s for s in tasks.load_samples()}
    return [by[n] for n in names]
data.load_split = load_split
from automationbench_skills.cli import main
raise SystemExit(main())
