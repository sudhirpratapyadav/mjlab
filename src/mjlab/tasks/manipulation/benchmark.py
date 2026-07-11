"""Query and sequencing API for the manipulation-diversity benchmark.

Thin layer over ``mjlab.tasks.registry``: filter registered tasks by their taxonomy
tags, build continual-learning orderings, and export a manifest.

Importing this module does NOT register tasks. Import the task packages first (e.g.
``import mjlab.tasks.manipulation.config.franka``) so the registry is populated, then
query here. ``all_benchmark_tasks`` reflects whatever has been imported.

Example:
    import mjlab.tasks.manipulation.config.franka  # populate registry
    from mjlab.tasks.manipulation import benchmark
    benchmark.tasks_by_skill()                     # {SkillFamily: [task_id, ...]}
    benchmark.fragility_graded_ordering()          # planar -> dexterous sequence
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from mjlab.tasks.manipulation.taxonomy import (
  Embodiment,
  Fragility,
  SkillFamily,
  TaskTaxonomy,
)
from mjlab.tasks.registry import list_taxonomy


def all_benchmark_tasks() -> dict[str, TaskTaxonomy]:
  """{task_id: taxonomy} for every registered task that carries taxonomy tags."""
  return list_taxonomy()


def filter_tasks(
  *,
  embodiment: Embodiment | None = None,
  skill: SkillFamily | None = None,
  fragility: Fragility | None = None,
  min_fragility: Fragility | None = None,
  contact_rich: bool | None = None,
  source: str | None = None,
) -> list[str]:
  """Return sorted task_ids matching all supplied filters (None = don't filter)."""
  out = []
  for task_id, tax in all_benchmark_tasks().items():
    if embodiment is not None and tax.embodiment is not embodiment:
      continue
    if skill is not None and tax.skill is not skill:
      continue
    if fragility is not None and tax.fragility is not fragility:
      continue
    if min_fragility is not None and tax.fragility < min_fragility:
      continue
    if contact_rich is not None and tax.contact_rich is not contact_rich:
      continue
    if source is not None and tax.source != source:
      continue
    out.append(task_id)
  return sorted(out)


def _group_by(key) -> dict:
  groups: dict = defaultdict(list)
  for task_id, tax in all_benchmark_tasks().items():
    groups[key(tax)].append(task_id)
  return {k: sorted(v) for k, v in groups.items()}


def tasks_by_embodiment() -> dict[Embodiment, list[str]]:
  """{Embodiment: [task_id, ...]}."""
  return _group_by(lambda t: t.embodiment)


def tasks_by_skill() -> dict[SkillFamily, list[str]]:
  """{SkillFamily: [task_id, ...]}."""
  return _group_by(lambda t: t.skill)


def tasks_by_fragility() -> dict[Fragility, list[str]]:
  """{Fragility: [task_id, ...]}."""
  return _group_by(lambda t: t.fragility)


# ---------------------------------------------------------------------------
# Continual-learning orderings
# ---------------------------------------------------------------------------
# These build task *sequences* for CL runs. They are deterministic given the set of
# registered tasks (no RNG) so experiments are reproducible; callers that want a
# random ordering can shuffle with their own seeded RNG.


def fragility_graded_ordering(
  embodiment: Embodiment | None = None,
  fragile_first: bool = True,
) -> list[str]:
  """Tasks ordered by fragility.

  With ``fragile_first=True`` (default) the most fragile task comes FIRST, so it must
  survive the most downstream consolidations — the ordering that most stresses
  catastrophic forgetting (cf. FINDINGS.md, where a fragile task-0 is hardest to
  retain). With ``fragile_first=False`` it grades easy->hard (planar first), a gentler
  curriculum.
  """
  tasks = filter_tasks(embodiment=embodiment) if embodiment else sorted(all_benchmark_tasks())
  taxo = all_benchmark_tasks()
  return sorted(
    tasks,
    key=lambda tid: (-taxo[tid].fragility if fragile_first else taxo[tid].fragility, tid),
  )


def skill_diverse_ordering(
  embodiment: Embodiment | None = None,
) -> list[str]:
  """Tasks ordered to maximize skill change between consecutive tasks.

  Round-robins across skill families so adjacent tasks exercise different skills — a
  curriculum that avoids "easy" same-skill transfer between neighbors. Deterministic.
  """
  by_skill = tasks_by_skill()
  if embodiment is not None:
    allowed = set(filter_tasks(embodiment=embodiment))
    by_skill = {s: [t for t in ts if t in allowed] for s, ts in by_skill.items()}
    by_skill = {s: ts for s, ts in by_skill.items() if ts}
  # Stable round-robin over skill families (sorted by skill name for determinism).
  buckets = [list(by_skill[s]) for s in sorted(by_skill, key=lambda s: s.value)]
  ordering: list[str] = []
  i = 0
  while any(buckets):
    b = buckets[i % len(buckets)]
    if b:
      ordering.append(b.pop(0))
    i += 1
    # drop emptied buckets to keep round-robin tight
    if i % len(buckets) == 0:
      buckets = [b for b in buckets if b]
      i = 0
      if not buckets:
        break
  return ordering


def export_manifest(path: str | Path | None = None) -> dict:
  """Build (and optionally write) a JSON manifest of all benchmark tasks.

  The manifest is the benchmark's public metadata: per-task embodiment / skill /
  fragility / source / license-provenance notes, plus summary counts. Written to
  ``path`` if given; always returned as a dict.
  """
  tasks = all_benchmark_tasks()
  by_emb = tasks_by_embodiment()
  by_skill = tasks_by_skill()
  by_frag = tasks_by_fragility()
  manifest = {
    "num_tasks": len(tasks),
    "num_distinct_skills": len(by_skill),
    "counts_by_embodiment": {e.value: len(v) for e, v in by_emb.items()},
    "counts_by_skill": {s.value: len(v) for s, v in by_skill.items()},
    "counts_by_fragility": {f.name.lower(): len(v) for f, v in by_frag.items()},
    "tasks": {tid: tax.as_dict() for tid, tax in sorted(tasks.items())},
  }
  if path is not None:
    Path(path).write_text(json.dumps(manifest, indent=2))
  return manifest
