"""Behavior-cloning teacher pipeline.

Stage 1 (collect_bc_data): roll out a classical expert with injected action
  noise (DART-style — the env visits perturbed states, labels are the
  expert's clean actions) to get state coverage beyond the expert corridor.
Stage 2 (train_bc): supervised BC training of a TeacherActorMLP on that data;
  validates in the env and saves a teacher.pkl fully compatible with the RL
  teachers (same params tree / normalizer / keys).
Stage 3 (collect_bc_teacher_dataset): deterministic rollouts of the BC policy
  to produce the distillation dataset (data.pkl) exactly like
  extract_teacher_dataset.py does for RL teachers.
"""
