# Public CL-V5 SI-15 summary

- Home: https://cl.sudhirpratapyadav.com/ — updated homepage card, CL-V5 now the "Latest" entry.
- Gallery: https://cl.sudhirpratapyadav.com/v5-cl-si/
- Remote directory: `/home/untu/sudhir/continual_learning/v5-cl-si`.
- Homepage backup before this update: `index.html.bak_20260919_100103` (same directory).

`build_gallery.py` builds 15 task cards from the rnd-s2 run's final checkpoint
(best of Block 3's 6 runs, avg 0.837), using `render_student.py`-rendered
success (or best-return failure) clips. Verified: homepage 200, gallery index
200, a sample video 200, all over HTTPS.

## Wave 1 (2026-09-18/19, first publication)

Published all 15 tasks from rnd-s2 (random ordering, seed 2). Fragile-first
mean 0.738 ± 0.049, random mean 0.795 ± 0.041 across the 6 Block 3 runs — see
EXPERIMENTS.md for the full breakdown and the task-fragility/mid-sequence
findings. This is the current best single run, not a guaranteed final result;
the page will be refreshed if the running fragile-last follow-up does better,
following the same iterative "wave" convention CL-V4 used.
