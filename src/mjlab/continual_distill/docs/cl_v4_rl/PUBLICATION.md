# Public RL teacher summary

User authorized publication on2026-09-13 via `untu_vps`.

- Home: https://cl.sudhirpratapyadav.com/ — newest CL-V4 card and latest link.
- Gallery: https://cl.sudhirpratapyadav.com/v4-rl/
- Remote directory: `/home/untu/sudhir/continual_learning/v4-rl`.
- Original home-page backup: `/home/untu/sudhir/cl24-site-backups/` (outside web root).

`build_gallery.py` builds24 summary/video cards using certificate-backed success clips for certified tasks and the latest completed recorded validation failure for unfinished tasks. Actual evaluation rates are independent of the selected example. Keep the output concise; do not copy checkpoints, raw traces, credentials or full logs to the site. Never use earlier classical-teacher videos for an RL card.

From the experiment worktree, after reviewing new evaluation videos:

```bash
export PYTHONPATH="$PWD/src"
/ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python src/mjlab/continual_distill/docs/cl_v4_rl/build_gallery.py --output src/mjlab/continual_distill/docs/cl_v4_rl/runs/public-gallery
rsync -a --delay-updates src/mjlab/continual_distill/docs/cl_v4_rl/runs/public-gallery/ untu_vps:sudhir/continual_learning/v4-rl/
```

The first publication was staged outside the web root, browser-checked and moved into a new directory. Future updates are authorized within the same summary scope. Preserve all older galleries and unrelated home-page cards. The root card intentionally has no hard-coded certification count; current counts are inside the generated gallery.

`verify_gallery.mjs` checks24 video playback, filter/search behavior and mobile overflow. On the VPS, an isolated Playwright install is at `/tmp/cl24-browser-check/node_modules/playwright`; use its `index.mjs` as PLAYWRIGHT_MODULE and the existing Chromium executable as CHROMIUM_PATH. The initial check expects14 certified /10 unfinished; update these assertions against the authoritative certificate count when teachers are newly certified. Verify public HTTPS responses and media SHA256s against generated `summary.json`; retain compact verification in evidence and screenshots under ignored runs. No web-server reload or service configuration change is needed.

Detailed numerical results, experiment records and retained checkpoints continue to go to the scoped W&B project. The gallery is a point-in-time summary, not a live training monitor.

Latest wave20 refresh:14 certified/10 in progress; all24 actual RL videos. Updated Cage11096 (97/128), Place6300 (96/128), Edge4998, Peg3400 and Lift8997. All newly selected clips reviewed before upload via untu_vps. Public checks pass newest home card/navigation,24-video playback, filters/search/mobile layout and50 HTTP200 SHA256 matches.

Wave21 refresh includes reviewed Lift10496, Throw7497, Pivot3499 and Peg3900 clips. Still14 certified/10 in progress. Newest homepage-card navigation,24 video playback, search/filters/mobile layout and50 HTTP200 SHA256 matches passed.
