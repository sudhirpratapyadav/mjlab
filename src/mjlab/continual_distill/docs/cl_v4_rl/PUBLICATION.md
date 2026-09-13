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

`verify_gallery.mjs` checks24 video playback, filter/search behavior and mobile overflow. On the VPS, an isolated Playwright install is at `/tmp/cl24-browser-check/node_modules/playwright`; use its `index.mjs` as PLAYWRIGHT_MODULE and the existing Chromium executable as CHROMIUM_PATH. The current check expects16 certified /8 unfinished; update these assertions against the authoritative certificate count when teachers are newly certified. Verify public HTTPS responses and media SHA256s against generated `summary.json`; retain compact verification in evidence and screenshots under ignored runs. No web-server reload or service configuration change is needed.

Detailed numerical results, experiment records and retained checkpoints continue to go to the scoped W&B project. The gallery is a point-in-time summary, not a live training monitor.

Latest wave20 refresh:14 certified/10 in progress; all24 actual RL videos. Updated Cage11096 (97/128), Place6300 (96/128), Edge4998, Peg3400 and Lift8997. All newly selected clips reviewed before upload via untu_vps. Public checks pass newest home card/navigation,24-video playback, filters/search/mobile layout and50 HTTP200 SHA256 matches.

Wave21 refresh includes reviewed Lift10496, Throw7497, Pivot3499 and Peg3900 clips. Still14 certified/10 in progress. Newest homepage-card navigation,24 video playback, search/filters/mobile layout and50 HTTP200 SHA256 matches passed.

Wave21 final refresh also includes Cage12600 latest90/128 (best97/128 remains in detailed records). All24 playback and50 HTTPS SHA256 checks repeated successfully after this new clip.


Wave22 refresh: reviewed final Cage13095 (99/128) and Strike6997 (21/128) clips published via untu_vps. Newest homepage card links to current RL teachers. Browser checks passed24 actual video playback,14/10 filters, search and mobile layout; all50 public files returnedHTTP200 and matched local SHA256 over verified TLS. No new certification.


Wave23 gallery refresh includes reviewed Place7400 (105/128), Peg5097 and Lift11995 clips. Newest homepage card/navigation,24 actual video playback,14/10 filters, search/mobile layout and50/50 HTTPS SHA256 matches passed. Full details remain in scoped W&B; fourteen certified teachers remain unchanged.


Wave24 final gallery refresh published via untu_vps: latest reviewed Edge6997, Place8299, Throw9496, Cage15094 and Strike8996 clips. Still14 certified/10 in progress. Throw displays both116/128 validation and107/128 failed confirmation with in-progress status. Newest homepage card/navigation, all24 actual video playback, filters/search/mobile layout and50/50 verified-TLS HTTP200 SHA256 matches passed. Full results and retained policies remain in scoped W&B.

Wave25 gallery includes latest reviewed Lift13994 and Peg7096 actual failure videos. Still14 certified/10 in progress. Newest homepage card/navigation,24 video playback, filters/search/mobile layout and50 verified-TLS HTTP200 SHA256 matches passed. Native success counts are unchanged by contact/control diagnostics.

Wave26 gallery published via untu_vps:15 certified/9 in progress, newly certified Place7500 success video and latest Reorient10597/Strike10995/Cage17093 reviewed actual clips. Cage shows117/128 validation and115/128 failed confirmation. Newest homepage card/navigation,24 video playback,15/9 filters/search/mobile layout and50/50 verified-TLS HTTP200 SHA256 matches passed.

Wave27 public gallery refreshed via untu_vps with reviewed Stack8197, pre-collapse Peg6200 and Edge8996 actual failure clips. Still15 certified/9 in progress. Newest homepage card/navigation,24 video playback,15/9 filters/search/mobile layout and50/50 verified-TLS HTTP200 SHA256 matches passed. Seven full trainers verified with corrected CPU affinity; GPU0 remains unused.

Wave28 public gallery refreshed via untu_vps with the reviewed Strike12994 actual failure clip and latest37/128 rate. Still15 certified/9 in progress. Newest homepage card/navigation, all24 video playback,15/9 filters, search/mobile layout and50/50 verified-TLS HTTP200 SHA256 matches passed. The concise gallery links to scoped W&B for full records.

Wave29 public gallery refreshed via untu_vps:16 certified/8 unfinished. Newly certified Cage19092 success clip and reviewed latest Lift15993 (56/128) and Reorient12596 (1/128) failure clips published. Newest homepage-card navigation,24 playable videos,16/8 filters/search/mobile layout and50/50 verified-TLS HTTP200 SHA256 matches passed. Seven full runs are scheduled onGPUs1–7 with8 CPUs per step; verify current UUID/PID placement before next launch. GPU0 unused.
