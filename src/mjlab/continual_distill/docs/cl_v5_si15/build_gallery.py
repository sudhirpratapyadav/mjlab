"""Build the public CL-V5 SI-15 gallery: one shared student, 15 tasks, videos
from the best run's final checkpoint plus per-task success and the
fragile-first vs random ordering comparison.
"""
import argparse
import html
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WANDB = "https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-cl15-si-20260918"

# task, success_rate (rnd-s2 final eval), position in random ordering (1-indexed)
TASKS = [
  ("ReachTarget", 1.000, 6),
  ("PushFlap", 1.000, 9),
  ("TurnLever", 0.828, 12),
  ("OpenDoor", 0.969, 13),
  ("OpenLid", 0.953, 14),
  ("PushButton", 0.891, 8),
  ("AxialExtract", 0.891, 11),
  ("FlipSwitch", 0.938, 4),
  ("SlideWindow", 0.969, 7),
  ("OpenDrawer", 0.516, 3),
  ("RotateValve", 0.969, 2),
  ("PushCuboid", 0.266, 10),
  ("ThrowToBin", 0.531, 5),
  ("ToppleBlock", 1.000, 1),
  ("DragPull", 0.828, 15),
]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--videos-dir", type=Path, default=HERE / "videos" / "rnd-s2")
  args = parser.parse_args()
  media_dir = args.output / "media"
  media_dir.mkdir(parents=True, exist_ok=True)

  cards = []
  for name, rate, position in sorted(TASKS, key=lambda t: -t[1]):
    src_video = next(args.videos_dir.glob(f"{name}-*.mp4"))
    label = src_video.stem.split("-")[-1]
    poster = args.videos_dir / f"{name}-{label}-0000.png"
    dest = media_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_video, dest / "policy.mp4")
    if poster.exists():
      shutil.copy2(poster, dest / "poster.png")
    badge = "good" if rate >= 0.9 else ("mid" if rate >= 0.5 else "low")
    pretty = " ".join(__import__("re").findall(r"[A-Z][a-z]*", name))
    cards.append(f'''<article class="task" data-tier="{badge}" data-name="{pretty.lower()}">
<div class="task-head"><h2>{html.escape(pretty)}</h2><span class="badge {badge}">{rate*100:.1f}%</span></div>
<div class="media"><video controls playsinline preload="none" poster="media/{name}/poster.png" aria-label="{html.escape(pretty)} student policy, {label} example"><source src="media/{name}/policy.mp4" type="video/mp4">Your browser cannot play this video. <a href="media/{name}/policy.mp4">Download</a>.</video></div>
<div class="task-body"><p>Position {position}/15 in the random training order · rnd-s2 checkpoint, {label} example</p></div></article>''')

  page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="description" content="CL-V5: one shared student policy trained continually across 15 tasks with Synaptic Intelligence.">
<title>CL-V5 &middot; SI continual learning</title>
<style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;background:#0b1019;color:#e9edf5}}*{{box-sizing:border-box}}body{{margin:0}}a{{color:#8ccaff;text-underline-offset:4px}}main{{max-width:1250px;margin:auto;padding:36px 16px 60px}}nav{{display:flex;justify-content:space-between;gap:16px;font-size:14px;margin-bottom:48px;flex-wrap:wrap}}.eyebrow{{color:#8ccaff;font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase}}h1{{font-size:clamp(28px,5vw,48px);letter-spacing:-.04em;margin:12px 0}}header>p{{color:#aab6c9;font-size:16px;max-width:760px;line-height:1.7;margin:0 0 24px}}.stats{{display:flex;gap:12px;margin:30px 0 20px;flex-wrap:wrap}}.stat{{background:#141d2b;border:1px solid #263449;border-radius:12px;padding:16px 20px;flex:1;min-width:140px}}.stat strong{{display:block;font-size:26px;font-weight:650}}.stat span{{font-size:12px;color:#aab6c9}}.stat.a strong{{color:#8ccaff}}.stat.b strong{{color:#92ebc8}}.curve{{background:#141d2b;border:1px solid #263449;border-radius:12px;padding:18px 20px;margin:0 0 32px;font-size:13px;color:#c6d3e5;line-height:1.8}}.curve b{{color:#e9edf5}}.explanation{{color:#aab6c9;line-height:1.65;font-size:14px;margin:0 0 32px;max-width:960px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.task{{border:1px solid #263449;background:#111a28;border-radius:12px;overflow:hidden}}.task-head{{padding:15px 16px;display:flex;align-items:center;justify-content:space-between;gap:8px}}.task h2{{font-size:15px;line-height:1.35;margin:0;font-weight:600}}.badge{{font-size:12px;font-weight:650;white-space:nowrap;border-radius:20px;padding:5px 10px}}.badge.good{{color:#92ebc8;background:#123d33}}.badge.mid{{color:#f4cb83;background:#3c3020}}.badge.low{{color:#f4a5a5;background:#3c1f20}}.media{{background:#19202b;aspect-ratio:16/9}}video{{display:block;width:100%;height:100%;object-fit:contain}}.task-body{{padding:14px 16px}}.task-body p{{font-size:12px;color:#aab6c9;margin:0;line-height:1.5}}footer{{color:#8392aa;font-size:12px;margin-top:36px;line-height:1.8}}@media(max-width:1000px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}@media(max-width:640px){{.grid{{grid-template-columns:1fr}}.stats{{gap:8px}}}}
</style></head><body><main>
<nav><a href="/">&larr; All experiments</a><a href="{WANDB}" target="_blank" rel="noopener">View W&amp;B &#8599;</a></nav>
<header><div class="eyebrow">MJLAB / CL-V5</div><h1>SI continual learning &middot; 15 tasks, one network</h1>
<p>A single shared student policy (one MLP trunk, per-task heads) trained sequentially across 15 of the CL-V4 RL teachers (each &gt;95% standalone validation success), using Synaptic Intelligence to resist forgetting earlier tasks while learning new ones. No architecture search &mdash; SI only, width 4096.</p>
<div class="stats"><div class="stat a"><strong>0.795 &plusmn; 0.041</strong><span>Random ordering, 3 seeds</span></div><div class="stat"><strong>0.738 &plusmn; 0.049</strong><span>Fragile-first ordering, 3 seeds</span></div><div class="stat"><strong>0.691 &plusmn; 0.028</strong><span>Fragile-last ordering, 3 seeds</span></div><div class="stat b"><strong>15</strong><span>Tasks, one shared network</span></div></div>
</header>
<div class="curve"><b>Scalability curve</b> (final average success across the full sequence): N=4 &rarr; <b>0.960</b> &middot; N=6 &rarr; <b>0.792</b> &middot; N=15 &rarr; <b>0.69&ndash;0.80</b> (best ordering: random). Roughly flat from N=6 to N=15, better than a naive extrapolation of the earlier N=4&rarr;N=6 drop would suggest.</div>
<p class="explanation">Videos below show the best run overall (random ordering, seed 2, final average 0.837) after training on all 15 tasks in sequence &mdash; a success episode where available, otherwise the best-return attempt. Per-task rate is that run's final environment evaluation after the whole sequence finished, not a standalone-teacher number. Three orderings were tried: <b>random won</b> (0.795), ahead of fragile-first (0.738) and fragile-last (0.691, the worst). Training the fragile tasks last didn't help &mdash; it pushed ten easy-for-the-teacher tasks to the front instead, and several of them (PushButton, FlipSwitch) turned out to be hard for the <i>student to retain</i> despite a perfect teacher. Standalone teacher difficulty and retention difficulty are evidently different properties. PushCuboid, ThrowToBin and OpenDrawer remain consistently the hardest tasks to retain across every ordering and seed tried.</p>
<div class="grid">
{"".join(cards)}
</div>
<footer>Detailed per-run results, all 9 seeds/orderings, and training curves are in the linked W&amp;B project. This page reflects the current best single run out of nine tried (three orderings &times; three seeds).</footer>
</main></body></html>'''
  (args.output / "index.html").write_text(page)
  print(f"Wrote {args.output/'index.html'} with {len(TASKS)} task cards")


if __name__ == "__main__":
  main()
