"""Serve the recorded task videos as a single browsable HTML gallery.

Companion to ``record_task_videos.py``. Reads that script's ``manifest.json`` and
renders every clip in one grid so the whole benchmark can be eyeballed at once —
scene layout, asset placement, camera framing, and which tasks look wrong.

Usage:
    python -m mjlab.scripts.serve_task_videos --video-dir benchmark_videos --port 8000

Then open http://localhost:8000 (or forward the port over SSH:
    ssh -N -L 8000:<node>:8000 <user>@<login-host>
).

Standard-library only (``http.server``), so it runs anywhere the repo does with no
extra dependencies.
"""

from __future__ import annotations

import http.server
import json
import socketserver
from dataclasses import dataclass
from functools import partial
from html import escape
from pathlib import Path

import tyro

_FRAGILITY_COLORS = {
  1: "#3b82f6",  # planar
  2: "#10b981",  # mild contact
  3: "#f59e0b",  # precision grasp
  4: "#ef4444",  # dexterous
}


def build_index_html(video_dir: Path) -> str:
  manifest_path = video_dir / "manifest.json"
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    entries = [e for e in manifest.get("videos", []) if e.get("ok")]
    failed = [e for e in manifest.get("videos", []) if not e.get("ok")]
    settings = manifest.get("settings", {})
  else:
    # Fall back to whatever .mp4 files are present.
    entries = [
      {"task_id": p.stem, "file": p.name} for p in sorted(video_dir.glob("*.mp4"))
    ]
    failed, settings = [], {}

  def group_of(entry: dict) -> str:
    tax = entry.get("taxonomy") or {}
    return tax.get("embodiment", "unknown")

  groups: dict[str, list[dict]] = {}
  for entry in entries:
    groups.setdefault(group_of(entry), []).append(entry)

  cards_by_group = []
  for group in sorted(groups):
    cards = []
    for entry in sorted(groups[group], key=lambda e: e["task_id"]):
      tax = entry.get("taxonomy") or {}
      frag = tax.get("fragility")
      color = _FRAGILITY_COLORS.get(frag, "#6b7280")
      chips = []
      if tax.get("skill"):
        chips.append(f'<span class="chip">{escape(str(tax["skill"]))}</span>')
      if frag is not None:
        chips.append(
          f'<span class="chip" style="background:{color}22;color:{color};'
          f'border-color:{color}55">tier {frag} · '
          f'{escape(str(tax.get("fragility_name", "")))}</span>'
        )
      if tax.get("contact_rich"):
        chips.append('<span class="chip">contact-rich</span>')

      meta = []
      if entry.get("obs_dim") is not None:
        meta.append(f'obs {entry["obs_dim"]}')
      if entry.get("action_dim") is not None:
        meta.append(f'act {entry["action_dim"]}')
      if entry.get("episode_length_s") is not None:
        meta.append(f'episode {entry["episode_length_s"]:g}s')

      notes = escape(str(tax.get("notes", "")))
      cards.append(f"""
        <div class="card">
          <video src="{escape(entry["file"])}" muted loop playsinline
                 preload="metadata" controls></video>
          <div class="body">
            <div class="title">{escape(entry["task_id"])}</div>
            <div class="chips">{"".join(chips)}</div>
            <div class="meta">{" · ".join(meta)}</div>
            {f'<details><summary>notes</summary><p>{notes}</p></details>' if notes else ""}
          </div>
        </div>""")
    cards_by_group.append(
      f'<h2>{escape(group)} <span class="count">{len(groups[group])}</span></h2>'
      f'<div class="grid">{"".join(cards)}</div>'
    )

  failed_html = ""
  if failed:
    items = "".join(
      f'<li><code>{escape(e["task_id"])}</code> — {escape(str(e.get("error", "")))}</li>'
      for e in failed
    )
    failed_html = f'<h2>failed to record <span class="count">{len(failed)}</span></h2><ul class="failed">{items}</ul>'

  settings_line = ""
  if settings:
    settings_line = escape(
      f"{settings.get('seconds', '?')}s · {settings.get('fps', '?')}fps · "
      f"{settings.get('agent', '?')} agent · seed {settings.get('seed', '?')}"
    )

  return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mjlab benchmark — task videos</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:24px; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;
         background:#0b0f17; color:#e5e7eb; }}
  header {{ margin-bottom:20px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:#9ca3af; font-size:13px; }}
  h2 {{ font-size:15px; margin:28px 0 12px; text-transform:uppercase;
        letter-spacing:.06em; color:#9ca3af; font-weight:600; }}
  .count {{ background:#1f2937; color:#9ca3af; border-radius:10px;
            padding:1px 8px; font-size:12px; margin-left:6px; }}
  .grid {{ display:grid; gap:16px;
           grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }}
  .card {{ background:#111827; border:1px solid #1f2937; border-radius:10px;
           overflow:hidden; }}
  video {{ width:100%; display:block; background:#000; aspect-ratio:4/3; }}
  .body {{ padding:10px 12px 12px; }}
  .title {{ font-weight:600; font-size:13px; margin-bottom:6px; word-break:break-word; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:5px; margin-bottom:6px; }}
  .chip {{ font-size:11px; padding:1px 7px; border-radius:20px;
           background:#1f2937; border:1px solid #374151; color:#d1d5db; }}
  .meta {{ font-size:11px; color:#6b7280; font-variant-numeric:tabular-nums; }}
  details {{ margin-top:8px; }} summary {{ cursor:pointer; font-size:11px; color:#6b7280; }}
  details p {{ font-size:12px; color:#9ca3af; margin:6px 0 0; }}
  .failed {{ color:#fca5a5; font-size:13px; }}
  .controls {{ display:flex; gap:8px; align-items:center; margin-top:10px; }}
  button {{ background:#1f2937; color:#e5e7eb; border:1px solid #374151;
            border-radius:6px; padding:5px 11px; font-size:12px; cursor:pointer; }}
  button:hover {{ background:#374151; }}
  input {{ background:#111827; color:#e5e7eb; border:1px solid #374151;
           border-radius:6px; padding:5px 10px; font-size:12px; min-width:220px; }}
</style></head>
<body>
<header>
  <h1>mjlab manipulation benchmark — task previews</h1>
  <div class="sub">{len(entries)} clips · {settings_line}</div>
  <div class="sub">Random agent: this shows scene layout and that the env steps — NOT task solvability.</div>
  <div class="controls">
    <button id="playall">play all</button>
    <button id="pauseall">pause all</button>
    <input id="filter" placeholder="filter by task name…">
  </div>
</header>
{"".join(cards_by_group)}
{failed_html}
<script>
  const vids = () => Array.from(document.querySelectorAll('video'));
  document.getElementById('playall').onclick = () => vids().forEach(v => v.play());
  document.getElementById('pauseall').onclick = () => vids().forEach(v => v.pause());
  document.getElementById('filter').addEventListener('input', e => {{
    const q = e.target.value.toLowerCase();
    document.querySelectorAll('.card').forEach(c => {{
      const t = c.querySelector('.title').textContent.toLowerCase();
      c.style.display = t.includes(q) ? '' : 'none';
    }});
  }});
  // Autoplay a clip while it is on screen; pause it once it scrolls away.
  const io = new IntersectionObserver(es => es.forEach(en => {{
    if (en.isIntersecting) en.target.play().catch(() => {{}});
    else en.target.pause();
  }}), {{threshold: 0.25}});
  vids().forEach(v => io.observe(v));
</script>
</body></html>
"""


@dataclass(frozen=True)
class ServeConfig:
  video_dir: Path = Path("benchmark_videos")
  port: int = 8000
  host: str = "0.0.0.0"
  write_html: bool = True
  """Also write index.html into video_dir, so the gallery can be opened offline."""


class _Handler(http.server.SimpleHTTPRequestHandler):
  def __init__(self, *args, video_dir: Path, **kwargs):
    self._video_dir = video_dir
    super().__init__(*args, directory=str(video_dir), **kwargs)

  def do_GET(self):  # noqa: N802 — http.server API
    if self.path in ("/", "/index.html"):
      body = build_index_html(self._video_dir).encode()
      self.send_response(200)
      self.send_header("Content-Type", "text/html; charset=utf-8")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
      return
    if self._serve_range():
      return
    super().do_GET()

  def _serve_range(self) -> bool:
    """Honour a single Range request so browsers can seek within a clip.

    SimpleHTTPRequestHandler ignores Range entirely and answers 200 with the whole
    file. Most browsers tolerate that for small clips but will not scrub reliably,
    so serve a proper 206 for the common ``bytes=start-[end]`` form.
    """
    header = self.headers.get("Range")
    if not header or not header.startswith("bytes="):
      return False
    path = Path(self.translate_path(self.path))
    if not path.is_file():
      return False
    try:
      start_s, _, end_s = header[len("bytes=") :].partition("-")
      size = path.stat().st_size
      start = int(start_s) if start_s else 0
      end = int(end_s) if end_s else size - 1
      end = min(end, size - 1)
      if start > end:
        return False
    except ValueError:
      return False

    length = end - start + 1
    ctype = self.guess_type(str(path))
    self.send_response(206)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    self.send_header("Accept-Ranges", "bytes")
    self.send_header("Content-Length", str(length))
    self.end_headers()
    with path.open("rb") as fh:
      fh.seek(start)
      remaining = length
      while remaining > 0:
        chunk = fh.read(min(64 * 1024, remaining))
        if not chunk:
          break
        self.wfile.write(chunk)
        remaining -= len(chunk)
    return True

  def log_message(self, fmt, *args):  # noqa: A002 — quieter logs
    if "index" in (args[0] if args else ""):
      super().log_message(fmt, *args)


def main(cfg: ServeConfig) -> None:
  video_dir = cfg.video_dir.resolve()
  if not video_dir.exists():
    raise SystemExit(f"video dir not found: {video_dir}")

  if cfg.write_html:
    (video_dir / "index.html").write_text(build_index_html(video_dir))
    print(f"wrote {video_dir / 'index.html'}")

  handler = partial(_Handler, video_dir=video_dir)
  socketserver.TCPServer.allow_reuse_address = True
  with socketserver.TCPServer((cfg.host, cfg.port), handler) as httpd:
    import socket

    print(f"serving {video_dir} at http://{socket.gethostname()}:{cfg.port}")
    print(f"  local:  http://localhost:{cfg.port}")
    print(f"  tunnel: ssh -N -L {cfg.port}:{socket.gethostname()}:{cfg.port} $USER@<login-host>")
    try:
      httpd.serve_forever()
    except KeyboardInterrupt:
      print("\nstopped")


if __name__ == "__main__":
  main(tyro.cli(ServeConfig))
