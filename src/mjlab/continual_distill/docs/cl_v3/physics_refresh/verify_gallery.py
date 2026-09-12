"""Decode local clips and verify published bytes and metadata over HTTPS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import ssl
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import imageio_ffmpeg

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--expected', type=int, default=25)
a = p.parse_args()
folders = sorted(path.parent for path in (a.root / 'videos').glob('*/result.json'))
if not a.expected:
  folders = [folder for folder in folders if (folder / 'published.json').exists()]
if a.expected:
  assert len(folders) == a.expected, (len(folders), a.expected)
base = 'https://cl.sudhirpratapyadav.com/v3/'
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
# uv's Python may not locate the host's RHEL trust store automatically.
ca_bundle = Path('/etc/pki/tls/certs/ca-bundle.crt')
ssl_context = ssl.create_default_context(cafile=str(ca_bundle) if ca_bundle.exists() else None)

def verify(folder):
  result = json.loads((folder / 'result.json').read_text())
  with urlopen(base + folder.name + '/result.json', timeout=30, context=ssl_context) as response:
    remote = json.load(response)
  assert result == remote, f'Stale result: {folder.name}'
  clips = []
  for filename in ('teacher.mp4', 'failure.mp4', 'thumb.jpg'):
    path = folder / filename
    if not path.exists():
      try:
        with urlopen(Request(base + folder.name + '/' + filename, method='HEAD'),
                     timeout=30, context=ssl_context):
          raise AssertionError(f'Obsolete remote file still present: {folder.name}/{filename}')
      except HTTPError as error:
        assert error.code == 404, (folder.name, filename, error.code)
      continue
    if path.suffix == '.mp4':
      subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    with urlopen(base + folder.name + '/' + filename, timeout=60, context=ssl_context) as response:
      digest = hashlib.sha256(response.read()).hexdigest()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest(), f'Stale bytes: {path}'
    clips.append({'file': filename, 'sha256': digest, 'bytes': path.stat().st_size})
  return {'task_id': folder.name, 'successes': result['num_success'], 'n': result['n'],
          'source_sha256': result['source_sha256'], 'verified_files': clips}

with ThreadPoolExecutor(max_workers=4) as pool:
  verified = list(pool.map(verify, folders))
pages = []
for url, path in [(base, Path(__file__).resolve().parent.parent / 'site/index.html'),
                  ('https://cl.sudhirpratapyadav.com/', a.root / 'home.html')]:
  with urlopen(url, timeout=30, context=ssl_context) as response:
    assert response.read() == path.read_bytes(), f'Stale gallery page: {url}'
  pages.append(url)
with urlopen('https://cl.sudhirpratapyadav.com/v3_before_physics_20260910/',
             timeout=30, context=ssl_context) as response:
  assert response.status == 200
report = {'tasks_verified': len(verified), 'pages_verified': pages, 'total_trials': sum(x['n'] for x in verified),
          'tasks_at_90_percent': sum(x['successes'] / x['n'] >= .9 for x in verified),
          'tasks': verified}
(a.root / ('verification.json' if a.expected else 'verification_partial.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'tasks'}, indent=2))
