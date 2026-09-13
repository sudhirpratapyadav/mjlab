"""Verify HTTPS content and actual encoded frame counts for a motion review."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request

import imageio_ffmpeg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--review-dir', type=Path, required=True)
  parser.add_argument('--public-url', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert not args.output.exists(), 'Verification receipt already exists'
  assert args.public_url.startswith('https://cl.sudhirpratapyadav.com/v4-rl/')
  metadata = json.loads((args.review_dir / 'videos.json').read_text())
  assert metadata['paired_initial_states_verified'] and metadata['paired_friction_verified']
  assert len(metadata['clips']) == 6
  encoded = []
  for clip in metadata['clips']:
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(args.review_dir / clip['video']))
    assert clip['fps'] == 50 and frames == clip['frames']
    assert abs(seconds - clip['simulation_seconds']) < 1e-6
    assert abs(seconds - frames / 50) < 1e-6
    encoded.append(dict(video=clip['video'], frames=frames, seconds=seconds, real_time=True))
  context = ssl.create_default_context(cafile='/etc/pki/tls/certs/ca-bundle.crt')
  files = []
  for name in ['index.html', 'summary.json'] + [clip['video'] for clip in metadata['clips']]:
    local = (args.review_dir / name).read_bytes()
    with urllib.request.urlopen(args.public_url.rstrip('/') + '/' + name,
                                context=context, timeout=60) as response:
      assert response.status == 200
      remote = response.read()
    assert hashlib.sha256(remote).digest() == hashlib.sha256(local).digest(), name
    files.append(dict(file=name, status=200, sha256=hashlib.sha256(remote).hexdigest()))
  report = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                url=args.public_url, files=files, encoded_video_verification=encoded,
                paired_initial_states_verified=True, paired_friction_verified=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(f'Verified {len(files)} HTTPS files and {len(encoded)} real-time videos: {args.public_url}')


if __name__ == '__main__':
  main()
