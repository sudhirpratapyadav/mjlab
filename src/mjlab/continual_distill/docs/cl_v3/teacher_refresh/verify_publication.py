"""Check the public HTTPS files against the exact reviewed local artifacts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=Path, default=here / "videos/Mjlab-Reorient-Object-Franka")
parser.add_argument("--report", type=Path, default=here / "publication_verification.json")
args = parser.parse_args()
folder = args.video
task_id = json.loads((folder / "result.json").read_text())["task_id"]
expected = {
  "index.html": hashlib.sha256(
    (here.parent / "site/index.html").read_bytes()
  ).hexdigest()
}
for name in ("result.json", "task.json", "teacher.mp4", "failure.mp4", "thumb.jpg"):
  if not (folder / name).exists():
    assert name in ("teacher.mp4", "failure.mp4"), name
    continue
  expected[f"{task_id}/{name}"] = hashlib.sha256(
    (folder / name).read_bytes()
  ).hexdigest()
for name in ("teacher.mp4", "failure.mp4"):
  if not (folder / name).exists():
    continue
  reader = imageio_ffmpeg.read_frames(str(folder / name))
  metadata = next(reader)
  reader.close()
  assert metadata["size"] == (1920, 1080), metadata
  assert metadata["fps"] == 50
  subprocess.run(
    [
      imageio_ffmpeg.get_ffmpeg_exe(),
      "-v",
      "error",
      "-i",
      str(folder / name),
      "-f",
      "null",
      "-",
    ],
    check=True,
  )
script = f"""import hashlib,json
from urllib.request import urlopen
expected=json.loads({json.dumps(expected)!r})
base="https://cl.sudhirpratapyadav.com/v3/"
verified=[]
for path,digest in expected.items():
    with urlopen(base+path,timeout=30) as response:
        data=response.read()
        assert response.status == 200
    assert hashlib.sha256(data).hexdigest()==digest,path
    verified.append({{"url":base+path,"status":200,"sha256":digest}})
print(json.dumps({{"files":verified,"public_content_matches_reviewed_artifacts":True}},indent=2))
"""
result = subprocess.run(
  ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "untu_vps", "python3", "-"],
  input=script,
  text=True,
  capture_output=True,
  check=True,
)
verified = json.loads(result.stdout)
verified["videos_decoded"] = sum((folder / name).exists() for name in ("teacher.mp4", "failure.mp4"))
verified["video_size"] = [1920, 1080]
verified["video_fps"] = 50
args.report.write_text(
  json.dumps(verified, indent=2) + "\n"
)
print(json.dumps(verified, indent=2))
