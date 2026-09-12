"""Save a W&B key for this experiment without changing the shared W&B login."""

import getpass
import os
from pathlib import Path
import sys
import tempfile


KEY_FILE = Path.home() / ".config/mjlab-cl24/wandb_api_key"


def main():
  if not sys.stdin.isatty():
    raise SystemExit("Run this command in your terminal to enter the key privately.")
  print("Open https://wandb.ai/authorize while signed in as sudhirpratapyadav.")
  key = getpass.getpass("Paste the W&B API key (input is hidden): ").strip()
  if not key or any(character.isspace() for character in key):
    raise SystemExit("No key saved: the key must be nonempty and contain no whitespace.")
  KEY_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  fd, temporary_path = tempfile.mkstemp(prefix=".wandb-key-", dir=KEY_FILE.parent)
  try:
    with os.fdopen(fd, "w") as stream:
      stream.write(key + "\n")
    os.replace(temporary_path, KEY_FILE)
  finally:
    if os.path.exists(temporary_path):
      os.unlink(temporary_path)
  print(f"Saved privately to {KEY_FILE} (permissions 600).")
  print("Tell the assistant 'done'; it will verify access and create the project.")
  print("Training has not been started. The global W&B login has not been changed.")


if __name__ == "__main__":
  main()
