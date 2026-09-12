"""W&B credentials and destination scoped to the CL24 experiment process."""

import os
from pathlib import Path

ENTITY = "sudhirpratapyadav-indian-institute-of-technology-jodhpur"
PROJECT = "mjlab-cl24-rl-teachers-20260912"
PRIVATE_DIR = Path.home() / ".config/mjlab-cl24"
KEY_FILE = PRIVATE_DIR / "wandb_api_key"


def configure(entity=ENTITY, project=PROJECT, key_file=KEY_FILE, offline=False):
  if not offline:
    key = Path(key_file).read_text().strip()
    if not key or any(character.isspace() for character in key):
      raise ValueError("The experiment W&B key is empty or malformed")
    os.environ["WANDB_API_KEY"] = key
  os.environ["WANDB_ENTITY"] = entity
  os.environ["WANDB_PROJECT"] = project
  os.environ["WANDB_MODE"] = "offline" if offline else "online"
  os.environ["WANDB_CONFIG_DIR"] = str(PRIVATE_DIR / "sdk")
  os.environ["NETRC"] = str(PRIVATE_DIR / "netrc")
  # RSL-RL's writer treats WANDB_USERNAME as an explicit entity, overriding
  # WANDB_ENTITY. Remove an inherited value only in this experiment process.
  os.environ.pop("WANDB_USERNAME", None)
