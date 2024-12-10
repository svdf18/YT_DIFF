import os
from dotenv import load_dotenv
from typing import Union
from json import dumps as json_dumps, load as json_load

def load_json(json_path: str) -> dict:
    """Load JSON file"""
    with open(json_path, "r") as f:
        return json_load(f)
    
def save_json(data: Union[dict, list], json_path: str, indent: int = 2) -> None:
    """Save JSON file with proper directory creation"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        f.write(json_dumps(data, indent=indent))

# Load environment variables
load_dotenv(override=True)

# Core paths
CONFIG_PATH = os.getenv("CONFIG_PATH")
MODELS_PATH = os.getenv("MODELS_PATH")
DEBUG_PATH = os.getenv("DEBUG_PATH")
SRC_PATH = os.getenv("PYTHONPATH")
CACHE_PATH = os.getenv("CACHE_PATH")

# Data and tools paths
DATASET_PATH = os.getenv("DATASET_PATH")
FFMPEG_PATH = os.getenv("FFMPEG_PATH")

# Optional settings
NO_GUI = int(os.getenv("NO_GUI") or 0) == 1
