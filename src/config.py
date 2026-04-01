import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "zhongkao"
OUTPUT_FILE = BASE_DIR / "data" / "zhongkao-v1.json"

DEEPSEEK_API_KEY = "sk-8b802cc484be4bc8aa8b3b2541d86b83"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SECTION = "zhongkao_v1"
DIFFICULTY = "medium"

BATCH_SIZE = 50
MAX_WORKERS = 5
