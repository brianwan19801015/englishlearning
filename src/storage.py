import json
from pathlib import Path


def load_existing(file_path: str) -> list[dict]:
    """加载已存在的JSON文件"""
    path = Path(file_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_exercises(exercises: list[dict], file_path: str):
    """保存练习题到JSON文件"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exercises, f, ensure_ascii=False, indent=2)


def deduplicate(exercises: list[dict]) -> list[dict]:
    """去重"""
    seen = set()
    unique = []
    
    for ex in exercises:
        ctx = ex.get("context", "")
        if ctx not in seen:
            seen.add(ctx)
            unique.append(ex)
    
    return unique
