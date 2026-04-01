from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import BATCH_SIZE, MAX_WORKERS, SECTION, DIFFICULTY
from .chain import get_word_transform_answer


def generate_exercise(exercise: dict) -> dict:
    """生成单个练习题"""
    context = exercise["context"]
    root = exercise["root"]
    
    answer = get_word_transform_answer(context, root)
    
    return {
        "context": context,
        "root": root,
        "answer": answer
    }


def generate_exercises_batch(exercises: list[dict], batch_size: int = BATCH_SIZE):
    """批量生成练习题"""
    results = []
    total = len(exercises)
    
    for i in range(0, total, batch_size):
        batch = exercises[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_exercise, ex): ex for ex in batch}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Processed {len(results)}/{total}")
    
    return results


def generate_all_exercises(exercises: list[dict]) -> list[dict]:
    """生成所有练习题"""
    return generate_exercises_batch(exercises)


def format_exercise(result: dict, index: int) -> dict:
    """格式化输出"""
    return {
        "id": f"{SECTION}_{index:03d}",
        "type": "word_transformation",
        "section": SECTION,
        "difficulty": DIFFICULTY,
        "context": result["context"],
        "correct_answers": [result["answer"]],
        "is_vocab": [True],
        "vocab_count": 1,
        "blanks_count": 1,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
