import sys
from pathlib import Path
from src.parser import load_all_exercises
from src.generator import generate_all_exercises, format_exercise
from src.storage import save_exercises, deduplicate, load_existing
from src.config import SOURCE_DIR, OUTPUT_FILE


def main():
    print("=" * 50)
    print("中考英语词性转换练习题生成系统")
    print("=" * 50)
    
    print("\n[1/4] 加载源文档...")
    exercises = load_all_exercises(str(SOURCE_DIR))
    print(f"  提取到 {len(exercises)} 道题目")
    
    print("\n[2/4] 去重...")
    exercises = deduplicate(exercises)
    print(f"  去重后 {len(exercises)} 道题目")
    
    print("\n[3/4] 生成答案 (调用DeepSeek API)...")
    results = generate_all_exercises(exercises)
    print(f"  生成完成 {len(results)} 道题目")
    
    print("\n[4/4] 保存结果...")
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(format_exercise(result, i))
    
    save_exercises(formatted, str(OUTPUT_FILE))
    print(f"  保存到 {OUTPUT_FILE}")
    
    print("\n" + "=" * 50)
    print(f"完成! 共生成 {len(formatted)} 道题目")
    print("=" * 50)


if __name__ == "__main__":
    main()
