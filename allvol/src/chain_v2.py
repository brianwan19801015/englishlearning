"""
LangChain Prompt 模板 -中考英语词性转换题生成

优化点：
1. 明确指定括号内是"源词"（动词原形/形容词原级）
2. 明确指定答案的词性类型
3. 添加语法检查规则
4. Few-shot 示例展示正确格式
5. 集成自动校验（Checker）
"""

import json
import re
from typing import List, Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# 导入校验器
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checker import ExerciseChecker

# ============================================
# 核心 Prompt 模板
# ============================================

WORD_TRANSFORMATION_TEMPLATE = """你是一个中考英语出题专家。请根据给定的词汇生成词性转换练习题。

## 任务要求

1. **括号内的词必须是源词**：
   - 动词 → 给出动词原形（如：act, advertise, cancel）
   - 形容词 → 给出形容词原级（如：cheap, happy, baggy）
   - 名词 → 给出名词单数形式（如：injury, harmony）

2. **答案必须是目标词性**：
   - 动词 → 名词/副词/第三人称单数/过去式/过去分词
   - 形容词 → 副词/比较级/最高级
   - 名词 → 复数/形容词

3. **句子必须符合英语语法**：
   - 名词前需要冠词（a/an/the）
   - 形容词修饰名词要正确位置
   - 副词修饰动词/形容词要正确位置

## 输出格式（JSON）

```json
{
  "id": "g8_001",
  "type": "word_transformation",
  "section": "grade8",
  "difficulty": "medium",
  "context": "完整的句子，单词用 ______ 填空，词性提示放在括号内",
  "correct_answers": ["正确答案，全大写"],
  "is_vocab": [true],
  "vocab_count": 1,
  "blanks_count": 1,
  "created_at": "2026-03-31T12:00:00.000Z"
}
```

## 词性转换规则参考

### ✅ 必须有形态变化

词性转换必须包含**拼写变化**，以下情况**不算**词性转换：

| 源词 → 答案 | 问题 | 正确做法 |
|-------------|------|----------|
| act → ACT | 拼写完全相同（仅大小写变化）❌ | act → ACTION ✅ |
| plan → PLAN | 拼写完全相同 ❌ | plan → PLANNING ✅ |
| say → SAID | 这是不规则变化，可以 ✅ | - |

### 有效转换类型

| 源词性 | 目标词性 | 示例 | 变化类型 |
|--------|----------|------|----------|
| 动词(v.) | 名词(n.) | act → ACTION / advertise → ADVERTISEMENT | 加后缀 (-ion, -ment) |
| 动词(v.) | 副词(adv.) | rapid → RAPIDLY / happy → HAPPILY | 加后缀 (-ly) |
| 动词(v.) | 过去式 | deal → DEALT / dig → DUG | 不规则变化 |
| 形容词(adj.) | 副词(adv.) | cheerful → CHEERFULLY / sudden → SUDDENLY | 加后缀 (-ly) |
| 形容词(adj.) | 比较级 | cheap → CHEAPER / happy → HAPPIER | 加后缀 (-er) |
| 形容词(adj.) | 最高级 | cheap → CHEAPEST / happy → HAPPIEST | 加后缀 (-est) |
| 名词(n.) | 复数 | battery → BATTERIES / harmony → HARMONIES | 加后缀 (-s/-es) |
| 名词(n.) | 形容词 | tradition → TRADITIONAL / music → MUSICAL | 加后缀 (-al) |

## 正确示例

示例 1：
- 源词：advertise（动词原形）
- 目标词性：名词
- 句子：The company decided to place an ______ in the newspaper to attract more customers.
- 答案：ADVERTISEMENT
- 变化：advertise → advertisement（加 -ment 后缀）✅

示例 2：
- 源词：rapid（形容词原级）
- 目标词性：副词
- 句子：The river flows ______ after the heavy rain, so we need to be careful.
- 答案：RAPIDLY
- 变化：rapid → rapidly（加 -ly 后缀）✅

示例 3：
- 源词：cheap（形容词原级）
- 目标词性：比较级
- 句子：The blue shirt is only $15, but the red one is even ______ than that.
- 答案：CHEAPER
- 变化：cheap → cheaper（加 -er 后缀）✅

示例 4：
- 源词：harmony（名词单数）
- 目标词性：复数
- 句子：The choir performed several beautiful ______ during the concert last night.
- 答案：HARMONIES
- 变化：harmony → harmonies（加 -es 后缀）✅

## 常见错误提醒

❌ 错误：括号给名词/形容词，答案直接用原词（无拼写变化）
   例如：act → ACT（仅大小写变化，不算词性转换）
✅ 正确：括号给源词（动词原形/形容词原级），答案必须变形

❌ 错误：名词前忘记加冠词（a/an/the）
   例如：announced new action（应为 a new action）
✅ 正确：可数名词前必须加冠词

❌ 错误：动词第三人称单数用错位置
   例如：The main attraction is... → 用了 ATTRACTS（动词形式）
   应该是：The main attraction is... → ATTRACTION（名词形式）
✅ 正确：区分动词形式和名词形式在句子中的位置

## 现在开始生成

请生成 10 道词性转换题，词汇来自初中英语教材（八年级上册）：
{word_list}

请直接输出 JSON 数组格式，不要有其他内容。
"""

# 创建 Prompt 模板
word_transformation_prompt = PromptTemplate(
    template=WORD_TRANSFORMATION_TEMPLATE,
    input_variables=["word_list"]
)


# ============================================
# 校验 Prompt（用于后处理检查）
# ============================================

VALIDATION_TEMPLATE = """你是一个英语语法检查专家。请检查以下题目是否有错误。

## 检查项目

1. **答案是否正确变形**：括号内的词是否正确转换为目标词性
2. **语法是否正确**：冠词、介词、搭配是否正确
3. **逻辑是否合理**：句子意思是否通顺

## 题目列表

{exercises}

## 输出格式

对于每道题，检查是否有问题。如果有问题，指出问题并给出修正建议。

输出 JSON 格式：
```json
[
  {
    "id": "g8_001",
    "has_error": true/false,
    "error_type": "答案错误/语法错误/逻辑错误",
    "suggestion": "修正建议"
  }
]
```
"""

validation_prompt = PromptTemplate(
    template=VALIDATION_TEMPLATE,
    input_variables=["exercises"]
)


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 示例词汇列表
    word_list = """act, advertise, cancel, cheap, cheerful, deal, death, destroy, discover, donate"""

    # 创建 Chain
    llm = OpenAI(temperature=0.7)
    chain = LLMChain(llm=llm, prompt=word_transformation_prompt)

    # 生成题目
    result = chain.run(word_list=word_list)
    print(result)


# ============================================
# 集成 Checker - 生成后自动校验
# ============================================

import json
import re
from typing import List, Dict, Any


class ExerciseGenerator:
    """带自动校验的练习题生成器"""

    def __init__(self, llm):
        self.llm = llm
        self.chain = LLMChain(llm=llm, prompt=word_transformation_prompt)
        self.checker = ExerciseChecker()

    def generate(self, word_list: str, max_retries: int = 3) -> List[Dict]:
        """
        生成练习题并自动校验

        Args:
            word_list: 词汇列表（逗号分隔）
            max_retries: 最大重试次数

        Returns:
            通过校验的练习题列表
        """
        # 1. 生成题目
        result = self.chain.run(word_list=word_list)

        # 2. 解析 JSON
        try:
            exercises = json.loads(result)
            if isinstance(exercises, dict):
                exercises = [exercises]
        except json.JSONDecodeError:
            print("❌ JSON 解析失败")
            return []

        # 3. 校验每道题
        valid_exercises = []
        invalid_exercises = []

        for exercise in exercises:
            issues = self.checker.check_exercise(exercise)
            if issues:
                invalid_exercises.append({
                    'exercise': exercise,
                    'issues': issues
                })
            else:
                valid_exercises.append(exercise)

        # 4. 输出报告
        print(f"\n📊 生成报告")
        print(f"  总生成: {len(exercises)}")
        print(f"  ✅ 通过: {len(valid_exercises)}")
        print(f"  ❌ 失败: {len(invalid_exercises)}")

        if invalid_exercises:
            print(f"\n⚠️ 失败原因:")
            for item in invalid_exercises:
                print(f"  - {item['exercise'].get('id', 'unknown')}: {item['issues'][0]['type']}")

        # 5. 如果通过率太低，重试
        if len(valid_exercises) < len(exercises) * 0.5 and max_retries > 0:
            print(f"\n🔄 通过率过低，剩余重试次数: {max_retries}")
            # 可以在这里添加重试逻辑

        return valid_exercises
