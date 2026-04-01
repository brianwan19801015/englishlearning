"""
LangChain Prompt 模板 - 中考英语词性转换题生成

优化点：
1. 明确指定括号内是"源词"（动词原形/形容词原级）
2. 明确指定答案的词性类型
3. 添加语法检查规则
4. Few-shot 示例展示正确格式
"""

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

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

| 源词性 | 目标词性 | 示例 |
|--------|----------|------|
| 动词(v.) | 名词(n.) | act → ACT / advertise → ADVERTISEMENT |
| 动词(v.) | 副词(adv.) | rapid → RAPIDLY / happy → HAPPILY |
| 动词(v.) | 过去式 | deal → DEALT / dig → DUG |
| 形容词(adj.) | 副词(adv.) | cheerful → CHEERFULLY / sudden → SUDDENLY |
| 形容词(adj.) | 比较级 | cheap → CHEAPER / happy → HAPPIER |
| 形容词(adj.) | 最高级 | cheap → CHEAPEST / happy → HAPPIEST |
| 名词(n.) | 复数 | battery → BATTERIES / harmony → HARMONIES |
| 名词(n.) | 形容词 | tradition → TRADITIONAL / music → MUSICAL |

## 正确示例

示例 1：
- 源词：advertise（动词原形）
- 目标词性：名词
- 句子：The company decided to place an ______ in the newspaper to attract more customers.
- 答案：ADVERTISEMENT

示例 2：
- 源词：rapid（形容词原级）
- 目标词性：副词
- 句子：The river flows ______ after the heavy rain, so we need to be careful.
- 答案：RAPIDLY

示例 3：
- 源词：cheap（形容词原级）
- 目标词性：比较级
- 句子：The blue shirt is only $15, but the red one is even ______ than that.
- 答案：CHEAPER

示例 4：
- 源词：harmony（名词单数）
- 目标词性：复数
- 句子：The choir performed several beautiful ______ during the concert last night.
- 答案：HARMONIES

## 常见错误提醒

❌ 错误：括号给名词/形容词，答案直接用原词（无变形）
✅ 正确：括号给源词（动词原形/形容词原级），答案必须变形

❌ 错误：名词前忘记加冠词（a/an/the）
✅ 正确：可数名词前必须加冠词

❌ 错误：动词第三人称单数用错（如 acts → ACTS 在名词位置）
✅ 正确：区分动词形式和名词形式

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
