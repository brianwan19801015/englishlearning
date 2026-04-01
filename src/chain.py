from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


def get_llm():
    """获取LLM实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL + "/v1",
        temperature=0.3,
        max_tokens=100
    )


def create_word_transform_prompt(context: str, root: str):
    """创建词性转换提示词"""
    system_msg = SystemMessage(content="""You are an experienced middle school English teacher in China. Your task is to help students practice word transformation (词性转换).

Given a sentence with a blank and a root word in parentheses, determine the CORRECT form of the word to fill the blank.

Rules:
1. Analyze the sentence to understand what grammatical form is needed (noun, verb, adjective, adverb, etc.)
2. Apply proper word transformation rules (e.g., adjective → adverb, noun → verb, etc.)
3. Return ONLY the transformed word in uppercase
4. NO explanation, just the answer

Examples:
- "She looks ____" (happy) → HAPPY
- "He runs very ____" (quick) → QUICKLY
- "The ____ of this book is interesting" (difficult) → DIFFICULTY""")
    
    human_msg = HumanMessage(content=f'Sentence: "{context}"\nRoot word: {root}\nAnswer:')
    
    return [system_msg, human_msg]


def get_word_transform_answer(context: str, root: str) -> str:
    """获取词性转换答案"""
    llm = get_llm()
    messages = create_word_transform_prompt(context, root)
    response = llm.invoke(messages)
    answer = response.content.strip().upper()
    
    # Clean up the answer
    answer = answer.replace("**", "").replace("ANSWER:", "").strip()
    answer = answer.split()[0] if answer.split() else root.upper()
    
    # Remove any punctuation
    answer = ''.join(c for c in answer if c.isalnum() or c == '-')
    
    return answer
