# 中考英语词性转换练习题生成系统

基于 LangChain + DeepSeek API 的自动化出题系统

## 项目结构

```
allvol/
├── zhongkao/                    # 源Word文档
│   └── *.docx
├── src/
│   ├── __init__.py
│   ├── config.py                # 配置
│   ├── parser.py                # Word文档解析
│   ├── chain.py                 # LangChain链定义
│   ├── generator.py             # 题目生成器
│   └── storage.py               # JSON存储
├── data/
│   └── zhongkao-v1.json         # 生成题目
├── main.py                     # 入口
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python main.py
```

## 题目格式

```json
{
  "id": "zhongkao_v1_001",
  "type": "word_transformation",
  "section": "zhongkao_v1",
  "difficulty": "medium",
  "context": "The population there has increased ______ in the past few years. (rapid)",
  "correct_answers": ["RAPIDLY"],
  "is_vocab": [true],
  "vocab_count": 1,
  "blanks_count": 1,
  "created_at": "2026-04-01T00:00:00.000Z"
}
```
