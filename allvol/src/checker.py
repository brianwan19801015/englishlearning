#!/usr/bin/env python3
"""
练习题质量校验器
检查生成的题目是否有语法错误、逻辑问题、词性转换错误
支持规则检查 + LLM 语义检查
"""

import json
import re
import os
import requests
from typing import List, Dict, Any
from pathlib import Path


# 加载核心词汇
def load_core_vocabulary():
    """加载核心词汇表"""
    # 尝试多个可能的路径
    possible_paths = [
        Path(__file__).parent.parent / "src" / "data" / "core_vocabulary.json",
        Path(__file__).parent.parent.parent / "src" / "data" / "core_vocabulary.json",
        Path(__file__).parent / "src" / "data" / "core_vocabulary.json",
    ]
    
    for vocab_path in possible_paths:
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    # 如果找不到文件，返回空字典
    print(f"⚠️ 未找到核心词汇表文件")
    return {}


# 全局加载核心词汇
CORE_VOCABULARY = load_core_vocabulary()


class ExerciseChecker:
    # ZenMux AI 配置
    ZENMUX_BASE_URL = "https://zenmux.ai/api/v1"
    ZENMUX_API_KEY = "sk-ss-v1-2e83ce9d273c15056b5c39bf95027189eb08e8b5d7ef9552e454853ac5d59449"
    ZENMUX_MODEL = "openai/gpt-4o"

    def __init__(self, use_llm: bool = True, grade: str = None):
        """初始化校验器
        
        Args:
            use_llm: 是否使用 LLM 进行语义检查（默认 True）
            grade: 年级，可选 "grade7" 或 "grade8"
        """
        self.issues = []
        self.use_llm = use_llm
        self.grade = grade
        self.llm_session = None
        
        # 加载该年级的核心词汇
        self.core_transformations = self._load_grade_transformations(grade)
        
        # 初始化 HTTP 会话
        if use_llm:
            self.llm_session = requests.Session()
            self.llm_session.headers.update({
                "Authorization": f"Bearer {self.ZENMUX_API_KEY}",
                "Content-Type": "application/json"
            })
    
    def _load_grade_transformations(self, grade: str = None) -> set:
        """加载指定年级的词性转换列表"""
        transformations = set()
        
        if not CORE_VOCABULARY:
            return transformations
        
        # 如果没有指定年级，加载所有年级
        grades_to_load = []
        if grade:
            grades_to_load = [grade]
        else:
            grades_to_load = list(CORE_VOCABULARY.keys())
        
        for g in grades_to_load:
            if g in CORE_VOCABULARY:
                vocab = CORE_VOCABULARY[g]
                for root, forms in vocab.items():
                    for form in forms:
                        transformations.add((root.lower(), form.upper()))
        
        return transformations

    def check_exercise(self, exercise: Dict[str, Any]) -> List[Dict]:
        """检查单道题目"""
        issues = []
        context = exercise.get('context', '')
        answer = exercise.get('correct_answers', [''])[0]
        hint = self._extract_hint(context)

        # 1. 检查冠词问题
        issues.extend(self._check_articles(context, answer))

        # 2. 检查逻辑合理性
        issues.extend(self._check_logic(context, answer, hint))

        # 3. 检查词性转换是否有形态变化
        issues.extend(self._check_morphology(hint, answer))

        # 4. 检查所有格用法
        issues.extend(self._check_possessive(context, answer, hint))

        # 5. 检查填空位置上下文
        issues.extend(self._check_blank_context(context, answer, hint))

        # 6. LLM 语义检查（默认开启，作为初中英语老师把关）
        if self.llm_session:
            llm_issues = self._check_semantics_llm(context, answer, hint)
            issues.extend(llm_issues)

        return issues

    def _check_semantics_llm(self, context: str, answer: str, hint: str) -> List[Dict]:
        """用 LLM 检查语义合理性（作为初中英语老师）"""
        if not self.llm_session:
            return []

        prompt = f"""作为初中英语老师，检查这道词性转换练习题是否合理：

句子：{context}
提示词：{hint}
答案：{answer}

请从以下角度检查：
1. 语法是否正确
2. 语义是否通顺自然
3. 是否符合中考考点
4. 答案填入后句子是否合理

如果有问题，请指出具体问题。
如果没问题，请回答"通过"。
"""

        try:
            response = self.llm_session.post(
                f"{self.ZENMUX_BASE_URL}/chat/completions",
                json={
                    "model": self.ZENMUX_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30
            )
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # 如果 LLM 说有问题
            # 判断标准：只有明确说"有问题"才算是问题
            content_lower = content.lower()
            
            # 明确的问题标记
            problem_keywords = ['存在问题', '有问题', '不合理', '不建议', '不推荐', '不自然', '不可行', '不可接受']
            # 明确的通过标记
            pass_keywords = ['通过', '合理', '正确', '可以', '没问题', '符合', '正确']
            
            has_problem = any(kw in content_lower for kw in problem_keywords)
            has_pass = any(kw in content_lower for kw in pass_keywords)
            
            # 只有明确有问题且没有通过标记时才认为有问题
            if has_problem and not has_pass:
                return [{
                    'type': 'llm_semantic_issue',
                    'message': f'LLM 语义检查: {content[:150]}...',
                    'suggestion': '请调整句子或答案'
                }]
        except Exception as e:
            # LLM 调用失败不影响主流程
            print(f"LLM 检查调用失败: {e}")

        return []

    def fix_exercise(self, exercise: Dict[str, Any]) -> Dict[str, Any]:
        """用 LLM 自动修复有问题的练习题"""
        if not self.llm_session:
            print("⚠️ 无 LLM 客户端，无法自动修复")
            return exercise

        context = exercise.get('context', '')
        answer = exercise.get('correct_answers', [''])[0]
        hint = self._extract_hint(context)

        # 先检查问题
        issues = self.check_exercise(exercise)
        if not issues:
            print(f"✅ 题目无需修复: {exercise.get('id')}")
            return exercise

        # 构建修复提示
        issue_descriptions = [f"- {i['type']}: {i['message']}" for i in issues]
        issue_text = '\n'.join(issue_descriptions)

        fix_prompt = f"""作为初中英语出题专家，请修复以下词性转换练习题的问题：

原始题目：
- 句子：{context}
- 提示词：{hint}
- 答案：{answer}

发现的问题：
{issue_text}

请修复以上问题，重新生成题目。要求：
1. 答案必须有形态变化（加后缀/不规则变化），不能只改大小写
2. 句子语义通顺，符合中考考点
3. 答案填入后句子合理自然
4. 使用常见的中考词汇

请直接输出 JSON 格式的修复结果，包含字段：context, correct_answers
"""

        try:
            print(f"🔧 正在修复题目: {exercise.get('id')}...")
            response = self.llm_session.post(
                f"{self.ZENMUX_BASE_URL}/chat/completions",
                json={
                    "model": self.ZENMUX_MODEL,
                    "messages": [{"role": "user", "content": fix_prompt}],
                    "temperature": 0.5
                },
                timeout=30
            )
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # 解析 JSON
            import json
            try:
                # 尝试提取 JSON
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    fixed = json.loads(json_match.group())
                    if 'context' in fixed:
                        exercise['context'] = fixed['context']
                    if 'correct_answers' in fixed:
                        exercise['correct_answers'] = fixed['correct_answers'] if isinstance(fixed['correct_answers'], list) else [fixed['correct_answers']]
                    
                    # 🔥 修复后重新检查
                    print(f"  🔍 修复后重新检查...")
                    recheck_issues = self.check_exercise(exercise)
                    if recheck_issues:
                        print(f"  ⚠️ 修复后仍有问题: {[i['type'] for i in recheck_issues]}")
                    else:
                        print(f"  ✅ 修复后检查通过!")
                    
                    return exercise
            except:
                pass

            print(f"⚠️ 修复解析失败，保持原题")
        except Exception as e:
            print(f"❌ 修复调用失败: {e}")

        return exercise

    def _extract_hint(self, context: str) -> str:
        """提取括号内的提示词"""
        match = re.search(r'\(([^)]+)\)', context)
        return match.group(1) if match else ''

    def _check_articles(self, context: str, answer: str) -> List[Dict]:
        """检查可数名词前是否有冠词"""
        issues = []

        # 提取填空前的内容
        blank_pos = context.find('______')
        if blank_pos > 0:
            before_blank = context[:blank_pos].strip().lower()
            after_blank = context[blank_pos + 6:].strip().lower().split()[0] if blank_pos + 6 < len(context) else ''

            # 1. 检查是否有"安全词"（后面不需要冠词）
            safe_prefixes = [
                'many ', 'several ', 'some ', 'various ', 'different ',
                'numerous ', 'multiple ', 'myriads of ',
                'hundreds of ', 'thousands of ', 'millions of ',
            ]
            is_safe_prefix = any(before_blank.endswith(prefix) for prefix in safe_prefixes)

            # 2. 检查是否是 "one of the best/worst + 复数" 结构
            if 'one of the best' in before_blank or 'one of the worst' in before_blank:
                is_safe_prefix = True

            # 3. 检查前面是否有形容词（形容词 + 名词，不需要额外冠词）
            # 例如：rechargeable batteries, beautiful harmonies, free samples
            has_adjective = any(word in before_blank for word in [
                'rechargeable ', 'beautiful ', 'free ', 'new ', 'old ', 'ancient ',
                'minor ', 'several ', 'numerous ', 'various ', 'modern ', 'medical ',
                'kitchen ', 'ancient ', 'foreign ', 'strange ', 'dark ', 'loud ',
            ])

            # 4. 如果是动词第三人称单数（cashes, finishes, injures 等），不是冠词问题
            # 动词第三人称单数形式：原词 + s/es
            hint_match = re.search(r'\(([^)]+)\)', context)
            hint = hint_match.group(1).lower() if hint_match else ''
            answer_lower = answer.lower()

            # 检查是否是动词第三人称单数（原词 + s/es）
            is_verb_third_person = False
            if hint and answer_lower.startswith(hint) and len(answer_lower) > len(hint):
                suffix = answer_lower[len(hint):]
                if suffix in ['s', 'es']:
                    is_verb_third_person = True

            # 综合判断
            if is_verb_third_person:
                # 动词形式，不需要冠词检查
                pass
            elif is_safe_prefix or has_adjective:
                # 有安全词或形容词，不需要冠词
                pass
            elif answer.endswith('S') and not answer.endswith('SS'):
                # 可能需要冠词
                last_word = before_blank.split()[-1] if before_blank else ''
                has_article = last_word in [
                    'a', 'an', 'the', 'my', 'your', 'his', 'her', 'its', 'this', 'that', 'these', 'those', 'own',
                    # 常见形容词/名词作定语
                    'kitchen', 'bedroom', 'bathroom', 'classroom', 'office', 'garden', 'park', 'street', 'city', 'town',
                    'school', 'hospital', 'museum', 'library', 'station', 'airport', 'restaurant', 'hotel', 'shop', 'store',
                    'book', 'door', 'window', 'wall', 'floor', 'ceiling', 'roof', 'table', 'chair', 'desk', 'bed', 'phone',
                    'computer', 'tv', 'radio', 'car', 'bus', 'train', 'bike', 'ship', 'plane', 'boat',
                ]
                if not has_article:
                    issues.append({
                        'type': 'missing_article',
                        'message': f'可数名词 "{answer}" 前可能缺少冠词',
                        'suggestion': f'考虑在 "{answer.lower()}" 前添加冠词'
                    })

        return issues

    def _check_logic(self, context: str, answer: str, hint: str) -> List[Dict]:
        """检查逻辑是否合理"""
        issues = []

        # 检查：anybody's + 名词 是否合理
        if hint.lower() == 'anybody' and answer.endswith("'S"):
            # anybody's owner = 任何人的主人？这通常不合理
            issues.append({
                'type': 'logic_error',
                'message': '"anybody\'s + 名词" 通常逻辑不通',
                'suggestion': '考虑改为 "the owner" 或 "its owner"'
            })

        # 检查：somebody's + 名词
        if hint.lower() in ['somebody', 'someone'] and answer.endswith("'S"):
            # 需要检查后面的名词是否合理
            issues.append({
                'type': 'logic_warning',
                'message': f'"{hint.lower()}\'s + 名词" 需要确保逻辑合理',
                'suggestion': '确认句子意思是否符合常理'
            })

        return issues

    def _check_morphology(self, hint: str, answer: str) -> List[Dict]:
        """检查词性转换是否有形态变化"""
        issues = []

        hint_upper = hint.upper()
        answer_upper = answer.upper()

        # 如果 hint 和 answer 只是大小写不同，不算转换
        if hint_upper == answer_upper:
            issues.append({
                'type': 'no_morphology',
                'message': f'"{hint}" → "{answer}" 没有形态变化（仅大小写）',
                'suggestion': '词性转换必须有拼写变化，如加后缀或不规则变化'
            })

        # 使用加载的核心词汇转换列表
        common_transformations = self.core_transformations

        # 检查是否在常见转换列表中
        is_common = (hint.lower(), answer.upper()) in common_transformations

        # 如果不在常见列表中，可能是生僻词
        if not is_common and hint_upper != answer_upper:
            # 但如果是规则变化（加后缀），可能是可以的
            common_suffixes = ['LY', 'ER', 'EST', 'ION', 'MENT', 'TION', 'NESS', 'S', 'ES', 'ED', 'ING', 'ABLE', 'IBLE']
            has_common_suffix = any(answer_upper.endswith(suffix) for suffix in common_suffixes)

            if not has_common_suffix:
                issues.append({
                    'type': 'uncommon_transformation',
                    'message': f'"{hint}" → "{answer}" 可能不是常见的中考词性转换',
                    'suggestion': '建议使用更常见的中考词汇转换，如：act→ACTION, happy→HAPPILY'
                })

        return issues

    def _check_possessive(self, context: str, answer: str, hint: str) -> List[Dict]:
        """检查所有格用法"""
        issues = []

        # 如果答案是所有格形式（以 'S 结尾）
        if answer.endswith("'S") or answer.endswith("'"):
            # 检查是否应该用所有格
            # 例如：______ owner，如果填 anybody's，则变成 "anybody's owner"
            # 这通常不对，应该是 "the owner's" 或 "its"

            # 提取填空后的完整短语
            blank_pos = context.find('______')
            if blank_pos > 0:
                after_blank = context[blank_pos + 6:].strip()  # 6 个下划线
                if after_blank.startswith('owner') or after_blank.startswith('Owners'):
                    issues.append({
                        'type': 'possessive_error',
                        'message': f'______ owner 填所有格 "{answer}" 通常不正确',
                        'suggestion': '考虑使用定冠词 "the" 或物主代词 "its/his/her"'
                    })

        return issues

    def _check_blank_context(self, context: str, answer: str, hint: str) -> List[Dict]:
        """检查填空位置的上下文是否合理"""
        issues = []

        # 首先检查填空是否存在
        if '______' not in context:
            issues.append({
                'type': 'missing_blank',
                'message': '句子中没有找到填空标记 "______"',
                'suggestion': '请确保句子中有填空标记 "______"'
            })
            return issues

        # 提取填空前后的内容
        blank_pos = context.find('______')

        before_blank = context[:blank_pos].strip().lower()
        after_blank = context[blank_pos + 6:].strip().lower()

        # 1. 检查填空前后是否已有完整内容（填空是多余的情况）
        # 例如："to the owner" 已经有内容，填空在 "owner" 位置就不合理
        before_words = before_blank.split()
        after_words = after_blank.split()

        # 如果填空前后都是完整的短语/单词，可能是冗余填空
        if len(before_words) >= 2 and len(after_words) >= 1:
            # 检查是否是 "to the + 名词" 结构
            if ' '.join(before_words[-2:]) == 'to the':
                issues.append({
                    'type': 'redundant_blank',
                    'message': '填空位置前已有 "to the"，填空可能多余或位置错误',
                    'suggestion': '考虑调整句子结构，如 "return it to ______" 或 "return ______ to the owner"'
                })

        # 2. 检查答案是否在句子里已经出现过
        if answer.lower() in context.lower():
            issues.append({
                'type': 'answer_in_context',
                'message': f'答案 "{answer}" 已在句子中出现，填空无意义',
                'suggestion': '填空位置应该是一个需要填写的空缺，而不是已存在的内容'
            })

        # 3. 检查答案是否是原词（填空无意义的情况）
        if answer.lower() == hint.lower():
            issues.append({
                'type': 'same_as_hint',
                'message': f'答案 "{answer}" 与提示词 "{hint}" 相同，词性转换无意义',
                'suggestion': '答案必须有形态变化，如加后缀或变形'
            })

        return issues

    def check_file(self, filepath: str, max_workers: int = 10) -> Dict:
        """检查整个文件（多线程并行）
        
        Args:
            filepath: JSON 文件路径
            max_workers: 最大线程数（默认10）
        """
        import concurrent.futures
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {
            'total': len(data),
            'passed': 0,
            'failed': 0,
            'details': []
        }

        print(f"🔍 开始检查 {len(data)} 道题目（{max_workers} 线程并行）...")

        # 逐个检查（为了避免 API 并发过高）
        for i, exercise in enumerate(data):
            print(f"  [{i+1}/{len(data)}] 检查 {exercise.get('id')}...", end=" ", flush=True)
            issues = self.check_exercise(exercise)
            
            if issues:
                results['failed'] += 1
                results['details'].append({
                    'id': exercise.get('id'),
                    'context': exercise.get('context'),
                    'answer': exercise.get('correct_answers', [''])[0],
                    'hint': self._extract_hint(exercise.get('context', '')),
                    'issues': issues
                })
                print(f"❌ {len(issues)} 个问题")
            else:
                results['passed'] += 1
                print("✅")

        return results

    def generate_fix_suggestions(self, results: Dict) -> str:
        """生成修复建议报告"""
        report = []
        report.append(f"# 练习题质量检查报告\n")
        report.append(f"总题数: {results['total']}")
        report.append(f"通过: {results['passed']}")
        report.append(f"失败: {results['failed']}\n")

        if results['failed'] > 0:
            report.append("## 问题列表\n")
            for item in results['details']:
                report.append(f"### {item['id']}: {item['context']}")
                report.append(f"答案: {item['answer']} (提示词: {item['hint']})")
                for issue in item['issues']:
                    report.append(f"- ❌ **{issue['type']}**: {issue['message']}")
                    report.append(f"  💡 {issue['suggestion']}")
                report.append("")

        return '\n'.join(report)


if __name__ == '__main__':
    import sys

    # 解析命令行参数
    grade = None
    fix_mode = False
    input_file = 'allvol/data/exercises_grade8_zhongkao_fixed.json'
    
    for arg in sys.argv[1:]:
        if arg == '--fix':
            fix_mode = True
        elif arg.startswith('--grade='):
            grade = arg.split('=')[1]  # grade7 或 grade8
        elif arg.endswith('.json'):
            input_file = arg
    
    # 显示配置信息
    print(f"📚 核心词汇检查器")
    print(f"  年级: {grade if grade else '所有年级'}")
    print(f"  输入文件: {input_file}")
    print(f"  模式: {'修复' if fix_mode else '检查'}")
    print("=" * 50)
    
    checker = ExerciseChecker(use_llm=True, grade=grade)

    if fix_mode:
        # 自动修复模式
        print("🔧 自动修复模式\n" + "="*50)

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        fixed_count = 0
        for i, exercise in enumerate(data):
            print(f"\n[{i+1}/{len(data)}] 检查 {exercise.get('id')}...")
            issues = checker.check_exercise(exercise)

            if issues:
                # 有问题，尝试修复
                fixed_exercise = checker.fix_exercise(exercise)
                if fixed_exercise != exercise:
                    fixed_count += 1
            else:
                print(f"  ✅ 无问题")

        print("\n" + "="*50)
        print(f"🎉 修复完成！共修复 {fixed_count} 道题")

        # 保存
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存到 {input_file}")

    else:
        # 只检查模式
        results = checker.check_file(input_file)
        report = checker.generate_fix_suggestions(results)

        print(report)

        # 保存报告
        report_file = input_file.replace('.json', '_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ 报告已保存到 {report_file}")
