#!/usr/bin/env python3
"""
练习题质量校验器
检查生成的题目是否有语法错误、逻辑问题、词性转换错误
"""

import json
import re
from typing import List, Dict, Any

class ExerciseChecker:
    def __init__(self):
        self.issues = []

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

        return issues

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

    def check_file(self, filepath: str) -> Dict:
        """检查整个文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = {
            'total': len(data),
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for exercise in data:
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
            else:
                results['passed'] += 1

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
    checker = ExerciseChecker()
    results = checker.check_file('allvol/data/exercises_grade8_zhongkao_fixed.json')
    report = checker.generate_fix_suggestions(results)

    print(report)

    # 保存报告
    with open('allvol/data/quality_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ 报告已保存到 allvol/data/quality_report.md")
