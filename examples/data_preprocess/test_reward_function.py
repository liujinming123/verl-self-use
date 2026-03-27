#!/usr/bin/env python3
import sys

sys.path.insert(0, "/vepfs/group03/user/ljm/git_program/verl")

from verl.utils.reward_score.qwen_grpo_car_dialogue import compute_score

test_cases = [
    # 测试匹配的用例
    ("qwen_grpo_car_dialogue", "答案是 1", 1, 1.0),
    ("qwen_grpo_car_dialogue", "答案是 0", 0, 1.0),
    ("qwen_grpo_car_dialogue", "1", 1, 1.0),
    ("qwen_grpo_car_dialogue", "0", 0, 1.0),
    ("qwen_grpo_car_dialogue", "答案是 1 后面内容", 1, 1.0),
    ("qwen_grpo_car_dialogue", "答案是 0 后面内容", 0, 1.0),
    # 测试不匹配的用例
    ("qwen_grpo_car_dialogue", "1", 0, -1.0),
    ("qwen_grpo_car_dialogue", "0", 1, -1.0),
    # 测试没有匹配数字的情况
    ("qwen_grpo_car_dialogue", "其他格式开头", 1, -1.0),
    # 测试边界情况：多位数字中的1和0（不应该匹配）
    ("qwen_grpo_car_dialogue", "答案是 10", 1, -1.0),
    ("qwen_grpo_car_dialogue", "答案是 100", 0, -1.0),
]

print("Running reward function tests...")
all_passed = True
for i, (data_source, solution_str, ground_truth, expected) in enumerate(test_cases, 1):
    result = compute_score(solution_str, ground_truth)
    passed = abs(result - expected) < 1e-6
    status = "✓ PASS" if passed else "✗ FAIL"
    all_passed = all_passed and passed
    print(f"Test {i}: {status} | Expected: {expected}, Got: {result}")

if all_passed:
    print("\nAll tests passed! ✓")
    sys.exit(0)
else:
    print("\nSome tests failed! ✗")
    sys.exit(1)
