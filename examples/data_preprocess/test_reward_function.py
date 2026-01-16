#!/usr/bin/env python3
import sys
sys.path.insert(0, '/vepfs/group03/user/ljm/git_program/verl')

from verl.utils.reward_score.qwen_grpo_car_dialogue import compute_score

test_cases = [
    ("qwen_grpo_car_dialogue", "**总结：** 是", 1, 1.0),
    ("qwen_grpo_car_dialogue", "**总结：** 否", 0, 1.0),
    ("qwen_grpo_car_dialogue", "**总结：** 是 后面内容", 1, 1.0),
    ("qwen_grpo_car_dialogue", "**总结：** 否 后面内容", 0, 1.0),
    ("qwen_grpo_car_dialogue", "**总结：** 是", 0, -1.0),
    ("qwen_grpo_car_dialogue", "**总结：** 否", 1, -1.0),
    ("qwen_grpo_car_dialogue", "其他格式开头", 1, -1.0),
]

print("Running reward function tests...")
all_passed = True
for i, (data_source, solution_str, ground_truth, expected) in enumerate(test_cases, 1):
    result = compute_score(data_source, solution_str, ground_truth)
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
