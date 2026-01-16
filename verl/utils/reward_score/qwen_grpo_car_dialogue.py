# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None):
    """
    奖励函数：比较模型输出与 ground_truth

    参数：
    - solution_str: 模型的完整输出文本，可能包含thinking内容
    - ground_truth: 真实标签（0 或 1）
    - data_source: 数据源名称（可选，为了兼容verl内置调用）
    - extra_info: 额外信息

    返回：
    - float: 1.0（匹配）或 -1.0（不匹配）
    """
    # 1. 去除thinking部分（Qwen3模型的thinking可能用特殊token包裹）
    # 常见的thinking格式：<|reserved_200006|>thinking content<|reserved_200007|> 或 <think>\n...\n</think>
    cleaned_str = solution_str

    # 去除 <|reserved_200006|>...<|reserved_200007|> 格式的thinking
    thinking_pattern = r"<\|reserved_200006\|>.*?<\|reserved_200007\|>"
    cleaned_str = re.sub(thinking_pattern, "", cleaned_str, flags=re.DOTALL)

    # 去除 <think>...</think> 格式的thinking
    cleaned_str = re.sub(r"<think>.*?</think>", "", cleaned_str, flags=re.DOTALL)

    # 去除多余的空白字符
    cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()

    # 2. 提取答案部分：查找 "**总结：** 是" 或 "**总结：** 否"
    answer_pattern = r"\*\*总结：\*\*\s*(是|否)"
    match = re.search(answer_pattern, cleaned_str)

    if not match:
        return -1.0

    predicted = match.group(1)
    predicted_label = 1 if predicted == "是" else 0
    ground_truth_int = int(ground_truth)

    return 1.0 if predicted_label == ground_truth_int else -1.0
