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

# 原始标注原因 → 标准原因信息映射表
# 格式：原始标签 → (原因编号, 标准原因文本, 对应结果)
REASON_MAPPING = {
    # 结果=1（骂车机）
    "有不文明用语，功能体验不支持": (1, "功能不支持：车机明确回复不支持用户请求的功能", 1),
    "有不文明用语，产品体验差": (2, "产品体验差：车机回复内容不符合用户预期，交互体验不佳；用户有指令时车机答非所问", 1),
    "有不文明用语，上下文中正常指令未响应": (3, "指令未响应：用户发出正常可理解的指令后，车机无任何回复", 1),
    "有不文明用语，功能重复说": (4, "功能重复响应：车机重复回复相同内容、反复询问相同问题，交互无进展", 1),
    "有不文明用语， 功能重复说": (4, "功能重复响应：车机重复回复相同内容、反复询问相同问题，交互无进展", 1),
    "有不文明用语，漏拒乱搭话": (5, "漏拒乱搭话：用户无指令时车机乱答话", 1),
    "有不文明用语，疑似asr识别错误": (6, "ASR识别错误：语音识别结果明显不符合用户意图（尤其导航/多媒体场景）", 1),
    "有不文明用语，上下文继承失败": (7, "上下文继承失败：车机无法理解用户基于对话上下文的连续指令", 1),
    
    # 结果=0（非车机问题）
    "有不文明用语，和车机体验无关": (8, "与车机无关：用户骂自己/骂旁人，不满与车机功能/体验完全无关", 0),
    "有不文明用语，上下文无明显差体验，属于全局退出": (9, "全局退出：车机功能正常执行，用户仅表达结束对话的全局退出意图", 0),
    "无不文明用语": (10, "无不文明用语：用户未表达任何不满或脏话", 0)
}

# 原因编号 → 标准原因信息映射表
REASON_ID_MAPPING = {v[0]: v for v in REASON_MAPPING.values()}

def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """
    奖励函数：同时校验模型输出的结果和原因是否正确

    参数：
    - solution_str: 模型的完整输出文本，可能包含thinking内容
    - ground_truth: 真实标签（0 或 1）
    - data_source: 数据源名称（可选，为了兼容verl内置调用）
    - extra_info: 额外信息，包含think_label字段（原始标注原因）

    返回：
    - float: 奖励值
      * +2.0: 结果正确 + 原因完全匹配
      * +0.5: 结果正确但原因不匹配
      * -1.0: 结果错误 / 提取不到结果 / 提取不到原因
    """
    # 1. 去除thinking部分（Qwen3模型的thinking可能用特殊token包裹）
    # 常见的thinking格式：<|reserved_200006|>thinking content<|reserved_200007|> 或 <think>\n...\n</think>
    cleaned_str = solution_str

    # 去除 <|reserved_200006|>...<|reserved_200007|> 格式的thinking
    thinking_pattern = r"<\|reserved_200006\|>.*?<\|reserved_200007\|>"
    cleaned_str = re.sub(thinking_pattern, "", cleaned_str, flags=re.DOTALL)

    # 去除 <think>...</think> 格式的thinking
    cleaned_str = re.sub(r"<think>.*?</think>", "", cleaned_str, flags=re.DOTALL)
    
    # 去除代码块标记（如果有的话）
    cleaned_str = re.sub(r"```|`", "", cleaned_str)

    # 去除多余的空白字符
    cleaned_str = re.sub(r"\s+", " ", cleaned_str).strip()

    # 2. 提取结果（0/1）：只匹配"结果:"后面的数字，避免和原因编号混淆
    answer_pattern = r"结果[：:]\s*(\d+)\b"
    answer_match = re.search(answer_pattern, cleaned_str, flags=re.IGNORECASE)
    if not answer_match:
        # 提取不到结果，返回-3.0
        return -3.0
    
    predicted_label = int(answer_match.group(1))
    # 结果只能是0或1，其他值视为无效
    if predicted_label not in (0, 1):
        return -3.0
    ground_truth_int = int(ground_truth)
    
    # 3. 提取原因编号
    reason_pattern = r"原因[：:]\s*(\d+)\s*\.?\s*(.*?)(?=\s*结果[：:]|\s*$)"
    reason_match = re.search(reason_pattern, cleaned_str, flags=re.IGNORECASE)
    
    # 先计算原因-结果匹配扣分项（无论结果对错，不匹配就扣1）
    reason_result_penalty = 0.0
    predicted_id = None
    predicted_reason_text = ""
    if reason_match and answer_match:
        predicted_id = int(reason_match.group(1))
        predicted_reason_text = reason_match.group(2).strip()
        
        # 检查原因和结果是否匹配：原因1-7对应结果1，原因8-10对应结果0
        reason_result_match = True
        if 1 <= predicted_id <=7 and predicted_label != 1:
            reason_result_match = False
        elif 8 <= predicted_id <=10 and predicted_label != 0:
            reason_result_match = False
        
        # 原因和结果不匹配，扣1.0
        if not reason_result_match:
            reason_result_penalty = -1.0
    
    # 4. 计算基础奖励：结果正确+2，结果错误-2
    if predicted_label == ground_truth_int:
        reward = 2.0
    else:
        reward = -2.0
    
    # 应用原因-结果不匹配扣分项
    reward += reason_result_penalty
    
    # 5. 判断原因是否正确（需要有think_label才校验，原因正确加0.5）
    reason_correct = False
    if extra_info is not None and "think_label" in extra_info and predicted_id is not None:
        # 标准化think_label：去除首尾空白 + 统一逗号左右无空格
        raw_think_label = extra_info["think_label"].strip()
        # 处理中文逗号前后的任意空白，统一格式
        raw_think_label = re.sub(r'\s*，\s*', '，', raw_think_label)
        standard_info = REASON_MAPPING.get(raw_think_label)
        if standard_info:
            standard_id, standard_reason, _ = standard_info
            
            # 优先判断编号是否匹配
            if predicted_id == standard_id:
                reason_correct = True
            # 编号没匹配上判断文本是否匹配
            else:
                def text_match(pred_text, std_text):
                    pred_text = pred_text.replace("：", ":").lower()
                    std_text = std_text.replace("：", ":").lower()
                    pred_key = pred_text.split(":")[0] if ":" in pred_text else pred_text
                    std_key = std_text.split(":")[0] if ":" in std_text else std_text
                    return pred_key in std_key or std_key in pred_key
                
                if text_match(predicted_reason_text, standard_reason):
                    reason_correct = True
    
    # 原因正确加0.5奖励
    if reason_correct:
        reward += 0.5
    
    # 最低奖励保底为-3.0（结果错-2 + 不匹配扣-1 = 最低-3）
    return max(reward, -3.0)
