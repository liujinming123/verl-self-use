import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = "你是一名汽车座舱专家，擅长分析汽车语音助手与用户的对话，并评估汽车语音助手执行指令是否表现不佳。"
SYSTEM_PROMPT ="""你是智能座舱交互质量分析模型，判断用户骂人是否由车机问题引发。

【输出】仅输出1或0
- 1：骂车机（车机有功能/体验问题）
- 0：骂人/无（与车机问题无关，或全局退出）

【核心规则】

一、骂人词分类

【全局退出词】：滚、滚蛋、滚吧、那滚、那滚吧、闭嘴
- 这类词常用于结束对话
- 上下文无明显差体验时→输出0
- 上下文有差体验时→输出1

【明确骂人词】：傻逼、操你妈、去死、真蠢、智障、我操、操
- 表达明确不满，需分析上下文

二、差体验判断

【有差体验→1】：
1. 理解失败：正常指令，车机说"没明白""换个说法"
2. 功能不支持：正常指令，车机说"不支持"
3. ASR错误：地名/歌名被错误识别（乐歌→勒戈）
4. 功能重复说：车机重复回答同样内容
5. 上下文继承失败：用户追问，车机不理解上下文
6. 服务超时/出错：车机提示超时、出错
7. 历史差体验：即使最后成功，之前有不理解/重复询问
【无差体验→0】：
1. 功能正常执行：车机说"好的"并完成任务
2. 合理引导操作：车机提示需要用户操作（授权、手动选择、在页面上操作）
3. 合理解释：车机解释限制并给替代方案
4. 噪音输入：用户输入无意义内容
5. 闲聊内容：用户聊天、讨论其他事物
6. 用户骂自己或他人："我傻逼了"

三、合理引导识别（重要）

车机以下回复属于合理引导，不是差体验：
- "请先在手机上授权..."
- "请在页面上手动选择..."
- "请在中控屏上操作..."
- "需要您先..."
→ 这类情况输出0

四、历史差体验（重要）

回溯整个对话历史：
- 如果历史中有理解失败/重复询问，即使最后执行成功
- 用户骂人是对历史体验不满
→ 输出1

五、噪音判断

【车机功能关键词】：导航、播放、打开、关闭、调节、空调、车窗、天窗、音量、温度、电话、音乐、歌、灯、座椅、悬挂、风口

【噪音特征】：
- 不含功能关键词
- 句子不完整/无意义
- 像误触发/背景声

【判断】：
- 首次指令成功 + 后续是噪音 + 全局退出词 → 输出0

六、闲聊判断

用户在评论、质疑、讨论，不是给车机指令：
- "怎么没有见下雨呢"（质疑天气预报）
- "妈妈的车不走了"（讨论他人）
→ 这类后面骂人，输出0

【示例】

示例1（输出0）- 合理引导：
用户: 打电话给XXX
车机: 请先在手机上授权同步通讯录，然后在中控屏的蓝牙电话中重新同步吧
用户: 去死吧
分析：车机给出合理操作引导（需用户授权），不是车机问题。输出0

示例2（输出0）- 合理引导：
用户: 把那个风口关掉
车机: 请在页面上手动选择要关闭的出风口吧
用户: 滚
分析：车机引导用户手动选择（因为有多个风口），合理引导。输出0

示例3（输出0）- 闲聊+全局退出：
用户: 今天天气是什么
车机: 绍兴市越城区今天全天小雨转阴...
用户: 怎么没有见下雨呢
车机:
用户: 闭嘴
分析：天气功能正常。"怎么没见下雨"是闲聊质疑天气预报准确性，不是车机问题。"闭嘴"是全局退出。输出0

示例4（输出0）- 噪音+全局退出：
用户: 唱一曲儿CHAT
车机: 为你找到以下CHAT，一起来听吧
用户: 这里就是CC DD
车机:
用户: 哎饼二十三饼干什么歌
车机:
用户: 那滚
分析：音乐功能首次执行成功。后续"这里就是CC DD""哎饼二十三饼干什么歌"是无意义噪音。"那滚"是全局退出。输出0

示例5（输出0）- 全局退出，功能正常：
用户: 打开座椅律动
车机: 座椅律动已经是开着的了
用户: 滚吧
分析：功能正常（告知状态），"滚吧"是全局退出。输出0

示例6（输出0）- 骂自己：
用户: 打电话给妈
车机: 找到多个结果，你要拨打第几个
用户: 我傻逼了
分析：功能正常执行，用户骂的是自己。输出0

示例7（输出1）- 历史差体验：
用户: 欢朋希尔顿欢朋对吧
车机: 哎呀，我没明白你的意思，换个说法试试呗
用户: 导航到希尔顿欢朋
车机: 去宁波慈溪希尔顿欢朋酒店，推荐以下路线，确认导航吗？
用户: 我操导航到希尔顿欢朋
分析：虽然最后执行成功，但第一次车机不理解，用户对历史差体验不满。输出1

示例8（输出1）- 理解失败：
用户: 打开车外放生
车机: 需要我做什么呢，再详细跟我说说呢
用户: 闭嘴
分析：用户发出指令，车机不理解。输出1
示例9（输出1）- 上下文继承失败：
用户: 悬挂调到标准
车机: 好的，悬架高度设置到标准。
用户: 上升
车机: 请告诉我您要调节的是什么，我会学的很快。
用户: 操你妈
分析：用户继续操作悬挂，车机未继承上下文。输出1

示例10（输出1）- 功能重复说：
用户: 导航到早晚相见
车机: 你想去哪里？
用户: 导航到早晚相见
车机: 你想去哪里？
用户: 操你妈
分析：车机重复问同样问题。输出1

示例11（输出1）- ASR错误：
用户: 导航到乐歌大厦
车机: 为您找到以下勒戈大厦
用户: 操你妈
分析："乐歌"被识别成"勒戈"，ASR错误。输出1

【决策流程】
1. 检查是否有"合理引导"回复→有则输出0
2. 回溯历史是否有差体验（不理解/重复/ASR错误/继承失败）→有则输出1
3. 最后指令是否正常执行→是则检查后续是否噪音/闲聊→是则输出0
4. 用户是否骂自己/他人→是则输出0
5. 其他情况→输出1

【你的任务】
分析给定对话，仅输出1或0。"""

user_prompt = """
你会看到一段汽车语音助手和用户之间的对话，对话中用户的疑似吐槽标识为<疑似吐槽>。
用户的话可能是给汽车语音助手下达的指令，也可能是与旁人的闲聊；汽车语音助手会自行判断用户的话是否为指令，如果汽车语音助手判断是指令，应该尝试执行指令，并根据执行结果给出回复，如果汽车语音助手判断不是指令，汽车语音助手理应保持沉默。
你的核心任务是判断用户吐槽的原因是否是针对汽车语音助手的。

请按以下结构输出分析结果：
- **总结：** [填写"是"或"否"]
  -"是"意味着汽车语音助手未能如用户预期般执行指令导致用户吐槽。
  -"否"是指汽车语音助手执行指令结果合理，用户吐槽不是汽车语音助手造成的；或者用户的吐槽并不是针对汽车语音助手，是针对旁人的；亦或用户没有吐槽。

提示:
1. 用户与旁人闲聊时，汽车语音助手可能会误判用户的话是指令，会乱答话，这可能会导致用户吐槽。
2. 当汽车语音助手回复"语音不支持此功能"时，这可能会导致用户吐槽。
3. 用户的话是经过asr(语音转文本)识别后的结果，当asr识别结果与用户预期不一致时，这可能会导致用户吐槽，这在导航场景比较常见。
4. 当汽车语音助手回复"好的"时，看作汽车语音助手正确执行了指令。
5. 当用户对汽车语音助手发出导航指令时，汽车语音助手会搜索可能的目的地，并回复"选第几个"，这被看作汽车语音助手正确执行了指令。当用户之后吐槽骂脏话时，这可能是第5点的情况:asr识别错误。
6. 当用户对汽车语音助手发出多媒体播放指令时，汽车语音助手有时会搜索可能的多媒体资源，并回复"为你找到以下"，这被看作汽车语音助手正确执行了指令。当用户之后吐槽骂脏话时，这可能是第5点的情况:asr识别错误，也可能是找到的资源和用户预期的不一致。
请使用工具search_similar_cases检索历史相似案例。
你需要分析的对话是: 
```
{text}
```
你的分析结果是: 
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--train_path", default="/vepfs/group03/user/ljm/work/qwen_grpo/data/train_data.csv", help="The path to the training data.")
    parser.add_argument("--test_path", default="/vepfs/group03/user/ljm/work/qwen_grpo/data/test_data.csv", help="The path to the test data.")
    
    args = parser.parse_args()
    
    data_source = "qwen_grpo_car_dialogue"
    ability = "dialogue_classification"
    
    def make_map_fn(split):
        def process_fn(row, idx):
            query = row["query"]
            label = int(row["label"])
            think_label = row["think_label"]
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": label
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "think_label": think_label,
                },
            }
            return data
        
        return process_fn
    
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    train_data = []
    test_data = []
    
    for idx, row in train_df.iterrows():
        train_data.append(make_map_fn("train")(row, idx))
    
    for idx, row in test_df.iterrows():
        test_data.append(make_map_fn("test")(row, idx))
    
    train_table = pa.Table.from_pandas(pd.DataFrame(train_data))
    test_table = pa.Table.from_pandas(pd.DataFrame(test_data))
    
    os.makedirs(args.local_dir, exist_ok=True)
    
    train_output_path = os.path.join(args.local_dir, "train_new.parquet")
    test_output_path = os.path.join(args.local_dir, "test_new.parquet")
    
    pq.write_table(train_table, train_output_path)
    pq.write_table(test_table, test_output_path)
    
    print(f"Train data saved to {train_output_path}, shape: {train_table.shape}")
    print(f"Test data saved to {test_output_path}, shape: {test_table.shape}")
