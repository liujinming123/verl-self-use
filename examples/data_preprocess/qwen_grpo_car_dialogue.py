import argparse
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = "你是一名汽车座舱专家，擅长分析汽车语音助手与用户的对话，并评估汽车语音助手执行指令是否表现不佳。"

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
                        "content": user_prompt.format(text=query),
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
    
    train_output_path = os.path.join(args.local_dir, "train.parquet")
    test_output_path = os.path.join(args.local_dir, "test.parquet")
    
    pq.write_table(train_table, train_output_path)
    pq.write_table(test_table, test_output_path)
    
    print(f"Train data saved to {train_output_path}, shape: {train_table.shape}")
    print(f"Test data saved to {test_output_path}, shape: {test_table.shape}")
