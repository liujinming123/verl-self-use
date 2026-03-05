#!/usr/bin/env python3
import pandas as pd
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/vepfs/group03/model-public/Qwen3-30B-A3B')

def convert_rlhf_to_sft(input_path, output_path):
    df = pd.read_parquet(input_path)
    
    sft_data = []
    
    for _, row in df.iterrows():
        prompt_messages = row['prompt']
        ground_truth = row['reward_model']['ground_truth']
        
        output_text = f"- **总结：** {'是' if ground_truth == 1 else '否'}"
        
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        sft_data.append({
            "prompt": prompt_str,
            "response": output_text
        })
    
    sft_df = pd.DataFrame(sft_data)
    sft_df.to_parquet(output_path, index=False)
    print(f"Converted {len(sft_df)} samples to {output_path}")

if __name__ == "__main__":
    base_dir = "/vepfs/group03/user/ljm/work/qwen_grpo/data/verl_format"
    
    convert_rlhf_to_sft(
        os.path.join(base_dir, "train.parquet"),
        os.path.join(base_dir, "train_sft.parquet")
    )
    
    convert_rlhf_to_sft(
        os.path.join(base_dir, "test.parquet"),
        os.path.join(base_dir, "test_sft.parquet")
    )
