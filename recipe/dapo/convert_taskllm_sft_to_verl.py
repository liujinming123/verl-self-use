#!/usr/bin/env python3
"""Convert taskllm_sft dialogue data to verl RL training format.

Input format (SFT parquet):
    - system: str
    - conversations: list[dict] with 'from' (human/assistant) and 'value'

Output format (verl RL parquet):
    - data_source: str
    - prompt: list[dict] (OpenAI chat format messages, up to last human turn)
    - ability: str
    - reward_model: dict {"style": "rule", "ground_truth": str}
    - extra_info: dict

Each conversation produces ONE training sample:
  - prompt = system + all turns except the last assistant response
  - ground_truth = text AFTER <|action|>\n in the last assistant response
  (The model generates the full assistant response including <|action|>\n,
   but the reward function only evaluates the part after <|action|>\n.)
"""

import pandas as pd
import argparse
import os


def extract_ground_truth(last_assistant_response):
    """Extract the action content after <|action|>\n from the last assistant response.

    Format: <|action|>\n<actual_content>
    Returns: <actual_content> (the part after <|action|>\n)

    If <|action|> is not found, returns the full response as ground_truth.
    """
    action_marker = "<|action|>"
    if action_marker in last_assistant_response:
        idx = last_assistant_response.index(action_marker)
        remainder = last_assistant_response[idx + len(action_marker):]
        # Skip the newline after <|action|>
        if remainder.startswith("\n"):
            remainder = remainder[1:]
        return remainder.strip()
    # No <|action|> marker: use full response as ground_truth
    return last_assistant_response.strip()


def convert_conversation(row_idx, system, conversations, split):
    """Convert one conversation to a single verl-format training sample.

    The prompt includes all turns EXCEPT the last assistant response.
    The ground_truth is extracted from the last assistant response
    (text AFTER <|action|>\n).
    """
    # Build full message list
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})

    # Find the last assistant turn
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return None

    # Prompt = all turns before the last assistant response
    prompt = messages[:last_assistant_idx]

    # Ground truth = content after <|action|>\n in the last assistant response
    last_response = messages[last_assistant_idx]["content"]
    ground_truth = extract_ground_truth(last_response)

    # Generation prefix = <|action|>\n (part of model input, not output)
    # The model sees <|action|>\n as part of the prompt and only generates ground_truth
    generation_prefix = None
    action_marker = "<|action|>"
    if action_marker in last_response:
        idx = last_response.index(action_marker)
        prefix_str = last_response[:idx + len(action_marker)]
        remainder_after_marker = last_response[idx + len(action_marker):]
        if remainder_after_marker.startswith("\n"):
            prefix_str += "\n"
        generation_prefix = prefix_str

    return {
        "data_source": "taskllm_dialogue",
        "prompt": prompt,
        "ability": "dialogue_response",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"index": row_idx, "split": split, "generation_prefix": generation_prefix},
    }


def main():
    parser = argparse.ArgumentParser(description="Convert taskllm_sft data to verl RL format")
    parser.add_argument("--input_file", type=str, required=True, help="Input parquet file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--split", type=str, default="train", help="Data split label (train/val/test)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input_file)

    all_samples = []
    skipped = 0
    for idx, row in df.iterrows():
        sample = convert_conversation(idx, row["system"], row["conversations"], args.split)
        if sample is not None:
            all_samples.append(sample)
        else:
            skipped += 1

    output_df = pd.DataFrame(all_samples)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_df.to_parquet(args.output_file)

    print(f"Converted {len(df)} conversations -> {len(output_df)} training samples (skipped {skipped})")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()