"""Exact match reward function for taskllm dialogue training.

With <|action|>\n in the prompt (generation_prefix), the model only generates
the action content. Ground_truth format examples:
  - "<其他>"           -> only tag, compare tag for exact match
  - "<音乐.播放>(title=No Frontiers,style=dj,media=音乐)" -> tag + params
    - tag "音乐.播放"  -> exact string match
    - params "title=No Frontiers,style=dj,media=音乐" -> parse as dict, compare keys & values

Reward: +1.0 for both tag and params match, -1.0 for any mismatch.
"""

import re


def parse_action(text):
    """Parse tag and params from action content string.

    Examples:
        "<其他>"          -> ("其他", None)
        "<音乐.播放>(title=x,style=dj,media=音乐)" -> ("音乐.播放", {"title":"x","style":"dj","media":"音乐"})
        "<>"              -> ("", None)

    Returns:
        (tag, params_dict) or None if no <> brackets found.
    """
    tag_match = re.search(r"<([^>]*)>", text)
    if not tag_match:
        return None
    tag = tag_match.group(1)

    params_match = re.search(r"\(([^)]+)\)", text)
    if params_match:
        params = {}
        for pair in params_match.group(1).split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                params[k.strip()] = v.strip()
        return (tag, params)
    return (tag, None)


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """Exact match reward: +1.0 for match, -1.0 for mismatch.

    <> tag content: exact string comparison.
    () param content: parse as dict, compare keys and values.
    Both must match for +1.0.
    """
    cleaned = solution_str.strip()
    gt = ground_truth.strip() if isinstance(ground_truth, str) else str(ground_truth).strip()

    parsed_pred = parse_action(cleaned)
    parsed_gt = parse_action(gt)

    if parsed_pred is None or parsed_gt is None:
        return {"score": -1.0, "acc": 0}

    # Compare tag (<content>)
    if parsed_pred[0] != parsed_gt[0]:
        return {"score": -1.0, "acc": 0}

    # Compare params ((content)) as dict
    if parsed_gt[1] is not None:
        if parsed_pred[1] is None or parsed_pred[1] != parsed_gt[1]:
            return {"score": -1.0, "acc": 0}
    elif parsed_pred[1] is not None:
        # gt has no params but pred has params
        return {"score": -1.0, "acc": 0}

    return {"score": 1.0, "acc": 1}