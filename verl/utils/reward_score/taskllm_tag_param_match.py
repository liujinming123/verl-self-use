"""Tag + params key + params value partial match reward for taskllm.

Scoring:
  - tag match:       +1
  - params keys match: +1 (exact key set equality)
  - params values match: +1 (all key-value pairs equal)

If ground truth has no params, only tag is scored (max score = 1).
If ground truth has params, max score = 3.
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
    """Partial match reward: tag +1, params keys +1, params values +1."""
    cleaned = solution_str.strip()
    gt = ground_truth.strip() if isinstance(ground_truth, str) else str(ground_truth).strip()

    parsed_pred = parse_action(cleaned)
    parsed_gt = parse_action(gt)

    if parsed_pred is None:
        return {"score": -1.0, "acc": 0}
    if parsed_gt is None:
        return {"score": 0.0, "acc": 0}

    score = 0.0
    tag_pred, params_pred = parsed_pred
    tag_gt, params_gt = parsed_gt

    # Tag match: +1
    if tag_pred == tag_gt:
        score += 1.0

    # Params scoring only when ground truth has params
    if params_gt is not None:
        if params_pred is not None:
            # Keys match: +1 (exact key set equality)
            if set(params_pred.keys()) == set(params_gt.keys()):
                score += 1.0
                # Values match: +1 (all key-value pairs equal)
                if params_pred == params_gt:
                    score += 1.0

    # Full match: tag + params all correct
    fully_matched = (tag_pred == tag_gt) and (
        (params_gt is None and params_pred is None)
        or (params_gt is not None and params_pred is not None and params_pred == params_gt)
    )
    return {"score": score, "acc": int(fully_matched)}