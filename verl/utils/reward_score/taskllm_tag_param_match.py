"""Tag + params key + params value partial match reward for taskllm.

Scoring:
  - tag match:         +1
  - params keys match:   +1 (exact key set equality; both empty also counts)
  - params values match: +1 (all key-value pairs equal; both empty also counts)

When both prediction and ground truth have no params, the empty param dicts
are treated as key set equal and value equal, so each contributes +1.
Max score = 3 regardless of whether GT has params.
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
    """Partial match reward: tag +1, params keys +1, params values +1.

    None params are normalized to empty dicts, so when both prediction and
    ground truth have no params, the empty key set and empty value mapping
    each count as a match (+1 +1).
    """
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

    # Normalize None to empty dict so the "both no params" case is treated
    # as key set equal and value equal.
    params_pred_norm = params_pred if params_pred is not None else {}
    params_gt_norm = params_gt if params_gt is not None else {}

    # Tag match: +1
    if tag_pred == tag_gt:
        score += 1.0

    # Params keys match: +1 (exact key set equality; both empty also counts)
    if set(params_pred_norm.keys()) == set(params_gt_norm.keys()):
        score += 1.0
        # Params values match: +1 (all key-value pairs equal; both empty also counts)
        if params_pred_norm == params_gt_norm:
            score += 1.0

    # Full match: tag + params all correct
    fully_matched = (tag_pred == tag_gt) and (params_pred_norm == params_gt_norm)
    return {"score": score, "acc": int(fully_matched)}