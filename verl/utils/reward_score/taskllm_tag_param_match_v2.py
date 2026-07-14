"""Tag + params key (F1) + params value (Levenshtein) partial match reward for taskllm.

Scoring:
  - tag match:         +1 (exact string equality)
  - params keys:       F1 score in [0, 1] (set-based, order-insensitive)
  - params values:     mean normalized Levenshtein similarity in [0, 1],
                        averaged over gt keys (missing key contributes 0)

Max score = 3 regardless of whether GT has params.

F1 on key sets:
  tp = |pred_keys ∩ gt_keys|
  precision = tp / |pred_keys|   (0 if pred empty)
  recall    = tp / |gt_keys|     (0 if gt empty)
  f1 = 2*p*r/(p+r)               (0 if p+r == 0)
  Both empty -> F1 = 1.0 (treated as full match).

Value similarity (per shared key k):
  sim = 1 - levenshtein(pred[k], gt[k]) / max(len(pred[k]), len(gt[k]))
  Both empty strings -> 1.0.

Value score:
  mean over k in gt_keys of (sim if k in pred else 0)
  gt has no keys -> 1.0 if pred also empty, else 0.0 (hallucinated params).
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


def _f1(pred_keys, gt_keys):
    """Set-based F1 over key collections.

    Both empty -> 1.0. One empty, other non-empty -> 0.0.
    """
    pred_set = set(pred_keys)
    gt_set = set(gt_keys)
    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0
    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a, b):
    """Standard Levenshtein distance (insert/delete/replace each cost 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def _lev_sim(a, b):
    """Normalized Levenshtein similarity in [0, 1]."""
    if a == b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / m


def _value_score(pred_params, gt_params):
    """Mean per-value Levenshtein similarity over gt keys.

    Missing key in pred contributes 0. If gt has no keys, returns 1.0 when
    pred is also empty (full match), 0.0 otherwise (hallucinated params).
    """
    if not gt_params:
        return 1.0 if not pred_params else 0.0
    total = 0.0
    for k, gv in gt_params.items():
        if k in pred_params:
            total += _lev_sim(pred_params[k], gv)
    return total / len(gt_params)


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """Partial match reward: tag +1, params keys F1 in [0,1], params values mean Lev-sim in [0,1]."""
    cleaned = solution_str.strip()
    gt = ground_truth.strip() if isinstance(ground_truth, str) else str(ground_truth).strip()

    parsed_pred = parse_action(cleaned)
    parsed_gt = parse_action(gt)

    if parsed_pred is None:
        return {"score": -1.0, "acc": 0}
    if parsed_gt is None:
        return {"score": 0.0, "acc": 0}

    tag_pred, params_pred = parsed_pred
    tag_gt, params_gt = parsed_gt

    params_pred_norm = params_pred if params_pred is not None else {}
    params_gt_norm = params_gt if params_gt is not None else {}

    score = 0.0

    # Tag match: +1 (exact)
    if tag_pred == tag_gt:
        score += 1.0

    # Params keys: F1 in [0, 1]
    score += _f1(params_pred_norm.keys(), params_gt_norm.keys())

    # Params values: mean normalized Levenshtein similarity in [0, 1]
    score += _value_score(params_pred_norm, params_gt_norm)

    # acc: strict full match (tag + exact full dict equality)
    fully_matched = (tag_pred == tag_gt) and (params_pred_norm == params_gt_norm)
    return {"score": score, "acc": int(fully_matched)}
