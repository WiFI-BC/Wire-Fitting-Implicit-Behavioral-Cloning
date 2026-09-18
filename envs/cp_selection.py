"""Candidate-cloud selection rules shared by the pixel envs."""
def select_from_cloud(q_values, cloud, selection="argmax", temperature=1.0,
                      score_norm="none"):
    """Turn a scored candidate cloud into one action index.

    WiFI-BC ranks a cloud of candidates; the ONLY published eval path so far was a
    hard argmax, which is maximally exposed to critic error — it deliberately
    seeks the candidate where Q is highest, which is disproportionately where Q
    is WRONG. `selection="sample"` draws from softmax(Q/temperature) instead
    (the IDQL selection rule), trading a little greediness for robustness to a
    miscalibrated critic.

    `score_norm` rescales WITHIN the cloud first, because raw Q magnitude drifts
    with the state: one temperature is near-greedy on some frames and
    near-uniform on others. Both transforms are strictly monotone, so neither
    can change an argmax — they exist for the sampling path.
      zscore  (q - mean) / std over the cloud; keeps relative gaps.
      rank    replace scores by rank mapped to [-1, 1]; scale-free and immune to
              a single wildly overestimated candidate, at the cost of
              discarding how much better the winner actually is.

    q_values: (1, N). cloud: (1, N, A). Returns an int index into N.
    """
    import torch as _t
    scores = q_values[0]
    if score_norm == "zscore":
        scores = (scores - scores.mean()) / (scores.std() + 1e-6)
    elif score_norm == "rank":
        order = _t.argsort(_t.argsort(scores)).to(scores.dtype)
        scores = 2.0 * order / max(1, scores.numel() - 1) - 1.0
    if selection == "sample":
        probs = _t.softmax(scores / max(float(temperature), 1e-6), dim=-1)
        return int(_t.multinomial(probs, 1).item())
    return int(scores.argmax().item())
