"""Step 4 -- turn per-frame probabilities into whole possessions with a hidden Markov model.

States H, A, D. Each frame contributes the classifier's log-probabilities (classifier.py), scaled
down because neighbouring frames are highly correlated. Changing state costs a fixed penalty, so
the decoder only switches when the evidence for the new state outweighs the cost -- which is
what makes possessions come out whole instead of flickering. Viterbi finds the best sequence for
each half independently.
"""

import numpy as np

from load import AWAY, HOME

# Params -- chosen on the external development games only (see tune.py), never on BAR-ATH.
EVIDENCE_SCALE = 1.0    # weight of one frame's log-probabilities
SWITCH_COST = 30.0      # log-penalty for H <-> A (= 2 x DEAD_COST: no cheaper detour via D)
DEAD_COST = 15.0        # log-penalty for entering or leaving D

DEAD = 2
STATES = np.array([HOME, AWAY, DEAD], dtype=np.int8)
DEFAULT_PARAMS = {"scale": EVIDENCE_SCALE, "switch": SWITCH_COST, "dead": DEAD_COST}


def transition_matrix(params):
    s, d = params["switch"], params["dead"]
    #          to H    to A    to D
    return -np.array([[0.0, s, d],      # from H
                      [s, 0.0, d],      # from A
                      [d, d, 0.0]])     # from D


def viterbi(log_emit, log_trans):
    """Best state index path. Loop over frames, vectorised over the 3 states."""
    n = len(log_emit)
    back = np.zeros((n, 3), dtype=np.int8)
    score = log_emit[0].copy()
    columns = np.arange(3)
    for t in range(1, n):
        candidates = score[:, None] + log_trans          # (from, to)
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], columns] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(log_probs, blocks, params, nobody=-1):
    """Per-frame codes: HOME / AWAY / DEAD inside the blocks, `nobody` outside them."""
    codes = np.full(len(log_probs), nobody, dtype=np.int8)
    log_trans = transition_matrix(params)
    for _, first, last in blocks:
        path = viterbi(log_probs[first:last + 1] * params["scale"], log_trans)
        codes[first:last + 1] = STATES[path]
    return codes
