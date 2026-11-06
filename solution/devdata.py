"""Development-game data shared by classifier.py, tune.py and dev_eval.py.

The development games are Metrica's public sample games (external.py). Their ball is hidden the
way BAR-ATH loses it (degrade), using only the lengths of BAR-ATH's tracked / untracked ball
stretches -- tracking statistics, never labels.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402

import external  # noqa: E402
import rules  # noqa: E402
from games import DATA_DIR, GAME, half_video_blocks  # noqa: E402
from load import AWAY, HOME  # noqa: E402

CACHE_DIR = os.path.join(DATA_DIR, "cache")


def to_pm(codes):
    """Our codes -> the scorer's 0..3 (H, A, D, X)."""
    out = np.full(len(codes), pm.X, dtype=np.int64)
    out[codes == HOME] = pm.H
    out[codes == AWAY] = pm.A
    out[codes == rules.DEAD] = pm.D
    return out


def cached_game(game):
    """(ball, players, team, blocks, gt) for one development game, cached as .npz under data/."""
    path = os.path.join(CACHE_DIR, "external_game{}.npz".format(game))
    if os.path.isfile(path):
        z = np.load(path)
        return z["ball"], z["players"], z["team"], [tuple(b) for b in z["blocks"]], z["gt"]
    ball, players, team, blocks, gt = external.load_game(game)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, ball=ball, players=players, team=team, blocks=np.array(blocks), gt=gt)
    return ball, players, team, blocks, gt


def bar_ath_gap_lengths():
    """Lengths of BAR-ATH's visible and missing ball runs inside the halves -- tracking only."""
    from load import load
    ball = load()[0]
    missing = np.isnan(ball[:, 0])
    visible_runs, missing_runs = [], []
    for _, first, last in half_video_blocks(GAME):
        m = missing[first:last + 1].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([1 - m[0]], m, [1 - m[-1]]))))
        for start, stop in zip(edges[:-1], edges[1:]):
            (missing_runs if m[start] else visible_runs).append(stop - start)
    return np.array(visible_runs), np.array(missing_runs)


def degrade(ball, gaps, seed):
    """Hide the dev ball the way BAR-ATH loses it: alternating visible and missing stretches with
    BAR-ATH's run lengths, on top of the dev data's own (mostly dead-ball) gaps. Seeded, so every
    run sees the same masks."""
    visible_runs, missing_runs = gaps
    rng = np.random.RandomState(seed)
    hide = np.zeros(len(ball), dtype=bool)
    frame, missing = 0, bool(rng.randint(2))
    while frame < len(ball):
        length = int(rng.choice(missing_runs if missing else visible_runs))
        hide[frame:frame + length] = missing
        frame += length
        missing = not missing
    out = ball.copy()
    out[hide] = np.nan
    return out
