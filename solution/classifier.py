"""The per-frame possession model: P(H), P(A), P(D) from features.frame_table.

Gradient-boosted trees (scikit-learn HistGradientBoostingClassifier), trained ONLY on the
external development games with their ball hidden the way BAR-ATH loses it (devdata.degrade).
Training masks use different seeds from the evaluation masks, so a model is never scored on the
exact gaps it was trained on. Fixed random_state and no early stopping: training is deterministic.

    python solution/classifier.py           # leave-one-game-out check, then train on all three

The trained model is saved to solution/possession_model.pkl (committed), so run.py reproduces
the submission without the external data.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import devdata
import external
import features
import rules
from load import AWAY, HOME

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "possession_model.pkl")
TRAIN_MASK_SEEDS = (101, 102, 103)  # evaluation uses seed = game number (1, 2, 3)
CLASSES = (HOME, AWAY, rules.DEAD)


def table(ball, players, team, blocks, pitch):
    touch = rules.touches(features.nearest_distance_per_team(ball, players, team))
    return features.frame_table(ball, players, team, blocks, pitch, touch)


def training_rows(games, gaps):
    X, y = [], []
    for game in games:
        ball, players, team, blocks, gt = devdata.cached_game(game)
        graded = gt != rules.NOBODY
        for seed in TRAIN_MASK_SEEDS:
            masked = devdata.degrade(ball, gaps, seed=seed * 10 + game)
            X.append(table(masked, players, team, blocks, external.PITCH)[graded])
            y.append(gt[graded])
    return np.concatenate(X), np.concatenate(y)


def fit(X, y):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=False,
                                           random_state=0)
    return model.fit(X, y)


def log_proba(model, X):
    """(n, 3) log-probabilities in CLASSES order (H, A, D)."""
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return np.log(np.clip(p[:, order], 1e-9, 1.0))


def load_model():
    with open(MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


def held_out_log_probs(gaps):
    """For each dev game, log-probs from a model trained on the other two (evaluation masks)."""
    out = {}
    for held in external.GAMES:
        X, y = training_rows([g for g in external.GAMES if g != held], gaps)
        model = fit(X, y)
        ball, players, team, blocks, gt = devdata.cached_game(held)
        masked = devdata.degrade(ball, gaps, seed=held)
        out[held] = (log_proba(model, table(masked, players, team, blocks, external.PITCH)),
                     blocks, gt)
    return out


if __name__ == "__main__":
    import decode
    import possession_metrics as pm

    gaps = devdata.bar_ath_gap_lengths()
    held = held_out_log_probs(gaps)
    for game, (lp, blocks, gt) in held.items():
        frame_acc = np.mean(np.array(CLASSES)[np.argmax(lp, axis=1)][gt != rules.NOBODY]
                            == gt[gt != rules.NOBODY])
        codes = decode.decode(lp, blocks, decode.DEFAULT_PARAMS)
        r = pm.evaluate(devdata.to_pm(gt), devdata.to_pm(codes))
        print("held-out game {}: frame acc {:.3f}  spellF1 {:.4f}  macroF1 {:.4f}".format(
            game, frame_acc, r["spell_f1_mean"], r["macro_f1"]))

    X, y = training_rows(external.GAMES, gaps)
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(fit(X, y), handle)
    print("wrote", MODEL_PATH)
