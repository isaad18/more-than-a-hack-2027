"""Choose the decoder's constants on the external development games. BAR-ATH is never scored here.

    python solution/tune.py

Every dev game is scored with log-probabilities from a classifier trained on the OTHER two games
(classifier.held_out_log_probs), so the costs are chosen on predictions the model did not fit.
Only the ratio of the transition costs to the evidence matters, so the scale is fixed at 1 and
the two costs are searched on a log grid. The winner is printed; copy it into decode.py.
"""

import itertools
import os

import numpy as np

import classifier
import decode
import devdata
from devdata import pm

SWITCH = (10, 20, 40, 60, 80, 120, 160)
DEAD = (5, 10, 15, 20, 30, 40, 80)
CACHE = os.path.join(devdata.CACHE_DIR, "held_out_log_probs.npz")


def held_out():
    """{game: (log_probs, blocks, gt)}, cached because training takes a couple of minutes."""
    if os.path.isfile(CACHE):
        z = np.load(CACHE)
        return {g: (z["lp{}".format(g)], [tuple(b) for b in z["blocks{}".format(g)]],
                    z["gt{}".format(g)]) for g in (1, 2, 3)}
    out = classifier.held_out_log_probs(devdata.bar_ath_gap_lengths())
    arrays = {}
    for g, (lp, blocks, gt) in out.items():
        arrays.update({"lp{}".format(g): lp, "blocks{}".format(g): np.array(blocks),
                       "gt{}".format(g): gt})
    np.savez(CACHE, **arrays)
    return out


def score(held, params):
    spells, macros = [], []
    for lp, blocks, gt in held.values():
        r = pm.evaluate(devdata.to_pm(gt), devdata.to_pm(decode.decode(lp, blocks, params)))
        spells.append(r["spell_f1_mean"])
        macros.append(r["macro_f1"])
    return float(np.mean(spells)), float(np.mean(macros)), spells


def main():
    held = held_out()
    results = []
    for switch, dead in itertools.product(SWITCH, DEAD):
        params = {"scale": 1.0, "switch": float(switch), "dead": float(dead)}
        spell, macro, per_game = score(held, params)
        results.append((spell, macro, params, per_game))
        print("switch {:>4} dead {:>4}  spellF1 {:.4f}  macroF1 {:.4f}  {}".format(
            switch, dead, spell, macro, np.round(per_game, 3).tolist()), flush=True)
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    print("\nbest 5:")
    for spell, macro, params, per_game in results[:5]:
        print("  spellF1 {:.4f}  macroF1 {:.4f}  {}  {}".format(
            spell, macro, params, np.round(per_game, 3).tolist()))


if __name__ == "__main__":
    main()
