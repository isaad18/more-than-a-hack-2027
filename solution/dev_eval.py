"""Score the current pipeline: held-out development games, plus sanity statistics on BAR-ATH.

    python solution/dev_eval.py

Three signals, in order of trust:
  1. dev games  -- Metrica sample games 1-3, labels from their event data, ball hidden the way
                   BAR-ATH loses it. Each game is scored with a classifier trained on the OTHER
                   two, so the number is out-of-sample. Decisions are made on the mean spellF1.
  2. sanity     -- shape of our BAR-ATH output: possession count, mean length, D share, X left.
  3. public     -- the BAR-ATH public five minutes. A bug alarm only: a collapse means something
                   broke; a small move means nothing. Never tune on it.
"""

import os

import numpy as np

import decode
import devdata
import tune
from devdata import pm
from games import GAME, half_video_blocks
from load import ROOT
from run import build_labels

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def main():
    held = tune.held_out()
    print("{:<10} {:>8} {:>8} {:>7} {:>7}".format("game", "spellF1", "macroF1", "poss", "gtposs"))
    spells, macros = [], []
    for game, (lp, blocks, gt) in held.items():
        gt_pm = devdata.to_pm(gt)
        pred_pm = devdata.to_pm(decode.decode(lp, blocks, decode.DEFAULT_PARAMS))
        r = pm.evaluate(gt_pm, pred_pm)
        runs = pm.valid_runs(gt_pm)
        spells.append(r["spell_f1_mean"])
        macros.append(r["macro_f1"])
        print("{:<10} {:>8.4f} {:>8.4f} {:>7} {:>7}".format(
            "dev " + str(game), spells[-1], macros[-1],
            len(pm.possession_spells(pm.fill_abstentions(pred_pm, runs), runs)),
            len(pm.possession_spells(gt_pm, runs))))
    print("{:<10} {:>8.4f} {:>8.4f}".format("DEV MEAN", np.mean(spells), np.mean(macros)))

    labels = build_labels()
    codes = pm.label_codes(labels)
    blocks = half_video_blocks(GAME)
    in_halves = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        in_halves[first:last + 1] = True
    runs = [(first, last + 1) for _, first, last in blocks]
    spells_bar = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
    graded = labels[in_halves]
    print("\nsanity BAR-ATH: {} possessions, mean {:.1f} s, D {:.1f}%, home share {:.1f}%, "
          "X inside halves {}".format(
              len(spells_bar), np.mean([(b - a) / 25 for a, b, _ in spells_bar]),
              100 * np.mean(graded == "D"),
              100 * np.sum(graded == "H") / max(1, np.sum(np.isin(graded, ["H", "A"]))),
              int(np.sum(graded == "X"))))

    public = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    r = pm.evaluate(public, codes)
    print("public 5 min (bug alarm only): spellF1 {:.4f}  macroF1 {:.4f}".format(
        r["spell_f1_mean"], r["macro_f1"]))


if __name__ == "__main__":
    main()
