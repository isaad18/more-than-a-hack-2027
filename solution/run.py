"""Step 5 -- run the whole pipeline on BAR-ATH and write a submission.

    python solution/run.py                    # -> evaluation/results/NoTime.csv
    python solution/run.py --name v1          # -> evaluation/results/v1.csv
    python evaluation/validate_submission.py evaluation/results/NoTime.csv
"""

import argparse
import os

import numpy as np

import rules
from load import AWAY, HOME, ROOT, load
from games import GAME, GAMES, half_video_blocks
from pipeline import predict

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
TEAM_NAME = "NoTime"


def to_letters(codes):
    """Codes -> H/A/D, X for NOBODY (outside the halves, or before anyone has been seen)."""
    labels = np.full(len(codes), "X", dtype="<U1")
    labels[codes == HOME] = "H"
    labels[codes == AWAY] = "A"
    labels[codes == rules.DEAD] = "D"
    return labels


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def build_labels():
    ball, players, team = load()
    return to_letters(predict(ball, players, team, half_video_blocks(GAME), GAMES[GAME]["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=TEAM_NAME)
    args = parser.parse_args()

    labels = build_labels()
    path = os.path.join(RESULTS_DIR, "{}.csv".format(args.name))
    write_submission(path, labels)
    counts = {label: int(np.sum(labels == label)) for label in "HADX"}
    print("wrote {}   H {H}  A {A}  D {D}  X {X}".format(os.path.relpath(path, ROOT), **counts))


if __name__ == "__main__":
    main()
