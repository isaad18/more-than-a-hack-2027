"""Step 1 -- load the tactical ATD into numpy arrays, in metres, indexed by video frame.

    ball     (n_frames, 2)       x, y in metres, NaN where the ball is not tracked
    players  (n_frames, 46, 2)   x, y in metres for every track slot, NaN where not tracked
    team     (46,)               0 = home (Barcelona), 1 = away (Athletic)

Parsing 195 MB of text takes a little while, so the arrays are cached next to the data as an .npz.
The cache lives under data/, which is gitignored -- it is derived tracking and must never be
published (RULES.md).
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

N_TRACKS = 46
HOME, AWAY = 0, 1
CACHE = os.path.join(DATA_DIR, "cache", "tactical_atd.npz")


def team_of_each_track(meta_path):
    """0 for a home slot, 1 for an away slot. The first <Team> in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_frames, pitch):
    """Every line "<frame>:<x,y,s>;...;<x,y,s>:<x,y,s>" flattened into one numeric table.

    Swapping ':' and ';' for ',' turns the file into a plain CSV with 1 + 46*3 + 3 columns, which
    pandas' C parser reads far faster than a Python loop.
    """
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).to_numpy()

    frame = table[:, 0].astype(np.int64) - 1  # frame_id counts from 1, video frames from 0
    length, width = pitch
    scale = np.array([length, width])

    players = np.full((n_frames, N_TRACKS, 2), np.nan)
    players[frame] = table[:, 1:1 + N_TRACKS * 3].reshape(-1, N_TRACKS, 3)[:, :, :2] * scale
    ball = np.full((n_frames, 2), np.nan)
    ball[frame] = table[:, 1 + N_TRACKS * 3:1 + N_TRACKS * 3 + 2] * scale
    return ball, players


def load(game=GAME, use_cache=True):
    """(ball, players, team) for the tactical recording. See the module docstring for shapes."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = team_of_each_track(meta_path)

    if use_cache and os.path.isfile(CACHE):
        cached = np.load(CACHE)
        return cached["ball"], cached["players"], team

    ball, players = parse_raw(raw_path, source_frames("tactical", game), GAMES[game]["pitch"])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, ball=ball, players=players)
    return ball, players, team


if __name__ == "__main__":
    ball, players, team = load(use_cache=False)
    print("frames {}   tracks {} (home {}, away {})".format(
        len(ball), players.shape[1], int(np.sum(team == HOME)), int(np.sum(team == AWAY))))
    print("ball tracked on {:.1f}% of frames".format(100 * np.mean(~np.isnan(ball[:, 0]))))
    print("frame 12710: ball at ({:.1f}, {:.1f}) m".format(*ball[12710]))
