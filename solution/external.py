"""Development games: Metrica's public sample data, used as an honest test set.

The competition game's labels are off limits (RULES.md), so progress is measured on three other
matches that Metrica publishes openly with tracking *and* event data:

    https://github.com/metrica-sports/sample-data   (data/Sample_Game_1, _2, _3)

Download into data/external/ (gitignored) with solution/get_external.sh. Their tracking loses the
ball on 32-41% of frames, close to BAR-ATH's 41%, so they are a realistic stand-in.

Possession labels are derived from the human event data by a fixed rule written before any
scoring (event data about *other* games is explicitly allowed):

  - the ball is dead from a BALL OUT (when it crosses the line), a FAULT RECEIVED, or a goal,
    until the next SET PIECE;
  - while alive, the team of the latest PASS / CARRY / SHOT / RECOVERY / SET PIECE has it.
    BALL LOST and CHALLENGE never move possession on their own -- the other team's next action
    does, which mirrors "a touch is not a change of possession".

This is a proxy for the organisers' definition, not a copy of it, so absolute scores here are not
comparable with BAR-ATH. Relative comparisons between versions of our pipeline are what it is for.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import rules
from games import DATA_DIR
from load import AWAY, HOME

EXTERNAL_DIR = os.path.join(DATA_DIR, "external")
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)

TEAM_EVENTS = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


def _tracking(game):
    """(ball, players, team, blocks) from kloppy, in metres, array index = frame_id - 1."""
    from kloppy import metrica
    warnings.filterwarnings("ignore")
    path = lambda name: os.path.join(EXTERNAL_DIR, "Sample_Game_{}_{}".format(game, name))

    if game == 3:
        dataset = metrica.load_tracking_epts(meta_data=path("metadata.xml"),
                                             raw_data=path("tracking.txt"), coordinates="metrica")
    else:
        dataset = metrica.load_tracking_csv(home_data=path("RawTrackingData_Home_Team.csv"),
                                            away_data=path("RawTrackingData_Away_Team.csv"),
                                            coordinates="metrica")
    df = dataset.to_df()
    index = df.frame_id.to_numpy() - 1
    n = int(index.max()) + 1
    scale = np.array(PITCH)

    ball = np.full((n, 2), np.nan)
    ball[index] = df[["ball_x", "ball_y"]].to_numpy() * scale

    home_team = dataset.metadata.teams[0]
    ids, team = [], []
    for side, club in ((HOME, dataset.metadata.teams[0]), (AWAY, dataset.metadata.teams[1])):
        for player in club.players:
            if "{}_x".format(player.player_id) in df.columns:
                ids.append(player.player_id)
                team.append(side)
    players = np.full((n, len(ids), 2), np.nan)
    for slot, pid in enumerate(ids):
        players[index, slot] = df[["{}_x".format(pid), "{}_y".format(pid)]].to_numpy() * scale

    blocks = [(int(p), int(g.frame_id.min()) - 1, int(g.frame_id.max()) - 1)
              for p, g in df.groupby("period_id")]
    return ball, players, np.array(team, dtype=np.int8), blocks, home_team.name


def _events(game, home_name):
    """[(team_code_or_None, type, subtype, period, start_frame, end_frame)] in file order."""
    if game == 3:
        with open(os.path.join(EXTERNAL_DIR, "Sample_Game_3_events.json")) as handle:
            data = json.load(handle)["data"]
        out = []
        for e in data:
            sub = e.get("subtypes")
            if isinstance(sub, list):
                sub = "-".join(s["name"] for s in sub)
            elif isinstance(sub, dict):
                sub = sub["name"]
            team = e["team"]["name"] if e.get("team") else None
            out.append((team, e["type"]["name"], sub or "", e["period"],
                        e["start"]["frame"], e["end"]["frame"]))
    else:
        df = pd.read_csv(os.path.join(EXTERNAL_DIR, "Sample_Game_{}_RawEventsData.csv".format(game)))
        out = [(r.Team, r.Type, r.Subtype if isinstance(r.Subtype, str) else "", r.Period,
                int(r._5), int(r._7)) for r in df.itertuples()]
    return [(None if t is None else (HOME if t == home_name else AWAY), ty, sub.upper(), p, s, e)
            for t, ty, sub, p, s, e in out]


def event_labels(game, n, blocks, home_name):
    """Per-frame codes derived from events by the rule in the module docstring. NOBODY = no GT."""
    changes = {}  # frame -> state; later events at the same frame win
    for team, kind, sub, _, start, end in _events(game, home_name):
        if kind == "BALL OUT":
            changes[end - 1] = rules.DEAD
        elif kind == "FAULT RECEIVED":
            changes[start - 1] = rules.DEAD
        elif kind == "SHOT" and "GOAL" in sub and "GOAL KICK" not in sub:
            changes[start - 1] = team
            changes[end - 1] = rules.DEAD
        elif kind in TEAM_EVENTS and team is not None:
            changes[start - 1] = team

    codes = np.full(n, rules.NOBODY, dtype=np.int8)
    for _, first, last in blocks:
        state = rules.NOBODY
        for frame in range(first, last + 1):
            state = changes.get(frame, state)
            codes[frame] = state
    return codes


def load_game(game):
    """(ball, players, team, blocks, gt_codes) for one development game."""
    ball, players, team, blocks, home_name = _tracking(game)
    return ball, players, team, blocks, event_labels(game, len(ball), blocks, home_name)
