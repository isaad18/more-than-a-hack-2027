"""Step 2 -- per-frame features computed from the raw positions.

Everything is vectorised over all 199319 frames at once; nothing here loops in Python.
"""

import warnings

import numpy as np

from load import AWAY, HOME

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

FPS = 25


def nearest_distance_per_team(ball, players, team):
    """(n_frames, 2): distance in metres from the ball to the closest home / away player.

    NaN where the ball, or every player of that team, is missing.
    """
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)  # (n_frames, 46)
    out = np.full((len(ball), 2), np.nan)
    for side in (HOME, AWAY):
        side_distance = distance[:, team == side]
        seen = ~np.all(np.isnan(side_distance), axis=1)
        out[seen, side] = np.nanmin(side_distance[seen], axis=1)
    return out


def ball_speed(ball):
    """m/s between consecutive frames. The ATD's own speed column is always NaN for the ball."""
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    return np.concatenate(([np.nan], step))


MAX_RUN_SPEED = 12.0  # m/s. Faster than any sprint: a tracker jump, not a player moving.


def centred_mean(values, half_window):
    """NaN-aware mean over a centred window of +/- half_window frames, via cumulative sums."""
    ok = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(values), index + half_window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def player_speeds(players):
    """(n_frames, n_tracks) m/s from frame-to-frame displacement, NaN for tracker jumps."""
    step = np.linalg.norm(np.diff(players, axis=0), axis=2) * FPS
    step = np.concatenate((np.full((1, players.shape[1]), np.nan), step))
    step[step > MAX_RUN_SPEED] = np.nan
    return step


def attack_sign(players, team, blocks):
    """(n_frames,) +1 where HOME attacks towards larger x, -1 where it attacks towards smaller x.

    Decided per half from where each team stands on average: a team defends the goal on its own
    side of its mean position. Works for any match without metadata about directions.
    """
    sign = np.zeros(len(players))
    for _, first, last in blocks:
        half = players[first:last + 1]
        home_x = np.nanmean(half[:, team == HOME, 0])
        away_x = np.nanmean(half[:, team == AWAY, 0])
        sign[first:last + 1] = 1.0 if home_x < away_x else -1.0
    return sign


def motion_features(players):
    """Whole-pitch movement, the stoppage signal: everyone walks while a restart is set up."""
    speed = player_speeds(players)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(speed, axis=1)
        fast = np.nanmean(speed > 4.0, axis=1)  # share of players running at > 4 m/s
    return np.column_stack([
        centred_mean(mean, 12),    # +/- 0.5 s
        centred_mean(mean, 50),    # +/- 2 s
        centred_mean(mean, 125),   # +/- 5 s
        centred_mean(fast, 50),
    ])


def shape_features(players, team, sign, pitch):
    """Team shape, oriented so positive x is towards the goal HOME attacks. The side in possession
    pushes up and stretches the pitch; the side out of it drops and compacts."""
    length, width = pitch
    x = (players[:, :, 0] - length / 2) * sign[:, None]
    y = players[:, :, 1]
    columns = []
    with np.errstate(invalid="ignore"):
        for side in (HOME, AWAY):
            columns += [np.nanmean(x[:, team == side], axis=1),
                        np.nanstd(x[:, team == side], axis=1),
                        np.nanstd(y[:, team == side], axis=1)]
        speed = player_speeds(players)
        for side in (HOME, AWAY):
            # Signed forward velocity of the team: who is pushing towards which goal.
            vx = np.diff(x[:, team == side], axis=0) * FPS
            vx[np.abs(vx) > MAX_RUN_SPEED] = np.nan
            columns.append(np.concatenate(([np.nan], np.nanmean(vx, axis=1))))
    out = np.column_stack(columns)
    return np.column_stack([centred_mean(out[:, k], 25) for k in range(out.shape[1])])


HISTORY_CAP = 1500  # frames (60 s). "Long ago" -- further back carries no extra information.


def frames_since(event, blocks):
    """Frames since `event` was last True, within each half; HISTORY_CAP where it never was."""
    out = np.full(len(event), HISTORY_CAP, dtype=np.float64)
    for _, first, last in blocks:
        e = event[first:last + 1]
        index = np.arange(len(e))
        last_seen = np.maximum.accumulate(np.where(e, index, -1))
        out[first:last + 1] = np.where(last_seen >= 0,
                                       np.minimum(index - last_seen, HISTORY_CAP), HISTORY_CAP)
    return out


def frames_until(event, blocks):
    """Frames until `event` is next True, within each half. The pipeline sees the whole match, so
    it may look ahead -- possession is decided by who controls the ball *next* as much as last."""
    reversed_blocks = [(h, len(event) - 1 - last, len(event) - 1 - first) for h, first, last in blocks]
    return frames_since(event[::-1], reversed_blocks)[::-1]


def distance_to_line(ball, pitch):
    """Metres from the ball to the nearest touchline or goal line (negative outside the pitch)."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))


def carried(values, blocks, backwards=False):
    """Last (or next) non-NaN value of each column, carried across the gaps, within each half."""
    out = np.full(values.shape, np.nan)
    for _, first, last in blocks:
        v = values[first:last + 1]
        if backwards:
            v = v[::-1]
        ok = ~np.isnan(v[:, 0])
        index = np.maximum.accumulate(np.where(ok, np.arange(len(v)), -1))
        filled = np.where((index >= 0)[:, None], v[np.maximum(index, 0)], np.nan)
        out[first:last + 1] = filled[::-1] if backwards else filled
    return out


def frame_table(ball, players, team, blocks, pitch, touch):
    """Every per-frame feature the possession model sees, one row per frame.

    Built only from positions, so it is computed identically for the development games and for
    BAR-ATH. NaN means "not available here" and the model handles it natively.
    """
    sign = attack_sign(players, team, blocks)
    length, width = pitch
    visible = ~np.isnan(ball[:, 0])
    nearest = nearest_distance_per_team(ball, players, team)
    speed = centred_mean(ball_speed(ball), 2)
    line = distance_to_line(ball, pitch)
    ball_forward = (ball[:, 0] - length / 2) * sign  # +: towards the goal HOME attacks

    # Where the ball was when last seen / where it is when next seen, carried over the gap.
    at_edges = np.column_stack([line, ball_forward, speed])
    before, after = carried(at_edges, blocks), carried(at_edges, blocks, backwards=True)

    columns = [
        motion_features(players),
        shape_features(players, team, sign, pitch),
        visible[:, None].astype(float),
        nearest,
        nearest[:, HOME:HOME + 1] - nearest[:, AWAY:AWAY + 1],
        speed[:, None], line[:, None], ball_forward[:, None],
        frames_since(touch == HOME, blocks)[:, None],
        frames_since(touch == AWAY, blocks)[:, None],
        frames_until(touch == HOME, blocks)[:, None],
        frames_until(touch == AWAY, blocks)[:, None],
        frames_since(visible, blocks)[:, None],
        frames_until(visible, blocks)[:, None],
        before, after,
    ]
    return np.column_stack(columns)
