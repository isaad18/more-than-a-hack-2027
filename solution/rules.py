"""Touch detection: which team, if either, is clearly on the ball in a frame.

Touches are not possession -- a touch can be a deflection or a lost duel. They are one input to
the per-frame model (features.frame_table, as "frames since / until the last / next touch"),
and the model plus the decoder decide what counts as control.

Both constants are football judgements, fixed before any scoring.
"""

import numpy as np

from load import AWAY, HOME

# Params
CONTROL_RADIUS = 1.5     # m. A player this close to the ball can play it.
CONTESTED_MARGIN = 0.5   # m. Both teams within this of each other -> nobody clearly has it.

NOBODY = -1
DEAD = 2


def touches(nearest):
    """Per frame: HOME or AWAY if exactly one team is clearly on the ball, else NOBODY."""
    home, away = nearest[:, HOME], nearest[:, AWAY]
    with np.errstate(invalid="ignore"):
        home_on = (home <= CONTROL_RADIUS) & ~(away <= home + CONTESTED_MARGIN)
        away_on = (away <= CONTROL_RADIUS) & ~(home <= away + CONTESTED_MARGIN)
    out = np.full(len(nearest), NOBODY, dtype=np.int8)
    out[home_on] = HOME
    out[away_on] = AWAY
    return out
