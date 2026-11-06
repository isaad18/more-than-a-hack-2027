"""Steps 2-4 for any match: positions in, per-frame possession codes out.

Kept separate from the BAR-ATH loader so the exact same function runs on the external
development games (external.py) and on the competition game (run.py).
"""

import classifier
import decode
import rules


def predict(ball, players, team, blocks, pitch, params=None, model=None):
    """Per-frame codes (load.HOME, load.AWAY, rules.DEAD, or rules.NOBODY) for one match.

    blocks is [(half, first_frame, last_frame)] in array indices; frames outside every block
    stay NOBODY, and each block is decoded on its own so possession never carries over halftime.
    """
    model = model or classifier.load_model()
    log_probs = classifier.log_proba(model, classifier.table(ball, players, team, blocks, pitch))
    return decode.decode(log_probs, blocks, params or decode.DEFAULT_PARAMS, nobody=rules.NOBODY)
