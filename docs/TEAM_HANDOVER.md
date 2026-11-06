# NoTime — team handover

Read this first if you are joining the team. It covers what the challenge is, what has been built,
why it was built that way, how to run it, and what is still open.

- **Deadline: 17 November 2026.** Submission is one Google Form per team; the link arrives by email.
- **Team size:** 3–5 members are needed to be eligible for the Innovation Challenge and the Final Pitch.
- **Repo:** https://github.com/isaad18/more-than-a-hack-2027 (a fork of `metrica-sports/more-than-a-hack-2027`).
- The organisers' official documents are `CHALLENGE.md`, `DATA.md`, `RULES.md`, `SUBMISSION.md`
  and `EVALUATION.md`. Our technical document for the jury is [`NoTime.md`](NoTime.md).

---

## 1. The challenge in five lines

1. We get one match: FC Barcelona (home, `H`) vs Athletic Club (away, `A`), LaLiga 2024-25.
2. For every frame of the tactical video (199,319 frames) we output who has the ball: `H`, `A`,
   `D` (dead ball) or `X` (abstain).
3. Only the two halves are scored: frames 11185–90651 and 110808–190197. Everything else is `X`.
4. We are ranked on **spellF1**, which counts how many real possessions we recover *as whole
   possessions*. macroF1 (per-frame) breaks ties. The official baseline scores 0.2433.
5. The real answer key is private. There is no live leaderboard: we find out after the deadline.

## 2. Rules we must never break (disqualification)

- **Labels must come from an algorithm.** Nobody may hand-label or hand-correct anything on this
  match, for any purpose, including "just to check" or "just for training".
- **No human-produced data about this match** (Opta, StatsBomb, event feeds, videos annotated by people).
- **The public 5-minute answer key** (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) is a
  pipeline check only. **Never choose a threshold or setting because it scores better there.**
  It is also misleading: the baseline that wins on it loses on the full match.
- **Constants must have a football reason**, never "it works on frame 54,321".
- **The repo must not contain match data or anything derived from it per frame** (no CSVs of our
  labels, no caches). `data/` and `evaluation/results/*.csv` are already gitignored; keep it that way.
- **The repo must regenerate our submitted CSV byte for byte.** If you change any code, the CSV we
  upload must be the one produced by the committed code.

## 3. What we built, step by step (the history)

### Step 0 — understanding the data
- The main input is the tracking file `BAR-ATH_tactical_atd.txt`: one line per frame, with the x/y
  of 46 player "slots" and the ball. Slots 1–24 are Barcelona and 25–46 Athletic. A slot is a
  tracklet, not a person: the same player can appear in several slots.
- Frame numbering: `video_frame = frame_id − 1`. Coordinates are 0–1; multiply by 105.3 × 68.0
  for metres. The ball-speed column is always empty, so we compute speed ourselves.
- We checked our data setup by regenerating the organisers' four baselines: all four came out
  byte-identical to the committed ones.

### Step 1 — v0, plain rules (no learning)
- A team "touches" the ball when its nearest player is within 1.5 m and the other team is not
  just as close. Possession switched only after about half a second of control.
- "Ball missing for 2 s → dead ball".
- Result: beat the baseline, but labelled **43% of the match as dead ball** (reality is around 20%).

### The key discovery
The ball is missing on 41% of the scored frames, and **it goes missing in long blocks during
live play**: 149 missing stretches, median 11 s, the longest 3.7 minutes. So "no ball" does not
mean "dead ball". Two other signals work without a ball:

- **Player movement:** during a stoppage everyone walks (about 1 m/s on average) versus about
  2 m/s in play. This holds in every match we looked at.
- **Team shape:** the team with the ball pushes up and spreads out. On its own, with no ball at
  all, this picks the right team about 79% of the time.

### Step 2 — a fair way to measure progress
We cannot score ourselves on this match, so we built a test set from **other matches**:
Metrica's three public sample games (https://github.com/metrica-sports/sample-data), which come
with tracking *and* event data. Event data on *other* games is allowed.

- Per-frame labels are derived from their events by a fixed rule (`solution/external.py`):
  dead from ball out / foul / goal until the next set piece; otherwise the team of the latest
  pass, carry, shot, recovery or set piece.
- **Trap we found:** those games lose the ball mostly when it is dead, so there "missing ball" is
  a near-perfect dead-ball signal, which would teach a model the wrong thing. We therefore
  **hide their ball the way BAR-ATH loses it** (alternating visible/missing stretches drawn
  from BAR-ATH's own stretch lengths; tracking statistics only, no labels). This is `degrade()`
  in `solution/devdata.py`.
- Every score below is **leave-one-game-out**: train on two sample games, score the third.

### Step 3 — v1, two small logistic regressions + HMM
Same idea as the DSLR project: one logistic regression for "dead vs alive" from movement, one
for "which team" from shape, plus touches. Their outputs are decoded by an HMM (see below).

### Step 4 — v2, the current submission
- **Per-frame model:** a gradient-boosted tree classifier (scikit-learn `HistGradientBoosting`,
  deterministic) outputs P(H), P(A), P(D) for every frame from about 40 features: movement,
  team shape, ball–player distances, frames since/until each team's last/next touch, and where
  the ball was before and after it vanished.
- **Sequence decoding:** a 3-state HMM decoded with Viterbi. Staying in the same state is free,
  changing team costs 30, entering or leaving dead ball costs 15. That is what turns flickering
  per-frame guesses into whole possessions, and it is how "a loose or contested ball stays with
  the last team" is implemented. The costs were picked by grid search on the held-out sample
  games, never on BAR-ATH.
- The halves are decoded separately. `X` appears only outside the halves.

### Scores (held-out sample games, official metric code)

| Version | spellF1 | macroF1 |
|---|---|---|
| v0 rules | 0.445 | 0.645 |
| v1 logistic regressions + HMM | 0.465 | 0.741 |
| **v2 boosted trees + HMM (submitted)** | **0.587** | **0.800** |

These labels are our own proxy, so the absolute numbers are not what we will get on BAR-ATH. They
are for comparing versions.

**Sanity checks on BAR-ATH** (no answer key needed): 339 possessions, mean 13.0 s, Barcelona
57.7% of live time, dead ball 30.8% of scored frames, no `X` inside the halves. The public
5 minutes give spellF1 0.513 against 0.326 for the best-spread baseline: the pipeline is not broken.

### Things we tried and dropped
- "Ball missing → dead" (v0): far too much dead ball, explained above.
- Features "length of the current missing-ball stretch" and "ball off the pitch": removed from the
  model during the v2 rework, where the remaining features did the job (see `features.py`).
- Smart Tagging: not used, because the sample games have no tagging, so we had no fair way to
  check whether it helps. It is the most obvious next input.
- Broadcast video/tracking: not aligned with the tactical frames, not used.

## 4. Setting up on your machine

```bash
git clone git@github.com:isaad18/more-than-a-hack-2027.git
cd more-than-a-hack-2027
uv venv --python 3.11 .venv            # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r solution/requirements.txt
```

**Match data** (from the organisers' download, never committed). `games.py` expects these names
under `data/` (copy or symlink the files you received):

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_homography.csv
data/tactical/BAR-ATH_tactical_smart_tagging.xml
data/broadcast/BAR-ATH_broadcast_atd.txt          (optional, unused)
data/broadcast/BAR-ATH_broadcast_atd.xml          (optional, unused)
data/broadcast/BAR-ATH_broadcast_homography.csv   (optional, unused)
```

Or set `MTAH_DATA_DIR` to wherever the data lives.

**Sample games** (only needed to retrain or evaluate): `solution/get_external.sh` downloads them
into `data/external/`.

## 5. Running it

All commands from the repo root:

```bash
.venv/bin/python solution/run.py            # -> evaluation/results/NoTime.csv (~10 s)
.venv/bin/python evaluation/validate_submission.py evaluation/results/NoTime.csv
.venv/bin/python evaluation/run_evaluation.py   # bug alarm on the public 5 min only
```

Expected MD5 of `NoTime.csv`: **`26a932514a583d70aa8e6542c4b96943`** (`md5 -q` on macOS,
`md5sum` on Linux). If you get a different checksum without changing code, tell the team.

Full rebuild including the model (~3 min):

```bash
solution/get_external.sh
.venv/bin/python solution/classifier.py     # held-out check, then retrain possession_model.pkl
.venv/bin/python solution/tune.py           # HMM cost grid on held-out sample games
.venv/bin/python solution/dev_eval.py       # held-out scores + BAR-ATH sanity stats + bug alarm
.venv/bin/python solution/run.py
```

`possession_model.pkl` is committed and depends on scikit-learn 1.9.1 exactly. Keep the pinned
versions in `solution/requirements.txt`.

## 6. Code map (`solution/`)

| File | What it does |
|---|---|
| `load.py` | Reads the BAR-ATH tracking into arrays in metres; caches to `data/cache/`. |
| `features.py` | Every per-frame feature (movement, shape, distances, touch timing, ball-gap context). |
| `rules.py` | Touch detection only (1.5 m radius, 0.5 m contested margin). |
| `external.py` | Loads the Metrica sample games and turns their events into per-frame labels. |
| `devdata.py` | Caching for sample games and `degrade()`, which hides their ball like BAR-ATH. |
| `classifier.py` | Trains / loads the boosted-tree model; leave-one-game-out check. |
| `decode.py` | The 3-state HMM and Viterbi; switch cost 30, dead cost 15. |
| `pipeline.py` | features → model → decoder, for any match. |
| `run.py` | Builds the submission CSV. |
| `tune.py` | Grid search of the HMM costs on held-out sample games. |
| `dev_eval.py` | Prints held-out scores, BAR-ATH sanity statistics and the public bug alarm. |

## 7. How we work

- **Measure every change on the held-out sample games** (`dev_eval.py`), then look at the BAR-ATH
  sanity stats. Only keep a change if the held-out spellF1 improves and the sanity stats stay
  plausible. The public 5 minutes are a smoke alarm, not a score.
- Re-check the MD5 after any change and update it in `docs/NoTime.md` and here.
- Commits are made by the repo owner (isaad18); send changes to them rather than pushing directly.

## 8. Open work (pick one)

1. **Too much dead ball on BAR-ATH** (30.8% vs ~20% expected). Our sample-game labels count a
   stoppage from ball-out until the restart, which makes stoppages long, and the model copies
   that. Ideas: shorten the dead spell in `external.py` (e.g. end it when players start moving
   again), or raise the dead-ball entry cost — tested on held-out games only.
2. **Smart Tagging set pieces** as restart hints for D boundaries. Needs a way to validate it
   without BAR-ATH labels.
3. **More labelled training games** (hand labels on *other* matches are allowed) or other
   public tracking+event datasets.
4. **Technical document:** proofread `docs/NoTime.md`, optionally export to `NoTime.pdf`.

## 9. Submission checklist

- [ ] Final code committed and pushed to the public fork.
- [ ] `NoTime.csv` regenerated from that commit; MD5 matches; validator passes.
- [ ] `NoTime.md` (or `.pdf`) up to date with the final scores and MD5.
- [ ] Repo contains no match data and no per-frame derived data.
- [ ] Google Form filled: CSV, repo URL, technical document, all 3–5 team members.
