#!/usr/bin/env bash
# Download Metrica's public sample games (the development set) into data/external/.
# Only needed to re-train the model; run.py uses the committed possession_model.pkl.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MTAH_DATA_DIR:-$ROOT/data}/external"
BASE="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
mkdir -p "$DEST"

for f in \
  Sample_Game_1/Sample_Game_1_RawEventsData.csv \
  Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
  Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv \
  Sample_Game_2/Sample_Game_2_RawEventsData.csv \
  Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
  Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
  Sample_Game_3/Sample_Game_3_events.json \
  Sample_Game_3/Sample_Game_3_metadata.xml \
  Sample_Game_3/Sample_Game_3_tracking.txt; do
  echo "downloading $f"
  curl -sfL -o "$DEST/$(basename "$f")" "$BASE/$f"
done
echo "done -> $DEST"
