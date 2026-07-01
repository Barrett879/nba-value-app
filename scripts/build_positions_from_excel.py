"""Convert Barrett's hand-reviewed positions Excel into a CSV the app reads.

player_positions_review.xlsx (sheet 'positions', columns Player/Primary/Secondary) is the
source of truth Barrett edits. openpyxl isn't a runtime dep and the app reads CSVs, so this
writes data/player_positions_reviewed.csv, which team_suitors.load_player_positions layers
ABOVE the NBA 2K source (the review wins) and below player_positions_override.csv.

Re-run after editing the Excel:  python scripts/build_positions_from_excel.py
"""
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
xl = pd.read_excel(ROOT / "player_positions_review.xlsx", sheet_name="positions")

rows = []
for _, r in xl.iterrows():
    name = str(r.get("Player", "")).strip()
    pri = str(r.get("Primary", "")).strip()
    sec = str(r.get("Secondary", "")).strip()
    if not name or not pri or pri.lower() == "nan":
        continue
    pos = pri if (not sec or sec.lower() == "nan") else f"{pri}/{sec}"
    rows.append((name, pos))

out = ROOT / "data" / "player_positions_reviewed.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "positions"])
    w.writerows(rows)
print(f"wrote {out.relative_to(ROOT)} ({len(rows)} players)")
