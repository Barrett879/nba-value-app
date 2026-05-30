#!/usr/bin/env python3
"""Learn the suitor-model DESIRE weights from historical free-agent signings.

The "Likely suitors" feature ranks teams by  offer x desire,  where
desire[tier][archetype] captures how keen a team of a given competitive TIMELINE
(title / playoff / bye / rebuild) is to PURSUE a player of a given ARCHETYPE
(star / young / prime / vet). Rather than hand-tuning those 16 numbers, this
script LEARNS them from who actually signed whom, and prints a block ready to
paste into team_suitors._DESIRE.

Method
------
1. Signings: for each consecutive season pair (2012-13 .. latest), a "new-team
   signing" is a player whose salary changed by >= NEW_CONTRACT_PCT (the model's
   own new-deal cutoff) AND who is on a different team the next season. The salary
   jump filters out pure trades; the team change isolates OUTSIDE interest, which
   is what the desire weight models (the incumbent is handled by Bird rights).
2. Signing team's tier: by its ACTUAL win% that season — the median W_PCT of its
   full-season players (GP >= 58), which is robust to a rested star skewing one
   player's record. Split top-8 / next-8 / next-7 / bottom-7 into the four tiers.
3. Player archetype: star (Barrett >= 20) / young (<25) / vet (>31) / prime (else).
4. Cap control: a rebuilder signs more of EVERYONE just because it has cap space.
   To isolate preference from opportunity, measure each tier's archetype MIX (the
   share of its own signings) and take LIFT vs the league-average mix — the
   cap-driven volume divides out, leaving preference.
5. Weight: normalize lift per archetype to max = 1  ->  a 0-1 desire weight / cell.

Caveats (read before trusting):
  - The star row is thin (~70 signings) -> smooth it by hand in _DESIRE.
  - Tiers use a win%-from-player-rows estimate, not official standings (verified
    accurate: real contenders land in 'title', tank teams in 'rebuild').
  - Lift removes the cap VOLUME confound; a residual (cap shifting the mix) remains.

Run:  python scripts/learn_desire_weights.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import build_ranked_projected, fetch_league_stats, SEASONS, NEW_CONTRACT_PCT  # noqa: E402

TIERS = ['title'] * 8 + ['playoff'] * 8 + ['bye'] * 7 + ['rebuild'] * 7
ARCHES = ['star', 'young', 'prime', 'vet']
TIER_NAMES = ['title', 'playoff', 'bye', 'rebuild']


def archetype(age, barrett):
    if barrett >= 20:                                 return 'star'
    if age == age and age is not None and age < 25:   return 'young'   # age == age guards NaN
    if age == age and age is not None and age > 31:   return 'vet'
    return 'prime'


def team_winpct(raw):
    """Each team's win% = median W_PCT of its full-season (GP >= 58) players."""
    full = raw[raw['GP'] >= 58]
    med = full.groupby('TEAM_ABBREVIATION')['W_PCT'].median()
    fb = raw.sort_values('GP', ascending=False).groupby('TEAM_ABBREVIATION')['W_PCT'].first()
    return med.reindex(fb.index).fillna(fb).sort_values(ascending=False)


def _weights_from(cell):
    """Lift-normalized desire weights from a (tier, archetype) signing counter."""
    tot_t = {t: sum(cell[(t, a)] for a in ARCHES) for t in TIER_NAMES}
    tot_a = {a: sum(cell[(t, a)] for t in TIER_NAMES) for a in ARCHES}
    g = sum(tot_t.values())
    des = {}
    for a in ARCHES:
        col = [(t, (cell[(t, a)] / tot_t[t]) / (tot_a[a] / g)
                if tot_t[t] and tot_a[a] and g else 0.0) for t in TIER_NAMES]
        mx = max(v for _, v in col) or 1.0
        for t, v in col:
            des[(t, a)] = v / mx
    return des


def main():
    yr = lambda s: int(s[:4])
    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)
             if yr(SEASONS[i]) >= 2013 and yr(SEASONS[i + 1]) >= 2012]
    cell = collections.Counter()
    cell_e, cell_l = collections.Counter(), collections.Counter()   # 2013-19 / 2020-25 holdout
    used = nsign = 0
    for prev, curr in pairs:
        try:
            raw = fetch_league_stats(prev, 'Regular Season')
            pdf = build_ranked_projected(prev)
            cdf = build_ranked_projected(curr)
        except Exception:
            continue
        if pdf.empty or cdf.empty or raw.empty or 'W_PCT' not in raw:
            continue
        used += 1
        tw = team_winpct(raw)
        tier = {t: TIERS[min(i, 29)] for i, t in enumerate(tw.index)}
        age = dict(zip(raw['PLAYER_ID'], raw.get('AGE', [])))
        cur = cdf[['PLAYER_ID', 'Team', 'salary']].rename(columns={'Team': 'ct', 'salary': 'sc'})
        mg = pdf[pdf['salary'] > 0].merge(cur, on='PLAYER_ID', how='left')
        mg = mg[mg['sc'].notna() & (mg['sc'] > 0)]
        mg['pct'] = (mg['sc'] - mg['salary']) / mg['salary']
        moved = mg[(mg['pct'].abs() >= NEW_CONTRACT_PCT) & (mg['Team'] != mg['ct'])]
        half = cell_e if yr(prev) <= 2019 else cell_l
        for _, r in moved.iterrows():
            dt = tier.get(r['ct'])
            if dt:
                key = (dt, archetype(age.get(r['PLAYER_ID']), r['barrett_score']))
                cell[key] += 1; half[key] += 1
                nsign += 1

    tot_t = {t: sum(cell[(t, a)] for a in ARCHES) for t in TIER_NAMES}
    tot_a = {a: sum(cell[(t, a)] for t in TIER_NAMES) for a in ARCHES}
    grand = sum(tot_t.values())
    print(f'season pairs: {used} | new-team signings: {nsign} | threshold: {NEW_CONTRACT_PCT}\n')

    print('RAW counts:        ' + ''.join(f'{a:>7}' for a in ARCHES))
    for t in TIER_NAMES:
        print(f'  {t:8} ' + ''.join(f'{cell[(t, a)]:7d}' for a in ARCHES))

    lift = {}
    for t in TIER_NAMES:
        for a in ARCHES:
            share = cell[(t, a)] / tot_t[t] if tot_t[t] else 0.0
            base = tot_a[a] / grand if grand else 0.0
            lift[(t, a)] = share / base if base else 0.0
    print('\nCAP-CONTROLLED LIFT (>1 = tier over-indexes on that archetype):')
    print('           ' + ''.join(f'{a:>7}' for a in ARCHES))
    for t in TIER_NAMES:
        print(f'  {t:8} ' + ''.join(f'{lift[(t, a)]:7.2f}' for a in ARCHES))

    print('\nDESIRE weights (lift normalized per archetype to max=1; paste into _DESIRE,')
    print('then hand-smooth the noisy `star` column):')
    des = {}
    for a in ARCHES:
        mx = max(lift[(t, a)] for t in TIER_NAMES) or 1.0
        for t in TIER_NAMES:
            des[(t, a)] = round(lift[(t, a)] / mx, 2)
    for t in TIER_NAMES:
        cells = ', '.join(f'"{a}": {des[(t, a)]:.2f}' for a in ARCHES)
        print(f'    "{t}":{" " * (8 - len(t))}{{{cells}}},')

    print('\nHOLDOUT — does it generalize? weights on 2013-19 vs 2020-25 signings:')
    we, wl = _weights_from(cell_e), _weights_from(cell_l)
    ev = [we[(t, a)] for t in TIER_NAMES for a in ARCHES]
    lv = [wl[(t, a)] for t in TIER_NAMES for a in ARCHES]
    nn = len(ev); me, ml = sum(ev) / nn, sum(lv) / nn
    den = (sum((e - me) ** 2 for e in ev) * sum((l - ml) ** 2 for l in lv)) ** 0.5 or 1.0
    r = sum((e - me) * (l - ml) for e, l in zip(ev, lv)) / den
    print(f'  early-vs-late weight correlation r = {r:.3f}  (high = generalizes)')
    flips = [f'{t}-{a}' for t in TIER_NAMES for a in ARCHES if abs(we[(t, a)] - wl[(t, a)]) > 0.35]
    print('  cells that FLIP (|delta|>0.35) = noise to flatten:', ', '.join(flips) or 'none')


if __name__ == '__main__':
    main()
