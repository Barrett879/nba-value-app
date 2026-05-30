#!/usr/bin/env python3
"""Train the "Likely suitors" DESTINATION model — predict which team a free agent signs with.

For every historical FA event (salary jump >= NEW_CONTRACT_PCT, 2013-2025), build one row
per candidate team (event x 30 teams) with self-contained features, label = 1 for the team
the player ACTUALLY signed with. A gradient-boosted classifier scores each team; live, its
score is z-blended (40%) with the hand-built rank (60%) per board.

Out-of-sample (train 2013-19, test 2020-25, 733 test events), vs the hand-score:
    hand-score (shipped)   top-1 48.2%  top-5 59.8%  mean-rank 7.30
    blend 40% GBM + 60% hand   top-1 52.8%  top-5 61.4%  mean-rank 7.11   <- ships

Features (all computable live from the board, no future-season leakage):
  is_inc, offer, offer_ratio, cap_space, pos_slot, tier_rank, value, age, barrett,
  and incumbent interactions (inc x age / value / barrett / offer_ratio).

Saves -> models/suitor_destination_v1.joblib  {model, feats, tier_rank, blend_w}
Run:   python scripts/train_suitor_destination.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
import joblib                                                                 # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier                   # noqa: E402
import team_suitors as ts                                                     # noqa: E402
from utils import (build_ranked_projected, fetch_league_stats,               # noqa: E402
                   fetch_player_positions_detailed, normalize, SEASONS,
                   NEW_CONTRACT_PCT, SALARY_CAP_M)

TIERS = ['title'] * 8 + ['playoff'] * 8 + ['bye'] * 7 + ['rebuild'] * 7
TIER_RANK = {'title': 3, 'playoff': 2, 'bye': 1, 'rebuild': 0}
FEATS = ['is_inc', 'offer', 'offer_ratio', 'cap_space', 'pos_slot', 'tier_rank',
         'value', 'age', 'barrett', 'inc_age', 'inc_value', 'inc_barrett', 'inc_ratio']
BLEND_W = 0.4
GBM_KW = dict(random_state=0, max_iter=400, learning_rate=0.05, max_depth=5, l2_regularization=2.0)
yr = lambda s: int(s[:4])


def _twpct(raw):
    full = raw[raw['GP'] >= 58]
    med = full.groupby('TEAM_ABBREVIATION')['W_PCT'].median()
    fb = raw.sort_values('GP', ascending=False).groupby('TEAM_ABBREVIATION')['W_PCT'].first()
    return med.reindex(fb.index).fillna(fb).sort_values(ascending=False)


def build_rows():
    pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)
             if yr(SEASONS[i]) >= 2013 and yr(SEASONS[i + 1]) >= 2012]
    rows, eid = [], 0
    for prev, curr in pairs:
        try:
            box = fetch_league_stats(prev, 'Regular Season')
            pdf = build_ranked_projected(prev); cdf = build_ranked_projected(curr)
            posmap = fetch_player_positions_detailed(prev, cache_v=3) or {}
        except Exception:
            continue
        if pdf.empty or cdf.empty or box.empty or 'W_PCT' not in box.columns:
            continue
        Y = yr(prev); capM = SALARY_CAP_M.get(curr, 140.0); mleM = 0.095 * capM
        payroll = pdf.groupby('Team')['salary'].sum() / 1e6
        teams = list(payroll.index)
        tw = _twpct(box); tier = {t: TIERS[min(i, 29)] for i, t in enumerate(tw.index)}
        age = dict(zip(box['PLAYER_ID'], box.get('AGE', [])))
        pr = pdf.copy()
        pr['pos'] = pr['Player'].map(lambda p: ts.resolve_position(p, posmap.get(normalize(p), ''), {}))
        pr['pn'] = pr['Player'].map(normalize)
        rosters = {t: pr[pr['Team'] == t][['pn', 'pos', 'barrett_score']].rename(
            columns={'barrett_score': 'barrett'}) for t in teams}
        cur = cdf[['PLAYER_ID', 'Team', 'salary']].rename(columns={'Team': 'ct', 'salary': 'sc'})
        mg = pdf[pdf['salary'] > 0].merge(cur, on='PLAYER_ID', how='left')
        mg = mg[mg['sc'].notna() & (mg['sc'] > 0)]; mg['pct'] = (mg['sc'] - mg['salary']) / mg['salary']
        for _, r in mg[mg['pct'].abs() >= NEW_CONTRACT_PCT].iterrows():
            T_act, inc = r['ct'], r['Team']
            if T_act not in teams:
                continue
            pid = int(r['PLAYER_ID']); value = r['sc'] / 1e6; bar = float(r['barrett_score'])
            a = age.get(pid); a = float(a) if a == a and a is not None else 27.0
            ppos = ts.resolve_position(r['Player'], posmap.get(normalize(r['Player']), ''), {})
            pn = normalize(r['Player'])
            for t in teams:
                is_inc = int(t == inc)
                rost = rosters[t]
                if is_inc:
                    rost = rost[rost['pn'] != pn]
                need = ts.roster_need(bar, ppos, rost.rename(columns={'pn': 'player'}))
                cap_space = max(0.0, capM - float(payroll.get(t, 0.0)))
                tool = max(cap_space, mleM, value if is_inc else 0.0)
                offer = min(value, tool); ratio = offer / value if value else 0.0
                rows.append(dict(
                    eid=eid, year=Y, label=int(t == T_act), is_inc=is_inc, offer=offer,
                    offer_ratio=ratio, cap_space=cap_space, pos_slot=min(need['slot'], 9),
                    tier_rank=TIER_RANK.get(tier.get(t, 'bye'), 1), value=value, age=a, barrett=bar,
                    inc_age=is_inc * a, inc_value=is_inc * value, inc_barrett=is_inc * bar,
                    inc_ratio=is_inc * ratio, tier=tier.get(t, 'bye')))
            eid += 1
    return pd.DataFrame(rows)


def _rank_eval(df, col):
    r = np.array([(g.sort_values(col, ascending=False)['label'].values == 1).argmax() + 1
                  for _, g in df.groupby('eid')])
    return (r == 1).mean() * 100, (r <= 5).mean() * 100, r.mean()


def main():
    df = build_rows()
    print(f'rows {len(df)} | events {df.eid.nunique()}')
    tr, te = df[df.year <= 2019].copy(), df[df.year >= 2020].copy()
    te['hand'] = te.apply(lambda x: x['offer'] * ts._FIT_FACTOR.get(int(x['pos_slot']), 0.10)
        * (1.0 if x['is_inc'] else ts.desire_weight(x['tier'], x['age'], x['value']))
        * (5.0 if x['is_inc'] else 1.0), axis=1)
    clf = HistGradientBoostingClassifier(**GBM_KW).fit(tr[FEATS], tr['label'])
    te['p'] = clf.predict_proba(te[FEATS])[:, 1]
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    te['zp'] = te.groupby('eid')['p'].transform(z); te['zh'] = te.groupby('eid')['hand'].transform(z)
    te['blend'] = BLEND_W * te['zp'] + (1 - BLEND_W) * te['zh']
    print('OUT-OF-SAMPLE (test 2020-25)      top1   top5   mean')
    print('  hand-score   %5.1f%% %5.1f%%  %5.2f' % _rank_eval(te, 'hand'))
    print('  blend 40/60  %5.1f%% %5.1f%%  %5.2f' % _rank_eval(te, 'blend'))

    final = HistGradientBoostingClassifier(**GBM_KW).fit(df[FEATS], df['label'])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'models', 'suitor_destination_v1.joblib')
    joblib.dump({'model': final, 'feats': FEATS, 'tier_rank': TIER_RANK, 'blend_w': BLEND_W}, out)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
