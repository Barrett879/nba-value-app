#!/usr/bin/env python3
"""Predictive backtest of the "Likely suitors" board.

For every historical free-agent event (a player whose salary jumped >= NEW_CONTRACT_PCT,
2013-2025), reconstruct the board from the PRIOR season — cap space from SALARY_CAP_M minus
team payroll, position fit, desire weights, skill fit — score all 30 teams, and check where
the player's ACTUAL signing team ranks. Ablates each component and sweeps the incumbent
(re-sign) weight.

Key findings (1,810 events):
  - 49% of FAs RE-SIGN with their own team — the single biggest signal.
  - Ablation (top-1 / top-5 / mean-rank vs random 3.3% / 16.7% / 15.5):
        money            5.4% / 24.2% / 13.71
        + position       6.9% / 27.5% / 12.70
        + desire        15.4% / 40.3% / 10.05   <- desire ~doubles accuracy (validates it)
        + skill         13.6% / 37.8% / 10.31   <- skill HURTS (now weighted 0 in the model)
  - Incumbent weight sweep on the +desire base: x5 -> top-1 45.9%, top-5 56.5% (3x better).
    -> team_suitors.INCUMBENT_WEIGHT = 5.0

Run:  python scripts/backtest_suitors.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # robust to invocation cwd
import signal
def h(s, f): raise TimeoutError()
signal.signal(signal.SIGALRM, h); signal.alarm(1500)
from utils import (build_ranked_projected, fetch_league_stats, fetch_advanced_stats,
                   fetch_player_positions_detailed, normalize, SEASONS, NEW_CONTRACT_PCT,
                   SALARY_CAP_M)
import team_suitors as ts
import numpy as np, collections

TIERS = ['title'] * 8 + ['playoff'] * 8 + ['bye'] * 7 + ['rebuild'] * 7
yr = lambda s: int(s[:4])
pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)
         if yr(SEASONS[i]) >= 2013 and yr(SEASONS[i + 1]) >= 2012]


def arche(age, b):
    if b >= 20: return 'star'
    if age == age and age is not None and age < 25: return 'young'
    if age == age and age is not None and age > 31: return 'vet'
    return 'prime'


def twpct(raw):
    full = raw[raw['GP'] >= 58]
    med = full.groupby('TEAM_ABBREVIATION')['W_PCT'].median()
    fb = raw.sort_values('GP', ascending=False).groupby('TEAM_ABBREVIATION')['W_PCT'].first()
    return med.reindex(fb.index).fillna(fb).sort_values(ascending=False)


# per event store the base components so we can re-rank under many configs cheaply
events = []   # each: {actual, inc, teams, offer{}, posf{}, des{}, skf{}}
resign = n = 0
for prev, curr in pairs:
    try:
        box = fetch_league_stats(prev, 'Regular Season')
        adv = fetch_advanced_stats(prev, 'Regular Season')
        pdf = build_ranked_projected(prev); cdf = build_ranked_projected(curr)
        posmap = fetch_player_positions_detailed(prev, cache_v=3) or {}
    except Exception:
        continue
    if pdf.empty or cdf.empty or box.empty or 'W_PCT' not in box.columns:
        continue
    capM = SALARY_CAP_M.get(curr, 140.0); mleM = 0.095 * capM
    payroll = pdf.groupby('Team')['salary'].sum() / 1e6
    teams = list(payroll.index)
    tw = twpct(box); tier = {t: TIERS[min(i, 29)] for i, t in enumerate(tw.index)}
    team_sk = ts.build_team_skills(box, adv)
    dsh = (1.0 - team_sk['shooting']).to_dict() if 'shooting' in team_sk else {}
    age = dict(zip(box['PLAYER_ID'], box.get('AGE', [])))
    bb = box[box['GP'] >= 30].copy(); vol = bb[bb['FG3A'] >= 2.0].copy()
    vol['s'] = (vol['FG3M'] / vol['FG3A']).rank(pct=True); shoot = dict(zip(vol['PLAYER_ID'], vol['s']))
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
        ppos = ts.resolve_position(r['Player'], posmap.get(normalize(r['Player']), ''), {})
        pn = normalize(r['Player']); a = arche(age.get(pid), bar); sh = shoot.get(pid, 0.25)
        n += 1; resign += (T_act == inc)
        offer, posf, des, skf = {}, {}, {}, {}
        for t in teams:
            is_inc = (t == inc)
            rost = rosters[t]
            if is_inc:                                    # FIX: he's not on his own roster as an incumbent
                rost = rost[rost['pn'] != pn]
            need = ts.roster_need(bar, ppos, rost.rename(columns={'pn': 'player'}))
            posf[t] = ts._FIT_FACTOR.get(need['slot'], 0.10)
            cap_space = max(0.0, capM - float(payroll.get(t, 0.0)))
            tool = max(cap_space, mleM, value if is_inc else 0.0)
            offer[t] = min(value, tool)
            des[t] = 1.0 if is_inc else ts.desire_weight(tier.get(t, ''), age.get(pid), value)
            skf[t] = sh * dsh.get(t, 0.5)
        events.append({'act': T_act, 'inc': inc, 'teams': teams,
                       'offer': offer, 'posf': posf, 'des': des, 'skf': skf})
signal.alarm(0)


def evalcfg(scorefn):
    rk = []
    for e in events:
        order = sorted(e['teams'], key=lambda t: -scorefn(e, t))
        rk.append(order.index(e['act']) + 1)
    rk = np.array(rk)
    return (rk == 1).mean() * 100, (rk <= 5).mean() * 100, rk.mean()


print(f'events: {n} | re-sign rate: {resign / n:.1%}\n')
print('ABLATION                         top1   top5   mean')
for name, fn in [
    ('money',                lambda e, t: e['offer'][t]),
    ('+position',            lambda e, t: e['offer'][t] * e['posf'][t]),
    ('+desire (best base)',  lambda e, t: e['offer'][t] * e['posf'][t] * e['des'][t]),
    ('+skill',               lambda e, t: e['offer'][t] * e['posf'][t] * e['des'][t] * (0.925 + 0.15 * e['skf'][t])),
]:
    print('  %-30s %5.1f%% %5.1f%%  %5.2f' % ((name,) + evalcfg(fn)))

print('\nINCUMBENT (re-sign) WEIGHT SWEEP on the +desire base       top1   top5   mean')
for M in [1, 2, 3, 5, 8, 12, 20]:
    fn = lambda e, t, M=M: e['offer'][t] * e['posf'][t] * e['des'][t] * (M if t == e['inc'] else 1.0)
    print('  incumbent x%-4d                                %5.1f%% %5.1f%%  %5.2f' % ((M,) + evalcfg(fn)))
