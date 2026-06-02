#!/usr/bin/env python3
"""Does age-gating the 'star' desire make the suitor board MORE accurate?

The live desire_weight buckets anyone projected >= $22M as a generic 'star' with a
flat, near-tier-blind preference, so a 36-yo max vet is courted by rebuilders about
as much as by contenders. Hypothesis: aging stars (value >= $22M AND age > 31)
should follow the data-derived VET pattern (vets favor title 1.00 vs ~0.5
elsewhere), so rebuilders stop topping their board.

Rebuilds the same 1,810-event board as backtest_suitors.py but stores the raw
(age, value, tier-per-team) so desire variants re-score cheaply, then reports
top-1 / top-5 / mean-rank under the shipped config (offer x posf x desire x
incumbent-x5) for each variant -- OVERALL and on the AGING-STAR SUBSET, which is
the only place the change bites (so overall will barely move; the subset is the
decision).

Run:  python -u scripts/experiment_age_gate.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import signal
def _h(s, f): raise TimeoutError()
signal.signal(signal.SIGALRM, _h); signal.alarm(1500)
from utils import (build_ranked_projected, fetch_league_stats, fetch_advanced_stats,
                   fetch_player_positions_detailed, normalize, SEASONS, NEW_CONTRACT_PCT,
                   SALARY_CAP_M)
import team_suitors as ts
import numpy as np

TIERS = ['title'] * 8 + ['playoff'] * 8 + ['bye'] * 7 + ['rebuild'] * 7
yr = lambda s: int(s[:4])
pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)
         if yr(SEASONS[i]) >= 2013 and yr(SEASONS[i + 1]) >= 2012]
INC_W = ts.INCUMBENT_WEIGHT   # shipped 5.0


def twpct(raw):
    full = raw[raw['GP'] >= 58]
    med = full.groupby('TEAM_ABBREVIATION')['W_PCT'].median()
    fb = raw.sort_values('GP', ascending=False).groupby('TEAM_ABBREVIATION')['W_PCT'].first()
    return med.reindex(fb.index).fillna(fb).sort_values(ascending=False)


events, n = [], 0
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
        ppos = ts.resolve_position(r['Player'], posmap.get(normalize(r['Player']), ''), {})
        pn = normalize(r['Player']); pa = age.get(pid)
        n += 1
        offer, posf = {}, {}
        for t in teams:
            is_inc = (t == inc)
            rost = rosters[t]
            if is_inc:
                rost = rost[rost['pn'] != pn]
            need = ts.roster_need(bar, ppos, rost.rename(columns={'pn': 'player'}))
            posf[t] = ts._FIT_FACTOR.get(need['slot'], 0.10)
            cap_space = max(0.0, capM - float(payroll.get(t, 0.0)))
            tool = max(cap_space, mleM, value if is_inc else 0.0)
            offer[t] = min(value, tool)
        events.append({'act': T_act, 'inc': inc, 'teams': teams, 'offer': offer,
                       'posf': posf, 'tier': tier, 'age': pa, 'value': value})
signal.alarm(0)


def _aging_star(age, value):
    a = float(age) if (age is not None and age == age) else 27.0
    return value >= 22.0 and a > 31


def d_cur(tl, age, value):                       # current: star row, age-blind
    return ts.desire_weight(tl, age, value)


def d_vet(tl, age, value):                       # aging star -> vet row
    if _aging_star(age, value):
        return ts._DESIRE[ts._timeline_key(tl)]['vet']
    return ts.desire_weight(tl, age, value)


def d_geo(tl, age, value):                       # aging star -> sqrt(star*vet)
    if _aging_star(age, value):
        k = ts._timeline_key(tl)
        return (ts._DESIRE[k]['star'] * ts._DESIRE[k]['vet']) ** 0.5
    return ts.desire_weight(tl, age, value)


def d_avg(tl, age, value):                       # aging star -> mean(star, vet)
    if _aging_star(age, value):
        k = ts._timeline_key(tl)
        return 0.5 * ts._DESIRE[k]['star'] + 0.5 * ts._DESIRE[k]['vet']
    return ts.desire_weight(tl, age, value)


VARIANTS = [('current (star flat)', d_cur), ('aging->vet', d_vet),
            ('aging->geo(star,vet)', d_geo), ('aging->avg(star,vet)', d_avg)]


def score(e, t, df):
    des = 1.0 if t == e['inc'] else df(e['tier'].get(t, ''), e['age'], e['value'])
    w = INC_W if t == e['inc'] else 1.0
    return e['offer'][t] * e['posf'][t] * des * w


def evalsub(df, mask):
    rk = []
    for e, m in zip(events, mask):
        if not m:
            continue
        order = sorted(e['teams'], key=lambda t: -score(e, t, df))
        rk.append(order.index(e['act']) + 1)
    rk = np.array(rk)
    return len(rk), (rk == 1).mean() * 100, (rk <= 5).mean() * 100, rk.mean()


aging_mask = [_aging_star(e['age'], e['value']) for e in events]
all_mask = [True] * len(events)
print(f"\nevents: {n} | aging stars (value>=22 & age>31): {sum(aging_mask)}\n", flush=True)
for label, mask in [('OVERALL', all_mask), ('AGING-STAR SUBSET', aging_mask)]:
    print(f"=== {label} (n={sum(mask)}) ===", flush=True)
    print(f"  {'variant':<24}{'top1':>7}{'top5':>7}{'mean':>7}", flush=True)
    for name, df in VARIANTS:
        k, t1, t5, mr = evalsub(df, mask)
        print(f"  {name:<24}{t1:>6.1f}%{t5:>6.1f}%{mr:>7.2f}", flush=True)
    print(flush=True)
