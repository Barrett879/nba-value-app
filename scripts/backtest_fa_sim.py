#!/usr/bin/env python3
"""Backtest a LEAGUE-WIDE, market-clearing free-agency simulation.

The shipped suitor board ranks teams for each player INDEPENDENTLY — it never
knows that a team already spent its cap on someone else, or filled its roster.
This script tests whether clearing the market (process FAs best-first, mutate
each team's cap / roster / positional depth as players sign, so spent teams drop
out for later FAs) predicts destinations any better than the independent ranking.

Same per-season reconstruction + scorer as backtest_suitors.py, but grouped by
season with mutable team state. Three modes, identical scorer, measured on the
same events:

  independent   - score vs the PRE-FA snapshot, no mutation (today's model).
  cleared/TF    - market-clearing with constraints, but advance state on the
                  ACTUAL signing (teacher-forced) -> per-event accuracy given a
                  realistic, de-cascaded state.
  cleared/free  - the real product: advance state on our OWN pick, errors cascade.

Run:  python scripts/backtest_fa_sim.py
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
import numpy as np, pandas as pd, collections

USE_GBM = os.environ.get("FA_SIM_NO_GBM") != "1"     # blend the destination model (parity with prod)
ROSTER_MAX = 15
MIN_M = 2.3                                           # veteran-minimum offer
TIERS = ['title'] * 8 + ['playoff'] * 8 + ['bye'] * 7 + ['rebuild'] * 7
yr = lambda s: int(s[:4])
pairs = [(SEASONS[i + 1], SEASONS[i]) for i in range(len(SEASONS) - 1)
         if yr(SEASONS[i]) >= 2013 and yr(SEASONS[i + 1]) >= 2012]


def twpct(raw):
    full = raw[raw['GP'] >= 58]
    med = full.groupby('TEAM_ABBREVIATION')['W_PCT'].median()
    fb = raw.sort_values('GP', ascending=False).groupby('TEAM_ABBREVIATION')['W_PCT'].first()
    return med.reindex(fb.index).fillna(fb).sort_values(ascending=False)


_DEST = ts._dest_model() if USE_GBM else None


def score_board(fa, st, mleM):
    """Rank teams for one FA against the CURRENT (mutable) league state, exactly
    like rank_suitors: offer x desire x incumbent, optional GBM blend. Each entry
    carries `feasible` (roster spot + a tool that can fund the offer)."""
    out = []
    for t in st['teams']:
        is_inc = (t == fa['inc'])
        rost = st['rost'][t]
        if is_inc:
            rost = rost[rost['player'] != fa['pn']]
        need = ts.roster_need(fa['bar'], fa['pos'], rost)
        if need['slot'] >= ts.INTEREST_DEPTH + (2 if is_inc else 0):
            continue
        cap = max(0.0, st['cap'][t]); val = fa['value']
        exc = mleM if not st['mle'][t] else 0.0
        tool = max(cap, exc, val if is_inc else 0.0)
        fit = 1.0 if is_inc else ts._FIT_FACTOR.get(need['slot'], 0.45)
        offer = min(val, tool, val * fit)
        if offer < 1.0:
            continue
        tl = st['tier'].get(t, '')
        des = 1.0 if is_inc else ts.desire_weight(tl, fa['age'], val)
        if (not is_inc) and t in ts._MARKET_PULL_TEAMS:
            des *= ts._MARKET_PULL
        rank = offer * des * (ts.INCUMBENT_WEIGHT if is_inc else 1.0)
        spot = st['count'][t] < ROSTER_MAX
        afford = (is_inc or cap + 1e-6 >= offer
                  or (not st['mle'][t] and val <= mleM + 1e-6) or val <= MIN_M + 1e-6)
        av = float(fa['age']) if fa['age'] is not None else 27.0
        raw_offer = min(val, tool); ratio = raw_offer / val if val else 0.0
        inc = float(is_inc)
        feat = [inc, raw_offer, ratio, cap, float(min(need['slot'], 9)),
                float(ts._TIER_RANK_MAP.get(ts._timeline_key(tl), 1)), val, av, fa['bar'],
                inc * av, inc * val, inc * fa['bar'], inc * ratio]
        out.append({'team': t, 'rank': rank, 'offer': offer,
                    'feasible': bool(spot and afford), 'feat': feat})
    if _DEST and len(out) > 1:
        try:
            proba = _DEST['model'].predict_proba(np.asarray([d['feat'] for d in out], float))[:, 1]
            hand = np.asarray([d['rank'] for d in out], float)
            zsc = lambda a: (a - a.mean()) / a.std() if a.std() > 1e-9 else a * 0.0
            w = float(_DEST.get('blend_w', 0.4))
            for d, b in zip(out, w * zsc(proba) + (1.0 - w) * zsc(hand)):
                d['rank'] = float(b)
        except Exception:
            pass
    out.sort(key=lambda d: -d['rank'])
    return out


def sign(st, team, fa, offer):
    """Apply a signing to the mutable state: take a roster spot, draw down the cap
    or burn the mid-level (Bird re-signs go over the cap, so just a spot)."""
    st['rost'][team] = pd.concat([st['rost'][team], pd.DataFrame(
        [{'player': fa['pn'], 'pos': fa['pos'], 'barrett': fa['bar']}])], ignore_index=True)
    st['count'][team] += 1
    if st['cap'][team] + 1e-6 >= offer:
        st['cap'][team] -= offer
    elif (not st['mle'][team]) and fa['value'] <= (st['mleM']) + 1e-6 and team != fa['inc']:
        st['mle'][team] = True


def fresh_state(base):
    return {**base,
            'cap': dict(base['cap0']), 'mle': {t: False for t in base['teams']},
            'count': dict(base['count0']),
            'rost': {t: base['rost0'][t].copy() for t in base['teams']}}


def offer_to(board, team):
    for d in board:
        if d['team'] == team:
            return d['offer']
    return None


# ── reconstruct each offseason, grouped by season ────────────────────────────
seasons = []     # each: {curr, base(state), fas(sorted best-first)}
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
    teams = list(pdf.groupby('Team').groups.keys())
    tw = twpct(box); tier = {t: TIERS[min(i, 29)] for i, t in enumerate(tw.index)}
    age = dict(zip(box['PLAYER_ID'], box.get('AGE', [])))
    pr = pdf.copy()
    pr['pos'] = pr['Player'].map(lambda p: ts.resolve_position(p, posmap.get(normalize(p), ''), {}))
    pr['pn'] = pr['Player'].map(normalize)
    # FA events first (salary jump >= threshold), so we can take them OFF each
    # team's books before computing pre-FA cap space + roster spots.
    cur = cdf[['PLAYER_ID', 'Team', 'salary']].rename(columns={'Team': 'ct', 'salary': 'sc'})
    mg = pdf[pdf['salary'] > 0].merge(cur, on='PLAYER_ID', how='left')
    mg = mg[mg['sc'].notna() & (mg['sc'] > 0)]; mg['pct'] = (mg['sc'] - mg['salary']) / mg['salary']
    fas = []
    for _, r in mg[mg['pct'].abs() >= NEW_CONTRACT_PCT].iterrows():
        if r['ct'] not in teams or r['Team'] not in teams:
            continue
        pid = int(r['PLAYER_ID'])
        fas.append({'pn': normalize(r['Player']), 'act': r['ct'], 'inc': r['Team'],
                    'value': r['sc'] / 1e6, 'bar': float(r['barrett_score']),
                    'pos': ts.resolve_position(r['Player'], posmap.get(normalize(r['Player']), ''), {}),
                    'age': age.get(pid)})
    fas.sort(key=lambda f: -f['value'])               # market order: stars sign first
    # PRE-FA books = players who stay under contract (drop the offseason's FAs,
    # who free up their salary + roster spot). This is what gives a team room.
    fa_set = {(f['inc'], f['pn']) for f in fas}
    pr['_fa'] = [(t, p) in fa_set for t, p in zip(pr['Team'], pr['pn'])]
    held = pr[~pr['_fa']]
    committed = held.groupby('Team')['salary'].sum() / 1e6
    rost0 = {t: held[held['Team'] == t][['pn', 'pos', 'barrett_score']].rename(
        columns={'pn': 'player', 'barrett_score': 'barrett'}).reset_index(drop=True) for t in teams}
    cap0 = {t: max(0.0, capM - float(committed.get(t, 0.0))) for t in teams}
    count0 = {t: len(rost0[t]) for t in teams}
    base = {'curr': curr, 'teams': teams, 'tier': tier, 'mleM': mleM,
            'rost0': rost0, 'cap0': cap0, 'count0': count0}
    seasons.append({'base': base, 'fas': fas})
signal.alarm(0)


# ── evaluate the three modes ─────────────────────────────────────────────────
def run():
    res = {m: {'rank': [], 'recent': []} for m in ['independent', 'cleared_tf', 'cleared_free']}
    act_infeasible = 0; total = 0
    for s in seasons:
        base = s['base']; mleM = base['mleM']; recent = yr(base['curr']) >= 2021
        st_tf, st_free = fresh_state(base), fresh_state(base)
        st_ind = fresh_state(base)                    # never mutated -> pre-FA snapshot
        for fa in s['fas']:
            total += 1
            # independent: static snapshot, ignore feasibility (today's model)
            b_ind = score_board(fa, st_ind, mleM)
            order = [d['team'] for d in b_ind]
            rk = order.index(fa['act']) + 1 if fa['act'] in order else 99
            res['independent']['rank'].append(rk)
            if recent: res['independent']['recent'].append(rk)
            # teacher-forced market clearing: feasible-ranked, advance on ACTUAL
            b_tf = score_board(fa, st_tf, mleM)
            feas = [d for d in b_tf if d['feasible']]
            ford = [d['team'] for d in feas]
            rk = ford.index(fa['act']) + 1 if fa['act'] in ford else 99
            res['cleared_tf']['rank'].append(rk)
            if recent: res['cleared_tf']['recent'].append(rk)
            if fa['act'] not in ford: act_infeasible += 1
            o = offer_to(b_tf, fa['act'])
            sign(st_tf, fa['act'], fa, o if o is not None else min(fa['value'], mleM))
            # free-running market clearing: advance on OUR pick (errors cascade)
            b_fr = score_board(fa, st_free, mleM)
            feas = [d for d in b_fr if d['feasible']]
            ford = [d['team'] for d in feas]
            rk = ford.index(fa['act']) + 1 if fa['act'] in ford else 99
            res['cleared_free']['rank'].append(rk)
            if recent: res['cleared_free']['recent'].append(rk)
            if feas:
                pick = feas[0]['team']
                sign(st_free, pick, fa, feas[0]['offer'])
    return res, act_infeasible, total


def report(tag, ranks):
    a = np.array(ranks)
    if not len(a):
        print(f'  {tag:<22} (no events)'); return
    print('  %-22s top1 %5.1f%%   top3 %5.1f%%   top5 %5.1f%%   mean %5.2f   (n=%d)'
          % (tag, (a == 1).mean() * 100, (a <= 3).mean() * 100, (a <= 5).mean() * 100, a.mean(), len(a)))


res, act_infeasible, total = run()
print(f'\nseasons: {len(seasons)} | FA events: {total} | GBM blend: {USE_GBM}')
print(f'actual destination filtered out by our cap/roster model: {act_infeasible}/{total} '
      f'({act_infeasible / max(total,1):.1%})\n')
print('ALL SEASONS (2013-2025)')
for m in ['independent', 'cleared_tf', 'cleared_free']:
    report(m, res[m]['rank'])
print('\nRECENT ONLY (2021-2025)')
for m in ['independent', 'cleared_tf', 'cleared_free']:
    report(m, res[m]['recent'])
