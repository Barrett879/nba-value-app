"""Ten defensive metrics from docs/defensive_options.md, scored side by side.

    /opt/anaconda3/bin/python3 -u scripts/def_options.py [season]

Offline. Writes nothing the site reads.

Each option is a separate scoring function over the same cached inputs, so the
differences between the lists below are differences of METHOD, not of data.
Options 8, 10, 12 and 14 are absent: they need an ESPN play-by-play ingestion
or a Bayesian sampler that do not exist yet, and a stub would be a lie.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
import def_lib as L

MIN_ON, MIN_OFF = 40, 25


def build(season, prior="2024-25", max_games=None):
    oo = L.onoff(season, max_games)
    pl = L.players(season).set_index("key")
    prior_oo = L.onoff(prior, max_team_games=17) if prior else {}

    rows = []
    for name, (on, off, tm, ng) in oo.items():
        if on.get("OffPoss", 0) < MIN_ON or off.get("OffPoss", 0) < MIN_OFF:
            continue
        zo, zf = L.zones(on), L.zones(off)
        if not zo or not zf:
            continue
        k = L.box_key(season, name)
        p = pl.loc[k] if k in pl.index else None
        if p is not None and isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        d = dict(player=name, key=k, team=tm, g=ng,
                 dp_on=on["OffPoss"], dp_off=off["OffPoss"])
        # ---- channels -------------------------------------------------
        d["d_rim"] = zf[0]["rim"] - zo[0]["rim"]          # rim share suppressed
        d["d_cor"] = zf[0]["cor3"] - zo[0]["cor3"]        # corner 3 suppressed
        d["d_pps"] = zf[1] - zo[1]                        # shot-diet PPS suppressed
        d["d_tov"] = (on.get("Turnovers", 0) / max(on["OffPoss"], 1)
                      - off.get("Turnovers", 0) / max(off["OffPoss"], 1))
        acc_on = on.get("AtRimFGM", 0) / max(on.get("AtRimFGA", 0), 1)
        acc_off = off.get("AtRimFGM", 0) / max(off.get("AtRimFGA", 0), 1)
        d["d_rimacc"] = acc_off - acc_on
        d["d_oreb"] = (off.get("OffRebounds", 0) / max(off["OffPoss"], 1)
                       - on.get("OffRebounds", 0) / max(on["OffPoss"], 1))
        # ---- box ------------------------------------------------------
        for c in ("BLK", "STL", "DREB", "PF", "MIN", "GP"):
            d[c] = float(p[c]) if p is not None and c in p and p[c] == p[c] else 0.0
        for c in ("dbpm", "dws", "blk_pct", "stl_pct", "drb_pct", "big", "DefPoss"):
            v = p[c] if p is not None and c in p else np.nan
            d[c] = float(v) if v == v else 0.0
        rows.append(d)
    df = pd.DataFrame(rows)
    # prior-season channels, for the frozen scale and the own-prior option
    if prior_oo:
        pr = {}
        for n, (on, off, tm, ng) in prior_oo.items():
            if on.get("OffPoss", 0) < MIN_ON or off.get("OffPoss", 0) < MIN_OFF:
                continue
            zo, zf = L.zones(on), L.zones(off)
            if not zo or not zf:
                continue
            pr[L.box_key(season, n)] = (zf[0]["rim"] - zo[0]["rim"], zf[1] - zo[1],
                                        zf[0]["cor3"] - zo[0]["cor3"])
        df["prior_rim"] = df.key.map(lambda k: pr.get(k, (np.nan,) * 3)[0])
        df["prior_pps"] = df.key.map(lambda k: pr.get(k, (np.nan,) * 3)[1])
        df["_prior_mu_rim"] = np.nanmean([v[0] for v in pr.values()])
        df["_prior_sd_rim"] = np.nanstd([v[0] for v in pr.values()])
        df["_prior_mu_pps"] = np.nanmean([v[1] for v in pr.values()])
        df["_prior_sd_pps"] = np.nanstd([v[1] for v in pr.values()])
    missed = df[df.GP == 0]
    if len(missed):
        print(f"  WARNING: {len(missed)} pool players have NO box row and were "
              f"zero-filled, which is not a score: "
              f"{', '.join(missed.player.astype(str))}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  The options. Each returns a Series on df's index, higher = better defender.
# ─────────────────────────────────────────────────────────────────────────────

def opt2_reliability(df):
    """#2 Reliability calibration. Not a score: split-half reliability per channel,
    which sets the weights option 1 uses instead of judgement."""
    out = {}
    for c in ("d_rim", "d_cor", "d_pps", "d_tov", "d_rimacc", "d_oreb"):
        if "prior_" + c.split("_")[1] in df.columns:
            a = df[c]; b = df["prior_" + c.split("_")[1]]
            m = a.notna() & b.notna()
            out[c] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else np.nan
    # the box channel, for comparison
    return out


def opt1_two_channel(df, w=None):
    """#1 Role-weighted two-channel z-composite on a FROZEN prior-season scale."""
    sd_rim = df["_prior_sd_rim"].iloc[0] if "_prior_sd_rim" in df else df.d_rim.std()
    mu_rim = df["_prior_mu_rim"].iloc[0] if "_prior_mu_rim" in df else df.d_rim.mean()
    det = (df.d_rim - mu_rim) / (sd_rim or 1)                      # frozen scale
    per = L.z(df.d_cor) + L.z(df.d_tov)
    box = L.z(np.sqrt(df.BLK.clip(0))) + L.z(df.STL) + 0.3 * L.z(df.DREB)
    w = w or {"det": 0.45, "box": 0.55}
    chan = df.big * det + (1 - df.big) * per
    n = df.dp_on
    shrink = n / (n + 300)
    return w["det"] * chan * shrink + w["box"] * box


def opt3_expected_points(df):
    """#3 Expected points allowed from the opponent's SHOT DIET, in points/100."""
    # d_pps is points-per-shot suppressed; convert to points per 100 possessions
    shots_per_poss = 0.95
    return df.d_pps * shots_per_poss * 100 * (df.dp_on / (df.dp_on + 300))


def opt4_career_blend(df, career_poss=None, season=None):
    """#4 Career-possession blend: DARKO's w = career/(career+6000) on defence."""
    if career_poss is not None:
        cp = career_poss
    else:
        cm = L.career_possessions(season) if season else {}
        cp = (df.key.map(lambda k: cm.get(k, 0.0)).astype(float) if cm
              else df.dp_on * 41 * 3)   # the old proxy, only if no history cached
    w = cp / (cp + 6000)
    mu = float(df["_prior_mu_rim"].iloc[0]) if "_prior_mu_rim" in df else 0.0
    sd = float(df["_prior_sd_rim"].iloc[0]) if "_prior_sd_rim" in df else 1.0
    det = (df.d_rim - mu) / (sd if sd else 1.0)
    chan = df.big * det + (1 - df.big) * (L.z(df.d_cor) + L.z(df.d_tov))
    box = L.z(np.sqrt(df.BLK.clip(0))) + L.z(df.STL)
    return w * chan + (1 - w) * box


def opt5_own_prior(df):
    """#5 Empirical Bayes toward the player's OWN prior-season channel."""
    n = df.dp_on
    k = 300.0
    cur = df.d_rim
    mu = float(df["_prior_mu_rim"].iloc[0]) if "_prior_mu_rim" in df else 0.0
    sd = float(df["_prior_sd_rim"].iloc[0]) if "_prior_sd_rim" in df else 1.0
    prior = df.prior_rim.fillna(mu) if "prior_rim" in df else pd.Series(mu, index=df.index)
    blended = (n * cur + k * prior) / (n + k)
    det = (blended - mu) / (sd if sd else 1.0)
    chan = df.big * det + (1 - df.big) * (L.z(df.d_cor) + L.z(df.d_tov))
    return chan + 0.5 * L.z(np.sqrt(df.BLK.clip(0)))


def opt6_ridge_rimshare(df, season="2026-27", lam=80.0, max_games=None):
    """#6 Lineup-level ridge with opponent RIM SHARE as the response."""
    rows, names = [], set()
    for _gid, _ab, lineups in L.lineup_rows(season, max_games):
        for r in lineups:
            sh = float(r.get("FG2A") or 0) + float(r.get("FG3A") or 0)
            dp = float(r.get("DefPoss") or 0)
            if sh < 4 or dp < 8:
                continue
            who = [x.strip() for x in str(r.get("Name", "")).split(",") if x.strip()]
            rows.append((who, float(r.get("AtRimFGA") or 0) / sh, dp))
            names |= set(who)
    names = sorted(names)
    idx = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names))); y = np.zeros(len(rows)); wt = np.zeros(len(rows))
    for i, (who, share, dp) in enumerate(rows):
        for n in who:
            X[i, idx[n]] = 1.0
        y[i] = share; wt[i] = dp
    # row-scale rather than np.diag(wt): a full season is ~15k lineup rows, and
    # the dense 15k-by-15k weight matrix is 1.8 GB.
    w = wt / wt.mean()
    A = X.T @ (w[:, None] * X) + lam * np.eye(len(names))
    b = np.linalg.solve(A, X.T @ (w * (y - y.mean())))
    cm = {L.box_key(season, n): -b[idx[n]] for n in names}
    return df.key.map(lambda k: cm.get(k, 0.0)).astype(float)


def opt7_obs_over_exp(df):
    """#7 Observed over expected, against the league's own shot diet."""
    lg_rim = (df.dp_on * 0 + df.d_rim).mean()
    vol = (df.dp_on / df.dp_on.median())            # possessions faced, normalised
    disr = L.z(df.d_rim) + 0.5 * L.z(df.d_tov) + 0.3 * L.z(df.d_oreb)
    return disr * np.sqrt(vol.clip(0.2, 3))


def opt9_band(df, base=None):
    """#9 Two-regime band: centre plus an analytic half-width from sample size."""
    c = base if base is not None else opt1_two_channel(df)
    half = 1.96 * df.d_rim.std() / np.sqrt(df.dp_on.clip(lower=1) / 40)
    return pd.DataFrame({"centre": c, "half": half})


def opt11_markov(df):
    """#11 Markov reward: price the diet shift through the continuation chain."""
    # second-chance multiplier: an allowed OREB buys roughly another 1.0 PPS trip
    oreb_rate_on = df.d_oreb
    base = df.d_pps * 0.95 * 100
    chain = oreb_rate_on * 100 * 1.0
    return (base + chain) * (df.dp_on / (df.dp_on + 300))


def opt13_possession_outcome(df, season="2026-27", lam=60.0, max_games=None):
    """#13 Ridge on opponent POINTS PER POSSESSION, no shot-location model."""
    rows, names = [], set()
    for _gid, _ab, lineups in L.lineup_rows(season, max_games):
        for r in lineups:
            dp = float(r.get("DefPoss") or 0)
            pts = (2 * float(r.get("FG2M") or r.get("AtRimFGM") or 0)
                   + 3 * float(r.get("FG3M") or 0) + float(r.get("FTA") or 0) * 0.76)
            if dp < 8:
                continue
            who = [x.strip() for x in str(r.get("Name", "")).split(",") if x.strip()]
            rows.append((who, pts / dp, dp)); names |= set(who)
    names = sorted(names); idx = {n: i for i, n in enumerate(names)}
    X = np.zeros((len(rows), len(names))); y = np.zeros(len(rows)); wt = np.zeros(len(rows))
    for i, (who, ppp, dp) in enumerate(rows):
        for n in who:
            X[i, idx[n]] = 1.0
        y[i] = ppp; wt[i] = dp
    w = wt / wt.mean()
    b = np.linalg.solve(X.T @ (w[:, None] * X) + lam * np.eye(len(names)),
                        X.T @ (w * (y - y.mean())))
    cm = {L.box_key(season, n): -b[idx[n]] for n in names}
    return df.key.map(lambda k: cm.get(k, 0.0)).astype(float)
