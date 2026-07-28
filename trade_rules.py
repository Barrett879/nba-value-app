"""Trade-legality engine for the Trade Machine hub (2023 CBA).

Pure rules layer: no Streamlit, no network, no caches. The UI hands this module
plain data (teams, players, picks, cash) and gets back a verdict with
plain-English violations and per-team cap impact.

Every figure and rule implements docs/plan_trade_machine.md ("Verified 2023 CBA
rules spec" -- adversarially verified against the official 2023 CBA PDF).
Rule numbers in comments cite that spec. Where the spec's Section 9 lists an
unresolved edge, the conservative default it prescribes is implemented and
marked "S9.<n>". NO CBA figures live in this file: they load from
data/cba_config_<season>.json, and a missing key raises at load time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CBAConfig:
    season: str
    cap: float
    tax_line: float
    apron1: float
    apron2: float
    min_team_salary: float
    # Salary matching below the first apron (spec 1.1): incoming <= greater of
    # [lesser of (pct_high*out + cushion) and (out + flat_add)] and
    # [pct_low*out + cushion]. A FORMULA, not tiers.
    match_pct_high: float
    match_flat_add: float
    match_pct_low: float
    match_cushion: float
    room_buffer: float               # room-team absorption buffer over the cap (1.9)
    apron_match_pct: float           # at/above apron 1 post-trade: flat, NO cushion (1.2)
    cash_annual_limit: float         # 5.15% of cap (8.1)
    ntmle: float
    tmle: float
    bae: float
    two_way_salary: float
    snt_min_years_excl_options: int  # 6.2
    snt_max_years_incl_options: int
    recently_signed_months: int      # 7.1
    recently_signed_floor: str       # Dec 15 (ISO)
    jan15_floor: str                 # 7.2 (ISO)
    aggregation_wait_months: int     # 1.11
    draftee_wait_days: int           # 5.10
    pick_trade_horizon_drafts: int   # 5.4
    next_draft_year: int
    roster_max_offseason: int        # 8.4
    roster_max_inseason: int
    roster_min: int
    two_way_slots: int

    @classmethod
    def load(cls, season: str) -> "CBAConfig":
        p = ROOT / "data" / f"cba_config_{season.replace('-', '_')}.json"
        raw = {k: v for k, v in json.loads(p.read_text()).items()
               if not k.startswith("_")}
        return cls(**raw)  # missing/extra keys raise loudly by design

    def matching_limit(self, outgoing: float) -> float:
        """Spec 1.1: max incoming salary for OUTGOING under the expanded TPE
        (below-first-apron regime)."""
        expanded = min(self.match_pct_high * outgoing + self.match_cushion,
                       outgoing + self.match_flat_add)
        return max(expanded, self.match_pct_low * outgoing + self.match_cushion)


# ── Trade data model ──────────────────────────────────────────────────────────

@dataclass
class TradePlayer:
    name: str
    salary: float                    # $M current-year cap hit
    signed_date: str | None = None   # ISO, for eligibility timing
    signed_via: str = "standard"     # standard | bird_resign | sign_and_trade | min | draftee
    raise_pct: float | None = None   # for the Jan-15 rule on Bird re-signs (7.2)
    acquired_date: str | None = None # for the 2-month aggregation freeze (1.11)
    consent_required: bool = False   # implicit veto / NTC (7.3-7.4)
    two_way: bool = False
    via_tpe: bool = False            # incoming: absorb into a trade exception (2.x)


@dataclass
class TradeException:
    """An outstanding traded-player exception the receiving team holds."""
    amount: float                    # $M remaining on the exception
    label: str = ""                  # departed player it is named for
    prior_season: bool = True        # created before this league year; unknown
                                     # defaults conservative (hard-cap trigger)


@dataclass
class TradePick:
    year: int
    round: int                       # 1 | 2
    protection: str = ""             # non-empty = protected/conditional
    swap: bool = False               # swap right, not the pick itself (5.5)


@dataclass
class TradeTeam:
    abbr: str
    payroll: float                   # $M committed BEFORE the trade
    roster_size: int                 # standard contracts before the trade
    out_players: list[TradePlayer] = field(default_factory=list)
    in_players: list[TradePlayer] = field(default_factory=list)
    out_picks: list[TradePick] = field(default_factory=list)
    in_picks: list[TradePick] = field(default_factory=list)
    cash_out: float = 0.0
    cash_in: float = 0.0
    cash_sent_ytd: float = 0.0       # against the annual limit (8.1)
    owned_future_firsts: list[int] = field(default_factory=list)  # draft years
    hard_cap: float | None = None    # $M if already hard-capped this cap year
    tpes: list[TradeException] = field(default_factory=list)

    @property
    def salary_out(self) -> float:   # two-ways count $0 both directions (1.8)
        return sum(p.salary for p in self.out_players if not p.two_way)

    @property
    def salary_in(self) -> float:
        return sum(p.salary for p in self.in_players if not p.two_way)

    def matchable_in(self) -> float:
        """Incoming that needs matching: min-exception arrivals need none (1.7);
        players absorbed into a trade exception need none either (2.x)."""
        return sum(p.salary for p in self.in_players
                   if not p.two_way and p.signed_via != "min" and not p.via_tpe)

    def payroll_after(self) -> float:
        return self.payroll - self.salary_out + self.salary_in

    def counted_out(self) -> list[TradePlayer]:
        return [p for p in self.out_players if not p.two_way]


@dataclass
class Trade:
    teams: list[TradeTeam]
    trade_date: str                  # ISO


# ── Verdict model ─────────────────────────────────────────────────────────────

@dataclass
class Violation:
    code: str
    team: str
    message: str
    rule_cite: str


@dataclass
class CapNote:
    """Non-blocking consequence or caveat the UI should surface."""
    code: str
    team: str
    message: str


@dataclass
class TeamImpact:
    abbr: str
    payroll_before: float
    payroll_after: float
    room_to_tax: float
    room_to_apron1: float
    room_to_apron2: float
    roster_after: int
    hard_cap_triggered: float | None


@dataclass
class TradeVerdict:
    legal: bool
    violations: list[Violation]
    notes: list[CapNote]
    impact: list[TeamImpact]


def _iso(d: str) -> date:
    return date.fromisoformat(d)


def _add_months(d: date, months: int) -> date:
    y, m = d.year, d.month + months
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def _is_offseason(d: date) -> bool:
    # July through the (approximate) late-October season start
    return (d.month, d.day) >= (7, 1) and (d.month, d.day) < (10, 21)


# ── Rule checks ───────────────────────────────────────────────────────────────
# Each returns (violations, notes). validate() runs all of them so the UI can
# show EVERY broken rule, not just the first.

def check_balanced_legs(trade: Trade, cfg: CBAConfig):
    """Structural: every asset sent must arrive somewhere in this trade."""
    v, n = [], []
    out_names = sorted(p.name for t in trade.teams for p in t.out_players)
    in_names = sorted(p.name for t in trade.teams for p in t.in_players)
    if out_names != in_names:
        v.append(Violation("UNBALANCED_PLAYERS", "-",
                           "Every player sent out must be received by another team in this trade.",
                           "structural"))
    total_cash_out = sum(t.cash_out for t in trade.teams)
    total_cash_in = sum(t.cash_in for t in trade.teams)
    if round(total_cash_out - total_cash_in, 6) != 0:
        v.append(Violation("UNBALANCED_CASH", "-",
                           "Cash sent and cash received across the trade must balance.",
                           "structural"))
    out_picks = sorted((p.year, p.round, p.swap) for t in trade.teams for p in t.out_picks)
    in_picks = sorted((p.year, p.round, p.swap) for t in trade.teams for p in t.in_picks)
    if out_picks != in_picks:
        v.append(Violation("UNBALANCED_PICKS", "-",
                           "Every pick sent out must be received by another team in this trade.",
                           "structural"))
    return v, n


def check_salary_matching(trade: Trade, cfg: CBAConfig):
    """Spec 1.1/1.2/1.3/1.4/1.9: regime chosen by POST-trade payroll."""
    v, n = [], []
    for t in trade.teams:
        need = t.matchable_in()
        if need <= 0:
            continue
        after = t.payroll_after()
        out = t.salary_out
        if after <= cfg.cap + cfg.room_buffer:
            continue  # fits under the cap + room buffer: no matching needed (1.9)
        if after > cfg.apron1:
            # at/above apron 1 post-trade: flat pct, NO cushion (1.2/1.3)
            limit = cfg.apron_match_pct * out
            if need > limit + 1e-9:
                which = "second apron" if after > cfg.apron2 else "first apron"
                v.append(Violation(
                    "MATCH_APRON", t.abbr,
                    f"{t.abbr} would finish above the {which} and can take back at most "
                    f"100% of the salary it sends out (${limit:.1f}M); it is taking back "
                    f"${need:.1f}M.",
                    "CBA Art. VII 6(j)(1); 6(j)(3)"))
        else:
            # below apron 1: expanded formula, but a genuinely under-cap team can
            # also absorb into room -- allowance is the better of the two (1.1/1.9)
            room_allow = max(0.0, cfg.cap - t.payroll) + cfg.room_buffer + t.salary_out
            limit = max(cfg.matching_limit(out), room_allow)
            if need > limit + 1e-9:
                v.append(Violation(
                    "MATCH_BELOW_APRON", t.abbr,
                    f"{t.abbr} sends out ${out:.1f}M, which allows taking back at most "
                    f"${limit:.1f}M; it is taking back ${need:.1f}M.",
                    "CBA Art. VII 6(j)(1)(iv)"))
            elif need > cfg.apron_match_pct * out + 1e-9:
                # S9.4 conservative default: any take-back above 100% flat
                # hard-caps at the first apron; warn inside the $250K sliver.
                sliver = need <= cfg.apron_match_pct * out + cfg.match_cushion + 1e-9
                n.append(CapNote(
                    "HARDCAP_APRON1", t.abbr,
                    f"Taking back more salary than it sends hard-caps {t.abbr} at the "
                    f"first apron (${cfg.apron1:.1f}M) for the rest of the season."
                    + (" (Within the $250K cushion this is contested; treated "
                       "conservatively as a hard cap.)" if sliver else "")))
    return v, n


def _assign_tpes(t: TradeTeam, cfg: CBAConfig):
    """Deterministically assign via_tpe incoming players to the team's
    exceptions. Fit = exception amount + the $250K allowance, SHARED by every
    player absorbed into that exception. Players are placed largest-salary
    first into the exception that fits with, in order of preference, no
    prior-season hard-cap trigger, then the tightest remaining capacity.
    Mirrored line-for-line by the JS engine; keep both in sync."""
    flagged = sorted((p for p in t.in_players if p.via_tpe and not p.two_way),
                     key=lambda p: (-p.salary, p.name))
    cap = [tpe.amount + cfg.match_cushion for tpe in t.tpes]
    assign, unfit = [], []
    for p in flagged:
        best = -1
        for i, tpe in enumerate(t.tpes):
            if cap[i] + 1e-9 < p.salary:
                continue
            if best < 0 or ((tpe.prior_season, cap[i])
                            < (t.tpes[best].prior_season, cap[best])):
                best = i
        if best < 0:
            unfit.append(p)
        else:
            cap[best] -= p.salary
            assign.append((p, best))
    return assign, unfit


def check_tpe_usage(trade: Trade, cfg: CBAConfig):
    """Spec 2.x: an incoming player may be absorbed into an outstanding
    traded-player exception INSTEAD of salary matching. Using an exception
    created in a prior season hard-caps the team at the first apron, so a team
    finishing above it cannot use one at all (S9 conservative reading of
    Art. VII 2(e)(4) row F)."""
    v, n = [], []
    for t in trade.teams:
        flagged = [p for p in t.in_players if p.via_tpe and not p.two_way]
        if not flagged:
            continue
        if not t.tpes:
            v.append(Violation(
                "TPE_NONE", t.abbr,
                f"{t.abbr} has no outstanding trade exception to absorb "
                f"{flagged[0].name} into.",
                "CBA Art. VII 6(j)(2)"))
            continue
        assign, unfit = _assign_tpes(t, cfg)
        for p in unfit:
            biggest = max(tpe.amount for tpe in t.tpes)
            v.append(Violation(
                "TPE_FIT", t.abbr,
                f"{p.name} (${p.salary:.1f}M) does not fit in {t.abbr}'s remaining "
                f"trade exceptions (largest: ${biggest:.2f}M plus the "
                f"${cfg.match_cushion:.2f}M allowance).",
                "CBA Art. VII 6(j)(2)"))
        used: dict[int, list[TradePlayer]] = {}
        for p, i in assign:
            used.setdefault(i, []).append(p)
        for i in sorted(used):
            tpe, ps = t.tpes[i], used[i]
            names = " and ".join(p.name for p in ps)
            total = sum(p.salary for p in ps)
            n.append(CapNote(
                "TPE_ABSORB", t.abbr,
                f"{names} (${total:.1f}M) fits into {t.abbr}'s ${tpe.amount:.2f}M "
                + (f"{tpe.label} " if tpe.label else "")
                + "trade exception, so no salary matching is needed there."))
        if any(t.tpes[i].prior_season for i in used):
            if t.payroll_after() > cfg.apron1 + 1e-9:
                v.append(Violation(
                    "TPE_PRIOR_APRON1", t.abbr,
                    f"{t.abbr} cannot use a trade exception created last season "
                    f"while finishing above the first apron (${cfg.apron1:.1f}M); "
                    f"this trade puts it at ${t.payroll_after():.1f}M.",
                    "CBA Art. VII 2(e)(4) row F"))
            else:
                n.append(CapNote(
                    "HARDCAP_APRON1_TPE", t.abbr,
                    f"Using a trade exception created last season hard-caps "
                    f"{t.abbr} at the first apron (${cfg.apron1:.1f}M) for the "
                    f"rest of the season."))
    return v, n


def check_apron2_restrictions(trade: Trade, cfg: CBAConfig):
    """Spec 3.1/3.2: post-trade second-apron teams cannot aggregate or send cash."""
    v, n = [], []
    for t in trade.teams:
        after = t.payroll_after()
        if after > cfg.apron2:
            if len(t.counted_out()) >= 2 and t.matchable_in() > 0:
                v.append(Violation(
                    "APRON2_AGGREGATION", t.abbr,
                    f"{t.abbr} would finish above the second apron (${cfg.apron2:.1f}M) "
                    f"and cannot combine two or more outgoing salaries to take salary back.",
                    "CBA Art. VII 6(j)(1)(ii); 2(e)(4) row H"))
            if t.cash_out > 0:
                v.append(Violation(
                    "APRON2_CASH", t.abbr,
                    f"{t.abbr} would finish above the second apron and cannot send cash "
                    f"in a trade.",
                    "CBA Art. VII 2(e)(4) row I"))
        else:
            # below apron 2 post-trade: these actions are legal but hard-cap (4b)
            if len(t.counted_out()) >= 2 and t.matchable_in() > 0:
                n.append(CapNote(
                    "HARDCAP_APRON2_AGG", t.abbr,
                    f"Aggregating salaries hard-caps {t.abbr} at the second apron "
                    f"(${cfg.apron2:.1f}M) for the rest of the season."))
            if t.cash_out > 0:
                n.append(CapNote(
                    "HARDCAP_APRON2_CASH", t.abbr,
                    f"Sending cash hard-caps {t.abbr} at the second apron "
                    f"(${cfg.apron2:.1f}M) for the rest of the season."))
    return v, n


def check_hard_caps(trade: Trade, cfg: CBAConfig):
    """Spec 4c.1: an existing hard cap is an absolute ceiling."""
    v, n = [], []
    for t in trade.teams:
        if t.hard_cap is not None and t.payroll_after() > t.hard_cap + 1e-9:
            v.append(Violation(
                "HARD_CAP_EXCEEDED", t.abbr,
                f"{t.abbr} is hard-capped at ${t.hard_cap:.1f}M this season and this "
                f"trade would put it at ${t.payroll_after():.1f}M. A hard cap cannot "
                f"be exceeded for any reason.",
                "CBA Art. VII 2(e)(2)"))
    return v, n


def check_stepien(trade: Trade, cfg: CBAConfig):
    """Spec 5.1-5.5 + S9.15: no two consecutive future drafts without a first;
    swaps keep the year covered; protected incoming picks do not cover a year;
    7-draft horizon."""
    v, n = [], []
    horizon_last = cfg.next_draft_year + cfg.pick_trade_horizon_drafts - 1
    for t in trade.teams:
        for p in t.out_picks + t.in_picks:
            if p.year > horizon_last:
                v.append(Violation(
                    "PICK_HORIZON", t.abbr,
                    f"The {p.year} pick is beyond the seven-draft limit "
                    f"(latest tradable draft: {horizon_last}).",
                    "NBA By-Laws / CBA pick-trading horizon"))
        # The consecutive-gap test only applies to a team actually sending a
        # first out of this trade; without an ownership ledger an empty list
        # would falsely read as "owns no firsts anywhere".
        sends_first = any(p.round == 1 and not p.swap for p in t.out_picks)
        if not sends_first:
            continue
        if not t.owned_future_firsts:
            n.append(CapNote(
                "STEPIEN_UNKNOWN", t.abbr,
                f"{t.abbr}'s future first-round ownership is not loaded, so the "
                f"Stepien rule was not checked for this trade."))
            continue
        before = set(t.owned_future_firsts)
        covered = set(before)
        for p in t.out_picks:
            if p.round == 1 and not p.swap:
                covered.discard(p.year)          # swap = year stays covered (5.5)
        for p in t.in_picks:
            if p.round == 1 and not p.swap and not p.protection:
                covered.add(p.year)              # protected does not cover (S9.15)
        # The rule bars trades that CREATE a consecutive-gap pair. A team whose
        # ledger already shows a conservative gap (protection structures the
        # league approved) is not violated by unrelated pick trades -- only a
        # NEW consecutive pair introduced by this trade counts.
        years = range(cfg.next_draft_year, horizon_last + 1)

        def _pairs(cov):
            run = [y for y in years if y not in cov]
            return {(a, b) for a, b in zip(run, run[1:]) if b == a + 1}

        new_pairs = _pairs(covered) - _pairs(before)
        if new_pairs:
            a, b = sorted(new_pairs)[0]
            v.append(Violation(
                "STEPIEN", t.abbr,
                f"{t.abbr} could be left without a first-round pick in "
                f"{a} and {b}, two consecutive future drafts. Teams must keep a "
                f"first in at least every other future draft.",
                "NBA By-Laws sec. 7 (Stepien rule)"))
    return v, n


def check_sign_and_trade(trade: Trade, cfg: CBAConfig):
    """Spec 6.4/6.5: no S&T once the season starts; the acquiring team must
    finish below the first apron and is hard-capped there."""
    v, n = [], []
    td = _iso(trade.trade_date)
    for t in trade.teams:
        snt_in = [p for p in t.in_players if p.signed_via == "sign_and_trade"]
        if not snt_in:
            continue
        if not _is_offseason(td):
            v.append(Violation(
                "SNT_SEASON", t.abbr,
                "Sign-and-trades are only allowed before the regular season starts.",
                "CBA Art. VII 8(e)(1)(v)"))
        if t.payroll_after() > cfg.apron1:
            v.append(Violation(
                "SNT_APRON1", t.abbr,
                f"{t.abbr} cannot finish a sign-and-trade above the first apron "
                f"(${cfg.apron1:.1f}M); this trade puts it at ${t.payroll_after():.1f}M.",
                "CBA Art. VII 2(e)(4) row C"))
        else:
            n.append(CapNote(
                "HARDCAP_APRON1_SNT", t.abbr,
                f"Acquiring a sign-and-trade player hard-caps {t.abbr} at the first "
                f"apron (${cfg.apron1:.1f}M) for the rest of the season."))
    return v, n


def check_player_eligibility(trade: Trade, cfg: CBAConfig):
    """Spec 7.1/7.2/7.3, 5.10, 1.11: signing-date freezes, consent, aggregation wait."""
    v, n = [], []
    td = _iso(trade.trade_date)
    dec15 = _iso(cfg.recently_signed_floor)
    jan15 = _iso(cfg.jan15_floor)
    for t in trade.teams:
        for p in t.out_players:
            if p.signed_date:
                sd = _iso(p.signed_date)
                if p.two_way or p.signed_via == "draftee":
                    ok_from = sd.fromordinal(sd.toordinal() + cfg.draftee_wait_days)
                    cite = "CBA 30-day rule"
                elif (p.signed_via == "bird_resign"
                      and (p.raise_pct or 0) > 20.0):        # S9.10: strictly >20%
                    ok_from = max(_add_months(sd, cfg.recently_signed_months), jan15)
                    cite = "CBA Art. VII 8(d)(iii) (Jan 15 rule)"
                elif p.signed_via != "sign_and_trade":       # initial S&T leg exempt (6.8)
                    ok_from = max(_add_months(sd, cfg.recently_signed_months), dec15)
                    cite = "CBA Art. VII 8(d) (Dec 15 rule)"
                else:
                    ok_from = None
                    cite = ""
                if ok_from and td < ok_from:
                    v.append(Violation(
                        "RECENTLY_SIGNED", t.abbr,
                        f"{p.name} signed on {p.signed_date} and cannot be traded "
                        f"until {ok_from.isoformat()}.",
                        cite))
            if p.consent_required:
                n.append(CapNote(
                    "CONSENT", t.abbr,
                    f"{p.name} must consent to this trade (no-trade rights)."))
            if p.acquired_date and len(t.counted_out()) >= 2:
                wait_until = _add_months(_iso(p.acquired_date), cfg.aggregation_wait_months)
                if td < wait_until:
                    v.append(Violation(
                        "AGGREGATION_WAIT", t.abbr,
                        f"{p.name} was acquired on {p.acquired_date} and cannot be "
                        f"combined with other salaries until {wait_until.isoformat()}.",
                        "CBA Art. VII 6(j)(4)(i)"))
    return v, n


def check_cash_and_roster(trade: Trade, cfg: CBAConfig):
    """Spec 8.1/8.4/8.5: annual cash limit; post-trade roster bounds."""
    v, n = [], []
    td = _iso(trade.trade_date)
    max_roster = cfg.roster_max_offseason if _is_offseason(td) else cfg.roster_max_inseason
    for t in trade.teams:
        if t.cash_out > 0 and t.cash_sent_ytd + t.cash_out > cfg.cash_annual_limit + 1e-9:
            v.append(Violation(
                "CASH_LIMIT", t.abbr,
                f"{t.abbr} would exceed the ${cfg.cash_annual_limit:.3f}M annual "
                f"cash-in-trades limit "
                f"(${t.cash_sent_ytd:.1f}M already sent + ${t.cash_out:.1f}M here).",
                "CBA Art. VII 8(a)"))
        std_delta = (len([p for p in t.in_players if not p.two_way])
                     - len([p for p in t.out_players if not p.two_way]))
        roster_after = t.roster_size + std_delta
        if roster_after > max_roster:
            v.append(Violation(
                "ROSTER_MAX", t.abbr,
                f"{t.abbr} would have {roster_after} standard contracts; the limit is "
                f"{max_roster} {'in the offseason' if max_roster == cfg.roster_max_offseason else 'during the season'}. "
                f"Waive or include another player first.",
                "roster limits (15+3 in season, 21 offseason)"))
        if roster_after < cfg.roster_min:
            n.append(CapNote(
                "ROSTER_MIN", t.abbr,
                f"{t.abbr} drops to {roster_after} standard contracts; below "
                f"{cfg.roster_min} is only allowed for two weeks at a time."))
    return v, n


CHECKS = [
    check_balanced_legs,
    check_salary_matching,
    check_tpe_usage,
    check_apron2_restrictions,
    check_hard_caps,
    check_stepien,
    check_sign_and_trade,
    check_player_eligibility,
    check_cash_and_roster,
]


def byc_outgoing(prior_salary: float, new_salary: float) -> float:
    """Spec 6.6: the sending team's matching number for a BYC sign-and-trade."""
    return max(prior_salary, 0.5 * new_salary)


def validate(trade: Trade, cfg: CBAConfig) -> TradeVerdict:
    violations: list[Violation] = []
    notes: list[CapNote] = []
    for chk in CHECKS:
        cv, cn = chk(trade, cfg)
        violations.extend(cv)
        notes.extend(cn)
    hardcap_notes = {x.team: cfg.apron1 if "APRON1" in x.code else cfg.apron2
                     for x in notes if x.code.startswith("HARDCAP")}
    impact = [
        TeamImpact(
            abbr=t.abbr,
            payroll_before=round(t.payroll, 1),
            payroll_after=round(t.payroll_after(), 1),
            room_to_tax=round(cfg.tax_line - t.payroll_after(), 1),
            room_to_apron1=round(cfg.apron1 - t.payroll_after(), 1),
            room_to_apron2=round(cfg.apron2 - t.payroll_after(), 1),
            roster_after=t.roster_size
            + len([p for p in t.in_players if not p.two_way])
            - len([p for p in t.out_players if not p.two_way]),
            hard_cap_triggered=hardcap_notes.get(t.abbr),
        )
        for t in trade.teams
    ]
    return TradeVerdict(legal=not violations, violations=violations,
                        notes=notes, impact=impact)
