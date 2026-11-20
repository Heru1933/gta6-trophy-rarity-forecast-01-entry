"""Earn-rate forecaster for gta6-trophy-rarity-forecast-01 (Python 3, standard library only).

The job: for one case (one JSON file in the input directory), forecast every trophy's earn
rate -- the percent of Exophase-tracked players who have earned it -- at the case's target
time, as five quantiles (5%, 25%, 50%, 75%, 95%). Scoring is pinball loss on the logit
scale, so the whole program works on logit(percent) and only converts back at the end.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), no network, one
case per run, an 8-second cap. Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The model, per trophy:

  1. Live segment. The house's early recording re-read a frozen page for hours, so a case's
     history can open with a long run of identical reads (same players tracked, same rates)
     that say nothing about the game's real trajectory. That prefix is dropped; fitting
     starts at the first read after it. A history that never unfreezes falls back to a
     wide persistence forecast.
  2. Rates are counts in disguise: a rate r in a read tracking N players means about
     r*N/100 earners. Each read's rate is replaced by the smoothed (count + 0.5)/(N + 1)
     so a 0% or 100% print still has a finite logit and small samples pull toward sense.
  3. Three candidate forecasts of the target logit, combined by their median so no single
     family's bad day decides the answer:
       - a saturating Weibull curve A*(1 - exp(-k*u^b)) in hours since launch, fitted by
         a deterministic grid plus pattern-search refinement, minimising the recent-
         weighted squared error on the logit scale;
       - the same curve re-anchored at the live segment's first read (level r0 at time
         u0), which fits tails whose launch phase the recording missed;
       - a straight line in logit against ln(1 + hours/24), the classic log-time trend;
       - persistence (the last read's rate) as the anchor candidate.
  4. The spread is honest about what is not known: a random-walk term growing with the
     square root of the horizon, the disagreement between the candidate families, the
     winning fits' residual scale, and the binomial noise of the last read's earner
     count. It widens further when the live data is thin, flat, or ends in a stale run
     of identical reads.

Everything is a pure function of the input bytes: no randomness, no clock, no network,
so two runs of a case write the same file byte for byte.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The five levels every row reports, and the standard normal's quantiles at them.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
# The scorer clamps percents into this range before taking logits; so do we.
LO, HI = 0.01, 99.99
# Fitting never uses more than this many reads per trophy (evenly thinned, ends kept).
MAX_POINTS = 64


def clamp_rate(x):
    return LO if x < LO else HI if x > HI else x


def logit_pct(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; a whole value comes back as an int for compact JSON."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    """An ISO-8601 timestamp ('2026-11-19T03:00:00.309Z') as epoch seconds, or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def read_signature(read):
    """What makes two reads the same read: the tracked-player count and every rate."""
    return (read.get("players_tracked"), tuple(read.get("rates") or ()))


def split_live(history):
    """Split off the frozen prefix; return (live reads, whole-history-frozen flag).

    The frozen prefix is the maximal run of reads identical to the first, signature by
    signature. The first read after it is the first one that can carry new information,
    so fitting uses the reads from there on. If the run covers the whole history, the
    single last read is returned and the flag is set.
    """
    n = len(history)
    if n == 0:
        return [], True
    first = read_signature(history[0])
    j = 0
    while j + 1 < n and read_signature(history[j + 1]) == first:
        j += 1
    if j == n - 1:
        return history[-1:], True
    if j >= 1:
        rest = history[j + 1:]
        return (rest if len(rest) >= 2 else history[j:]), False
    return history, False


def stale_tail_hours(segment, t0):
    """Hours covered by the run of identical reads at the end of the live segment.

    A page that stops changing near the checkpoint means the last reads are stale; the
    forecaster cannot see how stale, so it answers with a wider spread instead.
    """
    if len(segment) < 2:
        return 0.0
    last = read_signature(segment[-1])
    j = len(segment) - 1
    while j - 1 >= 0 and read_signature(segment[j - 1]) == last:
        j -= 1
    t_first = parse_ts(segment[j].get("at"))
    t_last = parse_ts(segment[-1].get("at"))
    if t_first is None or t_last is None:
        return 0.0
    return max(0.0, (t_last - t_first) / 3600.0)


def smoothed_rate(rate, players):
    """A printed rate as a smoothed percent: (earners + 0.5) / (players + 1)."""
    if players and players > 0:
        earners = rate * players / 100.0
        return clamp_rate(100.0 * (earners + 0.5) / (players + 1.0))
    return clamp_rate(rate)


def build_points(segment, t0, index):
    """(hours since t0, players, printed rate) for one trophy over the live segment."""
    pts = []
    for read in segment:
        rates = read.get("rates") or ()
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        t = parse_ts(read.get("at"))
        if t is None:
            continue
        players = read.get("players_tracked") or 0
        pts.append((max(0.0, (t - t0) / 3600.0), float(players), float(rate)))
    if len(pts) > MAX_POINTS:
        last = len(pts) - 1
        keep = sorted({round(i * last / (MAX_POINTS - 1)) for i in range(MAX_POINTS)})
        pts = [pts[i] for i in keep]
    return pts


def weights_for(m):
    """Recent reads weigh more: the curve's recent shape says the most about its future."""
    if m <= 1:
        return [1.0]
    return [0.35 + 0.65 * (i / (m - 1)) for i in range(m)]


def weibull_sse(us, ys, ws, powers, r0, asymptote, k):
    """Weighted mean squared logit error of r0 + (A - r0)*(1 - exp(-k*u^b)) on the points."""
    total = 0.0
    wsum = 0.0
    span = asymptote - r0
    for i in range(len(us)):
        pred = r0 + span * (1.0 - math.exp(-k * powers[i]))
        diff = logit_pct(pred) - ys[i]
        total += ws[i] * diff * diff
        wsum += ws[i]
    return total / wsum if wsum else float("inf")


def weibull_predict(r0, asymptote, k, b, u0, u_target):
    du = max(0.0, u_target - u0)
    return r0 + (asymptote - r0) * (1.0 - math.exp(-k * du ** b))


def fit_weibull(us, ys, ws, r0, u0, rmax, u_target):
    """Fit the saturating curve; (target logit, weighted SSE, parameter spread).

    The grid walks the asymptote (anchored to the largest rate seen, plus fixed high
    values), the half-saturation time in hours, and the shape exponent; a pattern search
    then steps each parameter multiplicatively around the winner with shrinking steps.
    Over a short window several asymptotes fit almost equally well while predicting very
    different futures, so the family's forecast is not the single winner's: it is the
    weighted median of the target predictions of every near-best fit (weights falling
    with the fit's excess error), and their weighted spread comes back as the parameter
    uncertainty. Deterministic: fixed grids, fixed order, first strict improvement wins.
    """
    du = [max(0.0, u - u0) for u in us]
    asymptotes = set()
    for mult in (1.08, 1.25, 1.6, 2.2, 3.2, 5.0, 8.0):
        asymptotes.add(min(100.0, rmax * mult))
    for fixed in (30.0, 45.0, 60.0, 75.0, 88.0, 96.0, 100.0):
        asymptotes.add(fixed)
    asymptotes = sorted(a for a in asymptotes if a > rmax * 1.02 and a > r0 + 0.05)
    if not asymptotes:
        asymptotes = [min(100.0, rmax * 1.2 + 0.5)]
    shapes = (0.5, 0.7, 1.0, 1.4, 1.9)
    half_times = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    ln2 = math.log(2.0)

    scored = []  # (sse, target logit) for every grid fit
    best = None  # (sse, asymptote, half_time, shape)
    for b in shapes:
        powers = [d ** b for d in du]
        for half in half_times:
            k = ln2 / (half ** b)
            for a in asymptotes:
                sse = weibull_sse(us, ys, ws, powers, r0, a, k)
                scored.append((sse, logit_pct(weibull_predict(r0, a, k, b, u0, u_target))))
                if best is None or sse < best[0]:
                    best = (sse, a, half, b)

    sse, a, half, b = best
    step_a, step_h, step_b = 0.18, 0.35, 0.22
    for _ in range(4):
        improved = False
        for cand in (
            (a * (1.0 + step_a), half, b),
            (a * (1.0 - step_a), half, b),
            (a, half * (1.0 + step_h), b),
            (a, half * (1.0 - step_h), b),
            (a, half, b + step_b),
            (a, half, max(0.2, b - step_b)),
        ):
            ca, ch, cb = cand
            if ca <= rmax * 1.01 or ca > 100.0 or ca <= r0 + 0.02:
                continue
            powers = [d ** cb for d in du]
            csse = weibull_sse(us, ys, ws, powers, r0, ca, ln2 / (ch ** cb))
            if csse < sse:
                sse, a, half, b = csse, ca, ch, cb
                improved = True
        if not improved:
            step_a *= 0.5
            step_h *= 0.5
            step_b *= 0.5
    k = ln2 / (half ** b)
    scored.append((sse, logit_pct(weibull_predict(r0, a, k, b, u0, u_target))))

    best_sse = min(s for s, _ in scored)
    near = [(s, lg) for s, lg in scored if s <= best_sse * 1.3 + 0.003]
    pairs = sorted((lg, 1.0 / (s - best_sse + 0.004)) for s, lg in near)
    total_w = sum(w for _, w in pairs)
    cumulative = 0.0
    centre = pairs[-1][0]
    for lg, w in pairs:
        cumulative += w
        if cumulative >= total_w / 2.0:
            centre = lg
            break
    spread = math.sqrt(sum(w * (lg - centre) ** 2 for lg, w in pairs) / total_w)
    return centre, best_sse, spread


def fit_loglin(us, ys, ws, u_target):
    """Weighted least squares of logit on ln(1 + hours/24); (target logit, weighted SSE)."""
    xs = [math.log1p(u / 24.0) for u in us]
    sw = sum(ws)
    sx = sum(w * x for w, x in zip(ws, xs))
    sy = sum(w * y for w, y in zip(ws, ys))
    sxx = sum(w * x * x for w, x in zip(ws, xs))
    sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope, intercept = 0.0, sy / sw
    else:
        slope = (sw * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / sw
    sse = 0.0
    for w, x, y in zip(ws, xs, ys):
        diff = intercept + slope * x - y
        sse += w * diff * diff
    pred = intercept + slope * math.log1p(u_target / 24.0)
    lo_lg, hi_lg = logit_pct(LO), logit_pct(HI)
    return min(hi_lg, max(lo_lg, pred)), sse / sw


def median(values):
    ordered = sorted(values)
    m = len(ordered)
    if m == 0:
        return 0.0
    if m % 2:
        return ordered[m // 2]
    return 0.5 * (ordered[m // 2 - 1] + ordered[m // 2])


def fallback_row(last_logit, sigma):
    return [round4(inv_logit_pct(last_logit + z * sigma)) for z in Z]


def forecast_trophy(case, segment, frozen, stale_hours, t0, u_target, horizon_days, index):
    """One trophy's five quantiles, as rounded output values."""
    # The last printed rate anywhere in the history, and its read's player count, anchor
    # both the fallback and the count-noise term even when the live segment is empty.
    last_rate, last_players = None, 0
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or ()
        if index < len(rates) and rates[index] is not None:
            last_rate = float(rates[index])
            last_players = read.get("players_tracked") or 0
            break
    if last_rate is None:
        return [0.01, 0.03, 0.08, 0.2, 0.8]
    last_logit = logit_pct(smoothed_rate(last_rate, last_players))
    earners = last_rate * last_players / 100.0 if last_players else 0.0
    sigma_count = min(0.8, math.sqrt(1.0 / max(earners, 0.5))) if last_players else 0.35

    pts = build_points(segment, t0, index)
    distinct = len({p[2] for p in pts})
    if frozen or len(pts) < 3 or distinct < 2:
        sigma = math.sqrt((0.55 * math.sqrt(horizon_days)) ** 2 + 0.3 ** 2 + sigma_count ** 2)
        return fallback_row(last_logit, min(2.2, max(0.3, sigma)))

    us = [p[0] for p in pts]
    ys = [logit_pct(smoothed_rate(p[2], p[1])) for p in pts]
    ws = weights_for(len(pts))
    rmax = max(smoothed_rate(p[2], p[1]) for p in pts)

    candidates = []  # (target logit, weighted sse or None)
    lg, sse, param_sd = fit_weibull(us, ys, ws, 0.0, 0.0, rmax, u_target)
    candidates.append((lg, sse))
    if us[0] > 1.0:
        r0 = smoothed_rate(pts[0][2], pts[0][1])
        lg, sse, anchored_sd = fit_weibull(us, ys, ws, r0, us[0], rmax, u_target)
        candidates.append((lg, sse))
        param_sd = max(param_sd, anchored_sd)
    lg, sse = fit_loglin(us, ys, ws, u_target)
    candidates.append((lg, sse))
    # Persistence: the last live read's logit, with its own scatter around the points.
    persist = ys[-1]
    psse = sum(w * (y - persist) ** 2 for w, y in zip(ws, ys)) / sum(ws)
    candidates.append((persist, psse))

    logits = [c[0] for c in candidates]
    mu = median(logits)
    param_spread = (max(logits) - min(logits)) / 2.0
    fit_sse = min(c[1] for c in candidates[:3])
    resid = min(1.0, max(0.04, math.sqrt(fit_sse)))
    sigma = math.sqrt(
        (0.32 * math.sqrt(horizon_days)) ** 2
        + param_spread ** 2
        + (0.6 * param_sd) ** 2
        + (0.6 * resid) ** 2
        + sigma_count ** 2
    )
    if stale_hours > 2.0:
        sigma *= 1.0 + min(0.8, stale_hours / 48.0)
    if distinct <= 2 and horizon_days >= 1.0:
        sigma = max(sigma, 0.45)
    sigma = min(2.2, max(0.22, sigma))

    row = []
    prev = 0.0
    for z in Z:
        value = round4(inv_logit_pct(mu + z * sigma))
        value = max(prev, min(100.0, float(value)))
        row.append(value)
        prev = value
    return row


def forecast_case(case):
    history = case.get("history") or []
    segment, frozen = split_live(history)

    release = (case.get("game") or {}).get("release_date")
    t0 = parse_ts(f"{release}T00:00:00Z") if release else None
    if t0 is None and history:
        t0 = parse_ts(history[0].get("at"))
    if t0 is None:
        t0 = 0.0
    target_t = parse_ts(case.get("target"))
    at_t = parse_ts(case.get("at"))
    if target_t is not None:
        u_target = max(0.0, (target_t - t0) / 3600.0)
    else:
        u_target = 0.0
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes:
        horizon_days = float(horizon_minutes) / 1440.0
    elif target_t is not None and at_t is not None:
        horizon_days = max(0.0, (target_t - at_t) / 86400.0)
    else:
        horizon_days = 1.0
    stale = stale_tail_hours(segment, t0) if not frozen else 0.0

    rows = []
    for index, trophy in enumerate(case.get("trophies") or []):
        try:
            q = forecast_trophy(case, segment, frozen, stale, t0, u_target, horizon_days, index)
        except Exception:
            q = [0.01, 0.03, 0.08, 0.2, 0.8]
        rows.append({"id": trophy.get("id"), "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
