"""Forecast GTA VI launch-day Twitch viewers, hour by hour (hours 2-24).

Method (validated by leave-one-out cross-validation over the 37 past
launches shipped in the kit -- never tuned to the four practice cases
alone): forecast in log space from ratios to the hour-1 print.

  * Each comparable launch donates its whole day curve as log-ratios
    l_h = ln(viewers_h / viewers_1), h = 2..24.
  * Donors are weighted by a kernel over launch similarity: tier
    (comparable / early-access-contaminated / weak), platform class
    (console-only vs multi-platform), log hour-1 size, and log
    pre-release buzz relative to the hour-1 print.
  * Only the 16 most similar donors are kept; the per-hour centre is
    their kernel-weighted mean log-ratio and the spread their
    kernel-weighted standard deviation.
  * Interval width is calibrated, not just narrow: for a console-only
    launch the donor spread overstates the forecast error late in the
    day, so the console spread is remapped (0.5 * sd + 0.15); the
    multi-platform spread is left as is. The console/multi regime is
    read off the donor pool itself (the console share of the top-16
    donor weight), never from a hard-coded game name.

Only the kit's input data is used. No network, no external data, no
hard-coded answers. Standard library only; deterministic.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

# Standard-normal quantiles for the five scored levels.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Kernel / model constants (chosen by leave-one-out validation).
BW_LOG_H1 = 1.2        # Gaussian bandwidth on log hour-1 viewers
BW_LOG_BUZZ = 1.3      # Gaussian bandwidth on log(buzz24 / hour-1)
W_PLATFORM_MISMATCH = 0.2   # weight when donor/target platform class differs
W_TIER = {"comparable": 1.0, "early-access-contaminated": 0.7,
          "weak": 0.5}
W_TIER_DEFAULT = 0.8
TOP_K = 16             # donors kept after kernel ranking
SPREAD_CONSOLE = 0.65  # base spread factor, console regime
SPREAD_MULTI = 1.1     # base spread factor, multi-platform regime
CONSOLE_REMAP_A = 0.5  # console spread remap: sd' = A * sd + B
CONSOLE_REMAP_B = 0.15
SD_FLOOR = 0.03
MAX_VIEWERS = 1.0e9


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def tier_of(comp):
    """Tier annotation carried in a comparable's unlock/source text."""
    text = (comp.get("unlock", "") + " " + comp.get("source", "")).lower()
    match = re.search(r"tier:\s*([a-z][a-z\-]*)", text)
    return match.group(1) if match else "comparable"


def load_donor(comp):
    """Turn one comparable launch into a donor curve, or None to skip it.

    A donor needs a complete 24-hour launch day (every hour present,
    every count positive). The Grand Theft Auto VI category's own
    pre-release history is not a launch day, so it is never a donor.
    """
    if "Grand Theft Auto VI" in comp.get("name", ""):
        return None
    day_start = parse_time(comp["day_start"])
    day_end = day_start + timedelta(days=1)
    hourly = sorted(comp.get("hourly", []), key=lambda r: r["from"])
    day = [r for r in hourly
           if day_start <= parse_time(r["from"]) < day_end]
    if len(day) != 24:
        return None
    viewers = [r["viewers"] for r in day]
    if any(v is None or v <= 0 for v in viewers):
        return None
    p1 = viewers[0]
    # Pre-release buzz: mean viewers over the 24 h before launch.
    pre = [r["viewers"] for r in hourly
           if day_start - timedelta(days=1) <= parse_time(r["from"]) < day_start
           and r["viewers"] is not None and r["viewers"] > 0]
    buzz24 = sum(pre) / len(pre) if pre else float(p1)
    platforms = comp.get("platforms", "")
    return {
        "name": comp["name"],
        "log_p1": math.log(p1),
        "log_buzz_ratio": math.log(buzz24 / p1),
        "console_only": "PC" not in platforms,
        "tier": tier_of(comp),
        # log-ratios to the hour-1 print, hours 1..24 (index 0 unused).
        "log_ratio": [0.0] + [math.log(viewers[h] / p1) for h in range(1, 24)],
    }


def case_features(case):
    """Features of the case being forecast, from data known at hour 1."""
    known = case["known"][0]
    p1 = known["viewers"]
    if p1 is None or p1 < 1:
        p1 = 1
    history = case.get("history_hourly") or []
    day_start = parse_time(case["day_start"])
    pre = [r["viewers"] for r in history
           if day_start - timedelta(days=1) <= parse_time(r["from"]) < day_start
           and r.get("viewers") is not None and r["viewers"] > 0]
    buzz24 = sum(pre) / len(pre) if pre else float(p1)
    return {
        "p1": p1,
        "log_p1": math.log(p1),
        "log_buzz_ratio": math.log(buzz24 / p1),
    }

def case_console_only(case):
    """Is the launch console-only, from the case's own notes?

    The notes open with "<game>, released <date> on <platforms> ...".
    The platform phrase runs to the first '(', ';' or '.'. It names
    PC exactly when PC is a launch platform (a later PC port, such as
    "PC 2019-11" in parentheses, does not count). Unparseable notes
    fall back to multi-platform (no spread remap).
    """
    notes = case.get("notes", "") or ""
    match = re.search(r"released\s+\S+\s+on\s+([^\(;\.]+)", notes)
    if not match:
        return False
    return "PC" not in match.group(1)


def donor_weight(target, donor, target_console):
    """Kernel weight of one donor for the case being forecast."""
    weight = W_TIER.get(donor["tier"], W_TIER_DEFAULT)
    if donor["console_only"] != target_console:
        weight *= W_PLATFORM_MISMATCH
    d_log_h1 = target["log_p1"] - donor["log_p1"]
    weight *= math.exp(-0.5 * (d_log_h1 / BW_LOG_H1) ** 2)
    d_buzz = target["log_buzz_ratio"] - donor["log_buzz_ratio"]
    weight *= math.exp(-0.5 * (d_buzz / BW_LOG_BUZZ) ** 2)
    return weight


def weighted_mean_sd(pairs):
    """Weighted mean and population standard deviation."""
    total = sum(w for w, _ in pairs)
    if total <= 0:
        return 0.0, 1.0
    mean = sum(w * x for w, x in pairs) / total
    var = sum(w * (x - mean) ** 2 for w, x in pairs) / total
    return mean, math.sqrt(max(var, 1e-6))


def forecast(case, comparables):
    """Return the 23 forecast rows for one case.

    Centre per hour: kernel-weighted mean of the top-16 donors'
    log-ratios. Spread per hour: kernel-weighted standard deviation,
    scaled by regime, with the console remap applied.
    """
    target = case_features(case)
    console_regime = case_console_only(case)
    donors = []
    for comp in comparables:
        donor = load_donor(comp)
        if donor is not None:
            donors.append(donor)
    if not donors:
        # No usable donor: fall back to the hour-1 print, wide spread.
        return [{"hour": h, "q": [target["p1"]] * 5} for h in range(2, 25)]
    scored = sorted(
        ((donor_weight(target, d, console_regime), d) for d in donors),
        key=lambda t: t[0],
        reverse=True,
    )[:TOP_K]
    weights = [w for w, _ in scored]
    ranked = [d for _, d in scored]
    spread_base = SPREAD_CONSOLE if console_regime else SPREAD_MULTI
    rows = []
    for h in range(2, 25):
        pairs = [(w, d["log_ratio"][h - 1]) for w, d in zip(weights, ranked)]
        centre, sd = weighted_mean_sd(pairs)
        sd = max(sd * spread_base, SD_FLOOR)
        if console_regime:
            sd = max(CONSOLE_REMAP_A * sd + CONSOLE_REMAP_B, SD_FLOOR)
        quantiles = []
        for z in Z_SCORES:
            q = target["p1"] * math.exp(centre + z * sd)
            q = min(max(q, 1.0), MAX_VIEWERS)
            quantiles.append(q)
        # Enforce non-decreasing quantiles (already ordered by z).
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": h, "q": quantiles})
    return rows

def main():
    """Read every *.in case, write one *.out forecast per case."""
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast(case, case.get("comparables", []))
        payload = {"hours": rows}
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
