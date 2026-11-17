[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Orochim25/gta6-twitch-ccu-forecast-01-entry-acct2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Orochim25/gta6-twitch-ccu-forecast-01-entry-acct2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers, hour by hour

Forecasts Twitch concurrent viewers in the Grand Theft Auto VI category
for hours 2–24 of release day (2026-11-19 UTC), five quantiles per hour.

## Method

Log-space ratio forecast from past launches (`main.py`, Python standard
library only, deterministic, no network):

- Each past launch in the kit donates its day curve as log-ratios to
  its hour-1 print, `ln(viewers_h / viewers_1)`.
- Donors are weighted by a similarity kernel: launch tier, platform
  class (console-only vs multi-platform), log hour-1 size, and log
  pre-release buzz relative to the hour-1 print. Only the 16 most
  similar donors are kept.
- Per hour, the centre is the kernel-weighted mean log-ratio and the
  spread the kernel-weighted standard deviation, scaled by regime.
  For console-only launches the spread is remapped (the donor spread
  overstates the forecast error late in the day); the regime is read
  from the case's own notes, never from a hard-coded game name.
- The Grand Theft Auto VI category's own pre-release history is never
  used as a donor (it is not a launch day).

## Validation

Leave-one-out cross-validation over the 37 past launches shipped in
the kit (never tuned to the four practice cases alone):

| split | mean pinball loss |
|---|---|
| all 37 launches | 0.381 |
| console-only launches | 0.155 |
| practice-like (console-only, comparable tier) | 0.143 |
| mega console launches (peak >= 90k) | 0.139 |

Practice-like typical miss 0.232 in log terms, 90% coverage 0.944.
Ideas tried and rejected by the same validation: per-hour bias
correction, median/trimmed/winsorised centres, shape/level splits,
channels blending, buzz-anchored levels, recency weighting, donor
smoothing, and persistence blending — none improved the practice-like
split without hurting the all-launches split.
