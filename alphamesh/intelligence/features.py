"""Deterministic feature extraction from 1-minute bars.

Pure functions over an immutable bar window. No I/O, no randomness, no clock
reads: the same bars always produce the same features, which is what makes the
audit journal reconstructable after the fact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from alphamesh.models.domain import Bar, MarketSnapshot

MINUTES_PER_TRADING_YEAR = 252 * 390
OPENING_RANGE_MINUTES = 30


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def pct_return(bars: Sequence[Bar], minutes: int) -> float:
    """Percentage return over the trailing ``minutes`` bars."""
    if len(bars) < minutes + 1:
        return 0.0
    start = bars[-(minutes + 1)].close
    end = bars[-1].close
    return _safe_div(end - start, start)


def rolling_vwap(bars: Sequence[Bar]) -> float:
    """Volume-weighted average price across the window.

    Uses each bar's own VWAP when Alpaca supplied one, else the typical price.
    """
    total_value = 0.0
    total_volume = 0.0
    for bar in bars:
        price = bar.vwap if bar.vwap is not None else (bar.high + bar.low + bar.close) / 3.0
        total_value += price * bar.volume
        total_volume += bar.volume
    if total_volume <= 0:
        return bars[-1].close if bars else 0.0
    return total_value / total_volume


def vwap_deviation(bars: Sequence[Bar]) -> float:
    """Signed distance of the last close from window VWAP, as a fraction."""
    if not bars:
        return 0.0
    vwap = rolling_vwap(bars)
    return _safe_div(bars[-1].close - vwap, vwap)


def average_true_range(bars: Sequence[Bar], period: int = 14) -> float:
    """Wilder-style ATR over the trailing ``period`` bars, in price units."""
    if len(bars) < 2:
        return 0.0
    trs: list[float] = []
    for prev, cur in pairwise(bars):
        trs.append(
            max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
        )
    window = trs[-period:] if len(trs) >= period else trs
    return sum(window) / len(window) if window else 0.0


def realized_volatility(bars: Sequence[Bar], period: int = 30) -> float:
    """Annualised realised volatility from log returns of 1-minute closes."""
    closes = [b.close for b in bars[-(period + 1) :]]
    if len(closes) < 3:
        return 0.0
    rets = [
        math.log(b / a) for a, b in pairwise(closes) if a > 0 and b > 0
    ]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(variance) * math.sqrt(MINUTES_PER_TRADING_YEAR)


def volume_acceleration(bars: Sequence[Bar], fast: int = 5, slow: int = 30) -> float:
    """Ratio of recent average volume to baseline average volume.

    1.0 means volume is running at its baseline; 2.0 means twice baseline.
    """
    if len(bars) < slow + fast:
        return 1.0
    recent = [b.volume for b in bars[-fast:]]
    baseline = [b.volume for b in bars[-(slow + fast) : -fast]]
    recent_avg = sum(recent) / len(recent)
    baseline_avg = sum(baseline) / len(baseline)
    if baseline_avg <= 0:
        return 1.0
    return recent_avg / baseline_avg


def opening_range_position(bars: Sequence[Bar], minutes: int = OPENING_RANGE_MINUTES) -> float:
    """Where the last close sits inside the opening range, scaled to [-1, 1].

    -1 is the bottom of the range, +1 the top; values beyond that mean the
    price has broken out of the range.
    """
    if len(bars) < minutes:
        return 0.0
    opening = bars[:minutes]
    hi = max(b.high for b in opening)
    lo = min(b.low for b in opening)
    span = hi - lo
    if span <= 0:
        return 0.0
    mid = (hi + lo) / 2.0
    return _safe_div(bars[-1].close - mid, span / 2.0)


def trend_strength(bars: Sequence[Bar], period: int = 30) -> float:
    """Signed R-weighted slope of a least-squares fit through recent closes.

    The result is a unitless value in roughly [-1, 1]: the sign carries the
    direction, the magnitude combines slope steepness with fit quality, so a
    noisy drift scores lower than a clean move of the same size.
    """
    closes = [b.close for b in bars[-period:]]
    n = len(closes)
    if n < 5:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(closes) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in closes)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, closes, strict=True))
    if sxx <= 0 or syy <= 0:
        return 0.0
    slope = sxy / sxx
    r_squared = (sxy * sxy) / (sxx * syy)
    # Normalise slope by mean price so the value is comparable across symbols.
    normalised = _safe_div(slope * n, mean_y) * 100.0
    return max(-1.0, min(1.0, normalised)) * r_squared


def distance_from_extreme(bars: Sequence[Bar], period: int = 60) -> tuple[float, float]:
    """Fractional distance of the last close below the window high and above
    the window low. Both values are non-negative."""
    window = bars[-period:]
    if not window:
        return 0.0, 0.0
    hi = max(b.high for b in window)
    lo = min(b.low for b in window)
    last = window[-1].close
    return abs(_safe_div(hi - last, hi)), abs(_safe_div(last - lo, lo))


def compute_features(snapshot: MarketSnapshot) -> dict[str, float]:
    """Full deterministic feature vector for one market snapshot."""
    bars = snapshot.bars
    if not bars:
        return {}

    atr = average_true_range(bars)
    hi_dist, lo_dist = distance_from_extreme(bars)
    last = bars[-1].close

    return {
        "ret_1m": pct_return(bars, 1),
        "ret_5m": pct_return(bars, 5),
        "ret_15m": pct_return(bars, 15),
        "vwap_deviation": vwap_deviation(bars),
        "atr": atr,
        "atr_pct": _safe_div(atr, last),
        "realized_vol": realized_volatility(bars),
        "volume_acceleration": volume_acceleration(bars),
        "opening_range_position": opening_range_position(bars),
        "trend_strength": trend_strength(bars),
        "distance_from_high": hi_dist,
        "distance_from_low": lo_dist,
        "bar_count": float(len(bars)),
        "last_price": last,
    }


__all__ = [
    "average_true_range",
    "compute_features",
    "distance_from_extreme",
    "opening_range_position",
    "pct_return",
    "realized_volatility",
    "rolling_vwap",
    "trend_strength",
    "volume_acceleration",
    "vwap_deviation",
]
