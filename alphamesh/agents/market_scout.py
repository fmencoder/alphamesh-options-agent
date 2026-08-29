"""Market scout: turns raw market data into scored opportunities.

The scout is the only component that talks to the market-data provider during
the discovery phase. It is deliberately cheap: no LLM call happens here, so the
agent can run every cycle without incurring model cost.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from alphamesh.alpaca.market_data import MarketDataProvider
from alphamesh.config import AppConfig
from alphamesh.intelligence.scoring import build_quant_signal
from alphamesh.models.domain import MarketSnapshot, QuantSignal

log = logging.getLogger(__name__)


class MarketScout:
    def __init__(self, config: AppConfig, data: MarketDataProvider) -> None:
        self.config = config
        self.data = data

    def symbols(self) -> Sequence[str]:
        return self.config.universe.symbols

    def scan(self) -> list[tuple[MarketSnapshot, QuantSignal]]:
        """Score every symbol in the universe, highest opportunity first."""
        results: list[tuple[MarketSnapshot, QuantSignal]] = []
        for symbol in self.symbols():
            try:
                snapshot = self.data.snapshot(
                    symbol, lookback_minutes=self.config.universe.bar_lookback_minutes
                )
            except Exception:
                log.exception("market data unavailable for %s; skipping", symbol)
                continue
            signal = build_quant_signal(
                snapshot, self.config.strategies, self.config.universe
            )
            results.append((snapshot, signal))
        results.sort(key=lambda pair: pair[1].quant_score, reverse=True)
        return results


__all__ = ["MarketScout"]
