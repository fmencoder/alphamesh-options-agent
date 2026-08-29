"""Regime agent: thin, testable wrapper around the regime classifier."""

from __future__ import annotations

from alphamesh.config import AppConfig
from alphamesh.intelligence.regime import classify
from alphamesh.models.domain import MarketSnapshot, QuantSignal, RegimeAssessment


class RegimeAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def assess(self, snapshot: MarketSnapshot, signal: QuantSignal) -> RegimeAssessment:
        return classify(snapshot, signal.features, self.config.universe)


__all__ = ["RegimeAgent"]
