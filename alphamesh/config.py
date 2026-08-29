"""Configuration loading for AlphaMesh.

Runtime settings come from the environment (``.env`` in development). Trading
policy comes from the YAML files in ``config/``. The two are kept separate on
purpose: the Risk Governor reads only ``risk.yaml``, and nothing in the AI path
is given a handle to it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


class Settings(BaseModel):
    """Runtime settings. Secrets are never rendered by ``__repr__``."""

    model_config = ConfigDict(frozen=True)

    paper: bool = True
    api_key_id: str = ""
    api_secret_key: str = Field(default="", repr=False)
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    anthropic_api_key: str = Field(default="", repr=False)
    llm_model: str = "claude-sonnet-5"
    database_path: Path = PROJECT_ROOT / "data" / "alphamesh.db"
    loop_seconds: int = 60
    dry_run: bool = True
    log_level: str = "INFO"
    data_source: str = "rest"
    capture_dir: Path = PROJECT_ROOT / "data" / "mcp_capture"
    alpaca_cli_path: str = "alpaca"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key_id and self.api_secret_key)

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)

    def redacted(self) -> dict[str, Any]:
        """Safe-to-log view. Secrets become fixed markers, never partial values."""
        return {
            "paper": self.paper,
            "api_key_id": "<set>" if self.api_key_id else "<unset>",
            "api_secret_key": "<set>" if self.api_secret_key else "<unset>",
            "anthropic_api_key": "<set>" if self.anthropic_api_key else "<unset>",
            "base_url": self.base_url,
            "data_url": self.data_url,
            "database_path": str(self.database_path),
            "data_source": self.data_source,
            "dry_run": self.dry_run,
            "loop_seconds": self.loop_seconds,
        }


def load_settings() -> Settings:
    """Build settings from the process environment."""
    db = _env("DATABASE_PATH") or str(PROJECT_ROOT / "data" / "alphamesh.db")
    cap = _env("ALPHAMESH_CAPTURE_DIR") or str(PROJECT_ROOT / "data" / "mcp_capture")
    return Settings(
        paper=_env_bool("ALPACA_PAPER", True),
        api_key_id=_env("APCA_API_KEY_ID"),
        api_secret_key=_env("APCA_API_SECRET_KEY"),
        base_url=_env("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets",
        data_url=_env("APCA_API_DATA_URL") or "https://data.alpaca.markets",
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        llm_model=_env("ALPHAMESH_LLM_MODEL") or "claude-sonnet-5",
        database_path=Path(db),
        loop_seconds=_env_int("ALPHAMESH_LOOP_SECONDS", 60),
        dry_run=_env_bool("ALPHAMESH_DRY_RUN", True),
        log_level=_env("ALPHAMESH_LOG_LEVEL") or "INFO",
        data_source=(_env("ALPHAMESH_DATA_SOURCE") or "rest").lower(),
        capture_dir=Path(cap),
        alpaca_cli_path=_env("ALPACA_CLI_PATH") or "alpaca",
    )


# --------------------------------------------------------------------------- #
# Policy files
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


class RiskLimits(BaseModel):
    """Immutable hard limits. Constructed once; never mutated at runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_defined_loss_per_trade: float
    high_confidence_max_defined_loss: float
    high_confidence_threshold: float
    absolute_max_defined_loss: float
    max_open_positions: int
    max_portfolio_defined_risk: float
    daily_loss_circuit_breaker: float
    correlation_groups: dict[str, list[str]]
    max_positions_per_correlation_group: int
    max_defined_risk_per_correlation_group: float
    max_quote_age_seconds: int
    max_relative_bid_ask_spread: float
    max_absolute_bid_ask_spread: float
    min_option_bid: float
    min_contract_day_volume: int
    min_top_of_book_size: int
    min_buying_power_multiple: float
    allowed_strategies: list[str]

    def cap_cents_for_confidence(self, confidence: float) -> int:
        """Per-trade defined-loss cap in cents, given judge confidence.

        The elevated high-confidence cap is still bounded by the absolute
        ceiling, so no confidence value can unlock unlimited risk.
        """
        base = self.max_defined_loss_per_trade
        if confidence >= self.high_confidence_threshold:
            base = max(base, self.high_confidence_max_defined_loss)
        capped = min(base, self.absolute_max_defined_loss)
        return round(capped * 100)

    def group_for(self, symbol: str) -> str | None:
        for group, members in self.correlation_groups.items():
            if symbol.upper() in {m.upper() for m in members}:
                return group
        return None


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quant_score_threshold: float
    min_judge_confidence: float
    min_dte: int
    max_dte: int
    bull_call_spread: dict[str, Any]
    bear_put_spread: dict[str, Any]
    max_debit_to_width_ratio: float
    limit_price_aggressiveness: float
    exits: dict[str, Any]


class UniverseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbols: list[str]
    min_bars_required: int
    bar_lookback_minutes: int


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    settings: Settings
    risk: RiskLimits
    strategies: StrategyConfig
    universe: UniverseConfig


def load_config(config_dir: Path | None = None, settings: Settings | None = None) -> AppConfig:
    directory = config_dir or CONFIG_DIR
    return AppConfig(
        settings=settings or load_settings(),
        risk=RiskLimits(**_load_yaml(directory / "risk.yaml")),
        strategies=StrategyConfig(**_load_yaml(directory / "strategies.yaml")),
        universe=UniverseConfig(**_load_yaml(directory / "universe.yaml")),
    )


@lru_cache(maxsize=1)
def default_config() -> AppConfig:
    return load_config()


__all__ = [
    "CONFIG_DIR",
    "PROJECT_ROOT",
    "AppConfig",
    "RiskLimits",
    "Settings",
    "StrategyConfig",
    "UniverseConfig",
    "default_config",
    "load_config",
    "load_settings",
]
