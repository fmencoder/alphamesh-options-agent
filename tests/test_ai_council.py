"""AI reasoning council. No test here makes a paid LLM call."""

from __future__ import annotations

import httpx
import pytest

from alphamesh.agents.bear_agent import BearAgent, heuristic_bear_argument
from alphamesh.agents.bull_agent import BullAgent, heuristic_bull_argument
from alphamesh.agents.evidence import build_evidence
from alphamesh.agents.judge_agent import JudgeAgent
from alphamesh.agents.strategy_agent import StrategyAgent, make_decision_id
from alphamesh.intelligence.reasoning import (
    AnthropicProvider,
    LLMUnavailableError,
    MalformedAIOutputError,
    NullProvider,
    ScriptedProvider,
    build_provider,
    extract_json,
)
from alphamesh.models.domain import Direction, ReasonCode, Regime, Strategy
from tests.conftest import make_regime, make_signal


@pytest.fixture
def evidence():  # type: ignore[no-untyped-def]
    return build_evidence(make_signal(), make_regime())


class TestEvidencePacket:
    def test_council_never_sees_account_or_risk_information(self, evidence) -> None:  # type: ignore[no-untyped-def]
        """Checked against the actual field names of AccountState and RiskLimits.

        Substring matching would be wrong here: ``opening_range_position`` is a
        legitimate market feature that merely contains the word "position".
        """
        from alphamesh.alpaca.types import AccountState
        from alphamesh.config import RiskLimits

        forbidden = set(AccountState.model_fields) | set(RiskLimits.model_fields)

        def keys(node: object) -> set[str]:
            if isinstance(node, dict):
                found = set(node)
                for value in node.values():
                    found |= keys(value)
                return found
            return set()

        assert keys(evidence) & forbidden == set()

    def test_evidence_is_a_whitelist(self, evidence) -> None:  # type: ignore[no-untyped-def]
        assert set(evidence) == {
            "symbol",
            "as_of",
            "quant_score",
            "quant_directional_bias",
            "regime",
            "regime_direction",
            "regime_confidence",
            "regime_risk_flags",
            "features",
        }


class TestJsonExtraction:
    def test_plain_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_json_wrapped_in_prose_and_fences(self) -> None:
        raw = 'Sure! Here it is:\n```json\n{"strategy": "NO_TRADE"}\n```\nHope that helps.'
        assert extract_json(raw) == {"strategy": "NO_TRADE"}

    @pytest.mark.parametrize("raw", ["", "   ", "no json here", "{not json}", "[1,2]"])
    def test_unusable_output_raises(self, raw: str) -> None:
        with pytest.raises(MalformedAIOutputError):
            extract_json(raw)


class TestProviders:
    def test_null_provider_is_unavailable(self) -> None:
        provider = NullProvider()
        assert not provider.available()
        with pytest.raises(LLMUnavailableError):
            provider.complete_json(system="s", user="u")

    def test_build_provider_without_key_returns_null(self) -> None:
        assert isinstance(build_provider("", "claude-sonnet-5"), NullProvider)

    def test_build_provider_with_key_returns_anthropic(self) -> None:
        assert isinstance(build_provider("sk-test", "claude-sonnet-5"), AnthropicProvider)

    def test_anthropic_network_failure_becomes_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = AnthropicProvider("sk-test", client=client)
        with pytest.raises(LLMUnavailableError):
            provider.complete_json(system="s", user="u")

    def test_anthropic_http_error_becomes_unavailable(self) -> None:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
        )
        provider = AnthropicProvider("sk-test", client=client)
        with pytest.raises(LLMUnavailableError):
            provider.complete_json(system="s", user="u")

    def test_anthropic_parses_a_well_formed_response(self) -> None:
        payload = {"content": [{"type": "text", "text": '{"strategy": "NO_TRADE"}'}]}
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        )
        provider = AnthropicProvider("sk-test", client=client)
        assert provider.complete_json(system="s", user="u") == {"strategy": "NO_TRADE"}

    def test_anthropic_never_puts_the_key_in_its_repr(self) -> None:
        provider = AnthropicProvider("sk-super-secret-value")
        assert "sk-super-secret-value" not in repr(provider)


class TestBullAndBearAgents:
    def test_agents_fall_back_to_heuristics_without_a_provider(self, evidence) -> None:  # type: ignore[no-untyped-def]
        bull = BullAgent(NullProvider()).argue(evidence)
        bear = BearAgent(NullProvider()).argue(evidence)
        assert bull.provider == "heuristic"
        assert bull.stance is Direction.BULLISH
        assert bear.stance is Direction.BEARISH

    def test_agents_use_the_provider_when_it_works(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [{"thesis": "Up.", "key_points": ["a", "b"], "conviction": 0.8}]
        )
        argument = BullAgent(provider).argue(evidence)
        assert argument.provider == "scripted"
        assert argument.conviction == 0.8

    def test_malformed_output_falls_back_rather_than_failing(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider([{"thesis": "", "key_points": []}])
        argument = BullAgent(provider).argue(evidence)
        assert argument.provider == "heuristic"

    def test_out_of_range_conviction_is_clamped(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [{"thesis": "Up.", "key_points": ["a"], "conviction": 99}]
        )
        assert BullAgent(provider).argue(evidence).conviction == 1.0

    def test_heuristic_conviction_respects_regime_alignment(self) -> None:
        bullish = build_evidence(make_signal(), make_regime())
        assert heuristic_bull_argument(bullish).conviction > 0.5
        assert heuristic_bear_argument(bullish).conviction == 0.0


class TestJudgeAgent:
    def test_falls_back_when_no_provider(self, evidence) -> None:  # type: ignore[no-untyped-def]
        bull = heuristic_bull_argument(evidence)
        bear = heuristic_bear_argument(evidence)
        verdict = JudgeAgent(NullProvider()).judge(evidence, bull, bear, make_regime())
        assert verdict.provider == "heuristic"
        assert ReasonCode.AI_UNAVAILABLE in verdict.reason_codes

    def test_valid_verdict_is_accepted(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [
                {
                    "strategy": "BULL_CALL_SPREAD",
                    "confidence": 0.8,
                    "bull_score": 0.9,
                    "bear_score": 0.2,
                    "rationale": "Trend up.",
                }
            ]
        )
        verdict = JudgeAgent(provider).judge(
            evidence,
            heuristic_bull_argument(evidence),
            heuristic_bear_argument(evidence),
            make_regime(),
        )
        assert verdict.strategy is Strategy.BULL_CALL_SPREAD
        assert verdict.confidence == 0.8

    @pytest.mark.parametrize(
        "strategy",
        ["IRON_CONDOR", "NAKED_CALL", "SELL_PUT", "BUY_STOCK", "bull_call_spread_v2", "1"],
    )
    def test_unsupported_strategy_is_refused_not_coerced(self, evidence, strategy) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [{"strategy": strategy, "confidence": 0.99, "rationale": "trust me"}]
        )
        verdict = JudgeAgent(provider).judge(
            evidence,
            heuristic_bull_argument(evidence),
            heuristic_bear_argument(evidence),
            make_regime(),
        )
        assert verdict.strategy is Strategy.NO_TRADE
        assert ReasonCode.AI_UNSUPPORTED_STRATEGY in verdict.reason_codes
        assert verdict.confidence == 0.0

    def test_malformed_verdict_falls_back(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider([{"confidence": 0.9}])
        verdict = JudgeAgent(provider).judge(
            evidence,
            heuristic_bull_argument(evidence),
            heuristic_bear_argument(evidence),
            make_regime(),
        )
        assert verdict.provider == "heuristic"

    def test_llm_outage_falls_back(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider([LLMUnavailableError("down")])
        verdict = JudgeAgent(provider).judge(
            evidence,
            heuristic_bull_argument(evidence),
            heuristic_bear_argument(evidence),
            make_regime(),
        )
        assert verdict.provider == "heuristic"
        assert ReasonCode.AI_UNAVAILABLE in verdict.reason_codes

    def test_directional_verdict_is_overridden_in_an_unstable_regime(self, evidence) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [{"strategy": "BULL_CALL_SPREAD", "confidence": 0.99, "rationale": "go"}]
        )
        unstable = make_regime(regime=Regime.UNSTABLE, direction=Direction.NEUTRAL)
        verdict = JudgeAgent(provider).judge(
            evidence,
            heuristic_bull_argument(evidence),
            heuristic_bear_argument(evidence),
            unstable,
        )
        assert verdict.strategy is Strategy.NO_TRADE
        assert ReasonCode.REGIME_UNSTABLE in verdict.reason_codes


class TestStrategyAgent:
    def test_ai_is_not_invoked_below_the_quant_gate(self, config) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [{"thesis": "x", "key_points": ["y"], "conviction": 1.0}] * 5
        )
        agent = StrategyAgent(config, provider)
        result = agent.decide(make_signal(quant_score=0.1, passes=False), make_regime())
        assert result.decision.strategy is Strategy.NO_TRADE
        assert provider.calls == [], "the council must not be consulted below the gate"

    def test_ai_is_not_invoked_in_an_unknown_regime(self, config) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider([{"thesis": "x", "key_points": ["y"]}] * 5)
        agent = StrategyAgent(config, provider)
        regime = make_regime(regime=Regime.UNKNOWN, direction=Direction.NEUTRAL)
        result = agent.decide(make_signal(), regime)
        assert result.decision.strategy is Strategy.NO_TRADE
        assert ReasonCode.REGIME_UNKNOWN in result.decision.reason_codes
        assert provider.calls == []

    def test_no_trade_always_carries_a_reason(self, config) -> None:  # type: ignore[no-untyped-def]
        agent = StrategyAgent(config, NullProvider())
        result = agent.decide(make_signal(quant_score=0.1, passes=False), make_regime())
        assert result.decision.no_trade_reason

    def test_confidence_below_the_floor_becomes_no_trade(self, config) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [
                {"thesis": "up", "key_points": ["a"], "conviction": 0.9},
                {"thesis": "down", "key_points": ["b"], "conviction": 0.1},
                {
                    "strategy": "BULL_CALL_SPREAD",
                    "confidence": 0.20,
                    "rationale": "weak",
                },
            ]
        )
        agent = StrategyAgent(config, provider)
        result = agent.decide(make_signal(), make_regime())
        assert result.decision.strategy is Strategy.NO_TRADE
        assert ReasonCode.LOW_JUDGE_CONFIDENCE in result.decision.reason_codes

    def test_tradable_decision_is_produced_when_everything_agrees(self, config) -> None:  # type: ignore[no-untyped-def]
        provider = ScriptedProvider(
            [
                {"thesis": "up", "key_points": ["a"], "conviction": 0.9},
                {"thesis": "down", "key_points": ["b"], "conviction": 0.1},
                {
                    "strategy": "BULL_CALL_SPREAD",
                    "confidence": 0.82,
                    "bull_score": 0.9,
                    "bear_score": 0.15,
                    "rationale": "clean uptrend",
                },
            ]
        )
        result = StrategyAgent(config, provider).decide(make_signal(), make_regime())
        assert result.decision.strategy is Strategy.BULL_CALL_SPREAD
        assert result.decision.direction is Direction.BULLISH
        assert result.decision.is_tradable

    def test_decision_id_is_deterministic(self) -> None:
        from tests.conftest import NOW

        assert make_decision_id("SPY", NOW, 0.7) == make_decision_id("SPY", NOW, 0.7)
        assert make_decision_id("SPY", NOW, 0.7) != make_decision_id("QQQ", NOW, 0.7)
