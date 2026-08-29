"""Alpaca CLI operational adapter."""

from __future__ import annotations

import pytest

from alphamesh.alpaca.cli_adapter import (
    READ_ONLY_COMMANDS,
    AlpacaCliAdapter,
    CliUnavailableError,
)


class TestAvailability:
    def test_missing_binary_reports_unavailable(self) -> None:
        adapter = AlpacaCliAdapter("definitely-not-a-real-binary-xyz")
        assert adapter.available() is False
        assert adapter.resolved_path() is None

    def test_status_is_honest_when_absent(self) -> None:
        status = AlpacaCliAdapter("definitely-not-a-real-binary-xyz").status()
        assert status["available"] is False
        assert status["detail"] == "not found on PATH"

    def test_calling_an_absent_cli_raises_rather_than_faking_output(self) -> None:
        adapter = AlpacaCliAdapter("definitely-not-a-real-binary-xyz")
        with pytest.raises(CliUnavailableError):
            adapter.run_named("account")

    def test_a_present_binary_is_resolved(self) -> None:
        """Uses /bin/echo purely to prove PATH resolution works."""
        adapter = AlpacaCliAdapter("echo")
        assert adapter.available() is True
        assert adapter.status()["available"] is True


class TestCommandAllowlist:
    def test_only_read_only_commands_are_permitted(self) -> None:
        adapter = AlpacaCliAdapter("echo")
        with pytest.raises(ValueError, match="not an allowlisted"):
            adapter.run_named("place_order")

    def test_no_allowlisted_command_can_place_an_order(self) -> None:
        for args in READ_ONLY_COMMANDS.values():
            assert args[1] in {"get", "list"}

    def test_allowlisted_command_executes(self) -> None:
        result = AlpacaCliAdapter("echo").run_named("account")
        assert result.ok
        assert "account get" in result.stdout
