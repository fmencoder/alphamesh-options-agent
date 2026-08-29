# Raw Alpaca MCP responses

Verbatim (trimmed) responses from the Alpaca MCP server, kept so the ingest code
in `alphamesh/alpaca/mcp_adapter.py` can be tested against the real payload
shape rather than against a guess at it. `tests/test_mcp_adapter.py` parses
these files and asserts the values match the CSV captures in the parent
directory.

Trimming is limited to reducing the number of bars and contracts, plus
redacting the account identifiers in `get_account_info.json` (`id` and
`account_number`). No field was renamed, reshaped or invented; the paper `PA`
prefix is preserved because the safety guard checks it.

Captured 2026-08-29 with the market closed (`get_clock` shows `is_open: false`).
