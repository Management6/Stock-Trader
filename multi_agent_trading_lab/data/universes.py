"""Config-driven stock universe helpers."""

from __future__ import annotations

from typing import Any


def resolve_universe_symbols(settings: dict[str, Any], universe_name: str | None = None) -> list[str]:
    """Resolve symbols from a named universe, falling back to data.symbols."""

    data_config = dict(settings.get("data", {}))
    universes = dict(settings.get("universes", {}))
    selected = universe_name or data_config.get("default_universe")
    if selected:
        if selected not in universes:
            raise KeyError(f"Unknown universe {selected!r}. Available universes: {', '.join(sorted(universes))}")
        return _clean_symbols(universes[selected])
    return _clean_symbols(data_config.get("symbols", settings.get("symbols", [])))


def validate_universe_symbols(universe_name: str, symbols: list[str]) -> list[str]:
    """Return operator-facing warnings for suspicious universe selections."""

    warnings: list[str] = []
    if not symbols:
        warnings.append(f"Universe {universe_name!r} has zero symbols.")
    if universe_name == "full_mix" and len(symbols) < 3:
        warnings.append("Universe 'full_mix' has fewer than 3 symbols; check config/settings.yaml.")
    return warnings


def _clean_symbols(symbols: Any) -> list[str]:
    if symbols is None:
        return []
    cleaned: list[str] = []
    for symbol in symbols:
        text = str(symbol).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
