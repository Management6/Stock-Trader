"""Structured market-data quality validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from multi_agent_trading_lab.data.data_sources import MarketData

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class DataQualityIssue:
    code: str
    severity: str
    symbol: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataQualityReport:
    passed: bool
    issues: list[DataQualityIssue]

    @property
    def warnings(self) -> list[DataQualityIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def errors(self) -> list[DataQualityIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "symbol": issue.symbol,
                    "message": issue.message,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
        }


def validate_market_data(
    data: MarketData,
    as_of_date: str | None = None,
    max_staleness_days: int | None = None,
    max_gap_days: int | None = None,
    block_on_missing_columns: bool = True,
    block_on_duplicate_dates: bool = True,
    block_on_invalid_prices: bool = True,
    block_on_stale_data: bool = False,
    block_on_large_gaps: bool = False,
    block_on_zero_volume: bool = False,
    expect_volume: bool = True,
) -> DataQualityReport:
    """Validate internal bar data and return structured issues."""

    issues: list[DataQualityIssue] = []
    for symbol, bars in data.items():
        issues.extend(
            _validate_symbol(
                symbol,
                bars,
                as_of_date,
                max_staleness_days,
                max_gap_days,
                block_on_missing_columns,
                block_on_duplicate_dates,
                block_on_invalid_prices,
                block_on_stale_data,
                block_on_large_gaps,
                block_on_zero_volume,
                expect_volume,
            )
        )
    return DataQualityReport(passed=not any(issue.severity == "error" for issue in issues), issues=issues)


def _validate_symbol(
    symbol: str,
    bars: list[dict[str, Any]],
    as_of_date: str | None,
    max_staleness_days: int | None,
    max_gap_days: int | None,
    block_on_missing_columns: bool,
    block_on_duplicate_dates: bool,
    block_on_invalid_prices: bool,
    block_on_stale_data: bool,
    block_on_large_gaps: bool,
    block_on_zero_volume: bool,
    expect_volume: bool,
) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    if not bars:
        return [
            DataQualityIssue(
                "missing_rows",
                "error",
                symbol,
                f"{symbol} has no market-data rows.",
                {"row_count": 0},
            )
        ]

    required_columns = REQUIRED_COLUMNS if expect_volume else tuple(column for column in REQUIRED_COLUMNS if column != "volume")
    missing = sorted({column for bar in bars for column in required_columns if column not in bar})
    if missing:
        issues.append(
            DataQualityIssue(
                "missing_columns",
                _severity(block_on_missing_columns),
                symbol,
                f"{symbol} is missing required OHLCV columns.",
                {"missing_columns": missing},
            )
        )

    dates = [str(bar.get("date", "")) for bar in bars]
    duplicate_dates = sorted({value for value in dates if value and dates.count(value) > 1})
    if duplicate_dates:
        issues.append(
            DataQualityIssue(
                "duplicate_dates",
                _severity(block_on_duplicate_dates),
                symbol,
                f"{symbol} has duplicate market-data dates.",
                {"dates": duplicate_dates},
            )
        )

    invalid_price_rows = [
        index
        for index, bar in enumerate(bars)
        if any(_not_positive_number(bar.get(column)) for column in PRICE_COLUMNS if column in bar)
    ]
    if invalid_price_rows:
        issues.append(
            DataQualityIssue(
                "invalid_prices",
                _severity(block_on_invalid_prices),
                symbol,
                f"{symbol} has non-positive price values.",
                {"rows": invalid_price_rows},
            )
        )

    zero_volume_rows = [index for index, bar in enumerate(bars) if "volume" in bar and _number_or_none(bar.get("volume")) == 0.0]
    if expect_volume and zero_volume_rows:
        issues.append(
            DataQualityIssue(
                "zero_volume",
                _severity(block_on_zero_volume),
                symbol,
                f"{symbol} has zero-volume bars.",
                {"rows": zero_volume_rows},
            )
        )

    parsed_dates = sorted(_parse_date(value) for value in dates if value)
    if as_of_date and max_staleness_days is not None and parsed_dates:
        latest_date = parsed_dates[-1]
        staleness_days = (_parse_date(as_of_date) - latest_date).days
        if staleness_days > max_staleness_days:
            issues.append(
                DataQualityIssue(
                    "stale_data",
                    _severity(block_on_stale_data),
                    symbol,
                    f"{symbol} latest bar is stale.",
                    {"latest_date": latest_date.isoformat(), "as_of_date": as_of_date, "staleness_days": staleness_days},
                )
            )

    if max_gap_days is not None and len(parsed_dates) >= 2:
        gaps = [
            {"from": previous.isoformat(), "to": current.isoformat(), "days": (current - previous).days}
            for previous, current in zip(parsed_dates, parsed_dates[1:])
            if (current - previous).days > max_gap_days
        ]
        if gaps:
            issues.append(
                DataQualityIssue(
                    "large_gap",
                    _severity(block_on_large_gaps),
                    symbol,
                    f"{symbol} has large date gaps.",
                    {"gaps": gaps},
                )
            )

    return issues


def _not_positive_number(value: Any) -> bool:
    number = _number_or_none(value)
    return number is None or number <= 0.0


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _severity(block: bool) -> str:
    return "error" if block else "warning"
