"""Order request and status models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class OrderRequest:
    """Provider-neutral order request."""

    symbol: str
    side: str
    quantity: int
    order_type: str = "market"
    limit_price: float | None = None
    timestamp: str = ""
    reason: str = ""
    source_strategy: str = ""
    estimated_price: float | None = None

    def normalized(self) -> "OrderRequest":
        timestamp = self.timestamp or datetime.now(UTC).isoformat()
        return OrderRequest(
            symbol=self.symbol.upper(),
            side=self.side.lower(),
            quantity=self.quantity,
            order_type=self.order_type.lower(),
            limit_price=self.limit_price,
            timestamp=timestamp,
            reason=self.reason,
            source_strategy=self.source_strategy,
            estimated_price=self.estimated_price,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OrderStatus:
    """Normalized broker order status."""

    order_id: str
    symbol: str
    side: str
    quantity: int
    status: str
    filled_quantity: int
    average_fill_price: float | None
    message: str
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

