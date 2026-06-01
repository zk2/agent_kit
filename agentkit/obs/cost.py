"""Token cost accounting.

Approximate published Anthropic prices in USD per 1M tokens (input, output). Used to attach a
per-node cost to traces so a run's spend is observable. Unknown models cost 0 (e.g. the eval stub).
"""

from __future__ import annotations

_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = _PRICES.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out, 6)
