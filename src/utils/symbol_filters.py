# src/utils/symbol_filters.py
from __future__ import annotations
from typing import Dict, Any, Optional
import math

class SymbolFilters:
    def __init__(self, info: Dict[str, Any]):
        # info is what you get from client.futures_symbol_info(symbol)
        self.info = info or {}
        self.filters = {f["filterType"]: f for f in self.info.get("filters", [])}

        lot = self.filters.get("LOT_SIZE") or self.filters.get("MARKET_LOT_SIZE") or {}
        pricef = self.filters.get("PRICE_FILTER") or {}
        notional = self.filters.get("MIN_NOTIONAL") or {}

        self.stepSize = float(lot.get("stepSize", "0.001"))       # ETHUSDT usually 0.001
        self.minQty   = float(lot.get("minQty", "0.001"))
        self.maxQty   = float(lot.get("maxQty", "100000000"))

        self.tickSize = float(pricef.get("tickSize", "0.01"))      # ETHUSDT usually 0.01
        self.minPrice = float(pricef.get("minPrice", "0.01"))
        self.maxPrice = float(pricef.get("maxPrice", "100000000"))

        # On Futures, MIN_NOTIONAL may exist (not always).
        self.minNotional = float(notional.get("notional", "0"))

    def _floor_to_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step) * step

    def quantize_qty(self, qty: float) -> float:
        q = max(self.minQty, min(qty, self.maxQty))
        q = self._floor_to_step(q, self.stepSize)
        # guard: flooring could go below minQty if qty < minQty
        if q < self.minQty:
            q = self.minQty
        return float(f"{q:.18f}".rstrip("0").rstrip("."))

    def quantize_price(self, price: float) -> float:
        p = max(self.minPrice, min(price, self.maxPrice))
        p = self._floor_to_step(p, self.tickSize)
        if p < self.minPrice:
            p = self.minPrice
        return float(f"{p:.18f}".rstrip("0").rstrip("."))

    def meets_notional(self, qty: float, price: float) -> bool:
        if self.minNotional <= 0:
            return True
        return (qty * price) >= self.minNotional