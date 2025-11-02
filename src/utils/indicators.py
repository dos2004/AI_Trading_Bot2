import pandas as pd
import numpy as np
from typing import Optional, Tuple

def _rma(series: pd.Series, period: int) -> pd.Series:
    # Wilder’s RMA == EMA with alpha = 1/period
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    TradingView-style RSI (Wilder's RSI)
    """
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if len(rsi.dropna()) else None

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Standard MACD (EMA fast/slow, EMA signal). Matches TV defaults.
    """
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < slow + signal:
        return None, None, None
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

def calculate_ema(prices: pd.Series, period: int) -> Optional[float]:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < period:
        return None
    ema = prices.ewm(span=period, adjust=False).mean()
    return float(ema.iloc[-1])

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Optional[float]:
    """
    TradingView-style ATR uses Wilder’s smoothing (RMA of TR)
    """
    high = pd.to_numeric(high, errors="coerce").dropna()
    low = pd.to_numeric(low, errors="coerce").dropna()
    close = pd.to_numeric(close, errors="coerce").dropna()
    n = min(len(high), len(low), len(close))
    if n < period + 1:
        return None
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _rma(tr, period)
    return float(atr.iloc[-1]) if len(atr.dropna()) else None

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0):
    """
    TV default: SMA + sample stdev (ddof=1). pandas .std() is sample by default.
    """
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < period:
        return None, None, None
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()  # ddof=1
    upper = sma + num_std * std
    lower = sma - num_std * std
    return float(sma.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])

def calculate_sma(prices: pd.Series, period: int) -> Optional[float]:
    prices = pd.to_numeric(prices, errors="coerce").dropna()
    if len(prices) < period:
        return None
    sma = prices.rolling(window=period).mean()
    return float(sma.iloc[-1])

def calculate_volume_ratio(current_volume: float, avg_volume: float) -> float:
    if avg_volume == 0:
        return 0.0
    return (current_volume / avg_volume) * 100

def calculate_change_percent(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100