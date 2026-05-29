"""
ScalpAI Trading Bot - v6.0
EMA 9/21 + RSI + Bollinger Bands + MACD + 50 EMA Trend Filter
Improved entry criteria: min score 4, BB bandwidth 1.2%, 50 EMA trend confirmation
"""
import os, time, logging, json, requests
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import numpy as np
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
SYMBOLS    = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
ENABLED_SYMBOLS = {s: True for s in SYMBOLS}  # per-symbol trading toggle

# Live prices from WebSocket
live_prices = {}
STATE_FILE  = "/tmp/scalp_state.json"
DIARY_DIR   = "/tmp/daily_diaries"

STRATEGY = {
    "ema_fast": 9, "ema_slow": 21, "ema_trend": 50,
    "enabled_regime_filter": True,
    "strategy_b_enabled":    True,   # Fefa MSS strategy
    "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
    "bb_period": 20, "bb_std": 2.0, "bb_min_bandwidth": 1.2,
    "stop_loss_pct": 1.5, "take_profit_pct": 3.0,
    "min_score": 4,
    "enabled_ema": True, "enabled_rsi_bb": True,
    "enabled_macd": True, "macd_threshold": 0,
    "enabled_trend_filter": True,
    "enabled_kronos": False, "kronos_endpoint": "",
}
STRATEGY_ORIGINAL = dict(STRATEGY)

bot_state = {
    "running": True, "killed": False,
    "positions": {}, "closed_trades": [], "diary": [],
    "day_pnl": 0.0, "total_trades": 0, "win_count": 0,
    "ai_changes": [],
    "account_cash": 0.0, "account_equity": 0.0, "account_buying_power": 0.0,
    "market_regime": "UNKNOWN",
    "regime_checked_at": None,
}

# ── Persistence ────────────────────────────────────────────────────────────────
def save_daily_diary():
    """Save today's diary entries to a daily file"""
    try:
        os.makedirs(DIARY_DIR, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = f"{DIARY_DIR}/{today}.json"
        # Get today's entries
        today_entries = [d for d in bot_state["diary"] 
                        if d.get("time", "") >= "00:00"]
        # Merge with existing if file exists
        existing = []
        if os.path.exists(filepath):
            with open(filepath) as f:
                existing = json.load(f).get("entries", [])
        # Combine and deduplicate
        all_entries = existing + [e for e in today_entries 
                                   if e not in existing]
        with open(filepath, "w") as f:
            json.dump({"date": today, "entries": all_entries}, f)
    except Exception as e:
        log.error(f"Daily diary save error: {e}")

def save_state():
    try:
        data = {
            "diary":         bot_state["diary"][-200:],
            "closed_trades": bot_state["closed_trades"][-100:],
            "day_pnl":       bot_state["day_pnl"],
            "total_trades":  bot_state["total_trades"],
            "win_count":     bot_state["win_count"],
            "ai_changes":    bot_state["ai_changes"],
            "strategy":      STRATEGY,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
        save_daily_diary()
    except Exception as e:
        log.error(f"Save state error: {e}")

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                data = json.load(f)
            bot_state["diary"]         = data.get("diary", [])
            bot_state["closed_trades"] = data.get("closed_trades", [])
            bot_state["day_pnl"]       = data.get("day_pnl", 0.0)
            bot_state["total_trades"]  = data.get("total_trades", 0)
            bot_state["win_count"]     = data.get("win_count", 0)
            bot_state["ai_changes"]    = data.get("ai_changes", [])
            saved_strategy = data.get("strategy", {})
            for k, v in saved_strategy.items():
                if k in STRATEGY:
                    STRATEGY[k] = v
            log.info(f"State loaded: {len(bot_state['diary'])} diary entries")
    except Exception as e:
        log.error(f"Load state error: {e}")

# ── Alpaca helpers ─────────────────────────────────────────────────────────────
def get_account_data():
    try:
        base = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
        r = requests.get(
            base + "/v2/account",
            headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET},
            timeout=5
        )
        if r.status_code == 200:
            d = r.json()
            bot_state["account_cash"]         = float(d.get("cash", 0))
            bot_state["account_equity"]       = float(d.get("equity", 0))
            bot_state["account_buying_power"] = float(d.get("buying_power", 0))
        else:
            log.error(f"Account fetch failed: {r.status_code}")
    except Exception as e:
        log.error(f"Account fetch error: {e}")

def sync_positions_from_alpaca():
    try:
        base = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
        r = requests.get(
            base + "/v2/positions",
            headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET},
            timeout=5
        )
        if r.status_code == 200:
            positions = r.json()
            live_syms = set()
            for pos in positions:
                sym = pos["symbol"]
                sym_slash = sym[:3] + "/" + sym[3:] if "/" not in sym else sym
                live_syms.add(sym)
                if sym not in bot_state["positions"]:
                    try:
                        open_at = pos.get("created_at", "")
                        if open_at:
                            dt = datetime.fromisoformat(open_at.replace("Z", "+00:00"))
                            open_time = dt.strftime("%H:%M")
                        else:
                            open_time = datetime.now().strftime("%H:%M")
                    except:
                        open_time = datetime.now().strftime("%H:%M")
                    bot_state["positions"][sym] = {
                        "entry":     float(pos["avg_entry_price"]),
                        "qty":       float(pos["qty"]),
                        "open_time": open_time,
                        "symbol":    sym_slash,
                        "current_price": float(pos.get("current_price", pos["avg_entry_price"])),
                        "unrealized_pnl": float(pos.get("unrealized_pl", 0)),
                        "unrealized_pct": float(pos.get("unrealized_plpc", 0)) * 100,
                    }
                else:
                    bot_state["positions"][sym]["current_price"] = float(pos.get("current_price", pos["avg_entry_price"]))
                    bot_state["positions"][sym]["unrealized_pnl"] = float(pos.get("unrealized_pl", 0))
                    bot_state["positions"][sym]["unrealized_pct"] = float(pos.get("unrealized_plpc", 0)) * 100
                    bot_state["positions"][sym]["qty"] = float(pos["qty"])
            # Remove closed positions
            for sym in list(bot_state["positions"].keys()):
                if sym not in live_syms:
                    del bot_state["positions"][sym]
        else:
            log.error(f"Position sync failed: {r.status_code}")
    except Exception as e:
        log.error(f"Position sync error: {e}")

def get_data_client():
    return CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

def get_trading_client():
    return TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER_MODE)

# ── Indicators ─────────────────────────────────────────────────────────────────
def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_bollinger(series, period=20, std=2.0):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    bw    = ((upper - lower) / mid * 100).iloc[-1]
    return upper, mid, lower, bw

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def detect_rsi_divergence(close, rsi, lookback=14):
    """
    Detect bullish RSI divergence:
    Price makes lower low but RSI makes higher low = bullish divergence (leading buy signal)
    Detect bearish RSI divergence:
    Price makes higher high but RSI makes lower high = bearish divergence (leading sell signal)
    """
    if len(close) < lookback * 2:
        return False, False
    
    try:
        prices = close.values
        rsi_vals = rsi.values
        
        # Find recent swing lows in price (for bullish divergence)
        bull_div = False
        bear_div = False
        
        # Look at last 3 swing points
        n = len(prices)
        
        # Bullish divergence: price lower low, RSI higher low
        # Find two recent lows
        recent_lows_price = []
        recent_lows_rsi = []
        for i in range(n-2, max(n-lookback*2, 2), -1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                recent_lows_price.append((i, prices[i]))
                recent_lows_rsi.append((i, rsi_vals[i]))
            if len(recent_lows_price) >= 2:
                break
        
        if len(recent_lows_price) >= 2:
            # Most recent low vs previous low
            p1_idx, p1_price = recent_lows_price[0]  # more recent
            p2_idx, p2_price = recent_lows_price[1]  # older
            r1 = recent_lows_rsi[0][1]
            r2 = recent_lows_rsi[1][1]
            # Bullish: price lower but RSI higher
            if p1_price < p2_price and r1 > r2 and r1 < 50:
                bull_div = True
        
        # Bearish divergence: price higher high, RSI lower high
        recent_highs_price = []
        recent_highs_rsi = []
        for i in range(n-2, max(n-lookback*2, 2), -1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                recent_highs_price.append((i, prices[i]))
                recent_highs_rsi.append((i, rsi_vals[i]))
            if len(recent_highs_price) >= 2:
                break
        
        if len(recent_highs_price) >= 2:
            p1_idx, p1_price = recent_highs_price[0]
            p2_idx, p2_price = recent_highs_price[1]
            r1 = recent_highs_rsi[0][1]
            r2 = recent_highs_rsi[1][1]
            # Bearish: price higher but RSI lower
            if p1_price > p2_price and r1 < r2 and r1 > 50:
                bear_div = True
        
        return bull_div, bear_div
    except:
        return False, False


def detect_support_resistance(close, lookback=50):
    """
    Detect key support and resistance levels from recent price history.
    Returns whether current price is near support (buy zone) or resistance (sell zone).
    """
    if len(close) < lookback:
        return False, False
    
    try:
        prices = close.values[-lookback:]
        current = prices[-1]
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(prices)-2):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                swing_highs.append(prices[i])
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                swing_lows.append(prices[i])
        
        if not swing_lows and not swing_highs:
            return False, False
        
        # Check if current price is near support (within 0.5%)
        near_support = any(abs(current - s) / s < 0.005 for s in swing_lows)
        # Check if current price is near resistance (within 0.5%)
        near_resistance = any(abs(current - r) / r < 0.005 for r in swing_highs)
        
        return near_support, near_resistance
    except:
        return False, False


def detect_fib_levels(close, lookback=50):
    """
    Detect if price is at a key Fibonacci retracement level (38.2%, 50%, 61.8%).
    Returns True if at support fib level (buy zone).
    """
    if len(close) < lookback:
        return False
    
    try:
        prices = close.values[-lookback:]
        current = prices[-1]
        high = max(prices)
        low = min(prices)
        diff = high - low
        
        if diff == 0:
            return False
        
        # Fibonacci levels
        fib_382 = high - 0.382 * diff
        fib_500 = high - 0.500 * diff
        fib_618 = high - 0.618 * diff
        
        # Check if price is within 0.3% of any fib level
        tolerance = 0.003
        for fib in [fib_382, fib_500, fib_618]:
            if abs(current - fib) / fib < tolerance:
                return True
        return False
    except:
        return False


# ── Strategy B: Fefa Market Structure Shift ──────────────────────────────────

def detect_swing_points(close, lookback=5):
    """Detect swing highs and lows in price series."""
    prices = close.values
    n = len(prices)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        if all(prices[i] > prices[i-j] for j in range(1, lookback+1)) and            all(prices[i] > prices[i+j] for j in range(1, lookback+1)):
            swing_highs.append((i, prices[i]))
        if all(prices[i] < prices[i-j] for j in range(1, lookback+1)) and            all(prices[i] < prices[i+j] for j in range(1, lookback+1)):
            swing_lows.append((i, prices[i]))
    return swing_highs, swing_lows

def get_1h_trend(data_client, symbol):
    """
    Determine 1H trend direction using market structure.
    Returns 'BULL' if higher highs + higher lows, 'BEAR' if lower lows + lower highs, 'NEUTRAL' otherwise.
    """
    try:
        from alpaca.data.timeframe import TimeFrameUnit
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            limit=50
        )
        bars = data_client.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        if len(df) < 10:
            return "NEUTRAL"
        close = df["close"]
        swing_highs, swing_lows = detect_swing_points(close, lookback=3)
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"
        # Check last 2 swing highs and lows
        recent_highs = [h[1] for h in swing_highs[-2:]]
        recent_lows  = [l[1] for l in swing_lows[-2:]]
        if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
            return "BULL"
        elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
            return "BEAR"
        return "NEUTRAL"
    except Exception as e:
        log.error(f"1H trend error {symbol}: {e}")
        return "NEUTRAL"

def detect_key_levels(data_client, symbol):
    """
    Find key support/resistance levels from 1H chart where price previously reversed.
    Returns list of key price levels.
    """
    try:
        from alpaca.data.timeframe import TimeFrameUnit
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            limit=48
        )
        bars = data_client.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        close = df["close"]
        swing_highs, swing_lows = detect_swing_points(close, lookback=2)
        levels = [h[1] for h in swing_highs[-3:]] + [l[1] for l in swing_lows[-3:]]
        return levels
    except:
        return []

def detect_mss_5m(data_client, symbol, trend_1h):
    """
    Detect Market Structure Shift on 5M chart.
    In BULL trend: look for MSS upward (higher high after lower lows = bullish MSS = BUY)
    In BEAR trend: look for MSS downward (lower low after higher highs = bearish MSS = SELL)
    Returns signal, entry price, stop loss level, and signal data.
    """
    try:
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=60)
        bars = data_client.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        if len(df) < 20:
            return "HOLD", {}

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        prices = close.values
        highs  = high.values
        lows   = low.values
        n = len(prices)

        swing_highs, swing_lows = detect_swing_points(close, lookback=2)

        current_price = float(prices[-1])
        sig_data = {"price": round(current_price, 4), "trend_1h": trend_1h}

        if trend_1h == "BULL":
            # Look for bullish MSS: series of lower lows followed by a higher high
            if len(swing_lows) >= 2 and len(swing_highs) >= 1:
                last_low1  = swing_lows[-1][1]
                last_low2  = swing_lows[-2][1] if len(swing_lows) >= 2 else last_low1
                last_high  = swing_highs[-1][1]
                last_high_idx = swing_highs[-1][0]
                # MSS: recent low is higher than previous low (structure shifting up)
                if last_low1 > last_low2 and last_high_idx > swing_lows[-1][0]:
                    # Price near key level (within 1%)
                    key_levels = detect_key_levels(data_client, symbol)
                    near_level = any(abs(current_price - lv) / lv < 0.01 for lv in key_levels) if key_levels else True
                    if near_level:
                        stop_loss = float(min(lows[-10:]))  # recent swing low
                        sl_pct    = abs(current_price - stop_loss) / current_price * 100
                        sig_data.update({"mss_type": "BULL_MSS", "stop_loss_price": round(stop_loss, 4), "sl_pct": round(sl_pct, 3)})
                        return "BUY", sig_data

        elif trend_1h == "BEAR":
            # Look for bearish MSS: series of higher highs followed by a lower low
            if len(swing_highs) >= 2 and len(swing_lows) >= 1:
                last_high1 = swing_highs[-1][1]
                last_high2 = swing_highs[-2][1] if len(swing_highs) >= 2 else last_high1
                last_low   = swing_lows[-1][1]
                last_low_idx = swing_lows[-1][0]
                if last_high1 < last_high2 and last_low_idx > swing_highs[-1][0]:
                    key_levels = detect_key_levels(data_client, symbol)
                    near_level = any(abs(current_price - lv) / lv < 0.01 for lv in key_levels) if key_levels else True
                    if near_level:
                        stop_loss = float(max(highs[-10:]))
                        sl_pct    = abs(stop_loss - current_price) / current_price * 100
                        sig_data.update({"mss_type": "BEAR_MSS", "stop_loss_price": round(stop_loss, 4), "sl_pct": round(sl_pct, 3)})
                        return "SELL", sig_data

        return "HOLD", sig_data
    except Exception as e:
        log.error(f"MSS detection error {symbol}: {e}")
        return "HOLD", {}


def get_market_regime(data_client):
    """
    Check if we are in a bull or bear market using BTC daily 200 EMA.
    Returns True if bull market (BTC above 200 EMA), False if bear.
    """
    try:
        from alpaca.data.timeframe import TimeFrameUnit
        req = CryptoBarsRequest(
            symbol_or_symbols="BTC/USD",
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            limit=210
        )
        bars = data_client.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == "BTC/USD"].copy()
        if len(df) < 200:
            return True  # not enough data, assume bull
        close = df["close"]
        ema200 = compute_ema(close, 200)
        current_price = float(close.iloc[-1])
        ema200_val = float(ema200.iloc[-1])
        is_bull = current_price > ema200_val
        log.info(f"Market regime: {'BULL' if is_bull else 'BEAR'} | BTC={current_price:.0f} vs 200EMA={ema200_val:.0f}")
        return is_bull
    except Exception as e:
        log.error(f"Market regime error: {e}")
        return True  # default to bull if error

def detect_bullish_candle(open_prices, close_prices, lookback=3):
    """
    Detect bullish candlestick patterns:
    - Hammer: small body, long lower wick
    - Bullish engulfing: current candle engulfs previous bearish candle
    """
    if len(close_prices) < lookback:
        return False
    
    try:
        opens = open_prices.values
        closes = close_prices.values
        
        # Check last candle for hammer pattern
        o = opens[-1]
        c = closes[-1]
        prev_o = opens[-2]
        prev_c = closes[-2]
        
        # Bullish engulfing: previous bearish, current bullish and larger
        if prev_c < prev_o and c > o:  # prev bearish, curr bullish
            if c > prev_o and o < prev_c:  # curr engulfs prev
                return True
        
        return False
    except:
        return False


def fetch_bars(data_client, symbol, limit=100):
    req  = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=limit)
    bars = data_client.get_crypto_bars(req)
    df   = bars.df.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    df.set_index("timestamp", inplace=True)
    return df

def generate_signal(symbol, data_client):
    try:
        df5    = fetch_bars(data_client, symbol, limit=100)
        df1    = fetch_bars(data_client, symbol, limit=60)
        close5 = df5["close"]
        close1 = df1["close"]

        # EMA signals on 5m
        ema_fast5  = compute_ema(close5, STRATEGY["ema_fast"])
        ema_slow5  = compute_ema(close5, STRATEGY["ema_slow"])
        ema_trend5 = compute_ema(close5, STRATEGY["ema_trend"])
        trend_bull = ema_fast5.iloc[-1] > ema_slow5.iloc[-1]
        trend_prev = ema_fast5.iloc[-2] > ema_slow5.iloc[-2]
        ema5_cross_up   = trend_bull and not trend_prev
        ema5_cross_down = not trend_bull and trend_prev

        # 50 EMA trend filter — price must be above 50 EMA to go long
        price_above_50ema = close5.iloc[-1] > ema_trend5.iloc[-1]

        # EMA on 1m
        ema_fast1 = compute_ema(close1, STRATEGY["ema_fast"])
        ema_slow1 = compute_ema(close1, STRATEGY["ema_slow"])
        ema1_bull = ema_fast1.iloc[-1] > ema_slow1.iloc[-1]

        # RSI
        rsi            = compute_rsi(close1, STRATEGY["rsi_period"])
        rsi_val        = rsi.iloc[-1]
        rsi_oversold   = rsi_val < STRATEGY["rsi_oversold"]
        rsi_overbought = rsi_val > STRATEGY["rsi_overbought"]

        # Bollinger Bands
        bb_up, bb_mid, bb_lo, bb_bw = compute_bollinger(close1, STRATEGY["bb_period"], STRATEGY["bb_std"])
        price   = close1.iloc[-1]
        bb_buy  = price < bb_lo.iloc[-1] and bb_bw >= STRATEGY["bb_min_bandwidth"]
        bb_sell = price > bb_up.iloc[-1]

        # MACD
        macd_line, signal_line, histogram = compute_macd(close1)
        macd_hist = histogram.iloc[-1]
        macd_bull = macd_hist > STRATEGY["macd_threshold"]
        macd_bear = macd_hist < -STRATEGY["macd_threshold"]

        # Leading indicators
        rsi_computed = compute_rsi(close1, STRATEGY["rsi_period"])
        bull_div, bear_div     = detect_rsi_divergence(close1, rsi_computed)
        near_support, near_res = detect_support_resistance(close1)
        at_fib                 = detect_fib_levels(close1)
        bull_candle            = detect_bullish_candle(df1["open"], close1)

        buy_score = sell_score = 0
        if STRATEGY["enabled_ema"]:
            if trend_bull and ema1_bull: buy_score  += 2
            if ema5_cross_up:            buy_score  += 1
            if not trend_bull:           sell_score += 1
            if ema5_cross_down:          sell_score += 2
        if STRATEGY["enabled_rsi_bb"]:
            if rsi_oversold:   buy_score  += 2
            if bb_buy:         buy_score  += 1
            if rsi_overbought: sell_score += 2
            if bb_sell:        sell_score += 1
        if STRATEGY["enabled_macd"]:
            if macd_bull: buy_score  += 1
            if macd_bear: sell_score += 1

        # Leading indicators (weighted higher as they predict vs confirm)
        if bull_div:      buy_score  += 2  # RSI bullish divergence = strong leading buy
        if bear_div:      sell_score += 2  # RSI bearish divergence = strong leading sell
        if near_support:  buy_score  += 1  # price at support level
        if near_res:      sell_score += 1  # price at resistance level
        if at_fib:        buy_score  += 1  # price at fibonacci level
        if bull_candle:   buy_score  += 1  # bullish candlestick pattern

        # 50 EMA trend filter — blocks buy if price below 50 EMA
        trend_filter_ok = price_above_50ema if STRATEGY["enabled_trend_filter"] else True

        log.info(f"{symbol} | price={price:.2f} RSI={rsi_val:.1f} MACD_H={macd_hist:.4f} BB_BW={bb_bw:.2f}% BUY={buy_score} SELL={sell_score} DIV={bull_div} S/R={near_support} FIB={at_fib}")

        sig_data = {
            "rsi": round(rsi_val, 1), "bb_bw": round(bb_bw, 2),
            "macd_hist": round(macd_hist, 4),
            "ema_trend": "BULL" if trend_bull else "BEAR",
            "above_50ema": bool(price_above_50ema),
            "rsi_bull_div": bool(bull_div),
            "rsi_bear_div": bool(bear_div),
            "near_support": bool(near_support),
            "near_resistance": bool(near_res),
            "at_fib": bool(at_fib),
            "bull_candle": bool(bull_candle),
            "price": round(price, 4),
        }

        min_score = STRATEGY["min_score"]
        if buy_score >= min_score and buy_score > sell_score and trend_filter_ok:
            return "BUY",  {**sig_data, "score": buy_score}
        elif sell_score >= min_score and sell_score > buy_score:
            return "SELL", {**sig_data, "score": sell_score}
        else:
            return "HOLD", {**sig_data, "score": max(buy_score, sell_score)}
    except Exception as e:
        log.error(f"Signal error for {symbol}: {e}")
        return "HOLD", {}

def diary_entry(symbol, text, entry_type="trade"):
    bot_state["diary"].append({
        "time":   datetime.now().strftime("%H:%M"),
        "symbol": symbol, "text": text, "type": entry_type,
    })
    save_state()

def ai_analyse_loss(trade):
    changes = []
    sig   = trade.get("signal_data", {})
    bb_bw = sig.get("bb_bw", 999)
    rsi   = sig.get("rsi", 50)
    if bb_bw < 1.0 and STRATEGY["bb_min_bandwidth"] < 1.5:
        old = STRATEGY["bb_min_bandwidth"]
        STRATEGY["bb_min_bandwidth"] = round(min(old + 0.1, 1.5), 1)
        changes.append(f"BB bandwidth raised {old}->{STRATEGY['bb_min_bandwidth']}")
    if rsi > 45 and STRATEGY["rsi_oversold"] >= 26:
        old = STRATEGY["rsi_oversold"]
        STRATEGY["rsi_oversold"] = max(old - 2, 20)
        changes.append(f"RSI oversold tightened {old}->{STRATEGY['rsi_oversold']}")
    if changes:
        note = " | ".join(changes)
        bot_state["ai_changes"].append({"time": datetime.now().isoformat(), "changes": note, "trade": trade["symbol"]})
        diary_entry("AI", f"Strategy updated: {note}", "ai")

def trading_loop():
    if not API_KEY or not API_SECRET:
        log.warning("No API keys")
        return
    trading_client = get_trading_client()
    data_client    = get_data_client()

    # Clear stale state file on startup
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        log.info("Cleared stale state file - starting fresh")
    # Ensure all required keys exist
    if "strategy_b" not in bot_state:
        bot_state["strategy_b"] = {"positions": {}, "closed_trades": [], "day_pnl": 0.0, "total_trades": 0, "win_count": 0, "diary": []}
    if "trend_1h_cache" not in bot_state:
        bot_state["trend_1h_cache"] = {}
    if "trend_1h_checked_at" not in bot_state:
        bot_state["trend_1h_checked_at"] = {}
    if "market_regime" not in bot_state:
        bot_state["market_regime"] = "UNKNOWN"
    if "regime_checked_at" not in bot_state:
        bot_state["regime_checked_at"] = None
    get_account_data()
    sync_positions_from_alpaca()

    log.info(f"ScalpAI v6.0 started | Paper={PAPER_MODE} | MinScore={STRATEGY['min_score']} | BB_BW={STRATEGY['bb_min_bandwidth']} | TrendFilter={STRATEGY['enabled_trend_filter']}")
    diary_entry("SYSTEM", f"Bot v6.0 started | Min score: {STRATEGY['min_score']} | BB min BW: {STRATEGY['bb_min_bandwidth']}% | 50 EMA trend filter: {'ON' if STRATEGY['enabled_trend_filter'] else 'OFF'}", "system")

    while True:
        try:
            if bot_state["killed"]:
                time.sleep(5)
                continue

            get_account_data()
            sync_positions_from_alpaca()
            data_client = get_data_client()

            # Check market regime every 15 minutes
            now = datetime.now()
            last_check = bot_state["regime_checked_at"]
            if STRATEGY["enabled_regime_filter"] and (last_check is None or (now - last_check).seconds >= 900):
                is_bull = get_market_regime(data_client)
                bot_state["market_regime"] = "BULL" if is_bull else "BEAR"
                bot_state["regime_checked_at"] = now
                if not is_bull:
                    log.warning("BEAR MARKET DETECTED - new entries blocked")
                    diary_entry("SYSTEM", "Bear market detected (BTC below 200 EMA) - new entries paused", "system")

            cash      = bot_state["account_cash"]
            per_trade = cash * 0.25 if cash > 0 else 2500

            for symbol in SYMBOLS:
                sym_key     = symbol.replace("/", "")
                in_position = sym_key in bot_state["positions"]
                # Skip if symbol trading is disabled
                if not ENABLED_SYMBOLS.get(symbol, True):
                    log.info(f"{symbol} trading disabled, skipping")
                    continue

                signal, sig_data = generate_signal(symbol, data_client)

                if in_position:
                    pos     = bot_state["positions"][sym_key]
                    price   = pos.get("current_price", sig_data.get("price", pos["entry"]))
                    if price == pos["entry"]:
                        price = sig_data.get("price", pos["entry"])
                    pnl_pct = (price - pos["entry"]) / pos["entry"] * 100

                    should_exit = False
                    exit_reason = ""
                    if pnl_pct >= STRATEGY["take_profit_pct"]:
                        should_exit = True
                        exit_reason = f"Take profit ({pnl_pct:.2f}%)"
                    elif pnl_pct <= -STRATEGY["stop_loss_pct"]:
                        should_exit = True
                        exit_reason = f"Stop loss ({pnl_pct:.2f}%)"
                    elif signal == "SELL":
                        should_exit = True
                        exit_reason = "SELL signal"

                    if should_exit:
                        try:
                            req   = MarketOrderRequest(symbol=symbol, qty=pos["qty"], side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                            trading_client.submit_order(req)
                            pnl = (price - pos["entry"]) * pos["qty"]
                            win = pnl > 0
                            trade = {
                                "symbol": symbol, "entry": pos["entry"], "exit": price,
                                "qty": pos["qty"], "pnl": round(pnl,2), "pct": round(pnl_pct,2),
                                "win": win, "time": pos["open_time"],
                                "close_time": datetime.now().strftime("%H:%M"),
                                "signal": exit_reason, "signal_data": sig_data
                            }
                            bot_state["closed_trades"].append(trade)
                            bot_state["day_pnl"]       = round(bot_state["day_pnl"] + pnl, 2)
                            bot_state["total_trades"] += 1
                            if win: bot_state["win_count"] += 1
                            del bot_state["positions"][sym_key]
                            diary_entry(symbol,
                                f"{'WIN' if win else 'LOSS'} | Entry ${pos['entry']:,.2f} -> ${price:,.2f} | "
                                f"P&L ${pnl:.2f} ({pnl_pct:+.2f}%) | {exit_reason}",
                                "win" if win else "loss"
                            )
                            if not win: ai_analyse_loss(trade)
                            save_state()
                        except Exception as e:
                            log.error(f"Exit order failed {symbol}: {e}")

                elif signal == "BUY" and not bot_state["killed"] and (not STRATEGY["enabled_regime_filter"] or bot_state["market_regime"] != "BEAR"):
                    try:
                        price = sig_data.get("price", 0)
                        if price > 0:
                            qty = round(per_trade / price, 6)
                            if qty > 0:
                                req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                trading_client.submit_order(req)
                                bot_state["positions"][sym_key] = {
                                    "entry": price, "qty": qty,
                                    "open_time": datetime.now().strftime("%H:%M"),
                                    "symbol": symbol
                                }
                                diary_entry(symbol,
                                    f"BUY | ${price:,.2f} | Qty {qty} | Budget ${per_trade:,.2f} | "
                                    f"Score {sig_data.get('score','?')} | RSI {sig_data.get('rsi','?')} "
                                    f"MACD_H {sig_data.get('macd_hist','?')} BB_BW {sig_data.get('bb_bw','?')}% "
                                    f"Above50EMA: {sig_data.get('above_50ema','?')}",
                                    "trade"
                                )
                                save_state()
                    except Exception as e:
                        log.error(f"Entry order failed {symbol}: {e}")

            # ── Strategy B: Fefa MSS ────────────────────────────────────────────
            if STRATEGY["strategy_b_enabled"]:
                for symbol in SYMBOLS:
                    if not ENABLED_SYMBOLS.get(symbol, True):
                        continue
                    sym_key = symbol.replace("/", "")

                    # Cache 1H trend for 15 minutes
                    now_dt = datetime.now()
                    last_trend_check = bot_state["trend_1h_checked_at"].get(sym_key)
                    if last_trend_check is None or (now_dt - last_trend_check).seconds >= 900:
                        trend = get_1h_trend(data_client, symbol)
                        bot_state["trend_1h_cache"][sym_key] = trend
                        bot_state["trend_1h_checked_at"][sym_key] = now_dt
                        log.info(f"[B] {symbol} 1H trend: {trend}")
                    else:
                        trend = bot_state["trend_1h_cache"].get(sym_key, "NEUTRAL")

                    # Skip neutral trend
                    if trend == "NEUTRAL":
                        continue

                    # Skip bear market entries
                    if STRATEGY["enabled_regime_filter"] and bot_state["market_regime"] == "BEAR" and trend == "BULL":
                        continue

                    b_positions = bot_state["strategy_b"]["positions"]
                    in_position_b = sym_key in b_positions

                    signal_b, sig_data_b = detect_mss_5m(data_client, symbol, trend)

                    if in_position_b:
                        pos_b   = b_positions[sym_key]
                        price_b = sig_data_b.get("price", pos_b["entry"])
                        pnl_pct_b = (price_b - pos_b["entry"]) / pos_b["entry"] * 100

                        should_exit_b = False
                        exit_reason_b = ""
                        if pnl_pct_b >= STRATEGY["take_profit_pct"]:
                            should_exit_b = True
                            exit_reason_b = f"Take profit ({pnl_pct_b:.2f}%)"
                        elif pnl_pct_b <= -STRATEGY["stop_loss_pct"]:
                            should_exit_b = True
                            exit_reason_b = f"Stop loss ({pnl_pct_b:.2f}%)"
                        elif signal_b == "SELL" and trend == "BULL":
                            should_exit_b = True
                            exit_reason_b = "MSS reversal"

                        if should_exit_b:
                            try:
                                req = MarketOrderRequest(symbol=symbol, qty=pos_b["qty"], side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                                trading_client.submit_order(req)
                                pnl_b = (price_b - pos_b["entry"]) * pos_b["qty"]
                                win_b = pnl_b > 0
                                trade_b = {"symbol": symbol, "entry": pos_b["entry"], "exit": price_b,
                                           "qty": pos_b["qty"], "pnl": round(pnl_b,2), "pct": round(pnl_pct_b,2),
                                           "win": win_b, "strategy": "B",
                                           "time": pos_b["open_time"], "close_time": datetime.now().strftime("%H:%M"),
                                           "signal": exit_reason_b}
                                bot_state["strategy_b"]["closed_trades"].append(trade_b)
                                bot_state["strategy_b"]["day_pnl"] = round(bot_state["strategy_b"]["day_pnl"] + pnl_b, 2)
                                bot_state["strategy_b"]["total_trades"] += 1
                                if win_b: bot_state["strategy_b"]["win_count"] += 1
                                del b_positions[sym_key]
                                diary_entry(symbol, f"[B-MSS] {'WIN' if win_b else 'LOSS'} | ${pos_b['entry']:,.2f} -> ${price_b:,.2f} | P&L ${pnl_b:.2f} ({pnl_pct_b:+.2f}%) | {exit_reason_b}", "win" if win_b else "loss")
                                save_state()
                            except Exception as e:
                                log.error(f"[B] Exit order failed {symbol}: {e}")

                    elif signal_b == "BUY" and not bot_state["killed"] and trend == "BULL":
                        price_b = sig_data_b.get("price", 0)
                        if price_b > 0:
                            try:
                                qty_b = round((cash * 0.25) / price_b, 6)
                                if qty_b > 0:
                                    req = MarketOrderRequest(symbol=symbol, qty=qty_b, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                    trading_client.submit_order(req)
                                    b_positions[sym_key] = {"entry": price_b, "qty": qty_b,
                                                            "open_time": datetime.now().strftime("%H:%M"), "symbol": symbol}
                                    diary_entry(symbol, f"[B-MSS] BUY | ${price_b:,.2f} | Qty {qty_b} | 1H: {trend} | {sig_data_b.get('mss_type','MSS')}", "trade")
                                    save_state()
                            except Exception as e:
                                log.error(f"[B] Entry order failed {symbol}: {e}")

            time.sleep(60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Loop error: {e}")
            # Re-initialize missing keys
            if "strategy_b" not in bot_state:
                bot_state["strategy_b"] = {"positions": {}, "closed_trades": [], "day_pnl": 0.0, "total_trades": 0, "win_count": 0, "diary": []}
            if "trend_1h_cache" not in bot_state:
                bot_state["trend_1h_cache"] = {}
            if "trend_1h_checked_at" not in bot_state:
                bot_state["trend_1h_checked_at"] = {}
            time.sleep(30)

# ── Flask API ──────────────────────────────────────────────────────────────────
def clean_nan(obj):
    # Handle None
    if obj is None:
        return None
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Convert numpy types to native Python first
    if hasattr(obj, '__module__') and type(obj).__module__ == 'numpy':
        try:
            obj = obj.item()
        except:
            try:
                obj = float(obj)
            except:
                return 0
    # Handle bool before int (bool is subclass of int)
    if isinstance(obj, bool):
        return obj
    # Handle floats
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    # Handle ints
    if isinstance(obj, int):
        return obj
    # Handle strings
    if isinstance(obj, str):
        return obj
    # Handle dicts
    if isinstance(obj, dict):
        return {str(k): clean_nan(v) for k, v in obj.items()}
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    # Fallback - convert to string
    try:
        return str(obj)
    except:
        return None

app = Flask(__name__)
CORS(app)

@app.after_request
def add_no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/status")
def status():
    get_account_data()
    sync_positions_from_alpaca()
    wins  = bot_state["win_count"]
    total = bot_state["total_trades"]
    payload = {
        "running":               bot_state["running"],
        "killed":                bot_state["killed"],
        "paper_mode":            PAPER_MODE,
        "positions":             bot_state["positions"],
        "closed_trades":         bot_state["closed_trades"][-50:],
        "diary":                 bot_state["diary"][-100:],
        "day_pnl":               bot_state["day_pnl"],
        "total_trades":          total,
        "win_rate":              round(wins/total*100) if total > 0 else 0,
        "ai_changes":            bot_state["ai_changes"],
        "strategy":              STRATEGY,
        "account_cash":          bot_state["account_cash"],
        "account_equity":        bot_state["account_equity"],
        "account_buying_power":  bot_state["account_buying_power"],
        "market_regime":         bot_state["market_regime"],
        "strategy_b":            {
            "positions":     bot_state.get("strategy_b", {}).get("positions", {}),
            "closed_trades": bot_state.get("strategy_b", {}).get("closed_trades", [])[-50:],
            "day_pnl":       bot_state.get("strategy_b", {}).get("day_pnl", 0.0),
            "total_trades":  bot_state.get("strategy_b", {}).get("total_trades", 0),
            "win_rate":      round(bot_state.get("strategy_b", {}).get("win_count", 0) / bot_state.get("strategy_b", {}).get("total_trades", 1) * 100) if bot_state.get("strategy_b", {}).get("total_trades", 0) > 0 else 0,
        },
        "trend_1h_cache":        bot_state.get("trend_1h_cache", {}),
        "version":               "6.0",
    }
    return jsonify(clean_nan(payload))

@app.route("/killswitch", methods=["POST"])
def killswitch():
    data = request.json or {}
    bot_state["killed"] = data.get("kill", True)
    status_str = "KILLED" if bot_state["killed"] else "RESUMED"
    diary_entry("SYSTEM", f"Kill switch {status_str}", "system")
    return jsonify({"killed": bot_state["killed"], "status": status_str})

@app.route("/settings", methods=["POST"])
def update_settings():
    data    = request.json or {}
    allowed = ["stop_loss_pct","take_profit_pct","rsi_oversold","rsi_overbought",
               "bb_min_bandwidth","min_score","enabled_ema","enabled_rsi_bb",
               "enabled_macd","macd_threshold","enabled_trend_filter",
               "enabled_kronos","kronos_endpoint"]
    for k in allowed:
        if k in data:
            STRATEGY[k] = data[k]
    diary_entry("SYSTEM", f"Settings updated: {json.dumps({k:data[k] for k in allowed if k in data})}", "system")
    save_state()
    return jsonify({"ok": True, "strategy": STRATEGY})

@app.route("/revert_ai", methods=["POST"])
def revert_ai():
    for k, v in STRATEGY_ORIGINAL.items():
        STRATEGY[k] = v
    bot_state["ai_changes"] = []
    diary_entry("SYSTEM", "AI strategy reverted.", "system")
    save_state()
    return jsonify({"ok": True, "strategy": STRATEGY})

@app.route("/diary")
def get_diary():
    return jsonify({"diary": bot_state["diary"]})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat(), "version": "6.0"})

@app.route("/history")
def get_history():
    try:
        base = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
        r = requests.get(
            base + "/v2/orders?status=closed&limit=200&direction=desc",
            headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET},
            timeout=5
        )
        if r.status_code != 200:
            return jsonify({"trades": [], "error": r.status_code})
        orders = r.json()
        from collections import defaultdict
        buys = defaultdict(list)
        sells = defaultdict(list)
        for o in orders:
            if not o.get("filled_at") or not o.get("filled_avg_price"):
                continue
            try:
                filled_at = datetime.fromisoformat(o["filled_at"].replace("Z","+00:00"))
                entry = {
                    "price": float(o["filled_avg_price"]),
                    "qty":   float(o.get("filled_qty", 0)),
                    "time":  filled_at.strftime("%Y-%m-%d %H:%M"),
                    "symbol": o["symbol"],
                }
                if o["side"] == "buy":
                    buys[o["symbol"]].append(entry)
                else:
                    sells[o["symbol"]].append(entry)
            except:
                pass
        paired = []
        for sym in set(list(buys.keys()) + list(sells.keys())):
            sym_buys  = sorted(buys[sym],  key=lambda x: x["time"])
            sym_sells = sorted(sells[sym], key=lambda x: x["time"])
            bi = si = 0
            while bi < len(sym_buys) and si < len(sym_sells):
                buy  = sym_buys[bi]
                sell = sym_sells[si]
                qty  = min(buy["qty"], sell["qty"])
                pnl  = (sell["price"] - buy["price"]) * qty
                pct  = (sell["price"] - buy["price"]) / buy["price"] * 100
                paired.append({
                    "symbol":     sym,
                    "entry":      buy["price"],
                    "exit":       sell["price"],
                    "qty":        round(qty, 6),
                    "pnl":        round(pnl, 2),
                    "pct":        round(pct, 2),
                    "win":        pnl > 0,
                    "open_time":  buy["time"],
                    "close_time": sell["time"],
                })
                bi += 1
                si += 1
        paired.sort(key=lambda x: x["close_time"], reverse=True)
        return jsonify({"trades": paired})
    except Exception as e:
        log.error(f"History error: {e}")
        return jsonify({"trades": [], "error": str(e)})

@app.route("/backtest", methods=["POST"])
def backtest():
    """Run strategy backtest on historical data"""
    try:
        data = request.json or {}
        symbol   = data.get("symbol", "BTC/USD")
        start_dt = data.get("start", "2026-02-01")
        end_dt   = data.get("end", "2026-02-28")
        sl_pct   = float(data.get("stop_loss", STRATEGY["stop_loss_pct"]))
        tp_pct   = float(data.get("take_profit", STRATEGY["take_profit_pct"]))

        from datetime import timezone
        start = datetime.fromisoformat(start_dt).replace(tzinfo=timezone.utc)
        end   = datetime.fromisoformat(end_dt).replace(tzinfo=timezone.utc)

        data_client = get_data_client()
        req  = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end, limit=10000)
        bars = data_client.get_crypto_bars(req)
        df   = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        df = df[["timestamp","open","high","low","close","volume"]].copy()
        df.set_index("timestamp", inplace=True)

        if len(df) < 50:
            return jsonify({"error": "Not enough data", "trades": []})

        close = df["close"]
        ema_fast  = compute_ema(close, STRATEGY["ema_fast"])
        ema_slow  = compute_ema(close, STRATEGY["ema_slow"])
        ema_trend = compute_ema(close, STRATEGY["ema_trend"])
        rsi       = compute_rsi(close, STRATEGY["rsi_period"])
        _, _, histogram = compute_macd(close)

        trades = []
        position = None
        starting_cash = 10000.0
        cash = starting_cash

        for i in range(50, len(df)):
            price = float(close.iloc[i])
            ts    = str(df.index[i])[:16]

            # Check exit
            if position:
                pnl_pct = (price - position["entry"]) / position["entry"] * 100
                if pnl_pct >= tp_pct:
                    pnl = (price - position["entry"]) * position["qty"]
                    cash += position["qty"] * price
                    trades.append({"symbol": symbol, "entry": position["entry"], "exit": price,
                                   "qty": position["qty"], "pnl": round(pnl,2), "pct": round(pnl_pct,2),
                                   "win": True, "open_time": position["time"], "close_time": ts,
                                   "signal": f"Take profit ({pnl_pct:.2f}%)"})
                    position = None
                    continue
                elif pnl_pct <= -sl_pct:
                    pnl = (price - position["entry"]) * position["qty"]
                    cash += position["qty"] * price
                    trades.append({"symbol": symbol, "entry": position["entry"], "exit": price,
                                   "qty": position["qty"], "pnl": round(pnl,2), "pct": round(pnl_pct,2),
                                   "win": False, "open_time": position["time"], "close_time": ts,
                                   "signal": f"Stop loss ({pnl_pct:.2f}%)"})
                    position = None
                    continue

            # Check entry
            if not position:
                trend_bull = ema_fast.iloc[i] > ema_slow.iloc[i]
                trend_prev = ema_fast.iloc[i-1] > ema_slow.iloc[i-1]
                above_50   = price > ema_trend.iloc[i]
                rsi_val    = float(rsi.iloc[i])
                macd_hist  = float(histogram.iloc[i])
                bb_up, _, bb_lo, bb_bw = compute_bollinger(close.iloc[max(0,i-30):i+1])

                buy_score = 0
                if trend_bull and ema_fast.iloc[i] > ema_slow.iloc[i]: buy_score += 2
                if trend_bull and not trend_prev: buy_score += 1
                if rsi_val < STRATEGY["rsi_oversold"]: buy_score += 2
                if price < float(bb_lo.iloc[-1]) and bb_bw >= STRATEGY["bb_min_bandwidth"]: buy_score += 1
                if macd_hist > 0: buy_score += 1

                if buy_score >= STRATEGY["min_score"] and above_50 and cash >= price * 0.01:
                    qty = round((cash * 0.25) / price, 6)
                    if qty > 0:
                        cash -= qty * price
                        position = {"entry": price, "qty": qty, "time": ts}

        # Close any open position at end
        if position:
            price = float(close.iloc[-1])
            pnl_pct = (price - position["entry"]) / position["entry"] * 100
            pnl = (price - position["entry"]) * position["qty"]
            trades.append({"symbol": symbol, "entry": position["entry"], "exit": price,
                           "qty": position["qty"], "pnl": round(pnl,2), "pct": round(pnl_pct,2),
                           "win": pnl > 0, "open_time": position["time"], "close_time": "end",
                           "signal": "End of period"})

        total_pnl  = sum(t["pnl"] for t in trades)
        wins       = sum(1 for t in trades if t["win"])
        win_rate   = round(wins/len(trades)*100) if trades else 0

        return jsonify({
            "symbol":     symbol,
            "start":      start_dt,
            "end":        end_dt,
            "trades":     trades,
            "total_trades": len(trades),
            "wins":       wins,
            "losses":     len(trades) - wins,
            "win_rate":   win_rate,
            "total_pnl":  round(total_pnl, 2),
            "starting_cash": starting_cash,
            "final_cash": round(cash + (position["qty"] * float(close.iloc[-1]) if position else 0), 2),
        })
    except Exception as e:
        log.error(f"Backtest error: {e}")
        return jsonify({"error": str(e), "trades": []})

@app.route("/bars")
def get_bars_route():
    sym = request.args.get("symbol", "BTC/USD")
    tf  = request.args.get("timeframe", "5")
    try:
        data_client = get_data_client()
        limit = 200 if tf == "5" else 120
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=12)
        req   = CryptoBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=limit
        )
        bars  = data_client.get_crypto_bars(req)
        df    = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == sym].copy()
        df = df[["timestamp","open","high","low","close","volume"]].copy()
        result = []
        for _, row in df.iterrows():
            result.append({
                "time":   int(row["timestamp"].timestamp()),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
            })
        return jsonify({"bars": result})
    except Exception as e:
        log.error(f"Bars fetch error: {e}")
        return jsonify({"bars": [], "error": str(e)})

@app.route("/diary/dates")
def get_diary_dates():
    """List all available diary dates"""
    try:
        os.makedirs(DIARY_DIR, exist_ok=True)
        files = sorted(os.listdir(DIARY_DIR), reverse=True)
        dates = [f.replace(".json", "") for f in files if f.endswith(".json")]
        return jsonify({"dates": dates})
    except Exception as e:
        return jsonify({"dates": [], "error": str(e)})

@app.route("/diary/<date>")
def get_diary_by_date(date):
    """Get diary entries for a specific date"""
    try:
        filepath = f"{DIARY_DIR}/{date}.json"
        if os.path.exists(filepath):
            with open(filepath) as f:
                return jsonify(json.load(f))
        return jsonify({"date": date, "entries": []})
    except Exception as e:
        return jsonify({"date": date, "entries": [], "error": str(e)})

@app.route("/toggle", methods=["POST"])
def toggle_symbol():
    data = request.json or {}
    symbol = data.get("symbol", "")
    enabled = data.get("enabled", True)
    if symbol in ENABLED_SYMBOLS:
        ENABLED_SYMBOLS[symbol] = enabled
        status = "enabled" if enabled else "disabled"
        diary_entry("SYSTEM", f"{symbol} trading {status}", "system")
        log.info(f"{symbol} trading {status}")
        return jsonify({"ok": True, "symbol": symbol, "enabled": enabled})
    return jsonify({"ok": False, "error": "Unknown symbol"})

@app.route("/toggles")
def get_toggles():
    return jsonify({"toggles": ENABLED_SYMBOLS})

@app.route("/prices")
def get_prices():
    """Real-time prices from WebSocket or latest bar"""
    prices = {}
    for sym in SYMBOLS:
        sym_key = sym.replace("/", "")
        if sym_key in live_prices:
            prices[sym] = live_prices[sym_key]
        elif sym_key in bot_state["positions"]:
            prices[sym] = bot_state["positions"][sym_key].get("current_price", 0)
    return jsonify({"prices": prices, "time": datetime.now().isoformat()})

def start_websocket():
    """Connect to Alpaca WebSocket for real-time crypto prices"""
    import websocket as ws_lib
    import threading

    def on_message(ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, list):
                for item in data:
                    if item.get("T") == "t":  # trade update
                        sym = item.get("S", "").replace("/", "")
                        price = item.get("p", 0)
                        if sym and price:
                            live_prices[sym] = float(price)
        except Exception as e:
            log.error(f"WebSocket message error: {e}")

    def on_open(ws):
        log.info("WebSocket connected to Alpaca")
        # Authenticate
        ws.send(json.dumps({"action": "auth", "key": API_KEY, "secret": API_SECRET}))
        # Subscribe to trades for all symbols
        syms = [s.replace("/", "") for s in SYMBOLS]
        ws.send(json.dumps({"action": "subscribe", "trades": syms}))

    def on_error(ws, error):
        log.error(f"WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        log.warning("WebSocket closed, reconnecting in 5s...")
        time.sleep(5)
        start_websocket()

    def run():
        url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        ws = ws_lib.WebSocketApp(url,
            on_message=on_message,
            on_open=on_open,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    log.info("WebSocket thread started")

@app.route("/")
def index():
    try:
        return open("/app/index.html").read()
    except Exception:
        return open("index.html").read()

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    # Start WebSocket for real-time prices
    try:
        import websocket
        start_websocket()
    except ImportError:
        log.warning("websocket-client not installed, skipping WebSocket feed")
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
