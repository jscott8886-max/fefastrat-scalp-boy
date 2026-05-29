"""
ScalpAI Bot 2 - Fefa MSS Strategy
Market Structure Shift + Key Levels + 1H Trend Bias
"""
import os, time, logging, json, requests, math
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
SYMBOLS    = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD"]
STATE_FILE = "/tmp/fefa_state.json"

STRATEGY = {
    "stop_loss_pct":    1.5,
    "take_profit_pct":  3.0,
    "position_size":    0.25,   # 25% of cash per trade
    "min_sl_pct":       0.2,    # minimum stop loss distance
    "max_sl_pct":       2.0,    # maximum stop loss distance
    "swing_lookback":   3,      # candles each side to confirm swing point
    "key_level_tolerance": 0.01, # 1% tolerance for key level proximity
    "enabled":          True,
}

bot_state = {
    "running":          True,
    "killed":           False,
    "positions":        {},
    "closed_trades":    [],
    "diary":            [],
    "day_pnl":          0.0,
    "total_trades":     0,
    "win_count":        0,
    "account_cash":     0.0,
    "account_equity":   0.0,
    "account_buying_power": 0.0,
    "trend_1h_cache":   {},
    "trend_1h_time":    {},
    "key_levels_cache": {},
    "key_levels_time":  {},
    "signals":          {},
}

# ── Persistence ────────────────────────────────────────────────────────────────
def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "diary":         bot_state["diary"][-200:],
                "closed_trades": bot_state["closed_trades"][-100:],
                "day_pnl":       bot_state["day_pnl"],
                "total_trades":  bot_state["total_trades"],
                "win_count":     bot_state["win_count"],
            }, f)
    except Exception as e:
        log.error(f"Save state error: {e}")

def diary_entry(symbol, text, entry_type="trade"):
    bot_state["diary"].append({
        "time":   datetime.now().strftime("%H:%M"),
        "symbol": symbol,
        "text":   text,
        "type":   entry_type,
    })
    save_state()

# ── Alpaca helpers ─────────────────────────────────────────────────────────────
def get_account_data():
    try:
        base = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
        r = requests.get(base + "/v2/account",
            headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            bot_state["account_cash"]         = float(d.get("cash", 0))
            bot_state["account_equity"]       = float(d.get("equity", 0))
            bot_state["account_buying_power"] = float(d.get("buying_power", 0))
    except Exception as e:
        log.error(f"Account fetch error: {e}")

def sync_positions_from_alpaca():
    try:
        base = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
        r = requests.get(base + "/v2/positions",
            headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}, timeout=5)
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
                        dt = datetime.fromisoformat(open_at.replace("Z", "+00:00"))
                        open_time = dt.strftime("%H:%M")
                    except:
                        open_time = datetime.now().strftime("%H:%M")
                    bot_state["positions"][sym] = {
                        "entry":          float(pos["avg_entry_price"]),
                        "qty":            float(pos["qty"]),
                        "open_time":      open_time,
                        "symbol":         sym_slash,
                        "current_price":  float(pos.get("current_price", pos["avg_entry_price"])),
                        "unrealized_pnl": float(pos.get("unrealized_pl", 0)),
                        "unrealized_pct": float(pos.get("unrealized_plpc", 0)) * 100,
                    }
                else:
                    bot_state["positions"][sym]["current_price"]  = float(pos.get("current_price", pos["avg_entry_price"]))
                    bot_state["positions"][sym]["unrealized_pnl"] = float(pos.get("unrealized_pl", 0))
                    bot_state["positions"][sym]["unrealized_pct"] = float(pos.get("unrealized_plpc", 0)) * 100
                    bot_state["positions"][sym]["qty"]            = float(pos["qty"])
            for sym in list(bot_state["positions"].keys()):
                if sym not in live_syms:
                    del bot_state["positions"][sym]
    except Exception as e:
        log.error(f"Position sync error: {e}")

def get_data_client():
    return CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

def get_trading_client():
    return TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER_MODE)

# ── Market Structure ───────────────────────────────────────────────────────────
def detect_swing_points(close, lookback=3):
    prices = close.values
    n = len(prices)
    swing_highs, swing_lows = [], []
    for i in range(lookback, n - lookback):
        if all(prices[i] > prices[i-j] for j in range(1, lookback+1)) and \
           all(prices[i] > prices[i+j] for j in range(1, lookback+1)):
            swing_highs.append((i, float(prices[i])))
        if all(prices[i] < prices[i-j] for j in range(1, lookback+1)) and \
           all(prices[i] < prices[i+j] for j in range(1, lookback+1)):
            swing_lows.append((i, float(prices[i])))
    return swing_highs, swing_lows

def get_1h_trend(data_client, symbol):
    """Get 1H market structure trend: BULL, BEAR, or NEUTRAL"""
    try:
        req = CryptoBarsRequest(symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour), limit=50)
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

def get_key_levels(data_client, symbol):
    """Get key S/R levels from 1H swing points"""
    try:
        req = CryptoBarsRequest(symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour), limit=48)
        bars = data_client.get_crypto_bars(req)
        df = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        close = df["close"]
        swing_highs, swing_lows = detect_swing_points(close, lookback=2)
        levels = [h[1] for h in swing_highs[-4:]] + [l[1] for l in swing_lows[-4:]]
        return levels
    except:
        return []

def detect_mss(data_client, symbol, trend_1h):
    """
    Detect Market Structure Shift on 5M chart.
    BULL trend: look for higher high after series of lower lows = BUY signal
    BEAR trend: look for lower low after series of higher highs = SELL signal (skip - no shorting)
    """
    try:
        req  = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=60)
        bars = data_client.get_crypto_bars(req)
        df   = bars.df.reset_index()
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
        current_price = float(prices[-1])

        swing_highs, swing_lows = detect_swing_points(close, lookback=STRATEGY["swing_lookback"])

        sig_data = {
            "price":    round(current_price, 4),
            "trend_1h": trend_1h,
            "strategy": "Fefa MSS",
        }

        if trend_1h == "BULL" and len(swing_lows) >= 2 and len(swing_highs) >= 1:
            last_low1     = swing_lows[-1][1]
            last_low2     = swing_lows[-2][1]
            last_high     = swing_highs[-1][1]
            last_high_idx = swing_highs[-1][0]

            # MSS: previous low was lower, now we have a higher low = structure shifting up
            if last_low1 > last_low2 and last_high_idx > swing_lows[-1][0]:
                # Check proximity to key level
                key_levels = bot_state["key_levels_cache"].get(symbol.replace("/",""), [])
                near_level = any(abs(current_price - lv) / lv < STRATEGY["key_level_tolerance"]
                                for lv in key_levels) if key_levels else True

                if near_level or len(key_levels) == 0:
                    stop_loss_price = float(min(lows[-8:]))
                    sl_pct = abs(current_price - stop_loss_price) / current_price * 100

                    # Only trade if stop loss is within acceptable range
                    if STRATEGY["min_sl_pct"] <= sl_pct <= STRATEGY["max_sl_pct"]:
                        sig_data.update({
                            "mss_type":         "Bullish MSS",
                            "stop_loss_price":  round(stop_loss_price, 4),
                            "sl_pct":           round(sl_pct, 3),
                            "near_key_level":   near_level,
                            "prev_low":         round(last_low2, 4),
                            "curr_low":         round(last_low1, 4),
                        })
                        log.info(f"{symbol} | BULL MSS detected | price={current_price:.2f} SL={stop_loss_price:.2f} ({sl_pct:.2f}%)")
                        return "BUY", sig_data

        sig_data["mss_type"] = "No MSS"
        return "HOLD", sig_data

    except Exception as e:
        log.error(f"MSS detection error {symbol}: {e}")
        return "HOLD", {"price": 0, "trend_1h": trend_1h}

# ── Trading Loop ───────────────────────────────────────────────────────────────
def trading_loop():
    if not API_KEY or not API_SECRET:
        log.warning("No API keys — bot idle")
        return

    trading_client = get_trading_client()
    data_client    = get_data_client()

    # Clear stale state
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        log.info("Cleared stale state")

    get_account_data()
    sync_positions_from_alpaca()

    log.info(f"Fefa MSS Bot started | Paper={PAPER_MODE}")
    diary_entry("SYSTEM",
        f"Fefa MSS Bot started | SL={STRATEGY['stop_loss_pct']}% | TP={STRATEGY['take_profit_pct']}% | "
        f"Size={int(STRATEGY['position_size']*100)}% per trade", "system")

    while True:
        try:
            if bot_state["killed"]:
                time.sleep(5)
                continue

            get_account_data()
            sync_positions_from_alpaca()
            data_client = get_data_client()

            cash      = bot_state["account_cash"]
            per_trade = cash * STRATEGY["position_size"] if cash > 0 else 2500
            now       = datetime.now()

            for symbol in SYMBOLS:
                sym_key = symbol.replace("/", "")

                # Update 1H trend every 15 minutes
                last_trend = bot_state["trend_1h_time"].get(sym_key)
                if last_trend is None or (now - last_trend).seconds >= 900:
                    trend = get_1h_trend(data_client, symbol)
                    bot_state["trend_1h_cache"][sym_key] = trend
                    bot_state["trend_1h_time"][sym_key]  = now
                    log.info(f"{symbol} 1H trend: {trend}")
                else:
                    trend = bot_state["trend_1h_cache"].get(sym_key, "NEUTRAL")

                # Update key levels every 30 minutes
                last_levels = bot_state["key_levels_time"].get(sym_key)
                if last_levels is None or (now - last_levels).seconds >= 1800:
                    levels = get_key_levels(data_client, symbol)
                    bot_state["key_levels_cache"][sym_key] = levels
                    bot_state["key_levels_time"][sym_key]  = now

                # Skip neutral trend
                if trend == "NEUTRAL":
                    bot_state["signals"][sym_key] = {"trend_1h": "NEUTRAL", "mss_type": "Waiting for trend"}
                    continue

                # Detect MSS signal
                signal, sig_data = detect_mss(data_client, symbol, trend)
                bot_state["signals"][sym_key] = sig_data

                in_position = sym_key in bot_state["positions"]

                if in_position:
                    pos     = bot_state["positions"][sym_key]
                    price   = pos.get("current_price", pos["entry"])
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
                        exit_reason = "MSS reversal signal"

                    if should_exit:
                        try:
                            req = MarketOrderRequest(symbol=symbol, qty=pos["qty"],
                                side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                            trading_client.submit_order(req)
                            pnl = (price - pos["entry"]) * pos["qty"]
                            win = pnl > 0
                            bot_state["closed_trades"].append({
                                "symbol": symbol, "entry": pos["entry"], "exit": price,
                                "qty": pos["qty"], "pnl": round(pnl,2), "pct": round(pnl_pct,2),
                                "win": win, "time": pos["open_time"],
                                "close_time": now.strftime("%H:%M"), "signal": exit_reason
                            })
                            bot_state["day_pnl"]       = round(bot_state["day_pnl"] + pnl, 2)
                            bot_state["total_trades"] += 1
                            if win: bot_state["win_count"] += 1
                            del bot_state["positions"][sym_key]
                            diary_entry(symbol,
                                f"{'WIN' if win else 'LOSS'} | ${pos['entry']:,.2f} → ${price:,.2f} | "
                                f"P&L ${pnl:.2f} ({pnl_pct:+.2f}%) | {exit_reason}",
                                "win" if win else "loss")
                            save_state()
                        except Exception as e:
                            log.error(f"Exit order failed {symbol}: {e}")

                elif signal == "BUY" and not bot_state["killed"]:
                    price = sig_data.get("price", 0)
                    if price > 0 and cash > 100:
                        try:
                            qty = round(per_trade / price, 6)
                            if qty > 0:
                                req = MarketOrderRequest(symbol=symbol, qty=qty,
                                    side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                trading_client.submit_order(req)
                                bot_state["positions"][sym_key] = {
                                    "entry":    price, "qty": qty,
                                    "open_time": now.strftime("%H:%M"), "symbol": symbol
                                }
                                diary_entry(symbol,
                                    f"BUY | ${price:,.2f} | Qty {qty} | Budget ${per_trade:,.2f} | "
                                    f"1H: {trend} | {sig_data.get('mss_type','MSS')} | "
                                    f"SL: ${sig_data.get('stop_loss_price',0):,.2f} ({sig_data.get('sl_pct',0):.2f}%)",
                                    "trade")
                                save_state()
                        except Exception as e:
                            log.error(f"Entry order failed {symbol}: {e}")

            time.sleep(60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Loop error: {e}")
            time.sleep(30)

# ── JSON helper ────────────────────────────────────────────────────────────────
def clean_nan(obj):
    if obj is None: return None
    if isinstance(obj, datetime): return obj.isoformat()
    if hasattr(obj, '__module__') and type(obj).__module__ == 'numpy':
        try: obj = obj.item()
        except: return 0
    if isinstance(obj, bool): return obj
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, int): return obj
    if isinstance(obj, str): return obj
    if isinstance(obj, dict): return {str(k): clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean_nan(v) for v in obj]
    try: return str(obj)
    except: return None

# ── Flask API ──────────────────────────────────────────────────────────────────
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
        "running":              bot_state["running"],
        "killed":               bot_state["killed"],
        "paper_mode":           PAPER_MODE,
        "positions":            bot_state["positions"],
        "closed_trades":        bot_state["closed_trades"][-50:],
        "diary":                bot_state["diary"][-100:],
        "day_pnl":              bot_state["day_pnl"],
        "total_trades":         total,
        "win_rate":             round(wins/total*100) if total > 0 else 0,
        "strategy":             STRATEGY,
        "signals":              bot_state["signals"],
        "trend_1h":             bot_state["trend_1h_cache"],
        "account_cash":         bot_state["account_cash"],
        "account_equity":       bot_state["account_equity"],
        "account_buying_power": bot_state["account_buying_power"],
        "version":              "Fefa-MSS-1.0",
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
    allowed = ["stop_loss_pct","take_profit_pct","position_size",
               "swing_lookback","key_level_tolerance"]
    for k in allowed:
        if k in data:
            STRATEGY[k] = data[k]
    diary_entry("SYSTEM", f"Settings updated", "system")
    return jsonify({"ok": True, "strategy": STRATEGY})

@app.route("/diary")
def get_diary():
    return jsonify({"diary": bot_state["diary"]})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat(), "version": "Fefa-MSS-1.0"})

@app.route("/history")
def get_history():
    try:
        base = "https://paper-api.alpaca.markets" if PAPER_MODE else "https://api.alpaca.markets"
        r = requests.get(base + "/v2/orders?status=closed&limit=200&direction=desc",
            headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}, timeout=5)
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
                entry = {"price": float(o["filled_avg_price"]), "qty": float(o.get("filled_qty",0)),
                         "time": filled_at.strftime("%Y-%m-%d %H:%M"), "symbol": o["symbol"]}
                if o["side"] == "buy": buys[o["symbol"]].append(entry)
                else: sells[o["symbol"]].append(entry)
            except: pass
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
                    "symbol": sym, "entry": buy["price"], "exit": sell["price"],
                    "qty": round(qty,6), "pnl": round(pnl,2), "pct": round(pct,2),
                    "win": pnl > 0, "open_time": buy["time"], "close_time": sell["time"],
                })
                bi += 1; si += 1
        paired.sort(key=lambda x: x["close_time"], reverse=True)
        return jsonify({"trades": paired})
    except Exception as e:
        return jsonify({"trades": [], "error": str(e)})

@app.route("/bars")
def get_bars_route():
    sym = request.args.get("symbol", "BTC/USD")
    tf  = request.args.get("timeframe", "5")
    try:
        data_client = get_data_client()
        limit = 200 if tf == "5" else 120
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=12)
        req   = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                  start=start, end=end, limit=limit)
        bars  = data_client.get_crypto_bars(req)
        df    = bars.df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"] == sym].copy()
        df = df[["timestamp","open","high","low","close","volume"]].copy()
        result = [{"time": int(r["timestamp"].timestamp()), "open": float(r["open"]),
                   "high": float(r["high"]), "low": float(r["low"]),
                   "close": float(r["close"]), "volume": float(r["volume"])}
                  for _, r in df.iterrows()]
        return jsonify({"bars": result})
    except Exception as e:
        return jsonify({"bars": [], "error": str(e)})

@app.route("/prices")
def get_prices():
    prices = {}
    for sym in SYMBOLS:
        sym_key = sym.replace("/", "")
        if sym_key in bot_state["positions"]:
            prices[sym] = bot_state["positions"][sym_key].get("current_price", 0)
    return jsonify({"prices": prices, "time": datetime.now().isoformat()})

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
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
Done
Select all of that output above (from the """ at the top to the last line app.run(host="0.0.0.0", port=port)) and paste it into the GitHub editor. Then click "Commit changes". Let me know when done!






