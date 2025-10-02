# main.py
import os
import threading
import asyncio
import logging
from datetime import datetime, timezone
import ccxt
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask
from telegram import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

# Load environment variables
load_dotenv()

# --------------------
# Config
# --------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
CHAT_ID = os.getenv("CHAT_ID")
BASE_CURRENCY = os.getenv("BASE_CURRENCY", "USDT")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "120"))
ALTS_LIST = [s.strip().upper() for s in os.getenv("ALTS_LIST", "BTC,ETH,AVAX,PEPE").split(",")]
MIN_PRICE_LESS_THAN = float(os.getenv("MIN_PRICE_LESS_THAN", "1.0"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger("crypto-bot")

# --------------------
# Keepalive server (optional for uptime monitoring)
# --------------------
app = Flask("keepalive")

@app.route("/")
def home():
    return "OK - Crypto Bot is Alive"

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# --------------------
# Binance setup
# --------------------
exchange = ccxt.binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "enableRateLimit": True,
    "options": {"adjustForTimeDifference": True}
})

# --------------------
# Technical Analysis helpers
# --------------------
def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 300):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","vol"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        logger.warning(f"fetch_ohlcv {symbol} failed: {e}")
        return None

def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ef = ema(series, fast)
    es = ema(series, slow)
    macd_line = ef - es
    sig = ema(macd_line, signal)
    hist = macd_line - sig
    return macd_line, sig, hist

def atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def format_price(p: float):
    p = float(p)
    if p >= 1:
        return f"{p:.2f}"
    elif p >= 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.6f}"

# --------------------
# Signal generator
# --------------------
def generate_signal(symbol_pair: str):
    df1h = fetch_ohlcv(symbol_pair, timeframe="1h", limit=300)
    df4h = fetch_ohlcv(symbol_pair, timeframe="4h", limit=200)
    if df1h is None or df1h.empty:
        return {"error": "No data for " + symbol_pair}

    close = df1h["close"]
    last = float(close.iloc[-1])

    # indicators
    ema20 = ema(close, 20).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    rsi1h = float(rsi(close, 14).iloc[-1])
    _, _, macd_hist = macd(close)
    macd_now = float(macd_hist.iloc[-1])
    atr14 = float(atr(df1h, 14).iloc[-1] if not atr(df1h, 14).isna().all() else 0.0)
    atr_unit = atr14 if atr14 > 0 else max(0.0001, last * 0.005)
    atr_pct = (atr_unit / last * 100) if last>0 else 0.0

    # trend 4h
    trend_bull_4h = False
    trend_bear_4h = False
    if df4h is not None and not df4h.empty:
        e20_4h = ema(df4h["close"], 20).iloc[-1]
        e50_4h = ema(df4h["close"], 50).iloc[-1]
        trend_bull_4h = e20_4h > e50_4h
        trend_bear_4h = e20_4h < e50_4h

    long_cond = (last > ema20) and (ema20 > ema50) and (rsi1h > 40 and rsi1h < 80) and (macd_now > 0)
    short_cond = (last < ema20) and (ema20 < ema50) and (rsi1h < 60 and rsi1h > 10) and (macd_now < 0)

    if trend_bull_4h:
        long_cond = long_cond or (ema20 > ema50 and rsi1h > 35)
    if trend_bear_4h:
        short_cond = short_cond or (ema20 < ema50 and rsi1h < 65)

    entry_low = last - 0.5 * atr_unit
    entry_high = last + 0.5 * atr_unit

    t1 = entry_high + 0.5 * atr_unit
    t2 = entry_high + 1.0 * atr_unit
    t3 = entry_high + 1.5 * atr_unit
    t4 = entry_high + 2.0 * atr_unit
    t5 = entry_high + 2.8 * atr_unit
    t6 = entry_high + 4.0 * atr_unit

    lookback = min(60, len(df1h))
    swing_low = float(df1h["low"].tail(lookback).min())
    swing_high = float(df1h["high"].tail(lookback).max())
    sl_long = max(swing_low - 0.5 * atr_unit, entry_low - 2 * atr_unit)
    sl_short = min(swing_high + 0.5 * atr_unit, entry_high + 2 * atr_unit)

    leverage = "20X-25X" if atr_pct <= 0.6 else "10X-20X" if atr_pct <= 1.5 else "5X-10X"

    signal = "WAIT"
    if long_cond and not short_cond:
        signal = "LONG"
    elif short_cond and not long_cond:
        signal = "SHORT"

    def fmt(v): return format_price(float(v))
    targets_str = " | ".join(fmt(x) for x in [t1,t2,t3,t4,t5,t6]) + " |"

    out = {
        "symbol": symbol_pair,
        "signal": signal,
        "leverage": leverage,
        "entry_low": fmt(entry_low),
        "entry_high": fmt(entry_high),
        "targets_str": targets_str,
        "stoploss": fmt(sl_long if signal=="LONG" else sl_short if signal=="SHORT" else sl_long),
        "price": fmt(last),
        "atr": fmt(atr_unit),
        "atr_pct": round(atr_pct,3),
        "rsi": round(rsi1h,2),
        "ema20": fmt(ema20),
        "ema50": fmt(ema50),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    return out

# --------------------
# Format messages
# --------------------
def format_signal_message(d: dict):
    if "error" in d:
        return f"{d.get('symbol','?')}: Error - {d.get('error')}"
    dir_text = "( Long )" if d["signal"]=="LONG" else "( Short )" if d["signal"]=="SHORT" else "( Wait )"
    sym = d["symbol"].replace("/", "/")
    lines = [
        f"#{sym}  {dir_text}",
        f"Leverage {d['leverage']}",
        f"Entry  {d['entry_low']} - {d['entry_high']}",
        f"Targets :- {d['targets_str']}",
        f"Stoploss = {d['stoploss']}",
        f"_Price_: {d['price']}  ATR: {d['atr']} ({d['atr_pct']}%)  RSI: {d['rsi']}",
        f"_Timestamp_: {d['timestamp']}"
    ]
    return "\n".join(lines)

# --------------------
# Telegram Handlers
# --------------------
SUBSCRIBERS = set()

async def start(update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    SUBSCRIBERS.add(user.id)
    await update.message.reply_text("Subscribed to signals. Use /signal SYMBOL to get one analysis.")

async def unsubscribe(update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    SUBSCRIBERS.discard(user.id)
    await update.message.reply_text("Unsubscribed.")

async def signal_cmd(update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /signal SYMBOL (eg: /signal AVAX/USDT)")
        return
    sym = args[0].upper()
    if "/" not in sym:
        sym = f"{sym}/{BASE_CURRENCY}"
    d = await asyncio.to_thread(generate_signal, sym)
    msg = format_signal_message(d)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def subscribers_cmd(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Subscribers count: {len(SUBSCRIBERS)}")

async def scan_and_send(context: ContextTypes.DEFAULT_TYPE):
    symbols = []
    for t in ALTS_LIST:
        sym = f"{t}/{BASE_CURRENCY}"
        try:
            tk = exchange.fetch_ticker(sym)
            last = float(tk.get("last",0) or 0)
            if last <= MIN_PRICE_LESS_THAN:
                symbols.append(sym)
        except Exception:
            continue

    messages = []
    for s in symbols:
        d = await asyncio.to_thread(generate_signal, s)
        if d.get("signal") in ("LONG","SHORT"):
            messages.append(format_signal_message(d))

    if not messages:
        messages = ["*Scan Complete*\nNo strong LONG/SHORT signals found this cycle."]

    app_telegram = context.application
    for chat_id in list(SUBSCRIBERS):
        for m in messages:
            try:
                await app_telegram.bot.send_message(chat_id=chat_id, text=m, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                continue

# --------------------
# Run Bot
# --------------------
async def main():
    if not TELEGRAM_TOKEN:
        logger.error("Missing TELEGRAM_BOT_TOKEN env variable.")
        return
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))
    application.add_handler(CommandHandler("signal", signal_cmd))
    application.add_handler(CommandHandler("subscribers", subscribers_cmd))

    job_queue: JobQueue = application.job_queue
    job_queue.run_repeating(scan_and_send, interval=POLL_INTERVAL, first=15)

    logger.info("Bot starting...")
    await application.run_polling()

if __name__ == "__main__":
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
