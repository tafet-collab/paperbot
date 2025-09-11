#!/usr/bin/env python3
# bot.py — Risk-first DAILY momentum bot with CPPI/lockbox and paper trading
# Modes:
#   python bot.py backtest --symbol BTC-USD --start 200
#   python bot.py paper --symbol BTC-USD --start 10
#   python bot.py reset --symbol BTC-USD

import argparse, os, json, math
import datetime as dt
from dateutil import tz
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("Install dependencies first: pip install -r requirements.txt")
    raise

DATA_DAYS = 800
FEE_BP = 10  # 0.10% per side
SLIP_BP = 5  # 0.05% per side


def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=tz.tzutc())


def sma(s, n): return s.rolling(n).mean()


def atr(df, n=20):
    c = df["Close"].astype(float)
    h = df["High"].astype(float) if "High" in df.columns else c
    l = df["Low"].astype(float) if "Low" in df.columns else c
    prev_c = c.shift()
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def load_history(symbol: str, days: int = DATA_DAYS) -> pd.DataFrame:
    period = f"{days}d"
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=lambda c: str(c).title())

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]

    df = df[keep].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.dropna(how="any")
    return df


def regime_filter(df: pd.DataFrame, vol_cap_annual=1.2) -> pd.Series:
    close = df["Close"].astype(float)  # force Series
    vol20 = close.pct_change().rolling(20).std() * np.sqrt(365)
    trend_on = sma(close, 50) > sma(close, 200)
    vol_ok = vol20 < vol_cap_annual
    return (trend_on & vol_ok).reindex(df.index, fill_value=False)


def momentum_signals(df: pd.DataFrame):
    a = atr(df, 20)
    close = df["Close"].astype(float)
    hi_src = df["High"].astype(float) if "High" in df.columns else close
    ma50 = sma(close, 50)
    hi20 = hi_src.rolling(20).max()

    entry_lvl = pd.concat([hi20, ma50 + 1.5 * a], axis=1).max(axis=1)
    long_entry = close > entry_lvl
    trail_stop = close - 2 * a
    return long_entry.fillna(False), trail_stop


def update_risk_state(equity, high_water, floor, lockbox, avgDailyProfit):
    high_water = max(high_water, equity)
    floor = max(floor, lockbox)
    cushion = max(0.0, equity - floor)
    CPPI_multiplier = 3.0
    max_allocation = 0.80 * equity
    alloc = min(CPPI_multiplier * cushion, max_allocation)
    risk_per_trade = min(0.005 * equity, 0.02 * cushion)
    open_risk_cap = 0.015 * equity
    day_loss_cap = min(0.01 * equity, 2.0 * max(0.0, avgDailyProfit))
    return {"high_water": high_water, "floor": floor, "cushion": cushion,
            "alloc": alloc, "risk_per_trade": risk_per_trade,
            "open_risk_cap": open_risk_cap, "day_loss_cap": day_loss_cap}


def position_size(entry, stop, risk_cash, min_notional, price_precision=6):
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0: return 0.0
    qty = max(0.0, math.floor((risk_cash / risk_per_unit) * (10 ** price_precision)) / (10 ** price_precision))
    if qty * entry < min_notional: return 0.0
    return qty


def apply_fees_slippage(price, side: str):
    bp = (FEE_BP + SLIP_BP) / 10000.0
    return price * (1 + bp) if side == "buy" else price * (1 - bp)


def backtest(symbol="BTC-USD", start_equity=200.0):
    df = load_history(symbol)
    df = df.iloc[220:]
    long_entry, trail_stop = momentum_signals(df)
    regime = regime_filter(df)
    equity = start_equity;
    pos_qty = 0.0;
    entry_px = None
    high_water = floor = start_equity;
    lockbox = 0.0;
    avgDailyProfit = 0.0
    pnl_hist = [];
    trades = [];
    prev_mtm = equity

    for i in range(1, len(df)):
        row = df.iloc[i];
        date = df.index[i];
        px = row["Close"];
        a = atr(df, 20).iloc[i]
        risk_state = update_risk_state(
            equity if pos_qty == 0 else equity + pos_qty * (px - entry_px),
            high_water,
            floor,
            lockbox,
            avgDailyProfit,
        )
        high_water = risk_state["high_water"]
        floor = risk_state["floor"]
        risk = risk_state
        if pos_qty > 0:
            stop = trail_stop.iloc[i]
            if not np.isnan(stop) and px <= stop:
                exit_px = apply_fees_slippage(stop, "sell")
                equity += pos_qty * (exit_px - entry_px)
                trades.append(("exit_stop", date.isoformat(), float(exit_px), float(pos_qty)))
                pos_qty = 0.0;
                entry_px = None
        if pos_qty == 0 and regime.iloc[i] and long_entry.iloc[i] and not np.isnan(a):
            stop = px - 2 * a
            qty = position_size(px, stop, risk["risk_per_trade"], min_notional=5.0, price_precision=6)
            if qty > 0:
                entry = apply_fees_slippage(px, "buy")
                pos_qty = qty;
                entry_px = entry
                trades.append(("entry", date.isoformat(), float(entry), float(qty)))
        equity_mtm = equity + (pos_qty * (px - entry_px) if pos_qty > 0 else 0.0)
        high_water = max(high_water, equity_mtm)
        day_pnl = equity_mtm - prev_mtm;
        prev_mtm = equity_mtm;
        pnl_hist.append(day_pnl)
        recent = pnl_hist[-7:];
        pos = [x for x in recent if x > 0]
        avgDailyProfit = np.mean(pos) if pos else 0.0
        if equity_mtm < 0.9 * high_water and pos_qty > 0:
            exit_px = apply_fees_slippage(px, "sell")
            equity += pos_qty * (exit_px - entry_px)
            trades.append(("forced_exit_dd", date.isoformat(), float(exit_px), float(pos_qty)))
            pos_qty = 0.0;
            entry_px = None
        if date.weekday() == 6:
            gain = max(0.0, (equity_mtm - floor))
            siphon = 0.5 * gain
            lockbox += siphon;
            floor = max(floor, lockbox)
    final_equity = equity + (pos_qty * (df["Close"].iloc[-1] - entry_px) if pos_qty > 0 else 0.0)
    return {"final_equity": float(final_equity), "trades": trades, "start_equity": start_equity}


def state_path(symbol): return f"state_{symbol.replace('-', '_')}.json"


def trades_path(symbol): return f"trades_{symbol.replace('-', '_')}.csv"


def load_state(symbol, start_equity):
    path = state_path(symbol)
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    st = {"equity": start_equity, "high_water": start_equity, "floor": start_equity,
          "lockbox": 0.0, "avgDailyProfit": 0.0, "last_processed_date": None,
          "position": {"qty": 0.0, "entry": None}}
    with open(path, "w") as f:
        json.dump(st, f, indent=2)
    if not os.path.exists(trades_path(symbol)):
        pd.DataFrame(columns=["date", "side", "price", "qty", "reason"]).to_csv(trades_path(symbol), index=False)
    return st


def save_state(symbol, st):
    with open(state_path(symbol), "w") as f: json.dump(st, f, indent=2)


def append_trade(symbol, date, side, price, qty, reason):
    df = pd.DataFrame([{"date": date, "side": side, "price": price, "qty": qty, "reason": reason}])
    if os.path.exists(trades_path(symbol)):
        df.to_csv(trades_path(symbol), mode="a", index=False, header=False)
    else:
        df.to_csv(trades_path(symbol), index=False)


def paper_step(symbol="BTC-USD", start_equity=10.0):
    df = load_history(symbol, days=400)
    if len(df) < 220:
        print("Not enough data.");
        return
    long_entry, trail_stop = momentum_signals(df)
    regime = regime_filter(df)
    bar_idx = -2  # last completed day
    date = df.index[bar_idx];
    px = df["Close"].iloc[bar_idx]
    a = atr(df, 20).iloc[bar_idx];
    stop_trail = trail_stop.iloc[bar_idx]
    entry_signal = bool(regime.iloc[bar_idx] and long_entry.iloc[bar_idx])
    st = load_state(symbol, start_equity)
    prev_day = df.index[-2].date()
    if st.get("last_processed_date") == str(prev_day):
        print("Already processed", prev_day);
        return
    equity = st["equity"];
    high_water = st["high_water"];
    floor = st["floor"];
    lockbox = st["lockbox"]
    avgDailyProfit = st["avgDailyProfit"];
    pos_qty = st["position"]["qty"];
    entry_px = st["position"]["entry"]
    equity_mtm = equity + (pos_qty * (px - entry_px) if pos_qty and entry_px else 0.0)
    risk_state = update_risk_state(equity_mtm, high_water, floor, lockbox, avgDailyProfit)
    high_water = risk_state["high_water"]
    floor = risk_state["floor"]
    risk = risk_state
    if pos_qty and entry_px and not np.isnan(stop_trail) and px <= stop_trail:
        exit_px = apply_fees_slippage(stop_trail, "sell")
        equity += pos_qty * (exit_px - entry_px)
        append_trade(symbol, str(date.date()), "sell", float(exit_px), float(pos_qty), "exit_stop")
        pos_qty = 0.0;
        entry_px = None
    if (not pos_qty) and entry_signal and not np.isnan(a):
        stop = px - 2 * a
        qty = position_size(px, stop, risk["risk_per_trade"], min_notional=0.0, price_precision=6)
        if qty > 0:
            entry = apply_fees_slippage(px, "buy")
            pos_qty = qty;
            entry_px = entry
            append_trade(symbol, str(date.date()), "buy", float(entry), float(qty), "entry")
    equity_mtm2 = equity + (pos_qty * (px - entry_px) if pos_qty and entry_px else 0.0)
    high_water = max(high_water, equity_mtm2)
    alpha = 0.3;
    day_pnl = equity_mtm2 - st["equity"]
    avgDailyProfit = (1 - alpha) * avgDailyProfit + alpha * max(0.0, day_pnl)
    if date.weekday() == 6 and equity_mtm2 > floor:
        gain = equity_mtm2 - floor;
        siphon = 0.5 * gain;
        lockbox += siphon;
        floor = max(floor, lockbox)
    st.update({"equity": float(equity_mtm2), "high_water": float(high_water), "floor": float(floor),
               "lockbox": float(lockbox), "avgDailyProfit": float(avgDailyProfit),
               "last_processed_date": str(prev_day),
               "position": {"qty": float(pos_qty or 0.0), "entry": float(entry_px) if entry_px else None}})
    save_state(symbol, st)
    print(
        f"[{date.date()}] equity={equity_mtm2:.2f} pos_qty={pos_qty or 0.0:.6f} entry={entry_px} floor={floor:.2f} cushion={equity_mtm2 - floor:.2f}")


def main():
    p = argparse.ArgumentParser(description="Daily momentum paper bot with CPPI/lockbox")
    p.add_argument("mode", choices=["backtest", "paper", "reset"], help="Operation mode")
    p.add_argument("--symbol", default="BTC-USD", help="Symbol, e.g., BTC-USD or ETH-USD")
    p.add_argument("--start", type=float, default=10.0, help="Starting equity")
    args = p.parse_args()
    if args.mode == "backtest":
        res = backtest(symbol=args.symbol, start_equity=args.start)
        print(
            f"Backtest {args.symbol} start={args.start:.2f} final={res['final_equity']:.2f} trades={len(res['trades'])}")
    elif args.mode == "paper":
        paper_step(symbol=args.symbol, start_equity=args.start)
    elif args.mode == "reset":
        sp, tp = state_path(args.symbol), trades_path(args.symbol)
        if os.path.exists(sp): os.remove(sp)
        if os.path.exists(tp): os.remove(tp)
        print("State reset.")


if __name__ == "__main__":
    main()
