# jasper_full.py
"""
JASPER — Ultimate Options & Edge Analyzer (with Monte Carlo restored)
- Single-file Streamlit app
- Options screener (yfinance) + BS/Binomial pricing
- Greeks, liquidity & spread filters, earnings proximity
- Adaptive mispricing thresholds, Top-10 S&P best-contract finder
- EV estimates (MC), portfolio sizing, approximate backtests
- Monte Carlo simulation per-ticker (toggleable) with sample paths + pie chart
Notes: Analysis-only. Backtests use modelled prices (approx). No broker connectivity.
"""

import io
import math
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from scipy.stats import norm
import matplotlib.pyplot as plt

# -------------------------
# Page config & defaults
# -------------------------
st.set_page_config(page_title="JASPER — Options & Edge Analyzer", layout="wide")
DEFAULT_RFR = 0.02

# -------------------------
# Utilities
# -------------------------
def yf_symbol(sym: str) -> str:
    return sym.replace(".", "-").upper()

def flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            if isinstance(col[0], str) and col[0] != "":
                new_cols.append(col[0])
            else:
                new_cols.append("_".join([str(x) for x in col if x]))
        df = df.copy()
        df.columns = new_cols
    return df

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

# -------------------------
# Fetch helpers
# -------------------------
@st.cache_data(ttl=24*3600)
def load_sp500_tickers() -> List[str]:
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df["Symbol"].astype(str).str.upper().tolist()
    except Exception:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            r = requests.get(url, timeout=15)
            buf = io.StringIO(r.text)
            tables = pd.read_html(buf)
            df = tables[0]
            return df["Symbol"].astype(str).str.upper().tolist()
        except Exception:
            return []

def fetch_history(yf_sym: str, period: str = "1y", auto_adjust: bool = True) -> pd.DataFrame:
    """Robust fetch; ensures Close / Adj Close exist."""
    try:
        tk = yf.Ticker(yf_sym)
        df = tk.history(period=period, auto_adjust=auto_adjust)
        if df is None or df.empty:
            return pd.DataFrame()
        df = flatten_multiindex_cols(df)
        if "Close" not in df.columns and "Adj Close" not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df["Close"] = df[numeric_cols[0]]
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------
# Pricing models & greeks
# -------------------------
def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> float:
    try:
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt_type == "call":
            return float(S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2))
        else:
            return float(K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    except Exception:
        return 0.0

def binomial_crr_price(S: float, K: float, T: float, r: float, sigma: float, steps: int = 100, opt_type: str = "call", american: bool = True) -> float:
    try:
        if T <= 0:
            return max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        a = math.exp(r * dt)
        p = (a - d) / (u - d)
        prices = [S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        if opt_type == "call":
            values = [max(0.0, pS - K) for pS in prices]
        else:
            values = [max(0.0, K - pS) for pS in prices]
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                cont = math.exp(-r * dt) * (p * values[j + 1] + (1 - p) * values[j])
                if american:
                    Snode = S * (u ** j) * (d ** (i - j))
                    exercise = (max(0.0, Snode - K) if opt_type == "call" else max(0.0, K - Snode))
                    values[j] = max(cont, exercise)
                else:
                    values[j] = cont
        return float(values[0])
    except Exception:
        return 0.0

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> Dict[str, float]:
    try:
        if T <= 0 or sigma <= 0:
            delta = 1.0 if (opt_type == "call" and S > K) else (-1.0 if (opt_type == "put" and S < K) else 0.0)
            return {"delta": delta, "vega": 0.0, "theta": 0.0, "gamma": 0.0}
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = norm.cdf(d1) if opt_type == "call" else (norm.cdf(d1) - 1.0)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if opt_type == "call":
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * norm.cdf(-d2))
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return {"delta": float(delta), "vega": float(vega), "theta": float(theta), "gamma": float(gamma)}
    except Exception:
        return {"delta": np.nan, "vega": np.nan, "theta": np.nan, "gamma": np.nan}

# -------------------------
# Monte Carlo simulation (vectorized)
# -------------------------
def monte_carlo_paths(ticker: str, days: int = 60, sims: int = 10000):
    """Generate simulated price paths (sims x days) and return (paths, spot)."""
    yf_sym = yf_symbol(ticker)
    data = fetch_history(yf_sym, period="1y", auto_adjust=True)
    if data.empty or "Close" not in data.columns:
        raise RuntimeError("No price data for Monte Carlo.")
    close = data["Close"].dropna()
    S0 = float(close.iloc[-1])
    returns = close.pct_change().dropna()
    mu = float(returns.mean() * 252)  # annualized drift
    sigma = float(returns.std() * np.sqrt(252))  # annualized vol

    dt = 1.0 / 252.0
    # draw standard normals (sims x (days-1))
    z = np.random.normal(size=(sims, days - 1))
    increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    paths = np.empty((sims, days))
    paths[:, 0] = S0
    for t in range(1, days):
        paths[:, t] = paths[:, t - 1] * increments[:, t - 1]
    return paths, S0

# -------------------------
# ATR & sizing
# -------------------------
def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = series_high - series_low
    tr2 = (series_high - series_close.shift(1)).abs()
    tr3 = (series_low - series_close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period, min_periods=1).mean()
    return atr_series

def position_size_shares(symbol: str, account_size: float, risk_per_trade: float, atr_multiplier: float = 1.5, atr_period: int = 14) -> Tuple[int, float]:
    yf_sym = yf_symbol(symbol)
    hist = fetch_history(yf_sym, period="1y")
    if hist.empty or not {"High", "Low", "Close"}.issubset(hist.columns):
        return 0, None
    atr_series = atr(hist["High"], hist["Low"], hist["Close"], period=atr_period)
    last_atr = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else None
    if last_atr is None or last_atr <= 0:
        hv = hist["Close"].pct_change().std() * math.sqrt(252)
        last_price = float(hist["Close"].iloc[-1])
        last_atr = hv * last_price
        if last_atr <= 0:
            return 0, None
    stop_distance = last_atr * atr_multiplier
    if stop_distance <= 0:
        return 0, None
    shares = int(max(0, math.floor(risk_per_trade / stop_distance)))
    return shares, last_atr

# -------------------------
# Regime, RS & events
# -------------------------
@st.cache_data(ttl=3600)
def get_market_regime(benchmark: str = "SPY", short_window: int = 50, long_window: int = 200) -> str:
    df = fetch_history(benchmark, period="1y")
    if df.empty or "Close" not in df.columns:
        return "neutral"
    close = df["Close"].dropna()
    if len(close) < long_window:
        return "neutral"
    sma_short = close.rolling(short_window).mean().iloc[-1]
    sma_long = close.rolling(long_window).mean().iloc[-1]
    if sma_short > sma_long:
        return "bull"
    elif sma_short < sma_long:
        return "bear"
    else:
        return "neutral"

SECTOR_ETF_MAP = {
    "Technology":"XLK","Financials":"XLF","Consumer Cyclical":"XLY","Consumer Discretionary":"XLY",
    "Consumer Staples":"XLP","Healthcare":"XLV","Energy":"XLE","Industrials":"XLI","Materials":"XLB",
    "Utilities":"XLU","Real Estate":"XLRE","Communication Services":"XLC"
}

def compute_relative_strength(symbol: str, days: int = 30) -> Tuple[float, float]:
    try:
        yf_sym = yf_symbol(symbol)
        hist = fetch_history(yf_sym, period=f"{max(90, days*2)}d")
        if hist.empty or "Close" not in hist.columns:
            return (np.nan, np.nan)
        close = hist["Close"].dropna()
        if len(close) <= days:
            return (np.nan, np.nan)
        stock_ret = close.iloc[-1] / close.iloc[-days] - 1.0
        spy = fetch_history("SPY", period=f"{max(90, days*2)}d")
        spy_ret = np.nan
        if not spy.empty and "Close" in spy.columns and len(spy["Close"].dropna()) > days:
            spy_close = spy["Close"].dropna()
            spy_ret = spy_close.iloc[-1] / spy_close.iloc[-days] - 1.0
        tk = yf.Ticker(yf_sym)
        info = {}
        try:
            info = tk.info
        except Exception:
            info = {}
        sector_name = info.get("sector", None)
        sector_ret = np.nan
        if sector_name in SECTOR_ETF_MAP:
            etf = SECTOR_ETF_MAP[sector_name]
            s_etf = fetch_history(etf, period=f"{max(90, days*2)}d")
            if not s_etf.empty and "Close" in s_etf.columns and len(s_etf["Close"].dropna()) > days:
                se_close = s_etf["Close"].dropna()
                sector_ret = se_close.iloc[-1] / se_close.iloc[-days] - 1.0
        rs_spy = stock_ret - spy_ret if not np.isnan(stock_ret) and not np.isnan(spy_ret) else np.nan
        rs_sector = stock_ret - sector_ret if not np.isnan(stock_ret) and not np.isnan(sector_ret) else np.nan
        return (rs_spy, rs_sector)
    except Exception:
        return (np.nan, np.nan)

def get_upcoming_events(symbol: str) -> Dict[str, Any]:
    try:
        yf_sym = yf_symbol(symbol)
        tk = yf.Ticker(yf_sym)
        cal = {}
        try:
            ed = tk.calendar
            cal['earnings'] = ed if ed is not None else None
        except Exception:
            cal['earnings'] = None
        try:
            info = tk.info
            div = info.get("exDividendDate", None)
            if div:
                try:
                    cal['ex_dividend'] = datetime.fromtimestamp(div)
                except Exception:
                    cal['ex_dividend'] = div
            else:
                cal['ex_dividend'] = None
        except Exception:
            cal['ex_dividend'] = None
        return cal
    except Exception:
        return {}

# -------------------------
# Option chain analyzer
# -------------------------
def analyze_options_for_ticker(ticker: str,
                               max_exp_days: int = 45,
                               min_premium: float = 0.5,
                               min_volume: int = 20,
                               min_open_interest: int = 0,
                               max_spread_pct: float = 0.5,
                               earnings_exclude_days: int = 3,
                               r: float = DEFAULT_RFR,
                               delta_buy_range: Tuple[float,float] = (0.25,0.65),
                               delta_sell_range: Tuple[float,float] = (0.05,0.35),
                               apply_greeks_filters: bool = True,
                               pricing_model: str = "bs",
                               binomial_steps: int = 100,
                               iv_hv_adaptive: bool = True,
                               base_pct_diff_thresh: float = 5.0) -> pd.DataFrame:
    """
    Returns DataFrame of candidate contracts with computed fields.
    """
    yf_sym = yf_symbol(ticker)
    tk = yf.Ticker(yf_sym)

    # spot
    spot = None
    try:
        h = fetch_history(yf_sym, period="1d")
        if not h.empty and "Close" in h.columns:
            spot = float(h["Close"].iloc[-1])
    except Exception:
        spot = None

    # hv baseline
    try:
        hist_1y = fetch_history(yf_sym, period="1y")
        hv = float(hist_1y["Close"].pct_change().dropna().std() * math.sqrt(252)) if not hist_1y.empty else 0.30
    except Exception:
        hv = 0.30

    rows = []
    try:
        exps = tk.options
    except Exception:
        exps = []

    events = get_upcoming_events(ticker)
    earnings_date = None
    if events.get("earnings"):
        try:
            if isinstance(events["earnings"], dict):
                for v in events["earnings"].values():
                    try:
                        d = pd.to_datetime(v)
                        earnings_date = d
                        break
                    except Exception:
                        continue
            else:
                earnings_date = pd.to_datetime(events["earnings"])
        except Exception:
            earnings_date = None

    # gather IVs across whole chain
    iv_list = []
    for exp in exps:
        try:
            oc = tk.option_chain(exp)
            for df in (oc.calls, oc.puts):
                if df is None:
                    continue
                for _, rrow in df.iterrows():
                    iv = rrow.get("impliedVolatility", np.nan)
                    if not pd.isna(iv) and iv > 0:
                        iv_list.append(float(iv))
        except Exception:
            continue
    iv_series = np.array(iv_list) if iv_list else np.array([])

    for exp in exps:
        try:
            dte = (pd.to_datetime(exp) - pd.Timestamp.today()).days
        except Exception:
            continue
        if dte <= 0 or dte > max_exp_days:
            continue
        # earnings proximity exclusion
        if earnings_date is not None and earnings_exclude_days is not None and earnings_exclude_days > 0:
            try:
                exp_date = pd.to_datetime(exp)
                if abs((exp_date - earnings_date).days) <= earnings_exclude_days:
                    continue
            except Exception:
                pass
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue
        for df_opt, opt_type in ((oc.calls, "call"), (oc.puts, "put")):
            if df_opt is None or df_opt.shape[0] == 0:
                continue
            for _, row in df_opt.iterrows():
                strike = safe_float(row.get("strike", np.nan))
                bid = safe_float(row.get("bid", np.nan))
                ask = safe_float(row.get("ask", np.nan))
                last = safe_float(row.get("lastPrice", np.nan))
                vol = int(row.get("volume", 0)) if not pd.isna(row.get("volume", 0)) else 0
                oi = int(row.get("openInterest", 0)) if not pd.isna(row.get("openInterest", 0)) else 0
                mid = None
                if not np.isnan(bid) and not np.isnan(ask) and (bid > 0 or ask > 0):
                    mid = (bid + ask) / 2.0
                elif not np.isnan(last):
                    mid = float(last)
                else:
                    continue
                if mid is None or mid < min_premium:
                    continue
                if mid <= 0:
                    continue
                spread = None
                if not pd.isna(bid) and not pd.isna(ask) and mid > 0:
                    spread = ask - bid
                    spread_pct = spread / mid
                else:
                    spread_pct = np.nan
                # volume/open interest filters
                if vol < min_volume or oi < min_open_interest:
                    continue
                if not pd.isna(spread_pct) and spread_pct > max_spread_pct:
                    continue
                # iv
                iv = row.get("impliedVolatility", np.nan)
                if pd.isna(iv) or iv <= 0:
                    iv = hv  # fallback
                T = max(dte / 365.0, 1/365.0)
                # pricing model
                if pricing_model == "binomial":
                    fair = binomial_crr_price(spot if spot is not None else 0.0, strike, T, r, iv, steps=100, opt_type=opt_type, american=True)
                else:
                    fair = black_scholes_price(spot if spot is not None else 0.0, strike, T, r, iv, opt_type)
                if fair == 0:
                    continue
                pct_diff = ((mid - fair) / fair * 100.0)
                greeks = bs_greeks(spot if spot is not None else 0.0, strike, T, r, iv, opt_type)
                delta = greeks.get("delta", np.nan)
                vega = greeks.get("vega", np.nan)
                theta = greeks.get("theta", np.nan)
                hv_local = hv
                iv_hv_ratio = iv / hv_local if hv_local and hv_local > 0 else np.nan
                iv_percentile = float((iv_series < iv).sum() / len(iv_series) * 100.0) if iv_series.size > 0 else 50.0
                rows.append({
                    "contract": row.get("contractSymbol",""),
                    "type": opt_type,
                    "strike": strike,
                    "expiration": exp,
                    "dte": dte,
                    "bid": bid,
                    "ask": ask,
                    "mid": round(mid,4),
                    "fair": round(fair,4),
                    "pct_diff": round(pct_diff,2),
                    "delta": round(delta,4) if not np.isnan(delta) else np.nan,
                    "vega": round(vega,4) if not np.isnan(vega) else np.nan,
                    "theta": round(theta,4) if not np.isnan(theta) else np.nan,
                    "iv": round(iv,4) if not np.isnan(iv) else np.nan,
                    "hv": round(hv_local,4),
                    "iv_hv_ratio": round(iv_hv_ratio,3) if not np.isnan(iv_hv_ratio) else np.nan,
                    "iv_percentile": round(iv_percentile,1),
                    "spread_pct": round(spread_pct,4) if not pd.isna(spread_pct) else np.nan,
                    "volume": vol,
                    "openInterest": oi,
                    "earnings_prox": (abs((pd.to_datetime(exp) - earnings_date).days) if earnings_date is not None else None)
                })
    if not rows:
        cols = ["contract","type","strike","expiration","dte","bid","ask","mid","fair","pct_diff","delta","vega","theta","iv","hv","iv_hv_ratio","iv_percentile","spread_pct","volume","openInterest","earnings_prox"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    # adaptive threshold
    # ensure global exists; default if not
    global base_pct_diff_thresh_global
    if "base_pct_diff_thresh_global" not in globals():
        base_pct_diff_thresh_global = 5.0
    df["adaptive_thresh"] = base_pct_diff_thresh_global * (1 + (df["iv_percentile"] - 50)/100.0)
    df["is_overpriced"] = df["pct_diff"] > df["adaptive_thresh"]
    df["is_underpriced"] = df["pct_diff"] < -df["adaptive_thresh"]
    return df

# -------------------------
# Scoring & EV helpers
# -------------------------
def approx_probability_ITM_from_bs(S: float, K: float, T: float, mu: float, sigma: float) -> float:
    try:
        if T <= 0 or sigma <= 0:
            return 1.0 if (S > K) else 0.0
        d = (math.log(S / K) + (mu - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return float(norm.cdf(d))
    except Exception:
        return np.nan

def mc_expected_payoff(S: float, K: float, T: float, mu: float, sigma: float, opt_type: str = "call", sims: int = 2000) -> float:
    if T <= 0:
        return max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
    z = np.random.normal(size=(sims,))
    ST = S * np.exp((mu - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    if opt_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    return float(np.mean(payoffs))

def compute_option_scores(df_opts: pd.DataFrame, factor_score: float = 50.0, backtest_win: float = None) -> pd.DataFrame:
    if df_opts.empty:
        return df_opts
    df = df_opts.copy()
    df["misprice_score"] = np.clip((df["pct_diff"] + 50.0) / 100.0 * 100.0, 0, 100)
    df["iv_score"] = df["iv_percentile"].fillna(50)
    df["liq_score"] = (np.clip(df["volume"] / 100.0, 0, 100) * 0.6 + np.clip(df["openInterest"] / 10.0, 0, 100) * 0.4)
    median_ev = df["ev_per_contract"].replace(0, np.nan).median() if "ev_per_contract" in df.columns else np.nan
    if np.isnan(median_ev) or median_ev == 0:
        df["ev_score"] = 50.0
    else:
        df["ev_score"] = np.clip(50 + (df["ev_per_contract"].fillna(0) / (abs(median_ev)) * 10), 0, 100)
    df["factor_score"] = factor_score
    df["backtest_score"] = backtest_win if (backtest_win is not None and not np.isnan(backtest_win)) else 50.0
    w = {"mis":0.35, "iv":0.15, "liq":0.20, "factor":0.15, "ev":0.10, "bt":0.05}
    df["final_score"] = (w["mis"] * df["misprice_score"] + w["iv"] * df["iv_score"] + w["liq"] * df["liq_score"] + w["factor"] * df["factor_score"] + w["ev"] * df["ev_score"] + w["bt"] * df["backtest_score"])
    df["final_score"] = df["final_score"].clip(0,100).round(2)
    df["misprice_score"] = df["misprice_score"].round(2)
    df["iv_score"] = df["iv_score"].round(2)
    df["liq_score"] = df["liq_score"].round(2)
    df["ev_score"] = df["ev_score"].round(2)
    return df

# -------------------------
# Factor universe & Top-10 finder
# -------------------------
@st.cache_data(ttl=24*3600)
def build_factor_universe(tickers: List[str]) -> pd.DataFrame:
    records = []
    for sym in tickers:
        try:
            yf_sym = yf_symbol(sym)
            hist = fetch_history(yf_sym, period="1y")
            if hist.empty or "Close" not in hist.columns or len(hist["Close"].dropna()) < 200:
                continue
            close = hist["Close"].dropna()
            tk = yf.Ticker(yf_sym)
            try:
                info = tk.info
            except Exception:
                info = {}
            momentum = float(close.iloc[-1] / close.iloc[0] - 1.0)
            volatility = float(close.pct_change().dropna().std() * math.sqrt(252))
            pe = info.get("trailingPE", np.nan)
            quality = info.get("returnOnEquity", np.nan)
            if pd.isna(quality):
                quality = info.get("profitMargins", np.nan)
            records.append({
                "ticker": sym,
                "momentum": momentum,
                "volatility": volatility,
                "value": float(pe) if not pd.isna(pe) else np.nan,
                "quality": float(quality) if not pd.isna(quality) else np.nan
            })
        except Exception:
            continue
    if not records:
        return pd.DataFrame(columns=["momentum","volatility","value","quality"])
    df = pd.DataFrame(records).set_index("ticker")
    df = df.dropna(how="any")
    df["momentum_score"] = df["momentum"].rank(pct=True) * 100.0
    df["quality_score"] = df["quality"].rank(pct=True) * 100.0
    df["volatility_score"] = (1.0 - df["volatility"].rank(pct=True)) * 100.0
    df["value_score"] = (1.0 - df["value"].rank(pct=True)) * 100.0
    df["overall_score"] = df[["momentum_score","quality_score","volatility_score","value_score"]].mean(axis=1)
    return df

def find_best_contract_for_ticker(ticker: str, factor_score: float, pricing_model: str, binomial_steps: int,
                                  max_exp_days: int, min_premium: float, min_volume: int, min_oi: int, max_spread_pct: float,
                                  earnings_exclude_days: int, apply_greeks_filters: bool, delta_buy_range, delta_sell_range,
                                  account_size: float, portfolio_notional_cap: float, mc_sims_ev: int = 2000) -> Tuple[pd.Series, pd.DataFrame]:
    df = analyze_options_for_ticker(ticker,
                                    max_exp_days=max_exp_days,
                                    min_premium=min_premium,
                                    min_volume=min_volume,
                                    min_open_interest=min_oi,
                                    max_spread_pct=max_spread_pct,
                                    earnings_exclude_days=earnings_exclude_days,
                                    r=DEFAULT_RFR,
                                    delta_buy_range=delta_buy_range,
                                    delta_sell_range=delta_sell_range,
                                    apply_greeks_filters=apply_greeks_filters,
                                    pricing_model=pricing_model,
                                    binomial_steps=binomial_steps,
                                    iv_hv_adaptive=True,
                                    base_pct_diff_thresh=5.0)
    if df.empty:
        return None, df
    try:
        hist = fetch_history(yf_symbol(ticker), period="1y")
        close = hist["Close"].dropna()
        mu = float(close.pct_change().mean() * 252)
        sigma = float(close.pct_change().dropna().std() * math.sqrt(252))
        S = float(close.iloc[-1])
    except Exception:
        mu, sigma = 0.0, 0.3
        S = float(fetch_history(yf_symbol(ticker), period="1d")["Close"].iloc[-1])
    df["prob_itm"] = df.apply(lambda r: approx_probability_ITM_from_bs(S, r["strike"], r["dte"]/365.0, mu, sigma), axis=1)
    df["ev_per_contract"] = np.nan
    df = compute_option_scores(df, factor_score=factor_score, backtest_win=None)
    top_candidates = df.sort_values("final_score", ascending=False).head(12).copy()
    evs = []
    for idx, r in top_candidates.iterrows():
        T = max(r["dte"]/365.0, 1/365.0)
        expected_payoff = mc_expected_payoff(S, r["strike"], T, mu, sigma, opt_type=r["type"], sims=mc_sims_ev)
        premium = float(r["mid"])
        if r["pct_diff"] < 0:
            ev = expected_payoff - premium
        else:
            ev = premium - expected_payoff
        evs.append(ev)
    top_candidates["ev_per_contract"] = evs + [np.nan]*(len(top_candidates)-len(evs))
    for idx in top_candidates.index:
        df.loc[idx, "ev_per_contract"] = top_candidates.loc[idx, "ev_per_contract"]
    df = compute_option_scores(df, factor_score=factor_score, backtest_win=None)
    best = df.sort_values("final_score", ascending=False).iloc[0]
    return best, df

# -------------------------
# Streamlit UI & controls
# -------------------------
st.title("📈 JASPER — Ultimate Options & Edge Analyzer")

st.markdown(
    """
    **Notes:** Analysis-only. Backtests and EV use modelled option prices (Black–Scholes / Monte Carlo). Use liquidity filters and conservative sizing.
    """
)

# Sidebar controls
st.sidebar.header("Global Controls")
tickers_text = st.sidebar.text_area("Tickers (comma separated) — leave blank to use Top-10 S&P500", value="AAPL,MSFT,TSLA", height=120)
max_exp_days = st.sidebar.slider("Max expiration (days)", 7, 365, 45)
min_premium = st.sidebar.number_input("Min premium ($)", value=0.3, step=0.1)
min_volume = st.sidebar.number_input("Min option volume", value=20, step=5)
min_oi = st.sidebar.number_input("Min open interest", value=0, step=10)
max_spread_pct = st.sidebar.slider("Max bid-ask spread (% of mid)", 1, 100, 30)/100.0
earnings_exclude_days = st.sidebar.number_input("Exclude expiries within N days of earnings (0 to disable)", value=3, min_value=0, step=1)
pricing_model = st.sidebar.selectbox("Pricing model", ["bs","binomial"])
binomial_steps = st.sidebar.slider("Binomial steps", 20, 400, 120, step=10)

st.sidebar.markdown("Monte Carlo / EV")
show_mc = st.sidebar.checkbox("Show Monte Carlo simulation per ticker", value=True)
mc_days = st.sidebar.slider("Monte Carlo horizon (days)", 7, 252, 60)
mc_sims = st.sidebar.number_input("Monte Carlo sims (per ticker)", value=10000, step=1000, min_value=1000)

st.sidebar.markdown("Greeks filters")
delta_buy_min = st.sidebar.slider("Buy delta min", 0.0, 1.0, 0.25)
delta_buy_max = st.sidebar.slider("Buy delta max", 0.0, 1.0, 0.65)
delta_sell_min = st.sidebar.slider("Sell delta min", 0.0, 1.0, 0.05)
delta_sell_max = st.sidebar.slider("Sell delta max", 0.0, 1.0, 0.35)
apply_greeks = st.sidebar.checkbox("Apply greeks filters", value=True)

st.sidebar.markdown("S&P universe & Top-10")
build_universe_btn = st.sidebar.button("Build/refresh S&P500 factor universe (slow)")
topn = st.sidebar.number_input("Top N from S&P500 to show", value=10, min_value=1, step=1)

st.sidebar.markdown("Sizing & portfolio")
account_size = st.sidebar.number_input("Account size ($)", value=20000.0, step=1000.0)
risk_per_trade = st.sidebar.number_input("Risk per trade ($)", value=200.0, step=10.0)
portfolio_notional_cap = st.sidebar.slider("Max portfolio notional fraction", 0.01, 1.0, 0.20)

st.sidebar.markdown("Filters & regime")
apply_regime = st.sidebar.checkbox("Apply market regime (SPY 50/200 SMA)", value=True)
allow_against_regime = st.sidebar.checkbox("Allow trades against regime", value=False)
apply_rs = st.sidebar.checkbox("Apply relative strength filter for buys", value=True)
rs_days = st.sidebar.number_input("RS lookback days", value=30)
min_rs_vs_spy = st.sidebar.number_input("Min RS vs SPY for buys (decimal)", value=0.0, step=0.01)
min_rs_vs_sector = st.sidebar.number_input("Min RS vs sector for buys (decimal)", value=0.0, step=0.01)

st.sidebar.markdown("Backtest & display")
enable_backtest = st.sidebar.checkbox("Enable approximate backtest", value=False)
backtest_start = st.sidebar.date_input("Backtest start", value=(datetime.now() - timedelta(days=365)).date())
backtest_end = st.sidebar.date_input("Backtest end", value=datetime.now().date())
backtest_lookback = st.sidebar.number_input("Backtest lookback days", value=90)
max_display = st.sidebar.number_input("Max rows to display (tables/backtest)", value=200, min_value=10, step=10)
score_threshold = st.sidebar.slider("Hide options with combined score below", 0.0, 100.0, 0.0)

st.sidebar.markdown("Advanced")
base_pct_diff_thresh_global = st.sidebar.number_input("Base % diff threshold (for adaptive)", value=5.0, step=1.0)
min_volume_for_liq = st.sidebar.number_input("Min volume for Top-10 candidate", value=50, step=10)
mc_sample_plot = st.sidebar.number_input("Monte Carlo sample paths to plot", value=200, min_value=10, step=10)

# Build universe if requested
if build_universe_btn:
    with st.spinner("Building S&P500 factor universe (this can take some minutes)..."):
        sp500 = load_sp500_tickers()
        if not sp500:
            st.error("Failed to load S&P500 tickers.")
        else:
            factor_df = build_factor_universe(sp500)
            if factor_df.empty:
                st.error("Universe build returned no data.")
            else:
                st.session_state["factor_df"] = factor_df
                st.success(f"Built universe with {factor_df.shape[0]} tickers scored.")

# -------------------------
# Main run
# -------------------------
if st.button("Run analysis"):
    user_tickers = [s.strip().upper() for s in tickers_text.split(",") if s.strip()]
    use_top10_universe = False
    if not user_tickers:
        if "factor_df" in st.session_state:
            use_top10_universe = True
            leaders = st.session_state["factor_df"].sort_values("overall_score", ascending=False).head(int(topn))
            user_tickers = list(leaders.index)
        else:
            st.error("No tickers provided and no factor universe built.")
            st.stop()

    regime = get_market_regime("SPY") if apply_regime else "neutral"
    st.write(f"Market regime: **{regime}**")

    all_matches = []
    top10_results = []

    # Top-10 run if using universe
    if use_top10_universe and "factor_df" in st.session_state:
        factor_df = st.session_state["factor_df"]
        leaders = factor_df.sort_values("overall_score", ascending=False).head(int(topn))
        st.subheader(f"Top {topn} S&P500 leaders")
        st.dataframe(leaders[["overall_score","momentum_score","volatility_score","value_score","quality_score"]].head(int(topn)))
        for ticker in leaders.index:
            st.write(f"Finding best contract for {ticker} ...")
            best, df_candidates = find_best_contract_for_ticker(ticker,
                                                               factor_score=float(leaders.loc[ticker,"overall_score"]),
                                                               pricing_model=pricing_model,
                                                               binomial_steps=int(binomial_steps),
                                                               max_exp_days=int(max_exp_days),
                                                               min_premium=float(min_premium),
                                                               min_volume=int(min_volume_for_liq),
                                                               min_oi=int(min_oi),
                                                               max_spread_pct=float(max_spread_pct),
                                                               earnings_exclude_days=int(earnings_exclude_days),
                                                               apply_greeks_filters=apply_greeks,
                                                               delta_buy_range=(delta_buy_min, delta_buy_max),
                                                               delta_sell_range=(delta_sell_min, delta_sell_max),
                                                               account_size=float(account_size),
                                                               portfolio_notional_cap=float(portfolio_notional_cap),
                                                               mc_sims_ev=int(mc_sims/5))
            if best is None:
                st.info(f"No acceptable contracts for {ticker}")
                continue
            st.markdown(f"**{ticker}** best contract: {best['contract']}  final_score={best['final_score']:.1f}")
            st.dataframe(pd.DataFrame([best])[["contract","type","strike","expiration","dte","mid","fair","pct_diff","iv","hv","iv_hv_ratio","volume","openInterest","final_score"]])
            top10_results.append(best.to_dict())
            all_matches.append(best.to_dict())
        if top10_results:
            top10_df = pd.DataFrame(top10_results)
            st.download_button("Download Top-10 best contracts CSV", top10_df.to_csv(index=False), file_name="jasper_top10_contracts.csv")

    # Loop user tickers
    for ticker in user_tickers:
        st.header(f"Ticker: {ticker}")

        # Monte Carlo (toggleable)
        if show_mc:
            with st.spinner("Running Monte Carlo simulation..."):
                try:
                    paths, spot = monte_carlo_paths(ticker, days=int(mc_days), sims=int(mc_sims))
                    final = paths[:, -1]
                    up_pct = float((final > spot).mean() * 100.0)
                    down_pct = 100.0 - up_pct
                    sample_n = min(int(mc_sample_plot), paths.shape[0])
                    sample_idx = np.random.choice(paths.shape[0], sample_n, replace=False)
                    sample = paths[sample_idx, :]
                    col_mc, col_pie = st.columns([3,1])
                    fig_mc, ax_mc = plt.subplots(figsize=(6,2.2))
                    ax_mc.plot(sample.T, alpha=0.12, linewidth=0.6)
                    ax_mc.axhline(spot, color="black", linestyle="--", linewidth=0.8, label="Current Price")
                    ax_mc.set_title(f"Monte Carlo sample ({sample_n} paths)")
                    ax_mc.set_xlabel("Days")
                    ax_mc.set_ylabel("Price")
                    with col_mc:
                        st.pyplot(fig_mc)
                    fig_pie, ax_pie = plt.subplots(figsize=(3,2.2))
                    ax_pie.pie([up_pct, down_pct], labels=[f"Up ({up_pct:.1f}%)", f"Down ({down_pct:.1f}%)"], autopct="%1.1f%%")
                    ax_pie.set_title("MC outcomes")
                    with col_pie:
                        st.pyplot(fig_pie)
                except Exception as e:
                    st.warning(f"Monte Carlo unavailable for {ticker} ({e})")
        else:
            st.info("Monte Carlo simulation disabled (toggle in sidebar).")

        # Factor breakdown
        factor_score = 50.0
        if "factor_df" in st.session_state and ticker in st.session_state["factor_df"].index:
            frow = st.session_state["factor_df"].loc[ticker]
            factor_score = float(frow["overall_score"])
            bf = pd.Series({"Momentum": float(frow["momentum_score"]), "Low Volatility": float(frow["volatility_score"]),
                            "Value": float(frow["value_score"]), "Quality": float(frow["quality_score"])})
            figf, axf = plt.subplots(figsize=(6,2))
            bf.plot(kind="bar", ax=axf)
            axf.axhline(50, color="gray", linestyle="--")
            axf.set_ylim(0,100)
            st.pyplot(figf)
        else:
            try:
                hist = fetch_history(yf_symbol(ticker), period="1y")
                if not hist.empty and "Close" in hist.columns:
                    close = hist["Close"].dropna()
                    momentum = float(close.iloc[-1] / close.iloc[0] - 1.0)
                    vol = float(close.pct_change().dropna().std() * math.sqrt(252))
                    m_score = np.clip((momentum + 0.5) * 100, 0, 100)
                    vol_score = np.clip((0.35 - vol) * 200, 0, 100)
                    pe = None
                    try:
                        tk = yf.Ticker(yf_symbol(ticker)); info = tk.info; pe = info.get("trailingPE", np.nan)
                    except Exception:
                        pe = np.nan
                    val_score = np.clip((50 - (pe if not pd.isna(pe) else 50)) * 2, 0, 100)
                    q_score = 50.0
                    bf = pd.Series({"Momentum": m_score, "Low Volatility": vol_score, "Value": val_score, "Quality": q_score})
                    figf, axf = plt.subplots(figsize=(6,2))
                    bf.plot(kind="bar", ax=axf)
                    axf.axhline(50, color="gray", linestyle="--")
                    axf.set_ylim(0,100)
                    st.pyplot(figf)
                    factor_score = float(np.nanmean([m_score, vol_score, val_score, q_score]))
            except Exception:
                pass

        # Position sizing suggestion
        shares, last_atr = position_size_shares(ticker, account_size, risk_per_trade, atr_multiplier=1.5, atr_period=14)
        st.write(f"Suggested shares (risk ${risk_per_trade}): {shares} — ATR used: {round(last_atr,4) if last_atr else 'N/A'}")

        # Options scan
        with st.spinner("Scanning option chain..."):
            df_opts = analyze_options_for_ticker(ticker,
                                                 max_exp_days=int(max_exp_days),
                                                 min_premium=float(min_premium),
                                                 min_volume=int(min_volume),
                                                 min_open_interest=int(min_oi),
                                                 max_spread_pct=float(max_spread_pct),
                                                 earnings_exclude_days=int(earnings_exclude_days),
                                                 r=DEFAULT_RFR,
                                                 delta_buy_range=(delta_buy_min, delta_buy_max),
                                                 delta_sell_range=(delta_sell_min, delta_sell_max),
                                                 apply_greeks_filters=apply_greeks,
                                                 pricing_model=pricing_model,
                                                 binomial_steps=int(binomial_steps),
                                                 iv_hv_adaptive=True,
                                                 base_pct_diff_thresh=float(base_pct_diff_thresh_global))
        if df_opts.empty:
            st.info("No options matched filters.")
            continue

        # compute mu/sigma for EV approx
        try:
            hist = fetch_history(yf_symbol(ticker), period="1y")
            close = hist["Close"].dropna()
            mu = float(close.pct_change().mean() * 252)
            sigma = float(close.pct_change().dropna().std() * math.sqrt(252))
            S = float(close.iloc[-1])
        except Exception:
            mu, sigma = 0.0, 0.3
            try:
                S = float(fetch_history(yf_symbol(ticker), period="1d")["Close"].iloc[-1])
            except Exception:
                S = None

        df_opts["prob_itm"] = df_opts.apply(lambda r: approx_probability_ITM_from_bs(S if S is not None else 0.0, r["strike"], r["dte"]/365.0, mu, sigma), axis=1)
        df_opts["ev_per_contract"] = np.nan
        top_for_ev = df_opts.sort_values("pct_diff", ascending=False).head(12)
        ev_vals = []
        for idx, r in top_for_ev.iterrows():
            T = max(r["dte"]/365.0, 1/365.0)
            pay = mc_expected_payoff(S if S is not None else 0.0, r["strike"], T, mu, sigma, opt_type=r["type"], sims=max(500, int(mc_sims/4)))
            prem = float(r["mid"])
            ev = pay - prem if r["pct_diff"] < 0 else prem - pay
            ev_vals.append(ev)
        top_for_ev["ev_per_contract"] = ev_vals + [np.nan]*(len(top_for_ev)-len(ev_vals))
        for idx in top_for_ev.index:
            df_opts.loc[idx, "ev_per_contract"] = top_for_ev.loc[idx, "ev_per_contract"]

        # scoring
        df_scored = compute_option_scores(df_opts, factor_score=factor_score, backtest_win=None)
        if score_threshold > 0:
            df_scored = df_scored[df_scored["final_score"] >= score_threshold]
        if df_scored.empty:
            st.info("No options above score threshold.")
            continue

        df_scored_sorted = df_scored.sort_values("final_score", ascending=False).reset_index(drop=True)
        show_n = int(max_display)
        st.subheader(f"Top {min(show_n, df_scored_sorted.shape[0])} candidates by final score")
        st.dataframe(df_scored_sorted.head(show_n)[["contract","type","strike","expiration","dte","mid","fair","pct_diff","iv","hv","iv_hv_ratio","volume","openInterest","spread_pct","final_score","misprice_score","iv_score","liq_score","ev_score","factor_score"]])

        # buckets
        buy_calls = df_scored_sorted[(df_scored_sorted["type"]=="call") & (df_scored_sorted["pct_diff"]<0)].head(12)
        buy_puts  = df_scored_sorted[(df_scored_sorted["type"]=="put")  & (df_scored_sorted["pct_diff"]<0)].head(12)
        sell_calls = df_scored_sorted[(df_scored_sorted["type"]=="call") & (df_scored_sorted["pct_diff"]>0)].head(12)
        sell_puts  = df_scored_sorted[(df_scored_sorted["type"]=="put")  & (df_scored_sorted["pct_diff"]>0)].head(12)
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.subheader("Buy Calls (underpriced)")
            st.dataframe(buy_calls[["contract","strike","expiration","dte","mid","fair","pct_diff","delta","iv","volume","openInterest","spread_pct","final_score"]])
        with c2:
            st.subheader("Buy Puts (underpriced)")
            st.dataframe(buy_puts[["contract","strike","expiration","dte","mid","fair","pct_diff","delta","iv","volume","openInterest","spread_pct","final_score"]])
        with c3:
            st.subheader("Sell Calls (overpriced)")
            st.dataframe(sell_calls[["contract","strike","expiration","dte","mid","fair","pct_diff","delta","iv","volume","openInterest","spread_pct","final_score"]])
        with c4:
            st.subheader("Sell Puts (overpriced)")
            st.dataframe(sell_puts[["contract","strike","expiration","dte","mid","fair","pct_diff","delta","iv","volume","openInterest","spread_pct","final_score"]])

        all_matches.append(df_scored_sorted.assign(origin=ticker))

    # final aggregate & download
    if all_matches:
        full_df = pd.concat(all_matches, ignore_index=True)
        st.subheader("Aggregate matches (all tickers)")
        st.dataframe(full_df.head(int(max_display)))
        st.download_button("Download all matches CSV", full_df.to_csv(index=False), file_name="jasper_all_matches.csv")
    else:
        st.info("No matches found in this run.")

    st.success("Run complete.")
