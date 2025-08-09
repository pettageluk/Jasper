# app.py
"""
Comprehensive single-file Streamlit app:
- Monte Carlo simulation (vectorized)
- Options screener (Yahoo option chains) with Black-Scholes fair prices and % diff
- Multi-factor scoring (Momentum, Value, Quality, Low Volatility) built from S&P-500 universe
- UI: ticker input, filters, Monte Carlo params, universe build, CSV download
- Robust handling for yfinance multi-index columns and missing fields
"""

import io
import time
from datetime import datetime
from math import exp

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Options + Monte Carlo + Multi-Factor Analyzer", layout="wide")
DEFAULT_RISK_FREE = 0.02

# ---------------------------
# Utilities
# ---------------------------
def yf_symbol(sym: str) -> str:
    """Map S&P style ticker (BRK.B) to Yahoo style (BRK-B)."""
    return sym.replace(".", "-").upper()

def flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """If df has MultiIndex columns from yfinance, flatten them to top-level names when possible."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # col often like ('Adj Close','AAPL') or ('Close','AAPL')
            if isinstance(col[0], str) and col[0] != "":
                new_cols.append(col[0])
            else:
                new_cols.append("_".join([str(x) for x in col if x]))
        df = df.copy()
        df.columns = new_cols
    return df

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

# ---------------------------
# S&P500 tickers loader (cached)
# ---------------------------
@st.cache_data(ttl=24 * 3600)
def load_sp500_tickers() -> list:
    """Load S&P500 tickers. Prefer DataHub; fallback to Wikipedia."""
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

# ---------------------------
# Data fetch helper (robust)
# ---------------------------
def fetch_history(yf_sym: str, period: str = "1y", auto_adjust: bool = True) -> pd.DataFrame:
    """Fetch history for symbol and flatten MultiIndex columns. Ensure 'Close'/'Adj Close' exist."""
    try:
        tk = yf.Ticker(yf_sym)
        df = tk.history(period=period, auto_adjust=auto_adjust)
        if df is None or df.empty:
            return pd.DataFrame()
        df = flatten_multiindex_cols(df)
        # ensure Close exists; fallback to the first numeric column
        if "Close" not in df.columns and "Adj Close" not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df["Close"] = df[numeric_cols[0]]
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        return df
    except Exception:
        return pd.DataFrame()

# ---------------------------
# Black-Scholes
# ---------------------------
def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Calculate Black-Scholes price. T in years, sigma annual vol."""
    try:
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    except Exception:
        return 0.0

# ---------------------------
# Monte Carlo simulation (vectorized)
# ---------------------------
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

# ---------------------------
# Options analysis
# ---------------------------
def analyze_options_for_ticker(ticker: str, max_exp_days: int, min_premium: float, r: float = DEFAULT_RISK_FREE) -> pd.DataFrame:
    """
    Pull option chains for ticker and compute Black-Scholes fair price and percent difference:
    Returns DataFrame with columns: contract,type,strike,expiration,dte,bid,ask,current,fair,pct_diff
    """
    yf_sym = yf_symbol(ticker)
    tk = yf.Ticker(yf_sym)

    # spot price robust
    spot_price = None
    try:
        hist = fetch_history(yf_sym, period="1d")
        if not hist.empty and "Close" in hist.columns:
            spot_price = float(hist["Close"].iloc[-1])
    except Exception:
        spot_price = None

    # fallback hist sigma
    try:
        hist_all = fetch_history(yf_sym, period="1y")
        hist_sigma = float(hist_all["Close"].pct_change().dropna().std() * np.sqrt(252)) if not hist_all.empty else 0.30
    except Exception:
        hist_sigma = 0.30

    rows = []
    try:
        exps = tk.options
    except Exception:
        exps = []

    for exp in exps:
        try:
            dte = (pd.to_datetime(exp) - pd.Timestamp.today()).days
        except Exception:
            continue
        if dte <= 0 or dte > max_exp_days:
            continue
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue
        for df, opt_type in [(oc.calls, "call"), (oc.puts, "put")]:
            if df is None or df.shape[0] == 0:
                continue
            for _, row in df.iterrows():
                strike = safe_float(row.get("strike", np.nan))
                bid = row.get("bid", np.nan)
                ask = row.get("ask", np.nan)
                last = row.get("lastPrice", np.nan)
                # choose current price: prefer bid/ask mid, else lastPrice
                current = None
                if pd.notna(bid) and pd.notna(ask) and (safe_float(bid, 0.0) > 0 or safe_float(ask, 0.0) > 0):
                    current = float((safe_float(bid, 0.0) + safe_float(ask, 0.0)) / 2.0)
                elif pd.notna(last):
                    current = safe_float(last, np.nan)
                else:
                    continue
                if current is None or current < min_premium:
                    continue
                iv = row.get("impliedVolatility", np.nan)
                sigma = float(iv) if pd.notna(iv) and iv > 0 else hist_sigma
                T = max(dte / 365.0, 1 / 365.0)
                fair = black_scholes_price(spot_price if spot_price is not None else 0.0, strike, T, r, sigma, opt_type)
                pct_diff = ((current - fair) / fair * 100.0) if fair > 0 else np.nan
                rows.append({
                    "contract": row.get("contractSymbol", ""),
                    "type": opt_type,
                    "strike": strike,
                    "expiration": exp,
                    "dte": dte,
                    "bid": bid,
                    "ask": ask,
                    "current": current,
                    "fair": round(fair, 4),
                    "pct_diff": round(pct_diff, 2)
                })
    if not rows:
        return pd.DataFrame(columns=["contract", "type", "strike", "expiration", "dte", "bid", "ask", "current", "fair", "pct_diff"])
    df = pd.DataFrame(rows)
    return df

# ---------------------------
# Multi-factor universe build & scoring
# ---------------------------
@st.cache_data(ttl=24 * 3600)
def build_factor_universe(tickers: list) -> pd.DataFrame:
    """
    Build universe: for each ticker fetch 1y history + info and compute raw factors:
    momentum, volatility, value (PE), quality (ROE/profitMargins).
    Returns DataFrame indexed by ticker and only tickers with complete factor data.
    """
    records = []
    for sym in tickers:
        yf_sym = yf_symbol(sym)
        try:
            hist = fetch_history(yf_sym, period="1y")
            if hist is None or hist.shape[0] < 200:
                continue
            close = hist["Close"].dropna()
            if close.shape[0] < 200:
                continue
            tk = yf.Ticker(yf_sym)
            info = tk.info
            momentum = float(close.iloc[-1] / close.iloc[0] - 1.0)
            volatility = float(close.pct_change().dropna().std() * np.sqrt(252))
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
        return pd.DataFrame(columns=["momentum", "volatility", "value", "quality"])
    df = pd.DataFrame(records).set_index("ticker")
    df = df.dropna(how="any")
    return df

def compute_percentile_scores(df_factors: pd.DataFrame) -> pd.DataFrame:
    """Map raw factor values to 0-100 percentiles (50 = median). Invert vol/value (lower better)."""
    df = df_factors.copy()
    df["momentum_score"] = df["momentum"].rank(pct=True) * 100.0
    df["quality_score"] = df["quality"].rank(pct=True) * 100.0
    df["volatility_score"] = (1.0 - df["volatility"].rank(pct=True)) * 100.0
    df["value_score"] = (1.0 - df["value"].rank(pct=True)) * 100.0
    df["overall_score"] = df[["momentum_score", "quality_score", "volatility_score", "value_score"]].mean(axis=1)
    return df

# ---------------------------
# UI - Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
ticker_input = st.sidebar.text_input("Ticker (e.g., AAPL)", value="AAPL").strip().upper()
max_exp_days = st.sidebar.slider("Max option expiration (days)", 7, 365, 45, step=1)
min_premium = st.sidebar.number_input("Minimum option premium ($)", value=0.5, min_value=0.0, step=0.1)
mc_days = st.sidebar.slider("Monte Carlo horizon (days)", 7, 252, 60)
mc_sims = st.sidebar.number_input("Monte Carlo simulations", value=10000, step=1000, min_value=1000)
build_universe_btn = st.sidebar.button("Build / Refresh S&P500 factor universe (slow)")

st.title("📈 JASPER — Options + Monte Carlo + Multi-Factor Analyzer")
st.markdown("Type a ticker in the sidebar, adjust filters, and click **Run analysis**.")

# Load S&P tickers
with st.spinner("Loading S&P 500 tickers..."):
    sp500_tickers = load_sp500_tickers()
if not sp500_tickers:
    st.warning("Could not load S&P500 tickers — network or source unavailable.")

# Build factor universe if requested (heavy)
if build_universe_btn:
    if not sp500_tickers:
        st.error("No S&P tickers available to build the universe.")
    else:
        with st.spinner("Building S&P500 factor universe (this can take several minutes)..."):
            factor_raw = build_factor_universe(sp500_tickers)
            if factor_raw.empty:
                st.error("Universe build found no tickers with complete data.")
            else:
                factor_scored = compute_percentile_scores(factor_raw)
                st.session_state["factor_df"] = factor_scored
                st.success(f"Built universe with {factor_scored.shape[0]} tickers scored.")

# ---------------------------
# Main analysis button
# ---------------------------
if st.button("Run analysis"):
    if not ticker_input:
        st.error("Enter a ticker in the sidebar.")
    else:
        ticker = ticker_input.upper()

        # Monte Carlo
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                paths, spot = monte_carlo_paths(ticker, days=int(mc_days), sims=int(mc_sims))
                final = paths[:, -1]
                up_pct = float((final > spot).mean() * 100.0)
                down_pct = 100.0 - up_pct
            except Exception as e:
                st.error(f"Monte Carlo simulation failed: {e}")
                paths, spot, up_pct, down_pct = None, None, None, None

        # Show Monte Carlo (left) and pie (right) side-by-side, small
        col_mc, col_pie = st.columns([3, 1])
        if paths is not None:
            sample_n = min(200, paths.shape[0])
            sample_idx = np.random.choice(paths.shape[0], sample_n, replace=False)
            sample = paths[sample_idx, :]

            fig_mc, ax_mc = plt.subplots(figsize=(6, 3))
            ax_mc.plot(sample.T, color="#1f77b4", alpha=0.12, linewidth=0.6)
            ax_mc.axhline(spot, color="black", linestyle="--", linewidth=0.8, label="Current Price")
            ax_mc.set_title(f"Monte Carlo sample ({sample_n} of {paths.shape[0]})")
            ax_mc.set_xlabel("Days")
            ax_mc.set_ylabel("Price")
            with col_mc:
                st.pyplot(fig_mc)

            fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
            ax_pie.pie([up_pct, down_pct], labels=[f"Up ({up_pct:.1f}%)", f"Down ({down_pct:.1f}%)"],
                       colors=["#2ca02c", "#d62728"], autopct="%1.1f%%")
            ax_pie.set_title("Monte Carlo outcomes")
            with col_pie:
                st.pyplot(fig_pie)
        else:
            st.warning("Monte Carlo unavailable for this ticker.")

        # Options analysis
        with st.spinner("Fetching option chains and computing fair prices..."):
            try:
                opts = analyze_options_for_ticker(ticker, max_exp_days=max_exp_days, min_premium=min_premium)
                if opts.empty:
                    st.info("No options matched the filters (expiration or premium).")
                else:
                    buy_calls = opts[(opts["type"] == "call") & (opts["pct_diff"] < 0)].sort_values("pct_diff").reset_index(drop=True)
                    buy_puts = opts[(opts["type"] == "put") & (opts["pct_diff"] < 0)].sort_values("pct_diff").reset_index(drop=True)
                    sell_calls = opts[(opts["type"] == "call") & (opts["pct_diff"] > 0)].sort_values("pct_diff", ascending=False).reset_index(drop=True)
                    sell_puts = opts[(opts["type"] == "put") & (opts["pct_diff"] > 0)].sort_values("pct_diff", ascending=False).reset_index(drop=True)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.subheader("Buy Calls (underpriced)")
                        st.dataframe(buy_calls.head(10)[["contract", "strike", "expiration", "dte", "current", "fair", "pct_diff"]].rename(
                            columns={"contract": "Contract", "strike": "Strike", "expiration": "Expiration", "dte": "DTE",
                                     "current": "Current Price", "fair": "Fair Price", "pct_diff": "% Diff"}))
                    with c2:
                        st.subheader("Buy Puts (underpriced)")
                        st.dataframe(buy_puts.head(10)[["contract", "strike", "expiration", "dte", "current", "fair", "pct_diff"]].rename(
                            columns={"contract": "Contract", "strike": "Strike", "expiration": "Expiration", "dte": "DTE",
                                     "current": "Current Price", "fair": "Fair Price", "pct_diff": "% Diff"}))
                    with c3:
                        st.subheader("Sell Calls (overpriced)")
                        st.dataframe(sell_calls.head(10)[["contract", "strike", "expiration", "dte", "current", "fair", "pct_diff"]].rename(
                            columns={"contract": "Contract", "strike": "Strike", "expiration": "Expiration", "dte": "DTE",
                                     "current": "Current Price", "fair": "Fair Price", "pct_diff": "% Diff"}))
                    with c4:
                        st.subheader("Sell Puts (overpriced)")
                        st.dataframe(sell_puts.head(10)[["contract", "strike", "expiration", "dte", "current", "fair", "pct_diff"]].rename(
                            columns={"contract": "Contract", "strike": "Strike", "expiration": "Expiration", "dte": "DTE",
                                     "current": "Current Price", "fair": "Fair Price", "pct_diff": "% Diff"}))

                    st.download_button("Download matched options (CSV)", opts.to_csv(index=False), file_name=f"{ticker}_matched_options.csv")
            except Exception as e:
                st.error(f"Options analysis failed: {e}")

        # Multi-factor scoring & bar chart
        st.subheader("Multi-Factor Score (50 = median)")

        if "factor_df" not in st.session_state:
            st.info("Build the S&P500 factor universe in the sidebar to get percentile-based scores. A quick single-stock estimate is below.")
            # quick single-stock heuristic estimate
            try:
                tk = yf.Ticker(yf_symbol(ticker))
                hist = fetch_history(yf_symbol(ticker), period="1y")
                if hist.empty or "Close" not in hist.columns:
                    st.warning("Insufficient price data for quick factor estimate.")
                else:
                    close = hist["Close"].dropna()
                    momentum = float(close.iloc[-1] / close.iloc[0] - 1.0)
                    vol = float(close.pct_change().dropna().std() * np.sqrt(252))
                    info = tk.info
                    pe = info.get("trailingPE", np.nan)
                    quality = info.get("returnOnEquity", np.nan)
                    if pd.isna(quality):
                        quality = info.get("profitMargins", np.nan)
                    m_score = np.clip((momentum + 0.5) * 100, 0, 100)
                    vol_score = np.clip((0.35 - vol) * 200, 0, 100)
                    val_score = np.clip((50 - (pe if not pd.isna(pe) else 50)) * 2, 0, 100)
                    q_score = np.clip((quality if not pd.isna(quality) else 0.1) * 100, 0, 100)
                    overall = np.nanmean([m_score, vol_score, val_score, q_score])
                    st.metric("Estimated Score (heuristic)", f"{overall:.1f}")
                    bf = pd.Series({"Momentum": m_score, "Low Volatility": vol_score, "Value": val_score, "Quality": q_score})
                    figf, axf = plt.subplots(figsize=(6, 3))
                    bf.plot(kind="bar", ax=axf, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
                    axf.axhline(50, color="gray", linestyle="--")
                    axf.set_ylim(0, 100)
                    axf.set_ylabel("Score (0-100; 50=avg)")
                    st.pyplot(figf)
            except Exception as e:
                st.warning(f"Quick factor estimate failed: {e}")
        else:
            factor_scored = st.session_state["factor_df"]
            if ticker.upper() in factor_scored.index:
                row = factor_scored.loc[ticker.upper()]
                overall = float(row["overall_score"])
                st.metric("Overall Score", f"{overall:.1f}")
                factors = {"Momentum": float(row["momentum_score"]),
                           "Low Volatility": float(row["volatility_score"]),
                           "Value": float(row["value_score"]),
                           "Quality": float(row["quality_score"])}
                bf = pd.Series(factors)
                figf, axf = plt.subplots(figsize=(6, 3))
                bf.plot(kind="bar", ax=axf, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
                axf.axhline(50, color="gray", linestyle="--", label="Median (50)")
                axf.set_ylim(0, 100)
                axf.set_ylabel("Score (0-100)")
                for idx, val in enumerate(bf.values):
                    axf.text(idx, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
                axf.legend()
                st.pyplot(figf)
            else:
                st.warning("Ticker not found in built universe or had incomplete factor data. Build/refresh the universe to compute percentile-based scores.")

        st.success("Analysis complete.")
