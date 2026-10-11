"""Streamlit app: portfolio VaR / Expected Shortfall with three methods.
Run:  streamlit run app.py
"""
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import riskkit as rk

st.set_page_config(page_title="Portfolio VaR", layout="wide")
st.title("Portfolio Value at Risk and Expected Shortfall")
st.caption("One-day risk from Historical, Parametric (Normal) and GARCH(1,1)-t methods, with a backtest.")

with st.sidebar:
    st.header("Inputs")
    tickers = [t.strip().upper() for t in st.text_input(
        "Tickers (comma separated)", "RELIANCE.NS, HDFCBANK.NS, INFY.NS, TCS.NS, ICICIBANK.NS").split(",") if t.strip()]
    wtxt = st.text_input("Weights (same order, blank = equal)", "")
    start = st.date_input("Start date", dt.date(2014, 1, 1),
                          min_value=dt.date(2005, 1, 1), max_value=dt.date(2040, 12, 31), key="start_v3")
    latest = st.checkbox("Use the latest available data", value=True)
    asof = None
    if not latest:
        asof = st.date_input("As-of date (use data up to this day)", dt.date(2025, 12, 31),
                             min_value=dt.date(2006, 1, 1), max_value=dt.date(2040, 12, 31), key="asof_v3")
    alpha = st.select_slider("Confidence level", [0.90, 0.95, 0.975, 0.99], value=0.99)
    value = st.number_input("Portfolio value", min_value=1000, value=10_000_000, step=100_000)
    run_bt = st.checkbox("Run backtest (slower, ~1 min)", value=False)

try:
    w = [float(x) for x in wtxt.split(",")] if wtxt.strip() else [1 / len(tickers)] * len(tickers)
    if len(w) != len(tickers):
        st.error("Number of weights must match number of tickers."); st.stop()
except ValueError:
    st.error("Weights must be numbers."); st.stop()


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def get_prices(tk, s, e):
    # e is None for "latest". yfinance's end date is exclusive, so add one day when given.
    end = None if e is None else (pd.Timestamp(e) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache = "data/prices.csv" if e is None else None   # CSV fallback only for latest-data runs
    return rk.load_prices(list(tk), str(s), end, cache_path=cache)


try:
    prices = get_prices(tuple(tickers), start, asof)
except Exception as e:
    st.error(f"Could not load prices: {e}"); st.stop()

if asof is not None:
    prices = prices.loc[:pd.Timestamp(asof)]
r = rk.portfolio_returns(prices, w)
if len(r) < 300:
    st.error("Need at least ~300 trading days. Use an earlier start date."); st.stop()
st.caption(f"Data used: {prices.index[0].date()} to {prices.index[-1].date()} "
           f"({len(prices)} trading days). The VaR below is the estimate for the NEXT trading day.")

tbl = rk.var_table(r, alpha, value)
c1, c2, c3 = st.columns(3)
for col, m in zip((c1, c2, c3), tbl.index):
    col.metric(f"{m}  {alpha:.1%} VaR", f"{value * tbl.loc[m, 'VaR']:,.0f}",
               f"{tbl.loc[m, 'VaR_%']:.2f}% of portfolio", delta_color="off")
st.dataframe(tbl[["VaR_%", "ES_%", "VaR_money", "ES_money"]].round(2))

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.hist(r * 100, bins=80, color="#cfd8e6", edgecolor="white", linewidth=0.3)
for m, col in rk.COLORS.items():
    ax.axvline(-tbl.loc[m, "VaR_%"], color=col, lw=1.8, label=f"{m}: {tbl.loc[m, 'VaR_%']:.2f}%")
ax.spines[["top", "right"]].set_visible(False); ax.set_xlabel("Daily return (%)"); ax.set_ylabel("Days")
ax.legend(frameon=False, fontsize=8)
st.pyplot(fig)

with st.expander("Return statistics"):
    st.dataframe(rk.return_stats(r).to_frame("value").round(4))

if run_bt:
    with st.spinner("Backtesting..."):
        bt = rk.rolling_backtest(r, alpha, 750, 20)
    st.subheader("Backtest")
    st.dataframe(rk.backtest_summary(bt, alpha).round(4))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(bt.index, bt["return"] * 100, color="#b9b8b2", lw=0.5)
    for m, col in rk.COLORS.items():
        ax.plot(bt.index, -bt[m] * 100, color=col, lw=1.2, label=m)
    ax.spines[["top", "right"]].set_visible(False); ax.set_ylabel("Return / -VaR (%)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    st.pyplot(fig)
st.caption("Educational project, not investment advice.")