"""
riskkit.py - market risk toolkit (Value at Risk, Expected Shortfall, GARCH, backtesting).

Conventions used everywhere in this file
- Returns are DECIMAL fractions (0.01 = 1%), unless a name ends in `_pct`.
- VaR and ES are returned as POSITIVE LOSS fractions (0.025 = "lose 2.5%").
- `alpha` is the CONFIDENCE level (0.99 = 99% VaR), so the tail probability is 1 - alpha.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy
from arch import arch_model

# Colours: first three slots of a validated categorical palette (blue, orange, aqua)
COLORS = {"Historical": "#2a78d6", "Parametric (Normal)": "#eb6834", "GARCH-t": "#1baf7a"}


# --------------------------------------------------------------------------- data
def simulate_prices(tickers, start, end, seed=7):
    """Fake prices with volatility clustering and fat tails.
    ONLY for testing the code offline. Never report results from this."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n, nu = len(idx), 6
    omega, a, b = 0.05, 0.10, 0.88  # GARCH(1,1) in %^2 units

    def garch_path():
        r, s2 = np.zeros(n), omega / (1 - a - b)
        for t in range(n):
            z = rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
            r[t] = np.sqrt(s2) * z
            s2 = omega + a * r[t] ** 2 + b * s2
        return r

    market = garch_path()
    cols = {}
    for t in tickers:
        idio = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu) * 0.6
        ret = (0.8 * market + idio + 0.04) / 100
        cols[t] = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame(cols, index=idx)


def load_prices(tickers, start, end=None, cache_path=None, simulated=False):
    """Download adjusted close prices with yfinance. Falls back to a CSV cache if the
    download fails. `simulated=True` returns fake data (offline testing only)."""
    tickers = list(tickers)
    if simulated:
        return simulate_prices(tickers, start, end or pd.Timestamp.today().strftime("%Y-%m-%d"))
    try:
        import yfinance as yf

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = close.dropna(how="all")
        missing = [t for t in tickers if t not in close.columns]
        if close.empty or missing:
            raise ValueError(f"No data for: {missing or tickers}")
        close = close[tickers].dropna()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            close.to_csv(cache_path)
        return close
    except Exception as exc:  # network down, bad ticker, rate limit...
        if cache_path and os.path.exists(cache_path):
            warnings.warn(f"Download failed ({exc}). Using cached file {cache_path}.")
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)
        raise RuntimeError(f"Could not download prices ({exc}) and no cache at {cache_path}.") from exc


def portfolio_returns(prices, weights):
    """Daily simple returns of a portfolio rebalanced to fixed weights every day."""
    w = np.asarray(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    simple = prices.pct_change().dropna()
    return (simple * w).sum(axis=1).rename("portfolio")


# --------------------------------------------------------------------------- descriptive stats
def return_stats(r):
    """Headline statistics that justify using GARCH-t instead of a plain Normal."""
    r = r.dropna()
    jb_stat, jb_p = stats.jarque_bera(r)
    return pd.Series({
        "observations": len(r),
        "mean_daily_%": r.mean() * 100,
        "std_daily_%": r.std() * 100,
        "annualised_vol_%": r.std() * np.sqrt(252) * 100,
        "skew": stats.skew(r),
        "excess_kurtosis": stats.kurtosis(r),  # 0 for a Normal
        "worst_day_%": r.min() * 100,
        "best_day_%": r.max() * 100,
        "jarque_bera_p": jb_p,  # < 0.05 => reject normality
    })


# --------------------------------------------------------------------------- VaR / ES
def hist_var_es(r, alpha=0.99):
    """Historical simulation: read the loss straight off the empirical distribution."""
    r = np.asarray(r)
    q = np.quantile(r, 1 - alpha)
    return -q, -r[r <= q].mean()


def param_var_es(r, alpha=0.99):
    """Variance-covariance method assuming Normal returns."""
    r = np.asarray(r)
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)  # negative
    var = -(mu + sd * z)
    es = -mu + sd * stats.norm.pdf(z) / (1 - alpha)
    return var, es


def fit_garch(r):
    """GARCH(1,1) with a constant mean and Student-t innovations. `r` in decimals."""
    am = arch_model(np.asarray(r) * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    return am.fit(disp="off")


def garch_var_es(mu_pct, sigma_pct, nu, alpha=0.99):
    """VaR/ES for one day given a conditional mean/vol (in %) and t-dof `nu`.
    The t quantile is rescaled to unit variance so `sigma` keeps its meaning."""
    scale = np.sqrt((nu - 2) / nu)
    q = stats.t.ppf(1 - alpha, nu)  # negative
    var = -(mu_pct + sigma_pct * scale * q)
    es_unit = stats.t.pdf(q, nu) / (1 - alpha) * (nu + q ** 2) / (nu - 1) * scale
    es = -mu_pct + sigma_pct * es_unit
    return var / 100, es / 100


def garch_next_day(r, alpha=0.99):
    """Fit GARCH-t on `r` and return next-day (VaR, ES, sigma_pct, fitted result)."""
    res = fit_garch(r)
    sigma = float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0]))
    var, es = garch_var_es(res.params["mu"], sigma, res.params["nu"], alpha)
    return var, es, sigma, res


def var_table(r, alpha=0.99, value=1_000_000):
    """One-day VaR and ES for all three methods, as fractions and in money."""
    hv, he = hist_var_es(r, alpha)
    pv, pe = param_var_es(r, alpha)
    gv, ge, _, _ = garch_next_day(r, alpha)
    t = pd.DataFrame(
        {"VaR": [hv, pv, gv], "ES": [he, pe, ge]},
        index=["Historical", "Parametric (Normal)", "GARCH-t"],
    )
    t["VaR_%"] = t["VaR"] * 100
    t["ES_%"] = t["ES"] * 100
    t["VaR_money"] = t["VaR"] * value
    t["ES_money"] = t["ES"] * value
    return t


# --------------------------------------------------------------------------- backtesting
def rolling_backtest(r, alpha=0.99, window=750, refit_every=20):
    """Walk forward through history. For each day, produce the VaR a risk manager would
    have computed the evening before, using ONLY past data. GARCH parameters are refit
    every `refit_every` days; in between, the volatility is updated with fixed parameters."""
    r = r.dropna()
    n = len(r)
    if n <= window + 50:
        raise ValueError(f"Need more than {window + 50} returns, got {n}. Use an earlier start date.")
    pct = r * 100

    # Historical and Parametric: rolling window, shifted so day t only sees data up to t-1
    roll = r.rolling(window)
    var_hist = -roll.quantile(1 - alpha).shift(1)
    var_param = -(roll.mean() + roll.std() * stats.norm.ppf(1 - alpha)).shift(1)

    # GARCH-t: refit in blocks, filter volatility forward inside each block
    garch_var = pd.Series(np.nan, index=r.index)
    garch_sigma = pd.Series(np.nan, index=r.index)
    for k in range(window, n, refit_every):
        end = min(k + refit_every, n)
        res = fit_garch(r.iloc[k - window:k])
        am = arch_model(pct.iloc[k - window:end], mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        fixed = am.fix(res.params)
        f = fixed.forecast(start=window - 1, horizon=1, reindex=False)
        v = f.variance.iloc[: end - k, 0].values  # forecast for days k .. end-1
        sig = np.sqrt(v)
        mu, nu = res.params["mu"], res.params["nu"]
        garch_var.iloc[k:end] = [garch_var_es(mu, s, nu, alpha)[0] for s in sig]
        garch_sigma.iloc[k:end] = sig

    out = pd.DataFrame({
        "return": r,
        "Historical": var_hist,
        "Parametric (Normal)": var_param,
        "GARCH-t": garch_var,
        "garch_sigma_pct": garch_sigma,
    }).iloc[window:]
    return out


def kupiec_pof(breaches, alpha):
    """Kupiec proportion-of-failures test: is the breach RATE what the confidence level promises?"""
    b = np.asarray(breaches, dtype=int)
    n, x, p = len(b), b.sum(), 1 - alpha
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    pi = x / n
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen_ind(breaches):
    """Christoffersen independence test: do breaches CLUSTER (bad - model is slow to react)?"""
    b = np.asarray(breaches, dtype=int)
    prev, cur = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (cur == 0))
    n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0))
    n11 = np.sum((prev == 1) & (cur == 1))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def basel_zone(breaches_per_250):
    """Basel 'traffic light' for 99% VaR, scaled to a 250-day year."""
    if breaches_per_250 <= 4:
        return "green"
    return "yellow" if breaches_per_250 <= 9 else "red"


def backtest_summary(bt, alpha=0.99):
    """One row per method: breaches vs expected, Kupiec, Christoffersen, verdict."""
    rows = []
    for m in ["Historical", "Parametric (Normal)", "GARCH-t"]:
        d = bt[["return", m]].dropna()
        breach = (d["return"] < -d[m]).astype(int).values
        n, x = len(breach), int(breach.sum())
        _, p_pof = kupiec_pof(breach, alpha)
        _, p_ind = christoffersen_ind(breach)
        per250 = x / n * 250
        rows.append({
            "method": m, "days": n, "breaches": x,
            "expected": round(n * (1 - alpha), 1), "breach_rate_%": x / n * 100,
            "per_250_days": per250, "basel_zone": basel_zone(per250) if alpha == 0.99 else "n/a",
            "kupiec_p": p_pof, "independence_p": p_ind,
            "verdict": "PASS" if (p_pof > 0.05 and p_ind > 0.05) else "FAIL",
        })
    return pd.DataFrame(rows).set_index("method")
