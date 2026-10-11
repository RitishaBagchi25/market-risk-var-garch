# Market Risk: GARCH Volatility, VaR and Expected Shortfall with Backtesting

Estimates one-day Value at Risk and Expected Shortfall for an Indian equity portfolio three ways
(Historical, Parametric Normal, GARCH(1,1) with Student-t errors) and **backtests** each with Kupiec,
Christoffersen and Basel traffic-light tests.

## Run
```
pip install -r requirements.txt
jupyter notebook market_risk_var_garch.ipynb     # Run All
streamlit run app.py
```
Prices come from yfinance and are cached in `data/prices.csv` (used automatically if a download fails).

## Files
- `riskkit.py` functions: data, stats, VaR/ES, GARCH, backtest, tests
- `market_risk_var_garch.ipynb` the analysis, with charts saved in `outputs/`
- `app.py` Streamlit app

## Method notes
- Returns are decimal simple returns; portfolio rebalanced daily to fixed weights.
- Backtest: 750-day rolling window, GARCH refit every 20 days, volatility filtered daily in between. Only past data is used for each day's VaR.
- Pass = Kupiec p > 0.05 and independence p > 0.05.


