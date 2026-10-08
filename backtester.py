"""Look-ahead-safe daily close-to-close moving-average backtester."""
import numpy as np
import pandas as pd

def backtest(prices, short_window=20, long_window=50, fee_bps=10, initial_capital=100000):
    """Signals at close t are executed for close-to-close return t+1.

    Cost model: one-way proportional turnover cost applied when exposure changes.
    Uses adjusted close data where available; does not model slippage or taxes.
    """
    if not 1 <= short_window < long_window:
        raise ValueError("Require 1 <= short_window < long_window")
    if fee_bps < 0 or initial_capital <= 0:
        raise ValueError("Invalid fees or capital")
    p=pd.Series(prices,dtype=float).dropna()
    if len(p) < long_window+2 or (p<=0).any():
        raise ValueError("Need enough strictly positive price observations")
    fast=p.rolling(short_window,min_periods=short_window).mean()
    slow=p.rolling(long_window,min_periods=long_window).mean()
    signal=(fast>slow).astype(int)
    signal.loc[slow.isna()]=0
    exposure=signal.shift(1).fillna(0).astype(int)
    returns=p.pct_change().fillna(0)
    turnover=exposure.diff().abs().fillna(exposure.abs())
    strategy_returns=exposure*returns-turnover*fee_bps/10000
    buyhold_returns=returns
    frame=pd.DataFrame({"price":p,"fast_ma":fast,"slow_ma":slow,"signal":signal,
                        "exposure":exposure,"asset_return":returns,
                        "turnover":turnover,"strategy_return":strategy_returns,
                        "buyhold_return":buyhold_returns})
    frame["strategy_equity"]=initial_capital*(1+strategy_returns).cumprod()
    frame["buyhold_equity"]=initial_capital*(1+buyhold_returns).cumprod()
    return frame

def metrics(equity, daily_returns, annualization=252):
    e=pd.Series(equity,dtype=float)
    r=pd.Series(daily_returns,dtype=float)
    if len(e)<2 or e.iloc[0]<=0 or (e<=0).any():
        raise ValueError("Equity must remain positive")
    # Include the initial starting capital in drawdown calculations.
    initial=e.iloc[0]/(1+r.iloc[0]) if 1+r.iloc[0]>0 else e.iloc[0]
    peak=pd.concat([pd.Series([initial]),e.reset_index(drop=True)],ignore_index=True).cummax()
    drawdown=pd.concat([pd.Series([initial]),e.reset_index(drop=True)],ignore_index=True)/peak-1
    years=len(r)/annualization
    cagr=(e.iloc[-1]/initial)**(1/years)-1
    vol=r.std(ddof=1)*np.sqrt(annualization) if len(r)>1 else 0
    sharpe=r.mean()*annualization/vol if vol>0 else float("nan")
    return {"total_return":e.iloc[-1]/initial-1,"cagr":cagr,
            "annual_volatility":vol,"sharpe_zero_rf":sharpe,
            "max_drawdown":drawdown.min()}
