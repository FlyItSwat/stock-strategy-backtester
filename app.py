import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from backtester import backtest, metrics

st.set_page_config(page_title="Stock Strategy Backtester",page_icon="📉",layout="wide")
st.title("📉 Moving-Average Strategy Backtester")
st.caption("Long-or-cash strategy versus buy-and-hold | educational research tool")
with st.sidebar:
    st.header("Inputs")
    ticker=st.text_input("Yahoo Finance ticker",value="SPY").strip().upper()
    start=st.date_input("Start date",value=pd.Timestamp("2018-01-01"))
    end=st.date_input("End date",value=pd.Timestamp.today())
    short=st.number_input("Fast moving average (days)",min_value=2,max_value=200,value=20)
    long=st.number_input("Slow moving average (days)",min_value=3,max_value=400,value=50)
    fee=st.number_input("Transaction fee (basis points per trade)",min_value=0.0,max_value=200.0,value=10.0)
    capital=st.number_input("Starting capital",min_value=1000.0,value=100000.0,step=1000.0)
    uploaded=st.file_uploader("Or upload daily CSV with Date and Close columns",type="csv")
    go_button=st.button("Run backtest",type="primary")
@st.cache_data(ttl=3600)
def fetch_prices(symbol,start_date,end_date):
    import yfinance as yf
    raw=yf.download(symbol,start=start_date,end=end_date,auto_adjust=True,progress=False)
    if raw.empty:
        raise ValueError("No prices returned. Check ticker and dates.")
    close=raw["Close"]
    if isinstance(close,pd.DataFrame):
        close=close.iloc[:,0]
    return close.rename("Close")
if go_button:
    try:
        if short>=long:
            raise ValueError("Fast window must be shorter than slow window.")
        if start>=end:
            raise ValueError("Start date must precede end date.")
        if uploaded is not None:
            raw=pd.read_csv(uploaded)
            if not {"Date","Close"}.issubset(raw.columns):
                raise ValueError("CSV needs Date and Close columns.")
            raw["Date"]=pd.to_datetime(raw["Date"],errors="raise")
            raw["Close"]=pd.to_numeric(raw["Close"],errors="raise")
            raw=raw.sort_values("Date").drop_duplicates("Date",keep="last").set_index("Date")
            prices=raw.loc[str(start):str(end),"Close"]
        else:
            if not ticker:
                raise ValueError("Enter a ticker.")
            prices=fetch_prices(ticker,str(start),str(end))
        frame=backtest(prices,short,long,fee,capital)
        s=metrics(frame.strategy_equity,frame.strategy_return)
        b=metrics(frame.buyhold_equity,frame.buyhold_return)
        st.subheader("Results")
        cols=st.columns(4)
        cols[0].metric("Strategy total return",f"{s['total_return']:.1%}")
        cols[1].metric("Buy & hold total return",f"{b['total_return']:.1%}")
        cols[2].metric("Strategy max drawdown",f"{s['max_drawdown']:.1%}")
        cols[3].metric("Strategy Sharpe (0% RF)",f"{s['sharpe_zero_rf']:.2f}")
        summary=pd.DataFrame({"Strategy":s,"Buy and hold":b})
        st.dataframe(summary.style.format("{:.2%}",subset=["Strategy","Buy and hold"],
                                          subset=pd.IndexSlice[["total_return","cagr","annual_volatility","max_drawdown"],:]))
        fig=go.Figure()
        for name,col in [("Strategy","strategy_equity"),("Buy & hold","buyhold_equity")]:
            fig.add_trace(go.Scatter(x=frame.index,y=frame[col],name=name))
        fig.update_layout(title="Equity curves",xaxis_title="Date",yaxis_title="Portfolio value")
        st.plotly_chart(fig,use_container_width=True)
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=frame.index,y=frame.price,name="Price"))
        fig2.add_trace(go.Scatter(x=frame.index,y=frame.fast_ma,name="Fast MA"))
        fig2.add_trace(go.Scatter(x=frame.index,y=frame.slow_ma,name="Slow MA"))
        fig2.update_layout(title="Price and moving averages")
        st.plotly_chart(fig2,use_container_width=True)
        st.download_button("Download backtest CSV",frame.to_csv(index_label="Date"),"backtest.csv","text/csv")
        st.warning("Historical results are not forecasts. Assumes signals generated at the close "
                   "and exposure applies to the next close-to-close interval; execution timing "
                   "and real-world fills may differ. No taxes, spreads, slippage, dividends on "
                   "cash, or interest on cash. Sharpe assumes 0% risk-free rate.")
    except Exception as exc:
        st.error(f"Unable to run backtest: {exc}")
else:
    st.info("Set inputs and click **Run backtest**. You can use a CSV without an internet connection.")
