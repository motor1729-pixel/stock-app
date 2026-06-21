from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from backtest import BacktestConfig, performance_metrics, run_backtest


st.set_page_config(page_title="국내주식 전략 실험실", page_icon="📈", layout="wide")


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        raise ValueError("시세를 찾지 못했습니다. 종목코드와 시장을 확인해 주세요.")
    return data


def stock_ticker(code: str, market: str) -> str:
    normalized = "".join(character for character in code if character.isdigit())
    if len(normalized) != 6:
        raise ValueError("종목코드는 숫자 6자리로 입력해 주세요. 예: 005930")
    suffix = ".KS" if market == "KOSPI" else ".KQ"
    return normalized + suffix


def price_chart(result: pd.DataFrame, ticker: str) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(x=result.index, y=result["Close"], name="수정 종가", line=dict(width=2))
    )
    chart.add_trace(
        go.Scatter(x=result.index, y=result["ShortMA"], name="단기 이동평균")
    )
    chart.add_trace(
        go.Scatter(x=result.index, y=result["LongMA"], name="장기 이동평균")
    )
    entries = result[result["Entry"]]
    exits = result[result["Exit"]]
    chart.add_trace(
        go.Scatter(
            x=entries.index,
            y=entries["Close"],
            mode="markers",
            name="매수 신호",
            marker=dict(symbol="triangle-up", size=11, color="#00a878"),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=exits.index,
            y=exits["Close"],
            mode="markers",
            name="매도 신호",
            marker=dict(symbol="triangle-down", size=11, color="#e45756"),
        )
    )
    chart.update_layout(
        title=f"{ticker} 가격과 매매 신호",
        yaxis_title="가격(원)",
        xaxis_title="날짜",
        hovermode="x unified",
        legend_orientation="h",
        height=520,
    )
    return chart


st.title("국내주식 전략 실험실")
st.caption("이동평균 전략을 과거 데이터로 실험하는 교육용 도구입니다. 투자 권유가 아닙니다.")

with st.sidebar:
    st.header("실험 설정")
    code = st.text_input("종목코드", value="005930", help="삼성전자: 005930")
    market = st.radio("시장", ["KOSPI", "KOSDAQ"], horizontal=True)
    period = st.selectbox(
        "분석 기간",
        options=["6mo", "1y", "3y", "5y", "10y"],
        index=2,
        format_func={
            "6mo": "6개월",
            "1y": "1년",
            "3y": "3년",
            "5y": "5년",
            "10y": "10년",
        }.get,
    )
    short_window = st.number_input("단기 이동평균(일)", 2, 200, 20)
    long_window = st.number_input("장기 이동평균(일)", 3, 400, 60)
    trading_cost = st.number_input(
        "편도 거래비용(%)",
        min_value=0.0,
        max_value=2.0,
        value=0.02,
        step=0.01,
        help="수수료·세금·슬리피지를 합친 가정값입니다.",
    )
    run = st.button("백테스트 실행", type="primary", width="stretch")

if run:
    try:
        ticker = stock_ticker(code, market)
        with st.spinner("가격 데이터를 불러오고 있습니다..."):
            prices = load_prices(ticker, period)
        config = BacktestConfig(
            short_window=int(short_window),
            long_window=int(long_window),
            trading_cost_pct=float(trading_cost),
        )
        result = run_backtest(prices, config)
        metrics = performance_metrics(result)
    except Exception as error:
        st.error(str(error))
        st.stop()

    columns = st.columns(5)
    columns[0].metric("전략 누적수익률", f"{metrics['total_return']:.1%}")
    columns[1].metric("단순 보유 수익률", f"{metrics['buy_hold_return']:.1%}")
    columns[2].metric("연환산 수익률", f"{metrics['cagr']:.1%}")
    columns[3].metric("최대 낙폭(MDD)", f"{metrics['max_drawdown']:.1%}")
    columns[4].metric("매수 횟수", f"{metrics['trade_count']}회")

    st.plotly_chart(price_chart(result, ticker), width="stretch")

    equity = result[["StrategyEquity", "BuyHoldEquity"]].rename(
        columns={"StrategyEquity": "이동평균 전략", "BuyHoldEquity": "단순 보유"}
    )
    st.subheader("1원을 투자했을 때의 자산 변화")
    st.line_chart(equity)

    with st.expander("상세 성과와 최근 데이터"):
        st.write(f"연환산 변동성: **{metrics['annual_volatility']:.1%}**")
        st.write(f"투자 중 상승일 비율: **{metrics['positive_day_ratio']:.1%}**")
        table = result[["Close", "ShortMA", "LongMA", "Position", "StrategyEquity"]]
        st.dataframe(table.tail(30).sort_index(ascending=False), width="stretch")
else:
    st.info("왼쪽에서 종목과 전략을 설정한 뒤 ‘백테스트 실행’을 눌러 주세요.")
    st.markdown(
        "**예시 종목코드:** 삼성전자 `005930` · SK하이닉스 `000660` · "
        "현대차 `005380` · 셀트리온 `068270`"
    )

st.divider()
st.caption(
    "가격은 Yahoo Finance의 수정 종가를 사용하며 지연·누락될 수 있습니다. "
    "백테스트 결과는 미래 수익을 보장하지 않습니다."
)
