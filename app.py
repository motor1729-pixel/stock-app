from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from backtest import BacktestConfig, performance_metrics, run_backtest


st.set_page_config(page_title="국내주식 전략 실험실", layout="wide")

PERIOD_DAYS = {
    "6mo": 140,
    "1y": 260,
    "3y": 780,
    "5y": 1300,
    "10y": 2600,
}
KST = timezone(timedelta(hours=9))


def _parse_naver_chart(content: bytes) -> pd.DataFrame:
    xml_text = content.decode("euc-kr")
    if xml_text.lstrip().startswith("<?xml"):
        xml_text = xml_text.split("?>", 1)[1]
    root = ElementTree.fromstring(xml_text)
    chart_data = root.find(".//chartdata")
    rows = []
    for item in root.findall(".//item"):
        values = item.attrib.get("data", "").split("|")
        if len(values) == 6:
            rows.append(values)
    if not rows:
        raise ValueError("시세 데이터가 없습니다.")

    data = pd.DataFrame(
        rows,
        columns=["Date", "Open", "High", "Low", "Close", "Volume"],
    )
    data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.set_index("Date").dropna(subset=["Close"]).sort_index()
    data.attrs["name"] = (
        chart_data.attrib.get("name", "") if chart_data is not None else ""
    )
    return data


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(symbol: str, period: str) -> pd.DataFrame:
    response = requests.get(
        "https://fchart.stock.naver.com/sise.nhn",
        params={
            "symbol": symbol,
            "timeframe": "day",
            "count": PERIOD_DAYS[period],
            "requestType": 0,
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    return _parse_naver_chart(response.content)


@st.cache_data(ttl=1800, show_spinner=False)
def load_news(company_name: str, limit: int = 10) -> list[dict[str, str]]:
    response = requests.get(
        "https://news.google.com/rss/search",
        params={
            "q": f'"{company_name}" 주식',
            "hl": "ko",
            "gl": "KR",
            "ceid": "KR:ko",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    news = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "제목 없음").strip()
        link = (item.findtext("link") or "").strip()
        source_element = item.find("source")
        source = (
            source_element.text.strip()
            if source_element is not None and source_element.text
            else "출처 미표시"
        )
        published = item.findtext("pubDate") or ""
        try:
            published_at = parsedate_to_datetime(published).astimezone(KST)
            published_text = published_at.strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            published_text = published
        news.append(
            {
                "title": title,
                "link": link,
                "source": source,
                "published": published_text,
            }
        )
    return news


def normalize_code(code: str) -> str:
    normalized = "".join(character for character in code if character.isdigit())
    if len(normalized) != 6:
        raise ValueError("종목코드는 숫자 6자리로 입력해 주세요. 예: 005930")
    return normalized


def calculate_rsi(close: pd.Series, window: int = 14) -> float:
    change = close.diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = -change.clip(upper=0).rolling(window).mean()
    latest_loss = float(loss.iloc[-1])
    if latest_loss == 0:
        return 100.0
    relative_strength = float(gain.iloc[-1]) / latest_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def technical_summary(
    result: pd.DataFrame, market_prices: pd.DataFrame | None
) -> dict[str, float | str]:
    close = result["Close"]
    current = float(close.iloc[-1])
    short_ma = float(result["ShortMA"].iloc[-1])
    long_ma = float(result["LongMA"].iloc[-1])
    if current > short_ma > long_ma:
        trend = "상승 정렬"
    elif current < short_ma < long_ma:
        trend = "하락 정렬"
    else:
        trend = "혼조 구간"

    rsi = calculate_rsi(close)
    if rsi >= 70:
        rsi_state = "과열권"
    elif rsi <= 30:
        rsi_state = "침체권"
    else:
        rsi_state = "중립권"

    recent = close.tail(252)
    yearly_low = float(recent.min())
    yearly_high = float(recent.max())
    high_low_position = (
        (current - yearly_low) / (yearly_high - yearly_low)
        if yearly_high > yearly_low
        else 0.5
    )
    stock_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    market_20d = 0.0
    if market_prices is not None and len(market_prices) > 20:
        market_20d = float(market_prices["Close"].pct_change(20).iloc[-1])
    relative_20d = stock_20d - market_20d
    relative_state = "시장 대비 우위" if relative_20d >= 0 else "시장 대비 열위"

    return {
        "trend": trend,
        "rsi": rsi,
        "rsi_state": rsi_state,
        "yearly_low": yearly_low,
        "yearly_high": yearly_high,
        "high_low_position": high_low_position,
        "stock_20d": stock_20d,
        "market_20d": market_20d,
        "relative_20d": relative_20d,
        "relative_state": relative_state,
    }


def price_chart(result: pd.DataFrame, name: str) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(x=result.index, y=result["Close"], name="종가", line=dict(width=2))
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
            name="매수 판단",
            marker=dict(symbol="triangle-up", size=11, color="#00a878"),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=exits.index,
            y=exits["Close"],
            mode="markers",
            name="매도 판단",
            marker=dict(symbol="triangle-down", size=11, color="#e45756"),
        )
    )
    chart.update_layout(
        title=f"{name} 가격과 이동평균 신호",
        yaxis_title="가격(원)",
        xaxis_title="날짜",
        hovermode="x unified",
        legend_orientation="h",
        height=500,
    )
    return chart


def won(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:,.0f}원"


st.title("국내주식 전략 실험실")
st.caption(
    "1주 기준 이동평균 백테스트와 시장·뉴스 점검 도구입니다. "
    "표시 내용은 투자 권유가 아닙니다."
)

with st.sidebar:
    st.header("실험 설정")
    code_input = st.text_input("종목코드", value="005930", help="삼성전자: 005930")
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
    run = st.button("분석 실행", type="primary", width="stretch")

if not run:
    st.info("왼쪽에서 종목과 전략을 설정한 뒤 ‘분석 실행’을 눌러 주세요.")
    st.markdown(
        "**예시 종목코드:** 삼성전자 `005930` · SK하이닉스 `000660` · "
        "현대차 `005380` · 셀트리온 `068270`"
    )
    st.stop()

try:
    code = normalize_code(code_input)
    with st.spinner("가격과 시장 데이터를 불러오고 있습니다..."):
        prices = load_prices(code, period)
        company_name = prices.attrs.get("name") or code
        try:
            market_prices = load_prices(market, "1y")
        except (requests.RequestException, ValueError, ElementTree.ParseError):
            market_prices = None
    config = BacktestConfig(
        short_window=int(short_window),
        long_window=int(long_window),
        trading_cost_pct=float(trading_cost),
    )
    result = run_backtest(prices, config)
    metrics = performance_metrics(result)
    summary = technical_summary(result, market_prices)
except (requests.RequestException, ValueError, ElementTree.ParseError) as error:
    st.error(f"데이터를 처리하지 못했습니다: {error}")
    st.stop()

current_close = float(result["Close"].iloc[-1])
previous_close = float(result["Close"].iloc[-2])
day_change = current_close / previous_close - 1.0

st.subheader(f"{company_name} ({code})")
first_row = st.columns(4)
first_row[0].metric("현재 종가", won(current_close), f"{day_change:+.2%}")
first_row[1].metric("1주 전략 누적손익", won(metrics["one_share_profit"], signed=True))
first_row[2].metric(
    "1주 단순보유 손익", won(metrics["buy_hold_one_share_profit"], signed=True)
)
first_row[3].metric(
    "현재 전략 상태", "1주 보유" if metrics["current_position"] else "현금 대기"
)

second_row = st.columns(4)
second_row[0].metric("전략 수익률", f"{metrics['total_return']:.1%}")
second_row[1].metric("단순보유 수익률", f"{metrics['buy_hold_return']:.1%}")
second_row[2].metric("최대 낙폭(MDD)", f"{metrics['max_drawdown']:.1%}")
second_row[3].metric("매수 판단 횟수", f"{metrics['trade_count']}회")

profit_tab, market_tab, news_tab, detail_tab = st.tabs(
    ["1주 손익", "시장 분석", "최신 뉴스", "상세 데이터"]
)

with profit_tab:
    st.plotly_chart(price_chart(result, company_name), width="stretch")
    st.subheader("첫날 종가 1주 금액으로 시작한 평가액")
    one_share_values = result[["OneShareValue", "BuyHoldOneShareValue"]].rename(
        columns={"OneShareValue": "이동평균 전략", "BuyHoldOneShareValue": "1주 단순보유"}
    )
    st.line_chart(one_share_values)
    st.caption(
        f"시작 기준금액 {won(metrics['initial_capital'])} · "
        "전략은 신호가 있을 때 정확히 1주만 보유하며 남은 금액은 현금으로 봅니다."
    )

with market_tab:
    analysis_columns = st.columns(3)
    analysis_columns[0].metric("추세 배열", str(summary["trend"]))
    analysis_columns[1].metric(
        "RSI(14)", f"{summary['rsi']:.1f}", str(summary["rsi_state"])
    )
    analysis_columns[2].metric(
        "20일 시장 대비", f"{summary['relative_20d']:+.1%}", str(summary["relative_state"])
    )
    st.write(
        f"최근 20거래일 종목 수익률은 **{summary['stock_20d']:+.1%}**, "
        f"{market} 지수는 **{summary['market_20d']:+.1%}**입니다."
    )
    st.write(
        f"최근 252거래일 범위는 **{won(summary['yearly_low'])} ~ "
        f"{won(summary['yearly_high'])}**이며, 현재가는 이 범위의 "
        f"**{summary['high_low_position']:.0%} 지점**입니다."
    )
    st.warning(
        "이 분석은 가격·이동평균·RSI·시장 상대수익률을 설명할 뿐, "
        "기업가치나 미래 주가를 판정하지 않습니다."
    )

with news_tab:
    st.caption(
        "Google 뉴스 RSS의 최신 제목입니다. 제목만으로 호재·악재를 단정하지 말고 "
        "기사 원문과 공시를 함께 확인하세요."
    )
    try:
        news_items = load_news(company_name)
    except (requests.RequestException, ElementTree.ParseError):
        news_items = []
    if news_items:
        for news_item in news_items:
            st.link_button(news_item["title"], news_item["link"], width="stretch")
            st.caption(f"{news_item['source']} · {news_item['published']} KST")
    else:
        st.info("현재 뉴스 데이터를 불러오지 못했습니다. 잠시 후 다시 실행해 주세요.")

with detail_tab:
    detail_columns = st.columns(3)
    detail_columns[0].metric("연환산 수익률", f"{metrics['cagr']:.1%}")
    detail_columns[1].metric("연환산 변동성", f"{metrics['annual_volatility']:.1%}")
    detail_columns[2].metric(
        "보유 중 상승일 비율", f"{metrics['positive_day_ratio']:.1%}"
    )
    table = result[
        [
            "Close",
            "ShortMA",
            "LongMA",
            "Position",
            "OneShareProfit",
            "OneShareValue",
        ]
    ].rename(
        columns={
            "Close": "종가",
            "ShortMA": "단기 이동평균",
            "LongMA": "장기 이동평균",
            "Position": "보유 여부",
            "OneShareProfit": "1주 전략 누적손익",
            "OneShareValue": "1주 전략 평가액",
        }
    )
    st.dataframe(table.tail(50).sort_index(ascending=False), width="stretch")

st.divider()
st.caption(
    "가격은 네이버 금융 일별 시세, 뉴스는 Google 뉴스 RSS를 사용하며 "
    "지연·누락될 수 있습니다. 백테스트 결과는 미래 수익을 보장하지 않습니다."
)
