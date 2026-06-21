from __future__ import annotations

from datetime import timedelta, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from backtest import BacktestConfig, performance_metrics, run_backtest


st.set_page_config(page_title="쉬운 국내주식 점검", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1280px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e4e9f2;
        border-radius: 12px;
        padding: 14px;
    }
    [data-testid="stMetricLabel"] {font-size: 0.9rem;}
    [data-testid="stMetricValue"] {font-size: 1.45rem;}
    @media (max-width: 768px) {
        .block-container {padding: 0.8rem 0.7rem 2rem;}
        h1 {font-size: 1.65rem !important;}
        h2 {font-size: 1.3rem !important;}
        [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: 0.5rem;}
        [data-testid="column"] {
            min-width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
        }
        [data-testid="stMetric"] {padding: 10px;}
        [data-testid="stMetricValue"] {font-size: 1.15rem;}
        .stTabs [data-baseweb="tab-list"] {overflow-x: auto; white-space: nowrap;}
        .stTabs [data-baseweb="tab"] {min-width: max-content;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PERIOD_DAYS = {"1y": 260, "3y": 780, "5y": 1300}
STYLE_CONFIG = {
    "빠른 판단": {
        "windows": (5, 20),
        "help": "변화를 빨리 잡지만 매매 신호가 자주 바뀔 수 있습니다.",
    },
    "균형 판단": {
        "windows": (20, 60),
        "help": "너무 빠르지도 느리지도 않은 기본 설정입니다.",
    },
    "큰 흐름 판단": {
        "windows": (60, 120),
        "help": "큰 추세만 확인해 신호가 느리지만 흔들림이 적습니다.",
    },
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
def load_stock_basic(code: str) -> dict[str, str]:
    response = requests.get(
        f"https://m.stock.naver.com/api/stock/{code}/basic",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    exchange = payload.get("stockExchangeType") or {}
    market = exchange.get("name") or exchange.get("nameEng") or "KOSPI"
    return {
        "name": payload.get("stockName") or code,
        "market": "KOSDAQ" if market == "KOSDAQ" else "KOSPI",
    }


@st.cache_data(ttl=1800, show_spinner=False)
def load_news(company_name: str, limit: int = 8) -> list[dict[str, str]]:
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
        source_element = item.find("source")
        published = item.findtext("pubDate") or ""
        try:
            published_text = (
                parsedate_to_datetime(published).astimezone(KST).strftime("%m-%d %H:%M")
            )
        except (TypeError, ValueError):
            published_text = published
        news.append(
            {
                "title": (item.findtext("title") or "제목 없음").strip(),
                "link": (item.findtext("link") or "").strip(),
                "source": (
                    source_element.text.strip()
                    if source_element is not None and source_element.text
                    else "출처 미표시"
                ),
                "published": published_text,
            }
        )
    return news


def normalize_code(code: str) -> str:
    normalized = "".join(character for character in code if character.isdigit())
    if len(normalized) != 6:
        raise ValueError("종목코드는 숫자 6자리로 입력해 주세요. 예: 005930")
    return normalized


def won(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:,.0f}원"


def last_signal_date(result: pd.DataFrame, column: str) -> str:
    dates = result.index[result[column]]
    return dates[-1].strftime("%Y-%m-%d") if len(dates) else "없음"


def decision_summary(result: pd.DataFrame) -> dict[str, str]:
    signal_now = int(result["Signal"].iloc[-1])
    signal_before = int(result["Signal"].iloc[-2])
    if signal_now == 1 and signal_before == 0:
        return {
            "title": "신규 매수 검토",
            "message": "오늘 종가에 매수 조건이 처음 발생했습니다. 다음 거래일에 분할매수를 검토하는 구간입니다.",
            "kind": "success",
        }
    if signal_now == 0 and signal_before == 1:
        return {
            "title": "매도 검토",
            "message": "오늘 종가에 매도 조건이 발생했습니다. 보유 중이라면 다음 거래일에 정리를 검토하는 구간입니다.",
            "kind": "error",
        }
    if signal_now == 1:
        return {
            "title": "보유 구간",
            "message": "매수 조건은 유지 중입니다. 이미 보유했다면 유지 구간이며, 새로 살 경우 급등 추격 여부를 별도로 확인해야 합니다.",
            "kind": "info",
        }
    return {
        "title": "신규 매수 대기",
        "message": "현재는 매수 조건이 아닙니다. 다음 매수 조건이 생길 때까지 현금으로 기다리는 구간입니다.",
        "kind": "warning",
    }


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
    short_line = float(result["ShortMA"].iloc[-1])
    long_line = float(result["LongMA"].iloc[-1])
    if current > short_line > long_line:
        trend = "상승 흐름"
    elif current < short_line < long_line:
        trend = "하락 흐름"
    else:
        trend = "방향 불분명"

    rsi = calculate_rsi(close)
    rsi_state = "과열 주의" if rsi >= 70 else "침체 구간" if rsi <= 30 else "보통"
    recent = close.tail(252)
    yearly_low = float(recent.min())
    yearly_high = float(recent.max())
    stock_20d = float(close.pct_change(20).iloc[-1])
    market_20d = (
        float(market_prices["Close"].pct_change(20).iloc[-1])
        if market_prices is not None and len(market_prices) > 20
        else 0.0
    )
    return {
        "trend": trend,
        "rsi": rsi,
        "rsi_state": rsi_state,
        "yearly_low": yearly_low,
        "yearly_high": yearly_high,
        "stock_20d": stock_20d,
        "market_20d": market_20d,
        "relative_20d": stock_20d - market_20d,
    }


def daily_price_chart(result: pd.DataFrame, name: str, days: int) -> go.Figure:
    view = result.tail(days)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=view.index,
            y=view["Close"],
            name="종가",
            line=dict(width=3, color="#1769e0"),
            hovertemplate="%{x|%Y-%m-%d}<br>종가 %{y:,.0f}원<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=view.index,
            y=view["ShortMA"],
            name="빠른 흐름선",
            line=dict(width=1.5, color="#42a5f5"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=view.index,
            y=view["LongMA"],
            name="느린 흐름선",
            line=dict(width=1.5, color="#ef5350"),
        )
    )
    entries = view[view["Entry"]]
    exits = view[view["Exit"]]
    figure.add_trace(
        go.Scatter(
            x=entries.index,
            y=entries["Close"],
            mode="markers",
            name="매수 조건",
            marker=dict(symbol="triangle-up", size=13, color="#00a878"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=exits.index,
            y=exits["Close"],
            mode="markers",
            name="매도 조건",
            marker=dict(symbol="triangle-down", size=13, color="#e45756"),
        )
    )
    figure.update_layout(
        title=f"{name} 최근 {days}거래일",
        yaxis_title="가격(원)",
        hovermode="x unified",
        legend_orientation="h",
        height=470,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    figure.update_xaxes(
        range=[view.index.min(), view.index.max()],
        rangebreaks=[dict(bounds=["sat", "mon"])],
        tickformat="%m-%d",
    )
    return figure


def value_chart(result: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=result.index,
            y=result["BuyHoldOneShareValue"],
            name="1주 계속 보유",
            line=dict(width=2, color="#1769e0"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=result.index,
            y=result["OneShareValue"],
            name="규칙대로 매매",
            line=dict(width=2, color="#42a5f5"),
        )
    )
    figure.update_layout(
        title="과거에 1주 가격으로 시작했다면",
        yaxis_title="평가액(원)",
        hovermode="x unified",
        legend_orientation="h",
        height=400,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    figure.update_xaxes(
        range=[result.index.min(), result.index.max()], tickformat="%Y-%m"
    )
    return figure


st.title("쉬운 국내주식 점검")
st.caption(
    "복잡한 지표 대신 오늘의 규칙상 행동, 투자 가능 수량, 위험 금액을 먼저 보여줍니다. "
    "수익을 보장하거나 개인별 매매를 지시하는 서비스는 아닙니다."
)

with st.sidebar:
    st.header("내 투자 계획")
    code_input = st.text_input("종목코드", value="005930", help="삼성전자: 005930")
    investment_amount = st.number_input(
        "투자할 금액(원)",
        min_value=10_000,
        max_value=1_000_000_000,
        value=1_000_000,
        step=100_000,
    )
    style = st.selectbox("판단 속도", list(STYLE_CONFIG), index=1)
    st.caption(STYLE_CONFIG[style]["help"])
    risk_pct = st.slider("감당할 최대 손실률", 1, 30, 7, format="%d%%")
    chart_days = st.selectbox(
        "그래프에 표시할 기간", [30, 60, 90, 180], index=2, format_func=lambda x: f"최근 {x}거래일"
    )
    with st.expander("고급 설정"):
        history_period = st.selectbox(
            "과거 검증 범위",
            ["1y", "3y", "5y"],
            index=1,
            format_func={"1y": "1년", "3y": "3년", "5y": "5년"}.get,
        )
        trading_cost = st.number_input(
            "편도 거래비용(%)", 0.0, 2.0, 0.02, 0.01
        )
    run = st.button("오늘 판단 보기", type="primary", width="stretch")

if not run:
    st.info("종목코드와 투자금액을 입력하고 ‘오늘 판단 보기’를 눌러 주세요.")
    st.markdown(
        "**예시:** 삼성전자 `005930` · SK하이닉스 `000660` · "
        "현대차 `005380` · 제주반도체 `080220`"
    )
    st.stop()

try:
    code = normalize_code(code_input)
    short_window, long_window = STYLE_CONFIG[style]["windows"]
    with st.spinner("종목과 시장 데이터를 확인하고 있습니다..."):
        basic = load_stock_basic(code)
        prices = load_prices(code, history_period)
        company_name = basic["name"] or prices.attrs.get("name") or code
        market = basic["market"]
        try:
            market_prices = load_prices(market, "1y")
        except (requests.RequestException, ValueError, ElementTree.ParseError):
            market_prices = None
    result = run_backtest(
        prices,
        BacktestConfig(short_window, long_window, float(trading_cost)),
    )
    metrics = performance_metrics(result)
    summary = technical_summary(result, market_prices)
except (requests.RequestException, ValueError, ElementTree.ParseError) as error:
    st.error(f"데이터를 처리하지 못했습니다: {error}")
    st.stop()

current_close = float(result["Close"].iloc[-1])
previous_close = float(result["Close"].iloc[-2])
day_change = current_close / previous_close - 1.0
decision = decision_summary(result)
cost_rate = float(trading_cost) / 100.0
possible_shares = int(investment_amount // (current_close * (1.0 + cost_rate)))
planned_cost = possible_shares * current_close * (1.0 + cost_rate)
cash_left = investment_amount - planned_cost
loss_price = current_close * (1.0 - risk_pct / 100.0)
planned_max_loss = possible_shares * current_close * risk_pct / 100.0
as_of = result.index[-1].strftime("%Y-%m-%d")

st.subheader(f"{company_name} ({code}) · {market}")
st.caption(f"판단 기준일: {as_of} 종가 · 장중 실시간 신호가 아닙니다.")

message_function = getattr(st, decision["kind"])
message_function(f"**규칙상 오늘 판단: {decision['title']}**\n\n{decision['message']}")

top_metrics = st.columns(4)
top_metrics[0].metric("현재 종가", won(current_close), f"{day_change:+.2%}")
top_metrics[1].metric("살 수 있는 수량", f"{possible_shares:,}주")
top_metrics[2].metric("예상 사용금액", won(planned_cost))
top_metrics[3].metric("남는 현금", won(cash_left))

risk_metrics = st.columns(4)
risk_metrics[0].metric("손실 제한 참고가격", won(loss_price), f"-{risk_pct}%")
risk_metrics[1].metric("계획상 최대 손실", won(planned_max_loss))
risk_metrics[2].metric("마지막 매수 조건일", last_signal_date(result, "Entry"))
risk_metrics[3].metric("마지막 매도 조건일", last_signal_date(result, "Exit"))

if possible_shares == 0:
    st.warning("입력한 투자금으로는 현재 가격 기준 1주를 살 수 없습니다.")

today_tab, calendar_tab, market_tab, news_tab, history_tab = st.tabs(
    ["오늘 판단", "일자별 보기", "시장 비교", "최신 뉴스", "과거 결과"]
)

with today_tab:
    st.plotly_chart(daily_price_chart(result, company_name, chart_days), width="stretch")
    st.markdown(
        f"""
        **이 화면에서 할 일**

        - 현재 판단: **{decision['title']}**
        - 마지막 매수 조건: **{last_signal_date(result, 'Entry')}**
        - 마지막 매도 조건: **{last_signal_date(result, 'Exit')}**
        - 입력 금액으로 가능한 수량: **{possible_shares:,}주**

        초록 삼각형은 매수 조건이 생긴 날, 빨간 삼각형은 매도 조건이 생긴 날입니다.
        조건은 종가 확정 후 계산되므로 실제 검토 시점은 다음 거래일입니다.
        """
    )

with calendar_tab:
    daily = result.tail(chart_days).copy()
    daily["전일 대비"] = daily["Close"].pct_change() * 100.0
    signal_change = daily["Signal"].diff()
    daily["하루 판단"] = "대기"
    daily.loc[daily["Signal"].eq(1), "하루 판단"] = "보유 구간"
    daily.loc[signal_change.eq(1), "하루 판단"] = "매수 조건"
    daily.loc[signal_change.eq(-1), "하루 판단"] = "매도 조건"
    daily_table = daily[["Close", "전일 대비", "하루 판단"]].rename(
        columns={"Close": "종가"}
    )
    st.dataframe(
        daily_table.sort_index(ascending=False),
        width="stretch",
        column_config={
            "종가": st.column_config.NumberColumn(format="%d원"),
            "전일 대비": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.caption("가장 최근 날짜부터 표시됩니다. 전일 대비는 휴장일을 제외한 거래일 기준입니다.")

with market_tab:
    comparison = st.columns(4)
    comparison[0].metric("현재 가격 흐름", str(summary["trend"]))
    comparison[1].metric("과열 정도", str(summary["rsi_state"]), f"점수 {summary['rsi']:.0f}")
    comparison[2].metric("종목 20일 변화", f"{summary['stock_20d']:+.1%}")
    comparison[3].metric(f"{market} 20일 변화", f"{summary['market_20d']:+.1%}")
    st.write(
        f"최근 1년 가격 범위는 **{won(summary['yearly_low'])} ~ "
        f"{won(summary['yearly_high'])}**입니다. 시장 대비 20일 차이는 "
        f"**{summary['relative_20d']:+.1%}**입니다."
    )
    st.warning("상승 흐름이나 과열 표시는 미래 상승을 보장하지 않습니다.")

with news_tab:
    st.caption("제목만으로 호재·악재를 단정하지 말고 원문과 기업 공시를 함께 확인하세요.")
    try:
        news_items = load_news(company_name)
    except (requests.RequestException, ElementTree.ParseError):
        news_items = []
    if news_items:
        for news_item in news_items:
            st.link_button(news_item["title"], news_item["link"], width="stretch")
            st.caption(f"{news_item['source']} · {news_item['published']} KST")
    else:
        st.info("현재 뉴스를 불러오지 못했습니다. 잠시 후 다시 실행해 주세요.")

with history_tab:
    history_metrics = st.columns(4)
    history_metrics[0].metric("규칙 매매 1주 손익", won(metrics["one_share_profit"], True))
    history_metrics[1].metric(
        "계속 보유 1주 손익", won(metrics["buy_hold_one_share_profit"], True)
    )
    history_metrics[2].metric("규칙 매매 최대 낙폭", f"{metrics['max_drawdown']:.1%}")
    history_metrics[3].metric("매수 조건 횟수", f"{metrics['trade_count']}회")
    st.plotly_chart(value_chart(result), width="stretch")
    st.caption(
        f"과거 시작 종가 {won(metrics['initial_capital'])}를 기준으로 정확히 1주만 "
        "보유하거나 현금으로 기다렸을 때를 비교합니다."
    )

st.divider()
st.caption(
    "가격은 네이버 금융 일별 시세, 뉴스는 Google 뉴스 RSS를 사용하며 지연·누락될 수 있습니다. "
    "규칙상 판단은 교육용 참고 정보이며 투자 손실에 대한 책임을 대신하지 않습니다."
)
