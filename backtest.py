from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    short_window: int = 20
    long_window: int = 60
    trading_cost_pct: float = 0.02


def run_backtest(prices: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Backtest a one-share, long-only moving-average strategy."""
    if config.short_window < 2:
        raise ValueError("단기 이동평균은 2일 이상이어야 합니다.")
    if config.short_window >= config.long_window:
        raise ValueError("장기 이동평균은 단기 이동평균보다 길어야 합니다.")
    if config.trading_cost_pct < 0:
        raise ValueError("거래비용은 0 이상이어야 합니다.")
    if "Close" not in prices.columns:
        raise ValueError("가격 데이터에 Close 열이 없습니다.")

    result = prices.copy().sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    result = result.dropna(subset=["Close"])
    if len(result) < config.long_window + 2:
        raise ValueError(
            f"최소 {config.long_window + 2}거래일의 가격 데이터가 필요합니다."
        )

    result["ShortMA"] = result["Close"].rolling(config.short_window).mean()
    result["LongMA"] = result["Close"].rolling(config.long_window).mean()
    result["Signal"] = (result["ShortMA"] > result["LongMA"]).astype(int)

    # A signal calculated at today's close applies to the next close-to-close interval.
    result["Position"] = result["Signal"].shift(1).fillna(0).astype(int)
    result["Turnover"] = result["Position"].diff().abs().fillna(
        result["Position"].abs()
    )
    result["MarketReturn"] = result["Close"].pct_change().fillna(0.0)

    cost_rate = config.trading_cost_pct / 100.0
    execution_price = result["Close"].shift(1).fillna(result["Close"])
    result["OneShareCost"] = execution_price * result["Turnover"] * cost_rate
    result["OneShareDailyPnl"] = (
        result["Close"].diff().fillna(0.0) * result["Position"]
        - result["OneShareCost"]
    )
    result["OneShareProfit"] = result["OneShareDailyPnl"].cumsum()

    initial_price = float(result["Close"].iloc[0])
    result["OneShareValue"] = initial_price + result["OneShareProfit"]
    result["BuyHoldOneShareProfit"] = result["Close"] - initial_price
    result["BuyHoldOneShareValue"] = result["Close"]

    result["StrategyReturn"] = (
        result["OneShareValue"].pct_change().fillna(0.0)
    )
    result["StrategyEquity"] = result["OneShareValue"] / initial_price
    result["BuyHoldEquity"] = result["BuyHoldOneShareValue"] / initial_price
    result["Entry"] = result["Signal"].diff().eq(1)
    result["Exit"] = result["Signal"].diff().eq(-1)
    return result


def performance_metrics(result: pd.DataFrame) -> dict[str, float | int]:
    if result.empty:
        raise ValueError("성과를 계산할 데이터가 없습니다.")

    value = result["OneShareValue"]
    running_high = value.cummax()
    drawdown = value / running_high - 1.0
    elapsed_days = max((result.index[-1] - result.index[0]).days, 1)
    years = elapsed_days / 365.25
    total_return = value.iloc[-1] / value.iloc[0] - 1.0
    cagr = (
        value.iloc[-1] / value.iloc[0]
    ) ** (1.0 / years) - 1.0 if value.iloc[-1] > 0 else -1.0
    volatility = result["StrategyReturn"].std(ddof=0) * np.sqrt(252)
    invested = result.loc[result["Position"].eq(1), "OneShareDailyPnl"]

    return {
        "initial_capital": float(value.iloc[0]),
        "one_share_profit": float(result["OneShareProfit"].iloc[-1]),
        "buy_hold_one_share_profit": float(result["BuyHoldOneShareProfit"].iloc[-1]),
        "current_value": float(value.iloc[-1]),
        "total_return": float(total_return),
        "buy_hold_return": float(result["BuyHoldEquity"].iloc[-1] - 1.0),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": float(volatility),
        "trade_count": int(result["Entry"].sum()),
        "positive_day_ratio": float((invested > 0).mean()) if not invested.empty else 0.0,
        "current_position": int(result["Position"].iloc[-1]),
    }
