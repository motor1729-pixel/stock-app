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
    """Run a long-only moving-average backtest without look-ahead bias."""
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

    # A signal calculated at today's close becomes a position on the next day.
    result["Position"] = result["Signal"].shift(1).fillna(0).astype(int)
    result["MarketReturn"] = result["Close"].pct_change().fillna(0.0)
    result["Turnover"] = result["Position"].diff().abs().fillna(
        result["Position"].abs()
    )
    cost_rate = config.trading_cost_pct / 100.0
    gross_return = result["MarketReturn"] * result["Position"]
    result["StrategyReturn"] = (1.0 + gross_return) * (
        1.0 - result["Turnover"] * cost_rate
    ) - 1.0
    result["StrategyEquity"] = (1.0 + result["StrategyReturn"]).cumprod()
    result["BuyHoldEquity"] = (1.0 + result["MarketReturn"]).cumprod()
    result["Entry"] = result["Signal"].diff().eq(1)
    result["Exit"] = result["Signal"].diff().eq(-1)
    return result


def performance_metrics(result: pd.DataFrame) -> dict[str, float | int]:
    if result.empty:
        raise ValueError("성과를 계산할 데이터가 없습니다.")

    equity = result["StrategyEquity"]
    running_high = equity.cummax()
    drawdown = equity / running_high - 1.0
    elapsed_days = max((result.index[-1] - result.index[0]).days, 1)
    years = elapsed_days / 365.25
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    volatility = result["StrategyReturn"].std(ddof=0) * np.sqrt(252)
    invested = result.loc[result["Position"].eq(1), "StrategyReturn"]

    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "buy_hold_return": float(result["BuyHoldEquity"].iloc[-1] - 1.0),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": float(volatility),
        "trade_count": int(result["Entry"].sum()),
        "positive_day_ratio": float((invested > 0).mean()) if not invested.empty else 0.0,
    }
