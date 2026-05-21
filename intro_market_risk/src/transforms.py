import numpy as np
import pandas as pd


def compute_daily_returns(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    returns = df.copy()
    value_cols = [col for col in returns.columns if col != "Fecha"]
    returns[value_cols] = returns[value_cols].pct_change(periods=periods)
    return returns


def compute_base100(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.copy()
    value_cols = [col for col in indexed.columns if col != "Fecha"]
    for col in value_cols:
        first_valid = indexed[col].dropna()
        if first_valid.empty:
            indexed[col] = pd.NA
        else:
            indexed[col] = indexed[col] / first_valid.iloc[0] * 100
    return indexed


def _format_price(value: float) -> str:
    return f"${value:,.0f}".replace(",", ".")


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def compute_series_snapshot(
    prices: pd.DataFrame, returns: pd.DataFrame, selected_series: str, periods: int = 1
) -> dict:
    price_series = prices[["Fecha", selected_series]].dropna()
    return_series = returns[["Fecha", selected_series]].dropna()

    last_price = price_series[selected_series].iloc[-1]
    min_price = price_series[selected_series].min()
    max_price = price_series[selected_series].max()
    last_return = return_series[selected_series].iloc[-1]
    avg_abs_return = return_series[selected_series].abs().mean()

    period_label = "diario" if periods == 1 else f"cada {periods} días"

    return {
        "last_price": _format_price(last_price),
        "last_return": _format_percent(last_return),
        "avg_abs_return": f"{_format_percent(avg_abs_return)} {period_label}",
        "price_range": f"{_format_price(min_price)} - {_format_price(max_price)}",
    }


def describe_return_behavior(snapshot: dict, selected_series: str, return_window_label: str) -> str:
    return (
        f"En {selected_series.lower()}, el último precio por sí solo no alcanza para hablar de riesgo. "
        f"Lo que empieza a importar es la magnitud de sus cambios {return_window_label.lower()}: "
        f"en promedio, la variación absoluta observada es {snapshot['avg_abs_return']}."
    )


def compute_rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    rolling = returns.copy()
    value_cols = [col for col in rolling.columns if col != "Fecha"]
    rolling[value_cols] = rolling[value_cols].rolling(window=window).std()
    return rolling


def compute_volatility_summary(returns: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in returns.columns if col != "Fecha"]
    rows = []
    for col in value_cols:
        series = returns[col].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "Serie": col,
                "Volatilidad": series.std(),
                "Cambio absoluto promedio": series.abs().mean(),
                "Peor caída": series.min(),
                "Mejor alza": series.max(),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("Volatilidad", ascending=False).reset_index(drop=True)


def compute_distribution_summary(returns: pd.DataFrame, selected_series: str) -> dict:
    series = returns[selected_series].dropna()
    if series.empty:
        return {
            "volatility": "0.00%",
            "avg_abs_return": "0.00%",
            "worst_day": "0.00%",
            "best_day": "0.00%",
        }
    return {
        "volatility": _format_percent(series.std()),
        "avg_abs_return": _format_percent(series.abs().mean()),
        "worst_day": _format_percent(series.min()),
        "best_day": _format_percent(series.max()),
    }


def describe_volatility_behavior(
    selected_series: str, distribution_summary: dict, ranking_position: int, total_series: int
) -> str:
    return (
        f"La volatilidad resume qué tan dispersos han sido los cambios de {selected_series.lower()}. "
        f"En la muestra disponible, ocupa la posición {ranking_position} de {total_series} en nivel de volatilidad, "
        f"con una desviación estándar de {distribution_summary['volatility']} y una caída observada mínima de "
        f"{distribution_summary['worst_day']}."
    )


def compute_historical_var(returns: pd.DataFrame, selected_series: str, confidence: float, horizon: int = 1) -> dict:
    indexed_returns = returns.set_index("Fecha")
    series = indexed_returns[selected_series].dropna()
    if series.empty:
        return {
            "var_return": 0.0,
            "var_return_pct": "0.00%",
            "worst_return_pct": "0.00%",
            "scenario_count": 0,
            "scaled_returns": pd.Series(dtype=float),
        }

    if horizon <= 1:
        scenario_returns = series
    else:
        scenario_returns = (1 + series).rolling(horizon).apply(np.prod, raw=True) - 1
        scenario_returns = scenario_returns.dropna()

    quantile = scenario_returns.quantile(1 - confidence)
    var_return = abs(min(quantile, 0))
    worst_return = scenario_returns.min()

    return {
        "var_return": var_return,
        "var_return_pct": _format_percent(var_return),
        "worst_return_pct": _format_percent(abs(min(worst_return, 0))),
        "scenario_count": int(scenario_returns.shape[0]),
        "scaled_returns": scenario_returns,
    }


def compute_inventory_var_amount(inventory_value: float, var_return: float) -> str:
    return _format_price(inventory_value * var_return)


def compute_expected_shortfall(returns: pd.Series, confidence: float) -> float:
    if returns.empty:
        return 0.0
    threshold = returns.quantile(1 - confidence)
    tail = returns[returns <= threshold]
    if tail.empty:
        return 0.0
    return abs(tail.mean())


def describe_var_behavior(
    selected_series: str,
    inventory_value: float,
    confidence: float,
    horizon: int,
    var_amount: str,
    var_return_pct: str,
) -> str:
    confidence_pct = f"{confidence * 100:.0f}%"
    horizon_label = "día" if horizon == 1 else f"{horizon} días"
    return (
        f"Si una persona mantiene un inventario de {selected_series.lower()} por un valor de "
        f"{_format_price(inventory_value)}, el VaR histórico sugiere que, con un nivel de confianza del "
        f"{confidence_pct} y un horizonte de {horizon_label}, la pérdida no superaría {var_amount}. "
        f"Eso equivale a una variación adversa aproximada de {var_return_pct}."
    )
