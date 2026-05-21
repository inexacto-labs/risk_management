from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st


PRESET_GROUPS = {
    "Tecnología": ["AAPL", "MSFT", "NVDA", "GOOGL"],
    "Consumo masivo": ["KO", "PG", "PEP", "WMT"],
    "Índices de referencia": ["SPY", "QQQ", "XLP", "XLK"],
}


@st.cache_data(show_spinner=False)
def load_market_data(tickers: tuple[str, ...], period: str = "3y") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Falta instalar la dependencia `yfinance`.") from exc

    if not tickers:
        raise RuntimeError("No se recibieron tickers para consultar.")

    data = yf.download(
        list(tickers),
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError("No fue posible descargar información para los tickers solicitados.")

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]}).copy()

    closes = closes.reset_index().rename(columns={"Date": "Fecha"})
    closes["Fecha"] = pd.to_datetime(closes["Fecha"])
    closes = closes.sort_values("Fecha").reset_index(drop=True)
    closes.columns = ["Fecha", *[str(col).upper() for col in closes.columns[1:]]]
    return closes


def normalize_tickers(values: Iterable[str]) -> list[str]:
    cleaned = []
    for value in values:
        ticker = value.strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned
