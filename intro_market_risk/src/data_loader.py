from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataPapa.csv"


@st.cache_data(show_spinner=False)
def load_papa_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.rename(columns=lambda col: col.strip())
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    numeric_cols = [col for col in df.columns if col != "Fecha"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Fecha").reset_index(drop=True)
    return df


def get_series_names(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "Fecha"]


def get_dataset_overview(df: pd.DataFrame) -> dict:
    series = get_series_names(df)
    return {
        "rows": len(df),
        "series_count": len(series),
        "series": series,
        "start": df["Fecha"].min().date().isoformat(),
        "end": df["Fecha"].max().date().isoformat(),
    }
