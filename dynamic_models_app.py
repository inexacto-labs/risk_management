import os
from pathlib import Path
from urllib.error import URLError

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Series Dinamicas y Riesgo",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1D3557;
            border-bottom: 1px solid #D9E2EC;
            padding-bottom: 4px;
            margin: 22px 0 10px;
        }
        .info-banner {
            background: #F1F7FF;
            border-left: 4px solid #457B9D;
            padding: 10px 14px;
            border-radius: 4px;
            margin: 8px 0 14px;
            color: #17324D;
        }
        .metric-card {
            background: #F8FAFC;
            border: 1px solid #D9E2EC;
            border-radius: 8px;
            padding: 10px 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


SERIES_CONFIG = {
    "gasoline": {
        "fred_id": "GASREGW",
        "label": "Gasolina regular",
        "color": "#D62828",
    },
    "oil": {
        "fred_id": "WCOILWTICO",
        "label": "Petroleo WTI",
        "color": "#1D3557",
    },
}

FX_TICKER = "EURUSD=X"
FX_PAIR_LABEL = "EUR/USD"
FX_PRICE_LABEL = "USD por EUR"


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def info_banner(text: str) -> None:
    st.markdown(f'<div class="info-banner">{text}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_statsmodels_modules():
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    return {
        "sm": sm,
        "acorr_ljungbox": acorr_ljungbox,
        "het_arch": het_arch,
        "ARIMA": ARIMA,
        "acf": acf,
        "adfuller": adfuller,
        "kpss": kpss,
        "pacf": pacf,
    }


@st.cache_resource(show_spinner=False)
def get_arch_model():
    from arch import arch_model

    return arch_model


@st.cache_resource(show_spinner=False)
def get_yfinance_module():
    import yfinance as yf

    return yf


@st.cache_data(show_spinner=False)
def load_fred_series(series_id: str, start_date: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["fecha", "valor"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna()
    df = df[df["fecha"] >= pd.to_datetime(start_date)].copy()
    return df


@st.cache_data(show_spinner=False)
def load_gasoline_oil_data(start_date: str) -> pd.DataFrame:
    gas = load_fred_series(SERIES_CONFIG["gasoline"]["fred_id"], start_date)
    oil = load_fred_series(SERIES_CONFIG["oil"]["fred_id"], start_date)

    gas = gas.set_index("fecha").resample("W-FRI").last().rename(columns={"valor": "gasoline"})
    oil = oil.set_index("fecha").resample("W-FRI").last().rename(columns={"valor": "oil"})

    df = gas.join(oil, how="inner").dropna().reset_index()
    df["log_gasoline"] = np.log(df["gasoline"])
    df["log_oil"] = np.log(df["oil"])
    df["dlog_gasoline"] = df["log_gasoline"].diff() * 100
    df["dlog_oil"] = df["log_oil"].diff() * 100
    df["year"] = df["fecha"].dt.year
    df["week"] = df["fecha"].dt.isocalendar().week.astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_fx_data(start_date: str) -> pd.DataFrame:
    fx_start_date = max(pd.Timestamp(start_date), pd.Timestamp("2017-01-01"))
    fx = load_fred_series("DEXUSEU", fx_start_date.strftime("%Y-%m-%d")).rename(columns={"valor": "fx"})
    fx = fx.dropna()
    fx["log_fx"] = np.log(fx["fx"])
    fx["fx_return"] = fx["log_fx"].diff() * 100
    fx["abs_fx_return"] = fx["fx_return"].abs()
    fx["fx_return_sq"] = fx["fx_return"] ** 2
    fx["year"] = fx["fecha"].dt.year
    return fx.dropna().reset_index(drop=True)


def make_level_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["gasoline"],
            mode="lines",
            name="Gasolina",
            line=dict(color=SERIES_CONFIG["gasoline"]["color"], width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["oil"],
            mode="lines",
            name="Petroleo",
            line=dict(color=SERIES_CONFIG["oil"]["color"], width=2.4),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="USD por galon", row=1, col=1)
    fig.update_yaxes(title_text="USD por barril", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Series en niveles", font=dict(size=20)),
        hovermode="x unified",
    )
    return fig


def make_log_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["log_gasoline"],
            mode="lines",
            name="ln(gasolina)",
            line=dict(color=SERIES_CONFIG["gasoline"]["color"], width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["log_oil"],
            mode="lines",
            name="ln(petroleo)",
            line=dict(color=SERIES_CONFIG["oil"]["color"], width=2.4),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="ln(precio)", row=1, col=1)
    fig.update_yaxes(title_text="ln(precio)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Series en logaritmos", font=dict(size=20)),
        hovermode="x unified",
    )
    return fig


def make_growth_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["dlog_gasoline"],
            mode="lines",
            name="Δln(gasolina)",
            line=dict(color=SERIES_CONFIG["gasoline"]["color"], width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["dlog_oil"],
            mode="lines",
            name="Δln(petroleo)",
            line=dict(color=SERIES_CONFIG["oil"]["color"], width=2.2),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Series listas para modelar", font=dict(size=20)),
        hovermode="x unified",
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Variacion logaritmica semanal (%)"),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    return fig


def make_fx_level_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["fx"],
            mode="lines",
            name=FX_PRICE_LABEL,
            line=dict(color="#1D3557", width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["log_fx"],
            mode="lines",
            name=f"ln({FX_PAIR_LABEL})",
            line=dict(color="#2A9D8F", width=2.4),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=FX_PRICE_LABEL, row=1, col=1)
    fig.update_yaxes(title_text=f"ln({FX_PAIR_LABEL})", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=620,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"{FX_PAIR_LABEL} en niveles y logaritmos", font=dict(size=20)),
        hovermode="x unified",
    )
    return fig


def make_fx_returns_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["fx_return"],
            mode="lines",
            name="Retorno logaritmico",
            line=dict(color="#D62828", width=1.6),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Retornos logaritmicos de {FX_PAIR_LABEL}", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=f"Δln({FX_PAIR_LABEL}) (%)"),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    return fig


def make_return_abs_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["fx_return"],
            mode="lines",
            name="Retorno",
            line=dict(color="#1D3557", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["abs_fx_return"],
            mode="lines",
            name="Retorno absoluto",
            line=dict(color="#E76F51", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Retorno (%)", row=1, col=1)
    fig.update_yaxes(title_text="|Retorno| (%)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Retornos y magnitud de los movimientos", font=dict(size=20)),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1, row=1, col=1)
    return fig


def make_squared_returns_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["fx_return_sq"],
            mode="lines",
            name="Retorno al cuadrado",
            line=dict(color="#6A4C93", width=1.6),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Retornos al cuadrado", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Retorno^2"),
        hovermode="x unified",
    )
    return fig


def make_acf_pacf_figure(values: np.ndarray, max_lag: int, title_prefix: str) -> go.Figure:
    modules = get_statsmodels_modules()
    acf_values = modules["acf"](values, nlags=max_lag, fft=True)
    pacf_values = modules["pacf"](values, nlags=max_lag, method="ywm")
    conf = 1.96 / np.sqrt(len(values))
    lags = np.arange(len(acf_values))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"ACF de {title_prefix}", f"PACF de {title_prefix}"),
    )

    for col_index, stats_values in enumerate([acf_values, pacf_values], start=1):
        fig.add_trace(
            go.Bar(
                x=lags[1:],
                y=stats_values[1:],
                marker_color="#457B9D",
                showlegend=False,
                hovertemplate="Rezago %{x}<br>Valor %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=col_index,
        )
        fig.add_hline(y=conf, line_dash="dot", line_color="#D62828", row=1, col=col_index)
        fig.add_hline(y=-conf, line_dash="dot", line_color="#D62828", row=1, col=col_index)
        fig.add_hline(y=0, line_color="#7A7A7A", row=1, col=col_index)
        fig.update_xaxes(title_text="Rezago", row=1, col=col_index)
        fig.update_yaxes(title_text="Correlacion", row=1, col=col_index)

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


@st.cache_data(show_spinner=False)
def summarize_ljung_box(values: tuple[float, ...], lag: int = 12) -> dict[str, float]:
    acorr_ljungbox = get_statsmodels_modules()["acorr_ljungbox"]
    lb = acorr_ljungbox(pd.Series(values).dropna(), lags=[lag], return_df=True)
    return {
        "stat": float(lb["lb_stat"].iloc[0]),
        "pvalue": float(lb["lb_pvalue"].iloc[0]),
        "lag": lag,
    }


def residual_interpretation(pvalue: float, lag: int) -> str:
    if pvalue >= 0.05:
        return (
            f"Con un p-valor de {pvalue:.4f} en Ljung-Box al rezago {lag}, no vemos evidencia fuerte de autocorrelacion remanente. "
            "Eso sugiere que los residuos se parecen mas a ruido blanco."
        )
    return (
        f"Con un p-valor de {pvalue:.4f} en Ljung-Box al rezago {lag}, si vemos evidencia de autocorrelacion remanente. "
        "Eso significa que los residuos aun guardan memoria y que el modelo o la regresion todavia no capturan toda la dinamica."
    )


def build_model_equation_text(family: str, order_p: int, model) -> str:
    params = model.params.to_dict()
    ar_terms = [name for name in params if name.startswith("ar.L")]
    oil_terms = [name for name in params if "dlog_oil" in name]

    if family == "AR":
        return (
            f"El modelo ganador es un AR({order_p}). En palabras, dice que el cambio de esta semana en la gasolina "
            "se explica con una combinacion lineal de sus propios cambios pasados. "
            f"Aqui no entra informacion externa: toda la estructura viene de la memoria propia de la serie."
        )

    if oil_terms:
        oil_name = oil_terms[0]
        oil_beta = params[oil_name]
        oil_text = (
            f"El coeficiente asociado al petroleo es {oil_beta:.3f}. "
            "Eso se interpreta como el efecto contemporaneo del cambio del petroleo sobre el cambio de la gasolina, "
            "manteniendo fija la parte autorregresiva."
        )
    else:
        oil_text = (
            "El modelo incluye al petroleo como variable exogena, aunque en esta especificacion el nombre del parametro puede cambiar "
            "segun como statsmodels etiquete la serie."
        )

    ar_text = (
        f"Ademas, el componente AR({order_p}) permite que la gasolina conserve memoria de sus propios movimientos pasados. "
        "Por eso un ARIMAX mezcla inercia interna con transmision de informacion externa."
    )

    return f"El modelo ganador es un ARIMAX({order_p},0,0). {oil_text} {ar_text}"


@st.cache_data(show_spinner=False)
def summarize_fx_diagnostics(returns: tuple[float, ...]) -> dict[str, float]:
    from scipy import stats

    modules = get_statsmodels_modules()
    acorr_ljungbox = modules["acorr_ljungbox"]
    het_arch = modules["het_arch"]

    series = pd.Series(returns).dropna()
    centered = series - series.mean()

    t_stat, t_pvalue = stats.ttest_1samp(series, popmean=0.0, nan_policy="omit")
    lb_returns = acorr_ljungbox(series, lags=[20], return_df=True)
    lb_abs = acorr_ljungbox(series.abs(), lags=[20], return_df=True)
    lb_sq = acorr_ljungbox(centered**2, lags=[12], return_df=True)
    arch_stat, arch_pvalue, _, _ = het_arch(centered, nlags=12)

    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "t_stat": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "lb_returns_pvalue": float(lb_returns["lb_pvalue"].iloc[0]),
        "lb_abs_pvalue": float(lb_abs["lb_pvalue"].iloc[0]),
        "lb_sq_pvalue": float(lb_sq["lb_pvalue"].iloc[0]),
        "arch_stat": float(arch_stat),
        "arch_pvalue": float(arch_pvalue),
    }


@st.cache_data(show_spinner=False)
def summarize_stationarity_tests(levels: tuple[float, ...], returns: tuple[float, ...]) -> pd.DataFrame:
    modules = get_statsmodels_modules()
    adfuller = modules["adfuller"]
    kpss = modules["kpss"]
    acorr_ljungbox = modules["acorr_ljungbox"]

    levels_series = pd.Series(levels).dropna()
    returns_series = pd.Series(returns).dropna()

    adf_levels = adfuller(levels_series, regression="ct", autolag="AIC")
    kpss_levels = kpss(levels_series, regression="ct", nlags="auto")
    adf_returns = adfuller(returns_series, regression="c", autolag="AIC")
    kpss_returns = kpss(returns_series, regression="c", nlags="auto")
    lb_returns = acorr_ljungbox(returns_series, lags=[20], return_df=True)

    rows = [
        {
            "Serie": f"ln({FX_PAIR_LABEL})",
            "Prueba": "ADF",
            "Hipotesis nula": "raiz unitaria",
            "Estadistico": float(adf_levels[0]),
            "p_valor": float(adf_levels[1]),
            "Lectura": "rechaza raiz unitaria" if adf_levels[1] < 0.05 else "no rechaza raiz unitaria",
        },
        {
            "Serie": f"ln({FX_PAIR_LABEL})",
            "Prueba": "KPSS",
            "Hipotesis nula": "estacionaria",
            "Estadistico": float(kpss_levels[0]),
            "p_valor": float(kpss_levels[1]),
            "Lectura": "rechaza estacionariedad" if kpss_levels[1] < 0.05 else "no rechaza estacionariedad",
        },
        {
            "Serie": "retornos",
            "Prueba": "ADF",
            "Hipotesis nula": "raiz unitaria",
            "Estadistico": float(adf_returns[0]),
            "p_valor": float(adf_returns[1]),
            "Lectura": "rechaza raiz unitaria" if adf_returns[1] < 0.05 else "no rechaza raiz unitaria",
        },
        {
            "Serie": "retornos",
            "Prueba": "KPSS",
            "Hipotesis nula": "estacionaria",
            "Estadistico": float(kpss_returns[0]),
            "p_valor": float(kpss_returns[1]),
            "Lectura": "rechaza estacionariedad" if kpss_returns[1] < 0.05 else "no rechaza estacionariedad",
        },
        {
            "Serie": "retornos",
            "Prueba": "Ljung-Box (20)",
            "Hipotesis nula": "sin autocorrelacion lineal",
            "Estadistico": float(lb_returns["lb_stat"].iloc[0]),
            "p_valor": float(lb_returns["lb_pvalue"].iloc[0]),
            "Lectura": "rechaza ruido blanco" if lb_returns["lb_pvalue"].iloc[0] < 0.05 else "no rechaza ruido blanco",
        },
    ]

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fit_volatility_model(
    returns: tuple[float, ...],
    family: str,
    arch_order: int,
    dist: str,
):
    arch_model = get_arch_model()
    series = pd.Series(returns).dropna()
    centered = series - series.mean()

    if family == "ARCH":
        model = arch_model(centered, mean="Zero", vol="ARCH", p=arch_order, dist=dist)
    else:
        model = arch_model(centered, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)

    fitted = model.fit(disp="off")
    std_resid = pd.Series(np.asarray(fitted.std_resid)).dropna()
    cond_vol = pd.Series(np.asarray(fitted.conditional_volatility)).dropna()
    result_df = pd.DataFrame(
        {
            "residual_std": std_resid.to_numpy(),
            "conditional_vol": cond_vol.to_numpy(),
        }
    )
    lb_resid = summarize_ljung_box(tuple(result_df["residual_std"].to_numpy()), lag=20)
    lb_sq = summarize_ljung_box(tuple((result_df["residual_std"] ** 2).to_numpy()), lag=20)

    return {
        "model": fitted,
        "result_df": result_df,
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "loglik": float(fitted.loglikelihood),
        "lb_resid_pvalue": lb_resid["pvalue"],
        "lb_sq_pvalue": lb_sq["pvalue"],
    }


def make_arch_diagnostic_chart(dates: pd.Series, returns: pd.Series, diagnostics_df: pd.DataFrame) -> go.Figure:
    aligned_dates = dates.iloc[-len(diagnostics_df):].reset_index(drop=True)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Volatilidad condicional",
            "Residuos estandarizados",
            "ACF residuos estandarizados",
            "ACF residuos estandarizados^2",
        ),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=aligned_dates,
            y=diagnostics_df["conditional_vol"],
            mode="lines",
            line=dict(color="#D62828", width=1.8),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=aligned_dates,
            y=diagnostics_df["residual_std"],
            mode="lines",
            line=dict(color="#1D3557", width=1.4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    modules = get_statsmodels_modules()
    acf_vals = modules["acf"](diagnostics_df["residual_std"].to_numpy(), nlags=20, fft=True)
    acf_sq_vals = modules["acf"]((diagnostics_df["residual_std"] ** 2).to_numpy(), nlags=20, fft=True)
    conf = 1.96 / np.sqrt(len(diagnostics_df))
    lags = np.arange(len(acf_vals))

    fig.add_trace(
        go.Bar(x=lags[1:], y=acf_vals[1:], marker_color="#457B9D", showlegend=False),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=lags[1:], y=acf_sq_vals[1:], marker_color="#6A4C93", showlegend=False),
        row=2,
        col=2,
    )

    for row, col in [(2, 1), (2, 2)]:
        fig.add_hline(y=conf, line_dash="dot", line_color="#D62828", row=row, col=col)
        fig.add_hline(y=-conf, line_dash="dot", line_color="#D62828", row=row, col=col)
        fig.add_hline(y=0, line_color="#7A7A7A", row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


@st.cache_data(show_spinner=False)
def build_volatility_forecast(
    dates: tuple[str, ...],
    returns: tuple[float, ...],
    levels: tuple[float, ...],
    log_levels: tuple[float, ...],
    family: str,
    arch_order: int,
    dist: str,
    horizon: int,
):
    result = fit_volatility_model(returns, family, arch_order, dist)
    model = result["model"]
    forecast = model.forecast(horizon=horizon, reindex=False)
    variance = np.asarray(forecast.variance.iloc[-1])
    sigma = np.sqrt(variance)
    returns_series = pd.Series(returns).dropna().reset_index(drop=True)
    date_index = pd.to_datetime(pd.Series(dates))
    level_series = pd.Series(levels).dropna().reset_index(drop=True)
    log_level_series = pd.Series(log_levels).dropna().reset_index(drop=True)

    future_dates = pd.bdate_range(
        start=date_index.iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
    )
    mean_return = float(returns_series.mean())
    mean_path_pct = np.repeat(mean_return, horizon)
    cum_mean_log = log_level_series.iloc[-1] + np.cumsum(mean_path_pct / 100)
    cum_sigma_log = np.sqrt(np.cumsum(variance)) / 100
    lower_log = cum_mean_log - 1.96 * cum_sigma_log
    upper_log = cum_mean_log + 1.96 * cum_sigma_log

    return pd.DataFrame(
        {
            "fecha": future_dates,
            "horizonte": np.arange(1, horizon + 1),
            "volatilidad_pct": sigma,
            "varianza_pct2": variance,
            "retorno_esperado_pct": mean_path_pct,
            "banda_inferior_pct": -1.96 * sigma,
            "banda_superior_pct": 1.96 * sigma,
            "fx_esperado": np.exp(cum_mean_log),
            "fx_inferior": np.exp(lower_log),
            "fx_superior": np.exp(upper_log),
            "ultimo_fx": float(level_series.iloc[-1]),
        }
    )


def make_volatility_forecast_chart(
    history_dates: pd.Series,
    diagnostics_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lookback: int = 120,
) -> go.Figure:
    aligned_dates = history_dates.iloc[-len(diagnostics_df):].reset_index(drop=True)
    history_plot = pd.DataFrame(
        {
            "fecha": aligned_dates,
            "volatilidad_pct": diagnostics_df["conditional_vol"].to_numpy(),
        }
    ).tail(lookback)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_plot["fecha"],
            y=history_plot["volatilidad_pct"],
            mode="lines",
            name="Volatilidad estimada",
            line=dict(color="#1D3557", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["volatilidad_pct"],
            mode="lines+markers",
            name="Volatilidad esperada",
            line=dict(color="#D62828", width=2.4),
        )
    )
    fig.add_vline(
        x=forecast_df["fecha"].iloc[0],
        line_dash="dash",
        line_color="#7A7A7A",
        line_width=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Volatilidad reciente y pronosticada", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Desviacion estandar esperada (%)"),
        hovermode="x unified",
    )
    return fig


def make_return_band_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame, lookback: int = 120) -> go.Figure:
    history_plot = history_df[["fecha", "fx_return"]].dropna().tail(lookback)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_plot["fecha"],
            y=history_plot["fx_return"],
            mode="lines",
            name="Retorno historico",
            line=dict(color="#1D3557", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["banda_superior_pct"],
            mode="lines",
            name="Banda superior 95%",
            line=dict(color="#2A9D8F", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["banda_inferior_pct"],
            mode="lines",
            name="Banda inferior 95%",
            line=dict(color="#2A9D8F", width=2),
            fill="tonexty",
            fillcolor="rgba(42,157,143,0.18)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["retorno_esperado_pct"],
            mode="lines+markers",
            name="Retorno esperado",
            line=dict(color="#D62828", width=2.2),
        )
    )
    fig.add_vline(
        x=forecast_df["fecha"].iloc[0],
        line_dash="dash",
        line_color="#7A7A7A",
        line_width=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Variaciones futuras con contexto historico", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=f"Δln({FX_PAIR_LABEL}) (%)"),
        hovermode="x unified",
    )
    return fig


def make_fx_business_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame, lookback: int = 120) -> go.Figure:
    history_plot = history_df[["fecha", "fx"]].dropna().tail(lookback)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_plot["fecha"],
            y=history_plot["fx"],
            mode="lines",
            name=FX_PRICE_LABEL,
            line=dict(color="#1D3557", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["fx_esperado"],
            mode="lines+markers",
            name="Trayectoria esperada",
            line=dict(color="#D62828", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["fecha"], forecast_df["fecha"][::-1]]),
            y=pd.concat([forecast_df["fx_superior"], forecast_df["fx_inferior"][::-1]]),
            fill="toself",
            fillcolor="rgba(214,40,40,0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Banda 95%",
        )
    )
    fig.add_vline(
        x=forecast_df["fecha"].iloc[0],
        line_dash="dash",
        line_color="#7A7A7A",
        line_width=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Tipo de cambio esperado en escala de negocio", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=FX_PRICE_LABEL),
        hovermode="x unified",
    )
    return fig


@st.cache_data(show_spinner=False)
def compute_acf_profile(values: tuple[float, ...], max_lag: int) -> pd.DataFrame:
    acf_fn = get_statsmodels_modules()["acf"]
    series = pd.Series(values).dropna().to_numpy()
    acf_values = acf_fn(series, nlags=max_lag, fft=True)
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "acf": acf_values[1:]})


def make_subsample_acf_chart(profiles: dict[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=len(profiles),
        subplot_titles=tuple(profiles.keys()),
        shared_yaxes=True,
    )
    palette = ["#1D3557", "#D62828", "#2A9D8F"]
    for idx, ((label, profile), color) in enumerate(zip(profiles.items(), palette), start=1):
        conf = 1.96 / np.sqrt(max(len(profile), 1))
        fig.add_trace(
            go.Bar(
                x=profile["lag"],
                y=profile["acf"],
                marker_color=color,
                showlegend=False,
                hovertemplate="Rezago %{x}<br>ACF %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.add_hline(y=conf, line_dash="dot", line_color="#D62828", row=1, col=idx)
        fig.add_hline(y=-conf, line_dash="dot", line_color="#D62828", row=1, col=idx)
        fig.add_hline(y=0, line_color="#7A7A7A", row=1, col=idx)
        fig.update_xaxes(title_text="Rezago", row=1, col=idx)
    fig.update_yaxes(title_text="Autocorrelacion", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=30, r=30, t=70, b=30),
        title=dict(text="ACF de retornos por submuestras", font=dict(size=20)),
    )
    return fig


def make_subsample_acf_squared_chart(profiles: dict[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=len(profiles),
        subplot_titles=tuple(profiles.keys()),
        shared_yaxes=True,
    )
    palette = ["#6A4C93", "#E76F51", "#2A9D8F"]
    for idx, ((label, profile), color) in enumerate(zip(profiles.items(), palette), start=1):
        conf = 1.96 / np.sqrt(max(len(profile), 1))
        fig.add_trace(
            go.Bar(
                x=profile["lag"],
                y=profile["acf"],
                marker_color=color,
                showlegend=False,
                hovertemplate="Rezago %{x}<br>ACF %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.add_hline(y=conf, line_dash="dot", line_color="#D62828", row=1, col=idx)
        fig.add_hline(y=-conf, line_dash="dot", line_color="#D62828", row=1, col=idx)
        fig.add_hline(y=0, line_color="#7A7A7A", row=1, col=idx)
        fig.update_xaxes(title_text="Rezago", row=1, col=idx)
    fig.update_yaxes(title_text="Autocorrelacion", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=30, r=30, t=70, b=30),
        title=dict(text="ACF de retornos al cuadrado por submuestras", font=dict(size=20)),
    )
    return fig


@st.cache_data(show_spinner=False)
def evaluate_short_ar_models(returns: tuple[float, ...], holdout_size: int = 60) -> pd.DataFrame:
    ARIMA = get_statsmodels_modules()["ARIMA"]
    series = pd.Series(returns).dropna().reset_index(drop=True)
    holdout = min(holdout_size, max(20, len(series) // 5))
    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    rows = []
    mean_forecast = np.repeat(train.mean(), len(test))
    mean_rmse = float(np.sqrt(np.mean((test.to_numpy() - mean_forecast) ** 2)))
    rows.append(
        {
            "modelo": "Media historica",
            "rmse": mean_rmse,
            "mae": float(np.mean(np.abs(test.to_numpy() - mean_forecast))),
            "directional_accuracy": float(np.mean(np.sign(test.to_numpy()) == np.sign(mean_forecast))),
        }
    )

    for order_p in [1, 3, 5]:
        try:
            model = ARIMA(train, order=(order_p, 0, 0), trend="c").fit()
            preds = np.asarray(model.get_forecast(steps=len(test)).predicted_mean)
            rows.append(
                {
                    "modelo": f"AR({order_p})",
                    "rmse": float(np.sqrt(np.mean((test.to_numpy() - preds) ** 2))),
                    "mae": float(np.mean(np.abs(test.to_numpy() - preds))),
                    "directional_accuracy": float(np.mean(np.sign(test.to_numpy()) == np.sign(preds))),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fit_contemporaneous_regression(df: pd.DataFrame):
    sm = get_statsmodels_modules()["sm"]
    reg_df = df[["fecha", "dlog_gasoline", "dlog_oil"]].dropna().copy()
    y = reg_df["dlog_gasoline"]
    x = reg_df[["dlog_oil"]]
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    reg_df["fitted"] = model.predict(x)
    reg_df["residual"] = model.resid
    return reg_df, model


@st.cache_data(show_spinner=False)
def fit_contemporaneous_regression_without_outliers(df: pd.DataFrame):
    sm = get_statsmodels_modules()["sm"]
    reg_df = df[["fecha", "dlog_gasoline", "dlog_oil"]].dropna().copy()
    y = reg_df["dlog_gasoline"]
    x = reg_df[["dlog_oil"]]
    base_model = sm.OLS(y, x).fit()
    influence = base_model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(reg_df)

    clean_df = reg_df.copy()
    clean_df["cooks_d"] = cooks_d
    clean_df["outlier_flag"] = clean_df["cooks_d"] > threshold
    filtered_df = clean_df.loc[~clean_df["outlier_flag"]].copy()

    filtered_y = filtered_df["dlog_gasoline"]
    filtered_x = filtered_df[["dlog_oil"]]
    filtered_model = sm.OLS(filtered_y, filtered_x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    filtered_df["fitted"] = filtered_model.predict(filtered_x)
    filtered_df["residual"] = filtered_model.resid

    excluded_df = clean_df.loc[clean_df["outlier_flag"], ["fecha", "dlog_gasoline", "dlog_oil", "cooks_d"]].copy()

    return filtered_df, filtered_model, excluded_df, threshold


def make_scatter_with_fit(reg_df: pd.DataFrame, model) -> go.Figure:
    slope = model.params["dlog_oil"]
    sorted_df = reg_df.sort_values("dlog_oil")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reg_df["dlog_oil"],
            y=reg_df["dlog_gasoline"],
            mode="markers",
            marker=dict(color="#457B9D", size=7, opacity=0.7),
            name="Observaciones",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_df["dlog_oil"],
            y=slope * sorted_df["dlog_oil"],
            mode="lines",
            line=dict(color="#D62828", width=2.5),
            name="Ajuste sin intercepto",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Relacion contemporanea entre cambios", font=dict(size=20)),
        xaxis=dict(title="Δln(petroleo) (%)"),
        yaxis=dict(title="Δln(gasolina) (%)"),
    )
    return fig


def make_scatter_with_residual_examples(reg_df: pd.DataFrame, model, max_points: int = 6) -> go.Figure:
    slope = model.params["dlog_oil"]
    sorted_df = reg_df.sort_values("dlog_oil")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reg_df["dlog_oil"],
            y=reg_df["dlog_gasoline"],
            mode="markers",
            marker=dict(color="#A8DADC", size=6, opacity=0.65),
            name="Observaciones",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_df["dlog_oil"],
            y=slope * sorted_df["dlog_oil"],
            mode="lines",
            line=dict(color="#1D3557", width=2.5),
            name="Recta ajustada",
        )
    )

    example_df = reg_df.copy()
    example_df["residual_abs"] = example_df["residual"].abs()
    example_df = example_df.nlargest(max_points, "residual_abs").sort_values("dlog_oil")

    for _, row in example_df.iterrows():
        fitted_y = slope * row["dlog_oil"]
        fig.add_trace(
            go.Scatter(
                x=[row["dlog_oil"], row["dlog_oil"]],
                y=[fitted_y, row["dlog_gasoline"]],
                mode="lines",
                line=dict(color="#D62828", width=2, dash="dot"),
                showlegend=False,
                hovertemplate=(
                    f"Fecha: {row['fecha']:%Y-%m-%d}<br>"
                    f"Residuo: {row['residual']:.3f}<extra></extra>"
                ),
            )
        )
    customdata = pd.DataFrame(
        {
            "fecha_label": example_df["fecha"].dt.strftime("%Y-%m-%d"),
            "residual": example_df["residual"],
        }
    ).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=example_df["dlog_oil"],
            y=example_df["dlog_gasoline"],
            mode="markers",
            marker=dict(color="#D62828", size=9, symbol="diamond"),
            name="Ejemplos de residuos",
            hovertemplate=(
                "Fecha: %{customdata[0]}<br>"
                "Δln(petroleo): %{x:.3f}<br>"
                "Δln(gasolina): %{y:.3f}<br>"
                "Residuo: %{customdata[1]:.3f}<extra></extra>"
            ),
            customdata=customdata,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=470,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Ejemplos visuales de residuos", font=dict(size=20)),
        xaxis=dict(title="Δln(petroleo) (%)"),
        yaxis=dict(title="Δln(gasolina) (%)"),
    )
    return fig


@st.cache_data(show_spinner=False)
def fit_candidate_model(
    model_family: str,
    order_p: int,
    y_train: tuple[float, ...],
    y_test: tuple[float, ...],
    x_train: tuple[float, ...],
    x_test: tuple[float, ...],
):
    modules = get_statsmodels_modules()
    ARIMA = modules["ARIMA"]
    acorr_ljungbox = modules["acorr_ljungbox"]

    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    x_train_series = pd.Series(x_train) if x_train else None
    x_test_series = pd.Series(x_test) if x_test else None

    if model_family == "AR":
        model = ARIMA(y_train_series, order=(order_p, 0, 0), trend="n").fit()
        forecast_res = model.get_forecast(steps=len(y_test_series))
    else:
        model = ARIMA(
            y_train_series,
            order=(order_p, 0, 0),
            trend="n",
            exog=x_train_series,
        ).fit()
        forecast_res = model.get_forecast(steps=len(y_test_series), exog=x_test_series)

    residuals = pd.Series(model.resid).dropna()
    lb = acorr_ljungbox(residuals, lags=[12], return_df=True)
    predictions = pd.Series(forecast_res.predicted_mean, index=y_test_series.index)
    rmse = float(np.sqrt(np.mean((y_test_series - predictions) ** 2)))

    return {
        "family": model_family,
        "order": order_p,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "rmse": rmse,
    }


@st.cache_data(show_spinner=False)
def evaluate_candidate_models(df: pd.DataFrame, max_p: int, holdout_size: int) -> pd.DataFrame:
    model_df = df[["fecha", "dlog_gasoline", "dlog_oil"]].dropna().copy()
    train = model_df.iloc[:-holdout_size].copy()
    test = model_df.iloc[-holdout_size:].copy()

    rows = []
    y_train = tuple(train["dlog_gasoline"].to_numpy())
    y_test = tuple(test["dlog_gasoline"].to_numpy())
    x_train = tuple(train["dlog_oil"].to_numpy())
    x_test = tuple(test["dlog_oil"].to_numpy())

    for family in ["AR", "ARIMAX"]:
        for order_p in range(1, max_p + 1):
            try:
                rows.append(
                    fit_candidate_model(
                        family,
                        order_p,
                        y_train,
                        y_test,
                        x_train,
                        x_test,
                    )
                )
            except Exception:
                continue

    return pd.DataFrame(rows).sort_values(["family", "rmse", "aic"]).reset_index(drop=True)


def choose_recommended_model(candidates: pd.DataFrame) -> pd.Series:
    valid = candidates[candidates["ljung_box_pvalue"] >= 0.05].copy()
    if valid.empty:
        valid = candidates.copy()
    valid = valid.sort_values(["rmse", "aic", "order"]).reset_index(drop=True)
    return valid.iloc[0]


@st.cache_data(show_spinner=False)
def fit_final_model(df: pd.DataFrame, family: str, order_p: int):
    ARIMA = get_statsmodels_modules()["ARIMA"]
    model_df = df[["fecha", "dlog_gasoline", "dlog_oil"]].dropna().copy()
    y = model_df["dlog_gasoline"]
    if family == "AR":
        model = ARIMA(y, order=(order_p, 0, 0), trend="n").fit()
    else:
        model = ARIMA(
            y,
            order=(order_p, 0, 0),
            trend="n",
            exog=model_df["dlog_oil"],
        ).fit()
    model_df["fitted"] = model.fittedvalues
    model_df["residual"] = model.resid
    return model_df, model


def make_model_fit_chart(model_df: pd.DataFrame, family: str, order_p: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=model_df["fecha"],
            y=model_df["dlog_gasoline"],
            mode="lines",
            name="Observado",
            line=dict(color=SERIES_CONFIG["gasoline"]["color"], width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=model_df["fecha"],
            y=model_df["fitted"],
            mode="lines",
            name="Ajustado",
            line=dict(color="#2A9D8F", width=2.2),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Ajuste del modelo {family}({order_p})", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Δln(gasolina) (%)"),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    return fig


def build_future_exog_path(df: pd.DataFrame, horizon: int, scenario: str) -> pd.Series:
    recent = df["dlog_oil"].dropna()
    last_value = float(recent.iloc[-1])
    avg_4 = float(recent.tail(4).mean())
    avg_12 = float(recent.tail(12).mean())

    if scenario == "Ultima observacion":
        value = last_value
    elif scenario == "Promedio ultimas 4 semanas":
        value = avg_4
    else:
        value = avg_12

    return pd.Series([value] * horizon)


@st.cache_data(show_spinner=False)
def build_forecast_outputs(
    df: pd.DataFrame,
    family: str,
    order_p: int,
    horizon: int,
    oil_scenario: str,
):
    model_df, model = fit_final_model(df, family, order_p)

    if family == "AR":
        forecast_res = model.get_forecast(steps=horizon)
        exog_assumption = None
    else:
        exog_assumption = build_future_exog_path(df, horizon, oil_scenario)
        forecast_res = model.get_forecast(steps=horizon, exog=exog_assumption)

    forecast_mean = pd.Series(forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    future_dates = pd.date_range(
        start=df["fecha"].iloc[-1] + pd.Timedelta(days=7),
        periods=horizon,
        freq="W-FRI",
    )

    last_log = float(df["log_gasoline"].iloc[-1])
    mean_log = last_log + forecast_mean.cumsum() / 100
    lower_log = last_log + conf_int.iloc[:, 0].cumsum() / 100
    upper_log = last_log + conf_int.iloc[:, 1].cumsum() / 100

    forecast_table = pd.DataFrame(
        {
            "fecha": future_dates,
            "forecast_dlog_gasoline": forecast_mean.to_numpy(),
            "lower_dlog_gasoline": conf_int.iloc[:, 0].to_numpy(),
            "upper_dlog_gasoline": conf_int.iloc[:, 1].to_numpy(),
            "forecast_log_gasoline": mean_log.to_numpy(),
            "lower_log_gasoline": lower_log.to_numpy(),
            "upper_log_gasoline": upper_log.to_numpy(),
            "forecast_gasoline": np.exp(mean_log.to_numpy()),
            "lower_gasoline": np.exp(lower_log.to_numpy()),
            "upper_gasoline": np.exp(upper_log.to_numpy()),
        }
    )

    if exog_assumption is not None:
        forecast_table["assumed_dlog_oil"] = exog_assumption.to_numpy()

    return model_df, model, forecast_table


def make_forecast_growth_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["fecha"],
            y=history_df["dlog_gasoline"],
            mode="lines",
            name="Historico",
            line=dict(color=SERIES_CONFIG["gasoline"]["color"], width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["forecast_dlog_gasoline"],
            mode="lines+markers",
            name="Pronostico",
            line=dict(color="#2A9D8F", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["fecha"], forecast_df["fecha"][::-1]]),
            y=pd.concat([forecast_df["upper_dlog_gasoline"], forecast_df["lower_dlog_gasoline"][::-1]]),
            fill="toself",
            fillcolor="rgba(42,157,143,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="IC 95%",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Pronostico en la escala modelada", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Δln(gasolina) (%)"),
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    return fig


def make_forecast_level_chart(df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["gasoline"],
            mode="lines",
            name="Historico",
            line=dict(color=SERIES_CONFIG["gasoline"]["color"], width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df["forecast_gasoline"],
            mode="lines+markers",
            name="Pronostico",
            line=dict(color="#2A9D8F", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["fecha"], forecast_df["fecha"][::-1]]),
            y=pd.concat([forecast_df["upper_gasoline"], forecast_df["lower_gasoline"][::-1]]),
            fill="toself",
            fillcolor="rgba(42,157,143,0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="IC 95%",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Pronostico en valores de negocio", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="USD por galon"),
        hovermode="x unified",
    )
    return fig


def format_table_for_display(df: pd.DataFrame, formats: dict[str, str]) -> pd.DataFrame:
    display_df = df.copy()
    for column, fmt in formats.items():
        if column not in display_df.columns:
            continue
        display_df[column] = display_df[column].map(
            lambda x: "" if pd.isna(x) else fmt.format(x)
        )
    return display_df


st.title("Series Dinamicas y Riesgo")
st.caption("Primera version de la app con dos modulos: series dinamicas con variable exogena y modelacion de volatilidad.")

with st.sidebar:
    st.header("Controles")
    start_date = st.date_input("Fecha inicial", value=pd.Timestamp("2010-01-01"))
    max_p = st.slider("Maximo orden AR a explorar", min_value=1, max_value=8, value=5)
    holdout_size = st.slider("Semanas para backtesting", min_value=12, max_value=104, value=52, step=4)
    forecast_horizon = st.slider("Horizonte de pronostico", min_value=4, max_value=16, value=8)
    oil_scenario = st.selectbox(
        "Supuesto para el petroleo futuro",
        [
            "Ultima observacion",
            "Promedio ultimas 4 semanas",
            "Promedio ultimas 12 semanas",
        ],
    )

data = None
fx_data = None
gas_error = None
fx_error = None

with st.spinner("Descargando datos externos..."):
    try:
        data = load_gasoline_oil_data(str(start_date))
    except Exception as exc:
        gas_error = exc

    try:
        fx_data = load_fx_data(str(start_date))
    except Exception as exc:
        fx_error = exc

if data is None or fx_data is None:
    st.error("No pude cargar todas las fuentes externas necesarias para ejecutar la app.")
    if gas_error is not None:
        st.caption(f"Gasolina/Petroleo: {type(gas_error).__name__}: {gas_error}")
    if fx_error is not None:
        st.caption(f"{FX_PAIR_LABEL}: {type(fx_error).__name__}: {fx_error}")
    st.stop()

if len(data.dropna()) <= holdout_size + max_p + 10:
    st.error("El modulo de gasolina queda con muy pocas observaciones para modelar con esos controles.")
    st.stop()

tab1, tab2 = st.tabs(["Gasolina y Petroleo", "Volatilidad FX"])

with tab1:
    info_banner(
        "En este modulo queremos responder una pregunta concreta: "
        "si el petroleo ayuda a explicar y pronosticar los cambios de la gasolina. "
        "La idea no es solo correr modelos, sino construir una receta clara para pasar de series en bruto a un pronostico interpretable."
    )

    section_title("1. La Pregunta del Modulo")
    info_banner(
        "La intuicion economica es sencilla: el petroleo es un insumo clave en el mercado de combustibles. "
        "Por eso esperamos que movimientos del petroleo tengan alguna relacion con los movimientos de la gasolina. "
        "Pero no sabemos todavia si esa relacion es fuerte, contemporanea o util para pronosticar. Ese es justamente el problema que vamos a resolver paso a paso."
    )
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Observaciones semanales", f"{len(data):,}")
    metric_col2.metric("Inicio de muestra", data["fecha"].min().strftime("%Y-%m-%d"))
    metric_col3.metric("Fin de muestra", data["fecha"].max().strftime("%Y-%m-%d"))

    section_title("2. Hoja de Ruta")
    info_banner(
        "La receta del modulo tiene cinco momentos. Primero miramos las series y entendemos su escala. "
        "Luego las transformamos para volverlas aptas para modelar. Despues medimos la relacion entre gasolina y petroleo. "
        "Enseguida comparamos un modelo que usa solo historia propia contra otro que tambien usa petroleo. "
        "Y al final convertimos el pronostico a una escala de negocio."
    )
    roadmap = pd.DataFrame(
        {
            "Paso": [
                "Ver las series",
                "Transformarlas",
                "Medir la relacion",
                "Comparar modelos",
                "Pronosticar y volver a niveles",
            ],
            "Pregunta que responde": [
                "Tienen tendencia o cambios de escala?",
                "Como las dejamos listas para modelar?",
                "El petroleo ayuda a explicar gasolina?",
                "Gana un AR o un ARIMAX?",
                "Como llegamos a precios proyectados?",
            ],
        }
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

    section_title("3. Ver las Series en Niveles")
    info_banner(
        "Arrancamos en niveles porque esta es la forma mas natural de leer las series. "
        "Aqui aun no modelamos nada. Solo buscamos tres cosas: tendencia, cambios de escala y momentos en los que las dos series parecen moverse juntas."
    )
    st.plotly_chart(make_level_chart(data), use_container_width=True)

    section_title("4. Pasar a Logaritmos")
    info_banner(
        "El logaritmo es util cuando queremos pensar en cambios relativos y no solo absolutos. "
        "En series de precios, esto suele volver mas comparable la dinamica y prepara el camino para hablar de crecimientos."
    )
    st.plotly_chart(make_log_chart(data), use_container_width=True)

    section_title("5. Dejar la Serie Lista para Modelar")
    info_banner(
        "El siguiente paso es tomar diferencias logaritmicas semanales. "
        "En la practica, esto se interpreta como una variacion porcentual aproximada. "
        "Lo hacemos porque los modelos suelen trabajar mejor con cambios que con niveles cuando las series tienen tendencia."
    )
    transformed = data[["fecha", "dlog_gasoline", "dlog_oil"]].dropna().copy()
    st.plotly_chart(make_growth_chart(transformed), use_container_width=True)
    info_banner(
        "A partir de aqui cambia la pregunta. Ya no preguntamos cuanto vale la gasolina, sino cuanto cambia de una semana a otra. "
        "Esa es la escala sobre la cual vamos a explicar, comparar y pronosticar."
    )
    transformed_table = format_table_for_display(
        transformed.tail(12).reset_index(drop=True),
        {"dlog_gasoline": "{:.3f}", "dlog_oil": "{:.3f}"},
    )
    st.dataframe(transformed_table, use_container_width=True, hide_index=True)

    section_title("6. Mirar la Memoria Propia de la Gasolina")
    info_banner(
        "Antes de meter el petroleo, conviene preguntar si la propia gasolina ya tiene memoria. "
        "La ACF nos muestra memoria total frente a rezagos pasados. La PACF nos muestra memoria directa una vez descontamos rezagos intermedios. "
        "Si encontramos patron, un modelo AR tiene sentido como punto de partida."
    )
    st.plotly_chart(
        make_acf_pacf_figure(transformed["dlog_gasoline"].to_numpy(), max_lag=20, title_prefix="Δln(gasolina)"),
        use_container_width=True,
    )

    section_title("7. Medir la Relacion entre Gasolina y Petroleo")
    info_banner(
        "Ahora si conectamos ambas series. La pregunta es: en la misma semana, cuando cambia el petroleo, tambien cambia la gasolina? "
        "Usamos una regresion sin intercepto porque estamos trabajando en cambios y queremos medir una elasticidad contemporanea simple."
    )
    reg_df, reg_model = fit_contemporaneous_regression(data)
    reg_col1, reg_col2, reg_col3 = st.columns(3)
    reg_col1.metric("Coeficiente del petroleo", f"{reg_model.params['dlog_oil']:.3f}")
    reg_col2.metric("p-valor de beta", f"{reg_model.pvalues['dlog_oil']:.4f}")
    reg_col3.metric("R cuadrado", f"{reg_model.rsquared:.3f}")
    st.plotly_chart(make_scatter_with_fit(reg_df, reg_model), use_container_width=True)
    info_banner(
        "Lectura economica: si beta es positiva y significativa, semanas con mayores cambios del petroleo tienden a coincidir con mayores cambios de la gasolina. "
        "Eso ya nos dice algo importante, pero todavia no sabemos si esa relacion basta para describir bien la dinamica."
    )

    section_title("7A. Regresion sin Observaciones Extremas")
    info_banner(
        "A veces unos pocos episodios muy extremos inclinan demasiado la recta de ajuste. "
        "Por eso vale la pena repetir la regresion quitando observaciones influyentes, medidas con distancia de Cook."
    )
    reg_df_clean, reg_model_clean, excluded_points, cook_threshold = fit_contemporaneous_regression_without_outliers(data)
    clean_col1, clean_col2, clean_col3 = st.columns(3)
    clean_col1.metric("Coeficiente sin extremos", f"{reg_model_clean.params['dlog_oil']:.3f}")
    clean_col2.metric("p-valor sin extremos", f"{reg_model_clean.pvalues['dlog_oil']:.4f}")
    clean_col3.metric("R cuadrado sin extremos", f"{reg_model_clean.rsquared:.3f}")
    st.plotly_chart(make_scatter_with_fit(reg_df_clean, reg_model_clean), use_container_width=True)
    info_banner(
        f"Aqui excluimos los puntos con distancia de Cook mayor que {cook_threshold:.4f}. "
        "Si el coeficiente cambia mucho frente a la regresion completa, eso sugiere que la relacion estaba siendo empujada por unas pocas semanas atipicas."
    )
    if not excluded_points.empty:
        st.dataframe(
            format_table_for_display(
                excluded_points.sort_values("fecha").reset_index(drop=True),
                {"dlog_gasoline": "{:.3f}", "dlog_oil": "{:.3f}", "cooks_d": "{:.4f}"},
            ),
            use_container_width=True,
            hide_index=True,
        )

    section_title("7B. Que son los Residuos y por que Importan")
    info_banner(
        "Un residuo es la parte del cambio de la gasolina que la regresion no logro explicar con el petroleo. "
        "En otras palabras, es la distancia vertical entre el dato observado y el valor ajustado por la recta."
    )
    st.plotly_chart(
        make_scatter_with_residual_examples(reg_df_clean, reg_model_clean, max_points=6),
        use_container_width=True,
    )
    info_banner(
        "Si los residuos fueran puro ruido, la regresion ya habria capturado lo esencial de la relacion. "
        "Pero si los residuos muestran patron o memoria, eso significa que todavia queda dinamica sin explicar y necesitamos un modelo mas rico."
    )

    section_title("8. Revisar lo que la Regresion No Explico")
    info_banner(
        "Los residuos son la parte que la regresion no logro explicar. "
        "Si ahi todavia aparece memoria temporal, la conclusion es clara: no basta con relacionar gasolina y petroleo en la misma semana. "
        "Todavia hay estructura dinamica por capturar."
    )
    st.plotly_chart(
        make_acf_pacf_figure(reg_df["residual"].to_numpy(), max_lag=20, title_prefix="residuos"),
        use_container_width=True,
    )
    reg_lb = summarize_ljung_box(tuple(reg_df["residual"].to_numpy()), lag=12)
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Ljung-Box residuos", f"{reg_lb['stat']:.2f}")
    res_col2.metric("p-valor residuos", f"{reg_lb['pvalue']:.4f}")
    info_banner(
        "Como leer esta grafica: si varios rezagos salen claramente de las bandas, los residuos no son ruido blanco. "
        "Eso quiere decir que, despues de explicar gasolina con petroleo en la misma semana, todavia queda una parte sistematica sin modelar."
    )
    info_banner(residual_interpretation(reg_lb["pvalue"], reg_lb["lag"]))

    section_title("9. Comparar dos Estrategias de Pronostico")
    info_banner(
        "Llegamos al punto central del modulo. Vamos a comparar dos estrategias. "
        "La primera usa solo la historia de la gasolina. La segunda usa la historia de la gasolina mas la informacion del petroleo. "
        "La decision no se toma por intuicion: la tomamos comparando ajuste, residuos y desempeno fuera de muestra."
    )
    with st.spinner("Evaluando candidatos AR y ARIMAX..."):
        candidates = evaluate_candidate_models(data, max_p=max_p, holdout_size=holdout_size)

    if candidates.empty:
        st.warning("No logre estimar candidatos con la configuracion actual.")
        st.stop()

    top_ar = candidates[candidates["family"] == "AR"].head(3)
    top_arimax = candidates[candidates["family"] == "ARIMAX"].head(3)

    top_cols = st.columns(2)
    top_cols[0].markdown("**Mejores candidatos AR**")
    top_cols[0].dataframe(
        format_table_for_display(
            top_ar,
            {"aic": "{:.2f}", "bic": "{:.2f}", "ljung_box_pvalue": "{:.4f}", "rmse": "{:.3f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )
    top_cols[1].markdown("**Mejores candidatos ARIMAX**")
    top_cols[1].dataframe(
        format_table_for_display(
            top_arimax,
            {"aic": "{:.2f}", "bic": "{:.2f}", "ljung_box_pvalue": "{:.4f}", "rmse": "{:.3f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )

    recommended = choose_recommended_model(candidates)
    info_banner(
        f"Modelo recomendado: {recommended['family']}({int(recommended['order'])}). "
        f"Lo elegimos porque combina un RMSE fuera de muestra de {recommended['rmse']:.3f} con un AIC de {recommended['aic']:.2f}. "
        "Ademas, si el p-valor de Ljung-Box es alto, eso sugiere que al modelo le queda menos estructura por explicar."
    )

    section_title("10. Entender el Modelo que Gano")
    info_banner(
        "Antes de pronosticar, vale la pena entender que significa el modelo ganador. "
        "La pregunta aqui no es solo si ajusta bien, sino que historia economica nos esta contando."
    )
    final_model_df, final_model = fit_final_model(
        data,
        family=str(recommended["family"]),
        order_p=int(recommended["order"]),
    )
    st.plotly_chart(
        make_model_fit_chart(final_model_df, str(recommended["family"]), int(recommended["order"])),
        use_container_width=True,
    )
    info_banner(build_model_equation_text(str(recommended["family"]), int(recommended["order"]), final_model))
    params_table = pd.DataFrame(
        {
            "Parametro": final_model.params.index,
            "Valor": final_model.params.values,
            "p_valor": final_model.pvalues.values,
        }
    )
    st.dataframe(
        format_table_for_display(params_table, {"Valor": "{:.4f}", "p_valor": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    if str(recommended["family"]) == "AR":
        info_banner(
            "Lectura economica: un modelo AR implica que la gasolina guarda memoria de sus propios movimientos recientes. "
            "Cuando un coeficiente AR es relevante, una parte del cambio actual se explica por la propia inercia de semanas anteriores."
        )
    else:
        info_banner(
            "Lectura economica: un modelo ARIMAX mezcla dos ideas. La primera es persistencia propia en la gasolina. "
            "La segunda es transmision de shocks desde el petroleo. Si el coeficiente del petroleo es relevante, cambios en el insumo energetico agregan poder explicativo al modelo."
        )
    info_banner(
        "Como leer los p-valores aqui: un p-valor bajo en un coeficiente AR sugiere que ese rezago realmente aporta memoria util. "
        "Un p-valor bajo en el coeficiente del petroleo sugiere que la variable exogena agrega informacion estadisticamente relevante. "
        "Si un coeficiente tiene p-valor alto, su aporte es menos claro y podriamos cuestionar si vale la pena mantenerlo."
    )

    section_title("11. Verificar los Residuos del Modelo Final")
    info_banner(
        "Despues de elegir el modelo, toca revisar si realmente limpiamos la dinamica. "
        "Un buen modelo deja residuos con poca autocorrelacion: eso significa que la parte predecible de la serie ya fue absorbida por la estructura estimada."
    )
    st.plotly_chart(
        make_acf_pacf_figure(final_model_df["residual"].dropna().to_numpy(), max_lag=20, title_prefix="residuos del modelo final"),
        use_container_width=True,
    )
    final_lb = summarize_ljung_box(tuple(final_model_df["residual"].dropna().to_numpy()), lag=12)
    final_res_col1, final_res_col2 = st.columns(2)
    final_res_col1.metric("Ljung-Box modelo final", f"{final_lb['stat']:.2f}")
    final_res_col2.metric("p-valor modelo final", f"{final_lb['pvalue']:.4f}")
    info_banner(
        "La diferencia entre este chequeo y el anterior es importante. Antes mirabamos los residuos de una regresion simple. "
        "Ahora miramos los residuos del modelo completo. Si las barras se reducen y el p-valor sube, el modelo efectivamente capturo mejor la estructura temporal."
    )
    info_banner(residual_interpretation(final_lb["pvalue"], final_lb["lag"]))

    section_title("12. Pronosticar en la Escala del Modelo")
    info_banner(
        "El primer pronostico sale en la escala en la que estimamos el modelo: cambios semanales de la gasolina. "
        "Eso esta bien estadisticamente, pero todavia no es la escala que un usuario de negocio quiere leer."
    )
    _, _, forecast_table = build_forecast_outputs(
        data,
        family=str(recommended["family"]),
        order_p=int(recommended["order"]),
        horizon=forecast_horizon,
        oil_scenario=oil_scenario,
    )
    st.plotly_chart(
        make_forecast_growth_chart(final_model_df, forecast_table),
        use_container_width=True,
    )

    section_title("13. Volver a la Escala de Negocio")
    info_banner(
        "Ahora deshacemos la transformacion paso a paso. Como solo usamos una diferencia logaritmica, el camino inverso es directo: "
        "pronosticamos el cambio, lo acumulamos sobre el ultimo logaritmo observado y luego aplicamos exponencial para regresar a precios."
    )
    example_row = forecast_table.iloc[0]
    example_text = (
        f"Ejemplo del primer paso: si el pronostico para la primera semana es {example_row['forecast_dlog_gasoline']:.3f}%, "
        f"sumamos ese cambio al ultimo ln(precio) observado y luego aplicamos exp(.) para obtener un precio proyectado de "
        f"{example_row['forecast_gasoline']:.3f} USD por galon."
    )
    info_banner(example_text)
    st.plotly_chart(make_forecast_level_chart(data, forecast_table), use_container_width=True)

    display_forecast = forecast_table.copy()
    st.dataframe(
        format_table_for_display(
            display_forecast,
            {
                "forecast_dlog_gasoline": "{:.3f}",
                "lower_dlog_gasoline": "{:.3f}",
                "upper_dlog_gasoline": "{:.3f}",
                "forecast_log_gasoline": "{:.4f}",
                "forecast_gasoline": "{:.3f}",
                "lower_gasoline": "{:.3f}",
                "upper_gasoline": "{:.3f}",
                "assumed_dlog_oil": "{:.3f}",
            },
        ),
        use_container_width=True,
        hide_index=True,
    )

    section_title("14. Cierre del Modulo")
    info_banner(
        f"En esta muestra, el modelo recomendado fue {recommended['family']}({int(recommended['order'])}). "
        "La lectura mas importante no es solo estadistica: queremos saber si la historia propia de la gasolina basta o si el petroleo agrega informacion util. "
        "Ese es el mensaje central del modulo y deja lista la transicion hacia pronosticos dinamicos mas ricos."
    )

with tab2:
    fx_arch_order = st.slider("Orden ARCH a explorar", min_value=1, max_value=12, value=5, key="fx_arch_order")
    fx_dist_label = st.selectbox(
        "Distribucion para volatilidad",
        ["Normal", "t-Student"],
        key="fx_dist",
    )
    fx_horizon = st.slider("Horizonte de volatilidad", min_value=5, max_value=30, value=10, key="fx_horizon")
    fx_dist = "normal" if fx_dist_label == "Normal" else "t"

    info_banner(
        f"En este modulo queremos responder otra pregunta distinta sobre {FX_PAIR_LABEL}: no tanto hacia donde va el tipo de cambio, "
        "sino cuan inestable puede estar. Aqui el foco pasa del pronostico del nivel al pronostico del riesgo."
    )

    section_title("1. La Pregunta del Modulo")
    info_banner(
        "En muchas series financieras el problema central no es predecir el retorno promedio, sino la intensidad de los movimientos. "
        "Eso es justamente lo que capturan los modelos de volatilidad condicional como ARCH y GARCH."
    )
    fx_col1, fx_col2, fx_col3 = st.columns(3)
    fx_col1.metric("Observaciones diarias", f"{len(fx_data):,}")
    fx_col2.metric("Inicio de muestra", fx_data["fecha"].min().strftime("%Y-%m-%d"))
    fx_col3.metric("Fin de muestra", fx_data["fecha"].max().strftime("%Y-%m-%d"))

    section_title("2. Hoja de Ruta")
    info_banner(
        f"La receta aqui tiene cinco momentos. Primero miramos {FX_PAIR_LABEL} y sus retornos. "
        "Luego validamos si los retornos parecen ruido blanco o si hay evidencia de clustering de volatilidad. "
        "Despues probamos el efecto ARCH. Luego comparamos un ARCH con un GARCH. Y al final pronosticamos volatilidad."
    )
    fx_roadmap = pd.DataFrame(
        {
            "Paso": [
                "Ver la serie",
                "Construir retornos",
                "Buscar clustering",
                "Comparar ARCH y GARCH",
                "Pronosticar volatilidad",
            ],
            "Pregunta que responde": [
                f"Como se mueve {FX_PAIR_LABEL}?",
                "En que escala modelamos?",
                "La volatilidad tiene memoria?",
                "Que estructura describe mejor el riesgo?",
                "Cuanto riesgo esperamos hacia adelante?",
            ],
        }
    )
    st.dataframe(fx_roadmap, use_container_width=True, hide_index=True)

    section_title("3. Ver el Tipo de Cambio")
    info_banner(
        f"Primero miramos {FX_PAIR_LABEL} en niveles y en logaritmos. Esto nos deja claro que el tipo de cambio puede cambiar de nivel con el tiempo, "
        "pero esa vista aun no nos dice mucho sobre la volatilidad."
    )
    st.plotly_chart(make_fx_level_chart(fx_data), use_container_width=True)

    section_title("4. Pasar a Retornos")
    info_banner(
        "Para estudiar volatilidad, la escala natural son los retornos logaritmicos. "
        f"Aqui la pregunta ya no es cuanto vale {FX_PAIR_LABEL}, sino cuanto cambia de un dia al siguiente."
    )
    st.plotly_chart(make_fx_returns_chart(fx_data), use_container_width=True)
    st.dataframe(
        format_table_for_display(
            fx_data[["fecha", "fx", "log_fx", "fx_return"]].tail(12).reset_index(drop=True),
            {"fx": "{:.4f}", "log_fx": "{:.4f}", "fx_return": "{:.3f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )

    section_title("5. Diagnostico Inicial de los Retornos")
    info_banner(
        "En muchas series financieras los retornos no muestran mucha autocorrelacion en promedio, "
        "pero si muestran persistencia en su magnitud. Esa es una pista temprana de volatilidad condicional."
    )
    fx_diag = summarize_fx_diagnostics(tuple(fx_data["fx_return"].to_numpy()))
    diag_cols = st.columns(4)
    diag_cols[0].metric("Media (%)", f"{fx_diag['mean']:.4f}")
    diag_cols[1].metric("Desviacion estandar (%)", f"{fx_diag['std']:.4f}")
    diag_cols[2].metric("p-valor t-test", f"{fx_diag['t_pvalue']:.4f}")
    diag_cols[3].metric("p-valor Ljung-Box retornos", f"{fx_diag['lb_returns_pvalue']:.4f}")
    st.plotly_chart(make_return_abs_chart(fx_data), use_container_width=True)
    acf_cols = st.columns(2)
    acf_cols[0].plotly_chart(
        make_acf_pacf_figure(fx_data["fx_return"].to_numpy(), max_lag=24, title_prefix="retornos"),
        use_container_width=True,
    )
    acf_cols[1].plotly_chart(
        make_acf_pacf_figure(fx_data["abs_fx_return"].to_numpy(), max_lag=24, title_prefix="retornos absolutos"),
        use_container_width=True,
    )
    info_banner(
        "Si los retornos tienen poca autocorrelacion pero los retornos absolutos si muestran patron, la lectura economica es potente: "
        "el signo del retorno es dificil de anticipar, pero la intensidad de los movimientos si parece agruparse en el tiempo."
    )

    section_title("6. Pruebas Formales: Raiz Unitaria y Ruido Blanco")
    info_banner(
        "Aqui hacemos un chequeo estadistico mas directo. ADF prueba como hipotesis nula que la serie tiene raiz unitaria. "
        "KPSS prueba lo contrario: como hipotesis nula toma que la serie es estacionaria. Ljung-Box no prueba raiz unitaria, "
        "pero si ayuda a ver si los retornos se parecen a ruido blanco en media."
    )
    stationarity_tests = summarize_stationarity_tests(
        tuple(fx_data["log_fx"].to_numpy()),
        tuple(fx_data["fx_return"].to_numpy()),
    )
    st.dataframe(
        format_table_for_display(
            stationarity_tests,
            {"Estadistico": "{:.4f}", "p_valor": "{:.4f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )
    info_banner(
        "La lectura esperada en tipos de cambio suele ser esta: el logaritmo del nivel no rechaza raiz unitaria o incluso rechaza estacionariedad, "
        "mientras que los retornos si se ven mucho mas compatibles con estacionariedad. Eso no significa que la serie no sea modelable; "
        "significa que no conviene modelar el nivel con herramientas que asumen estacionariedad en media."
    )
    info_banner(
        "Si ademas Ljung-Box no rechaza ruido blanco en retornos, la conclusion tipica es que la media diaria es dificil de predecir. "
        "En cambio, si |r|, r^2 y el test ARCH si muestran dependencia, la parte modelable no esta tanto en la media sino en la volatilidad."
    )

    section_title("7. Validar el Efecto ARCH")
    info_banner(
        "El efecto ARCH significa que la varianza condicional depende del pasado. "
        "Una manera simple de verlo es estudiar los retornos al cuadrado: si ellos muestran autocorrelacion, entonces la volatilidad no es constante."
    )
    st.plotly_chart(make_squared_returns_chart(fx_data), use_container_width=True)
    st.plotly_chart(
        make_acf_pacf_figure(fx_data["fx_return_sq"].to_numpy(), max_lag=24, title_prefix="retornos al cuadrado"),
        use_container_width=True,
    )
    arch_cols = st.columns(3)
    arch_cols[0].metric("p-valor Ljung-Box |r|", f"{fx_diag['lb_abs_pvalue']:.4f}")
    arch_cols[1].metric("p-valor Ljung-Box r^2", f"{fx_diag['lb_sq_pvalue']:.4f}")
    arch_cols[2].metric("p-valor ARCH test", f"{fx_diag['arch_pvalue']:.4f}")
    info_banner(
        "Si estos p-valores son bajos, rechazamos la idea de volatilidad constante. "
        "Eso abre la puerta para usar un modelo ARCH o GARCH en lugar de tratar la varianza como fija."
    )

    section_title("8. Comparar un Modelo ARCH con un Modelo GARCH")
    info_banner(
        "Un modelo ARCH explica la volatilidad actual con shocks recientes. "
        "Un modelo GARCH agrega otra idea: la propia volatilidad pasada tambien tiene memoria. "
        "Eso permite modelos mas parsimoniosos y, muchas veces, mas realistas."
    )
    with st.spinner("Estimando modelos de volatilidad..."):
        arch_result = fit_volatility_model(
            tuple(fx_data["fx_return"].to_numpy()),
            family="ARCH",
            arch_order=fx_arch_order,
            dist=fx_dist,
        )
        garch_result = fit_volatility_model(
            tuple(fx_data["fx_return"].to_numpy()),
            family="GARCH",
            arch_order=fx_arch_order,
            dist=fx_dist,
        )

    comparison_df = pd.DataFrame(
        [
            {
                "Modelo": f"ARCH({fx_arch_order})",
                "AIC": arch_result["aic"],
                "BIC": arch_result["bic"],
                "LogLik": arch_result["loglik"],
                "p-valor LB residuos": arch_result["lb_resid_pvalue"],
                "p-valor LB residuos^2": arch_result["lb_sq_pvalue"],
            },
            {
                "Modelo": "GARCH(1,1)",
                "AIC": garch_result["aic"],
                "BIC": garch_result["bic"],
                "LogLik": garch_result["loglik"],
                "p-valor LB residuos": garch_result["lb_resid_pvalue"],
                "p-valor LB residuos^2": garch_result["lb_sq_pvalue"],
            },
        ]
    )
    st.dataframe(
        format_table_for_display(
            comparison_df,
            {
                "AIC": "{:.2f}",
                "BIC": "{:.2f}",
                "LogLik": "{:.2f}",
                "p-valor LB residuos": "{:.4f}",
                "p-valor LB residuos^2": "{:.4f}",
            },
        ),
        use_container_width=True,
        hide_index=True,
    )

    chosen_family = "ARCH" if arch_result["aic"] <= garch_result["aic"] else "GARCH"
    chosen_result = arch_result if chosen_family == "ARCH" else garch_result
    chosen_label = f"ARCH({fx_arch_order})" if chosen_family == "ARCH" else "GARCH(1,1)"
    info_banner(
        f"Modelo recomendado en esta corrida: {chosen_label}. "
        "La decision combina parsimonia, ajuste y limpieza de residuos estandarizados."
    )

    section_title("9. Entender el Modelo de Volatilidad")
    info_banner(
        f"En estos modelos no estamos explicando el retorno promedio de {FX_PAIR_LABEL}, sino la varianza condicional. "
        "Por eso los coeficientes hablan de persistencia del riesgo, no de direccion del tipo de cambio."
    )
    st.dataframe(
        format_table_for_display(
            pd.DataFrame(
                {
                    "Parametro": chosen_result["model"].params.index,
                    "Valor": chosen_result["model"].params.values,
                }
            ),
            {"Valor": "{:.4f}"},
        ),
        use_container_width=True,
        hide_index=True,
    )
    if chosen_family == "ARCH":
        info_banner(
            "Lectura economica: un ARCH dice que la volatilidad de hoy responde principalmente a shocks recientes. "
            "Cuando alguno de esos coeficientes es grande, un movimiento fuerte tiende a dejar una estela de riesgo en los dias siguientes."
        )
    else:
        info_banner(
            "Lectura economica: un GARCH(1,1) dice que la volatilidad depende tanto de los shocks recientes como de la propia volatilidad pasada. "
            "Eso captura muy bien la idea de que los periodos turbulentos suelen tardar en disiparse."
        )

    section_title("10. Verificar los Residuos Estandarizados")
    info_banner(
        "Despues de ajustar un modelo de volatilidad, revisamos si los residuos estandarizados se parecen mas a ruido blanco. "
        "Tambien miramos sus cuadrados para ver si queda volatilidad sin capturar."
    )
    st.plotly_chart(
        make_arch_diagnostic_chart(fx_data["fecha"], fx_data["fx_return"], chosen_result["result_df"]),
        use_container_width=True,
    )
    diag_model_cols = st.columns(2)
    diag_model_cols[0].metric("p-valor LB residuos std", f"{chosen_result['lb_resid_pvalue']:.4f}")
    diag_model_cols[1].metric("p-valor LB residuos std^2", f"{chosen_result['lb_sq_pvalue']:.4f}")
    info_banner(
        "Si el p-valor sobre residuos estandarizados^2 sube frente al analisis inicial, eso sugiere que el modelo absorbio una parte importante del clustering de volatilidad."
    )

    section_title("11. Pronostico de Volatilidad")
    info_banner(
        f"El forecast final ya no habla de la direccion puntual de {FX_PAIR_LABEL}, sino del tamano esperado de sus movimientos. "
        "Para que esto sea interpretable, conviene verlo con historia reciente, en la escala de variaciones y tambien en la escala del tipo de cambio."
    )
    vol_forecast = build_volatility_forecast(
        tuple(fx_data["fecha"].dt.strftime("%Y-%m-%d").to_numpy()),
        tuple(fx_data["fx_return"].to_numpy()),
        tuple(fx_data["fx"].to_numpy()),
        tuple(fx_data["log_fx"].to_numpy()),
        family=chosen_family,
        arch_order=fx_arch_order,
        dist=fx_dist,
        horizon=fx_horizon,
    )
    st.plotly_chart(
        make_volatility_forecast_chart(
            fx_data["fecha"],
            chosen_result["result_df"],
            vol_forecast,
        ),
        use_container_width=True,
    )
    forecast_cols = st.columns(2)
    forecast_cols[0].plotly_chart(
        make_return_band_chart(fx_data, vol_forecast),
        use_container_width=True,
    )
    forecast_cols[1].plotly_chart(
        make_fx_business_forecast_chart(fx_data, vol_forecast),
        use_container_width=True,
    )
    last_fx = float(fx_data["fx"].iloc[-1])
    first_step = vol_forecast.iloc[0]
    info_banner(
        f"Lectura rapida del primer horizonte: con un ultimo nivel observado de {last_fx:.4f} {FX_PRICE_LABEL}, "
        f"el modelo espera una volatilidad diaria de {first_step['volatilidad_pct']:.3f}%. "
        f"Eso se traduce en una banda aproximada de {first_step['fx_inferior']:.4f} a {first_step['fx_superior']:.4f} {FX_PRICE_LABEL} al 95%."
    )
    st.dataframe(
        format_table_for_display(
            vol_forecast[
                [
                    "fecha",
                    "horizonte",
                    "volatilidad_pct",
                    "retorno_esperado_pct",
                    "banda_inferior_pct",
                    "banda_superior_pct",
                    "fx_esperado",
                    "fx_inferior",
                    "fx_superior",
                ]
            ],
            {
                "volatilidad_pct": "{:.3f}",
                "retorno_esperado_pct": "{:.3f}",
                "banda_inferior_pct": "{:.3f}",
                "banda_superior_pct": "{:.3f}",
                "fx_esperado": "{:.4f}",
                "fx_inferior": "{:.4f}",
                "fx_superior": "{:.4f}",
            },
        ),
        use_container_width=True,
        hide_index=True,
    )

    section_title("12. Cierre del Modulo")
    info_banner(
        f"En esta muestra, el modelo recomendado fue {chosen_label}. "
        f"La conclusion clave es que {FX_PAIR_LABEL} puede no ser facil de predecir en direccion, "
        "pero su volatilidad si suele mostrar memoria. Ese es el espacio natural para ARCH y GARCH."
    )
