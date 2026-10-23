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
from scipy.stats import chi2, norm


st.set_page_config(
    page_title="Risk Lab EAFIT",
    page_icon="📉",
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
        .concept-card {
            background: #F8FAFC;
            border: 1px solid #D9E2EC;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 12px;
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


VAR_SERIES = {
    "oil": {"fred_id": "WCOILWTICO", "label": "Petroleo WTI", "color": "#1D3557"},
    "gasoline": {"fred_id": "GASREGW", "label": "Gasolina regular", "color": "#D62828"},
    "energy_cpi": {"fred_id": "T5YIE", "label": "Inflacion esperada 5Y", "color": "#2A9D8F"},
}

VAR_LABELS = {
    "oil": "Petroleo WTI",
    "gasoline": "Gasolina regular",
    "energy_cpi": "Inflacion esperada 5Y",
}

VAR_CHART_COLORS = {
    "oil": "#1D3557",
    "gasoline": "#D62828",
    "energy_cpi": "#2A9D8F",
}

VAR_PORTFOLIOS = {
    "S&P 500": {"tickers": ["^GSPC"], "weights": [1.0]},
    "WTI": {"tickers": ["CL=F"], "weights": [1.0]},
    "Portafolio 60/40": {"tickers": ["SPY", "IEF"], "weights": [0.6, 0.4]},
    "Portafolio personalizado": {"tickers": [], "weights": []},
}

MODEL_NAMES = {
    "historical": "Simulacion historica",
    "varcov": "Var-Cov normal",
    "ewma": "EWMA",
}


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def info_banner(text: str) -> None:
    st.markdown(f'<div class="info-banner">{text}</div>', unsafe_allow_html=True)


def concept_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="concept-card"><strong>{title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )


def diff_label(key: str, time_suffix: str = "t") -> str:
    return f"Δln({VAR_LABELS[key]}) [{time_suffix}]"


@st.cache_resource(show_spinner=False)
def get_statsmodels_modules():
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests

    return {
        "sm": sm,
        "VAR": VAR,
        "adfuller": adfuller,
        "acorr_ljungbox": acorr_ljungbox,
        "grangercausalitytests": grangercausalitytests,
    }


@st.cache_resource(show_spinner=False)
def get_arch_model():
    from arch import arch_model

    return arch_model


@st.cache_resource(show_spinner=False)
def get_yfinance_module():
    import yfinance as yf

    return yf


@st.cache_data(show_spinner=False, ttl=3600)
def load_fred_series(series_id: str, start_date: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["fecha", "valor"]
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna()
    return df[df["fecha"] >= pd.to_datetime(start_date)].copy()


@st.cache_data(show_spinner=False, ttl=3600)
def load_var_dataset(start_date: str) -> pd.DataFrame:
    gas = load_fred_series(VAR_SERIES["gasoline"]["fred_id"], start_date)
    oil = load_fred_series(VAR_SERIES["oil"]["fred_id"], start_date)
    third = load_fred_series(VAR_SERIES["energy_cpi"]["fred_id"], start_date)

    gas_w = gas.set_index("fecha").resample("W-FRI").last().rename(columns={"valor": "gasoline"})
    oil_w = oil.set_index("fecha").resample("W-FRI").last().rename(columns={"valor": "oil"})

    # La ventana base debe ser identica a la del modulo dinamico: el cruce semanal de gasolina y petroleo.
    base = gas_w.join(oil_w, how="inner").dropna()

    third_w = third.set_index("fecha").resample("W-FRI").last().rename(columns={"valor": "energy_cpi"})
    # Alineamos la tercera serie a la base semanal de gasolina-petroleo para que no recorte el tramo final.
    third_w = third_w.reindex(base.index).ffill()

    df = base.join(third_w, how="left").dropna().reset_index()

    for key in VAR_SERIES:
        df[f"log_{key}"] = np.log(df[key])
        df[f"dlog_{key}"] = df[f"log_{key}"].diff() * 100

    return df.dropna().reset_index(drop=True)


def build_var_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"fecha": df["fecha"]})
    for key in VAR_SERIES:
        result[key] = df[f"dlog_{key}"]
    return result.dropna().reset_index(drop=True)


def make_var_level_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    ordered_keys = ["oil", "gasoline", "energy_cpi"]
    y_titles = ["USD por barril", "USD por galon", "Porcentaje"]

    for idx, key in enumerate(ordered_keys, start=1):
        fig.add_trace(
            go.Scatter(
                x=df["fecha"],
                y=df[key],
                mode="lines",
                name=VAR_LABELS[key],
                line=dict(color=VAR_CHART_COLORS[key], width=2.3),
            ),
            row=idx,
            col=1,
        )
        fig.update_yaxes(title_text=y_titles[idx - 1], row=idx, col=1)

    fig.update_xaxes(title_text="Fecha", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=760,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Series en niveles", font=dict(size=20)),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def make_var_return_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key in ["oil", "gasoline", "energy_cpi"]:
        fig.add_trace(
            go.Scatter(
                x=df["fecha"],
                y=df[key],
                mode="lines",
                name=VAR_LABELS[key],
                line=dict(color=VAR_CHART_COLORS[key], width=2.0),
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Series estacionarias para el VAR", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Variacion logaritmica semanal (%)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    corr = df.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=[VAR_LABELS.get(c, c) for c in corr.columns],
            y=[VAR_LABELS.get(c, c) for c in corr.index],
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 3),
            texttemplate="%{text}",
            hovertemplate="Fila: %{y}<br>Columna: %{x}<br>Correlacion: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text=title, font=dict(size=18)),
    )
    return fig


def positional_correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_values[mask], y_values[mask])[0, 1])


def prepare_lagged_pair_dataframe(
    data: pd.DataFrame,
    predictor: str,
    response: str,
    lag: int,
) -> pd.DataFrame:
    if lag == 0:
        pair_df = pd.DataFrame(
            {
                "x": data[predictor].to_numpy(),
                "y": data[response].to_numpy(),
            },
            index=data.index,
        )
    else:
        pair_df = pd.DataFrame(
            {
                "x": data[predictor].iloc[:-lag].to_numpy(),
                "y": data[response].iloc[lag:].to_numpy(),
            }
        )
    return pair_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def compute_cross_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    lags = np.arange(-max_lag, max_lag + 1)
    values = []
    for lag in lags:
        if lag < 0:
            corr = positional_correlation(x.iloc[:lag].to_numpy(), y.iloc[-lag:].to_numpy())
        elif lag > 0:
            corr = positional_correlation(x.iloc[lag:].to_numpy(), y.iloc[:-lag].to_numpy())
        else:
            corr = positional_correlation(x.to_numpy(), y.to_numpy())
        values.append(corr)
    return pd.DataFrame({"lag": lags, "corr": values})


def make_cross_corr_chart(ccf_df: pd.DataFrame, x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ccf_df["lag"],
            y=ccf_df["corr"],
            marker_color="#457B9D",
            hovertemplate="Rezago: %{x}<br>Correlacion: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#7A7A7A", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(
            text=f"Correlacion cruzada entre {x_label} y {y_label}",
            font=dict(size=18),
        ),
        xaxis=dict(title="Rezago de la primera serie"),
        yaxis=dict(title="Correlacion"),
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def compute_lagged_correlation_matrices(data: pd.DataFrame, max_lag: int) -> dict[int, pd.DataFrame]:
    matrices = {}
    columns = list(data.columns)

    for lag in range(0, max_lag + 1):
        matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
        for response in columns:
            for predictor in columns:
                if lag == 0:
                    value = positional_correlation(data[response].to_numpy(), data[predictor].to_numpy())
                else:
                    value = positional_correlation(
                        data[response].iloc[lag:].to_numpy(),
                        data[predictor].iloc[:-lag].to_numpy(),
                    )
                matrix.loc[response, predictor] = value
        matrices[lag] = matrix
    return matrices


def make_lagged_heatmap(matrix: pd.DataFrame, lag: int, title_prefix: str) -> go.Figure:
    x_suffix = "t" if lag == 0 else f"t-{lag}"
    y_labels = [diff_label(idx, "t") for idx in matrix.index]
    x_labels = [diff_label(col, x_suffix) for col in matrix.columns]
    subtitle = "t con t" if lag == 0 else f"t con t-{lag}"
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(matrix.values, 3),
            texttemplate="%{text}",
            hovertemplate="Serie objetivo: %{y}<br>Serie rezagada: %{x}<br>Correlacion: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text=f"{title_prefix} ({subtitle})", font=dict(size=18)),
    )
    return fig


@st.cache_data(show_spinner=False)
def summarize_top_lagged_links(data: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    matrices = compute_lagged_correlation_matrices(data, max_lag)
    for lag, matrix in matrices.items():
        for response in matrix.index:
            for predictor in matrix.columns:
                if response == predictor:
                    continue
                value = float(matrix.loc[response, predictor])
                rows.append(
                    {
                        "Rezago": lag,
                        "Serie objetivo": VAR_LABELS.get(response, response),
                        "Serie rezagada": VAR_LABELS.get(predictor, predictor),
                        "Correlacion": value,
                        "Correlacion abs": abs(value),
                    }
                )
    summary = pd.DataFrame(rows).sort_values("Correlacion abs", ascending=False).reset_index(drop=True)
    return summary


def make_pair_lag_scatter(data: pd.DataFrame, predictor: str, response: str, lag: int) -> go.Figure:
    aligned = prepare_lagged_pair_dataframe(data, predictor, response, lag)
    if lag == 0:
        x_label = diff_label(predictor, "t")
    else:
        x_label = diff_label(predictor, f"t-{lag}")

    corr = positional_correlation(aligned["x"].to_numpy(), aligned["y"].to_numpy())
    x_series = data[predictor].dropna()
    y_series = data[response].dropna()

    def padded_range(series: pd.Series) -> list[float]:
        min_val = float(series.min())
        max_val = float(series.max())
        span = max_val - min_val
        pad = 0.08 * span if span > 0 else 0.5
        return [min_val - pad, max_val + pad]

    x_range = padded_range(x_series)
    y_range = padded_range(y_series)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=aligned["x"],
            y=aligned["y"],
            mode="markers",
            marker=dict(color="#457B9D", size=7, opacity=0.75),
            name="Observaciones",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=390,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(
            text=(
                f"Relacion entre {x_label} y {diff_label(response, 't')}"
                f" | corr = {corr:.3f}"
            ),
            font=dict(size=18),
        ),
        xaxis=dict(title=x_label, range=x_range),
        yaxis=dict(title=diff_label(response, "t"), range=y_range),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def interpret_pair_relationship(predictor: str, response: str, lag: int, corr: float) -> str:
    sign_text = "positiva" if corr >= 0 else "negativa"
    strength = abs(corr)
    if strength >= 0.50:
        intensity = "fuerte"
    elif strength >= 0.25:
        intensity = "moderada"
    else:
        intensity = "debil"

    if lag == 0:
        return (
            f"Aqui la lectura es contemporanea: comparamos {VAR_LABELS[predictor]} y {VAR_LABELS[response]} en el mismo periodo. "
            f"La asociacion es {sign_text} y {intensity} (corr = {corr:.3f}). "
            "Si inviertes los ejes, la nube cambia de orientacion visual pero la relacion estadistica es la misma, porque sigue siendo la misma comparacion en t."
        )

    return (
        f"Aqui la lectura ya no es contemporanea sino direccional: comparamos {VAR_LABELS[predictor]} en t-{lag} contra {VAR_LABELS[response]} en t. "
        f"La asociacion es {sign_text} y {intensity} (corr = {corr:.3f}). "
        "Si intercambias predictor y objetivo, ya no estas invirtiendo ejes sobre la misma relacion; estas formulando otra pregunta econometrica distinta."
    )


def explain_correlation_vs_regression(corr_value: float) -> str:
    return (
        f"La correlacion contemporanea entre Δln(petroleo) y Δln(gasolina) en esta muestra es {corr_value:.3f}. "
        "Eso no tiene por que coincidir con el coeficiente de una regresion de gasolina sobre petroleo. "
        "La correlacion es una medida simetrica de comovimiento; la regresion, en cambio, depende de cual variable pones como respuesta y de la especificacion del modelo."
    )


@st.cache_data(show_spinner=False)
def fit_no_intercept_pair_regression(
    x_values: tuple[float, ...],
    y_values: tuple[float, ...],
) -> dict[str, float]:
    sm = get_statsmodels_modules()["sm"]
    pair_df = pd.DataFrame({"x": x_values, "y": y_values}).dropna()
    model = sm.OLS(pair_df["y"], pair_df[["x"]]).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    return {
        "beta": float(model.params["x"]),
        "pvalue": float(model.pvalues["x"]),
        "rsquared": float(model.rsquared),
    }


def make_scatter_matrix(data: pd.DataFrame) -> go.Figure:
    variables = list(data.columns)
    labels = {key: diff_label(key, "t") for key in variables}
    fig = make_subplots(
        rows=len(variables),
        cols=len(variables),
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )

    for row_idx, row_var in enumerate(variables, start=1):
        for col_idx, col_var in enumerate(variables, start=1):
            if row_idx == col_idx:
                fig.add_trace(
                    go.Histogram(
                        x=data[row_var],
                        nbinsx=20,
                        marker=dict(color="#A8DADC", line=dict(color="white", width=0.4)),
                        showlegend=False,
                        hovertemplate="Valor: %{x:.2f}<br>Frecuencia: %{y}<extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                continue

            x_var, y_var = sorted([row_var, col_var], key=lambda item: variables.index(item))
            fig.add_trace(
                go.Scatter(
                    x=data[x_var],
                    y=data[y_var],
                    mode="markers",
                    marker=dict(color="#457B9D", size=5, opacity=0.65),
                    showlegend=False,
                    hovertemplate=(
                        f"{labels[x_var]}: %{{x:.2f}}<br>"
                        f"{labels[y_var]}: %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

            x_min, x_max = float(data[x_var].min()), float(data[x_var].max())
            y_min, y_max = float(data[y_var].min()), float(data[y_var].max())
            x_pad = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
            y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.5
            fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad], row=row_idx, col=col_idx)
            fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad], row=row_idx, col=col_idx)

    for idx, var in enumerate(variables, start=1):
        fig.update_xaxes(title_text=labels[var], row=len(variables), col=idx)
        fig.update_yaxes(title_text=labels[var], row=idx, col=1)

    fig.update_layout(
        template="plotly_white",
        height=820,
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(
            text="Nube contemporanea sobre diferencias logaritmicas",
            font=dict(size=18),
        ),
        bargap=0.04,
    )
    return fig


def summarize_lagged_matrix(matrix: pd.DataFrame, lag: int) -> str:
    values = matrix.to_numpy(dtype=float)
    diagonal_mask = np.eye(values.shape[0], dtype=bool)
    off_diag = values[~diagonal_mask]
    mean_abs_off_diag = float(np.nanmean(np.abs(off_diag))) if off_diag.size else np.nan
    values_no_diag = values.copy()
    values_no_diag[diagonal_mask] = np.nan
    max_idx = np.unravel_index(np.nanargmax(np.abs(values_no_diag)), values.shape)
    strongest_value = float(values[max_idx])
    response = matrix.index[max_idx[0]]
    predictor = matrix.columns[max_idx[1]]

    timing = "en el mismo periodo" if lag == 0 else f"cuando la serie de la columna se rezaga {lag} periodo(s)"
    return (
        f"En esta matriz, la correlacion cruzada promedio fuera de la diagonal es {mean_abs_off_diag:.3f}. "
        f"La relacion mas fuerte aparece entre {VAR_LABELS.get(response, response)} y {VAR_LABELS.get(predictor, predictor)} "
        f"con un valor de {strongest_value:.3f}, {timing}."
    )


def interpret_top_lagged_links(top_links_df: pd.DataFrame) -> str:
    if top_links_df.empty:
        return "No aparecen relaciones rezagadas suficientemente claras en la muestra seleccionada."

    first = top_links_df.iloc[0]
    lag = int(first["Rezago"])
    corr = float(first["Correlacion"])
    sign_text = "positiva" if corr >= 0 else "negativa"
    if lag == 0:
        timing = "en el mismo periodo"
    else:
        timing = f"cuando la serie predictora va rezagada {lag} periodo(s)"

    return (
        f"La relacion rezagada mas intensa del sistema es entre {first['Serie rezagada']} y {first['Serie objetivo']}, "
        f"con una correlacion {sign_text} de {corr:.3f} {timing}. "
        "Eso no prueba causalidad por si solo, pero si identifica una direccion de transmision que merece prueba formal."
    )


def interpret_lag_selection(lag_df: pd.DataFrame) -> str:
    clean = lag_df.dropna().copy()
    if clean.empty:
        return "No fue posible construir una recomendacion clara de rezagos con la muestra disponible."

    aic_choice = int(clean.loc[clean["AIC"].idxmin(), "Rezago"])
    bic_choice = int(clean.loc[clean["BIC"].idxmin(), "Rezago"])
    hqic_choice = int(clean.loc[clean["HQIC"].idxmin(), "Rezago"])

    if aic_choice == bic_choice == hqic_choice:
        return (
            f"AIC, BIC y HQIC coinciden en VAR({aic_choice}). Eso sugiere una decision bastante estable: "
            "el sistema parece tener una profundidad dinamica clara y no hay mucho conflicto entre ajuste y parsimonia."
        )

    return (
        f"AIC prefiere VAR({aic_choice}), BIC prefiere VAR({bic_choice}) y HQIC prefiere VAR({hqic_choice}). "
        "Cuando esto ocurre, la lectura usual es que AIC esta dispuesto a pagar con mas parametros a cambio de mejor ajuste, "
        "mientras BIC castiga con mas fuerza la complejidad. Para una app docente conviene usar BIC como referencia parsimoniosa y luego revisar si los resultados economicos son estables."
    )


def criteria_choices(lag_df: pd.DataFrame) -> dict[str, int]:
    clean = lag_df.dropna().copy()
    if clean.empty:
        return {}
    return {
        "AIC": int(clean.loc[clean["AIC"].idxmin(), "Rezago"]),
        "BIC": int(clean.loc[clean["BIC"].idxmin(), "Rezago"]),
        "HQIC": int(clean.loc[clean["HQIC"].idxmin(), "Rezago"]),
    }


@st.cache_data(show_spinner=False)
def run_adf_summary(data: pd.DataFrame) -> pd.DataFrame:
    adfuller = get_statsmodels_modules()["adfuller"]
    rows = []
    for column in data.columns:
        stat, pvalue, *_ = adfuller(data[column].dropna(), autolag="AIC")
        rows.append(
            {
                "Serie": VAR_LABELS.get(column, column),
                "ADF estadistico": stat,
                "ADF p-valor": pvalue,
                "Conclusion": "Estacionaria" if pvalue < 0.05 else "No estacionaria",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def select_var_order(data: pd.DataFrame, maxlags: int) -> pd.DataFrame:
    VAR = get_statsmodels_modules()["VAR"]
    model = VAR(data)
    selected = model.select_order(maxlags=maxlags)

    def read_ic(metric_name: str, lag: int):
        metric_values = selected.ics[metric_name]
        if hasattr(metric_values, "get"):
            return metric_values.get(lag)
        if isinstance(metric_values, (list, tuple)):
            if 0 <= lag < len(metric_values):
                return metric_values[lag]
            return np.nan
        return np.nan

    rows = []
    for lag in range(1, maxlags + 1):
        rows.append(
            {
                "Rezago": lag,
                "AIC": read_ic("aic", lag),
                "BIC": read_ic("bic", lag),
                "HQIC": read_ic("hqic", lag),
                "FPE": read_ic("fpe", lag),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fit_var_results(data: pd.DataFrame, lags: int):
    VAR = get_statsmodels_modules()["VAR"]
    return VAR(data).fit(lags)


@st.cache_data(show_spinner=False)
def compute_residual_diagnostics(data: pd.DataFrame, lags: int) -> pd.DataFrame:
    acorr_ljungbox = get_statsmodels_modules()["acorr_ljungbox"]
    results = fit_var_results(data, lags)
    rows = []
    for column in results.resid.columns:
        lb = acorr_ljungbox(results.resid[column], lags=[12], return_df=True)
        rows.append(
            {
                "Ecuacion": VAR_LABELS.get(column, column),
                "Ljung-Box Q(12)": float(lb["lb_stat"].iloc[0]),
                "p-valor": float(lb["lb_pvalue"].iloc[0]),
                "Lectura": "Residuos cercanos a ruido blanco" if float(lb["lb_pvalue"].iloc[0]) >= 0.05 else "Todavia queda dependencia",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_granger_table(data: pd.DataFrame, maxlag: int) -> pd.DataFrame:
    granger = get_statsmodels_modules()["grangercausalitytests"]
    rows = []
    columns = list(data.columns)

    for caused in columns:
        for causing in columns:
            if caused == causing:
                continue
            pair = data[[caused, causing]].dropna()
            test = granger(pair, maxlag=int(maxlag))
            pvalues = [float(test[lag][0]["ssr_ftest"][1]) for lag in range(1, maxlag + 1)]
            best_p = min(pvalues)
            best_lag = int(np.argmin(pvalues) + 1)
            rows.append(
                {
                    "Objetivo key": caused,
                    "Predictora key": causing,
                    "Serie objetivo": VAR_LABELS.get(caused, caused),
                    "Serie que ayuda a predecir": VAR_LABELS.get(causing, causing),
                    "Mejor rezago": best_lag,
                    "p-valor minimo": best_p,
                    "Conclusion": "Hay evidencia de causalidad de Granger" if best_p < 0.05 else "No hay evidencia fuerte",
                }
            )
    return pd.DataFrame(rows)


def make_granger_heatmap(granger_df: pd.DataFrame) -> go.Figure:
    labels = list(VAR_LABELS.values())
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, row in granger_df.iterrows():
        matrix.loc[row["Serie objetivo"], row["Serie que ayuda a predecir"]] = row["p-valor minimo"]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="YlOrRd_r",
            zmin=0,
            zmax=0.10,
            text=np.round(matrix.values, 4),
            texttemplate="%{text}",
            hovertemplate="Objetivo: %{y}<br>Predictora: %{x}<br>p-valor: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text="Mapa de p-valores de causalidad de Granger", font=dict(size=18)),
    )
    return fig


def interpret_granger_pair(granger_df: pd.DataFrame, predictor: str, response: str) -> str:
    row = granger_df.loc[
        (granger_df["Objetivo key"] == response) & (granger_df["Predictora key"] == predictor)
    ]
    if row.empty:
        return "No fue posible construir la lectura de Granger para este par."

    row = row.iloc[0]
    pvalue = float(row["p-valor minimo"])
    lag = int(row["Mejor rezago"])

    if pvalue < 0.01:
        strength = "evidencia muy fuerte"
    elif pvalue < 0.05:
        strength = "evidencia estadistica razonable"
    elif pvalue < 0.10:
        strength = "evidencia debil"
    else:
        strength = "no hay evidencia estadistica suficiente"

    if pvalue < 0.05:
        return (
            f"Para la pregunta '{VAR_LABELS[predictor]} ayuda a pronosticar {VAR_LABELS[response]}', hay {strength}. "
            f"El mejor resultado aparece con rezago {lag} y p-valor {pvalue:.4f}. "
            "La lectura correcta es predictiva: la historia de la serie explicativa agrega informacion util sobre la serie objetivo."
        )

    return (
        f"Para la pregunta '{VAR_LABELS[predictor]} ayuda a pronosticar {VAR_LABELS[response]}', {strength}. "
        f"El mejor rezago fue {lag}, pero el p-valor quedo en {pvalue:.4f}. "
        "Eso no demuestra ausencia de relacion economica; solo indica que, con esta especificacion lineal y esta muestra, la ganancia predictiva no es convincente."
    )


def summarize_granger_system(granger_df: pd.DataFrame) -> str:
    significant = granger_df.loc[granger_df["p-valor minimo"] < 0.05].copy()
    if significant.empty:
        return (
            "En conjunto, el mapa de Granger no encuentra relaciones predictivas lineales suficientemente fuertes al 5%. "
            "Eso sugiere que el sistema puede estar dominado por comovimiento contemporaneo, por dinamicas mas complejas o por senales debiles en esta muestra."
        )

    significant = significant.sort_values("p-valor minimo")
    top = significant.iloc[0]
    count = len(significant)
    return (
        f"En conjunto, aparecen {count} relaciones con evidencia al 5%. "
        f"La mas clara es {top['Serie que ayuda a predecir']} -> {top['Serie objetivo']}, con p-valor {top['p-valor minimo']:.4f}. "
        "La idea no es leer esto como causalidad estructural, sino como un mapa de direcciones donde la informacion pasada parece viajar con mas fuerza."
    )


def format_var_param_name(param_name: str) -> str:
    if param_name == "const":
        return "Constante"
    if ".L" in param_name:
        var_name, lag_text = param_name.split(".L")
        return f"{diff_label(var_name, f't-{lag_text}')}"
    return param_name


def build_var_equation_table(results) -> pd.DataFrame:
    params = results.params.copy()
    rows = []
    for equation_name in params.columns:
        for param_name, coef in params[equation_name].items():
            rows.append(
                {
                    "Ecuacion": VAR_LABELS.get(equation_name, equation_name),
                    "Termino": format_var_param_name(param_name),
                    "Coeficiente": float(coef),
                    "Abs coef": abs(float(coef)),
                }
            )
    return pd.DataFrame(rows)


def build_var_equation_texts(results) -> list[dict[str, str]]:
    params = results.params.copy()
    outputs = []
    for equation_name in params.columns:
        pieces = []
        for param_name, coef in params[equation_name].items():
            if param_name == "const":
                continue
            pieces.append(f"{coef:.3f}·{format_var_param_name(param_name)}")
        rhs = " + ".join(pieces[:8]) if pieces else "solo constante"
        outputs.append(
            {
                "equation_name": equation_name,
                "label": VAR_LABELS.get(equation_name, equation_name),
                "equation_text": f"{diff_label(equation_name, 't')} = {rhs}",
            }
        )
    return outputs


def interpret_var_equation(results, target_variable: str) -> str:
    equation = results.params[target_variable]
    non_const = equation.drop(labels=["const"], errors="ignore")
    if non_const.empty:
        return "La ecuacion esta dominada por la constante y casi no muestra respuesta a rezagos relevantes."

    strongest = non_const.abs().sort_values(ascending=False).head(3)
    pieces = []
    for param_name in strongest.index:
        coef = float(non_const[param_name])
        sign = "positivo" if coef >= 0 else "negativo"
        pieces.append(f"{format_var_param_name(param_name)} ({sign}, {coef:.3f})")

    return (
        f"En la ecuacion de {VAR_LABELS[target_variable]}, los terminos mas influyentes son {', '.join(pieces)}. "
        "Eso resume que parte de la dinamica viene de memoria propia y que parte viene de rezagos de las otras variables del sistema."
    )


def residual_guidance(residual_df: pd.DataFrame) -> str:
    problematic = residual_df.loc[residual_df["p-valor"] < 0.05]
    if problematic.empty:
        return (
            "No vemos evidencia fuerte de autocorrelacion remanente en las ecuaciones al rezago de prueba. "
            "Eso apoya la idea de que el VAR capturo una parte razonable de la dinamica lineal del sistema."
        )

    eq_names = ", ".join(problematic["Ecuacion"].tolist())
    return (
        f"Si los residuos de {eq_names} todavia muestran estructura, el mensaje no es 'el modelo no sirve', sino 'el sistema aun no absorbio toda la dinamica'. "
        "Los siguientes pasos tipicos son: probar mas rezagos, revisar cambios estructurales, considerar transformaciones adicionales o enriquecer el sistema con variables relevantes omitidas."
    )


def make_irf_chart(results, impulse: str, response: str, horizon: int) -> go.Figure:
    irf = results.irf(horizon)
    impulse_idx = list(results.names).index(impulse)
    response_idx = list(results.names).index(response)
    values = irf.irfs[:, response_idx, impulse_idx]
    stderr = irf.stderr(orth=False)[:, response_idx, impulse_idx]
    steps = np.arange(len(values))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=values,
            mode="lines+markers",
            name="IRF",
            line=dict(color="#1D3557", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([steps, steps[::-1]]),
            y=np.concatenate([values + 1.96 * stderr, (values - 1.96 * stderr)[::-1]]),
            fill="toself",
            fillcolor="rgba(69,123,157,0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="IC 95%",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(
            text=f"Impulso-respuesta: shock en {VAR_LABELS[impulse]} sobre {VAR_LABELS[response]}",
            font=dict(size=18),
        ),
        xaxis=dict(title="Horizonte"),
        yaxis=dict(title="Respuesta"),
    )
    return fig


def interpret_irf(results, impulse: str, response: str, horizon: int) -> str:
    irf = results.irf(horizon)
    impulse_idx = list(results.names).index(impulse)
    response_idx = list(results.names).index(response)
    values = irf.irfs[:, response_idx, impulse_idx]

    peak_idx = int(np.argmax(np.abs(values)))
    peak_value = float(values[peak_idx])
    sign_text = "positiva" if peak_value >= 0 else "negativa"

    decay_idx = None
    for idx in range(peak_idx + 1, len(values)):
        if abs(values[idx]) <= 0.1 * abs(peak_value):
            decay_idx = idx
            break

    if decay_idx is None:
        persistence_text = "la respuesta sigue siendo material durante casi todo el horizonte analizado"
    else:
        persistence_text = f"la respuesta se disipa de forma importante hacia el horizonte {decay_idx}"

    if peak_idx == 0:
        timing_text = "el mayor efecto aparece de inmediato"
    else:
        timing_text = f"el mayor efecto aparece con retraso, alrededor del horizonte {peak_idx}"

    return (
        f"Un shock en {VAR_LABELS[impulse]} genera una respuesta {sign_text} en {VAR_LABELS[response]}. "
        f"El impacto maximo es {peak_value:.3f} y {timing_text}; despues, {persistence_text}. "
        f"Aqui ese {peak_value:.3f} debe leerse en la escala del modelo: es un cambio adicional en {diff_label(response, 't')} provocado por un shock unitario en {VAR_LABELS[impulse]}. "
        "En otras palabras, no es un nivel de precio ni una elasticidad directa, sino la magnitud de la respuesta dinamica de la variable en su version estacionaria. "
        "Esta lectura ayuda a pasar de la pregunta 'quien ayuda a predecir a quien' a la pregunta 'como se transmite economicamente un shock en el sistema'."
    )


def make_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame, column: str) -> go.Figure:
    lower_col = f"{column}_lower"
    upper_col = f"{column}_upper"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["fecha"],
            y=history_df[column],
            mode="lines",
            name="Historico",
            line=dict(color=VAR_CHART_COLORS[column], width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["fecha"],
            y=forecast_df[column],
            mode="lines+markers",
            name="Pronostico",
            line=dict(color="#E76F51", width=2.2, dash="dash"),
            marker=dict(size=7),
        )
    )
    if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df["fecha"], forecast_df["fecha"][::-1]]),
                y=pd.concat([forecast_df[upper_col], forecast_df[lower_col][::-1]]),
                fill="toself",
                fillcolor="rgba(231,111,81,0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="IC 95%",
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Pronostico VAR en niveles para {VAR_LABELS[column]}", font=dict(size=18)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Nivel de la serie"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


@st.cache_data(show_spinner=False)
def build_var_forecast(
    data: pd.DataFrame,
    raw_df: pd.DataFrame,
    lags: int,
    steps: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    results = fit_var_results(data, lags)
    point_forecast, lower_forecast, upper_forecast = results.forecast_interval(
        data.values[-lags:],
        steps=steps,
        alpha=alpha,
    )
    forecast_diff = pd.DataFrame(point_forecast, columns=data.columns)
    lower_diff = pd.DataFrame(lower_forecast, columns=data.columns)
    upper_diff = pd.DataFrame(upper_forecast, columns=data.columns)
    last_date = pd.to_datetime(data.index[-1]) if isinstance(data.index, pd.DatetimeIndex) else None
    if last_date is None:
        forecast_dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=steps, freq="W-FRI")
    else:
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=steps, freq="W-FRI")

    forecast_df = pd.DataFrame({"fecha": forecast_dates})

    for key in data.columns:
        last_level = float(raw_df[key].iloc[-1])
        last_log = float(np.log(last_level))

        point_log_path = last_log + forecast_diff[key].cumsum() / 100
        lower_log_path = last_log + lower_diff[key].cumsum() / 100
        upper_log_path = last_log + upper_diff[key].cumsum() / 100

        forecast_df[key] = np.exp(point_log_path)
        forecast_df[f"{key}_lower"] = np.exp(lower_log_path)
        forecast_df[f"{key}_upper"] = np.exp(upper_log_path)
        forecast_df[f"{key}_dlog"] = forecast_diff[key].to_numpy()
        forecast_df[f"{key}_dlog_lower"] = lower_diff[key].to_numpy()
        forecast_df[f"{key}_dlog_upper"] = upper_diff[key].to_numpy()

    return forecast_df


@st.cache_data(show_spinner=False, ttl=3600)
def load_market_prices(tickers: tuple[str, ...], start_date: str) -> pd.DataFrame:
    yf = get_yfinance_module()
    prices = yf.download(
        list(tickers),
        start=start_date,
        auto_adjust=True,
        progress=False,
    )
    if prices.empty:
        raise ValueError("No fue posible descargar precios para el activo seleccionado.")

    close = prices["Close"] if isinstance(prices.columns, pd.MultiIndex) else prices.rename(columns={"Close": tickers[0]})[tickers[0]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close.dropna(how="all").ffill().dropna()
    close.columns = list(tickers)
    return close


def build_portfolio_returns(price_df: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    returns = np.log(price_df / price_df.shift(1)).dropna()
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    out = pd.DataFrame({"fecha": portfolio_returns.index, "retorno": portfolio_returns.values})
    out["precio_indice"] = 100 * np.exp(out["retorno"].cumsum())
    return out.reset_index(drop=True)


def build_custom_portfolio_inputs() -> tuple[list[str], list[float], str]:
    ticker_cols = st.columns(3)
    tickers = []
    weights = []
    default_tickers = ["SPY", "IEF", "GLD"]
    default_weights = [0.5, 0.3, 0.2]

    for idx, col in enumerate(ticker_cols, start=1):
        with col:
            ticker = st.text_input(
                f"Ticker {idx}",
                value=default_tickers[idx - 1],
                key=f"custom_ticker_{idx}",
            ).strip().upper()
            weight = st.number_input(
                f"Peso {idx}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weights[idx - 1]),
                step=0.05,
                key=f"custom_weight_{idx}",
            )
            if ticker:
                tickers.append(ticker)
                weights.append(weight)

    total_weight = sum(weights)
    label = " / ".join(tickers) if tickers else "Portafolio personalizado"
    return tickers, weights, label


def make_price_return_chart(df: pd.DataFrame, label: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["precio_indice"],
            mode="lines",
            name="Indice base 100",
            line=dict(color="#1D3557", width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["fecha"],
            y=df["retorno"] * 100,
            mode="lines",
            name="Retorno diario",
            line=dict(color="#D62828", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="Base 100", row=1, col=1)
    fig.update_yaxes(title_text="Retorno (%)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=620,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"{label}: precios y retornos", font=dict(size=20)),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def compute_ewma_sigma(window_returns: np.ndarray, lam: float) -> float:
    sigma2 = np.var(window_returns[: min(len(window_returns), 30)], ddof=1)
    for value in window_returns[1:]:
        sigma2 = lam * sigma2 + (1 - lam) * value * value
    return float(np.sqrt(max(sigma2, 1e-12)))


@st.cache_data(show_spinner=False)
def run_var_models(returns: tuple[float, ...], window: int, alpha: float) -> pd.DataFrame:
    series = pd.Series(returns, dtype=float)
    dates = pd.RangeIndex(len(series))
    z_alpha = norm.ppf(alpha)
    lam = 0.94
    rows = []

    for t in range(window, len(series)):
        sample = series.iloc[t - window:t].values
        current_return = float(series.iloc[t])

        hist_var = -np.quantile(sample, alpha)
        mu = float(np.mean(sample))
        sigma = float(np.std(sample, ddof=1))
        varcov_var = -(mu + sigma * z_alpha)
        ewma_sigma = compute_ewma_sigma(sample, lam)
        ewma_var = -(0.0 + ewma_sigma * z_alpha)

        row = {
            "idx": int(dates[t]),
            "realized_return": current_return,
            "historical": hist_var,
            "varcov": varcov_var,
            "ewma": ewma_var,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def kupiec_test(exceptions: int, total: int, alpha: float) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    pi_hat = exceptions / total
    if pi_hat == 0:
        lr = -2 * total * np.log(1 - alpha)
        return float(lr), float(1 - chi2.cdf(lr, df=1))
    if pi_hat >= 1:
        return np.inf, 0.0
    lr = -2 * (
        (total - exceptions) * np.log((1 - alpha) / (1 - pi_hat))
        + exceptions * np.log(alpha / pi_hat)
    )
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def christoffersen_independence_test(hits: np.ndarray) -> tuple[float, float]:
    if len(hits) < 2:
        return np.nan, np.nan

    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(hits[:-1], hits[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def safe_term(count: int, prob: float) -> float:
        if count == 0:
            return 0.0
        prob = min(max(prob, 1e-10), 1 - 1e-10)
        return count * np.log(prob)

    ll_ind = (
        safe_term(n00 + n10, 1 - pi)
        + safe_term(n01 + n11, pi)
    )
    ll_markov = (
        safe_term(n00, 1 - pi0)
        + safe_term(n01, pi0)
        + safe_term(n10, 1 - pi1)
        + safe_term(n11, pi1)
    )
    lr = -2 * (ll_ind - ll_markov)
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def basel_zone(exceptions: int) -> str:
    if exceptions <= 4:
        return "Verde"
    if exceptions <= 9:
        return "Amarilla"
    return "Roja"


def summarize_backtest(var_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    model_columns = [col for col in ["historical", "varcov", "ewma"] if col in var_df.columns]
    for column in model_columns:
        clean = var_df[["realized_return", column]].dropna().copy()
        hits = (clean["realized_return"] < -clean[column]).astype(int)
        exceptions = int(hits.sum())
        total = int(len(clean))
        ratio = exceptions / max(total * alpha, 1)
        kupiec_lr, kupiec_p = kupiec_test(exceptions, total, alpha)
        ind_lr, ind_p = christoffersen_independence_test(hits.to_numpy())

        basel_text = "No aplica"
        if np.isclose(alpha, 0.01) and total >= 250:
            basel_text = basel_zone(int(hits.iloc[-250:].sum()))

        rows.append(
            {
                "Modelo": MODEL_NAMES[column],
                "Excepciones": exceptions,
                "Observaciones": total,
                "Tasa observada": exceptions / total if total else np.nan,
                "Breach ratio": ratio,
                "Kupiec LR": kupiec_lr,
                "Kupiec p-valor": kupiec_p,
                "Independencia LR": ind_lr,
                "Independencia p-valor": ind_p,
                "Semaforo Basilea": basel_text,
            }
        )
    return pd.DataFrame(rows)


def make_var_backtest_chart(var_df: pd.DataFrame, selected_models: list[str], asset_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=var_df["fecha"],
            y=var_df["realized_return"] * 100,
            mode="lines",
            name="Retorno realizado",
            line=dict(color="#1D3557", width=1.6),
        )
    )
    color_map = {
        "historical": "#D62828",
        "varcov": "#2A9D8F",
        "ewma": "#6A4C93",
    }
    for model in selected_models:
        fig.add_trace(
            go.Scatter(
                x=var_df["fecha"],
                y=-var_df[model] * 100,
                mode="lines",
                name=f"VaR {MODEL_NAMES[model]}",
                line=dict(color=color_map[model], width=2.0),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Backtesting VaR para {asset_label}", font=dict(size=20)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Retorno / umbral VaR (%)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    return fig


def make_exception_bar(summary_df: pd.DataFrame, alpha: float) -> go.Figure:
    expected = summary_df["Observaciones"] * alpha
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary_df["Modelo"],
            y=summary_df["Excepciones"],
            name="Excepciones observadas",
            marker_color="#457B9D",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary_df["Modelo"],
            y=expected,
            mode="markers+lines",
            name="Excepciones esperadas",
            marker=dict(color="#D62828", size=10),
            line=dict(color="#D62828", dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Excepciones observadas vs esperadas", font=dict(size=18)),
        yaxis=dict(title="Numero de excepciones"),
    )
    return fig


def make_single_var_exception_chart(var_df: pd.DataFrame, model_name: str, asset_label: str) -> go.Figure:
    clean = var_df[["fecha", "realized_return", model_name]].dropna().copy()
    exceptions = clean["realized_return"] < -clean[model_name]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean["fecha"],
            y=clean["realized_return"] * 100,
            mode="lines",
            name="Retorno realizado",
            line=dict(color="#1D3557", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=clean["fecha"],
            y=-clean[model_name] * 100,
            mode="lines",
            name=f"VaR {MODEL_NAMES[model_name]}",
            line=dict(color="#D62828", width=2.0),
        )
    )
    flagged = clean.loc[exceptions]
    fig.add_trace(
        go.Scatter(
            x=flagged["fecha"],
            y=flagged["realized_return"] * 100,
            mode="markers",
            name="Excepciones",
            marker=dict(color="#E76F51", size=8, symbol="x"),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"VaR y excepciones para {asset_label}", font=dict(size=18)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Retorno / umbral VaR (%)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def explain_var_method(model_name: str) -> str:
    texts = {
        "historical": "Usa directamente la experiencia pasada: el VaR sale del percentil de la ventana historica sin imponer una forma paramétrica.",
        "varcov": "Supone una distribucion aproximadamente normal y resume el riesgo con media y desviacion estandar de la ventana reciente.",
        "ewma": "Tambien es paramétrico, pero da mas peso a lo reciente para reaccionar mas rapido a cambios de volatilidad.",
    }
    return texts[model_name]


def estimate_portfolio_return_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df / price_df.shift(1)).dropna()


@st.cache_data(show_spinner=False)
def run_monte_carlo_portfolio(
    price_df: pd.DataFrame,
    weights: tuple[float, ...],
    horizon_days: int,
    n_sims: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    returns = estimate_portfolio_return_matrix(price_df)
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()
    weights_arr = np.array(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    rng = np.random.default_rng(seed)

    sims = rng.multivariate_normal(mu, cov, size=(n_sims, horizon_days))
    asset_cum_returns = sims.sum(axis=1)
    portfolio_log_returns = asset_cum_returns @ weights_arr
    portfolio_path_returns = np.cumsum(sims @ weights_arr, axis=1)

    initial_value = 1000.0
    final_values = initial_value * np.exp(portfolio_log_returns)
    path_values = initial_value * np.exp(portfolio_path_returns)
    pnl = final_values - initial_value

    return {
        "mu": mu,
        "cov": cov,
        "portfolio_log_returns": portfolio_log_returns,
        "final_values": final_values,
        "path_values": path_values,
        "pnl": pnl,
        "initial_value": initial_value,
    }


def make_monte_carlo_histogram(mc_result: dict, confidence: float, label: str) -> go.Figure:
    pnl = np.asarray(mc_result["pnl"])
    var_level = -np.quantile(pnl, 1 - confidence)
    es = -pnl[pnl <= np.quantile(pnl, 1 - confidence)].mean()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pnl,
            nbinsx=50,
            marker=dict(color="#A8DADC", line=dict(color="white", width=0.5)),
            name="P&L simulado",
        )
    )
    fig.add_vline(x=np.mean(pnl), line_color="#1D3557", line_width=2, line_dash="dash")
    fig.add_vline(x=-var_level, line_color="#D62828", line_width=2)
    fig.add_vline(x=-es, line_color="#E76F51", line_width=2, line_dash="dot")
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Distribucion simulada de P&L para {label}", font=dict(size=18)),
        xaxis=dict(title="P&L sobre base 100"),
        yaxis=dict(title="Frecuencia"),
        showlegend=False,
    )
    return fig


def make_monte_carlo_terminal_chart(mc_result: dict, label: str) -> go.Figure:
    final_values = np.asarray(mc_result["final_values"])
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=50,
            marker=dict(color="#457B9D", line=dict(color="white", width=0.5)),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Valor final simulado del portafolio: {label}", font=dict(size=18)),
        xaxis=dict(title="Valor final"),
        yaxis=dict(title="Frecuencia"),
        showlegend=False,
    )
    return fig


def make_monte_carlo_path_chart(mc_result: dict, label: str, max_paths: int = 60) -> go.Figure:
    path_values = np.asarray(mc_result["path_values"])
    n_paths, horizon = path_values.shape
    shown = min(max_paths, n_paths)
    fig = go.Figure()
    for i in range(shown):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, horizon + 1),
                y=path_values[i],
                mode="lines",
                line=dict(color="rgba(69,123,157,0.18)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    median_path = np.median(path_values, axis=0)
    p10 = np.quantile(path_values, 0.10, axis=0)
    p90 = np.quantile(path_values, 0.90, axis=0)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([np.arange(1, horizon + 1), np.arange(horizon, 0, -1)]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor="rgba(231,111,81,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Banda 10%-90%",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, horizon + 1),
            y=median_path,
            mode="lines",
            line=dict(color="#D62828", width=2.5),
            name="Trayectoria mediana",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Futuros posibles del valor del portafolio: {label}", font=dict(size=18)),
        xaxis=dict(title="Dias hacia adelante"),
        yaxis=dict(title="Valor del portafolio"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_monte_carlo_history_and_paths_chart(
    returns_df: pd.DataFrame,
    mc_result: dict,
    label: str,
    lookback_days: int = 21,
    max_paths: int = 60,
) -> go.Figure:
    initial_value = float(mc_result["initial_value"])
    path_values = np.asarray(mc_result["path_values"])
    n_paths, horizon = path_values.shape
    shown = min(max_paths, n_paths)

    history = returns_df.tail(lookback_days).copy()
    history["rebased_value"] = history["precio_indice"] / history["precio_indice"].iloc[-1] * initial_value
    history_x = np.arange(-len(history) + 1, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_x,
            y=history["rebased_value"],
            mode="lines",
            line=dict(color="#1D3557", width=2.6),
            name="Ultimo mes observado",
        )
    )

    for i in range(shown):
        future_path = np.concatenate([[initial_value], path_values[i]])
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, horizon + 1),
                y=future_path,
                mode="lines",
                line=dict(color="rgba(69,123,157,0.18)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    median_path = np.median(path_values, axis=0)
    p10 = np.quantile(path_values, 0.10, axis=0)
    p90 = np.quantile(path_values, 0.90, axis=0)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([np.arange(1, horizon + 1), np.arange(horizon, 0, -1)]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor="rgba(231,111,81,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Banda 10%-90%",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(0, horizon + 1),
            y=np.concatenate([[initial_value], median_path]),
            mode="lines",
            line=dict(color="#D62828", width=2.8),
            name="Trayectoria mediana",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Ultimo mes observado y futuros simulados: {label}", font=dict(size=18)),
        xaxis=dict(title="Dias respecto a hoy"),
        yaxis=dict(title="Valor del portafolio"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def monte_carlo_summary(mc_result: dict, confidence: float) -> dict[str, float]:
    pnl = np.asarray(mc_result["pnl"])
    q = np.quantile(pnl, 1 - confidence)
    tail = pnl[pnl <= q]
    one_day_path_values = np.asarray(mc_result["path_values"])[:, 0]
    one_day_pnl = one_day_path_values - float(mc_result["initial_value"])
    one_day_q = np.quantile(one_day_pnl, 1 - confidence)
    one_day_tail = one_day_pnl[one_day_pnl <= one_day_q]
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "var": float(-one_day_q),
        "es": float(-np.mean(one_day_tail)),
        "horizon_var": float(-q),
        "horizon_es": float(-np.mean(tail)),
        "prob_loss": float(np.mean(pnl < 0)),
        "median_final_value": float(np.median(np.asarray(mc_result["final_values"]))),
    }


def stress_scenario_catalog() -> dict[str, dict[str, object]]:
    return {
        "Recesion global": {
            "description": "Las acciones caen con fuerza, los activos defensivos resisten mejor y el apetito por riesgo se deteriora.",
            "shocks": [-0.18, -0.08, -0.12],
        },
        "Shock inflacionario": {
            "description": "Suben las tasas, los bonos sufren y los activos sensibles a inflacion o commodities tienden a sostenerse mejor.",
            "shocks": [-0.10, -0.15, 0.06],
        },
        "Recuperacion de mercado": {
            "description": "Mejora el sentimiento, suben los activos de riesgo y los refugios quedan rezagados.",
            "shocks": [0.12, 0.03, 0.07],
        },
    }


def preset_stress_shocks(n_assets: int, preset_name: str) -> list[float]:
    catalog = stress_scenario_catalog()
    base = catalog.get(preset_name, {}).get("shocks", [0.0, 0.0, 0.0])
    if len(base) < n_assets:
        base = list(base) + [base[-1] if base else 0.0] * (n_assets - len(base))
    return list(base[:n_assets])


def sync_stress_scenario_defaults(selected_tickers: tuple[str, ...], preset_name: str) -> None:
    tickers_key = "|".join(selected_tickers)
    state_key = "stress_defaults_signature"
    new_signature = f"{preset_name}::{tickers_key}"
    old_signature = st.session_state.get(state_key)

    if old_signature == new_signature:
        return

    default_shocks = preset_stress_shocks(len(selected_tickers), preset_name)
    for ticker, shock in zip(selected_tickers, default_shocks):
        st.session_state[f"stress_shock_{ticker}"] = float(shock)
    st.session_state[state_key] = new_signature


def run_stress_test(
    latest_prices: pd.Series,
    tickers: list[str],
    weights: list[float],
    shocks: list[float],
) -> pd.DataFrame:
    weights_arr = np.array(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    portfolio_value = 100.0
    current_values = portfolio_value * weights_arr
    shocked_values = current_values * (1 + np.array(shocks, dtype=float))

    rows = []
    for idx, ticker in enumerate(tickers):
        rows.append(
            {
                "Activo": ticker,
                "Peso": weights_arr[idx],
                "Precio actual": float(latest_prices.iloc[idx]),
                "Shock": float(shocks[idx]),
                "Valor inicial": float(current_values[idx]),
                "Valor estresado": float(shocked_values[idx]),
                "Contribucion P&L": float(shocked_values[idx] - current_values[idx]),
            }
        )
    return pd.DataFrame(rows)


def make_stress_contribution_chart(stress_df: pd.DataFrame, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stress_df["Activo"],
            y=stress_df["Contribucion P&L"],
            marker_color=["#D62828" if x < 0 else "#2A9D8F" for x in stress_df["Contribucion P&L"]],
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=f"Contribucion al P&L bajo stress: {label}", font=dict(size=18)),
        xaxis=dict(title="Activo"),
        yaxis=dict(title="Contribucion al P&L"),
        showlegend=False,
    )
    return fig


def safe_readable_pvalue(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.0001:
        return "< 0.0001"
    return f"{value:.4f}"


st.title("Risk Lab EAFIT")
st.caption("VAR · Causalidad de Granger · VaR · Backtesting")

with st.sidebar:
    st.header("Controles")
    active_module = st.radio(
        "Modulo",
        ["VAR y Causalidad", "VaR y Backtesting", "Monte Carlo", "Stress Testing"],
    )

    if active_module == "VAR y Causalidad":
        start_date = st.date_input("Fecha inicial", value=pd.Timestamp("2010-01-01"))
        max_lag = st.slider("Maximo rezago a evaluar", min_value=2, max_value=8, value=4)
        forecast_steps = st.slider("Horizonte de pronostico", min_value=4, max_value=16, value=8)
        irf_horizon = st.slider("Horizonte impulso-respuesta", min_value=4, max_value=20, value=12)
    else:
        var_start_date = st.date_input("Fecha inicial", value=pd.Timestamp("2018-01-01"))
        asset_choice = st.selectbox("Activo o portafolio", list(VAR_PORTFOLIOS.keys()))
        confidence = st.selectbox("Nivel de confianza", [0.95, 0.99], index=1, format_func=lambda x: f"{int(x * 100)}%")
        if active_module == "VaR y Backtesting":
            rolling_window = st.slider("Ventana rolling", min_value=125, max_value=500, value=250, step=5)
        elif active_module == "Monte Carlo":
            simulation_horizon = st.slider("Horizonte de simulacion (dias)", min_value=5, max_value=60, value=20)
            n_simulations = st.slider("Numero de simulaciones", min_value=1000, max_value=20000, value=5000, step=1000)
            mc_seed = st.number_input("Semilla", min_value=1, max_value=99999, value=12, step=1)
        else:
            stress_preset = st.selectbox(
                "Escenario base",
                list(stress_scenario_catalog().keys()),
            )
        custom_tickers: list[str] = []
        custom_weights: list[float] = []
        custom_label = ""
        if asset_choice == "Portafolio personalizado":
            st.caption("Define hasta 3 activos y sus pesos")
            custom_tickers, custom_weights, custom_label = build_custom_portfolio_inputs()
        if active_module == "VaR y Backtesting":
            run_portfolio_module = st.button("Confirmar y correr backtesting", type="primary")
        elif active_module == "Monte Carlo":
            run_portfolio_module = st.button("Confirmar y correr simulacion", type="primary")
        else:
            run_portfolio_module = st.button("Confirmar y correr stress test", type="primary")


if active_module == "VAR y Causalidad":
    info_banner(
        "Este modulo toma tres series economicas relacionadas con energia para responder una pregunta central: "
        "quien ayuda a predecir a quien, y como se propagan los shocks entre petroleo, gasolina e inflacion esperada a 5 años."
    )

    try:
        raw_df = load_var_dataset(pd.Timestamp(start_date).strftime("%Y-%m-%d"))
    except (URLError, ValueError, pd.errors.EmptyDataError) as exc:
        st.error(f"No fue posible cargar los datos macro desde FRED: {exc}")
        st.stop()

    model_df = build_var_model_frame(raw_df)
    data_for_var = model_df.set_index("fecha")
    lagged_real = compute_lagged_correlation_matrices(data_for_var, max_lag)
    top_links_df = summarize_top_lagged_links(data_for_var, max_lag).head(10)

    section_title("1. La idea econometrica antes del modelo")
    concept_card(
        "Intuicion del modelo",
        "Un modelo VAR trata a todas las variables como endogenas. Eso significa que cada serie se explica con sus propios rezagos "
        "y con los rezagos de las demas. En vez de preguntar solo si X explica Y hoy, preguntamos si la historia conjunta del sistema "
        "ayuda a entender la dinamica completa."
    )
    concept_card(
        "Causalidad de Granger",
        "La causalidad de Granger no significa causalidad filosofica o estructural. Significa algo mas operativo: "
        "si los rezagos de una serie mejoran el pronostico de otra, decimos que la primera Granger-causa a la segunda. "
        "Es una nocion de precedencia informativa."
    )

    section_title("2. Ver las series en niveles, pero no modelarlas asi")
    info_banner(
        "Los niveles sirven para entender el contexto economico: tendencia, episodios de choque y posibles retrasos visuales entre variables. "
        "Pero no son una buena base para medir dependencia dinamica, porque la tendencia comun puede inflar relaciones que en realidad no son informativas."
    )
    st.plotly_chart(make_var_level_chart(raw_df), use_container_width=True)
    info_banner(
        "La conclusion de esta figura es solo economica: el petroleo y la gasolina reaccionan rapido, mientras que las expectativas de inflacion suelen moverse con otra inercia. "
        "Todavia no estamos concluyendo nada sobre correlacion o causalidad."
    )

    section_title("3. Llevar las series a una escala comparable y modelable")
    info_banner(
        "Para que el VAR sea estable, trabajamos con variaciones logaritmicas semanales. Eso reduce tendencia y cambia la pregunta: "
        "ya no modelamos niveles, sino movimientos porcentuales aproximados."
    )
    st.plotly_chart(make_var_return_chart(model_df), use_container_width=True)
    info_banner(
        "Punto clave: el problema no era correlacionar variables en unidades distintas, porque la correlacion no cambia por reescalar linealmente una serie. "
        "El problema verdadero era trabajar con niveles no estacionarios. Por eso el analisis de correlacion y causalidad se hace sobre diferencias logaritmicas."
    )

    transform_left, transform_right = st.columns([1.1, 0.9])
    with transform_left:
        st.plotly_chart(make_scatter_matrix(data_for_var), use_container_width=True)
    with transform_right:
        adf_df = run_adf_summary(data_for_var)
        st.dataframe(
            adf_df.style.format({"ADF estadistico": "{:.3f}", "ADF p-valor": "{:.4f}"}),
            use_container_width=True,
        )
        contemp_oil_gas_corr = positional_correlation(
            data_for_var["oil"].to_numpy(),
            data_for_var["gasoline"].to_numpy(),
        )
        info_banner(
            "La nube multivariada es una comparacion contemporanea sobre diferencias logaritmicas: cada punto toma Δln de las series en el mismo periodo t. "
            "Las celdas simetricas repiten exactamente la misma nube para no introducir confusion innecesaria por invertir ejes. "
            "Sirve para ver comovimiento instantaneo, pero no alcanza para hablar de precedencia temporal. "
            "Por eso despues pasamos a matrices por rezagos y a causalidad de Granger."
        )
        info_banner(explain_correlation_vs_regression(contemp_oil_gas_corr))

    section_title("4. Leer el sistema por rezagos, no por una sola correlacion")
    info_banner(
        "Desde aqui todo se calcula sobre diferencias logaritmicas. Cada celda de la matriz responde: que tanto se asocia Δln de la serie de la fila en t "
        "con Δln de la serie de la columna en t-k. Esto ya es una lectura multivariada de dependencia dinamica."
    )
    info_banner(
        "Cuando k = 0, la lectura es contemporanea y la matriz debe ser simetrica. Cuando k > 0, la lectura es direccional: "
        "fila en t contra columna en t-k. En ese caso, intercambiar fila y columna cambia la pregunta econometrica."
    )
    info_banner(
        "Importante: las correlaciones rezagadas se calculan alineando observaciones por posicion temporal, no por igualdad de fecha. "
        "Es decir, se compara realmente t contra t-k."
    )
    lag_view = st.slider("Rezago a visualizar en la matriz real", min_value=0, max_value=max_lag, value=min(2, max_lag), key="lag_view_real")
    left_col, right_col = st.columns([1.05, 0.95])
    with left_col:
        st.plotly_chart(
            make_lagged_heatmap(lagged_real[lag_view], lag_view, "Matriz de correlacion multivariada"),
            use_container_width=True,
        )
        info_banner(
            summarize_lagged_matrix(lagged_real[lag_view], lag_view)
        )
    with right_col:
        st.dataframe(
            top_links_df[["Rezago", "Serie objetivo", "Serie rezagada", "Correlacion"]].style.format({"Correlacion": "{:.3f}"}),
            use_container_width=True,
        )
        info_banner(
            interpret_top_lagged_links(top_links_df)
        )

    section_title("5. Bajar del sistema completo a una relacion puntual")
    info_banner(
        "Despues de ver la matriz completa, ahora si tiene sentido estudiar una pareja de series. "
        "La idea es pasar del mapa general a una pregunta concreta de transmision."
    )
    pair_left, pair_mid, pair_right = st.columns(3)
    with pair_left:
        pair_predictor = st.selectbox("Serie rezagada", list(data_for_var.columns), index=0, format_func=lambda x: VAR_LABELS[x])
    with pair_mid:
        pair_response = st.selectbox("Serie objetivo", list(data_for_var.columns), index=1, format_func=lambda x: VAR_LABELS[x])
    with pair_right:
        pair_lag = st.selectbox("Rezago de la relacion", list(range(0, max_lag + 1)), index=0)

    pair_chart_left, pair_chart_right = st.columns([1.0, 1.0])
    pair_aligned = prepare_lagged_pair_dataframe(data_for_var, pair_predictor, pair_response, pair_lag)
    with pair_chart_left:
        ccf_df = compute_cross_correlation(data_for_var[pair_predictor], data_for_var[pair_response], max_lag=max_lag)
        st.plotly_chart(
            make_cross_corr_chart(ccf_df, VAR_LABELS[pair_predictor], VAR_LABELS[pair_response]),
            use_container_width=True,
        )
    with pair_chart_right:
        pair_corr = positional_correlation(
            pair_aligned["x"].to_numpy(),
            pair_aligned["y"].to_numpy(),
        )
        st.plotly_chart(
            make_pair_lag_scatter(data_for_var, pair_predictor, pair_response, pair_lag),
            use_container_width=True,
        )
    if pair_lag == 0 and len(pair_aligned) >= 10:
        reg_stats = fit_no_intercept_pair_regression(
            tuple(pair_aligned["x"].to_numpy()),
            tuple(pair_aligned["y"].to_numpy()),
        )
        reg_c1, reg_c2, reg_c3 = st.columns(3)
        reg_c1.metric("Coeficiente sin intercepto", f"{reg_stats['beta']:.3f}")
        reg_c2.metric("p-valor", f"{reg_stats['pvalue']:.4f}")
        reg_c3.metric("R cuadrado", f"{reg_stats['rsquared']:.3f}")
        info_banner(
            f"Como aqui estamos en lag 0, esta lectura se puede comparar directamente con la otra app. "
            f"La correlacion contemporanea es {pair_corr:.3f}, mientras la regresion sin intercepto entrega beta = {reg_stats['beta']:.3f} y R² = {reg_stats['rsquared']:.3f}. "
            "No son la misma estadistica y por eso no deben coincidir numericamente."
        )
    else:
        info_banner(
            interpret_pair_relationship(pair_predictor, pair_response, pair_lag, pair_corr)
        )

    section_title("6. Elegir el numero de rezagos del VAR")
    info_banner(
        "El numero de rezagos define cuanta memoria permitimos dentro del sistema. Si usamos pocos, podemos perder transmision relevante; "
        "si usamos demasiados, corremos el riesgo de sobreajustar y volver opaca la interpretacion."
    )
    lag_df = select_var_order(data_for_var, max_lag)
    lag_left, lag_right = st.columns([1.15, 0.85])
    with lag_left:
        st.dataframe(
            lag_df.style.format({"AIC": "{:.4f}", "BIC": "{:.4f}", "HQIC": "{:.4f}", "FPE": "{:.6f}"}),
            use_container_width=True,
        )
    with lag_right:
        concept_card(
            "Como leer los criterios",
            "Todos buscan buen ajuste, pero penalizan la complejidad de manera distinta. "
            "AIC suele tolerar modelos mas ricos; BIC suele ser mas conservador; HQIC queda en un punto intermedio."
        )
        info_banner(interpret_lag_selection(lag_df))

    choices = criteria_choices(lag_df)
    aic_choice = choices.get("AIC")
    bic_choice = choices.get("BIC")
    hqic_choice = choices.get("HQIC")
    distinct_choices = sorted(set(choices.values()))

    if len(distinct_choices) > 1:
        lag_options = []
        for lag in distinct_choices:
            labels = [name for name, value in choices.items() if value == lag]
            lag_options.append((f"VAR({lag}) - sugerido por {', '.join(labels)}", lag))

        selected_lag_label = st.selectbox(
            "Orden del VAR para el resto del analisis",
            [label for label, _ in lag_options],
            index=next((idx for idx, (_, lag) in enumerate(lag_options) if lag == bic_choice), 0),
        )
        selected_lag_map = {label: lag for label, lag in lag_options}
        suggested_lag = selected_lag_map[selected_lag_label]
        info_banner(
            f"Como los criterios no coinciden, puedes escoger el modelo con el que quieres continuar. "
            f"En esta corrida estas trabajando con VAR({suggested_lag}). "
            "Todas las pruebas y resultados de las secciones siguientes se recalculan con esa eleccion."
        )
    else:
        suggested_lag = distinct_choices[0]
        info_banner(
            f"Los criterios coinciden en VAR({suggested_lag}), asi que usamos ese orden para el resto del analisis."
        )

    results = fit_var_results(data_for_var, suggested_lag)

    section_title("7. Causalidad de Granger como prueba formal")
    info_banner(
        "Aqui ya no preguntamos solo si dos series se parecen por rezagos. La pregunta ahora es mas exigente: "
        "si conozco la historia pasada de una serie, mejora realmente mi pronostico sobre otra?"
    )
    concept_left, concept_right = st.columns(2)
    with concept_left:
        concept_card(
            "Hipotesis nula",
            "Los rezagos de la serie predictora no agregan informacion util para pronosticar la serie objetivo, una vez se controla por la propia historia de la serie objetivo."
        )
    with concept_right:
        concept_card(
            "Que significa rechazarla",
            "Rechazar la nula no prueba causalidad economica profunda. Lo que prueba es precedencia predictiva: el pasado de una serie contiene senal util sobre la otra."
        )
    granger_df = compute_granger_table(data_for_var, suggested_lag)
    pair_g_left, pair_g_right = st.columns(2)
    with pair_g_left:
        granger_predictor = st.selectbox(
            "Serie cuya historia quiero probar",
            list(data_for_var.columns),
            index=0,
            format_func=lambda x: VAR_LABELS[x],
            key="granger_predictor",
        )
    with pair_g_right:
        granger_response = st.selectbox(
            "Serie que quiero pronosticar mejor",
            list(data_for_var.columns),
            index=1,
            format_func=lambda x: VAR_LABELS[x],
            key="granger_response",
        )

    info_banner(interpret_granger_pair(granger_df, granger_predictor, granger_response))

    left_col, right_col = st.columns([1.2, 1.0])
    with left_col:
        st.dataframe(
            granger_df[["Serie objetivo", "Serie que ayuda a predecir", "Mejor rezago", "p-valor minimo", "Conclusion"]].style.format({"p-valor minimo": "{:.4f}"}),
            use_container_width=True,
        )
    with right_col:
        st.plotly_chart(make_granger_heatmap(granger_df), use_container_width=True)
    info_banner(summarize_granger_system(granger_df))

    significant_links = granger_df.loc[granger_df["p-valor minimo"] < 0.05]
    if significant_links.empty:
        info_banner(
            "En esta ventana no aparece evidencia fuerte de causalidad de Granger al 5%. Eso no implica ausencia de relacion economica, "
            "solo que la estructura predictiva lineal no es contundente con esta especificacion."
        )
    else:
        first_link = significant_links.iloc[0]
        info_banner(
            f"Una lectura posible es que {first_link['Serie que ayuda a predecir']} ayuda a predecir {first_link['Serie objetivo']} "
            f"en al menos un rezago, con p-valor minimo de {first_link['p-valor minimo']:.4f}."
        )

    section_title("8. Leer las ecuaciones y revisar residuos")
    info_banner(
        "Un VAR no es una sola ecuacion: es un sistema. Cada variable del modelo tiene su propia ecuacion y en cada una entran sus rezagos "
        "y los rezagos de las otras variables. Por eso conviene leer el bloque completo y no solo una variable aislada."
    )
    equation_texts = build_var_equation_texts(results)
    for item in equation_texts:
        concept_card(item["label"], item["equation_text"])

    equation_table = build_var_equation_table(results)
    eq_left, eq_right = st.columns([1.15, 0.85])
    with eq_left:
        st.dataframe(
            equation_table[["Ecuacion", "Termino", "Coeficiente"]].style.format({"Coeficiente": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
    with eq_right:
        target_variable = st.selectbox(
            "Ecuacion a interpretar",
            list(data_for_var.columns),
            format_func=lambda x: VAR_LABELS[x],
        )
        info_banner(interpret_var_equation(results, target_variable))

    residual_df = compute_residual_diagnostics(data_for_var, suggested_lag)
    section_title("8A. Diagnostico de residuos")
    info_banner(
        "Despues de estimar el sistema, revisamos si en los residuos todavia queda memoria lineal. "
        "Si queda estructura, eso sugiere que el VAR aun no absorbio toda la dinamica relevante."
    )
    st.dataframe(
        residual_df.style.format({"Ljung-Box Q(12)": "{:.2f}", "p-valor": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    info_banner(residual_guidance(residual_df))

    section_title("9. Funciones impulso-respuesta")
    info_banner(
        "La IRF muestra como un shock inesperado en una variable se transmite hacia otra a lo largo del tiempo. "
        "Es una de las salidas mas potentes del VAR porque pasa de prediccion a propagacion dinamica."
    )
    irf_left, irf_right = st.columns(2)
    with irf_left:
        impulse_choice = st.selectbox(
            "Variable del shock",
            list(data_for_var.columns),
            index=0,
            format_func=lambda x: VAR_LABELS[x],
        )
    with irf_right:
        response_choice = st.selectbox(
            "Variable respuesta",
            list(data_for_var.columns),
            index=1,
            format_func=lambda x: VAR_LABELS[x],
        )
    st.plotly_chart(
        make_irf_chart(results, impulse_choice, response_choice, irf_horizon),
        use_container_width=True,
    )
    info_banner(
        interpret_irf(results, impulse_choice, response_choice, irf_horizon)
    )

    section_title("10. Pronostico multivariado")
    info_banner(
        "El pronostico VAR se estima sobre diferencias logaritmicas, pero aqui lo devolvemos a niveles para que la lectura economica sea directa. "
        "Ademas mostramos un intervalo de confianza aproximado en la misma escala de la serie."
    )
    forecast_df = build_var_forecast(data_for_var, raw_df[["fecha", "oil", "gasoline", "energy_cpi"]], suggested_lag, forecast_steps)
    forecast_target = st.selectbox(
        "Serie a pronosticar",
        list(data_for_var.columns),
        index=1,
        format_func=lambda x: VAR_LABELS[x],
        key="forecast_target",
    )
    history_levels = raw_df[["fecha", forecast_target]].copy()
    st.plotly_chart(
        make_forecast_chart(history_levels, forecast_df, forecast_target),
        use_container_width=True,
    )
    info_banner(
        f"El punto central del pronostico muestra la trayectoria esperada de {VAR_LABELS[forecast_target]} en niveles. "
        "La banda refleja la incertidumbre acumulada a medida que avanzamos en el horizonte."
    )

    section_title("11. Cierre del modulo")
    info_banner(
        "La secuencia correcta es: usar niveles para contexto, transformar a una escala estacionaria, leer dependencia por rezagos, "
        "probar precedencia predictiva con Granger y despues usar el VAR para transmision y pronostico. "
        "La pieza decisiva no es igualar unidades, sino trabajar con series comparables y estadisticamente modelables."
    )

elif active_module == "VaR y Backtesting":
    alpha = 1 - confidence
    info_banner(
        "Este modulo no arranca en las pruebas. Arranca recordando que es VaR, como se construye y que significa una excepcion. "
        "Solo despues pasamos al backtesting para evaluar si la medida estaba cumpliendo lo que prometia."
    )

    concept_cols = st.columns(3)
    with concept_cols[0]:
        concept_card(
            "Que promete el VaR",
            f"Un VaR al {int(confidence * 100)}% resume una perdida umbral. La lectura es: en condiciones normales, solo esperamos una perdida peor en alrededor de {int(alpha * 100)} de cada 100 dias."
        )
    with concept_cols[1]:
        concept_card(
            "Que es una excepcion",
            "Una excepcion ocurre cuando la perdida realizada supera el VaR estimado para ese dia. Es el evento basico que luego contamos en el backtesting."
        )
    with concept_cols[2]:
        concept_card(
            "Que hace el backtesting",
            "Contrasta si el numero y el patron temporal de excepciones son compatibles con el nivel de riesgo anunciado por el modelo."
        )

    if asset_choice == "Portafolio personalizado":
        if len(custom_tickers) == 0:
            st.error("Ingresa al menos un ticker para construir el portafolio personalizado.")
            st.stop()
        if any(weight <= 0 for weight in custom_weights):
            st.error("Todos los pesos del portafolio personalizado deben ser positivos.")
            st.stop()
        if sum(custom_weights) <= 0:
            st.error("La suma de los pesos debe ser mayor que cero.")
            st.stop()
        selected_tickers = tuple(custom_tickers)
        selected_weights = custom_weights
        asset_label = custom_label
    else:
        config = VAR_PORTFOLIOS[asset_choice]
        selected_tickers = tuple(config["tickers"])
        selected_weights = config["weights"]
        asset_label = asset_choice

    if not run_portfolio_module:
        info_banner(
            "Ajusta activo(s), pesos, fecha inicial y parametros de riesgo. "
            "Cuando estes listo, pulsa `Confirmar y correr backtesting` para descargar datos y ejecutar el analisis."
        )
        st.stop()

    try:
        prices = load_market_prices(selected_tickers, pd.Timestamp(var_start_date).strftime("%Y-%m-%d"))
    except Exception as exc:
        st.error(f"No fue posible descargar precios de mercado: {exc}")
        st.stop()

    returns_df = build_portfolio_returns(prices, selected_weights)
    st.plotly_chart(make_price_return_chart(returns_df, asset_label), use_container_width=True)

    section_title("1. La serie sobre la que medimos el riesgo")
    info_banner(
        "El VaR no se calcula sobre precios, sino sobre retornos o P&L. Aqui usamos retornos logaritmicos diarios y una ventana rolling para que la medida se actualice con el mercado."
    )
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Observaciones", f"{len(returns_df):,}")
    summary_col2.metric("Volatilidad diaria", f"{returns_df['retorno'].std() * 100:.2f}%")
    summary_col3.metric("Retorno medio diario", f"{returns_df['retorno'].mean() * 100:.3f}%")

    section_title("2. Construir el VaR")
    info_banner(
        "Cada modelo toma la misma ventana de informacion pasada y produce un umbral VaR diario. "
        "La diferencia entre modelos esta en como resumen la distribucion de riesgo."
    )
    var_df = run_var_models(tuple(returns_df["retorno"]), rolling_window, alpha)
    var_df["fecha"] = returns_df["fecha"].iloc[rolling_window:].reset_index(drop=True)

    selected_models = ["historical", "varcov", "ewma"]

    walkthrough_model = st.selectbox(
        "Modelo para explicar el VaR paso a paso",
        selected_models,
        format_func=lambda x: MODEL_NAMES[x],
    )
    info_banner(explain_var_method(walkthrough_model))
    st.plotly_chart(
        make_single_var_exception_chart(var_df, walkthrough_model, asset_label),
        use_container_width=True,
    )
    info_banner(
        "La linea VaR marca la perdida umbral estimada para cada fecha. "
        "Cuando el retorno realizado cae por debajo de esa linea, aparece una excepcion."
    )

    section_title("3. Que significa hacer backtesting")
    info_banner(
        f"Si el VaR esta calibrado al {int(confidence * 100)}%, esperamos aproximadamente {alpha:.0%} de excepciones. "
        "Backtesting significa revisar si ese conteo y su patron temporal son razonables."
    )
    backtest_df = summarize_backtest(var_df, alpha)
    overview_cols = st.columns(3)
    selected_row = backtest_df.loc[backtest_df["Modelo"] == MODEL_NAMES[walkthrough_model]].iloc[0]
    overview_cols[0].metric("Excepciones del modelo", int(selected_row["Excepciones"]))
    overview_cols[1].metric("Tasa observada", f"{selected_row['Tasa observada']:.2%}")
    overview_cols[2].metric("Breach ratio", f"{selected_row['Breach ratio']:.2f}")

    section_title("4. Comparar modelos de VaR")
    info_banner(
        "Una vez entendido un modelo, ahora si tiene sentido comparar varias metodologias. "
        "La tabla resume cobertura total, agrupacion de excepciones y, cuando aplica, el semaforo regulatorio."
    )
    st.dataframe(
        backtest_df.style.format(
            {
                "Tasa observada": "{:.3%}",
                "Breach ratio": "{:.2f}",
                "Kupiec LR": "{:.3f}",
                "Kupiec p-valor": "{:.4f}",
                "Independencia LR": "{:.3f}",
                "Independencia p-valor": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.plotly_chart(make_exception_bar(backtest_df, alpha), use_container_width=True)
    st.plotly_chart(
        make_var_backtest_chart(var_df, selected_models, asset_label),
        use_container_width=True,
    )

    section_title("5. Como leer las pruebas")
    read_col1, read_col2 = st.columns(2)
    with read_col1:
        info_banner(
            "Kupiec pregunta si el numero total de excepciones es compatible con la tasa teorica. "
            "Un p-valor bajo sugiere que el modelo subestima o sobreestima riesgo."
        )
        best_kupiec = backtest_df.sort_values("Kupiec p-valor", ascending=False).iloc[0]
        st.write(
            f"Mejor desempeno por cobertura incondicional: **{best_kupiec['Modelo']}** "
            f"(p-valor {safe_readable_pvalue(float(best_kupiec['Kupiec p-valor']))})."
        )
    with read_col2:
        info_banner(
            "La prueba de independencia mira si las excepciones aparecen aisladas o en racimos. "
            "Si se agrupan demasiado, el modelo puede estar llegando tarde a los cambios de volatilidad."
        )
        best_ind = backtest_df.sort_values("Independencia p-valor", ascending=False).iloc[0]
        st.write(
            f"Mejor desempeno por independencia: **{best_ind['Modelo']}** "
            f"(p-valor {safe_readable_pvalue(float(best_ind['Independencia p-valor']))})."
        )

    if np.isclose(alpha, 0.01):
        section_title("6. Semaforo de Basilea")
        info_banner(
            "Para VaR al 99%, Basilea popularizo un semaforo simple sobre las ultimas 250 observaciones: verde, amarilla o roja segun el numero de excepciones."
        )
        st.dataframe(backtest_df[["Modelo", "Excepciones", "Semaforo Basilea"]], use_container_width=True)

    section_title("7. Cierre del modulo")
    info_banner(
        "La idea central es simple: un modelo VaR produce una promesa probabilistica, y el backtesting revisa si esa promesa se sostuvo en los datos. "
        "Por eso no estamos backtesteando 'econometria' en abstracto; estamos validando si un umbral de riesgo realmente cubria las perdidas observadas."
    )

elif active_module == "Monte Carlo":
    alpha = 1 - confidence
    info_banner(
        "Este modulo toma el mismo portafolio del bloque de VaR, pero en vez de resumir el riesgo con una formula puntual, "
        "genera muchos escenarios futuros posibles para construir una distribucion simulada de resultados."
    )

    concept_cols = st.columns(3)
    with concept_cols[0]:
        concept_card(
            "Que hace Monte Carlo",
            "Parte de la historia observada del portafolio y genera muchos futuros posibles para responder una pregunta de negocio: cuanto podria valer el portafolio dentro de unos dias."
        )
    with concept_cols[1]:
        concept_card(
            "Que obtenemos",
            "Un abanico de trayectorias y valores finales posibles. Eso permite hablar de escenarios buenos, centrales y adversos, no solo de estadisticos."
        )
    with concept_cols[2]:
        concept_card(
            "En que se diferencia del VaR clasico",
            "Aqui no nos quedamos con una sola formula. Construimos explicitamente muchos escenarios y dejamos que la distribucion simulada nos muestre la cola del riesgo."
        )

    if asset_choice == "Portafolio personalizado":
        if len(custom_tickers) == 0:
            st.error("Ingresa al menos un ticker para construir el portafolio personalizado.")
            st.stop()
        if any(weight <= 0 for weight in custom_weights):
            st.error("Todos los pesos del portafolio personalizado deben ser positivos.")
            st.stop()
        if sum(custom_weights) <= 0:
            st.error("La suma de los pesos debe ser mayor que cero.")
            st.stop()
        selected_tickers = tuple(custom_tickers)
        selected_weights = custom_weights
        asset_label = custom_label
    else:
        config = VAR_PORTFOLIOS[asset_choice]
        selected_tickers = tuple(config["tickers"])
        selected_weights = config["weights"]
        asset_label = asset_choice

    if not run_portfolio_module:
        info_banner(
            "Configura el portafolio, el horizonte y el numero de simulaciones. "
            "Cuando estes listo, pulsa `Confirmar y correr simulacion`."
        )
        st.stop()

    try:
        prices = load_market_prices(selected_tickers, pd.Timestamp(var_start_date).strftime("%Y-%m-%d"))
    except Exception as exc:
        st.error(f"No fue posible descargar precios de mercado: {exc}")
        st.stop()

    returns_df = build_portfolio_returns(prices, selected_weights)
    st.plotly_chart(make_price_return_chart(returns_df, asset_label), use_container_width=True)

    section_title("1. Definir el punto de partida")
    info_banner(
        "Primero fijamos el portafolio actual y usamos su historia reciente para generar futuros plausibles. "
        "La pregunta principal no es tecnica, sino practica: como podria verse el valor del portafolio durante el proximo horizonte?"
    )
    mc_result = run_monte_carlo_portfolio(
        prices,
        tuple(selected_weights),
        simulation_horizon,
        n_simulations,
        int(mc_seed),
    )
    mc_summary = monte_carlo_summary(mc_result, confidence)

    setup_cols = st.columns(3)
    setup_cols[0].metric("Valor base del portafolio", f"{mc_result['initial_value']:.2f}")
    setup_cols[1].metric("Horizonte", f"{simulation_horizon} dias")
    setup_cols[2].metric("Simulaciones", f"{n_simulations:,}")

    section_title("2. Ver futuros posibles del portafolio")
    info_banner(
        f"Aqui generamos {n_simulations:,} escenarios a {simulation_horizon} dias. "
        "Cada linea representa un futuro posible para el valor del portafolio. La banda central resume la zona donde cae una buena parte de esos resultados."
    )
    mc_cols = st.columns(4)
    mc_cols[0].metric("Valor mediano final", f"{mc_summary['median_final_value']:.2f}")
    mc_cols[1].metric("P&L esperado", f"{mc_summary['mean_pnl']:.2f}")
    mc_cols[2].metric("Prob. de perdida", f"{mc_summary['prob_loss']:.2%}")
    mc_cols[3].metric("VaR diario Monte Carlo", f"{mc_summary['var']:.2f}")

    st.plotly_chart(
        make_monte_carlo_history_and_paths_chart(returns_df, mc_result, asset_label),
        use_container_width=True,
    )
    st.plotly_chart(make_monte_carlo_terminal_chart(mc_result, asset_label), use_container_width=True)
    st.plotly_chart(make_monte_carlo_histogram(mc_result, confidence, asset_label), use_container_width=True)
    info_banner(
        "La grafica principal conecta el ultimo mes observado con muchos caminos simulados hacia adelante. "
        "La trayectoria mediana te da un escenario central y la banda te deja ver dispersion. "
        "El histograma de valores finales y el histograma de P&L ayudan a traducir esos caminos en preguntas de negocio: "
        "cuanto podria valer el portafolio y cual seria una perdida severa pero plausible."
    )
    info_banner(
        f"Para mantener coherencia con el curso, el VaR Monte Carlo mostrado arriba es diario: se calcula con la distribucion del primer dia simulado. "
        f"Las trayectorias y los valores finales, en cambio, corresponden al horizonte completo de {simulation_horizon} dias."
    )

    section_title("3. Cierre del modulo")
    info_banner(
        "Monte Carlo es util porque convierte incertidumbre en escenarios visibles. "
        "En vez de quedarte con un solo numero, puedes hablar de trayectorias, rangos esperados y resultados adversos sobre el valor del portafolio."
    )

elif active_module == "Stress Testing":
    info_banner(
        "Stress testing responde una pregunta distinta a Monte Carlo: no busca muchos escenarios probables, sino pocos escenarios severos e interpretables."
    )

    concept_cols = st.columns(3)
    with concept_cols[0]:
        concept_card(
            "Logica del stress test",
            "Aplicamos shocks directos a los activos del portafolio y observamos el impacto total sobre su valor."
        )
    with concept_cols[1]:
        concept_card(
            "Que aprendemos",
            "No una probabilidad, sino una vulnerabilidad: cuanto perderiamos si ocurriera un escenario adverso concreto."
        )
    with concept_cols[2]:
        concept_card(
            "Diferencia frente a Monte Carlo",
            "Monte Carlo distribuye muchos futuros posibles. Stress testing fuerza un escenario severo especifico para ver sensibilidad y concentracion de perdidas."
        )

    if asset_choice == "Portafolio personalizado":
        if len(custom_tickers) == 0:
            st.error("Ingresa al menos un ticker para construir el portafolio personalizado.")
            st.stop()
        if any(weight <= 0 for weight in custom_weights):
            st.error("Todos los pesos del portafolio personalizado deben ser positivos.")
            st.stop()
        if sum(custom_weights) <= 0:
            st.error("La suma de los pesos debe ser mayor que cero.")
            st.stop()
        selected_tickers = tuple(custom_tickers)
        selected_weights = custom_weights
        asset_label = custom_label
    else:
        config = VAR_PORTFOLIOS[asset_choice]
        selected_tickers = tuple(config["tickers"])
        selected_weights = config["weights"]
        asset_label = asset_choice

    catalog = stress_scenario_catalog()
    scenario_description = str(catalog[stress_preset]["description"])
    n_assets = len(selected_tickers)
    sync_stress_scenario_defaults(selected_tickers, stress_preset)
    section_title("1. Escenario narrativo")
    concept_card(stress_preset, scenario_description)
    info_banner(
        "Debajo de la narrativa puedes ajustar los shocks por activo. "
        "La idea es partir de una historia economica coherente y luego aterrizarla en impactos concretos sobre el portafolio. "
        "Los valores aparecen precargados como sugerencia y puedes modificarlos libremente."
    )
    shock_cols = st.columns(max(n_assets, 1))
    manual_shocks = []
    for idx, ticker in enumerate(selected_tickers):
        with shock_cols[idx]:
            widget_key = f"stress_shock_{ticker}"
            widget_kwargs = {
                "label": f"Shock {ticker}",
                "min_value": -0.95,
                "max_value": 2.0,
                "step": 0.01,
                "format": "%.2f",
                "key": widget_key,
            }
            if widget_key not in st.session_state:
                widget_kwargs["value"] = 0.0
            shock_value = st.number_input(
                **widget_kwargs,
            )
            manual_shocks.append(shock_value)

    suggested_df = pd.DataFrame(
        {
            "Activo": list(selected_tickers),
            "Shock sugerido": manual_shocks,
        }
    )
    st.dataframe(
        suggested_df.style.format({"Shock sugerido": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    if not run_portfolio_module:
        info_banner(
            "Configura el portafolio y los shocks del escenario. "
            "Cuando estes listo, pulsa `Confirmar y correr stress test`."
        )
        st.stop()

    try:
        prices = load_market_prices(selected_tickers, pd.Timestamp(var_start_date).strftime("%Y-%m-%d"))
    except Exception as exc:
        st.error(f"No fue posible descargar precios de mercado: {exc}")
        st.stop()

    latest_prices = prices.iloc[-1]
    returns_df = build_portfolio_returns(prices, list(selected_weights))
    st.plotly_chart(make_price_return_chart(returns_df, asset_label), use_container_width=True)

    section_title("2. Traducir la narrativa a impacto de portafolio")
    info_banner(
        "Cada shock se interpreta como un cambio porcentual inmediato sobre el valor del activo. "
        "El objetivo es ver cuanto perderia o ganaria el portafolio si la historia descrita arriba se materializara."
    )
    stress_df = run_stress_test(latest_prices, list(selected_tickers), list(selected_weights), manual_shocks)
    total_initial = float(stress_df["Valor inicial"].sum())
    total_stressed = float(stress_df["Valor estresado"].sum())
    total_pnl = float(stress_df["Contribucion P&L"].sum())

    stress_cols = st.columns(3)
    stress_cols[0].metric("Valor inicial", f"{total_initial:.2f}")
    stress_cols[1].metric("Valor bajo stress", f"{total_stressed:.2f}")
    stress_cols[2].metric("P&L del escenario", f"{total_pnl:.2f}")

    section_title("3. Ver contribuciones al resultado")
    st.dataframe(
        stress_df.style.format(
            {
                "Peso": "{:.2%}",
                "Precio actual": "{:.2f}",
                "Shock": "{:.2%}",
                "Valor inicial": "{:.2f}",
                "Valor estresado": "{:.2f}",
                "Contribucion P&L": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(make_stress_contribution_chart(stress_df, asset_label), use_container_width=True)
    info_banner(
        "La clave aqui no es una probabilidad de cola, sino una pregunta de vulnerabilidad. "
        "Que activo explica la mayor parte de la perdida? Donde esta la concentracion del riesgo del portafolio?"
    )

    section_title("4. Cierre del modulo")
    info_banner(
        "Stress testing complementa al VaR y a Monte Carlo. "
        "Mientras el VaR resume un umbral y Monte Carlo genera distribuciones, el stress test obliga a mirar escenarios concretos y severos."
    )
