from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from plotly.subplots import make_subplots
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf


st.set_page_config(
    page_title="Ingresos Corporativos MX",
    page_icon="📊",
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
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_FILE = Path(__file__).with_name("dataMX.xlsx")
COMPANY_COLUMNS = {
    "Bimbo": "Bimbo",
    "Coca Cola": "Coca Cola",
    "Walmart": "Walmart",
}
ANALYSIS_COLUMNS = {
    "Bimbo": "Bimbo",
    "Coca Cola": "Coca Cola",
    "Walmart": "Walmart",
    "Real GDP": "Real GDP",
}
COLORS = {
    "Bimbo": "#D62828",
    "Coca Cola": "#1D3557",
    "Walmart": "#2A9D8F",
    "Real GDP": "#6A4C93",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    df = df.rename(columns={"Mes": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df[df["fecha"].dt.year.between(2013, 2025)].copy()
    df["anio"] = df["fecha"].dt.year
    df["trimestre"] = df["fecha"].dt.quarter
    return df


def transform_series(df: pd.DataFrame, companies: list[str], mode: str) -> pd.DataFrame:
    result = pd.DataFrame({"fecha": df["fecha"]})

    for company in companies:
        series = df[COMPANY_COLUMNS[company]].astype(float)
        if mode == "Valores absolutos":
            transformed = series
        elif mode == "Variacion porcentual":
            transformed = series.pct_change() * 100
        elif mode == "Diferencia logaritmica":
            transformed = np.log(series).diff() * 100
        else:
            transformed = np.log(series)

        result[company] = transformed

    return result


def make_chart(
    df: pd.DataFrame,
    companies: list[str],
    mode: str,
    chart_title: str | None = None,
) -> go.Figure:
    labels = {
        "Valores absolutos": "Ingresos trimestrales",
        "Variacion porcentual": "Variacion trimestral (%)",
        "Diferencia logaritmica": "Diferencia logaritmica trimestral (%)",
        "Logaritmos": "Logaritmo natural de los ingresos",
    }

    hover_templates = {
        "Valores absolutos": "%{y:,.0f}",
        "Variacion porcentual": "%{y:.2f}%",
        "Diferencia logaritmica": "%{y:.2f}%",
        "Logaritmos": "%{y:.4f}",
    }

    fig = go.Figure()
    for company in companies:
        fig.add_trace(
            go.Scatter(
                x=df["fecha"],
                y=df[company],
                mode="lines+markers",
                name=company,
                line=dict(color=COLORS[company], width=2.4),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{company}</b><br>"
                    "Fecha: %{x|%Y-%m-%d}<br>"
                    f"{labels[mode]}: {hover_templates[mode]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=70, b=30),
        title=dict(text=chart_title or labels[mode], font=dict(size=20)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=labels[mode]),
    )

    if mode == "Logaritmos":
        fig.update_yaxes(range=[10, 13])

    if mode != "Valores absolutos":
        fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)

    return fig


def build_data_table(
    df: pd.DataFrame,
    companies: list[str],
    absolute_df: pd.DataFrame,
    log_df: pd.DataFrame,
    percent_df: pd.DataFrame,
    log_diff_df: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "fecha": df["fecha"],
            "anio": df["anio"],
            "trimestre": df["trimestre"],
        }
    )

    for company in companies:
        table[f"{company}_absoluto"] = absolute_df[company]
        table[f"{company}_log"] = log_df[company]
        table[f"{company}_var_pct"] = percent_df[company]
        table[f"{company}_var_log"] = log_diff_df[company]

    return table


def build_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    analysis_df = pd.DataFrame({"fecha": df["fecha"]})
    for label, column in ANALYSIS_COLUMNS.items():
        analysis_df[label] = np.log(df[column].astype(float)).diff() * 100
    return analysis_df.dropna().reset_index(drop=True)


def make_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_df.values, 3),
            texttemplate="%{text}",
            hovertemplate="Fila: %{y}<br>Columna: %{x}<br>Correlacion: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=40, t=60, b=40),
        title=dict(text="Matriz de correlacion", font=dict(size=20)),
    )
    return fig


def fit_simple_regression(analysis_df: pd.DataFrame, predictor: str):
    reg_df = analysis_df[["fecha", "Coca Cola", predictor]].dropna().copy()
    y = reg_df["Coca Cola"]
    x = sm.add_constant(reg_df[predictor])
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    reg_df["ajustado"] = model.predict(x)
    reg_df["residuo"] = model.resid
    return reg_df, model


def fit_regression_without_influential_outliers(analysis_df: pd.DataFrame, predictor: str):
    reg_df = analysis_df[["fecha", "Coca Cola", predictor]].dropna().copy()
    x_full = sm.add_constant(reg_df[predictor])
    base_model = sm.OLS(reg_df["Coca Cola"], x_full).fit()
    influence = base_model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(reg_df)

    filtered_df = reg_df.loc[cooks_d <= threshold].copy()
    filtered_df["cook_d"] = cooks_d[cooks_d <= threshold]
    outliers_df = reg_df.loc[cooks_d > threshold].copy()
    outliers_df["cook_d"] = cooks_d[cooks_d > threshold]

    # Para el caso del PIB, removemos explícitamente 2020 en la versión pedagógica
    # "sin outliers" porque los trimestres de pandemia dominan la nube de puntos.
    if predictor == "Real GDP":
        pandemic_mask = reg_df["fecha"].dt.year == 2020
        pandemic_df = reg_df.loc[pandemic_mask].copy()
        if not pandemic_df.empty:
            pandemic_df["cook_d"] = cooks_d[pandemic_mask.to_numpy()]
            outliers_df = pd.concat([outliers_df, pandemic_df], ignore_index=True)
        filtered_df = filtered_df.loc[filtered_df["fecha"].dt.year != 2020].copy()
        outliers_df = outliers_df.drop_duplicates(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    x_filtered = sm.add_constant(filtered_df[predictor])
    filtered_model = sm.OLS(filtered_df["Coca Cola"], x_filtered).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 1},
    )
    filtered_df["ajustado"] = filtered_model.predict(x_filtered)
    filtered_df["residuo"] = filtered_model.resid
    return filtered_df, filtered_model, outliers_df, threshold


def make_regression_scatter(reg_df: pd.DataFrame, predictor: str) -> go.Figure:
    order = np.argsort(reg_df[predictor].values)
    x_sorted = reg_df[predictor].values[order]
    y_hat_sorted = reg_df["ajustado"].values[order]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=reg_df[predictor],
            y=reg_df["Coca Cola"],
            mode="markers",
            name="Observaciones",
            marker=dict(color="#457B9D", size=8, opacity=0.8),
            hovertemplate=(
                f"{predictor}: %{{x:.2f}}%<br>"
                "Coca Cola: %{y:.2f}%<br>"
                "Fecha: %{customdata|%Y-%m-%d}<extra></extra>"
            ),
            customdata=reg_df["fecha"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=y_hat_sorted,
            mode="lines",
            name="Recta estimada",
            line=dict(color="#D62828", width=2.5),
            hovertemplate="Ajuste: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Regresion simple sobre diferencias logaritmicas", font=dict(size=18)),
        xaxis=dict(title=f"{predictor} - variacion logaritmica trimestral (%)"),
        yaxis=dict(title="Coca Cola - variacion logaritmica trimestral (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_residual_chart(reg_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residuos en el tiempo", "Histograma de residuos"))
    fig.add_trace(
        go.Scatter(
            x=reg_df["fecha"],
            y=reg_df["residuo"],
            mode="lines+markers",
            name="Residuos",
            line=dict(color="#1D3557", width=2),
            marker=dict(size=6),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Residuo: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1, row=1, col=1)
    fig.add_trace(
        go.Histogram(
            x=reg_df["residuo"],
            nbinsx=15,
            marker=dict(color="#A8DADC", line=dict(color="white", width=0.6)),
            name="Histograma",
            hovertemplate="Residuo: %{x:.2f}<br>Frecuencia: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text="Diagnostico basico de residuos", font=dict(size=18)),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Fecha", row=1, col=1)
    fig.update_yaxes(title_text="Residuo", row=1, col=1)
    fig.update_xaxes(title_text="Residuo", row=1, col=2)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
    return fig


def run_adf_test(series: pd.Series) -> tuple[float | None, float | None]:
    clean = series.dropna()
    if len(clean) < 12 or clean.nunique() <= 1:
        return None, None
    stat, pvalue, *_ = adfuller(clean)
    return float(stat), float(pvalue)


def build_modeling_recipe(df: pd.DataFrame, series_name: str) -> dict[str, pd.Series]:
    original = df[series_name].astype(float)
    log_series = np.log(original)
    seasonal_diff = log_series.diff(4)
    ready_series = seasonal_diff.diff()
    return {
        "original": original,
        "log": log_series,
        "seasonal_diff": seasonal_diff,
        "ready": ready_series,
    }


def make_single_series_chart(
    x: pd.Series,
    y: pd.Series,
    title: str,
    y_title: str,
    color: str,
    add_zero_line: bool = False,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=color, width=2.3),
            marker=dict(size=6),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Valor: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=y_title),
        showlegend=False,
    )
    if add_zero_line:
        fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    return fig


def make_recipe_summary_table(recipe_series: dict[str, pd.Series], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {
        "original": "Serie original",
        "log": "Logaritmo natural",
        "seasonal_diff": "Diferencia estacional logarítmica (lag 4)",
        "ready": "Diferencia adicional sobre la serie estacional",
    }
    for key, label in labels.items():
        stat, pvalue = run_adf_test(recipe_series[key])
        clean = recipe_series[key].dropna()
        rows.append(
            {
                "Paso": label,
                "Observaciones utiles": int(len(clean)),
                "Media": float(clean.mean()) if len(clean) else np.nan,
                "Desv. est.": float(clean.std()) if len(clean) else np.nan,
                "ADF estadistico": stat,
                "ADF p-valor": pvalue,
            }
        )
    return pd.DataFrame(rows)


def make_correlation_lag_chart(
    values: np.ndarray,
    title: str,
    y_title: str,
    confidence: float,
) -> go.Figure:
    lags = np.arange(len(values))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lags,
            y=values,
            marker_color="#457B9D",
            hovertemplate="Rezago: %{x}<br>Valor: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1)
    fig.add_hline(y=confidence, line_dash="dash", line_color="#D62828", line_width=1)
    fig.add_hline(y=-confidence, line_dash="dash", line_color="#D62828", line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Rezago"),
        yaxis=dict(title=y_title),
        showlegend=False,
    )
    return fig


def compute_time_dependence_diagnostics(series: pd.Series, max_lag: int = 12) -> dict:
    clean = series.dropna()
    n = len(clean)
    usable_lag = min(max_lag, max(1, n // 2 - 1))
    acf_vals = acf(clean, nlags=usable_lag, fft=False)
    pacf_vals = pacf(clean, nlags=usable_lag, method="ywm")
    conf = 1.96 / np.sqrt(n)
    lb = acorr_ljungbox(clean, lags=list(range(1, usable_lag + 1)), return_df=True)
    return {
        "acf": acf_vals,
        "pacf": pacf_vals,
        "conf": conf,
        "ljung_box": lb.reset_index().rename(columns={"index": "lag", "lb_stat": "Q", "lb_pvalue": "pvalue"}),
        "n": n,
        "usable_lag": usable_lag,
    }


def make_stylized_pattern_chart(pattern_name: str) -> go.Figure:
    lags = np.arange(0, 9)
    if pattern_name == "AR":
        acf_vals = np.array([1.00, 0.72, 0.50, 0.34, 0.23, 0.15, 0.10, 0.06, 0.03])
        pacf_vals = np.array([1.00, 0.72, 0.05, 0.02, 0.00, -0.01, 0.00, 0.00, 0.00])
        title = "Patron tipico AR(p): PACF corta, ACF decae"
    elif pattern_name == "MA":
        acf_vals = np.array([1.00, 0.65, 0.06, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00])
        pacf_vals = np.array([1.00, 0.65, 0.42, 0.28, 0.18, 0.10, 0.06, 0.03, 0.01])
        title = "Patron tipico MA(q): ACF corta, PACF decae"
    else:
        acf_vals = np.array([1.00, 0.58, 0.33, 0.18, 0.10, 0.06, 0.03, 0.02, 0.01])
        pacf_vals = np.array([1.00, 0.52, 0.24, 0.14, 0.08, 0.05, 0.03, 0.02, 0.01])
        title = "Patron tipico ARIMA/ARMA: ACF y PACF decaen"

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF estilizada", "PACF estilizada"))
    fig.add_trace(
        go.Bar(x=lags, y=acf_vals, marker_color="#457B9D", hovertemplate="Rezago: %{x}<br>ACF: %{y:.2f}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=lags, y=pacf_vals, marker_color="#D62828", hovertemplate="Rezago: %{x}<br>PACF: %{y:.2f}<extra></extra>"),
        row=1,
        col=2,
    )
    for col in [1, 2]:
        fig.add_hline(y=0, line_dash="dot", line_color="#7A7A7A", line_width=1, row=1, col=col)
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=30, r=30, t=70, b=30),
        title=dict(text=title, font=dict(size=17)),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Rezago", row=1, col=1)
    fig.update_xaxes(title_text="Rezago", row=1, col=2)
    fig.update_yaxes(title_text="Correlacion", row=1, col=1)
    fig.update_yaxes(title_text="Correlacion parcial", row=1, col=2)
    return fig


def suggest_family_from_diagnostics(acf_vals: np.ndarray, pacf_vals: np.ndarray, conf: float) -> str:
    acf_sig = np.where(np.abs(acf_vals[1:]) > conf)[0] + 1
    pacf_sig = np.where(np.abs(pacf_vals[1:]) > conf)[0] + 1
    acf_count = len(acf_sig)
    pacf_count = len(pacf_sig)

    if pacf_count <= 2 and acf_count > pacf_count:
        return "AR"
    if acf_count <= 2 and pacf_count > acf_count:
        return "MA"
    return "ARIMA"


def fit_univariate_model(series: pd.Series, family: str, p: int, q: int):
    clean = series.dropna()
    if family == "AR":
        order = (p, 0, 0)
    elif family == "MA":
        order = (0, 0, q)
    else:
        order = (p, 0, q)

    model = ARIMA(clean, order=order, trend="c").fit()
    fitted = pd.DataFrame(
        {
            "fecha": clean.index,
            "observado": clean.values,
            "ajustado": model.fittedvalues,
            "residuo": model.resid,
        }
    )
    resid_lb = acorr_ljungbox(model.resid, lags=list(range(1, min(12, max(2, len(clean) // 2 - 1)) + 1)), return_df=True)
    resid_lb = resid_lb.reset_index().rename(columns={"index": "lag", "lb_stat": "Q", "lb_pvalue": "pvalue"})
    return model, fitted, resid_lb, order


def search_candidate_models(series: pd.Series, max_p: int = 4, max_q: int = 4) -> pd.DataFrame:
    clean = series.dropna()
    candidates: list[dict] = []

    def residual_score(resid: pd.Series) -> tuple[float | None, int]:
        max_lag = min(8, max(2, len(clean) // 2 - 1))
        lb = acorr_ljungbox(resid, lags=list(range(1, max_lag + 1)), return_df=True)
        min_pvalue = float(lb["lb_pvalue"].min()) if not lb.empty else np.nan
        rejected = int((lb["lb_pvalue"] < 0.05).sum()) if not lb.empty else 0
        return min_pvalue, rejected

    for p in range(max_p + 1):
        if p == 0:
            continue
        try:
            model = ARIMA(clean, order=(p, 0, 0), trend="c").fit()
            min_pvalue, rejected = residual_score(model.resid)
            candidates.append(
                {
                    "familia": "AR",
                    "p": p,
                    "q": 0,
                    "orden": f"({p},0,0)",
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "resid_min_pvalue": min_pvalue,
                    "resid_rechazos_lb": rejected,
                }
            )
        except Exception:
            continue

    for q in range(max_q + 1):
        if q == 0:
            continue
        try:
            model = ARIMA(clean, order=(0, 0, q), trend="c").fit()
            min_pvalue, rejected = residual_score(model.resid)
            candidates.append(
                {
                    "familia": "MA",
                    "p": 0,
                    "q": q,
                    "orden": f"(0,0,{q})",
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "resid_min_pvalue": min_pvalue,
                    "resid_rechazos_lb": rejected,
                }
            )
        except Exception:
            continue

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            if p == 0 or q == 0:
                continue
            try:
                model = ARIMA(clean, order=(p, 0, q), trend="c").fit()
                min_pvalue, rejected = residual_score(model.resid)
                candidates.append(
                    {
                        "familia": "ARIMA",
                        "p": p,
                        "q": q,
                        "orden": f"({p},0,{q})",
                        "aic": float(model.aic),
                        "bic": float(model.bic),
                        "resid_min_pvalue": min_pvalue,
                        "resid_rechazos_lb": rejected,
                    }
                )
            except Exception:
                continue

    candidate_df = pd.DataFrame(candidates)
    if candidate_df.empty:
        return candidate_df

    candidate_df["pasa_ljung_box"] = candidate_df["resid_min_pvalue"] > 0.05
    candidate_df = candidate_df.sort_values(
        by=["pasa_ljung_box", "resid_rechazos_lb", "aic", "bic"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return candidate_df


def top_candidates_by_family(candidate_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    pieces = []
    for family in ["AR", "MA", "ARIMA"]:
        family_df = candidate_df[candidate_df["familia"] == family].head(top_n).copy()
        if not family_df.empty:
            pieces.append(family_df)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def economic_model_interpretation(model_result, family: str, series_name: str) -> str:
    ar_params = {k: v for k, v in model_result.params.items() if str(k).startswith("ar.L")}
    ma_params = {k: v for k, v in model_result.params.items() if str(k).startswith("ma.L")}

    if family == "AR":
        if ar_params:
            strongest_name = max(ar_params, key=lambda k: abs(ar_params[k]))
            strongest_value = ar_params[strongest_name]
            return (
                f"Un modelo AR sugiere que la dinámica de {series_name} depende sobre todo de su propia inercia. "
                f"En términos económicos, cuando la serie se mueve hoy, parte de ese movimiento tiende a arrastrarse "
                f"hacia los siguientes trimestres. El coeficiente más visible aquí es {strongest_name} = {strongest_value:.3f}: "
                f"si es positivo, los movimientos recientes tienden a continuar; si fuera negativo, la serie tendería a corregirse "
                f"más rápidamente en el siguiente periodo."
            )
        return (
            f"Un modelo AR sugiere que {series_name} se explica por persistencia interna: el pasado reciente ayuda a predecir el presente."
        )

    if family == "MA":
        if ma_params:
            strongest_name = max(ma_params, key=lambda k: abs(ma_params[k]))
            strongest_value = ma_params[strongest_name]
            return (
                f"Un modelo MA sugiere que {series_name} responde sobre todo a choques transitorios que siguen repercutiendo "
                f"durante algunos trimestres. Económicamente, esto se parece a eventos inesperados que no desaparecen de inmediato, "
                f"sino que dejan una estela corta en la serie. El coeficiente más visible es {strongest_name} = {strongest_value:.3f}: "
                f"su magnitud indica qué tan fuerte es esa transmisión del choque pasado al periodo actual."
            )
        return (
            f"Un modelo MA sugiere que {series_name} está dominada por choques recientes más que por una persistencia fuerte de sus propios niveles pasados."
        )

    pieces = []
    if ar_params:
        strongest_ar = max(ar_params, key=lambda k: abs(ar_params[k]))
        pieces.append(f"el componente autorregresivo más fuerte es {strongest_ar} = {ar_params[strongest_ar]:.3f}")
    if ma_params:
        strongest_ma = max(ma_params, key=lambda k: abs(ma_params[k]))
        pieces.append(f"el componente de medias móviles más fuerte es {strongest_ma} = {ma_params[strongest_ma]:.3f}")

    detail = "; ".join(pieces) if pieces else "hay mezcla de persistencia y choques pasados"
    return (
        f"Un modelo ARIMA/ARMA sugiere que {series_name} combina dos fuerzas: persistencia interna y reacción a choques recientes. "
        f"Económicamente, eso quiere decir que la serie no solo arrastra parte de su propia dinámica, sino que también absorbe "
        f"eventos inesperados de periodos anteriores. En este ajuste, {detail}."
    )


def invert_ready_forecast_to_levels(
    original_log: pd.Series,
    ready_mean: pd.Series,
    ready_lower: pd.Series,
    ready_upper: pd.Series,
) -> pd.DataFrame:
    seasonal_hist = original_log - original_log.shift(4)

    def rebuild_path(ready_path: pd.Series) -> tuple[pd.Series, pd.Series]:
        log_map = {idx: float(val) for idx, val in original_log.dropna().items()}
        seasonal_map = {idx: float(val) for idx, val in seasonal_hist.dropna().items()}
        future_log = {}
        future_seasonal = {}

        for current_date, ready_value in ready_path.items():
            prev_quarter = current_date - pd.DateOffset(months=3)
            prev_year = current_date - pd.DateOffset(months=12)

            prev_seasonal = seasonal_map[prev_quarter]
            current_seasonal = float(ready_value) + prev_seasonal
            seasonal_map[current_date] = current_seasonal
            future_seasonal[current_date] = current_seasonal

            prev_log_year = log_map[prev_year]
            current_log = current_seasonal + prev_log_year
            log_map[current_date] = current_log
            future_log[current_date] = current_log

        future_log_series = pd.Series(future_log).sort_index()
        future_level_series = np.exp(future_log_series)
        return future_log_series, future_level_series

    log_mean, level_mean = rebuild_path(ready_mean)
    log_lower, level_lower = rebuild_path(ready_lower)
    log_upper, level_upper = rebuild_path(ready_upper)

    return pd.DataFrame(
        {
            "fecha": ready_mean.index,
            "ready_forecast": ready_mean.values,
            "ready_lower": ready_lower.values,
            "ready_upper": ready_upper.values,
            "log_forecast": log_mean.values,
            "log_lower": log_lower.values,
            "log_upper": log_upper.values,
            "level_forecast": level_mean.values,
            "level_lower": level_lower.values,
            "level_upper": level_upper.values,
        }
    )


def build_reconstruction_explainer(
    original_log: pd.Series,
    ready_mean: pd.Series,
) -> pd.DataFrame:
    seasonal_hist = original_log - original_log.shift(4)
    log_map = {idx: float(val) for idx, val in original_log.dropna().items()}
    seasonal_map = {idx: float(val) for idx, val in seasonal_hist.dropna().items()}
    rows = []

    for current_date, ready_value in ready_mean.items():
        prev_quarter = current_date - pd.DateOffset(months=3)
        prev_year = current_date - pd.DateOffset(months=12)

        prev_seasonal = seasonal_map[prev_quarter]
        current_seasonal = float(ready_value) + prev_seasonal
        seasonal_map[current_date] = current_seasonal

        prev_log_year = log_map[prev_year]
        current_log = current_seasonal + prev_log_year
        current_level = float(np.exp(current_log))
        log_map[current_date] = current_log

        rows.append(
            {
                "fecha": current_date,
                "serie_modelada": float(ready_value),
                "diff_estacional_previa": float(prev_seasonal),
                "nueva_diff_estacional": float(current_seasonal),
                "log_hace_4_trimestres": float(prev_log_year),
                "nuevo_log": float(current_log),
                "valor_absoluto": current_level,
            }
        )

    return pd.DataFrame(rows)


def make_forecast_chart(
    historical_dates: pd.Series,
    historical_values: pd.Series,
    forecast_dates: pd.Series,
    forecast_mean: pd.Series,
    forecast_lower: pd.Series,
    forecast_upper: pd.Series,
    title: str,
    y_title: str,
    color: str,
    value_format: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical_dates,
            y=historical_values,
            mode="lines",
            name="Historico",
            line=dict(color=color, width=2.2),
            hovertemplate=f"Fecha: %{{x|%Y-%m-%d}}<br>Valor: {value_format}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_upper,
            mode="lines",
            line=dict(color="rgba(214,40,40,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(214,40,40,0.18)",
            line=dict(color="rgba(214,40,40,0)"),
            name="IC 95%",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_mean,
            mode="lines+markers",
            name="Pronostico",
            line=dict(color="#D62828", width=2.4, dash="dash"),
            marker=dict(size=6),
            hovertemplate=f"Fecha: %{{x|%Y-%m-%d}}<br>Pronostico: {value_format}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_model_fit_chart(fitted_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted_df["fecha"],
            y=fitted_df["observado"],
            mode="lines+markers",
            name="Observado",
            line=dict(color="#1D3557", width=2.2),
            marker=dict(size=5),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Observado: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fitted_df["fecha"],
            y=fitted_df["ajustado"],
            mode="lines",
            name="Ajustado",
            line=dict(color="#D62828", width=2, dash="dash"),
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Ajustado: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=30, t=60, b=30),
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Fecha"),
        yaxis=dict(title="Serie lista para modelar"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


df = load_data()

st.title("Ingresos Corporativos MX")
st.markdown(
    "Explora ingresos trimestrales corporativos y luego analiza relaciones estadísticas "
    "entre las series usando una estructura por módulos."
)

with st.sidebar:
    st.header("Controles")
    selected_companies = st.multiselect(
        "Empresas",
        options=list(COMPANY_COLUMNS.keys()),
        default=["Bimbo", "Coca Cola"],
        max_selections=3,
    )
    show_logs = st.toggle("Logaritmos", value=False)
    show_differences = st.toggle("Diferencias", value=False)

if not selected_companies:
    st.info("Selecciona al menos una empresa para visualizar la serie.")
    st.stop()

absolute_df = transform_series(df, selected_companies, "Valores absolutos")
log_df = transform_series(df, selected_companies, "Logaritmos")
percent_df = transform_series(df, selected_companies, "Variacion porcentual")
log_diff_df = transform_series(df, selected_companies, "Diferencia logaritmica")
analysis_df = build_analysis_dataframe(df)
guide_tab, module1_tab, module2_tab, module3_tab = st.tabs(
    [
        "Guia - Receta",
        "Modulo 1 - Series",
        "Modulo 2 - Correlacion y Regresion",
        "Modulo 3 - Preparacion para Modelar",
    ]
)

with guide_tab:
    st.markdown(
        '<div class="section-title">Receta para modelar una serie de negocio</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Este tab resume la lógica completa de la app como una hoja de ruta.
            La idea es que el estudiante entienda <b>qué se hace, en qué orden y por qué</b>,
            antes de entrar al detalle técnico de cada módulo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    recipe_steps = pd.DataFrame(
        {
            "Paso": [
                "1. Definir la variable",
                "2. Ver la serie en niveles",
                "3. Transformar la serie",
                "4. Comparar con otras variables",
                "5. Revisar memoria temporal",
                "6. Proponer familias de modelo",
                "7. Seleccionar estadísticamente",
                "8. Validar residuos",
                "9. Pronosticar",
                "10. Regresar a escala de negocio",
            ],
            "Pregunta central": [
                "¿Qué quiero explicar o pronosticar?",
                "¿Hay tendencia, estacionalidad o cambios de escala?",
                "¿Qué transformación deja la serie más estable?",
                "¿La serie se mueve con competidores o con el PIB?",
                "¿La serie depende de sus propios rezagos?",
                "¿Parece más AR, MA o ARIMA?",
                "¿Qué modelo deja mejor balance entre ajuste y simplicidad?",
                "¿Los residuos quedaron razonablemente limpios?",
                "¿Qué viene en los próximos trimestres?",
                "¿Cómo traduzco el forecast a valores útiles para negocio?",
            ],
            "Dónde verlo en la app": [
                "Selección inicial de empresa/serie",
                "Modulo 1",
                "Modulo 1 y Modulo 3",
                "Modulo 2",
                "Modulo 3 - ACF, PACF y Ljung-Box",
                "Modulo 3 - patrones típicos",
                "Modulo 3 - selección de candidatos",
                "Modulo 3 - diagnóstico del modelo",
                "Modulo 3 - pronóstico",
                "Modulo 3 - reconstrucción a niveles",
            ],
        }
    )
    st.dataframe(recipe_steps, width="stretch", hide_index=True)

    st.markdown(
        '<div class="section-title">Cómo usar la app en orden</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Una ruta pedagógica recomendada sería:
            <br><br>
            <b>Primero</b>, entrar al Módulo 1 para entender la serie en niveles, logaritmos y variaciones.
            <br>
            <b>Segundo</b>, usar el Módulo 2 para discutir relaciones con otras variables y separar correlación de causalidad.
            <br>
            <b>Tercero</b>, entrar al Módulo 3 para transformar, diagnosticar, seleccionar el modelo y pronosticar.
            <br><br>
            Esa secuencia reproduce bastante bien la lógica con la que normalmente se enseña modelación de series en negocios:
            primero leer datos, luego entender relaciones y finalmente modelar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Errores comunes que la receta evita</div>',
        unsafe_allow_html=True,
    )
    common_mistakes = pd.DataFrame(
        {
            "Error común": [
                "Modelar directamente la serie en niveles",
                "Elegir un ARIMA solo por intuición",
                "Confiar solo en R-cuadrado o AIC",
                "Pronosticar y olvidar volver a escala de negocio",
                "Confundir correlación con causalidad",
            ],
            "Cómo lo corrige la app": [
                "Obliga a revisar tendencia, estacionalidad y transformaciones",
                "Usa patrones ACF/PACF y luego selección estadística",
                "Pide validar residuos con Ljung-Box",
                "Reconstruye paso a paso hasta valores absolutos",
                "Separa el módulo de regresión y lo interpreta con cautela",
            ],
        }
    )
    st.dataframe(common_mistakes, width="stretch", hide_index=True)

    st.markdown(
        '<div class="section-title">Mensaje final para clase</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La enseñanza clave no es solo “cómo correr un modelo”, sino <b>cómo pensar una serie</b>:
            leerla, transformarla, justificar el modelo, validarlo y traducir el resultado a una decisión de negocio.
            Esa es la lógica completa que articula todos los tabs de la app.
        </div>
        """,
        unsafe_allow_html=True,
    )

with module1_tab:
    st.markdown(
        '<div class="section-title">Resumen Inicial</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Primero mostramos los niveles observados de ingresos para cada empresa seleccionada.
            Estas métricas permiten identificar el último dato disponible y comparar si el cambio
            reciente entre trimestres fue positivo o negativo antes de aplicar cualquier transformación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(len(selected_companies))
    for col, company in zip(metric_cols, selected_companies):
        last_value = absolute_df[company].dropna().iloc[-1]
        previous_value = absolute_df[company].dropna().iloc[-2]
        delta = last_value - previous_value
        col.metric(
            label=f"{company} - ultimo trimestre",
            value=f"{last_value:,.0f}",
            delta=f"{delta:,.0f}",
        )

    st.markdown(
        '<div class="section-title">Serie en Niveles</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La gráfica principal conserva siempre los ingresos en términos absolutos.
            Esta es la referencia base del análisis, porque nos deja ver tamaño, tendencia y
            estacionalidad en la escala original de la variable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_logs:
        st.markdown(
            """
            <div class="info-banner">
                Al activar <b>Logaritmos</b>, añadimos una segunda gráfica con el logaritmo natural
                de los ingresos. Esta transformación comprime la escala y facilita comparar tasas de
                crecimiento relativas entre empresas de distinto tamaño.
            </div>
            """,
            unsafe_allow_html=True,
        )
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(
                make_chart(
                    absolute_df,
                    selected_companies,
                    "Valores absolutos",
                    chart_title="Ingresos trimestrales",
                ),
                width="stretch",
            )
        with chart_col2:
            st.plotly_chart(
                make_chart(
                    log_df,
                    selected_companies,
                    "Logaritmos",
                    chart_title="Logaritmo natural de los ingresos",
                ),
                width="stretch",
            )
    else:
        st.plotly_chart(
            make_chart(
                absolute_df,
                selected_companies,
                "Valores absolutos",
                chart_title="Ingresos trimestrales",
            ),
            width="stretch",
        )

    if show_differences:
        st.subheader("Variaciones")
        st.markdown(
            """
            <div class="info-banner">
                Las diferencias muestran cómo cambia la serie de un trimestre al siguiente.
                La variación porcentual expresa el cambio relativo usual, mientras que la diferencia
                logarítmica aproxima una tasa de crecimiento continua y suele ser muy útil en series financieras.
            </div>
            """,
            unsafe_allow_html=True,
        )
        diff_col1, diff_col2 = st.columns(2)
        with diff_col1:
            st.plotly_chart(
                make_chart(
                    percent_df,
                    selected_companies,
                    "Variacion porcentual",
                    chart_title="Variacion porcentual trimestral",
                ),
                width="stretch",
            )
        with diff_col2:
            st.plotly_chart(
                make_chart(
                    log_diff_df,
                    selected_companies,
                    "Diferencia logaritmica",
                    chart_title="Variacion logaritmica trimestral",
                ),
                width="stretch",
            )

    st.markdown(
        '<div class="section-title">Estructura de Datos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La tabla final reúne la serie original y sus principales transformaciones.
            Sirve para que los estudiantes vean cómo pasamos de los niveles observados a logaritmos,
            variaciones porcentuales y diferencias logarítmicas dentro de un panel trimestral.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Estructura de datos")
    table_df = build_data_table(
        df,
        selected_companies,
        absolute_df,
        log_df,
        percent_df,
        log_diff_df,
    )
    display_df = table_df.copy()
    display_df["fecha"] = display_df["fecha"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, width="stretch", hide_index=True)

with module2_tab:
    st.markdown(
        '<div class="section-title">Modulo 2: Correlacion y Regresion</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            En este módulo dejamos de trabajar con ingresos en niveles y pasamos a diferencias logarítmicas trimestrales.
            Esto es importante porque las series en niveles suelen tener tendencia y estacionalidad, lo que puede generar
            regresiones espurias. Al usar cambios logarítmicos, el análisis se enfoca en tasas de crecimiento comparables.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Paso 1 - Transformacion para inferencia</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La variable usada para inferencia es <b>Δlog(X)</b> multiplicada por 100. En términos prácticos, esta cantidad
            se interpreta como una tasa de crecimiento trimestral aproximada. La misma transformación se aplica a Bimbo,
            Coca Cola, Walmart y al PIB real, para que todas las series queden en una escala comparable.
        </div>
        """,
        unsafe_allow_html=True,
    )
    transform_preview = analysis_df.copy()
    transform_preview["fecha"] = transform_preview["fecha"].dt.strftime("%Y-%m-%d")
    st.dataframe(transform_preview, width="stretch", hide_index=True)

    st.markdown(
        '<div class="section-title">Paso 2 - Matriz de correlacion</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La matriz de correlación resume cómo se mueven conjuntamente las tasas de crecimiento trimestral.
            Un valor cercano a 1 indica asociación positiva fuerte; cercano a -1, asociación negativa; y alrededor de 0,
            poca relación lineal o monotónica, según el método elegido.
        </div>
        """,
        unsafe_allow_html=True,
    )

    corr_method = st.selectbox(
        "Metodo de correlacion",
        options=["pearson", "spearman"],
        index=0,
        help="Pearson mide asociación lineal; Spearman mide asociación monotónica basada en rangos.",
        key="corr_method_tab2",
    )
    corr_df = analysis_df[list(ANALYSIS_COLUMNS.keys())].corr(method=corr_method)
    st.plotly_chart(make_correlation_heatmap(corr_df), width="stretch")
    st.dataframe(corr_df.style.format("{:.3f}"), width="stretch")

    st.markdown(
        '<div class="section-title">Paso 3 - Regresion para explicar Coca Cola</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Ahora estimamos una regresión lineal simple donde la variable dependiente es la variación logarítmica
            trimestral de Coca Cola. La variable explicativa la eliges tú. El modelo estimado es:
            <br><br>
            <b>Δlog(Coca Cola)<sub>t</sub> = α + β Δlog(X)<sub>t</sub> + ε<sub>t</sub></b>
            <br><br>
            Reportamos errores robustos tipo HAC para ser un poco más cuidadosos con posible autocorrelación
            y heterocedasticidad en series trimestrales.
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = st.selectbox(
        "Variable explicativa para Coca Cola",
        options=["Bimbo", "Walmart", "Real GDP"],
        index=2,
        key="predictor_tab2",
    )
    reg_df, reg_model = fit_simple_regression(analysis_df, predictor)

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Observaciones", f"{int(reg_model.nobs)}")
    metric_b.metric("Beta", f"{reg_model.params[predictor]:.3f}")
    metric_c.metric("p-valor beta", f"{reg_model.pvalues[predictor]:.4f}")
    metric_d.metric("R-cuadrado", f"{reg_model.rsquared:.3f}")

    st.markdown(
        """
        <div class="info-banner">
            La pendiente β mide cuánto cambia, en promedio, el crecimiento trimestral de Coca Cola cuando la variable
            explicativa aumenta 1 punto porcentual en su diferencia logarítmica. El p-valor ayuda a evaluar si esa
            relación es estadísticamente distinguible de cero bajo este modelo simple.
        </div>
        """,
        unsafe_allow_html=True,
    )

    scatter_col, resid_col = st.columns(2)
    with scatter_col:
        st.plotly_chart(make_regression_scatter(reg_df, predictor), width="stretch")
    with resid_col:
        st.plotly_chart(make_residual_chart(reg_df), width="stretch")

    st.markdown(
        '<div class="section-title">Paso 4 - Resultados numericos e interpretacion</div>',
        unsafe_allow_html=True,
    )

    results_table = pd.DataFrame(
        {
            "Parametro": ["Intercepto", predictor],
            "Coeficiente": [reg_model.params["const"], reg_model.params[predictor]],
            "Error std. HAC": [reg_model.bse["const"], reg_model.bse[predictor]],
            "t-stat": [reg_model.tvalues["const"], reg_model.tvalues[predictor]],
            "p-valor": [reg_model.pvalues["const"], reg_model.pvalues[predictor]],
        }
    )
    st.dataframe(results_table.style.format({"Coeficiente": "{:.4f}", "Error std. HAC": "{:.4f}", "t-stat": "{:.3f}", "p-valor": "{:.4f}"}), width="stretch", hide_index=True)
    st.markdown(
        """
        <div class="info-banner">
            <b>Cómo leer los p-valores:</b> el p-valor del <b>intercepto</b> evalúa si, cuando la variable explicativa
            toma el valor cero, el crecimiento esperado de Coca Cola es estadísticamente distinto de cero. En muchos
            contextos su interpretación económica es limitada, pero sigue siendo útil como parte de la especificación.
            <br><br>
            El p-valor del <b>coeficiente beta</b> evalúa si existe evidencia estadística de una relación lineal entre
            la variación de la variable explicativa y la variación de Coca Cola. Si el p-valor es pequeño, rechazamos
            la hipótesis nula de que <b>β = 0</b> y concluimos que la asociación estimada es estadísticamente significativa
            dentro de este modelo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dw_stat = durbin_watson(reg_model.resid)
    interpretation = (
        f"Con esta especificación, un aumento de 1 punto porcentual en la variación logarítmica de {predictor} "
        f"se asocia en promedio con un cambio de {reg_model.params[predictor]:.3f} puntos porcentuales en la "
        f"variación logarítmica de Coca Cola."
    )
    st.markdown(
        f"""
        <div class="info-banner">
            <b>Lectura económica:</b> {interpretation}
            <br><br>
            <b>Durbin-Watson:</b> {dw_stat:.3f}. Valores cercanos a 2 sugieren que no hay una autocorrelación lineal
            fuerte en los residuos; valores muy alejados de 2 nos invitan a ser más cuidadosos con la inferencia.
            <br><br>
            <b>Nota docente:</b> este modelo es útil para ilustrar asociación contemporánea entre crecimientos trimestrales,
            no para afirmar causalidad. Para una etapa posterior podríamos extenderlo con más regresores, rezagos o pruebas
            formales de estacionariedad y cointegración.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predictor == "Real GDP":
        st.markdown(
            '<div class="section-title">Paso 5 - Regresion sin observaciones influyentes</div>',
            unsafe_allow_html=True,
        )
        filtered_reg_df, filtered_model, outliers_df, cook_threshold = fit_regression_without_influential_outliers(
            analysis_df,
            predictor,
        )
        st.markdown(
            f"""
            <div class="info-banner">
                Cuando usamos el PIB real, la pandemia introduce observaciones muy extremas que pueden dominar la pendiente
                estimada. Para mostrar ese efecto, identificamos <b>observaciones influyentes</b> con la distancia de Cook.
                Como regla práctica, marcamos como influyente cualquier punto con Cook's D mayor que <b>4/n = {cook_threshold:.4f}</b>.
                Después reestimamos la misma regresión excluyendo esas observaciones.
            </div>
            """,
            unsafe_allow_html=True,
        )

        filt_a, filt_b, filt_c, filt_d = st.columns(4)
        filt_a.metric("Observaciones filtradas", f"{int(filtered_model.nobs)}")
        filt_b.metric("Beta sin outliers", f"{filtered_model.params[predictor]:.3f}")
        filt_c.metric("p-valor beta", f"{filtered_model.pvalues[predictor]:.4f}")
        filt_d.metric("R-cuadrado", f"{filtered_model.rsquared:.3f}")

        filtered_scatter_col, filtered_resid_col = st.columns(2)
        with filtered_scatter_col:
            st.plotly_chart(make_regression_scatter(filtered_reg_df, predictor), width="stretch")
        with filtered_resid_col:
            st.plotly_chart(make_residual_chart(filtered_reg_df), width="stretch")

        filtered_results = pd.DataFrame(
            {
                "Parametro": ["Intercepto", predictor],
                "Coeficiente": [filtered_model.params["const"], filtered_model.params[predictor]],
                "Error std. HAC": [filtered_model.bse["const"], filtered_model.bse[predictor]],
                "t-stat": [filtered_model.tvalues["const"], filtered_model.tvalues[predictor]],
                "p-valor": [filtered_model.pvalues["const"], filtered_model.pvalues[predictor]],
            }
        )
        st.dataframe(
            filtered_results.style.format(
                {
                    "Coeficiente": "{:.4f}",
                    "Error std. HAC": "{:.4f}",
                    "t-stat": "{:.3f}",
                    "p-valor": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            """
            <div class="info-banner">
                La interpretación de los p-valores se mantiene igual en la regresión filtrada.
                El p-valor del intercepto indica si el crecimiento promedio de Coca Cola sería distinto de cero cuando
                el crecimiento del PIB es cero; el p-valor de beta indica si la relación lineal entre ambas variables
                sigue siendo estadísticamente distinta de cero después de retirar las observaciones extremas.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not outliers_df.empty:
            removed_display = outliers_df.copy()
            removed_display["fecha"] = removed_display["fecha"].dt.strftime("%Y-%m-%d")
            removed_display = removed_display.rename(
                columns={
                    "Coca Cola": "Coca Cola var_log",
                    predictor: f"{predictor} var_log",
                    "cook_d": "Cook's D",
                }
            )
            st.markdown(
                """
                <div class="info-banner">
                    La siguiente tabla muestra qué fechas fueron tratadas como observaciones influyentes.
                    En la práctica, este tipo de puntos no se elimina automáticamente en una investigación formal:
                    primero se interpreta económicamente por qué ocurrieron. Aquí los mostramos aparte para ilustrar
                    cómo cambia una regresión cuando algunos eventos extraordinarios dominan la muestra.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                removed_display.style.format(
                    {
                        "Coca Cola var_log": "{:.3f}",
                        f"{predictor} var_log": "{:.3f}",
                        "Cook's D": "{:.4f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        filtered_dw = durbin_watson(filtered_model.resid)
        filtered_interpretation = (
            f"Sin las observaciones influyentes, un aumento de 1 punto porcentual en la variación logarítmica de {predictor} "
            f"se asocia con un cambio promedio de {filtered_model.params[predictor]:.3f} puntos porcentuales en Coca Cola."
        )
        st.markdown(
            f"""
            <div class="info-banner">
                <b>Lectura económica sin outliers:</b> {filtered_interpretation}
                <br><br>
                <b>Durbin-Watson:</b> {filtered_dw:.3f}.
                <br><br>
                <b>Comparación docente:</b> este ejercicio deja ver cómo algunos episodios excepcionales, como la pandemia,
                pueden alterar de manera importante el signo, la magnitud o la significancia de una regresión.
                Comparar ambos resultados es útil para discutir sensibilidad del modelo y calidad de inferencia.
            </div>
            """,
            unsafe_allow_html=True,
        )

with module3_tab:
    st.markdown(
        '<div class="section-title">Modulo 3: Receta para dejar una serie lista para modelar</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Antes de ajustar un modelo temporal, conviene transformar la serie paso a paso.
            En series trimestrales como estas, normalmente buscamos reducir tres problemas:
            <b>escala creciente</b>, <b>estacionalidad</b> y <b>tendencia</b>. El objetivo de esta receta es llegar
            a una versión más estable y más adecuada para modelación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    modeling_series = st.selectbox(
        "Serie para preparar",
        options=list(ANALYSIS_COLUMNS.keys()),
        index=1,
        key="modeling_series_tab3",
    )
    recipe = build_modeling_recipe(df, ANALYSIS_COLUMNS[modeling_series])

    st.markdown(
        '<div class="section-title">Paso 1 - Serie original</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Empezamos en niveles para observar el comportamiento natural de la serie.
            Aquí suelen aparecer tendencia de largo plazo, cambios de escala y patrones estacionales.
            Esta visualización sirve como diagnóstico inicial, no como versión final para modelar.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_single_series_chart(
            df["fecha"],
            recipe["original"],
            f"{modeling_series} - serie original",
            "Nivel",
            COLORS[modeling_series],
        ),
        width="stretch",
    )

    st.markdown(
        '<div class="section-title">Paso 2 - Logaritmo natural</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            El logaritmo natural comprime la escala y vuelve más comparables los cambios relativos a lo largo del tiempo.
            Esta transformación es muy común en finanzas y macroeconomía porque una misma variación absoluta no tiene el
            mismo significado cuando la serie es pequeña que cuando es grande.
        </div>
        """,
        unsafe_allow_html=True,
    )
    log_col1, log_col2 = st.columns(2)
    with log_col1:
        st.plotly_chart(
            make_single_series_chart(
                df["fecha"],
                recipe["original"],
                f"{modeling_series} - niveles",
                "Nivel",
                COLORS[modeling_series],
            ),
            width="stretch",
        )
    with log_col2:
        st.plotly_chart(
            make_single_series_chart(
                df["fecha"],
                recipe["log"],
                f"{modeling_series} - logaritmo natural",
                "ln(serie)",
                COLORS[modeling_series],
            ),
            width="stretch",
        )

    st.markdown(
        '<div class="section-title">Paso 3 - Diferencia estacional trimestral</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Como trabajamos con datos trimestrales, una forma natural de atacar la estacionalidad es restar el valor
            observado cuatro trimestres atrás. En logaritmos, esta diferencia estacional compara cada trimestre con el
            mismo trimestre del año anterior.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_single_series_chart(
            df["fecha"],
            recipe["seasonal_diff"],
            f"{modeling_series} - diferencia estacional logarítmica",
            "Δ4 log(serie)",
            COLORS[modeling_series],
            add_zero_line=True,
        ),
        width="stretch",
    )

    st.markdown(
        '<div class="section-title">Paso 4 - Diferencia adicional para estabilizar tendencia</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Después de remover estacionalidad, aún puede quedar tendencia o persistencia en la media.
            Por eso aplicamos una diferencia adicional. El resultado final es una serie mucho más cercana a una
            dinámica estacionaria, que es justo lo que normalmente buscamos antes de estimar un ARIMA o SARIMA.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_single_series_chart(
            df["fecha"],
            recipe["ready"],
            f"{modeling_series} - serie final lista para modelar",
            "ΔΔ4 log(serie)",
            COLORS[modeling_series],
            add_zero_line=True,
        ),
        width="stretch",
    )

    st.markdown(
        '<div class="section-title">Paso 5 - Resumen de transformaciones</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La tabla resume cómo cambia la serie a medida que avanzamos en la receta.
            Incluimos una prueba ADF como referencia práctica: p-valores más bajos suelen ser más consistentes con
            estacionariedad. No es la única evidencia que importa, pero sí una señal útil para decidir con qué versión
            continuar al módulo de modelación.
        </div>
        """,
        unsafe_allow_html=True,
    )
    recipe_summary = make_recipe_summary_table(recipe, df)
    st.dataframe(
        recipe_summary.style.format(
            {
                "Media": "{:.4f}",
                "Desv. est.": "{:.4f}",
                "ADF estadistico": "{:.4f}",
                "ADF p-valor": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        """
        <div class="info-banner">
            <b>Serie recomendada para modelar ahora:</b> la última transformación, <b>ΔΔ4 log(serie)</b>.
            Esta versión suele ser una buena candidata para el siguiente paso del curso: revisar ACF, PACF,
            Ljung-Box y luego estimar un modelo temporal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    final_series_df = pd.DataFrame(
        {
            "fecha": df["fecha"],
            "serie_original": recipe["original"],
            "log_serie": recipe["log"],
            "diff_estacional_log": recipe["seasonal_diff"],
            "serie_lista_modelar": recipe["ready"],
        }
    ).copy()
    final_series_df["fecha"] = final_series_df["fecha"].dt.strftime("%Y-%m-%d")
    st.dataframe(final_series_df, width="stretch", hide_index=True)

    st.markdown(
        '<div class="section-title">Seccion 2 - Dependencia temporal paso a paso</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Ahora tomamos la serie final <b>lista para modelar</b> y revisamos si todavía conserva memoria temporal.
            Esa memoria es justamente la materia prima de los modelos ARIMA: si una serie depende de sus rezagos,
            podemos intentar modelarla con sus propios valores pasados.
        </div>
        """,
        unsafe_allow_html=True,
    )

    diagnostics = compute_time_dependence_diagnostics(recipe["ready"], max_lag=12)

    st.markdown(
        '<div class="section-title">Paso 6 - Funcion de Autocorrelacion (ACF)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="info-banner">
            La <b>ACF</b> (función de autocorrelación) responde una pregunta muy simple:
            <b>qué tanto se parece la serie actual a sus propios valores pasados</b>.
            <br><br>
            Si la barra del rezago 1 es alta, significa que el valor de este trimestre se mueve de forma parecida
            al del trimestre inmediatamente anterior. Si la barra del rezago 4 es alta, significa que la serie todavía
            se parece bastante a lo que ocurría hace un año, lo cual puede revelar persistencia o estacionalidad.
            <br><br>
            En otras palabras, la ACF muestra la <b>memoria total</b> de la serie en distintos rezagos, sin separar
            si esa relación viene directamente de ese rezago o si pasa a través de otros rezagos intermedios.
            <br><br>
            Las líneas punteadas rojas marcan una banda de referencia aproximada de ±1.96/sqrt(n), donde
            n = {diagnostics["n"]}. Barras que superan claramente esa banda sugieren autocorrelación relevante
            en ese rezago.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_correlation_lag_chart(
            diagnostics["acf"],
            f"ACF - {modeling_series} lista para modelar",
            "Autocorrelacion",
            diagnostics["conf"],
        ),
        width="stretch",
    )

    st.markdown(
        '<div class="section-title">Paso 7 - Funcion de Autocorrelacion Parcial (PACF)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La <b>PACF</b> (función de autocorrelación parcial) intenta responder una pregunta más fina:
            <b>qué parte de la relación con un rezago específico es directa</b>, una vez quitamos el efecto
            de los rezagos más cortos.
            <br><br>
            Por ejemplo, si el rezago 2 parece importante en la ACF, puede ser porque realmente el segundo rezago
            importa por sí mismo, o simplemente porque el rezago 1 ya arrastra información. La PACF ayuda a separar
            esas dos historias.
            <br><br>
            Pedagógicamente, una forma simple de pensarlo es esta:
            <br>
            <b>ACF = memoria total observada</b>
            <br>
            <b>PACF = memoria directa de cada rezago</b>
            <br><br>
            Esa comparación suele ser muy útil para pensar si la serie se parece más a una dinámica autorregresiva
            o a una dinámica de medias móviles.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_correlation_lag_chart(
            diagnostics["pacf"],
            f"PACF - {modeling_series} lista para modelar",
            "Autocorrelacion parcial",
            diagnostics["conf"],
        ),
        width="stretch",
    )

    st.markdown(
        '<div class="section-title">Paso 8 - Prueba de Ljung-Box</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La prueba de <b>Ljung-Box</b> resume de forma conjunta si varios rezagos tienen autocorrelación.
            La hipótesis nula dice que la serie no presenta autocorrelación hasta el rezago evaluado.
            <br><br>
            Si el p-valor es pequeño, rechazamos esa hipótesis y concluimos que todavía hay dependencia temporal
            aprovechable para modelar. Si el p-valor es grande, la serie se parece más a un ruido sin mucha estructura.
        </div>
        """,
        unsafe_allow_html=True,
    )
    ljung_box_df = diagnostics["ljung_box"].copy()
    st.dataframe(
        ljung_box_df.style.format({"Q": "{:.4f}", "pvalue": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    significant_lags = ljung_box_df.loc[ljung_box_df["pvalue"] < 0.05, "lag"].tolist()
    if significant_lags:
        interpretation_text = (
            f"Hay evidencia de autocorrelación conjunta al menos hasta los rezagos {significant_lags}. "
            "Eso sugiere que la serie todavía tiene estructura temporal que un modelo puede intentar capturar."
        )
    else:
        interpretation_text = (
            "No aparece evidencia fuerte de autocorrelación conjunta al 5% en los rezagos evaluados. "
            "Eso sugiere una serie más cercana a ruido blanco en esta etapa."
        )

    st.markdown(
        f"""
        <div class="info-banner">
            <b>Lectura integrada:</b> {interpretation_text}
            <br><br>
            En la práctica, el siguiente paso consiste en combinar lo que ves en la ACF, la PACF y Ljung-Box
            para proponer un modelo inicial. Si la dependencia temporal sigue viva, entonces ya estamos listos
            para intentar un ARIMA o SARIMA en la siguiente sección.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Seccion 3 - Introduccion a modelos AR, MA y ARIMA</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Una vez tenemos una serie más estable, el siguiente paso es pensar cómo describir su dinámica.
            Aquí aparecen tres familias clásicas:
            <br><br>
            <b>AR(p)</b>: la serie se explica con sus propios rezagos.
            <br>
            <b>MA(q)</b>: la serie se explica con errores o choques pasados.
            <br>
            <b>ARIMA(p,d,q)</b>: combina ambas ideas y además permite diferenciar la serie cuando todavía no está lista.
            <br><br>
            En nuestro caso, como ya construimos una serie transformada y lista para modelar, trabajaremos con
            <b>d = 0</b>. Es decir, vamos a estimar modelos sobre la serie ya diferenciada fuera del modelo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Paso 9 - Como se reconocen tipicamente estos modelos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Antes de estimar, conviene mirar patrones típicos en ACF y PACF:
            <br><br>
            <b>AR(p):</b> la PACF suele mostrar un corte relativamente claro en pocos rezagos, mientras que la ACF decae de forma gradual.
            <br>
            <b>MA(q):</b> la ACF suele cortar en pocos rezagos, mientras la PACF decae gradualmente.
            <br>
            <b>ARIMA / ARMA:</b> tanto ACF como PACF suelen decaer sin un corte perfectamente limpio, porque hay mezcla de memoria directa y de choques pasados.
            <br><br>
            Esta parte es importante: <b>ACF y PACF no “adivinan” el modelo</b>. Solo ayudan a proponer candidatos razonables.
            La decisión final debe combinar teoría, criterios de información y diagnóstico de residuos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pattern_table = pd.DataFrame(
        {
            "Familia": ["AR(p)", "MA(q)", "ARIMA/ARMA"],
            "Idea central": [
                "La serie depende de sus propios rezagos",
                "La serie depende de choques pasados",
                "La serie combina rezagos y choques pasados",
            ],
            "Patron ACF tipico": [
                "Decae gradualmente",
                "Corta en pocos rezagos",
                "Suele decaer sin corte limpio",
            ],
            "Patron PACF tipico": [
                "Corta en pocos rezagos",
                "Decae gradualmente",
                "Suele decaer sin corte limpio",
            ],
        }
    )
    st.dataframe(pattern_table, width="stretch", hide_index=True)

    pattern_col1, pattern_col2, pattern_col3 = st.columns(3)
    with pattern_col1:
        st.plotly_chart(make_stylized_pattern_chart("AR"), width="stretch")
    with pattern_col2:
        st.plotly_chart(make_stylized_pattern_chart("MA"), width="stretch")
    with pattern_col3:
        st.plotly_chart(make_stylized_pattern_chart("ARIMA"), width="stretch")

    st.markdown(
        '<div class="section-title">Paso 10 - Inspeccion visual de la serie elegida</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Antes de lanzar una búsqueda automática, conviene volver a mirar la ACF y la PACF de la serie concreta
            que estamos modelando. El objetivo aquí no es tomar la decisión final, sino construir una <b>intuición inicial</b>:
            si la ACF parece cortar rápido y la PACF decae, podríamos pensar en MA; si ocurre al revés, en AR;
            si ambas decaen, un ARIMA/ARMA puede ser más razonable.
        </div>
        """,
        unsafe_allow_html=True,
    )

    inspection_col1, inspection_col2 = st.columns(2)
    with inspection_col1:
        st.plotly_chart(
            make_correlation_lag_chart(
                diagnostics["acf"],
                f"ACF observada - {modeling_series}",
                "Autocorrelacion",
                diagnostics["conf"],
            ),
            width="stretch",
        )
    with inspection_col2:
        st.plotly_chart(
            make_correlation_lag_chart(
                diagnostics["pacf"],
                f"PACF observada - {modeling_series}",
                "Autocorrelacion parcial",
                diagnostics["conf"],
            ),
            width="stretch",
        )

    suggested_family = suggest_family_from_diagnostics(
        diagnostics["acf"],
        diagnostics["pacf"],
        diagnostics["conf"],
    )
    suggestion_text = {
        "AR": "La PACF parece concentrar los rezagos más importantes y la ACF se va apagando gradualmente. Eso sugiere empezar mirando candidatos AR.",
        "MA": "La ACF parece cortar relativamente rápido y la PACF se disipa con más lentitud. Eso sugiere empezar mirando candidatos MA.",
        "ARIMA": "Ni la ACF ni la PACF muestran un corte especialmente limpio. Eso sugiere una dinámica mixta, así que conviene considerar candidatos ARIMA/ARMA.",
    }
    st.markdown(
        f"""
        <div class="info-banner">
            <b>Lectura intuitiva para esta serie:</b> {suggestion_text[suggested_family]}
            <br><br>
            Esta lectura todavía no decide el modelo final. Solo nos da un punto de partida razonable antes de aplicar
            selección estadística formal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Paso 11 - Seleccion estadistica de candidatos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Ahora sí pasamos a una selección más sistemática. La lógica será:
            <br><br>
            1. Estimar varios modelos AR, MA y ARIMA con órdenes pequeños.
            <br>
            2. Compararlos con <b>AIC</b> y <b>BIC</b>.
            <br>
            3. Revisar si sus residuos pasan razonablemente la prueba de <b>Ljung-Box</b>.
            <br><br>
            Así evitamos elegir “a ojo” y convertimos la identificación del modelo en un proceso más profesional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    order_col1, order_col2 = st.columns(2)
    with order_col1:
        max_p_search = st.slider(
            "Maximo p a explorar",
            min_value=1,
            max_value=6,
            value=4,
            key="max_p_search_tab3",
        )
    with order_col2:
        max_q_search = st.slider(
            "Maximo q a explorar",
            min_value=1,
            max_value=6,
            value=4,
            key="max_q_search_tab3",
        )

    ready_series_clean = recipe["ready"].dropna()
    ready_series_indexed = pd.Series(ready_series_clean.values, index=df.loc[ready_series_clean.index, "fecha"], name=modeling_series)
    candidate_models_df = search_candidate_models(ready_series_indexed, max_p=max_p_search, max_q=max_q_search)

    if candidate_models_df.empty:
        st.warning("No fue posible estimar candidatos con la configuración actual.")
        st.stop()

    family_top_df = top_candidates_by_family(candidate_models_df, top_n=3)
    st.markdown(
        """
        <div class="info-banner">
            En lugar de mezclar todos los modelos en una sola lista, primero comparamos a los mejores dentro de cada familia.
            Eso permite ver si la serie parece describirse mejor con rezagos puros (AR), con choques pasados (MA),
            o con una combinación de ambos (ARIMA).
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        family_top_df.style.format(
            {
                "aic": "{:.2f}",
                "bic": "{:.2f}",
                "resid_min_pvalue": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Paso 12 - Regla de seleccion y modelo recomendado</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La regla de selección que seguiremos es:
            <br><br>
            1. Preferir modelos cuyos residuos pasen mejor la prueba de Ljung-Box.
            <br>
            2. Entre esos, preferir menor AIC y menor BIC.
            <br>
            3. Si dos modelos son muy parecidos, preferir el más simple.
            <br><br>
            Con esa lógica, la app propone un modelo recomendado y luego te deja inspeccionarlo en detalle.
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_candidate = candidate_models_df.iloc[0]
    recommendation_text = (
        f"El modelo recomendado es {best_candidate['familia']} {best_candidate['orden']} "
        f"porque combina un buen comportamiento de residuos, AIC={best_candidate['aic']:.2f} "
        f"y BIC={best_candidate['bic']:.2f}."
    )
    rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
    rec_col1.metric("Familia recomendada", str(best_candidate["familia"]))
    rec_col2.metric("Orden recomendado", str(best_candidate["orden"]))
    rec_col3.metric("AIC", f"{best_candidate['aic']:.2f}")
    rec_col4.metric("Pasa Ljung-Box", "Si" if bool(best_candidate["pasa_ljung_box"]) else "No")
    st.markdown(
        f"""
        <div class="info-banner">
            <b>Modelo recomendado:</b> {recommendation_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    inspect_options_df = family_top_df.copy()
    selected_label = st.selectbox(
        "Modelo a inspeccionar en detalle",
        options=[
            f"{row.familia} {row.orden} | AIC={row.aic:.2f} | BIC={row.bic:.2f} | pasa LB={bool(row.pasa_ljung_box)}"
            for _, row in inspect_options_df.iterrows()
        ],
        index=0,
        key="selected_candidate_tab3",
    )
    selected_idx = [
        f"{row.familia} {row.orden} | AIC={row.aic:.2f} | BIC={row.bic:.2f} | pasa LB={bool(row.pasa_ljung_box)}"
        for _, row in inspect_options_df.iterrows()
    ].index(selected_label)
    selected_candidate = inspect_options_df.iloc[selected_idx]
    model_result, fitted_df, residual_lb_df, used_order = fit_univariate_model(
        ready_series_indexed,
        selected_candidate["familia"],
        int(selected_candidate["p"]),
        int(selected_candidate["q"]),
    )

    st.markdown(
        '<div class="section-title">Paso 13 - Diagnostico del modelo seleccionado</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="info-banner">
            El mejor candidato global según el ranking actual es <b>{best_candidate["familia"]} {best_candidate["orden"]}</b>.
            Ese ranking prioriza modelos con mejor comportamiento en residuos y luego menor AIC/BIC.
            <br><br>
            Abajo puedes inspeccionar en detalle cualquiera de los mejores modelos por familia. La idea ya no es “adivinar”
            un orden, sino revisar candidatos razonables encontrados por un proceso sistemático.
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_metric_a, model_metric_b, model_metric_c, model_metric_d = st.columns(4)
    model_metric_a.metric("Familia", str(selected_candidate["familia"]))
    model_metric_b.metric("Orden usado", str(used_order))
    model_metric_c.metric("AIC", f"{model_result.aic:.2f}")
    model_metric_d.metric("BIC", f"{model_result.bic:.2f}")
    sigma_col1, sigma_col2, sigma_col3 = st.columns(3)
    sigma_col1.metric("Sigma2", f"{float(model_result.params.get('sigma2', np.nan)):.4f}")
    sigma_col2.metric("Min p-valor residuos", f"{float(selected_candidate['resid_min_pvalue']):.4f}")
    sigma_col3.metric("Rechazos Ljung-Box", f"{int(selected_candidate['resid_rechazos_lb'])}")

    fit_col1, fit_col2 = st.columns(2)
    with fit_col1:
        st.plotly_chart(
            make_model_fit_chart(
                fitted_df,
                f"{selected_candidate['familia']} {used_order} - ajuste sobre la serie lista para modelar",
            ),
            width="stretch",
        )
    with fit_col2:
        st.plotly_chart(make_residual_chart(fitted_df), width="stretch")

    st.markdown(
        '<div class="section-title">Paso 14 - Parametros estimados</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La tabla siguiente resume los coeficientes del modelo. Los parámetros AR suelen interpretarse como
            persistencia de la serie; los parámetros MA, como efecto de choques pasados. El p-valor ayuda a ver
            si cada parámetro aparece estadísticamente distinto de cero en esta especificación. Aquí ya estamos
            leyendo un modelo que llegó a esta etapa después de un filtro estadístico previo, no de una elección manual.
        </div>
        """,
        unsafe_allow_html=True,
    )
    param_table = pd.DataFrame(
        {
            "Parametro": model_result.params.index,
            "Coeficiente": model_result.params.values,
            "Error std.": model_result.bse,
            "z-stat": model_result.tvalues,
            "p-valor": model_result.pvalues,
        }
    )
    st.dataframe(
        param_table.style.format(
            {
                "Coeficiente": "{:.4f}",
                "Error std.": "{:.4f}",
                "z-stat": "{:.3f}",
                "p-valor": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        f"""
        <div class="info-banner">
            <b>Lectura económica del modelo:</b> {economic_model_interpretation(model_result, selected_candidate["familia"], modeling_series)}
            <br><br>
            <b>Cómo leer los coeficientes:</b> en un modelo AR, los coeficientes muestran cuánta persistencia propia tiene la serie.
            En un modelo MA, muestran cuánto siguen pesando los choques recientes. En un modelo mixto ARIMA/ARMA, ambos tipos
            de coeficientes conviven: unos capturan inercia y otros capturan la propagación de perturbaciones pasadas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Paso 15 - Prueba sobre residuos del modelo</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Después de ajustar un modelo, no basta con mirar el AIC o la gráfica. También queremos saber
            si los residuos quedaron parecidos a ruido blanco. Si todavía muestran autocorrelación,
            eso sugiere que el modelo no capturó toda la dinámica temporal de la serie.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        residual_lb_df.style.format({"Q": "{:.4f}", "pvalue": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    residual_significant = residual_lb_df.loc[residual_lb_df["pvalue"] < 0.05, "lag"].tolist()
    if residual_significant:
        model_diagnostic_text = (
            f"Los residuos todavía muestran evidencia de autocorrelación en rezagos como {residual_significant}. "
            "Eso sugiere que el modelo actual podría mejorarse."
        )
    else:
        model_diagnostic_text = (
            "No aparece evidencia fuerte de autocorrelación en los residuos al 5%. "
            "Eso es una buena señal: el modelo parece haber capturado una parte importante de la dinámica temporal."
        )

    st.markdown(
        f"""
        <div class="info-banner">
            <b>Lectura final del modelo:</b> {model_diagnostic_text}
            <br><br>
            <b>Idea docente:</b> aquí la selección ya no depende de “probar porque sí”, sino de un flujo más serio:
            patrones sugeridos por ACF/PACF, búsqueda de candidatos, comparación con AIC/BIC y validación con residuos.
            Esa es la lógica que conecta identificación, estimación y validación.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Ver tabla completa de candidatos"):
        st.dataframe(
            candidate_models_df.style.format(
                {
                    "aic": "{:.2f}",
                    "bic": "{:.2f}",
                    "resid_min_pvalue": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        '<div class="section-title">Seccion 4 - Pronostico y regreso a escala de negocio</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            El modelo se estimó sobre la serie transformada, así que el pronóstico sale primero en esa misma escala.
            Después debemos reconstruir paso a paso la serie para volver a una escala interpretable para negocio.
            En este caso, el camino es:
            <br><br>
            <b>serie lista para modelar -> diferencia estacional logarítmica -> logaritmo -> valores absolutos</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    forecast_steps = st.selectbox(
        "Horizonte de pronostico",
        options=[4, 8],
        index=0,
        key="forecast_steps_tab3",
    )
    forecast_result = model_result.get_forecast(steps=forecast_steps)
    conf_int = forecast_result.conf_int(alpha=0.05)
    future_dates = pd.date_range(
        start=ready_series_indexed.index[-1] + pd.DateOffset(months=3),
        periods=forecast_steps,
        freq="QS",
    )
    ready_forecast_df = pd.DataFrame(
        {
            "fecha": future_dates,
            "mean": forecast_result.predicted_mean.values,
            "lower": conf_int.iloc[:, 0].values,
            "upper": conf_int.iloc[:, 1].values,
        }
    )
    ready_forecast_df = ready_forecast_df.set_index("fecha")

    level_forecast_df = invert_ready_forecast_to_levels(
        recipe["log"].set_axis(df["fecha"]),
        ready_forecast_df["mean"],
        ready_forecast_df["lower"],
        ready_forecast_df["upper"],
    )
    reconstruction_df = build_reconstruction_explainer(
        recipe["log"].set_axis(df["fecha"]),
        ready_forecast_df["mean"],
    )

    st.markdown(
        '<div class="section-title">Paso 16 - Pronostico en la serie modelada</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Primero mostramos el pronóstico exactamente en la escala en que fue estimado el modelo.
            Esto es importante porque aquí es donde la inferencia estadística del AR, MA o ARIMA es más directa.
            La franja sombreada representa un intervalo de confianza aproximado al 95%.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_forecast_chart(
            ready_series_indexed.index.to_series(),
            ready_series_indexed,
            level_forecast_df["fecha"],
            level_forecast_df["ready_forecast"],
            level_forecast_df["ready_lower"],
            level_forecast_df["ready_upper"],
            "Pronostico en la serie lista para modelar",
            "ΔΔ4 log(serie)",
            COLORS[modeling_series],
            "%{y:.3f}",
        ),
        width="stretch",
    )

    st.markdown(
        '<div class="section-title">Paso 17 - Reconstruccion hacia logaritmos y niveles</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            Para volver a una escala útil para negocio, reconstruimos la serie hacia atrás:
            <br><br>
            1. Sumamos el pronóstico a la diferencia estacional acumulada.
            <br>
            2. Recuperamos el <b>logaritmo pronosticado</b>.
            <br>
            3. Aplicamos la exponencial para regresar a <b>valores absolutos</b>.
            <br><br>
            Esta reconstrucción usa la estructura temporal que ya habíamos construido durante la preparación de la serie.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            <b>Idea clave para no perderse:</b>
            <br><br>
            El modelo no pronostica directamente el ingreso de negocio. Pronostica la serie transformada
            <b>ΔΔ4 log(serie)</b>. Por eso debemos deshacer la receta en orden inverso:
            <br><br>
            1. Partimos del pronóstico de <b>ΔΔ4 log</b>.
            <br>
            2. Le sumamos la diferencia estacional previa para recuperar <b>Δ4 log</b>.
            <br>
            3. Le sumamos el logaritmo de hace cuatro trimestres para recuperar <b>log(serie)</b>.
            <br>
            4. Aplicamos exponencial para recuperar el <b>valor absoluto</b>.
            <br><br>
            En otras palabras: estamos <b>deshaciendo las transformaciones</b> exactamente en el orden inverso en que las construimos.
        </div>
        """,
        unsafe_allow_html=True,
    )
    recon_col1, recon_col2 = st.columns(2)
    with recon_col1:
        st.plotly_chart(
            make_forecast_chart(
                df["fecha"],
                recipe["log"],
                level_forecast_df["fecha"],
                level_forecast_df["log_forecast"],
                level_forecast_df["log_lower"],
                level_forecast_df["log_upper"],
                "Pronostico reconstruido en logaritmos",
                "log(serie)",
                COLORS[modeling_series],
                "%{y:.3f}",
            ),
            width="stretch",
        )
    with recon_col2:
        st.plotly_chart(
            make_forecast_chart(
                df["fecha"],
                recipe["original"],
                level_forecast_df["fecha"],
                level_forecast_df["level_forecast"],
                level_forecast_df["level_lower"],
                level_forecast_df["level_upper"],
                "Pronostico final en valores absolutos",
                "Valor absoluto",
                COLORS[modeling_series],
                "%{y:,.0f}",
            ),
            width="stretch",
        )

    st.markdown(
        '<div class="section-title">Paso 18 - Reconstruccion guiada trimestre por trimestre</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La tabla siguiente muestra exactamente cómo se reconstruye cada pronóstico.
            Léela de izquierda a derecha:
            <br><br>
            <b>serie modelada</b> -> <b>nueva diferencia estacional</b> -> <b>nuevo log</b> -> <b>valor absoluto</b>
            <br><br>
            Así el estudiante puede seguir la cadena de cálculo sin tratar la reconstrucción como una “caja negra”.
        </div>
        """,
        unsafe_allow_html=True,
    )
    recon_display = reconstruction_df.copy()
    recon_display["fecha"] = recon_display["fecha"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        recon_display.style.format(
            {
                "serie_modelada": "{:.4f}",
                "diff_estacional_previa": "{:.4f}",
                "nueva_diff_estacional": "{:.4f}",
                "log_hace_4_trimestres": "{:.4f}",
                "nuevo_log": "{:.4f}",
                "valor_absoluto": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    if not reconstruction_df.empty:
        example_row = reconstruction_df.iloc[0]
        st.markdown(
            f"""
            <div class="info-banner">
                <b>Ejemplo con el primer trimestre pronosticado:</b>
                <br><br>
                1. El modelo entrega <b>ΔΔ4 log = {example_row['serie_modelada']:.4f}</b>.
                <br>
                2. La diferencia estacional previa era <b>{example_row['diff_estacional_previa']:.4f}</b>.
                Entonces la nueva diferencia estacional queda en
                <b>{example_row['serie_modelada']:.4f} + {example_row['diff_estacional_previa']:.4f} = {example_row['nueva_diff_estacional']:.4f}</b>.
                <br>
                3. El logaritmo observado hace cuatro trimestres era <b>{example_row['log_hace_4_trimestres']:.4f}</b>.
                Entonces el nuevo logaritmo proyectado es
                <b>{example_row['nueva_diff_estacional']:.4f} + {example_row['log_hace_4_trimestres']:.4f} = {example_row['nuevo_log']:.4f}</b>.
                <br>
                4. Finalmente, al aplicar exponencial obtenemos el valor de negocio proyectado:
                <b>exp({example_row['nuevo_log']:.4f}) = {example_row['valor_absoluto']:,.0f}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Paso 19 - Tabla final de estimaciones</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-banner">
            La tabla siguiente resume el pronóstico en todas las escalas relevantes.
            Didácticamente, esto ayuda a ver que el modelo trabaja sobre una transformación estadística,
            pero la decisión de negocio termina necesitando valores en la escala original.
        </div>
        """,
        unsafe_allow_html=True,
    )
    forecast_display = level_forecast_df.copy()
    forecast_display["fecha"] = forecast_display["fecha"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        forecast_display.style.format(
            {
                "ready_forecast": "{:.4f}",
                "ready_lower": "{:.4f}",
                "ready_upper": "{:.4f}",
                "log_forecast": "{:.4f}",
                "log_lower": "{:.4f}",
                "log_upper": "{:.4f}",
                "level_forecast": "{:,.0f}",
                "level_lower": "{:,.0f}",
                "level_upper": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Paso 20 - Conclusion final del modulo</div>',
        unsafe_allow_html=True,
    )
    final_business_text = (
        f"Para {modeling_series}, el proceso completo sugiere un modelo {selected_candidate['familia']} {used_order} "
        f"como candidato razonable. La serie se preparó para remover escala, estacionalidad y tendencia; luego se identificó "
        f"un modelo con base en ACF/PACF, criterios de información y residuos; y finalmente se generó un pronóstico que "
        f"volvimos a llevar hasta valores absolutos para interpretación económica."
    )
    st.markdown(
        f"""
        <div class="info-banner">
            <b>Conclusion del modulo:</b> {final_business_text}
            <br><br>
            <b>Mensaje docente clave:</b> en series temporales, modelar bien no significa solo “pronosticar”.
            Significa transformar correctamente la serie, justificar el modelo, validar residuos y luego regresar
            el resultado a la escala donde realmente se toman decisiones.
        </div>
        """,
        unsafe_allow_html=True,
    )
