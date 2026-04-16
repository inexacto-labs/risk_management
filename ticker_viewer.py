import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuración de la página ─────────────────────────────────────────────
st.set_page_config(
    page_title="Visor de Precios",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Visor de Precios Históricos")
st.markdown("Selecciona hasta **3 tickers** y un **rango de fechas**.")

TODAY = datetime.date.today()
MIN_DATE = datetime.date(2010, 1, 1)
COLORS = ["#2196F3", "#FF5722", "#4CAF50"]
COLORS_FILL = ["rgba(33,150,243,0.12)", "rgba(255,87,34,0.12)", "rgba(76,175,80,0.12)"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

def make_price_chart(data: dict, title: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    for i, (ticker, series) in enumerate(data.items()):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=ticker,
            line=dict(color=COLORS[i], width=1.8),
            hovertemplate=f"<b>{ticker}</b><br>Fecha: %{{x|%Y-%m-%d}}<br>Precio: $%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
        yaxis=dict(title="Precio ajustado (USD)", tickprefix="$"),
        xaxis=dict(title="Fecha"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return fig


def make_returns_chart(data: dict, title: str, height: int = 380) -> go.Figure:
    fig = go.Figure()
    for i, (ticker, series) in enumerate(data.items()):
        log_ret = np.log(series / series.shift(1)).dropna() * 100
        fig.add_trace(go.Scatter(
            x=log_ret.index, y=log_ret.values,
            mode="lines", name=ticker,
            line=dict(color=COLORS[i], width=1.2),
            hovertemplate=f"<b>{ticker}</b><br>Fecha: %{{x|%Y-%m-%d}}<br>Retorno log: %{{y:.3f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified",
        yaxis=dict(title="Retorno log diario (%)", ticksuffix="%"),
        xaxis=dict(title="Fecha"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return fig


def make_single_price(ticker: str, series, color: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=ticker,
        line=dict(color=color, width=1.8),
        hovertemplate=f"<b>{ticker}</b><br>Fecha: %{{x|%Y-%m-%d}}<br>Precio: $%{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{ticker} — Precio de Cierre Ajustado", font=dict(size=13)),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified", showlegend=False,
        yaxis=dict(title="Precio ajustado (USD)", tickprefix="$"),
        xaxis=dict(title="Fecha"),
    )
    return fig


def make_single_returns(ticker: str, series, color: str, fill_color: str, height: int = 300) -> go.Figure:
    log_ret = np.log(series / series.shift(1)).dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_ret.index, y=log_ret.values,
        mode="lines", name=ticker,
        line=dict(color=color, width=1.2),
        fill="tozeroy", fillcolor=fill_color,
        hovertemplate=f"<b>{ticker}</b><br>Fecha: %{{x|%Y-%m-%d}}<br>Retorno log: %{{y:.3f}}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(
        title=dict(text=f"{ticker} — Retorno Logarítmico Diario", font=dict(size=13)),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified", showlegend=False,
        yaxis=dict(title="Retorno log diario (%)", ticksuffix="%"),
        xaxis=dict(title="Fecha"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — VaR
# ══════════════════════════════════════════════════════════════════════════════

def var_historico(port_returns: pd.Series, monto: float, confianza: float):
    """Devuelve (VaR, percentil_usado, retornos_ordenados)."""
    alpha = 1 - confianza
    var_pct = np.percentile(port_returns, alpha * 100)   # negativo
    var_dollar = abs(var_pct / 100) * monto
    return var_dollar, var_pct, port_returns.sort_values()


def var_varianza_covarianza(log_rets: pd.DataFrame, pesos: np.ndarray,
                             monto: float, confianza: float):
    """Devuelve (VaR, z, sigma_port, cov_matrix, medias, stds)."""
    medias = log_rets.mean()
    stds   = log_rets.std()
    cov    = log_rets.cov()
    w      = np.array(pesos)
    sigma2_port = w @ cov.values @ w
    sigma_port  = np.sqrt(sigma2_port)
    z           = stats.norm.ppf(1 - confianza)   # negativo
    var_dollar  = abs(z) * (sigma_port / 100) * monto
    return var_dollar, z, sigma_port, cov, medias, stds


def make_histogram_normalidad(data: dict, height: int = 420) -> go.Figure:
    """
    Histograma de densidad de retornos log diarios por ticker con:
      - curva normal ajustada
      - sombreado de colas empíricas vs normales
    """
    n = len(data)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=list(data.keys()),
        horizontal_spacing=0.08,
    )

    for i, (ticker, series) in enumerate(data.items(), start=1):
        log_ret = np.log(series / series.shift(1)).dropna() * 100
        mu, sigma = log_ret.mean(), log_ret.std()
        x_range = np.linspace(mu - 4.5 * sigma, mu + 4.5 * sigma, 400)
        y_normal = stats.norm.pdf(x_range, mu, sigma)

        # Histograma normalizado a densidad
        fig.add_trace(go.Histogram(
            x=log_ret.values,
            histnorm="probability density",
            nbinsx=60,
            name=ticker,
            marker=dict(color=COLORS[i - 1], opacity=0.55, line=dict(width=0.3, color="white")),
            showlegend=False,
            hovertemplate="Retorno: %{x:.3f}%<br>Densidad: %{y:.4f}<extra></extra>",
        ), row=1, col=i)

        # Curva normal ajustada
        fig.add_trace(go.Scatter(
            x=x_range, y=y_normal,
            mode="lines", name=f"Normal ({ticker})",
            line=dict(color="white", width=2.5, dash="dash"),
            showlegend=(i == 1),
        ), row=1, col=i)

        # Sombrear cola izquierda empírica (peor 5%) — zona roja
        p5 = np.percentile(log_ret, 5)
        x_tail_l = x_range[x_range <= p5]
        y_tail_l = stats.norm.pdf(x_tail_l, mu, sigma)
        if len(x_tail_l):
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_tail_l, x_tail_l[::-1]]),
                y=np.concatenate([y_tail_l, np.zeros(len(y_tail_l))]),
                fill="toself", fillcolor="rgba(220,50,50,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Cola izquierda (5% observado)",
                showlegend=(i == 1),
                hoverinfo="skip",
            ), row=1, col=i)

        # Sombrear cola derecha empírica (mejor 5%) — zona verde tenue
        p95 = np.percentile(log_ret, 95)
        x_tail_r = x_range[x_range >= p95]
        y_tail_r = stats.norm.pdf(x_tail_r, mu, sigma)
        if len(x_tail_r):
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_tail_r, x_tail_r[::-1]]),
                y=np.concatenate([y_tail_r, np.zeros(len(y_tail_r))]),
                fill="toself", fillcolor="rgba(50,180,50,0.13)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Cola derecha (5% observado)",
                showlegend=(i == 1),
                hoverinfo="skip",
            ), row=1, col=i)

        fig.update_xaxes(title_text="Retorno log diario (%)", row=1, col=i)
        fig.update_yaxes(title_text="Densidad" if i == 1 else "", row=1, col=i)

    fig.update_layout(
        height=height,
        title=dict(text="Distribución de retornos vs. distribución normal ajustada", font=dict(size=14)),
        template="plotly_white",
        margin=dict(l=60, r=20, t=70, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    )
    return fig


def hist_distribution_chart(port_returns: pd.Series, var_pct: float,
                              confianza: float, color: str = "#2196F3") -> go.Figure:
    """Histograma de retornos del portafolio con línea de VaR."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_returns.values,
        nbinsx=60,
        name="Retornos del portafolio",
        marker_color=color,
        opacity=0.7,
        hovertemplate="Retorno: %{x:.3f}%<br>Frecuencia: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=var_pct,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"VaR {int(confianza*100)}%: {var_pct:.3f}%",
        annotation_position="top right",
        annotation_font=dict(color="red", size=12),
    )
    fig.update_layout(
        title="Distribución de retornos diarios del portafolio",
        height=350, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(title="Retorno diario (%)"),
        yaxis=dict(title="Frecuencia"),
        showlegend=False,
    )
    return fig


def normal_distribution_chart(sigma_port: float, z: float,
                                confianza: float) -> go.Figure:
    """Curva normal con área de pérdida sombreada."""
    x = np.linspace(-4 * sigma_port, 4 * sigma_port, 400)
    y = stats.norm.pdf(x, 0, sigma_port)
    var_x = z * sigma_port   # negativo

    fig = go.Figure()
    # Área de cola izquierda (zona de pérdida)
    x_tail = x[x <= var_x]
    y_tail = stats.norm.pdf(x_tail, 0, sigma_port)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_tail, x_tail[::-1]]),
        y=np.concatenate([y_tail, np.zeros(len(y_tail))]),
        fill="toself", fillcolor="rgba(255,0,0,0.15)",
        line=dict(color="rgba(255,0,0,0)"),
        name=f"Zona de pérdida ({(1-confianza)*100:.0f}%)",
    ))
    # Curva normal completa
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color="#2196F3", width=2),
        name="Distribución normal",
    ))
    # Línea vertical del VaR
    fig.add_vline(
        x=var_x,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"z = {z:.3f} → VaR {int(confianza*100)}%",
        annotation_position="top right",
        annotation_font=dict(color="red", size=12),
    )
    fig.update_layout(
        title="Distribución normal del portafolio — Zona de pérdida",
        height=350, template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(title="Retorno diario (%)"),
        yaxis=dict(title="Densidad de probabilidad"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — EWMA / GARCH
# ══════════════════════════════════════════════════════════════════════════════

def kupiec_test(n_exc: int, n_days: int, confidence: float) -> tuple:
    """LR statistic y p-valor de la prueba POF de Kupiec."""
    from scipy.stats import chi2
    alpha = 1 - confidence
    if n_exc == 0:
        return 0.0, 1.0
    p_hat = n_exc / n_days
    if p_hat >= 1:
        return np.inf, 0.0
    N1, N0 = n_exc, n_days - n_exc
    lr = -2 * (N1 * np.log(alpha / p_hat) + N0 * np.log((1 - alpha) / (1 - p_hat)))
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def make_bt_chart(test_dates, actual_ret: np.ndarray,
                  var_lines: dict, height: int = 380) -> go.Figure:
    """Retornos reales + líneas VaR de cada método, excepciones marcadas."""
    fig = go.Figure()
    METHOD_COLORS = {"Histórico": "#2196F3", "Var-Cov": "#4CAF50",
                     "EWMA": "#FF9800", "GARCH": "#E91E63"}

    # Retornos como barras
    bar_colors = ["rgba(220,50,50,0.7)" if r < 0 else "rgba(50,180,50,0.5)"
                  for r in actual_ret]
    fig.add_trace(go.Bar(
        x=test_dates, y=actual_ret,
        name="Retorno portafolio", marker_color=bar_colors,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}%<extra>Retorno</extra>",
    ))

    for method, var_pct in var_lines.items():
        color = METHOD_COLORS[method]
        fig.add_trace(go.Scatter(
            x=test_dates, y=var_pct,
            mode="lines", name=f"VaR {method}",
            line=dict(color=color, width=1.5, dash="dash"),
            hovertemplate=f"VaR {method}: %{{y:.3f}}%<extra></extra>",
        ))
        # Marcar excepciones
        exc_mask = actual_ret < var_pct
        if exc_mask.any():
            fig.add_trace(go.Scatter(
                x=test_dates[exc_mask], y=actual_ret[exc_mask],
                mode="markers", name=f"Excepción {method}",
                marker=dict(color=color, size=8, symbol="x",
                            line=dict(width=2, color=color)),
                showlegend=False,
                hovertemplate=f"Excepción {method}: %{{y:.3f}}%<extra></extra>",
            ))

    fig.add_hline(y=0, line=dict(color="gray", width=0.8, dash="dot"))
    fig.update_layout(
        title="Retornos reales vs. VaR diario — ventana de backtest",
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=55, b=60),
        hovermode="x unified",
        yaxis=dict(title="Retorno log diario (%)"),
        xaxis=dict(title="Fecha"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        barmode="relative",
    )
    return fig


def make_bt_single(test_dates, actual_ret: np.ndarray,
                   var_pct: np.ndarray, method: str,
                   color: str, height: int = 280) -> go.Figure:
    """Gráfico individual de backtest para un solo método."""
    exc_mask = actual_ret < var_pct
    fig = go.Figure()
    bar_cols = ["rgba(220,50,50,0.65)" if r < 0 else "rgba(150,150,150,0.4)"
                for r in actual_ret]
    fig.add_trace(go.Bar(
        x=test_dates, y=actual_ret, name="Retorno",
        marker_color=bar_cols,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}%<extra>Retorno</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=var_pct, mode="lines", name="VaR",
        line=dict(color=color, width=2),
        hovertemplate="VaR: %{y:.3f}%<extra></extra>",
    ))
    if exc_mask.any():
        fig.add_trace(go.Scatter(
            x=test_dates[exc_mask], y=actual_ret[exc_mask],
            mode="markers", name="Excepción",
            marker=dict(color="red", size=9, symbol="x",
                        line=dict(width=2, color="red")),
            hovertemplate="Excepción: %{y:.3f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="gray", width=0.8, dash="dot"))
    n_exc = int(exc_mask.sum())
    fig.update_layout(
        title=f"{method} — {n_exc} excepciones",
        height=height, template="plotly_white",
        margin=dict(l=55, r=15, t=45, b=35),
        hovermode="x unified", showlegend=False,
        yaxis=dict(title="(%)"), xaxis=dict(title=""),
    )
    return fig


def compute_ewma_vol(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """Volatilidad EWMA diaria (mismas unidades que returns)."""
    var = np.empty(len(returns))
    var[0] = returns.iloc[0] ** 2
    for t in range(1, len(returns)):
        var[t] = lambda_ * var[t - 1] + (1 - lambda_) * returns.iloc[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=returns.index)


@st.cache_data(show_spinner=False)
def fit_garch(returns_values: np.ndarray, _index) -> tuple:
    """
    Ajusta GARCH(1,1) y devuelve (volatilidad condicional diaria, parámetros,
    pronóstico σ_{T+1}).  Cacheado para no reestimar en cada interacción.
    (_index con guion bajo para que st.cache_data no intente hashearlo)
    """
    am = arch_model(returns_values, vol="Garch", p=1, q=1,
                    mean="Constant", dist="normal", rescale=False)
    res = am.fit(disp="off")
    cond_vol = pd.Series(res.conditional_volatility, index=_index)
    forecast = res.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(forecast.variance.values[-1, 0]))
    params = res.params
    return cond_vol, params, sigma_next


def make_vol_comparison_chart(roll_vol: pd.Series, ewma_vol: pd.Series,
                               garch_vol: pd.Series, height: int = 420) -> go.Figure:
    """Serie de tiempo con las tres volatilidades anualizadas."""
    ann = np.sqrt(252)
    fig = go.Figure()
    for series, name, color, dash in [
        (roll_vol * ann,  "Clásica (rolling 30d)",  "#90A4AE", "dot"),
        (ewma_vol * ann,  "EWMA",                   "#FF9800", "solid"),
        (garch_vol * ann, "GARCH(1,1)",              "#E91E63", "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=name,
            line=dict(color=color, width=1.6, dash=dash),
            hovertemplate=f"<b>{name}</b><br>Fecha: %{{x|%Y-%m-%d}}<br>Vol anual: %{{y:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Volatilidad del portafolio: Clásica vs EWMA vs GARCH(1,1)", font=dict(size=14)),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
        yaxis=dict(title="Volatilidad anualizada (%)", ticksuffix="%"),
        xaxis=dict(title="Fecha"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — ANÁLISIS DE PORTAFOLIO
# ══════════════════════════════════════════════════════════════════════════════

def make_portfolio_returns_chart(data: dict, pesos: np.ndarray, height: int = 400) -> go.Figure:
    """Retornos log diarios individuales + retorno del portafolio en una sola gráfica."""
    fig = go.Figure()
    log_rets = pd.DataFrame({
        t: np.log(s / s.shift(1)).dropna() * 100 for t, s in data.items()
    }).dropna()
    port_ret = log_rets @ pesos

    # Activos individuales — líneas finas y transparentes
    for i, ticker in enumerate(log_rets.columns):
        fig.add_trace(go.Scatter(
            x=log_rets.index, y=log_rets[ticker].values,
            mode="lines", name=ticker,
            line=dict(color=COLORS[i], width=0.9),
            opacity=0.55,
            hovertemplate=f"<b>{ticker}</b><br>%{{x|%Y-%m-%d}}: %{{y:.3f}}%<extra></extra>",
        ))

    # Portafolio — línea gruesa destacada
    fig.add_trace(go.Scatter(
        x=port_ret.index, y=port_ret.values,
        mode="lines", name="Portafolio",
        line=dict(color="#FFD700", width=2.2),
        hovertemplate="<b>Portafolio</b><br>%{x|%Y-%m-%d}: %{y:.3f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(
        title=dict(text="Retornos logarítmicos diarios — activos y portafolio", font=dict(size=14)),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=55, b=60),
        hovermode="x unified",
        yaxis=dict(title="Retorno log diario (%)", ticksuffix="%"),
        xaxis=dict(title="Fecha"),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return fig


def make_var_diversification_chart(var_data: dict, height: int = 480) -> go.Figure:
    """
    4 subplots (uno por método) con barras de VaR para cada activo al 100%
    y para el portafolio — ilustra el beneficio de la diversificación.
    """
    methods   = list(var_data.keys())          # 4 métodos
    labels    = list(var_data[methods[0]].keys())  # portfolio + tickers
    bar_colors = ["#FFD700"] + COLORS          # gold para portafolio, colores para activos

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=methods,
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (row, col), method in zip(positions, methods):
        values = list(var_data[method].values())
        colors = bar_colors[:len(labels)]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"${v:,.0f}" for v in values],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}<br>VaR: $%{y:,.2f}<extra></extra>",
            ),
            row=row, col=col,
        )
        # Línea horizontal en el valor del portafolio para referencia visual
        port_val = var_data[method][labels[0]]
        fig.add_hline(
            y=port_val,
            line=dict(color="#FFD700", dash="dot", width=1.5),
            row=row, col=col,
        )
        fig.update_yaxes(title_text="VaR (USD)", tickprefix="$", row=row, col=col)

    fig.update_layout(
        title=dict(
            text="VaR por método: portafolio vs. 100% en cada activo",
            font=dict(size=15),
        ),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=80, b=50),
    )
    return fig


def make_es_diversification_chart(es_data: dict, height: int = 480) -> go.Figure:
    """Igual que VaR pero para Expected Shortfall."""
    methods    = list(es_data.keys())
    labels     = list(es_data[methods[0]].keys())
    bar_colors = ["#FFD700"] + COLORS

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=methods,
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (row, col), method in zip(positions, methods):
        values = list(es_data[method].values())
        colors = bar_colors[:len(labels)]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"${v:,.0f}" for v in values],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}<br>ES: $%{y:,.2f}<extra></extra>",
            ),
            row=row, col=col,
        )
        port_val = es_data[method][labels[0]]
        fig.add_hline(
            y=port_val,
            line=dict(color="#FFD700", dash="dot", width=1.5),
            row=row, col=col,
        )
        fig.update_yaxes(title_text="ES (USD)", tickprefix="$", row=row, col=col)

    fig.update_layout(
        title=dict(
            text="Expected Shortfall por método: portafolio vs. 100% en cada activo",
            font=dict(size=15),
        ),
        height=height, template="plotly_white",
        margin=dict(l=60, r=20, t=80, b=50),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Parámetros")

    ticker1 = st.text_input("Ticker 1", value="AAPL").strip().upper()
    ticker2 = st.text_input("Ticker 2 (opcional)", value="MSFT").strip().upper()
    ticker3 = st.text_input("Ticker 3 (opcional)", value="").strip().upper()

    st.divider()

    st.markdown("**Fecha de inicio**")
    date_start = st.date_input(
        "Fecha de inicio", value=datetime.date(2020, 1, 1),
        min_value=MIN_DATE, max_value=TODAY, label_visibility="collapsed",
    )
    st.markdown("**Fecha de fin**")
    date_end = st.date_input(
        "Fecha de fin", value=TODAY,
        min_value=MIN_DATE, max_value=TODAY, label_visibility="collapsed",
    )

    st.divider()

    graficar     = st.button("Graficar", type="primary", use_container_width=True)
    ver_retornos = st.toggle("Ver retornos", value=False)

    st.divider()
    st.markdown("### Portafolio")

    monto = st.number_input(
        "Monto invertido (USD)", min_value=1_000, max_value=100_000_000,
        value=100_000, step=1_000, format="%d",
        help="Capital total del portafolio en dólares.",
    )
    confianza = st.select_slider(
        "Nivel de confianza",
        options=[0.90, 0.95, 0.99], value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
        help="A mayor confianza, más conservador es el VaR.",
    )
    lambda_ewma = st.number_input(
        "λ EWMA", min_value=0.80, max_value=0.99,
        value=0.94, step=0.01, format="%.2f",
        help="Factor de decaimiento EWMA. RiskMetrics usa 0.94.",
    )

    st.markdown("**Pesos del portafolio** — deben sumar **100%**")
    _sidebar_tickers = [t for t in [ticker1, ticker2, ticker3] if t]
    pesos_raw = []
    for _t in _sidebar_tickers:
        _default = round(100 / len(_sidebar_tickers), 1)
        _p = st.number_input(
            f"Peso {_t} (%)", min_value=0.0, max_value=100.0,
            value=_default, step=0.1, format="%.1f", key=f"sb_w_{_t}",
        )
        pesos_raw.append(_p)

    _suma = sum(pesos_raw) if pesos_raw else 0
    if pesos_raw and abs(_suma - 100) > 0.01:
        st.warning(f"Los pesos suman {_suma:.1f}% — deben ser 100%.")
        pesos_ok = False
    else:
        if pesos_raw:
            st.success("✓ Pesos OK")
        pesos_ok = bool(pesos_raw)

    pesos = np.array(pesos_raw) / 100.0 if pesos_ok else np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

if graficar:
    tickers = [t for t in [ticker1, ticker2, ticker3] if t]

    if not tickers:
        st.error("Ingresa al menos un ticker.")
        st.stop()
    if date_start >= date_end:
        st.error("La fecha de inicio debe ser anterior a la de fin.")
        st.stop()

    data, errors = {}, []

    with st.spinner("Descargando datos..."):
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=str(date_start), end=str(date_end),
                                 auto_adjust=True, progress=False)
                if df.empty:
                    errors.append(f"{ticker} (sin datos)")
                    continue
                if hasattr(df.columns, "levels"):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df["Close"].dropna()
            except Exception as e:
                errors.append(f"{ticker} ({e})")

    if errors:
        st.warning(f"Problemas con: {', '.join(errors)}")
    if not data:
        st.error("No se pudieron obtener datos para los tickers indicados.")
        st.stop()

    st.session_state["data"]       = data
    st.session_state["date_start"] = date_start
    st.session_state["date_end"]   = date_end


# ══════════════════════════════════════════════════════════════════════════════
# RENDER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if "data" not in st.session_state:
    st.info("Configura los parámetros en la barra lateral y haz clic en **Graficar**.")
    st.stop()

data       = st.session_state["data"]
date_start = st.session_state["date_start"]
date_end   = st.session_state["date_end"]
names      = list(data.keys())

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Precios & Retornos",
    "⚠️ VaR del Portafolio",
    "📊 EWMA & GARCH",
    "📋 Análisis del Portafolio",
    "🔬 Backtesting",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Precios & Retornos
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if ver_retornos:
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(make_price_chart(data, f"Precios — {', '.join(names)}"),
                            use_container_width=True)
        with col_right:
            st.plotly_chart(make_returns_chart(data, f"Retornos log diarios — {', '.join(names)}"),
                            use_container_width=True)
    else:
        st.plotly_chart(make_price_chart(data, f"Precios — {', '.join(names)}"),
                        use_container_width=True)

    st.divider()

    for i, (ticker, series) in enumerate(data.items()):
        color = COLORS[i]
        if ver_retornos:
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(make_single_price(ticker, series, color), use_container_width=True)
            with col_right:
                st.plotly_chart(make_single_returns(ticker, series, color, COLORS_FILL[i]),
                                use_container_width=True)
        else:
            st.plotly_chart(make_single_price(ticker, series, color), use_container_width=True)

    st.divider()
    st.subheader("Resumen del periodo")
    cols = st.columns(len(data))
    for col_ui, (ticker, series) in zip(cols, data.items()):
        ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
        col_ui.metric(label=ticker, value=f"${series.iloc[-1]:.2f}",
                      delta=f"{ret:+.1f}% en el periodo")
        if ver_retornos:
            log_ret   = np.log(series / series.shift(1)).dropna() * 100
            vol_diaria = log_ret.std()
            vol_anual  = vol_diaria * np.sqrt(252)
            col_ui.metric(label=f"{ticker} — Volatilidad",
                          value=f"{vol_anual:.2f}% anual",
                          delta=f"{vol_diaria:.3f}% diaria",
                          delta_color="off")

    # ── Histograma de normalidad ───────────────────────────────────────────
    if ver_retornos:
        st.divider()
        st.subheader("¿Se distribuyen normal los retornos?")
        st.markdown(
            "Una de las hipótesis más importantes —y más cuestionadas— en finanzas es que los "
            "retornos siguen una **distribución normal**. El histograma muestra la distribución "
            "empírica (barras) frente a la curva normal teórica con la misma media y desviación "
            "estándar (línea punteada). Las **colas sombreadas** revelan los excesos respecto a la normal."
        )

        st.plotly_chart(make_histogram_normalidad(data), use_container_width=True)

        # ── Prueba de Jarque-Bera por ticker ──────────────────────────────
        st.markdown("### Prueba de normalidad: Jarque-Bera")
        st.markdown(
            "La prueba **Jarque-Bera** evalúa si los retornos son consistentes con una distribución "
            "normal, midiendo simultáneamente dos características:"
        )
        col_exp1, col_exp2 = st.columns(2)
        col_exp1.info(
            "**Asimetría (skewness):** mide si la distribución es simétrica. "
            "Una distribución normal tiene asimetría = 0. "
            "Asimetría negativa → cola izquierda más pesada (más días de pérdidas extremas)."
        )
        col_exp2.info(
            "**Curtosis (kurtosis):** mide el 'peso' de las colas. "
            "Una distribución normal tiene curtosis = 3. "
            "Curtosis > 3 → colas más gruesas que la normal (*leptocúrtica*), "
            "lo que implica más eventos extremos de los que la teoría predice."
        )
        st.markdown(
            r"La estadística JB combina ambas: "
            r"$JB = \dfrac{n}{6}\left(S^2 + \dfrac{(K-3)^2}{4}\right)$ "
            r"donde $S$ = asimetría, $K$ = curtosis, $n$ = número de observaciones. "
            "Bajo la hipótesis nula de normalidad, JB sigue una distribución χ² con 2 grados de libertad."
        )

        jb_cols = st.columns(len(data))
        for col_jb, (ticker, series) in zip(jb_cols, data.items()):
            log_ret = np.log(series / series.shift(1)).dropna() * 100
            jb_stat, p_val = stats.jarque_bera(log_ret)
            skew = float(stats.skew(log_ret))
            kurt = float(stats.kurtosis(log_ret, fisher=False))  # curtosis total (normal = 3)
            rechaza = p_val < 0.05

            col_jb.markdown(f"**{ticker}**")
            col_jb.metric("Estadístico JB", f"{jb_stat:,.2f}")
            col_jb.metric("p-valor", f"{p_val:.2e}",
                          delta="Rechaza normalidad ✗" if rechaza else "No rechaza normalidad ✓",
                          delta_color="inverse" if rechaza else "off")
            col_jb.metric("Asimetría", f"{skew:.4f}",
                          delta="Sesgo negativo (cola izq.)" if skew < -0.1
                          else ("Sesgo positivo (cola der.)" if skew > 0.1 else "Aproximadamente simétrico"),
                          delta_color="off")
            col_jb.metric("Curtosis", f"{kurt:.4f}",
                          delta=f"{'Colas gruesas (leptocúrtica)' if kurt > 3.5 else 'Similar a normal'}",
                          delta_color="inverse" if kurt > 3.5 else "off")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VaR del Portafolio
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Valor en Riesgo (VaR) del Portafolio")
    st.markdown(
        "El **VaR** responde a una pregunta concreta: *¿cuánto dinero puedo perder en un día "
        "en el peor escenario posible, dado un nivel de confianza?* "
        "Calculamos ese número con dos metodologías distintas para que puedas comparar sus supuestos y resultados."
    )

    st.info(
        f"💼 **Portafolio activo:** monto **${monto:,.0f} USD** · "
        f"confianza **{int(confianza*100)}%** · "
        f"pesos: {' / '.join([f'{n} {p*100:.1f}%' for n, p in zip(names, pesos)]) if pesos_ok else '—'}  \n"
        "Puedes cambiar estos valores en la barra lateral izquierda."
    )

    calcular = st.button("Calcular VaR", type="primary", disabled=not pesos_ok)

    if calcular and pesos_ok:
        pesos = np.array(pesos_raw) / 100.0

        # Construir DataFrame de retornos log diarios
        log_rets = pd.DataFrame({
            t: np.log(s / s.shift(1)).dropna() * 100
            for t, s in data.items()
        }).dropna()

        # Retorno diario del portafolio (suma ponderada)
        port_returns = log_rets @ pesos

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════
        # MÉTODO 1 — SIMULACIÓN HISTÓRICA
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## Método 1: Simulación Histórica")
        st.markdown(
            "La simulación histórica **no asume ninguna distribución estadística**. "
            "Simplemente usa los retornos pasados como escenarios de lo que podría volver a ocurrir, "
            "y pregunta: *¿cuál fue el peor día en el X% de los casos?*"
        )

        # Paso 1
        with st.expander("📌 Paso 1 — Retornos logarítmicos diarios de cada activo", expanded=True):
            st.markdown(
                "Calculamos el **retorno logarítmico diario** de cada activo con la fórmula:"
            )
            st.latex(r"r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)")
            st.markdown(
                "Usamos logaritmos porque son **aditivos en el tiempo** y tienen mejores propiedades "
                "estadísticas que los retornos simples. Multiplicamos por 100 para expresarlos en porcentaje."
            )
            st.dataframe(
                log_rets.tail(10).style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
            )
            st.caption(f"Mostrando los últimos 10 días de {len(log_rets)} observaciones totales.")

        # Paso 2
        with st.expander("📌 Paso 2 — Retorno diario del portafolio", expanded=True):
            st.markdown(
                "Combinamos los retornos individuales en un único **retorno de portafolio**, "
                "ponderando por el peso asignado a cada activo:"
            )
            formula_pesos = " + ".join(
                [f"w_{{{t}}} \\cdot r_{{{t}}}" for t in names]
            )
            st.latex(rf"r_{{portafolio}} = {formula_pesos}")

            valores_pesos = "  |  ".join([f"**{t}**: {p*100:.1f}%" for t, p in zip(names, pesos)])
            st.markdown(f"Con los pesos: {valores_pesos}")

            port_df = port_returns.to_frame(name="Retorno portafolio (%)")
            st.dataframe(
                port_df.tail(10).style.format("{:.4f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Retorno medio diario", f"{port_returns.mean():.4f}%")
            c2.metric("Desv. estándar diaria", f"{port_returns.std():.4f}%")
            c3.metric("Observaciones", f"{len(port_returns):,}")

        # Paso 3
        with st.expander("📌 Paso 3 — Ordenar los retornos de peor a mejor", expanded=True):
            st.markdown(
                "Ordenamos todos los retornos históricos del portafolio de **menor a mayor**. "
                "Los valores más negativos representan los peores días históricos."
            )
            sorted_rets = port_returns.sort_values()
            worst10 = sorted_rets.head(10).to_frame(name="Retorno (%)")
            worst10.index = worst10.index.strftime("%Y-%m-%d")
            st.markdown("**Los 10 peores días del portafolio:**")
            st.dataframe(
                worst10.style.format("{:.4f}").background_gradient(cmap="RdYlGn"),
                use_container_width=True,
            )

        # Paso 4
        with st.expander("📌 Paso 4 — Encontrar el percentil crítico", expanded=True):
            alpha = 1 - confianza
            var_pct = np.percentile(port_returns, alpha * 100)
            st.markdown(
                f"Con un nivel de confianza del **{int(confianza*100)}%**, buscamos el "
                f"**percentil {alpha*100:.0f}%** de la distribución de retornos. "
                f"Esto significa: *el {alpha*100:.0f}% de los días históricos, el portafolio perdió más que este valor.*"
            )
            st.latex(
                rf"\text{{VaR}}_{{{int(confianza*100)}\%}} = \text{{Percentil}}_{{\alpha={alpha*100:.0f}\%}}"
                rf"\left(r_{{portafolio}}\right)"
            )
            st.info(
                f"**Percentil {alpha*100:.0f}%** de los retornos históricos = **{var_pct:.4f}%** diario"
            )

        # Paso 5
        with st.expander("📌 Paso 5 — Convertir a valor monetario", expanded=True):
            var_hist = abs(var_pct / 100) * monto
            st.markdown(
                "Multiplicamos el retorno crítico (en términos absolutos) por el monto invertido "
                "para obtener la **pérdida máxima esperada en dólares**:"
            )
            st.latex(
                rf"\text{{VaR}}_\$ = \left| {var_pct:.4f}\% \right| \times \${monto:,.0f}"
                rf"= \${var_hist:,.2f}"
            )
            st.success(
                f"**VaR Histórico ({int(confianza*100)}% confianza): ${var_hist:,.2f} USD**  \n"
                f"En el {alpha*100:.0f}% de los días, el portafolio podría perder más de esta cantidad."
            )

        # Paso 6 — ES Histórico
        with st.expander("📌 Paso 6 — Expected Shortfall (ES) histórico", expanded=True):
            cola = port_returns[port_returns <= var_pct]
            es_pct = cola.mean()
            es_hist = abs(es_pct / 100) * monto
            st.markdown(
                "El **Expected Shortfall** (también llamado *CVaR* o *Conditional VaR*) responde "
                "una pregunta que el VaR deja sin contestar: *si el día es de los malos, "
                "¿cuánto se pierde **en promedio**?*  \n\n"
                "Mientras el VaR marca el umbral, el ES mide la pérdida **esperada en la cola**, "
                "promediando todos los escenarios que superan ese umbral:"
            )
            st.latex(
                r"ES_{\alpha} = -\,\mathbb{E}\!\left[\,r \;\middle|\; r \leq \text{VaR}_{\alpha}\,\right]"
            )
            st.markdown(
                f"Tomamos los **{len(cola)} días** en que el retorno fue igual o peor al VaR "
                f"({var_pct:.4f}%) y calculamos su promedio:"
            )

            worst_df = cola.sort_values().to_frame(name="Retorno (%)")
            worst_df.index = worst_df.index.strftime("%Y-%m-%d")
            st.dataframe(
                worst_df.style.format("{:.4f}").background_gradient(cmap="Reds_r"),
                use_container_width=True,
                height=180,
            )
            st.latex(
                rf"ES_\$ = \left| {es_pct:.4f}\% \right| \times \${monto:,.0f} = \${es_hist:,.2f}"
            )
            st.warning(
                f"**ES Histórico ({int(confianza*100)}% confianza): ${es_hist:,.2f} USD**  \n"
                f"En los días que superan el VaR, la pérdida promedio esperada es esta cantidad. "
                f"Siempre es mayor que el VaR."
            )

        # Gráfico histórico
        st.plotly_chart(
            hist_distribution_chart(port_returns, var_pct, confianza),
            use_container_width=True,
        )

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════
        # MÉTODO 2 — VARIANZA-COVARIANZA
        # ══════════════════════════════════════════════════════════════════
        st.markdown("## Método 2: Varianza-Covarianza (Paramétrico)")
        st.markdown(
            "Este método **asume que los retornos siguen una distribución normal**. "
            "En lugar de mirar los datos directamente, construye un modelo matemático del portafolio "
            "usando medias, desviaciones estándar y correlaciones entre activos."
        )

        var_vc, z_score, sigma_port, cov_matrix, medias, stds = var_varianza_covarianza(
            log_rets, pesos, monto, confianza
        )

        # Paso 1
        with st.expander("📌 Paso 1 — Estadísticas individuales de cada activo", expanded=True):
            st.markdown(
                "Calculamos la **media** (retorno esperado) y la **desviación estándar** "
                "(volatilidad) de los retornos diarios de cada activo:"
            )
            stats_df = pd.DataFrame({
                "Media diaria (%)": medias,
                "Desv. estándar diaria (%)": stds,
                "Volatilidad anualizada (%)": stds * np.sqrt(252),
            }).round(4)
            st.dataframe(stats_df.style.format("{:.4f}"), use_container_width=True)
            st.caption("La volatilidad anualizada se obtiene multiplicando la diaria por √252 (días hábiles en un año).")

        # Paso 2
        with st.expander("📌 Paso 2 — Matriz de covarianza", expanded=True):
            st.markdown(
                "La **matriz de covarianza** mide cómo se mueven los activos juntos. "
                "Los valores en la diagonal son las varianzas individuales; "
                "los valores fuera de la diagonal miden la relación entre pares de activos. "
                "Si dos activos tienen covarianza positiva, tienden a subir y bajar al mismo tiempo."
            )
            st.latex(r"\Sigma = \begin{pmatrix} \sigma_1^2 & \sigma_{12} & \cdots \\ \sigma_{21} & \sigma_2^2 & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}")
            st.dataframe(
                cov_matrix.style.format("{:.6f}").background_gradient(cmap="Blues"),
                use_container_width=True,
            )
            corr_matrix = log_rets.corr()
            st.markdown("**Matriz de correlación** (más fácil de interpretar: va de -1 a +1):")
            st.dataframe(
                corr_matrix.style.format("{:.4f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                use_container_width=True,
            )

        # Paso 3
        with st.expander("📌 Paso 3 — Varianza del portafolio", expanded=True):
            sigma2_port = pesos @ cov_matrix.values @ pesos
            st.markdown(
                "La varianza del portafolio **no es simplemente el promedio ponderado** de las varianzas "
                "individuales — también captura el efecto de la diversificación a través de las covarianzas:"
            )
            st.latex(r"\sigma^2_p = \mathbf{w}^\top \, \Sigma \, \mathbf{w}")

            pesos_str = " \\\\ ".join([f"w_{{{t}}} = {p:.3f}" for t, p in zip(names, pesos)])
            st.markdown("Vector de pesos utilizado:")
            st.latex(rf"\mathbf{{w}} = \begin{{pmatrix}} {pesos_str} \end{{pmatrix}}")

            c1, c2 = st.columns(2)
            c1.metric("Varianza del portafolio (%²)", f"{sigma2_port:.6f}")
            c2.metric("Desv. estándar del portafolio (%)", f"{sigma_port:.4f}")

        # Paso 4
        with st.expander("📌 Paso 4 — Desviación estándar del portafolio", expanded=True):
            st.markdown(
                "Sacamos raíz cuadrada de la varianza para obtener la **volatilidad diaria del portafolio**, "
                "que es la medida que usaremos para calcular el VaR:"
            )
            st.latex(r"\sigma_p = \sqrt{\sigma^2_p} = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}")
            st.info(f"**Volatilidad diaria del portafolio: {sigma_port:.4f}%**  \n"
                    f"Equivalente anualizada: **{sigma_port * np.sqrt(252):.4f}%**")

        # Paso 5
        with st.expander("📌 Paso 5 — Z-score del nivel de confianza", expanded=True):
            st.markdown(
                f"Dado que asumimos distribución normal, el **z-score** nos dice cuántas desviaciones "
                f"estándar por debajo de la media cae el percentil crítico. "
                f"Para un nivel de confianza del **{int(confianza*100)}%**, el z es:"
            )
            st.latex(rf"z_{{{int(confianza*100)}\%}} = {z_score:.4f}")
            st.markdown(
                f"*Interpretación: en una distribución normal estándar, solo el {(1-confianza)*100:.0f}% "
                f"de los valores cae por debajo de {z_score:.4f} desviaciones estándar de la media.*"
            )

        # Paso 6
        with st.expander("📌 Paso 6 — Cálculo final del VaR", expanded=True):
            st.markdown("Combinamos todos los elementos para obtener el VaR en dólares:")
            st.latex(
                r"\text{VaR}_\$ = \text{Monto} \times \sigma_p \times |z|"
            )
            st.latex(
                rf"\text{{VaR}}_\$ = \${monto:,.0f} \times {sigma_port/100:.6f} \times {abs(z_score):.4f} = \${var_vc:,.2f}"
            )
            st.success(
                f"**VaR Varianza-Covarianza ({int(confianza*100)}% confianza): ${var_vc:,.2f} USD**  \n"
                f"Asumiendo normalidad, el portafolio no debería perder más de esta cantidad "
                f"en el {(1-confianza)*100:.0f}% peor de los días."
            )

        # Paso 7 — ES Varianza-Covarianza
        with st.expander("📌 Paso 7 — Expected Shortfall (ES) paramétrico", expanded=True):
            alpha = 1 - confianza
            phi_z = stats.norm.pdf(abs(z_score))   # densidad de la normal en |z|
            es_vc = monto * (sigma_port / 100) * phi_z / alpha
            st.markdown(
                "Bajo el supuesto de normalidad, el ES tiene una fórmula cerrada que depende "
                "de la **densidad** de la distribución normal evaluada en el z-score crítico:"
            )
            st.latex(
                r"ES_\$ = \text{Monto} \times \sigma_p \times \frac{\varphi(|z|)}{\alpha}"
            )
            st.markdown(
                r"Donde $\varphi(\cdot)$ es la función de densidad de la normal estándar y "
                r"$\alpha = 1 - \text{confianza}$ es el tamaño de la cola."
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("φ(|z|) — densidad en z", f"{phi_z:.6f}")
            c2.metric("α — tamaño de la cola", f"{alpha:.2f}  ({alpha*100:.0f}%)")
            c3.metric("φ(|z|) / α", f"{phi_z/alpha:.6f}")

            st.latex(
                rf"ES_\$ = \${monto:,.0f} \times {sigma_port/100:.6f} \times "
                rf"\frac{{{phi_z:.6f}}}{{{alpha:.2f}}} = \${es_vc:,.2f}"
            )
            st.warning(
                f"**ES Varianza-Covarianza ({int(confianza*100)}% confianza): ${es_vc:,.2f} USD**  \n"
                f"Esta es la pérdida promedio esperada en la cola bajo el supuesto de normalidad."
            )

        # Gráfico normal
        st.plotly_chart(
            normal_distribution_chart(sigma_port, z_score, confianza),
            use_container_width=True,
        )

        # ── Comparativa final ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Comparativa de resultados")
        st.markdown(
            "Ambos métodos responden la misma pregunta pero con supuestos distintos. "
            "La diferencia entre ellos te dice cuánto importa el supuesto de normalidad."
        )

        comp_cols = st.columns(4)
        comp_cols[0].metric("VaR Histórico", f"${var_hist:,.2f}",
                            delta="No asume distribución", delta_color="off")
        comp_cols[1].metric("ES Histórico", f"${es_hist:,.2f}",
                            delta=f"+{(es_hist/var_hist - 1)*100:.1f}% sobre el VaR hist.", delta_color="off")
        comp_cols[2].metric("VaR Var-Cov", f"${var_vc:,.2f}",
                            delta="Asume normalidad", delta_color="off")
        comp_cols[3].metric("ES Var-Cov", f"${es_vc:,.2f}",
                            delta=f"+{(es_vc/var_vc - 1)*100:.1f}% sobre el VaR V-C", delta_color="off")

        st.info(
            "💡 **VaR vs ES:** El ES siempre es mayor que el VaR porque mide el promedio de la cola, "
            "no solo su umbral. Por eso el ES es preferido por reguladores (Basilea III) como medida de riesgo.  \n"
            "**¿Qué método usar?** Si los retornos muestran *colas gruesas* —como confirma la prueba "
            "Jarque-Bera en el Tab 1— el VaR y ES históricos serán más conservadores y realistas. "
            "El método Var-Cov puede **subestimar** el riesgo en mercados turbulentos."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ANÁLISIS DEL PORTAFOLIO
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Análisis Completo del Portafolio")

    if not pesos_ok:
        st.warning("Define los pesos del portafolio en la barra lateral para continuar.")
        st.stop()

    # ── Botón único + resumen de parámetros ───────────────────────────────
    param_col, btn_col = st.columns([4, 1])
    param_col.info(
        f"💼 **${monto:,.0f} USD** · confianza **{int(confianza*100)}%** · "
        f"λ = {lambda_ewma:.2f} · "
        f"pesos: {' / '.join([f'{n} {p*100:.1f}%' for n, p in zip(names, pesos)])}"
    )
    calcular_todo = btn_col.button(
        "⟳ Calcular todo", type="primary", use_container_width=True, key="btn_todo",
        help="Recalcula con los parámetros actuales del sidebar.",
    )

    if calcular_todo:
        st.session_state.pop("analisis_tab4", None)

    if not calcular_todo and "analisis_tab4" not in st.session_state:
        st.info("Ajusta los parámetros en el sidebar y haz clic en **⟳ Calcular todo**.")
        st.stop()

    # ── Cálculo completo (solo cuando se pulsa el botón) ──────────────────
    if calcular_todo:
        with st.spinner("Calculando... (GARCH puede tardar unos segundos)"):
            _log_rets = pd.DataFrame({
                t: np.log(s / s.shift(1)).dropna() * 100 for t, s in data.items()
            }).dropna()
            _port_ret = _log_rets @ pesos
            _roll_vol = _port_ret.rolling(30).std().dropna()
            _ewma_vol = compute_ewma_vol(_port_ret, lambda_ewma).loc[_roll_vol.index]
            _gc, _, _gsn = fit_garch(_port_ret.values, _port_ret.index)
            _garch_vol   = _gc.loc[_roll_vol.index]

            _z     = stats.norm.ppf(1 - confianza)
            _alpha = 1 - confianza
            _phi_z = stats.norm.pdf(abs(_z))
            _sigma_port = np.sqrt(pesos @ _log_rets.cov().values @ pesos)

            def _calc(ret_s, method):
                if method == "Histórico":
                    vp = np.percentile(ret_s, _alpha * 100)
                    return abs(vp / 100) * monto, abs(ret_s[ret_s <= vp].mean() / 100) * monto
                if method == "Var-Cov":
                    sig = ret_s.std()
                    return abs(_z)*(sig/100)*monto, (sig/100)*monto*_phi_z/_alpha
                if method == "EWMA":
                    ew = compute_ewma_vol(ret_s, lambda_ewma).iloc[-1]
                    return abs(_z)*(ew/100)*monto, (ew/100)*monto*_phi_z/_alpha
                # GARCH
                _, _, gsn_ = fit_garch(ret_s.values, ret_s.index)
                return abs(_z)*(gsn_/100)*monto, (gsn_/100)*monto*_phi_z/_alpha

            _vp_h = np.percentile(_port_ret, _alpha * 100)
            _port_var = {
                "Histórico": abs(_vp_h/100)*monto,
                "Var-Cov":   abs(_z)*(_sigma_port/100)*monto,
                "EWMA":      abs(_z)*(compute_ewma_vol(_port_ret, lambda_ewma).iloc[-1]/100)*monto,
                "GARCH":     abs(_z)*(_gsn/100)*monto,
            }
            _port_es = {
                "Histórico": abs(_port_ret[_port_ret <= _vp_h].mean()/100)*monto,
                "Var-Cov":   (_sigma_port/100)*monto*_phi_z/_alpha,
                "EWMA":      (compute_ewma_vol(_port_ret, lambda_ewma).iloc[-1]/100)*monto*_phi_z/_alpha,
                "GARCH":     (_gsn/100)*monto*_phi_z/_alpha,
            }

            _methods = ["Histórico", "Var-Cov", "EWMA", "GARCH"]
            _vars_ind = {m: {"Portafolio": _port_var[m]} for m in _methods}
            _es_ind   = {m: {"Portafolio": _port_es[m]}  for m in _methods}
            for method in _methods:
                for ticker in names:
                    v, e = _calc(_log_rets[ticker], method)
                    _vars_ind[method][ticker] = v
                    _es_ind[method][ticker]   = e

        st.session_state["analisis_tab4"] = dict(
            log_rets=_log_rets, port_ret=_port_ret,
            roll_vol=_roll_vol, ewma_vol=_ewma_vol,
            garch_vol=_garch_vol, gsn=_gsn,
            vars_ind=_vars_ind, es_ind=_es_ind,
        )

    # ── Cargar desde sesión ────────────────────────────────────────────────
    _r         = st.session_state["analisis_tab4"]
    _log_rets  = _r["log_rets"];  _port_ret  = _r["port_ret"]
    _roll_vol  = _r["roll_vol"];  _ewma_vol  = _r["ewma_vol"]
    _garch_vol = _r["garch_vol"]; _gsn       = _r["gsn"]
    _vars_ind  = _r["vars_ind"];  _es_ind    = _r["es_ind"]

    # ── Sección 1: Retornos ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 1. Retornos del portafolio y activos individuales")
    st.markdown(
        "La línea dorada es el retorno diario del **portafolio combinado**. "
        "Al compararlo con los activos individuales se observa cómo la diversificación "
        "**reduce la amplitud de los movimientos extremos**."
    )
    st.plotly_chart(make_portfolio_returns_chart(data, pesos), use_container_width=True)

    # ── Sección 2: Volatilidades ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 2. Evolución de la volatilidad del portafolio")
    st.plotly_chart(make_vol_comparison_chart(_roll_vol, _ewma_vol, _garch_vol),
                    use_container_width=True)
    v1, v2, v3 = st.columns(3)
    v1.metric("Vol. clásica (30d)", f"{_roll_vol.iloc[-1]*np.sqrt(252):.2f}% anual",
              delta=f"{_roll_vol.iloc[-1]:.4f}% diaria", delta_color="off")
    v2.metric("Vol. EWMA",         f"{_ewma_vol.iloc[-1]*np.sqrt(252):.2f}% anual",
              delta=f"{_ewma_vol.iloc[-1]:.4f}% diaria", delta_color="off")
    v3.metric("Vol. GARCH (T+1)",  f"{_gsn*np.sqrt(252):.2f}% anual",
              delta=f"{_gsn:.4f}% diaria", delta_color="off")

    # ── Sección 3: VaR / ES — diversificación ─────────────────────────────
    st.markdown("---")
    st.markdown("## 3. VaR y ES: beneficio de la diversificación")
    st.markdown(
        "Cada gráfico muestra el VaR calculado con un método distinto. "
        "Las barras comparan lo que se arriesga en el **portafolio actual** vs. "
        "lo que se arriesgaría si el **100% de la inversión** estuviera en un solo activo."
    )

    st.markdown("### VaR por método")
    st.plotly_chart(make_var_diversification_chart(_vars_ind), use_container_width=True)

    st.markdown("### Expected Shortfall por método")
    st.plotly_chart(make_es_diversification_chart(_es_ind), use_container_width=True)

    st.markdown("### Tabla resumen")
    _methods_list = ["Histórico", "Var-Cov", "EWMA", "GARCH"]
    _rows = []
    for label in ["Portafolio"] + names:
        row = {"Activo / Portafolio": label}
        for method in _methods_list:
            row[f"VaR {method}"] = f"${_vars_ind[method][label]:,.2f}"
            row[f"ES {method}"]  = f"${_es_ind[method][label]:,.2f}"
        _rows.append(row)
    st.dataframe(pd.DataFrame(_rows).set_index("Activo / Portafolio"), use_container_width=True)

    st.markdown("### Beneficio de la diversificación")
    div_cols = st.columns(len(_methods_list))
    for col_d, method in zip(div_cols, _methods_list):
        max_ind    = max(_vars_ind[method][t] for t in names)
        ahorro     = max_ind - _vars_ind[method]["Portafolio"]
        col_d.metric(method, f"${ahorro:,.2f} menos",
                     delta=f"−{ahorro/max_ind*100:.1f}% vs. activo más riesgoso",
                     delta_color="inverse")

    st.success(
        "💡 **Teorema de Markowitz en acción:** el riesgo del portafolio es siempre "
        "menor o igual al promedio ponderado de los riesgos individuales, "
        "gracias a que los activos no están perfectamente correlacionados."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EWMA & GARCH
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Modelos de Volatilidad Dinámica: EWMA y GARCH")
    st.markdown(
        "La volatilidad de los activos financieros **no es constante en el tiempo**. "
        "Durante crisis los mercados se vuelven mucho más volátiles, y luego se calman. "
        "Este fenómeno se llama **clustering de volatilidad**. Los modelos EWMA y GARCH "
        "capturan esta dinámica, a diferencia de la desviación estándar clásica que trata "
        "todos los días con el mismo peso."
    )

    st.info(
        f"💼 **Portafolio activo:** monto **${monto:,.0f} USD** · "
        f"confianza **{int(confianza*100)}%** · λ EWMA **{lambda_ewma:.2f}** · "
        f"pesos: {' / '.join([f'{n} {p*100:.1f}%' for n, p in zip(names, pesos)]) if pesos_ok else '—'}  \n"
        "Puedes cambiar estos valores en la barra lateral izquierda."
    )

    calcular_t3 = st.button("Calcular", type="primary", disabled=not pesos_ok, key="btn_t3")

    if calcular_t3 and pesos_ok:
        # Retornos log diarios del portafolio
        log_rets_all = pd.DataFrame({
            t: np.log(s / s.shift(1)).dropna() * 100
            for t, s in data.items()
        }).dropna()
        port_ret = log_rets_all @ pesos

        # ── Volatilidades ─────────────────────────────────────────────────
        roll_vol  = port_ret.rolling(30).std().dropna()
        ewma_vol  = compute_ewma_vol(port_ret, lambda_ewma).loc[roll_vol.index]

        with st.spinner("Estimando GARCH(1,1)... (puede tomar unos segundos)"):
            garch_cond_vol, garch_params, sigma_garch_next = fit_garch(
                port_ret.values, port_ret.index
            )
        garch_vol = garch_cond_vol.loc[roll_vol.index]

        # ─────────────────────────────────────────────────────────────────
        # SECCIÓN 1: Comparativa de volatilidades
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Evolución de la volatilidad del portafolio")
        st.markdown(
            "La siguiente gráfica muestra las tres estimaciones de volatilidad anualizada "
            "a lo largo del tiempo. Observa cómo EWMA y GARCH reaccionan más rápido a los "
            "choques (p.ej. COVID-19 en 2020, crisis bancarias de 2023), mientras que la "
            "volatilidad clásica los *suaviza* dentro de la ventana de 30 días."
        )
        st.plotly_chart(
            make_vol_comparison_chart(roll_vol, ewma_vol, garch_vol),
            use_container_width=True,
        )

        vol_actual_roll  = roll_vol.iloc[-1]  * np.sqrt(252)
        vol_actual_ewma  = ewma_vol.iloc[-1]  * np.sqrt(252)
        vol_actual_garch = sigma_garch_next    * np.sqrt(252)

        vm1, vm2, vm3 = st.columns(3)
        vm1.metric("Volatilidad clásica (actual)", f"{vol_actual_roll:.2f}%",
                   delta="Promedio ventana 30d", delta_color="off")
        vm2.metric("Volatilidad EWMA (actual)", f"{vol_actual_ewma:.2f}%",
                   delta=f"λ = {lambda_ewma:.2f}", delta_color="off")
        vm3.metric("Volatilidad GARCH (pronóstico)", f"{vol_actual_garch:.2f}%",
                   delta="σ_{T+1} 1-day ahead", delta_color="off")

        t3_z = stats.norm.ppf(1 - confianza)

        # ─────────────────────────────────────────────────────────────────
        # SECCIÓN 2: VaR EWMA paso a paso
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## VaR con EWMA")
        st.markdown(
            "El modelo **EWMA** (*Exponentially Weighted Moving Average*) asigna más peso "
            "a los retornos recientes y menos a los pasados. A diferencia de la ventana "
            "rodante que trata todos los días igual, EWMA *olvida* el pasado de forma gradual."
        )

        with st.expander("📌 Paso 1 — Idea intuitiva del EWMA", expanded=True):
            st.markdown(
                "Imagina que quieres estimar cuán nervioso está el mercado *hoy*. "
                "Lo que pasó ayer importa más que lo que pasó hace un mes. "
                "EWMA formaliza esta idea con un parámetro **λ (lambda)**:"
            )
            st.latex(r"\sigma^2_t = \lambda \cdot \sigma^2_{t-1} + (1-\lambda)\cdot r_{t-1}^2")
            st.markdown(
                f"Con **λ = {lambda_ewma:.2f}** (valor actual), el día de ayer tiene peso "
                f"**{(1-lambda_ewma)*100:.0f}%**, el de anteayer **{(1-lambda_ewma)*lambda_ewma*100:.1f}%**, "
                f"y así sucesivamente. Cuanto más cercano a 1 sea λ, más lento olvida el pasado."
            )
            pesos_ewma = [(1 - lambda_ewma) * lambda_ewma**k for k in range(10)]
            fig_pesos = go.Figure(go.Bar(
                x=[f"t-{k}" for k in range(10)],
                y=pesos_ewma,
                marker_color="#FF9800",
                text=[f"{p*100:.2f}%" for p in pesos_ewma],
                textposition="outside",
            ))
            fig_pesos.update_layout(
                title="Pesos asignados a cada día (λ = {:.2f})".format(lambda_ewma),
                height=280, template="plotly_white",
                margin=dict(l=40, r=20, t=50, b=40),
                yaxis=dict(title="Peso", tickformat=".2%"),
                xaxis=dict(title="Día"),
            )
            st.plotly_chart(fig_pesos, use_container_width=True)

        with st.expander("📌 Paso 2 — Evolución de la varianza EWMA", expanded=True):
            st.markdown(
                "Aplicamos la fórmula recursiva sobre los retornos diarios del portafolio. "
                "La varianza EWMA se actualiza cada día con el retorno observado:"
            )
            ewma_var_series = ewma_vol ** 2
            fig_ewma_var = go.Figure()
            fig_ewma_var.add_trace(go.Scatter(
                x=ewma_var_series.index, y=ewma_var_series.values,
                mode="lines", line=dict(color="#FF9800", width=1.5),
                hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Varianza EWMA: %{y:.6f}<extra></extra>",
            ))
            fig_ewma_var.update_layout(
                title="Varianza EWMA diaria del portafolio (%²)",
                height=300, template="plotly_white",
                margin=dict(l=60, r=20, t=50, b=40),
                yaxis=dict(title="Varianza (%²)"),
                xaxis=dict(title="Fecha"),
            )
            st.plotly_chart(fig_ewma_var, use_container_width=True)

            e1, e2, e3 = st.columns(3)
            e1.metric("σ² EWMA hoy (%²)",  f"{ewma_vol.iloc[-1]**2:.6f}")
            e2.metric("σ EWMA hoy (%)",    f"{ewma_vol.iloc[-1]:.4f}")
            e3.metric("σ EWMA anualizada", f"{ewma_vol.iloc[-1]*np.sqrt(252):.4f}%")

        with st.expander("📌 Paso 3 — VaR diario con EWMA", expanded=True):
            sigma_ewma_hoy = ewma_vol.iloc[-1]
            var_ewma = abs(t3_z) * (sigma_ewma_hoy / 100) * monto
            st.markdown(
                "Usamos la volatilidad EWMA del último día como estimado de la volatilidad "
                "de **mañana**, asumiendo distribución normal:"
            )
            st.latex(r"\text{VaR}^{EWMA}_\$ = \text{Monto} \times \sigma^{EWMA}_t \times |z_\alpha|")
            st.latex(
                rf"\text{{VaR}}^{{EWMA}}_\$ = \${monto:,.0f} \times "
                rf"{sigma_ewma_hoy/100:.6f} \times {abs(t3_z):.4f} = \${var_ewma:,.2f}"
            )
            alpha_t3 = 1 - confianza
            phi_z_t3 = stats.norm.pdf(abs(t3_z))
            es_ewma = abs(t3_z) * (sigma_ewma_hoy / 100) * monto * phi_z_t3 / (abs(t3_z) * alpha_t3)
            es_ewma = (sigma_ewma_hoy / 100) * monto * phi_z_t3 / alpha_t3
            st.success(f"**VaR EWMA ({int(confianza*100)}%): ${var_ewma:,.2f} USD**")

            st.markdown("**Expected Shortfall EWMA:**")
            st.latex(
                r"ES^{EWMA}_\$ = \text{Monto} \times \sigma^{EWMA}_t \times \frac{\varphi(|z|)}{\alpha}"
            )
            st.latex(
                rf"ES^{{EWMA}}_\$ = \${monto:,.0f} \times {sigma_ewma_hoy/100:.6f} \times "
                rf"\frac{{{phi_z_t3:.6f}}}{{{alpha_t3:.2f}}} = \${es_ewma:,.2f}"
            )
            st.warning(f"**ES EWMA ({int(confianza*100)}%): ${es_ewma:,.2f} USD**")

        # ─────────────────────────────────────────────────────────────────
        # SECCIÓN 3: VaR GARCH paso a paso
        # ─────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## VaR con GARCH(1,1)")
        st.markdown(
            "El modelo **GARCH** (*Generalized Autoregressive Conditional Heteroskedasticity*) "
            "es el estándar de la industria para modelar volatilidad. Captura dos efectos clave: "
            "los choques recientes (*ARCH*) y la persistencia de la volatilidad pasada (*GARCH*)."
        )

        with st.expander("📌 Paso 1 — La ecuación GARCH(1,1)", expanded=True):
            st.markdown(
                "La varianza condicional en el tiempo *t* depende de tres componentes:"
            )
            st.latex(
                r"\sigma^2_t = \underbrace{\omega}_{\text{cte. base}} + "
                r"\underbrace{\alpha \cdot \varepsilon^2_{t-1}}_{\text{choque reciente (ARCH)}} + "
                r"\underbrace{\beta \cdot \sigma^2_{t-1}}_{\text{volatilidad pasada (GARCH)}}"
            )
            st.markdown(
                "- **ω (omega):** varianza de largo plazo escalada. Si los demás términos fueran cero, "
                "la varianza sería constante e igual a ω.\n"
                "- **α (alpha):** reacción a noticias recientes. Un α alto → el modelo reacciona "
                "fuerte a los choques (el mercado es nervioso).\n"
                "- **β (beta):** persistencia de la volatilidad. Un β alto → la volatilidad tarda "
                "mucho en volver a su nivel normal.\n"
                "- **α + β < 1** garantiza que el proceso sea estacionario (la volatilidad no explota)."
            )

        with st.expander("📌 Paso 2 — Parámetros estimados", expanded=True):
            omega = garch_params["omega"]
            alpha = garch_params["alpha[1]"]
            beta  = garch_params["beta[1]"]
            mu    = garch_params["mu"]
            vol_lp = np.sqrt(omega / (1 - alpha - beta)) if (alpha + beta) < 1 else np.nan

            st.markdown("Estimamos los parámetros por **máxima verosimilitud** sobre los retornos históricos del portafolio:")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("ω (omega)",  f"{omega:.6f}")
            p2.metric("α (alpha)",  f"{alpha:.4f}",
                      delta="Sensibilidad a choques", delta_color="off")
            p3.metric("β (beta)",   f"{beta:.4f}",
                      delta="Persistencia", delta_color="off")
            p4.metric("α + β",      f"{alpha+beta:.4f}",
                      delta="< 1 → estacionario ✓" if alpha+beta < 1 else "≥ 1 → no estacionario ✗",
                      delta_color="normal" if alpha+beta < 1 else "inverse")

            if not np.isnan(vol_lp):
                st.info(
                    f"**Volatilidad de largo plazo:** σ_LP = √(ω / (1 − α − β)) = "
                    f"**{vol_lp:.4f}% diario** → **{vol_lp*np.sqrt(252):.2f}% anual**  \n"
                    "Hacia este nivel converge la volatilidad en el largo plazo."
                )
            st.latex(
                rf"\sigma^2_t = {omega:.6f} + {alpha:.4f}\,\varepsilon^2_{{t-1}} + {beta:.4f}\,\sigma^2_{{t-1}}"
            )

        with st.expander("📌 Paso 3 — Volatilidad condicional histórica", expanded=True):
            st.markdown(
                "Con los parámetros estimados, reconstruimos la volatilidad condicional para "
                "cada día del periodo. Esta es la *firma* del GARCH: reacciona a los shocks "
                "y luego decae gradualmente."
            )
            fig_garch_vol = go.Figure()
            fig_garch_vol.add_trace(go.Scatter(
                x=garch_vol.index, y=garch_vol.values * np.sqrt(252),
                mode="lines", line=dict(color="#E91E63", width=1.5),
                hovertemplate="Fecha: %{x|%Y-%m-%d}<br>σ GARCH anual: %{y:.2f}%<extra></extra>",
            ))
            if not np.isnan(vol_lp):
                fig_garch_vol.add_hline(
                    y=vol_lp * np.sqrt(252),
                    line=dict(color="gray", dash="dot", width=1.2),
                    annotation_text=f"Vol. largo plazo: {vol_lp*np.sqrt(252):.2f}%",
                    annotation_position="top right",
                )
            fig_garch_vol.update_layout(
                title="Volatilidad condicional GARCH(1,1) — anualizada (%)",
                height=320, template="plotly_white",
                margin=dict(l=60, r=20, t=50, b=40),
                yaxis=dict(title="Volatilidad anualizada (%)", ticksuffix="%"),
                xaxis=dict(title="Fecha"),
                showlegend=False,
            )
            st.plotly_chart(fig_garch_vol, use_container_width=True)

        with st.expander("📌 Paso 4 — Pronóstico de volatilidad a 1 día", expanded=True):
            st.markdown(
                "A diferencia de EWMA que usa la varianza del último día observado, "
                "GARCH genera un **pronóstico formal** de la varianza del día *siguiente* "
                "usando la ecuación del modelo con los valores más recientes:"
            )
            eps2_last = port_ret.iloc[-1] ** 2
            sigma2_last = garch_cond_vol.iloc[-1] ** 2
            sigma2_next = omega + alpha * eps2_last + beta * sigma2_last
            st.latex(
                rf"\hat{{\sigma}}^2_{{T+1}} = {omega:.6f} + {alpha:.4f} \times "
                rf"{eps2_last:.6f} + {beta:.4f} \times {sigma2_last:.6f}"
            )
            g1, g2, g3 = st.columns(3)
            g1.metric("ε²_T (retorno² último día)", f"{eps2_last:.6f}")
            g2.metric("σ²_T (var. condicional hoy)", f"{sigma2_last:.6f}")
            g3.metric("σ²_{T+1} (pronóstico)", f"{sigma2_next:.6f}")
            st.info(
                f"**σ_{{T+1}} = {sigma_garch_next:.4f}% diario** → "
                f"**{sigma_garch_next*np.sqrt(252):.2f}% anual**"
            )

        with st.expander("📌 Paso 5 — VaR diario con GARCH", expanded=True):
            var_garch = abs(t3_z) * (sigma_garch_next / 100) * monto
            es_garch  = (sigma_garch_next / 100) * monto * phi_z_t3 / alpha_t3
            st.markdown(
                "Usamos el pronóstico σ_{T+1} como estimado de riesgo para el día de mañana:"
            )
            st.latex(r"\text{VaR}^{GARCH}_\$ = \text{Monto} \times \hat{\sigma}_{T+1} \times |z_\alpha|")
            st.latex(
                rf"\text{{VaR}}^{{GARCH}}_\$ = \${monto:,.0f} \times "
                rf"{sigma_garch_next/100:.6f} \times {abs(t3_z):.4f} = \${var_garch:,.2f}"
            )
            st.success(f"**VaR GARCH ({int(confianza*100)}%): ${var_garch:,.2f} USD**")

            st.markdown("**Expected Shortfall GARCH:**")
            st.latex(
                r"ES^{GARCH}_\$ = \text{Monto} \times \hat{\sigma}_{T+1} \times \frac{\varphi(|z|)}{\alpha}"
            )
            st.latex(
                rf"ES^{{GARCH}}_\$ = \${monto:,.0f} \times {sigma_garch_next/100:.6f} \times "
                rf"\frac{{{phi_z_t3:.6f}}}{{{alpha_t3:.2f}}} = \${es_garch:,.2f}"
            )
            st.warning(f"**ES GARCH ({int(confianza*100)}%): ${es_garch:,.2f} USD**")

        # ── Comparativa final ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Comparativa de los tres enfoques")
        st.markdown(
            "Cada modelo usa una forma distinta de estimar la volatilidad actual del portafolio:"
        )
        comp_data = {
            "Modelo": ["Clásico (rolling 30d)", "EWMA", "GARCH(1,1)"],
            "σ diaria (%)": [
                f"{roll_vol.iloc[-1]:.4f}",
                f"{ewma_vol.iloc[-1]:.4f}",
                f"{sigma_garch_next:.4f}",
            ],
            "σ anualizada (%)": [
                f"{vol_actual_roll:.2f}",
                f"{vol_actual_ewma:.2f}",
                f"{vol_actual_garch:.2f}",
            ],
            f"VaR {int(confianza*100)}% (USD)": [
                f"${abs(t3_z)*(roll_vol.iloc[-1]/100)*monto:,.2f}",
                f"${var_ewma:,.2f}",
                f"${var_garch:,.2f}",
            ],
            f"ES {int(confianza*100)}% (USD)": [
                f"${(roll_vol.iloc[-1]/100)*monto*phi_z_t3/alpha_t3:,.2f}",
                f"${es_ewma:,.2f}",
                f"${es_garch:,.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(comp_data).set_index("Modelo"), use_container_width=True)

        st.info(
            "💡 **¿Por qué difieren?**  \n"
            "- **Clásico**: asigna el mismo peso a todos los días de la ventana de 30 días. "
            "Es lento en reaccionar y lento en olvidar.  \n"
            "- **EWMA**: da más peso a los días recientes. Reacciona rápido a los choques "
            "pero también los olvida rápido.  \n"
            "- **GARCH**: estima la velocidad de reacción (α) y de olvido (β) directamente "
            "de los datos. Es el más flexible y el preferido en la práctica financiera."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Backtesting del VaR")
    st.markdown(
        "El **backtesting** evalúa si un modelo de VaR es preciso: comparamos el VaR "
        "predicho cada día contra la pérdida real observada al día siguiente. "
        "Si el modelo es correcto, las **excepciones** (días donde la pérdida supera el VaR) "
        "deben ocurrir exactamente con la frecuencia prometida por el nivel de confianza."
    )

    if not pesos_ok:
        st.warning("Define los pesos del portafolio en la barra lateral para continuar.")
        st.stop()

    N_TEST = 100

    # Comprobar que hay suficientes datos
    _all_log = pd.DataFrame({
        t: np.log(s / s.shift(1)).dropna() * 100 for t, s in data.items()
    }).dropna()
    _all_port = _all_log @ pesos

    if len(_all_port) <= N_TEST + 50:
        st.error(f"Se necesitan más de {N_TEST + 50} días de retornos. Amplía el rango de fechas.")
        st.stop()

    param_col, btn_col = st.columns([4, 1])
    param_col.info(
        f"💼 **${monto:,.0f} USD** · confianza **{int(confianza*100)}%** · "
        f"λ = {lambda_ewma:.2f} · "
        f"pesos: {' / '.join([f'{n} {p*100:.1f}%' for n, p in zip(names, pesos)])}  \n"
        f"Ventana de entrenamiento: **{len(_all_port) - N_TEST} días** · "
        f"Ventana de test: **{N_TEST} días**"
    )
    run_bt = btn_col.button(
        "⟳ Ejecutar BT", type="primary", use_container_width=True, key="btn_bt",
        help="Corre el backtest con los parámetros actuales.",
    )

    if run_bt:
        st.session_state.pop("bt_results", None)

    if not run_bt and "bt_results" not in st.session_state:
        st.info("Haz clic en **⟳ Ejecutar BT** para correr el backtest.")
        st.stop()

    # ── Cálculo del backtest ───────────────────────────────────────────────
    if run_bt:
        with st.spinner("Ejecutando backtest (GARCH puede tardar unos segundos)..."):
            _z     = stats.norm.ppf(1 - confianza)
            _alpha = 1 - confianza
            WINDOW = 250  # días de entrenamiento para métodos rolling

            # Separar entrenamiento / test
            _train = _all_port.iloc[:-N_TEST]
            _test  = _all_port.iloc[-N_TEST:]

            # EWMA: calcular sobre serie completa, usar valores t-1 en test
            _ewma_full = compute_ewma_vol(_all_port, lambda_ewma)

            # GARCH: ajustar sobre entrenamiento, obtener vol condicional + forecasts en test
            _gc_full, _, _ = fit_garch(_all_port.values, _all_port.index)
            # La vol condicional en t es la estimada usando datos hasta t-1
            _garch_sigma_test = _gc_full.iloc[-N_TEST:].values  # diaria en %

            # Calcular VaR diario (en %) para cada método en la ventana de test
            _var_hist_pct = np.empty(N_TEST)
            _var_vc_pct   = np.empty(N_TEST)
            _var_ewma_pct = np.empty(N_TEST)

            n_total = len(_all_port)
            for i in range(N_TEST):
                t = n_total - N_TEST + i          # índice absoluto del día de test
                w_start = max(0, t - WINDOW)
                _window_ret = _all_port.iloc[w_start:t]

                # Histórico
                _var_hist_pct[i] = np.percentile(_window_ret, _alpha * 100)

                # Var-Cov
                _var_vc_pct[i] = _z * _window_ret.std()

                # EWMA: usar vol del día anterior
                _var_ewma_pct[i] = _z * _ewma_full.iloc[t - 1]

            # GARCH: usar sigma condicional estimada para el período de test
            _var_garch_pct = _z * _garch_sigma_test   # negativo porque z < 0

            _test_dates  = _test.index
            _actual      = _test.values
            _var_lines   = {
                "Histórico": _var_hist_pct,
                "Var-Cov":   _var_vc_pct,
                "EWMA":      _var_ewma_pct,
                "GARCH":     _var_garch_pct,
            }

            # Excepciones y prueba de Kupiec
            _bt_summary = {}
            for method, var_pct in _var_lines.items():
                exc_mask  = _actual < var_pct
                n_exc     = int(exc_mask.sum())
                lr, pval  = kupiec_test(n_exc, N_TEST, confianza)
                pct_exc   = n_exc / N_TEST * 100
                esperado  = _alpha * 100
                # Zona Basilea (adaptada a 100 días)
                if n_exc <= int(_alpha * N_TEST * 0.8):
                    zona = "🟢 Verde"
                elif n_exc <= int(_alpha * N_TEST * 2.0):
                    zona = "🟡 Amarilla"
                else:
                    zona = "🔴 Roja"
                _bt_summary[method] = dict(
                    n_exc=n_exc, pct_exc=pct_exc, esperado=esperado,
                    lr=lr, pval=pval, zona=zona,
                    acepta=(pval >= 0.05),
                )

        st.session_state["bt_results"] = dict(
            var_lines=_var_lines, actual=_actual,
            test_dates=_test_dates, summary=_bt_summary,
        )

    # ── Render resultados ─────────────────────────────────────────────────
    _bt = st.session_state["bt_results"]
    _var_lines  = _bt["var_lines"]
    _actual     = _bt["actual"]
    _test_dates = _bt["test_dates"]
    _summary    = _bt["summary"]

    # ── Gráfico combinado ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Retornos reales vs. VaR predicho")
    st.markdown(
        "Las **barras** muestran el retorno diario real del portafolio. "
        "Las **líneas punteadas** son los umbrales de VaR de cada método. "
        "Los **×** marcan las excepciones de cada modelo."
    )
    st.plotly_chart(
        make_bt_chart(_test_dates, _actual, _var_lines),
        use_container_width=True,
    )

    # ── Gráficos individuales ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Detalle por método")
    METHOD_COLORS = {"Histórico": "#2196F3", "Var-Cov": "#4CAF50",
                     "EWMA": "#FF9800", "GARCH": "#E91E63"}
    bt_cols = st.columns(2)
    for idx, (method, var_pct) in enumerate(_var_lines.items()):
        with bt_cols[idx % 2]:
            st.plotly_chart(
                make_bt_single(_test_dates, _actual, var_pct,
                               method, METHOD_COLORS[method]),
                use_container_width=True,
            )

    # ── Tabla de resultados ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Resultados del backtesting")
    st.markdown(
        "La **prueba de Kupiec** (POF — Proportion of Failures) evalúa formalmente "
        "si la tasa de excepciones observada es estadísticamente compatible con el nivel "
        "de confianza del modelo. La hipótesis nula es que el modelo es correcto."
    )
    st.latex(
        r"LR_{POF} = -2\ln\!\left[\frac{p_0^{N_1}(1-p_0)^{N_0}}{\hat{p}^{N_1}(1-\hat{p})^{N_0}}\right] \sim \chi^2(1)"
    )
    st.markdown(
        r"Donde $p_0 = 1 - \text{confianza}$ es la tasa esperada, "
        r"$\hat{p} = N_1 / T$ la tasa observada, $N_1$ = excepciones, $N_0 = T - N_1$."
    )

    _rows_bt = []
    for method, s in _summary.items():
        _rows_bt.append({
            "Método": method,
            "Excepciones": s["n_exc"],
            "Esperadas": f"{s['esperado']:.1f}%  →  {s['esperado']/100*N_TEST:.1f}",
            "Tasa obs. (%)": f"{s['pct_exc']:.1f}",
            "LR Kupiec": f"{s['lr']:.3f}",
            "p-valor": f"{s['pval']:.4f}",
            "Acepta H₀ (5%)": "✓ Sí" if s["acepta"] else "✗ No",
            "Zona Basilea": s["zona"],
        })

    st.dataframe(
        pd.DataFrame(_rows_bt).set_index("Método"),
        use_container_width=True,
    )

    # ── Métricas destacadas ────────────────────────────────────────────────
    met_cols = st.columns(4)
    for col_m, (method, s) in zip(met_cols, _summary.items()):
        col_m.metric(
            label=method,
            value=f"{s['n_exc']} excepciones",
            delta=f"p-valor Kupiec: {s['pval']:.3f}",
            delta_color="normal" if s["acepta"] else "inverse",
        )

    # ── Conclusión ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Conclusión")

    # Clasificar modelos
    _aceptados = [m for m, s in _summary.items() if s["acepta"]]
    _rechazados = [m for m, s in _summary.items() if not s["acepta"]]
    # Mejor: aceptado con excepciones más cercanas a la esperada
    _esperado_n = _alpha * N_TEST
    _mejor = min(
        _summary.keys(),
        key=lambda m: abs(_summary[m]["n_exc"] - _esperado_n)
    )

    if _aceptados:
        st.success(
            f"**Modelos que pasan la prueba de Kupiec al 5%:** {', '.join(_aceptados)}.  \n"
            f"El modelo con excepciones más cercanas al valor esperado ({_esperado_n:.1f}) "
            f"es **{_mejor}** con **{_summary[_mejor]['n_exc']} excepciones**."
        )
    else:
        st.error(
            "Ningún modelo pasa la prueba de Kupiec. Esto puede indicar que la ventana "
            "de test incluye un período de alta turbulencia atípica, o que los modelos "
            "requieren recalibración con parámetros distintos."
        )

    if _rechazados:
        st.warning(f"**Modelos rechazados:** {', '.join(_rechazados)}.  \n"
                   "Un p-valor bajo indica que la tasa de excepciones es incompatible "
                   "con el nivel de confianza declarado.")

    st.info(
        "💡 **Interpretación de las zonas de Basilea:**  \n"
        "- 🟢 **Verde**: el número de excepciones es consistente con el modelo — se acepta.  \n"
        "- 🟡 **Amarilla**: posible debilidad del modelo, se requiere análisis adicional.  \n"
        "- 🔴 **Roja**: el modelo subestima sistemáticamente el riesgo — se rechaza.  \n\n"
        "En general, un modelo de VaR **sobrestima** el riesgo si tiene muy pocas excepciones "
        "(muy conservador → capital inmovilizado innecesariamente), y lo **subestima** si tiene "
        "demasiadas (peligroso → capital insuficiente para cubrir pérdidas reales)."
    )
