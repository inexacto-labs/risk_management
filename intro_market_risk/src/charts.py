import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


COLORS = {
    "Papa criolla limpia": "#2D6A4F",
    "Papa criolla sucia": "#40916C",
    "Papa pastusa": "#BC6C25",
    "Papa sabanera": "#6C757D",
}


def _line_width(series: str, highlight: str | None) -> float:
    return 3.2 if series == highlight else 1.8


def _opacity(series: str, highlight: str | None) -> float:
    return 1.0 if series == highlight else 0.55


def _base_layout(title_y: float = 0.96) -> dict:
    return {
        "template": "plotly_white",
        "margin": {"l": 10, "r": 10, "t": 24, "b": 10},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": title_y, "x": 0},
        "hovermode": "x unified",
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }


def make_price_chart(df: pd.DataFrame, series_list: list[str], highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    for series in series_list:
        fig.add_trace(
            go.Scatter(
                x=df["Fecha"],
                y=df[series],
                mode="lines",
                name=series,
                line={"color": COLORS.get(series, "#2D6A4F"), "width": _line_width(series, highlight)},
                opacity=_opacity(series, highlight),
            )
        )
    fig.update_layout(**_base_layout())
    fig.update_yaxes(title="Precio")
    fig.update_xaxes(title=None)
    return fig


def make_return_chart(
    df: pd.DataFrame, series_list: list[str], highlight: str | None = None, periods: int = 1
) -> go.Figure:
    fig = go.Figure()
    for series in series_list:
        fig.add_trace(
            go.Scatter(
                x=df["Fecha"],
                y=df[series] * 100,
                mode="lines",
                name=series,
                line={"color": COLORS.get(series, "#2D6A4F"), "width": _line_width(series, highlight)},
                opacity=_opacity(series, highlight),
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#9AA0A6")
    fig.update_layout(**_base_layout())
    fig.update_yaxes(title=f"Retorno % ({periods} periodo{'s' if periods > 1 else ''})")
    fig.update_xaxes(title=None)
    return fig


def make_base100_chart(df: pd.DataFrame, series_list: list[str], highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    for series in series_list:
        fig.add_trace(
            go.Scatter(
                x=df["Fecha"],
                y=df[series],
                mode="lines",
                name=series,
                line={"color": COLORS.get(series, "#2D6A4F"), "width": _line_width(series, highlight)},
                opacity=_opacity(series, highlight),
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color="#9AA0A6")
    fig.update_layout(**_base_layout())
    fig.update_yaxes(title="Base 100")
    fig.update_xaxes(title=None)
    return fig


def make_histogram_chart(df: pd.DataFrame, selected_series: str) -> go.Figure:
    series = df[selected_series].dropna() * 100
    fig = px.histogram(
        x=series,
        nbins=40,
        labels={"x": "Retorno %"},
        color_discrete_sequence=[COLORS.get(selected_series, "#2D6A4F")],
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Frecuencia")
    fig.update_xaxes(title="Retorno %")
    return fig


def make_rolling_vol_chart(df: pd.DataFrame, series_list: list[str], highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    for series in series_list:
        fig.add_trace(
            go.Scatter(
                x=df["Fecha"],
                y=df[series] * 100,
                mode="lines",
                name=series,
                line={"color": COLORS.get(series, "#2D6A4F"), "width": _line_width(series, highlight)},
                opacity=_opacity(series, highlight),
            )
        )
    fig.update_layout(**_base_layout())
    fig.update_yaxes(title="Volatilidad rolling %")
    fig.update_xaxes(title=None)
    return fig


def make_volatility_bar_chart(summary: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    colors = [COLORS.get(name, "#2D6A4F") if name == highlight else "#B7C9BE" for name in summary["Serie"]]
    fig = go.Figure(
        go.Bar(
            x=summary["Serie"],
            y=summary["Volatilidad"] * 100,
            marker_color=colors,
            hovertemplate="%{x}<br>Volatilidad: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Volatilidad %")
    fig.update_xaxes(title=None)
    return fig


def make_var_distribution_chart(
    returns: pd.Series,
    confidence: float,
    selected_series: str,
) -> go.Figure:
    series = returns.dropna() * 100
    threshold = series.quantile(1 - confidence) if not series.empty else 0
    fig = px.histogram(
        x=series,
        nbins=45,
        labels={"x": "Retorno %"},
        color_discrete_sequence=[COLORS.get(selected_series, "#2D6A4F")],
    )
    fig.add_vline(
        x=threshold,
        line_color="#8B1E3F",
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"Percentil adverso {(1 - confidence) * 100:.0f}%",
        annotation_position="top",
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Frecuencia")
    fig.update_xaxes(title="Retorno %")
    return fig


def make_var_tail_chart(
    returns: pd.Series,
    confidence: float,
    selected_series: str,
) -> go.Figure:
    series = returns.dropna().sort_values() * 100
    threshold = series.quantile(1 - confidence) if not series.empty else 0
    colors = ["#8B1E3F" if value <= threshold else COLORS.get(selected_series, "#2D6A4F") for value in series]
    fig = go.Figure(
        go.Bar(
            x=list(range(1, len(series) + 1)),
            y=series.values,
            marker_color=colors,
            hovertemplate="Escenario %{x}<br>Retorno %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=threshold, line_color="#8B1E3F", line_dash="dash")
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Retorno %")
    fig.update_xaxes(title="Escenarios ordenados")
    return fig


def make_bond_price_curve_chart(curve_df: pd.DataFrame, base_ytm: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df["Tasa"] * 100,
            y=curve_df["Precio"],
            mode="lines",
            line={"color": "#BC6C25", "width": 3},
            name="Precio del bono",
        )
    )
    fig.add_vline(x=base_ytm * 100, line_dash="dash", line_color="#2D6A4F")
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Precio del bono")
    fig.update_xaxes(title="Tasa de rendimiento %")
    return fig


def make_bond_cashflow_chart(cashflows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cashflows["Tiempo (años)"],
            y=cashflows["Flujo"],
            marker_color="#40916C",
            hovertemplate="Año %{x}<br>Flujo %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Flujo")
    fig.update_xaxes(title="Tiempo (años)")
    return fig


def make_duration_approx_chart(shocks_bps: list[int], actual_pct: list[float], duration_pct: list[float], duration_conv_pct: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shocks_bps, y=actual_pct, mode="lines", name="Cambio real", line={"color": "#1F2A1F", "width": 3}))
    fig.add_trace(go.Scatter(x=shocks_bps, y=duration_pct, mode="lines", name="Solo duración", line={"color": "#2D6A4F", "width": 2}))
    fig.add_trace(go.Scatter(x=shocks_bps, y=duration_conv_pct, mode="lines", name="Duración + convexidad", line={"color": "#BC6C25", "width": 2}))
    fig.add_hline(y=0, line_dash="dot", line_color="#9AA0A6")
    fig.update_layout(**_base_layout())
    fig.update_yaxes(title="Cambio de precio %")
    fig.update_xaxes(title="Shock de tasa (pbs)")
    return fig


def make_monte_carlo_loss_chart(losses: np.ndarray, confidence: float) -> go.Figure:
    threshold = np.quantile(losses, confidence) if losses.size else 0.0
    fig = px.histogram(
        x=losses,
        nbins=50,
        labels={"x": "Pérdida"},
        color_discrete_sequence=["#8B1E3F"],
    )
    fig.add_vline(
        x=threshold,
        line_color="#1F2A1F",
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"VaR {confidence * 100:.0f}%",
        annotation_position="top",
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Frecuencia")
    fig.update_xaxes(title="Pérdida simulada")
    return fig


def make_rate_scenarios_chart(delta_rates: np.ndarray) -> go.Figure:
    fig = px.histogram(
        x=delta_rates * 10000,
        nbins=50,
        labels={"x": "Cambio en tasa (pbs)"},
        color_discrete_sequence=["#2D6A4F"],
    )
    fig.update_layout(**_base_layout(), showlegend=False)
    fig.update_yaxes(title="Frecuencia")
    fig.update_xaxes(title="Cambio en tasa (pbs)")
    return fig


def make_rate_path_chart(rate_paths: np.ndarray, max_paths: int = 50) -> go.Figure:
    fig = go.Figure()
    if rate_paths.size == 0:
        fig.update_layout(**_base_layout(), showlegend=False)
        return fig
    sample = rate_paths[:max_paths]
    x_axis = list(range(sample.shape[1]))
    for path in sample:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=path * 100,
                mode="lines",
                line={"color": "rgba(45,106,79,0.18)", "width": 1.1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    mean_path = sample.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=mean_path * 100,
            mode="lines",
            line={"color": "#1F2A1F", "width": 3},
            name="Trayectoria promedio",
        )
    )
    fig.update_layout(**_base_layout())
    fig.update_yaxes(title="Tasa %")
    fig.update_xaxes(title="Día del horizonte")
    return fig
