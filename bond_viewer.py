"""
bond_viewer.py — Análisis de Renta Fija
Streamlit app para valoración de bonos, duración, convexidad y VaR.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import pandas_datareader as pdr
from scipy.stats import norm, chi2

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Renta Fija — EAFIT",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .metric-label { color: #a6adc8; font-size: 0.78rem; margin-bottom: 2px; }
    .metric-value { color: #cdd6f4; font-size: 1.3rem; font-weight: 700; }
    .metric-sub   { color: #6c7086; font-size: 0.72rem; }
    .step-header  { color: #89b4fa; font-size: 1.05rem; font-weight: 600;
                    border-left: 3px solid #89b4fa; padding-left: 8px; margin: 18px 0 6px; }
    .formula-box  { background: #181825; border: 1px solid #45475a; border-radius: 6px;
                    padding: 10px 14px; font-family: monospace; font-size: 0.85rem;
                    color: #cba6f7; margin: 6px 0 12px; }
    .info-banner  { background: #1e3a5f; border-left: 4px solid #89b4fa;
                    padding: 10px 14px; border-radius: 4px; margin: 8px 0; }
    .warn-banner  { background: #3d2b1f; border-left: 4px solid #fab387;
                    padding: 10px 14px; border-radius: 4px; margin: 8px 0; }
    .section-title { font-size: 1.15rem; font-weight: 700; color: #cdd6f4;
                     border-bottom: 1px solid #313244; padding-bottom: 4px; margin: 20px 0 10px; }
</style>
""", unsafe_allow_html=True)

# ── FRED codes ────────────────────────────────────────────────────────────────
FRED_CODES = {
    "1 año (DGS1)":   "DGS1",
    "2 años (DGS2)":  "DGS2",
    "5 años (DGS5)":  "DGS5",
    "7 años (DGS7)":  "DGS7",
    "10 años (DGS10)": "DGS10",
    "20 años (DGS20)": "DGS20",
    "30 años (DGS30)": "DGS30",
}
FREQ_OPTS = {"Anual (1/año)": 1, "Semestral (2/año)": 2, "Trimestral (4/año)": 4}
CONF_OPTS = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_MAIN   = "#89b4fa"   # blue
COLOR_ACCENT = "#a6e3a1"   # green
COLOR_WARN   = "#fab387"   # orange
COLOR_DANGER = "#f38ba8"   # red
COLOR_PURPLE = "#cba6f7"
COLOR_YELLOW = "#f9e2af"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — Bond Math
# ═══════════════════════════════════════════════════════════════════════════════

def build_cashflows(face: float, coupon_rate: float, freq: int,
                    settle: date, maturity: date) -> pd.DataFrame:
    """Build a DataFrame of cash-flow dates and amounts."""
    months_per_period = 12 // freq
    coupon_amount = face * coupon_rate / freq

    # Generate payment dates backwards from maturity
    payment_dates = []
    d = maturity
    while d > settle:
        payment_dates.append(d)
        # subtract months_per_period months
        month = d.month - months_per_period
        year  = d.year
        while month <= 0:
            month += 12
            year  -= 1
        last_day = (date(year, month % 12 + 1, 1) - timedelta(days=1)).day if month < 12 else 31
        day = min(d.day, last_day)
        d = date(year, month, day)

    payment_dates = sorted(payment_dates)

    rows = []
    for i, pd_date in enumerate(payment_dates, 1):
        t_years = (pd_date - settle).days / 365.0
        cf = coupon_amount + (face if pd_date == maturity else 0)
        rows.append({"n": i, "fecha": pd_date, "t_años": t_years,
                     "flujo": cf, "es_principal": pd_date == maturity})
    return pd.DataFrame(rows)


def price_bond(cf_df: pd.DataFrame, ytm: float, freq: int) -> float:
    """DCF bond price given cash-flow DataFrame and YTM (annual decimal)."""
    r = ytm / freq
    total = 0.0
    for _, row in cf_df.iterrows():
        t_periods = row["t_años"] * freq
        total += row["flujo"] / (1 + r) ** t_periods
    return total


def price_bond_from_ytm(face: float, coupon_rate: float, freq: int,
                         settle: date, maturity: date, ytm: float) -> float:
    cf_df = build_cashflows(face, coupon_rate, freq, settle, maturity)
    return price_bond(cf_df, ytm, freq)


def macaulay_duration(cf_df: pd.DataFrame, ytm: float, freq: int,
                      price: float) -> float:
    r = ytm / freq
    dur = 0.0
    for _, row in cf_df.iterrows():
        t_periods = row["t_años"] * freq
        pv = row["flujo"] / (1 + r) ** t_periods
        dur += row["t_años"] * pv
    return dur / price


def convexity(cf_df: pd.DataFrame, ytm: float, freq: int, price: float) -> float:
    r = ytm / freq
    conv = 0.0
    for _, row in cf_df.iterrows():
        t_periods = row["t_años"] * freq
        pv = row["flujo"] / (1 + r) ** t_periods
        conv += t_periods * (t_periods + 1) * pv
    conv /= price * (1 + r) ** 2
    return conv / freq ** 2   # annualise


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — FRED Data
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def get_fred_yields(series: str, start: str = "2015-01-01") -> pd.Series:
    df = pdr.get_data_fred(series, start=start)
    s = df[series].dropna() / 100.0   # percent → decimal
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — Kupiec POF
# ═══════════════════════════════════════════════════════════════════════════════

def kupiec_test(n_exc: int, n_days: int, confidence: float):
    alpha = 1 - confidence
    if n_exc == 0:
        return 0.0, 1.0
    p_hat = n_exc / n_days
    if p_hat >= 1.0:
        return np.inf, 0.0
    N1, N0 = n_exc, n_days - n_exc
    lr = -2 * (N1 * np.log(alpha / p_hat) + N0 * np.log((1 - alpha) / (1 - p_hat)))
    return float(lr), float(1 - chi2.cdf(lr, df=1))


def basel_zone(n_exc: int) -> tuple:
    if n_exc <= 4:
        return "🟢 Verde", "#a6e3a1"
    elif n_exc <= 9:
        return "🟡 Amarilla", "#f9e2af"
    else:
        return "🔴 Roja", "#f38ba8"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📐 Renta Fija")
    st.caption("Valoración · Duración · VaR")
    st.divider()

    st.markdown("**Parámetros del Bono**")
    face_value  = st.number_input("Valor Nominal ($)", min_value=100.0,
                                   max_value=1_000_000.0, value=1_000.0, step=100.0)
    coupon_pct  = st.number_input("Tasa Cupón (% anual)", min_value=0.0,
                                   max_value=30.0, value=5.0, step=0.25)
    freq_label  = st.selectbox("Frecuencia de Pago", list(FREQ_OPTS.keys()), index=1)
    freq        = FREQ_OPTS[freq_label]

    today = date.today()
    settle_date  = st.date_input("Fecha de Liquidación", value=today,
                                  min_value=date(2000, 1, 1), max_value=today)
    default_mat  = date(today.year + 10, today.month, today.day)
    maturity_date = st.date_input("Fecha de Vencimiento",
                                   value=default_mat,
                                   min_value=date(today.year + 1, 1, 1),
                                   max_value=date(2060, 12, 31))

    ytm_pct = st.number_input("YTM actual (% anual)", min_value=0.01,
                               max_value=30.0, value=5.50, step=0.05)

    st.divider()
    st.markdown("**Parámetros de Riesgo**")
    monto_inv   = st.number_input("Monto Invertido ($)", min_value=1_000.0,
                                   value=100_000.0, step=1_000.0, format="%.0f")
    conf_label  = st.selectbox("Nivel de Confianza", list(CONF_OPTS.keys()), index=1)
    confianza   = CONF_OPTS[conf_label]

    st.divider()
    st.markdown("**Datos de Mercado (FRED)**")
    tenor_label = st.selectbox("Tenor de Referencia", list(FRED_CODES.keys()), index=4)
    fred_code   = FRED_CODES[tenor_label]

    st.divider()
    calcular_btn = st.button("⟳ Calcular todo", type="primary", use_container_width=True)

# ── Derived base parameters ───────────────────────────────────────────────────
coupon_rate = coupon_pct / 100.0
ytm         = ytm_pct   / 100.0
z_score     = norm.ppf(confianza)
alpha_level = 1 - confianza

if maturity_date <= settle_date:
    st.error("⚠️ La fecha de vencimiento debe ser posterior a la fecha de liquidación.")
    st.stop()

# ── Build cash flows & price ──────────────────────────────────────────────────
cf_df = build_cashflows(face_value, coupon_rate, freq, settle_date, maturity_date)
if cf_df.empty:
    st.error("No se generaron flujos de caja. Revisa las fechas.")
    st.stop()

precio = price_bond(cf_df, ytm, freq)
cf_df["pv"] = cf_df.apply(
    lambda row: row["flujo"] / (1 + ytm / freq) ** (row["t_años"] * freq), axis=1
)
mac_dur = macaulay_duration(cf_df, ytm, freq, precio)
mod_dur = mac_dur / (1 + ytm / freq)
dv01    = mod_dur * precio * 0.0001          # price change per 1bp
conv    = convexity(cf_df, ytm, freq, precio)
n_bonds = monto_inv / precio                 # number of bonds

# ── FRED data load (cached) ───────────────────────────────────────────────────
with st.spinner(f"Cargando yields FRED ({fred_code})…"):
    try:
        yields_series = get_fred_yields(fred_code)
        fred_ok = True
    except Exception as e:
        st.warning(f"No se pudo cargar datos FRED: {e}")
        yields_series = pd.Series(dtype=float)
        fred_ok = False

yield_changes = yields_series.diff().dropna() if fred_ok else pd.Series(dtype=float)

# ── Session state for recalculation ──────────────────────────────────────────
if calcular_btn:
    for key in ["var_computed", "bt_computed"]:
        st.session_state.pop(key, None)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "💵 Valoración",
    "📏 Duración & Convexidad",
    "⚠️ VaR del Bono",
    "🔬 Backtesting",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Valoración
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Valoración por Descuento de Flujos (DCF)</div>',
                unsafe_allow_html=True)

    # ── Step 1: Parameters ──
    st.markdown('<div class="step-header">Paso 1 — Parámetros del Bono</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    def mbox(label, value, sub=""):
        return f"""<div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>"""

    with c1:
        st.markdown(mbox("Valor Nominal", f"${face_value:,.0f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(mbox("Tasa Cupón", f"{coupon_pct:.2f}%", freq_label), unsafe_allow_html=True)
    with c3:
        years_to_mat = (maturity_date - settle_date).days / 365
        st.markdown(mbox("Plazo", f"{years_to_mat:.2f} años",
                          f"Vence {maturity_date.strftime('%d %b %Y')}"), unsafe_allow_html=True)
    with c4:
        st.markdown(mbox("YTM", f"{ytm_pct:.2f}%", "Rendimiento al vencimiento"), unsafe_allow_html=True)

    # ── Step 2: Formula ──
    st.markdown('<div class="step-header">Paso 2 — Fórmula de Valoración</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="formula-box">
P = Σ [ C / (1 + r)ᵗ ]  +  F / (1 + r)ᵀ

donde:
  C = cupón por período = F × (tasa_cupón / m)
  r = YTM por período   = YTM_anual / m
  m = pagos por año     (1, 2 ó 4)
  t = período de cada flujo
  T = número total de períodos
  F = valor nominal (principal)
</div>
""", unsafe_allow_html=True)

    coupon_per_period = face_value * coupon_rate / freq
    r_per_period      = ytm / freq
    total_periods     = len(cf_df)

    col_f, col_v = st.columns([1, 1])
    with col_f:
        st.markdown(f"""
<div class="info-banner">
<b>Cupón por período:</b> ${coupon_per_period:,.2f}<br>
<b>Tasa por período (r):</b> {r_per_period*100:.4f}%<br>
<b>Total de períodos:</b> {total_periods}
</div>
""", unsafe_allow_html=True)
    with col_v:
        pv_coupons   = cf_df.loc[~cf_df["es_principal"], "pv"].sum()
        pv_principal = cf_df.loc[cf_df["es_principal"],  "pv"].sum() - \
                       cf_df.loc[cf_df["es_principal"],  "pv"].sum() + \
                       cf_df.loc[cf_df["es_principal"],  "flujo"].iloc[-1] / \
                       (1 + r_per_period) ** (cf_df.loc[cf_df["es_principal"], "t_años"].iloc[-1] * freq)
        pv_coupons_only = sum(
            row["flujo"] / (1 + r_per_period) ** (row["t_años"] * freq)
            for _, row in cf_df.iterrows() if not row["es_principal"]
        )
        # also separate principal PV from last coupon row
        last_row = cf_df[cf_df["es_principal"]].iloc[-1]
        t_last   = last_row["t_años"] * freq
        pv_principal_only = face_value / (1 + r_per_period) ** t_last
        pv_last_coupon    = coupon_per_period / (1 + r_per_period) ** t_last
        pv_coupons_all    = pv_coupons_only + pv_last_coupon

        premium_discount = precio - face_value
        pd_label = "Prima" if premium_discount > 0 else ("Descuento" if premium_discount < 0 else "Par")
        st.markdown(f"""
<div class="info-banner">
<b>PV cupones:</b> ${pv_coupons_all:,.2f}<br>
<b>PV principal:</b> ${pv_principal_only:,.2f}<br>
<b>{pd_label}:</b> ${abs(premium_discount):,.2f}
</div>
""", unsafe_allow_html=True)

    # ── Step 3: Price result ──
    st.markdown('<div class="step-header">Paso 3 — Precio del Bono</div>',
                unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown(mbox("Precio Limpio (DCF)", f"${precio:,.4f}",
                          f"${precio / face_value * 100:.4f} por $100 de nominal"),
                    unsafe_allow_html=True)
    with pc2:
        st.markdown(mbox("Monto Invertido", f"${monto_inv:,.0f}",
                          f"≈ {n_bonds:.4f} bonos"), unsafe_allow_html=True)
    with pc3:
        pct_label = "Prima" if premium_discount > 0 else "Descuento"
        st.markdown(mbox(pct_label, f"${abs(premium_discount):,.4f}",
                          f"{abs(premium_discount/face_value)*100:.4f}% del nominal"),
                    unsafe_allow_html=True)

    # ── Step 4: Cash flow table ──
    st.markdown('<div class="step-header">Paso 4 — Tabla de Flujos Descontados</div>',
                unsafe_allow_html=True)
    display_cf = cf_df[["n", "fecha", "t_años", "flujo", "pv"]].copy()
    display_cf.columns = ["N°", "Fecha", "Tiempo (años)", "Flujo ($)", "Valor Presente ($)"]
    display_cf["Tiempo (años)"]    = display_cf["Tiempo (años)"].round(4)
    display_cf["Flujo ($)"]        = display_cf["Flujo ($)"].map("${:,.2f}".format)
    display_cf["Valor Presente ($)"] = display_cf["Valor Presente ($)"].map("${:,.4f}".format)
    display_cf["Fecha"]            = display_cf["Fecha"].astype(str)
    st.dataframe(display_cf, use_container_width=True, hide_index=True)

    # ── Step 5: Charts ──
    st.markdown('<div class="step-header">Paso 5 — Visualización de Flujos</div>',
                unsafe_allow_html=True)

    fig_cf = make_subplots(rows=1, cols=2,
                            subplot_titles=("Flujos de Caja por Período",
                                            "Estructura de Precio (VPN)"),
                            column_widths=[0.55, 0.45],
                            specs=[[{"type": "xy"}, {"type": "domain"}]])

    # Bar chart of cash flows
    colors_bar = [COLOR_WARN if row["es_principal"] else COLOR_MAIN
                  for _, row in cf_df.iterrows()]
    fig_cf.add_trace(go.Bar(
        x=[str(d) for d in cf_df["fecha"]],
        y=cf_df["flujo"],
        marker_color=colors_bar,
        name="Flujo ($)",
        text=[f"${v:,.0f}" for v in cf_df["flujo"]],
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)
    fig_cf.add_trace(go.Bar(
        x=[str(d) for d in cf_df["fecha"]],
        y=cf_df["pv"],
        marker_color=[COLOR_ACCENT if row["es_principal"] else "rgba(137,180,250,0.4)"
                      for _, row in cf_df.iterrows()],
        name="VP ($)",
        showlegend=False,
    ), row=1, col=1)

    # Pie chart of price composition
    fig_cf.add_trace(go.Pie(
        labels=["VP Cupones", "VP Principal"],
        values=[pv_coupons_all, pv_principal_only],
        marker_colors=[COLOR_MAIN, COLOR_WARN],
        hole=0.4,
        textinfo="label+percent",
        showlegend=True,
    ), row=1, col=2)

    fig_cf.update_layout(
        template=PLOTLY_TEMPLATE,
        height=380,
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation="h", y=-0.12),
        bargap=0.25,
    )
    fig_cf.update_xaxes(tickangle=-45, row=1, col=1)
    st.plotly_chart(fig_cf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Duración & Convexidad
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Duración, DV01 y Convexidad</div>',
                unsafe_allow_html=True)

    # ── Step 1: Macaulay Duration ──
    st.markdown('<div class="step-header">Paso 1 — Duración de Macaulay</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="formula-box">
D_Mac = Σ [ t × PV(Cᵢ) ] / P

Promedio ponderado del tiempo de cobro de los flujos,
usando el valor presente de cada flujo como ponderador.
</div>
""", unsafe_allow_html=True)

    # Show weighted contribution table (condensed)
    cf_dur = cf_df.copy()
    cf_dur["peso"]    = cf_dur["pv"] / precio
    cf_dur["t×peso"]  = cf_dur["t_años"] * cf_dur["peso"]

    st.markdown("**Contribución de cada flujo a la Duración:**")
    display_dur = cf_dur[["n", "fecha", "t_años", "pv", "peso", "t×peso"]].copy()
    display_dur.columns = ["N°", "Fecha", "t (años)", "VP ($)", "Peso", "t × Peso"]
    display_dur["VP ($)"]   = display_dur["VP ($)"].map("${:,.4f}".format)
    display_dur["Peso"]     = display_dur["Peso"].map("{:.6f}".format)
    display_dur["t × Peso"] = display_dur["t × Peso"].map("{:.6f}".format)
    display_dur["t (años)"] = display_dur["t (años)"].round(4)
    display_dur["Fecha"]    = display_dur["Fecha"].astype(str)
    st.dataframe(display_dur, use_container_width=True, hide_index=True)

    # ── Step 2: Modified Duration, DV01, Convexity ──
    st.markdown('<div class="step-header">Paso 2 — Duración Modificada, DV01 y Convexidad</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="formula-box">
D_Mod = D_Mac / (1 + r)     donde r = YTM / m = {ytm_pct:.2f}% / {freq} = {ytm/freq*100:.4f}%

DV01  = D_Mod × P × 0.0001  (cambio en precio por 1 punto base)

Convexidad = Σ [ t×(t+1) × VP(Cᵢ) ] / [ P × (1+r)² × m² ]
</div>
""", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(mbox("Duración Macaulay", f"{mac_dur:.4f} años",
                          "Vida promedio ponderada"), unsafe_allow_html=True)
    with d2:
        st.markdown(mbox("Duración Modificada", f"{mod_dur:.4f} años",
                          "Sensibilidad al YTM"), unsafe_allow_html=True)
    with d3:
        st.markdown(mbox("DV01", f"${dv01:,.4f}",
                          "Por $1 de nominal · 1bp"), unsafe_allow_html=True)
    with d4:
        st.markdown(mbox("Convexidad", f"{conv:.4f}",
                          "Curvatura precio-yield"), unsafe_allow_html=True)

    # ── Interpretation ──
    st.markdown('<div class="step-header">Paso 3 — Interpretación</div>',
                unsafe_allow_html=True)
    delta_ytm_eg = 0.01  # 100 bp
    dp_duration  = -mod_dur * precio * delta_ytm_eg
    dp_convex    = 0.5 * conv * precio * delta_ytm_eg ** 2
    dp_total     = dp_duration + dp_convex

    st.markdown(f"""
<div class="info-banner">
Si el YTM sube <b>100 pb</b> (de {ytm_pct:.2f}% → {ytm_pct+1:.2f}%):

• Efecto Duración:   ΔP ≈ −D_Mod × P × Δy = −{mod_dur:.4f} × ${precio:,.2f} × 0.01
                        = <b>${dp_duration:,.2f}</b><br>
• Efecto Convexidad: ΔP ≈ ½ × Conv × P × Δy² = ½ × {conv:.4f} × ${precio:,.2f} × 0.0001
                        = <b>${dp_convex:,.2f}</b><br>
• <b>Cambio Total Estimado: ${dp_total:,.2f}</b><br><br>
La convexidad <i>reduce</i> la pérdida cuando suben tasas (y amplía la ganancia cuando bajan).
</div>
""", unsafe_allow_html=True)

    # ── Step 4: Price-Yield Curve ──
    st.markdown('<div class="step-header">Paso 4 — Curva Precio-Yield</div>',
                unsafe_allow_html=True)

    ytm_range = np.linspace(max(0.0001, ytm - 0.05), ytm + 0.05, 200)
    prices_exact = [price_bond_from_ytm(face_value, coupon_rate, freq,
                                         settle_date, maturity_date, y)
                    for y in ytm_range]

    # Duration approximation (tangent line)
    prices_dur   = [precio - mod_dur * precio * (y - ytm) for y in ytm_range]
    # Duration + convexity approximation
    prices_conv  = [precio - mod_dur * precio * (y - ytm)
                    + 0.5 * conv * precio * (y - ytm) ** 2 for y in ytm_range]

    fig_py = go.Figure()
    fig_py.add_trace(go.Scatter(x=ytm_range * 100, y=prices_exact,
                                 name="Precio Real", line=dict(color=COLOR_MAIN, width=2.5)))
    fig_py.add_trace(go.Scatter(x=ytm_range * 100, y=prices_dur,
                                 name="Aprox. Duración", line=dict(color=COLOR_WARN, width=1.5,
                                                                    dash="dash")))
    fig_py.add_trace(go.Scatter(x=ytm_range * 100, y=prices_conv,
                                 name="Aprox. Dur + Convex.", line=dict(color=COLOR_ACCENT, width=1.5,
                                                                         dash="dot")))
    fig_py.add_vline(x=ytm_pct, line_dash="dot", line_color=COLOR_PURPLE, line_width=1.5)
    fig_py.add_annotation(x=ytm_pct, y=precio, text=f"  YTM={ytm_pct:.2f}%<br>  P=${precio:,.2f}",
                           showarrow=True, arrowhead=2, arrowcolor=COLOR_PURPLE,
                           font=dict(color=COLOR_PURPLE, size=11))

    fig_py.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        xaxis_title="YTM (%)",
        yaxis_title="Precio ($)",
        title="Curva Precio-Yield: Real vs Aproximaciones Lineales y Cuadráticas",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=55, b=80, l=60, r=40),
    )
    st.plotly_chart(fig_py, use_container_width=True)

    st.markdown("""
<div class="warn-banner">
<b>Nota:</b> La curva precio-yield es <i>convexa</i> — la duración (línea recta) subestima el precio
cuando el YTM sube y lo sobreestima cuando baja. La corrección de convexidad (curva punteada) captura
casi perfectamente el precio real para movimientos moderados de tasas.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — VaR del Bono
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Valor en Riesgo del Bono</div>',
                unsafe_allow_html=True)

    if not fred_ok or yield_changes.empty:
        st.error("No se pudieron cargar datos FRED para calcular el VaR. Verifica la conexión.")
        st.stop()

    # ── Info banner ──
    n_obs = len(yield_changes)
    sigma_dy = float(yield_changes.std())
    st.markdown(f"""
<div class="info-banner">
<b>Datos FRED ({fred_code}):</b> {n_obs:,} cambios diarios de yield
 · Período: {yield_changes.index[0].strftime("%d %b %Y")} → {yield_changes.index[-1].strftime("%d %b %Y")}
<br><b>Volatilidad histórica Δy:</b> {sigma_dy*10000:.2f} pb/día
 · <b>Confianza:</b> {conf_label} (z = {z_score:.4f})
 · <b>Monto Invertido:</b> ${monto_inv:,.0f}
</div>
""", unsafe_allow_html=True)

    # ── Step 1: Yield data visualization ──
    st.markdown('<div class="step-header">Paso 1 — Yields Históricas y Cambios Diarios</div>',
                unsafe_allow_html=True)

    fig_y = make_subplots(rows=2, cols=1,
                           subplot_titles=("Nivel del Yield (%)", "Cambio Diario en Yield (pb)"),
                           vertical_spacing=0.12)
    fig_y.add_trace(go.Scatter(x=yields_series.index, y=yields_series * 100,
                                line=dict(color=COLOR_MAIN, width=1.5), name="Yield (%)"),
                    row=1, col=1)
    fig_y.add_trace(go.Bar(x=yield_changes.index, y=yield_changes * 10000,
                            marker_color=np.where(yield_changes >= 0, COLOR_DANGER, COLOR_ACCENT),
                            name="Δy (pb)"), row=2, col=1)

    fig_y.add_hline(y=ytm_pct, line_dash="dot", line_color=COLOR_YELLOW,
                    annotation_text=f"YTM actual: {ytm_pct:.2f}%",
                    annotation_position="bottom right", row=1, col=1)

    fig_y.update_layout(template=PLOTLY_TEMPLATE, height=420,
                         margin=dict(t=40, b=40, l=50, r=40),
                         showlegend=False)
    st.plotly_chart(fig_y, use_container_width=True)

    # ── Step 2: Duration-based VaR ──
    st.markdown('<div class="step-header">Paso 2 — VaR por Duración Modificada</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="formula-box">
VaR_dur = D_Mod × Monto × |Δy_VaR|

donde:
  Δy_VaR = z × σ(Δy)       para VaR paramétrico (normal)
  Δy_VaR = percentil empírico de Δy   para VaR histórico

  D_Mod  = {mod_dur:.4f} años
  Monto  = ${monto_inv:,.0f}
  σ(Δy)  = {sigma_dy*10000:.4f} pb/día
</div>
""", unsafe_allow_html=True)

    # Parametric
    delta_y_param  = z_score * sigma_dy
    var_dur_param  = mod_dur * monto_inv * abs(delta_y_param)
    # Historical
    delta_y_hist   = float(yield_changes.quantile(1 - confianza))   # negative: yield falls → bond gains; we want loss side
    # For bonds: loss occurs when yield RISES (positive Δy)
    delta_y_hist_up = float(yield_changes.quantile(confianza))
    var_dur_hist   = mod_dur * monto_inv * abs(delta_y_hist_up)

    # ES (Expected Shortfall)
    tail_changes = yield_changes[yield_changes >= yield_changes.quantile(confianza)]
    es_delta_y   = float(tail_changes.mean()) if len(tail_changes) > 0 else delta_y_hist_up
    es_dur_hist  = mod_dur * monto_inv * abs(es_delta_y)
    es_dur_param = mod_dur * monto_inv * (norm.pdf(z_score) / (1 - confianza)) * sigma_dy

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.markdown(mbox(f"VaR Dur. Paramétrico ({conf_label})",
                          f"${var_dur_param:,.0f}",
                          f"Δy = {delta_y_param*10000:.2f} pb"),
                    unsafe_allow_html=True)
    with v2:
        st.markdown(mbox(f"VaR Dur. Histórico ({conf_label})",
                          f"${var_dur_hist:,.0f}",
                          f"Δy = {delta_y_hist_up*10000:.2f} pb"),
                    unsafe_allow_html=True)
    with v3:
        st.markdown(mbox(f"ES Dur. Paramétrico ({conf_label})",
                          f"${es_dur_param:,.0f}",
                          "Expected Shortfall"), unsafe_allow_html=True)
    with v4:
        st.markdown(mbox(f"ES Dur. Histórico ({conf_label})",
                          f"${es_dur_hist:,.0f}",
                          "Expected Shortfall"), unsafe_allow_html=True)

    # ── Step 3: Full repricing VaR ──
    st.markdown('<div class="step-header">Paso 3 — VaR por Repricing Completo</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="formula-box">
VaR_reprice = |P(YTM + Δy_VaR) − P(YTM)| × n_bonos

Recalcula el precio exacto del bono bajo el escenario de estrés de tasas.
Este método captura la NO-LINEALIDAD (convexidad) que el enfoque de duración ignora.

P actual  = ${precio:,.4f}
n° bonos  = {n_bonds:.4f}
</div>
""", unsafe_allow_html=True)

    # Parametric full-reprice
    ytm_up_param = ytm + delta_y_param
    price_up_p   = price_bond_from_ytm(face_value, coupon_rate, freq,
                                        settle_date, maturity_date, ytm_up_param)
    var_reprice_param = abs(price_up_p - precio) * n_bonds

    # Historical full-reprice
    ytm_up_hist = ytm + delta_y_hist_up
    price_up_h  = price_bond_from_ytm(face_value, coupon_rate, freq,
                                       settle_date, maturity_date, ytm_up_hist)
    var_reprice_hist = abs(price_up_h - precio) * n_bonds

    # ES full-reprice (parametric ES Δy)
    delta_y_es_param = (norm.pdf(z_score) / (1 - confianza)) * sigma_dy
    ytm_es_param     = ytm + delta_y_es_param
    price_es_p       = price_bond_from_ytm(face_value, coupon_rate, freq,
                                            settle_date, maturity_date, ytm_es_param)
    es_reprice_param = abs(price_es_p - precio) * n_bonds

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(mbox(f"VaR Reprice Paramétrico ({conf_label})",
                          f"${var_reprice_param:,.0f}",
                          f"P estrés = ${price_up_p:,.4f}"),
                    unsafe_allow_html=True)
    with r2:
        st.markdown(mbox(f"VaR Reprice Histórico ({conf_label})",
                          f"${var_reprice_hist:,.0f}",
                          f"P estrés = ${price_up_h:,.4f}"),
                    unsafe_allow_html=True)
    with r3:
        st.markdown(mbox(f"ES Reprice Paramétrico ({conf_label})",
                          f"${es_reprice_param:,.0f}",
                          "Expected Shortfall"),
                    unsafe_allow_html=True)

    # ── Step 4: Comparison chart ──
    st.markdown('<div class="step-header">Paso 4 — Comparación de Métodos</div>',
                unsafe_allow_html=True)

    methods = ["VaR Dur.\nParamétrico", "VaR Reprice\nParamétrico",
               "VaR Dur.\nHistórico",   "VaR Reprice\nHistórico"]
    values  = [var_dur_param, var_reprice_param, var_dur_hist, var_reprice_hist]
    bar_colors = [COLOR_MAIN, COLOR_ACCENT, COLOR_WARN, COLOR_PURPLE]

    fig_cmp = go.Figure(go.Bar(
        x=methods, y=values,
        marker_color=bar_colors,
        text=[f"${v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        yaxis_title="VaR ($)",
        title=f"VaR Comparado — Confianza {conf_label} · Monto ${monto_inv:,.0f}",
        margin=dict(t=55, b=60, l=60, r=40),
        yaxis=dict(range=[0, max(values) * 1.3]),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Diff annotation
    diff_pct = (var_reprice_param - var_dur_param) / var_dur_param * 100 if var_dur_param > 0 else 0
    direction = "subestima" if diff_pct > 0 else "sobreestima"
    st.markdown(f"""
<div class="info-banner">
<b>Diferencia Duración vs Repricing (paramétrico):</b> {abs(diff_pct):.2f}%<br>
El modelo de duración <b>{direction}</b> el VaR respecto al repricing completo.
Esto se debe a que la duración ignora la convexidad del bono.
Cuanto mayor sea el movimiento de tasas, mayor será la diferencia.
</div>
""", unsafe_allow_html=True)

    # Store for backtesting
    st.session_state["var_computed"] = {
        "var_dur_param": var_dur_param,
        "var_reprice_param": var_reprice_param,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Backtesting
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Backtesting del VaR</div>',
                unsafe_allow_html=True)

    if not fred_ok or yield_changes.empty:
        st.error("No se pudieron cargar datos FRED para el backtesting. Verifica la conexión.")
        st.stop()

    N_TEST = 100
    if len(yield_changes) < N_TEST + 250:
        st.warning(f"Datos insuficientes para backtesting (se necesitan al menos {N_TEST+250} días).")
        st.stop()

    # ── Setup ──
    st.markdown('<div class="step-header">Configuración del Backtest</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="info-banner">
<b>Período de prueba:</b> últimos {N_TEST} días de yield changes
 · <b>Ventana de estimación:</b> 250 días anteriores al período de prueba<br>
<b>Método 1 — VaR por Duración:</b> rolling σ(Δy) × z × D_Mod × Monto<br>
<b>Método 2 — VaR por Repricing:</b> repricing completo con Δy_VaR rolling histórico
</div>
""", unsafe_allow_html=True)

    N_TRAIN   = len(yield_changes) - N_TEST
    test_chg  = yield_changes.iloc[N_TRAIN:]       # last N_TEST
    WINDOW    = 250

    # ── Rolling VaR calculation ──
    var_dur_roll     = []
    var_reprice_roll = []

    for i in range(N_TEST):
        start_i = N_TRAIN - WINDOW + i
        end_i   = N_TRAIN + i
        window_data = yield_changes.iloc[start_i:end_i]

        # Duration VaR
        sig_dy_i     = float(window_data.std())
        var_d_i      = mod_dur * monto_inv * z_score * sig_dy_i
        var_dur_roll.append(var_d_i)

        # Repricing VaR (historical quantile)
        q_dy_i       = float(window_data.quantile(confianza))
        ytm_stress_i = ytm + abs(q_dy_i)
        p_stress_i   = price_bond_from_ytm(face_value, coupon_rate, freq,
                                            settle_date, maturity_date, ytm_stress_i)
        var_r_i      = abs(p_stress_i - precio) * n_bonds
        var_reprice_roll.append(var_r_i)

    # ── Actual P&L ──
    # P&L(t) = -D_Mod × P × Δy(t) × n_bonds  (duration-based P&L, negative when yield rises)
    # For backtesting, a "loss" is when P&L < -VaR
    actual_pnl = pd.Series(
        [-mod_dur * precio * float(dy) * n_bonds for dy in test_chg.values],
        index=test_chg.index
    )

    var_dur_s     = pd.Series(var_dur_roll,     index=test_chg.index)
    var_reprice_s = pd.Series(var_reprice_roll, index=test_chg.index)

    exc_dur     = (actual_pnl < -var_dur_s).sum()
    exc_reprice = (actual_pnl < -var_reprice_s).sum()

    lr_dur,     p_dur     = kupiec_test(exc_dur,     N_TEST, confianza)
    lr_reprice, p_reprice = kupiec_test(exc_reprice, N_TEST, confianza)

    zone_dur,     col_dur     = basel_zone(exc_dur)
    zone_reprice, col_reprice = basel_zone(exc_reprice)

    # ── Results summary ──
    st.markdown('<div class="step-header">Resultados del Backtesting</div>',
                unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**VaR por Duración Modificada**")
        st.markdown(mbox("Excesos Observados", f"{exc_dur} / {N_TEST}",
                          f"Esperados: {(1-confianza)*N_TEST:.1f}"), unsafe_allow_html=True)
        st.markdown(mbox("Zona Basel", zone_dur, ""), unsafe_allow_html=True)
        st.markdown(mbox("Kupiec LR", f"{lr_dur:.3f}", f"p-valor: {p_dur:.4f}"),
                    unsafe_allow_html=True)

    with b2:
        st.markdown(f"**VaR por Repricing Completo**")
        st.markdown(mbox("Excesos Observados", f"{exc_reprice} / {N_TEST}",
                          f"Esperados: {(1-confianza)*N_TEST:.1f}"), unsafe_allow_html=True)
        st.markdown(mbox("Zona Basel", zone_reprice, ""), unsafe_allow_html=True)
        st.markdown(mbox("Kupiec LR", f"{lr_reprice:.3f}", f"p-valor: {p_reprice:.4f}"),
                    unsafe_allow_html=True)

    # ── Backtest chart ──
    st.markdown('<div class="step-header">Gráfica de Backtesting</div>',
                unsafe_allow_html=True)

    exc_dur_mask     = actual_pnl < -var_dur_s
    exc_reprice_mask = actual_pnl < -var_reprice_s

    fig_bt = go.Figure()

    # P&L bars
    fig_bt.add_trace(go.Bar(
        x=actual_pnl.index,
        y=actual_pnl.values,
        marker_color=np.where(actual_pnl >= 0, "rgba(166,227,161,0.5)", "rgba(243,139,168,0.5)"),
        name="P&L diario ($)",
    ))

    # VaR lines
    fig_bt.add_trace(go.Scatter(
        x=var_dur_s.index, y=-var_dur_s.values,
        line=dict(color=COLOR_MAIN, width=1.8, dash="solid"),
        name=f"−VaR Duración ({conf_label})",
    ))
    fig_bt.add_trace(go.Scatter(
        x=var_reprice_s.index, y=-var_reprice_s.values,
        line=dict(color=COLOR_WARN, width=1.8, dash="dash"),
        name=f"−VaR Repricing ({conf_label})",
    ))

    # Exceedance markers — Duration
    exc_dates_d  = actual_pnl[exc_dur_mask].index
    exc_vals_d   = actual_pnl[exc_dur_mask].values
    if len(exc_dates_d):
        fig_bt.add_trace(go.Scatter(
            x=exc_dates_d, y=exc_vals_d,
            mode="markers",
            marker=dict(color=COLOR_MAIN, size=9, symbol="x"),
            name=f"Exceso Duración ({exc_dur})",
        ))

    # Exceedance markers — Repricing
    exc_dates_r = actual_pnl[exc_reprice_mask].index
    exc_vals_r  = actual_pnl[exc_reprice_mask].values
    if len(exc_dates_r):
        fig_bt.add_trace(go.Scatter(
            x=exc_dates_r, y=exc_vals_r,
            mode="markers",
            marker=dict(color=COLOR_WARN, size=9, symbol="circle-open"),
            name=f"Exceso Repricing ({exc_reprice})",
        ))

    fig_bt.update_layout(
        template=PLOTLY_TEMPLATE,
        height=430,
        xaxis_title="Fecha",
        yaxis_title="P&L / VaR ($)",
        title=f"Backtesting — {N_TEST} días · {tenor_label} · Confianza {conf_label}",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=55, b=100, l=60, r=40),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Kupiec explanation ──
    st.markdown('<div class="step-header">Interpretación — Prueba de Kupiec (POF)</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="formula-box">
H₀: la frecuencia real de excesos = {(1-confianza)*100:.0f}% (modelo es correcto)

LR = −2 × [ N₁·ln(α/p̂) + N₀·ln((1−α)/(1−p̂)) ]   ~  χ²(1)

donde N₁ = excesos, N₀ = no-excesos, α = {alpha_level:.2f}, p̂ = N₁/T

Rechazamos H₀ si p-valor < 0.05 → el modelo tiene mal calibrado el VaR.
</div>
""", unsafe_allow_html=True)

    # ── Basel zones explanation ──
    st.markdown('<div class="step-header">Zonas de Semáforo de Basilea (250 días)</div>',
                unsafe_allow_html=True)
    basel_data = {
        "Zona": ["🟢 Verde", "🟡 Amarilla", "🔴 Roja"],
        "Excesos": ["0 – 4", "5 – 9", "≥ 10"],
        "Interpretación": [
            "Modelo adecuado — sin penalización",
            "Zona de advertencia — revisar modelo",
            "Modelo deficiente — factor de penalización Basilea (+0.40 a +1.00)",
        ],
    }
    st.dataframe(pd.DataFrame(basel_data), use_container_width=True, hide_index=True)

    # ── Conclusion ──
    st.markdown('<div class="step-header">Conclusión Automática</div>',
                unsafe_allow_html=True)

    conclusions = []
    for method, exc, zone, col, lr, pv in [
        ("Duración",   exc_dur,     zone_dur,     col_dur,     lr_dur,     p_dur),
        ("Repricing",  exc_reprice, zone_reprice, col_reprice, lr_reprice, p_reprice),
    ]:
        expected = (1 - confianza) * N_TEST
        if pv > 0.05:
            kupiec_msg = f"no se rechaza H₀ (p={pv:.4f} > 0.05) — el modelo está bien calibrado."
        else:
            kupiec_msg = f"se rechaza H₀ (p={pv:.4f} < 0.05) — el VaR está <b>mal calibrado</b>."
        conclusions.append(f"""
<div class="info-banner" style="border-left-color:{col};">
<b>Método {method}:</b> {exc} excesos en {N_TEST} días (esperados: {expected:.1f}) · Zona {zone}<br>
Kupiec: {kupiec_msg}
</div>""")

    for c in conclusions:
        st.markdown(c, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Herramienta educativa — Renta Fija · EAFIT · Datos de mercado: FRED (Federal Reserve Bank of St. Louis)")
