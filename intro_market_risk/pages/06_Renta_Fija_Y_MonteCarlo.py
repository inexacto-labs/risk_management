import numpy as np
import streamlit as st

from src.charts import (
    make_bond_cashflow_chart,
    make_bond_price_curve_chart,
    make_duration_approx_chart,
    make_monte_carlo_loss_chart,
    make_rate_path_chart,
    make_rate_scenarios_chart,
)
from src.fixed_income import (
    bond_risk_measures,
    estimate_price_change,
    price_curve,
    price_fixed_rate_bond,
    run_bond_var_monte_carlo,
)
from src.ui import apply_app_style, render_chart_info, render_page_header, render_stat_card


st.set_page_config(
    page_title="Renta fija y Monte Carlo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_style()

with st.sidebar:
    st.markdown("## Configura el bono")
    face_value = st.number_input("Valor nominal", min_value=1000, max_value=5000000, value=100000, step=1000)
    coupon_rate_pct = st.slider("Cupón anual (%)", min_value=0.0, max_value=20.0, value=8.0, step=0.25)
    ytm_pct = st.slider("Tasa de mercado / YTM (%)", min_value=1.0, max_value=20.0, value=9.0, step=0.25)
    maturity_years = st.slider("Plazo (años)", min_value=1, max_value=15, value=5, step=1)
    payments_per_year = st.selectbox("Pagos por año", options=[1, 2, 4], index=1)
    shock_bps = st.slider("Shock para sensibilidad (pbs)", min_value=-300, max_value=300, value=100, step=25)
    rate_vol_pct = st.slider("Volatilidad anual de la tasa (%)", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
    horizon_days = st.slider("Horizonte VaR (días)", min_value=1, max_value=20, value=10, step=1)
    confidence_pct = st.select_slider("Confianza", options=[90, 95, 99], value=95)
    simulations = st.select_slider("Simulaciones Monte Carlo", options=[1000, 5000, 10000, 20000], value=10000)

coupon_rate = coupon_rate_pct / 100
ytm = ytm_pct / 100
rate_vol_annual = rate_vol_pct / 100
confidence = confidence_pct / 100
delta_rate = shock_bps / 10000

measures = bond_risk_measures(face_value, coupon_rate, ytm, maturity_years, payments_per_year)
curve_df = price_curve(face_value, coupon_rate, maturity_years, payments_per_year, ytm)
current_price = measures["price"]
shocked_price = price_fixed_rate_bond(
    face_value,
    coupon_rate,
    max(ytm + delta_rate, 0.0001),
    maturity_years,
    payments_per_year,
)
actual_change_pct = (shocked_price / current_price) - 1
duration_change_pct = estimate_price_change(measures["modified_duration"], measures["convexity"], delta_rate, False)
duration_conv_change_pct = estimate_price_change(measures["modified_duration"], measures["convexity"], delta_rate, True)

comparison_shocks = [-200, -100, -50, 50, 100, 200]
actual_curve = []
duration_curve = []
duration_conv_curve = []
for bps in comparison_shocks:
    dr = bps / 10000
    scenario_price = price_fixed_rate_bond(
        face_value,
        coupon_rate,
        max(ytm + dr, 0.0001),
        maturity_years,
        payments_per_year,
    )
    actual_curve.append(((scenario_price / current_price) - 1) * 100)
    duration_curve.append(estimate_price_change(measures["modified_duration"], measures["convexity"], dr, False) * 100)
    duration_conv_curve.append(estimate_price_change(measures["modified_duration"], measures["convexity"], dr, True) * 100)

mc = run_bond_var_monte_carlo(
    face_value=face_value,
    coupon_rate=coupon_rate,
    ytm=ytm,
    maturity_years=maturity_years,
    payments_per_year=payments_per_year,
    rate_vol_annual=rate_vol_annual,
    horizon_days=horizon_days,
    confidence=confidence,
    simulations=simulations,
)

render_page_header(
    eyebrow="Pantalla 6",
    title="Renta fija: del crédito al bono, y del bono al VaR con Monte Carlo",
    subtitle=(
        "El paso de renta variable a renta fija exige cambiar el objeto de análisis. "
        "Aquí el riesgo no surge principalmente de una acción que cambia de precio por sí sola, "
        "sino de cómo cambian las tasas de interés y cómo esas tasas revaloran un flujo contractual."
    ),
)

st.markdown("### Una analogía para entrar al tema")
st.markdown(
    """
    Piensa en un crédito: alguien presta dinero hoy y recibe pagos en el futuro. Un bono puede leerse de manera parecida,
    solo que desde la perspectiva del inversionista que compra esos flujos. Por eso, en renta fija no basta con mirar
    el precio actual. Lo esencial es entender cómo cambian los flujos descontados cuando se mueve la tasa de interés.
    """
)
st.markdown(
    """
    Dicho de otra forma: una acción suele analizarse como un precio que fluctúa en el mercado; un bono, en cambio,
    se entiende mejor como una secuencia de pagos futuros cuyo valor hoy depende de la tasa con la que se descuentan.
    """
)

top1, top2, top3, top4 = st.columns(4)
with top1:
    render_stat_card("Precio del bono", f"${current_price:,.0f}".replace(",", "."))
with top2:
    render_stat_card("Duración modificada", f"{measures['modified_duration']:.2f}")
with top3:
    render_stat_card("Convexidad", f"{measures['convexity']:.2f}")
with top4:
    render_stat_card("VaR Monte Carlo", f"${mc['var_amount']:,.0f}".replace(",", "."))

st.markdown(
    f"""
    <div class="callout">
      <strong>Lectura inicial</strong><br/>
      Con una tasa de mercado de <strong>{ytm_pct:.2f}%</strong>, el bono vale aproximadamente
      <strong>${current_price:,.0f}</strong>. Si las tasas suben, ese precio cae; si bajan, el precio sube.
      Esa sensibilidad se resume primero con duración y convexidad, y luego se lleva a una distribución de pérdidas
      cuando simulamos posibles movimientos de tasa con Monte Carlo.
    </div>
    """.replace(",", "."),
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### 1. El bono como paquete de flujos")
    render_chart_info(
        title="Flujos del bono",
        steps=[
            "Se define un valor nominal, una tasa cupón, un plazo y una frecuencia de pago.",
            "Cada periodo genera un cupón y el último flujo devuelve además el principal.",
            "El bono se puede entender como la suma de esos flujos futuros.",
        ],
        formula="Flujo final = cupón + principal; flujos intermedios = solo cupón",
        note="Esta es la parte pedagógica clave: primero entendemos el bono como contrato de flujos antes de medir su riesgo.",
    )
    st.plotly_chart(make_bond_cashflow_chart(measures["cashflows"]), use_container_width=True)

with col2:
    st.markdown("### 2. Cómo se calcula el precio del bono")
    render_chart_info(
        title="Valoración del bono por descuento de flujos",
        steps=[
            "Se toma cada flujo futuro del bono: cupones y devolución del principal.",
            "Cada flujo se divide por un factor de descuento que depende de la tasa de mercado y del tiempo.",
            "El precio del bono es la suma de todos esos valores presentes.",
        ],
        formula="Precio = Σ Flujo_t / (1 + tasa/m)^periodo_t",
        note="Esta es la base de todo lo que sigue: si cambia la tasa de descuento, cambia el valor presente de cada flujo y, por tanto, el precio del bono.",
    )
    st.dataframe(
        measures["cashflows"][
            ["Periodo", "Tiempo (años)", "Flujo", "Factor de descuento", "Valor presente"]
        ].style.format(
            {
                "Tiempo (años)": "{:.2f}",
                "Flujo": "{:,.0f}",
                "Factor de descuento": "{:.4f}",
                "Valor presente": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### 3. Precio y tasa: la relación central")
st.markdown(
    f"""
    Con el bono configurado arriba, el precio actual es **${current_price:,.0f}**. Si la tasa de mercado sube por encima del
    cupón, los flujos futuros se descuentan más fuerte y el precio cae. Si la tasa baja, esos mismos flujos valen más hoy
    y el precio sube.
    """.replace(",", ".")
)
render_chart_info(
    title="Relación inversa entre tasa y precio",
    steps=[
        "Se conservan los mismos flujos del bono.",
        "Se recalcula el precio para distintos niveles de tasa.",
        "Al subir la tasa, aumenta el descuento aplicado a cada flujo y el precio baja.",
    ],
    formula="Si tasa ↑ entonces precio ↓; si tasa ↓ entonces precio ↑",
    note="Esta relación es el corazón del riesgo de mercado en renta fija.",
)
st.plotly_chart(make_bond_price_curve_chart(curve_df, ytm), use_container_width=True)

st.markdown("### 4. De la intuición a la sensibilidad")
st.markdown(
    f"""
    Si la tasa cambia en <strong>{shock_bps} pbs</strong>, el cambio real de precio sería aproximadamente
    <strong>{actual_change_pct * 100:.2f}%</strong>. Con solo duración, estimaríamos
    <strong>{duration_change_pct * 100:.2f}%</strong>; al incluir convexidad, la aproximación pasa a
    <strong>{duration_conv_change_pct * 100:.2f}%</strong>.
    """,
    unsafe_allow_html=True,
)
render_chart_info(
    title="Duración y convexidad como aproximaciones",
    steps=[
        "La duración modificada aproxima el cambio porcentual del precio ante un pequeño cambio en la tasa.",
        "La convexidad corrige el error cuando el movimiento de tasas deja de ser pequeño.",
        "La comparación con el cambio real muestra por qué ambas métricas importan.",
    ],
    formula="ΔP/P ≈ -D_mod·Δr + 0.5·Convexidad·(Δr)^2",
    note="Este paso es útil porque conecta una intuición geométrica con una medición más exacta del riesgo.",
)
st.plotly_chart(
    make_duration_approx_chart(comparison_shocks, actual_curve, duration_curve, duration_conv_curve),
    use_container_width=True,
)

st.markdown("### 5. VaR con Monte Carlo sobre la tasa de interés")
st.markdown(
    """
    Aquí cambiamos de metodología de forma deliberada. En vez de tomar retornos históricos de un activo, ahora
    simulamos posibles variaciones de la tasa de interés. Cada tasa simulada produce un precio distinto del bono;
    esa revaloración genera una distribución de ganancias y pérdidas, y sobre esa distribución calculamos el VaR.
    """
)
st.markdown(
    """
    Para hacerlo más claro, la simulación no se presenta solo como un resultado final. También mostramos trayectorias
    posibles de la tasa a lo largo del horizonte. Cada trayectoria representa desplazamientos paralelos del nivel de tasa
    en el tiempo; al final de cada una, revaloramos el bono y medimos la pérdida o ganancia resultante.
    """
)
st.info(
    "En esta versión pedagógica usamos una única tasa de referencia para descontar todos los flujos del bono. No estamos modelando una curva completa de tasas, sino un movimiento paralelo del nivel de tasa."
)

var1, var2, var3, var4 = st.columns(4)
with var1:
    render_stat_card("VaR Monte Carlo", f"${mc['var_amount']:,.0f}".replace(",", "."))
with var2:
    render_stat_card("VaR como % del precio", f"{mc['var_pct'] * 100:.2f}%")
with var3:
    render_stat_card("Expected Shortfall", f"${mc['expected_shortfall']:,.0f}".replace(",", "."))
with var4:
    render_stat_card("Simulaciones", f"{simulations:,}".replace(",", "."))

mc_col1, mc_col2 = st.columns(2, gap="large")
with mc_col1:
    st.markdown("### 6. Trayectorias simuladas de tasa")
    render_chart_info(
        title="Trayectorias Monte Carlo de la tasa",
        steps=[
            "Se simulan incrementos diarios de tasa con una distribución normal centrada en cero.",
            "Los incrementos se acumulan a lo largo del horizonte para construir trayectorias posibles de la tasa.",
            "Cada trayectoria representa una secuencia de movimientos paralelos del nivel de tasa.",
        ],
        formula="r_t = r_(t-1) + ε_t, con ε_t ~ Normal(0, σ_tasa / √252)",
        note="La línea oscura muestra la trayectoria promedio; las líneas suaves muestran escenarios posibles alrededor de ella.",
    )
    st.plotly_chart(make_rate_path_chart(mc["rate_paths"]), use_container_width=True)

with mc_col2:
    st.markdown("### 7. Distribución de pérdidas del bono")
    render_chart_info(
        title="VaR Monte Carlo del bono",
        steps=[
            "De cada trayectoria se toma la tasa terminal del horizonte.",
            "Cada tasa terminal produce un precio del bono.",
            "La diferencia entre el precio simulado y el precio actual define la ganancia o pérdida.",
            "El VaR se obtiene como el percentil de pérdidas asociado al nivel de confianza elegido.",
        ],
        formula="Pérdida = Precio actual - Precio simulado; VaR = percentil_c(pérdidas)",
        note="Este cambio de metodología es pedagógicamente potente: muestra que el riesgo puede medirse con herramientas distintas según el activo.",
    )
    st.plotly_chart(make_monte_carlo_loss_chart(mc["losses"], confidence), use_container_width=True)

st.markdown("### 8. Distribución de choques terminales de tasa")
render_chart_info(
    title="Choques terminales de tasa",
    steps=[
        "Se observa el cambio acumulado de la tasa entre el inicio y el final del horizonte.",
        "Ese cambio resume el efecto total de la trayectoria simulada sobre la tasa terminal.",
        "La dispersión de estos choques ayuda a entender por qué el precio del bono termina en valores distintos.",
    ],
    formula="Δr_h = r_h - r_0",
    note="Esta vista complementa las trayectorias y ayuda a conectar simulación de tasas con revaloración del bono.",
)
st.plotly_chart(make_rate_scenarios_chart(mc["terminal_rates"] - ytm), use_container_width=True)

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### Cómo leer esta pantalla")
    st.markdown(
        """
        - En renta fija, el riesgo de mercado entra principalmente por la tasa de interés.
        - El bono es un flujo contractual revaluado con una tasa de descuento.
        - La duración resume sensibilidad lineal; la convexidad mejora la aproximación.
        - Monte Carlo permite pasar de sensibilidad puntual a distribución de pérdidas.
        """
    )

with right:
    st.markdown("### Cierre conceptual")
    st.markdown(
        """
        La intuición no cambia: seguimos hablando de exposición bajo incertidumbre. Lo que cambia es la forma de modelar
        el factor de riesgo. En acciones trabajábamos con retornos del precio; en bonos, con movimientos de la tasa que
        revaloran flujos futuros. Esa diferencia justifica el cambio metodológico hacia Monte Carlo.
        """
    )

nav1, nav2 = st.columns([1, 1])
with nav1:
    st.page_link("pages/05_Acciones_Y_Sectores.py", label="Volver a Pantalla 5", icon="⬅️")
with nav2:
    st.page_link("pages/06_Renta_Fija_Y_MonteCarlo.py", label="Seguir explorando esta pantalla", icon="🔎")
