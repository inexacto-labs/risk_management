import streamlit as st
import pandas as pd

from src.charts import make_var_distribution_chart, make_var_tail_chart
from src.data_loader import get_series_names, load_papa_data
from src.transforms import (
    compute_expected_shortfall,
    compute_historical_var,
    compute_inventory_var_amount,
    compute_daily_returns,
    describe_var_behavior,
)
from src.ui import apply_app_style, render_chart_info, render_page_header, render_stat_card


st.set_page_config(
    page_title="VaR como intuición",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_style()

data = load_papa_data()
series_names = get_series_names(data)
returns = compute_daily_returns(data, periods=1)

with st.sidebar:
    st.markdown("## Configura el caso")
    selected_series = st.selectbox("Tipo de papa", options=series_names, index=0)
    inventory_value = st.number_input(
        "Valor del inventario (COP)",
        min_value=100000,
        max_value=50000000,
        value=5000000,
        step=100000,
    )
    confidence_pct = st.select_slider("Nivel de confianza", options=[90, 95, 99], value=95)
    horizon = st.slider("Horizonte (días)", min_value=1, max_value=10, value=1, step=1)

confidence = confidence_pct / 100
var_result = compute_historical_var(returns, selected_series, confidence, horizon=horizon)
var_amount = compute_inventory_var_amount(inventory_value, var_result["var_return"])
expected_shortfall = compute_expected_shortfall(var_result["scaled_returns"], confidence)
expected_shortfall_amount = compute_inventory_var_amount(inventory_value, expected_shortfall)
insight = describe_var_behavior(
    selected_series,
    inventory_value,
    confidence,
    horizon,
    var_amount,
    var_result["var_return_pct"],
)

scenario_series = var_result["scaled_returns"].dropna().sort_values()
threshold = scenario_series.quantile(1 - confidence) if not scenario_series.empty else 0.0
scenario_table = pd.DataFrame(
    {
        "Fecha de variación": scenario_series.index,
        "Retorno": scenario_series.values,
    }
)
scenario_table["Fecha de variación"] = pd.to_datetime(scenario_table["Fecha de variación"]).dt.strftime("%Y-%m-%d")
scenario_table.insert(0, "Escenario", range(1, len(scenario_table) + 1))
scenario_table["Retorno %"] = scenario_table["Retorno"].map(lambda value: f"{value * 100:.2f}%")
scenario_table["Resultado estimado (COP)"] = scenario_table["Retorno"].map(
    lambda value: f"${inventory_value * value:,.0f}".replace(",", ".")
)
scenario_table["Zona VaR"] = scenario_table["Retorno"].map(
    lambda value: "Escenario adverso" if value <= threshold else "Fuera de cola VaR"
)
scenario_table_display = scenario_table[
    ["Escenario", "Fecha de variación", "Retorno %", "Resultado estimado (COP)", "Zona VaR"]
]
tail_table_display = scenario_table_display[scenario_table_display["Zona VaR"] == "Escenario adverso"]

render_page_header(
    eyebrow="Pantalla 4",
    title="Valor en Riesgo: traducir variación a pérdida potencial",
    subtitle=(
        "Hasta ahora hemos visto que los precios cambian y que esa variación puede ser más o menos intensa. "
        "El VaR aparece cuando queremos responder una pregunta práctica: cuánto podría perder una posición "
        "bajo un nivel de confianza y en un horizonte dado."
    ),
)

top1, top2, top3, top4 = st.columns(4)
with top1:
    render_stat_card("VaR estimado", var_amount)
with top2:
    render_stat_card("VaR como retorno", var_result["var_return_pct"])
with top3:
    render_stat_card("Peor escenario observado", var_result["worst_return_pct"])
with top4:
    render_stat_card("Escenarios utilizados", f"{var_result['scenario_count']}")

st.markdown(
    f"""
    <div class="callout">
      <strong>Lectura del resultado</strong><br/>
      {insight}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Caso de referencia")
st.markdown(
    """
    Pensemos en un comerciante que compra inventario de papa y lo mantiene por algunos días antes de venderlo.
    La pregunta ya no es solamente si el precio puede bajar, sino cuánto podría afectar esa caída al valor del inventario.
    """
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### 1. Ubicar el percentil adverso")
    render_chart_info(
        title="Distribución de retornos y percentil adverso",
        steps=[
            "Se calculan los retornos diarios de la serie seleccionada.",
            "Si el horizonte es mayor a un día, se construyen retornos acumulados de ese número de días.",
            "Luego se ubica el percentil correspondiente a la cola izquierda de la distribución.",
        ],
        formula="VaR histórico = |percentil_(1-c)(retornos del horizonte)|",
        note="Si el nivel de confianza es 95%, se observa el 5% más adverso de la distribución histórica. Para horizontes mayores a un día, la app usa variaciones acumuladas del horizonte elegido.",
    )
    st.caption(
        "La línea punteada marca el punto a partir del cual empiezan los escenarios más adversos según el nivel de confianza elegido."
    )
    st.plotly_chart(
        make_var_distribution_chart(var_result["scaled_returns"], confidence, selected_series),
        use_container_width=True,
    )

with col2:
    st.markdown("### 2. Observar la cola de pérdidas")
    render_chart_info(
        title="Escenarios ordenados de peor a mejor",
        steps=[
            "Se ordenan los retornos históricos de menor a mayor.",
            "Se identifican los retornos que quedan por debajo del umbral del VaR.",
            "Esos escenarios representan la zona de pérdidas más extremas dentro de la muestra.",
        ],
        formula="Cola izquierda = {retornos : retorno <= percentil_(1-c)}",
        note="Esta vista ayuda a mostrar que el VaR resume un umbral, pero no agota toda la información sobre escenarios extremos.",
    )
    st.caption(
        "Las barras resaltadas muestran los escenarios que caen dentro de la zona más adversa de la distribución."
    )
    st.plotly_chart(
        make_var_tail_chart(var_result["scaled_returns"], confidence, selected_series),
        use_container_width=True,
    )

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### Cómo leer esta pantalla")
    st.markdown(
        f"""
        - El VaR no dice cuál será la pérdida exacta, sino un umbral razonable bajo un criterio estadístico.
        - Con confianza del {confidence_pct}%, la pérdida no superaría el monto estimado en la mayoría de los casos.
        - El resultado depende de tres decisiones: la serie elegida, el horizonte y el nivel de confianza.
        """
    )

with right:
    st.markdown("### Más allá del VaR")
    st.markdown(
        f"""
        - El VaR estimado para este caso es **{var_amount}**.
        - Si miramos solo los escenarios peores que el VaR, la pérdida promedio sería cercana a **{expected_shortfall_amount}**.
        - Esa idea prepara el camino para discutir después medidas como el Expected Shortfall.
        """
    )

st.markdown("### Tabla de escenarios")
render_chart_info(
    title="Tabla de retornos e impacto sobre el inventario",
        steps=[
            "Se toman los retornos históricos ya ajustados al horizonte seleccionado.",
            "Cada retorno se convierte en una pérdida o ganancia potencial sobre el valor del inventario.",
            "Luego se marca si ese escenario cae o no dentro de la zona adversa definida por el VaR.",
        ],
        formula="Resultado_t = Inventario x Retorno_t",
        note="La tabla permite pasar del gráfico a casos concretos: cada fila representa un escenario histórico posible.",
    )
st.caption(
    "Esta tabla permite ver, escenario por escenario, cómo se traducen los retornos en ganancias o pérdidas monetarias sobre el inventario."
)

tab1, tab2 = st.tabs(["Todos los escenarios", "Solo zona VaR"])
with tab1:
    st.dataframe(scenario_table_display, use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(tail_table_display, use_container_width=True, hide_index=True)

nav1, nav2 = st.columns([1, 1])
with nav1:
    st.page_link("pages/03_Volatilidad.py", label="Volver a Pantalla 3", icon="⬅️")
with nav2:
    st.page_link("pages/05_Acciones_Y_Sectores.py", label="Ir a Pantalla 5: acciones", icon="➡️")
