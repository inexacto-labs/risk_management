import streamlit as st

from src.charts import (
    make_histogram_chart,
    make_return_chart,
    make_rolling_vol_chart,
    make_volatility_bar_chart,
)
from src.data_loader import get_series_names, load_papa_data
from src.transforms import (
    compute_daily_returns,
    compute_distribution_summary,
    compute_rolling_volatility,
    compute_volatility_summary,
    describe_volatility_behavior,
)
from src.ui import apply_app_style, render_page_header, render_stat_card
from src.ui import render_chart_info


st.set_page_config(
    page_title="Volatilidad",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_style()

data = load_papa_data()
series_names = get_series_names(data)

with st.sidebar:
    st.markdown("## Configura el análisis")
    selected_series = st.selectbox("Serie principal", options=series_names, index=0)
    compare_series = st.multiselect(
        "Series para comparar",
        options=series_names,
        default=series_names,
    )
    rolling_window = st.slider("Ventana rolling (días)", min_value=10, max_value=90, value=30, step=5)

returns = compute_daily_returns(data, periods=1)
rolling_vol = compute_rolling_volatility(returns, window=rolling_window)
vol_summary = compute_volatility_summary(returns)
distribution_summary = compute_distribution_summary(returns, selected_series)

ranking_row = vol_summary.index[vol_summary["Serie"] == selected_series]
ranking_position = int(ranking_row[0]) + 1 if len(ranking_row) else 0
insight = describe_volatility_behavior(
    selected_series,
    distribution_summary,
    ranking_position,
    len(vol_summary),
)

render_page_header(
    eyebrow="Pantalla 3",
    title="Volatilidad: cuando la variación empieza a volverse medible",
    subtitle=(
        "Después de ver que los retornos cuentan una historia más útil que los niveles, "
        "el siguiente paso es medir la dispersión de esos cambios. A esa dispersión la "
        "llamamos volatilidad."
    ),
)

top1, top2, top3, top4 = st.columns(4)
with top1:
    render_stat_card("Volatilidad", distribution_summary["volatility"])
with top2:
    render_stat_card("Cambio absoluto promedio", distribution_summary["avg_abs_return"])
with top3:
    render_stat_card("Peor caída observada", distribution_summary["worst_day"])
with top4:
    render_stat_card("Mejor alza observada", distribution_summary["best_day"])

st.markdown(
    f"""
    <div class="callout">
      <strong>Idea clave</strong><br/>
      {insight}
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### 1. Ver la dispersión de los retornos")
    render_chart_info(
        title="Histograma de retornos",
        steps=[
            "Se toman todos los retornos diarios disponibles de la serie seleccionada.",
            "Esos retornos se agrupan en intervalos o bins.",
            "La altura de cada barra muestra cuántas veces apareció un retorno dentro de ese rango.",
        ],
        formula="Frecuencia(rango i) = número de retornos ubicados dentro del intervalo i",
        note="Si los retornos se concentran cerca de cero, la serie es más estable. Si se dispersan, la volatilidad aumenta.",
    )
    st.caption(
        "Cuando los cambios se concentran cerca de cero, la serie es más estable. Cuando se dispersan más, la incertidumbre aumenta."
    )
    st.plotly_chart(
        make_histogram_chart(returns, selected_series),
        use_container_width=True,
    )

with col2:
    st.markdown("### 2. Ver cómo cambia en el tiempo")
    render_chart_info(
        title="Volatilidad rolling",
        steps=[
            "Se calcula primero la serie de retornos diarios.",
            "Luego, para cada fecha, se toma una ventana móvil de los últimos n días.",
            "Dentro de esa ventana se calcula la desviación estándar de los retornos.",
        ],
        formula="Volatilidad_t = std(Retorno_(t-n+1), ..., Retorno_t)",
        note="En la app, el usuario puede cambiar n para observar cómo cambia la suavidad o sensibilidad de la medida.",
    )
    st.caption(
        "La volatilidad no tiene por qué ser constante. Esta vista ayuda a identificar periodos de mayor o menor tensión."
    )
    st.plotly_chart(
        make_rolling_vol_chart(rolling_vol, compare_series, highlight=selected_series),
        use_container_width=True,
    )

st.markdown("### Comparación entre variedades")
render_chart_info(
    title="Comparación de volatilidad entre series",
    steps=[
        "Para cada variedad se toma la serie completa de retornos diarios disponibles.",
        "Se calcula la desviación estándar de esos retornos.",
        "Las barras ordenan y comparan ese resultado entre variedades.",
    ],
    formula="Volatilidad_serie = std(Retornos de la serie)",
    note="Una desviación estándar más alta indica mayor dispersión histórica de los cambios diarios.",
)
st.caption(
    "Esta barra resume qué series han mostrado mayor dispersión en sus retornos diarios dentro de la muestra observada."
)
st.plotly_chart(
    make_volatility_bar_chart(vol_summary, highlight=selected_series),
    use_container_width=True,
)
st.info("Más volatilidad no significa necesariamente un activo peor. Significa, en esta etapa del curso, mayor incertidumbre de corto plazo sobre sus variaciones.")

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### Cómo leer esta pantalla")
    st.markdown(
        """
        - La volatilidad no mide si el precio sube o baja, sino qué tan intensos son sus cambios.
        - Una serie puede tener una tendencia clara y aun así ser muy volátil.
        - Dos series con precios parecidos pueden implicar niveles de riesgo muy distintos.
        """
    )

with right:
    st.markdown("### Traducción a negocio")
    st.markdown(
        """
        Si una variedad muestra mayor volatilidad, eso implica un entorno más incierto para quien:

        - compra inventario,
        - fija precios de venta,
        - proyecta márgenes,
        - o necesita anticipar pérdidas potenciales.
        """
    )

st.markdown("### Retornos de referencia")
render_chart_info(
    title="Serie de retornos diarios",
    steps=[
        "Se transforma cada precio observado en un cambio porcentual frente al día anterior.",
        "Luego se grafica esa secuencia de variaciones a lo largo del tiempo.",
        "La línea cero actúa como referencia entre alzas y caídas.",
    ],
    formula="Retorno_t = (Precio_t / Precio_(t-1)) - 1",
    note="Este gráfico funciona como puente entre la pantalla 2 y las medidas de volatilidad mostradas arriba.",
)
st.caption(
    "Si quieres volver a mirar la serie de cambios diarios antes de pasar al siguiente concepto, aquí tienes la comparación directa."
)
st.plotly_chart(
    make_return_chart(returns, compare_series, highlight=selected_series, periods=1),
    use_container_width=True,
)

nav1, nav2 = st.columns([1, 1])
with nav1:
    st.page_link("pages/02_Precios_y_Retornos.py", label="Volver a Pantalla 2", icon="⬅️")
with nav2:
    st.page_link("pages/04_VaR_Como_Intuicion.py", label="Ir a Pantalla 4: VaR", icon="➡️")
