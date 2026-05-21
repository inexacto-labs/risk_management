import streamlit as st

from src.charts import (
    make_base100_chart,
    make_price_chart,
    make_return_chart,
)
from src.data_loader import get_series_names, load_papa_data
from src.transforms import (
    compute_base100,
    compute_daily_returns,
    compute_series_snapshot,
    describe_return_behavior,
)
from src.ui import apply_app_style, render_chart_info, render_page_header, render_stat_card


st.set_page_config(
    page_title="Precios y retornos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_style()

data = load_papa_data()
series_names = get_series_names(data)

with st.sidebar:
    st.markdown("## Explora la serie")
    selected_series = st.selectbox("Tipo de papa", options=series_names, index=0)
    compare_series = st.multiselect(
        "Series para comparar",
        options=series_names,
        default=series_names,
    )
    return_window_label = st.selectbox(
        "Escala del retorno",
        options=["Diario", "Semanal aproximado (5 días)"],
        index=0,
    )

render_page_header(
    eyebrow="Pantalla 2",
    title="Mismo mercado, dos historias: niveles y retornos",
    subtitle=(
        "El precio nos dice en qué nivel está una serie. El retorno nos dice "
        "qué tanto cambia. Para hablar de riesgo, casi siempre la segunda historia "
        "es la que importa más."
    ),
)

window = 1 if return_window_label == "Diario" else 5
returns = compute_daily_returns(data, periods=window)
base100 = compute_base100(data)
snapshot = compute_series_snapshot(data, returns, selected_series, periods=window)
insight = describe_return_behavior(snapshot, selected_series, return_window_label)

top1, top2, top3, top4 = st.columns(4)
with top1:
    render_stat_card("Último precio", snapshot["last_price"])
with top2:
    render_stat_card("Cambio reciente", snapshot["last_return"])
with top3:
    render_stat_card("Promedio móvil del cambio", snapshot["avg_abs_return"])
with top4:
    render_stat_card("Rango histórico", snapshot["price_range"])

st.markdown(
    f"""
    <div class="callout">
      <strong>Idea clave</strong><br/>
      {insight} Dicho de forma más directa: el precio responde cuánto vale; el retorno responde qué tanto cambia.
    </div>
    """,
    unsafe_allow_html=True,
)

price_col, return_col = st.columns(2, gap="large")
with price_col:
    st.markdown("### 1. Ver los niveles")
    render_chart_info(
        title="Serie de precios en niveles",
        steps=[
            "Se toma la columna de precios observados para cada variedad.",
            "Cada punto del gráfico corresponde al precio reportado en una fecha.",
            "No se transforma la serie: aquí se muestran los niveles originales.",
        ],
        formula="Precio_t = valor observado en la fecha t",
        note="Este gráfico es descriptivo. Sirve para ubicar tendencias, máximos, mínimos y cambios de régimen.",
    )
    st.caption(
        "Aquí vemos el precio como tal. Esta vista ayuda a ubicar máximos, mínimos y tendencias generales."
    )
    st.plotly_chart(
        make_price_chart(data, compare_series, highlight=selected_series),
        use_container_width=True,
    )

with return_col:
    st.markdown("### 2. Ver los cambios")
    render_chart_info(
        title="Retornos porcentuales",
        steps=[
            "Se compara el precio actual con el precio de uno o varios periodos atrás.",
            "La diferencia se expresa como proporción del precio anterior.",
            "En esta pantalla, la escala puede ser diaria o semanal aproximada de 5 días.",
        ],
        formula="Retorno_t = (Precio_t / Precio_(t-k)) - 1",
        note="Cuando k = 1, el retorno es diario. Cuando k = 5, aproximamos un cambio semanal.",
    )
    st.caption(
        "Aquí ya no importa tanto el nivel, sino cuánto cambia la serie entre un momento y otro."
    )
    st.plotly_chart(
        make_return_chart(returns, compare_series, highlight=selected_series, periods=window),
        use_container_width=True,
    )

st.markdown("### Si todas empezaran en el mismo punto")
render_chart_info(
    title="Serie indexada base 100",
    steps=[
        "Se identifica el primer valor disponible de cada serie.",
        "Cada observación posterior se divide por ese primer valor.",
        "El resultado se multiplica por 100 para que todas las series arranquen en el mismo punto.",
    ],
    formula="Índice_t = (Precio_t / Precio_inicial) * 100",
    note="Esta transformación facilita comparar trayectorias relativas entre series con niveles iniciales distintos.",
)
st.caption(
    "Indexar en base 100 ayuda a comparar trayectorias sin que nos distraigan los distintos niveles iniciales."
)
st.plotly_chart(
    make_base100_chart(base100, compare_series, highlight=selected_series),
    use_container_width=True,
)

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### Cómo leer esta pantalla")
    st.markdown(
        """
        - En precios, una serie puede parecer "más cara" solo porque su nivel es mayor.
        - En retornos, lo que miramos es qué tan bruscos son los cambios.
        - Para medir riesgo, nos interesa mucho más la intensidad y frecuencia de esas variaciones.
        """
    )

with right:
    st.markdown("### Traducción a negocio")
    st.markdown(
        """
        Si compras inventario o dependes del precio para tu margen, no basta con saber si la papa vale 1.200 o 2.000.
        Lo importante es saber:

        - cuánto puede cambiar,
        - con qué frecuencia,
        - y qué tan doloroso sería un cambio adverso.
        """
    )

nav1, nav2 = st.columns([1, 1])
with nav1:
    st.page_link("introduccion.py", label="Volver al inicio", icon="⬅️")
with nav2:
    st.page_link(
        "pages/03_Volatilidad.py",
        label="Ir a Pantalla 3: volatilidad",
        icon="➡️",
    )
