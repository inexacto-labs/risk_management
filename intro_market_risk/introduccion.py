from pathlib import Path

import streamlit as st

from src.data_loader import get_dataset_overview, load_papa_data
from src.ui import apply_app_style, render_page_header, render_stat_card


st.set_page_config(
    page_title="Introducción",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_style()

data = load_papa_data()
overview = get_dataset_overview(data)

render_page_header(
    eyebrow="Sesión 1",
    title="Riesgo de mercado: una introducción desde la experiencia económica cotidiana",
    subtitle=(
        "Esta sesión propone una aproximación conceptual al riesgo de mercado a "
        "partir de un contexto cercano: la evolución de los precios de la papa. "
        "El objetivo inicial no es entrar de inmediato en instrumentos financieros "
        "complejos, sino construir una intuición sólida sobre variación, "
        "incertidumbre y exposición."
    ),
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_stat_card("Observaciones", f"{overview['rows']:,}".replace(",", "."))
with col2:
    render_stat_card("Series de precios", str(overview["series_count"]))
with col3:
    render_stat_card("Fecha inicial", overview["start"])
with col4:
    render_stat_card("Fecha final", overview["end"])

st.markdown(
    """
    <div class="callout">
      <strong>Idea orientadora de la sesión</strong><br/>
      Antes de estudiar el riesgo de mercado en acciones, bonos o portafolios,
      conviene observar cómo una variación de precios afecta decisiones reales de
      compra, margen, inventario y flujo de caja.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("### Propósito de esta primera parte")
    st.markdown(
        """
        - Distinguir entre pérdida observada y riesgo como variación posible.
        - Entender por qué el análisis de precios no se agota en el nivel de la serie.
        - Comparar series tanto en niveles como en cambios porcentuales.
        - Construir la intuición necesaria para introducir después volatilidad y VaR.
        """
    )

    st.markdown("### Lógica de trabajo")
    st.markdown(
        """
        Cada pantalla sigue una secuencia pedagógica común:

        1. Una pregunta de contexto o de decisión.
        2. Una visualización o exploración del dato.
        3. Una conclusión breve sobre lo observado.
        4. La formalización conceptual correspondiente.
        """
    )

with right:
    st.markdown("### Caso de referencia")
    st.markdown(
        """
        A lo largo de la sesión trabajaremos con situaciones como estas:

        - Un comerciante que compra inventario de papa.
        - Un restaurante cuyo costo depende del precio del insumo.
        - Un analista que necesita identificar escenarios adversos.

        En todos los casos aparece una pregunta común:
        **si el precio cambia de manera significativa, qué tan expuesta queda la decisión?**
        """
    )

    st.markdown("### Siguiente paso")
    st.page_link(
        "pages/02_Precios_y_Retornos.py",
        label="Ir a Pantalla 2: precios y retornos",
        icon="📈",
    )

st.markdown("### Series disponibles para el análisis")
series_cols = st.columns(len(overview["series"]))
for col, name in zip(series_cols, overview["series"]):
    with col:
        render_stat_card("Variedad", name.replace("Papa ", ""))

st.markdown("### Recorrido de la sesión")
steps = st.columns(6)
labels = [
    ("1", "Introducir el problema", "Riesgo entendido como variación e incertidumbre."),
    ("2", "Comparar precios y retornos", "Una misma serie puede leerse de maneras distintas."),
    ("3", "Estudiar la volatilidad", "La dispersión de los cambios empieza a medirse."),
    ("4", "Llegar al VaR", "El riesgo se resume como pérdida potencial bajo un criterio estadístico."),
    ("5", "Pasar a acciones", "La intuición cotidiana se traduce al lenguaje del mercado."),
    ("6", "Cambiar a renta fija", "El riesgo ahora entra por la tasa y la valoración del bono."),
]
for col, (idx, title, body) in zip(steps, labels):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
              <div class="step-index">{idx}</div>
              <div class="step-title">{title}</div>
              <div class="step-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Ver nota técnica sobre los datos"):
    st.markdown(
        f"""
        - Archivo cargado desde `{Path('data/dataPapa.csv')}`.
        - La aplicación conserva los faltantes reales del dataset.
        - En las siguientes pantallas compararemos niveles de precio y retornos porcentuales.
        """
    )
