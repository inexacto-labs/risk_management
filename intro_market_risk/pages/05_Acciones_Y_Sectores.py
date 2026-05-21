import streamlit as st

from src.charts import (
    make_base100_chart,
    make_price_chart,
    make_return_chart,
    make_rolling_vol_chart,
    make_var_distribution_chart,
    make_var_tail_chart,
)
from src.market_data import PRESET_GROUPS, load_market_data, normalize_tickers
from src.transforms import (
    compute_base100,
    compute_daily_returns,
    compute_expected_shortfall,
    compute_distribution_summary,
    compute_historical_var,
    compute_inventory_var_amount,
    compute_rolling_volatility,
    compute_volatility_summary,
)
from src.ui import apply_app_style, render_chart_info, render_page_header, render_stat_card


st.set_page_config(
    page_title="Acciones y sectores",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_style()

with st.sidebar:
    st.markdown("## Configura la comparación")
    preset_group = st.selectbox("Canasta sugerida", options=list(PRESET_GROUPS.keys()), index=0)
    preset_tickers = PRESET_GROUPS[preset_group]
    custom_tickers = st.text_input(
        "Tickers adicionales o alternativos",
        value=",".join(preset_tickers),
        help="Escribe tickers separados por comas. Ejemplo: AAPL,MSFT,KO,PG",
    )
    period_label = st.selectbox(
        "Ventana histórica",
        options=[("1 año", "1y"), ("3 años", "3y"), ("5 años", "5y")],
        index=1,
        format_func=lambda item: item[0],
    )
    rolling_window = st.slider("Ventana rolling (días)", min_value=10, max_value=90, value=30, step=5)
    position_value = st.number_input(
        "Valor de la posición (COP)",
        min_value=100000,
        max_value=500000000,
        value=10000000,
        step=100000,
    )
    confidence_pct = st.select_slider("Confianza para VaR", options=[90, 95, 99], value=95)
    var_horizon = st.slider("Horizonte VaR (días)", min_value=1, max_value=10, value=1, step=1)

tickers = normalize_tickers(custom_tickers.split(","))

render_page_header(
    eyebrow="Pantalla 5",
    title="De la papa a las acciones: misma lógica, activos distintos",
    subtitle=(
        "Ahora trasladamos la intuición construida con precios de papa al análisis de acciones. "
        "La pregunta de fondo no cambia: qué tanto varía el precio, cómo se comportan sus retornos "
        "y qué diferencias aparecen entre sectores con perfiles de riesgo distintos."
    ),
)

st.markdown("### Puente conceptual")
st.markdown(
    """
    Lo que hicimos con la papa puede traducirse de manera directa al lenguaje financiero:

    - precio de la papa -> precio de la acción,
    - cambio porcentual -> retorno,
    - dispersión de retornos -> volatilidad,
    - pérdida potencial sobre inventario -> pérdida potencial sobre una posición.

    La lógica analítica es la misma. Lo que cambia es el tipo de activo y el contexto económico.
    """
)

if not tickers:
    st.warning("Ingresa al menos un ticker para continuar con el análisis.")
    st.stop()

period = period_label[1]

try:
    market_data = load_market_data(tuple(tickers), period=period)
except RuntimeError as exc:
    st.error(
        f"No fue posible cargar datos de mercado: {exc} "
        "Si estás ejecutando la app localmente, revisa que `yfinance` esté instalado y que haya acceso a internet."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Ocurrió un problema al descargar la información de mercado: {exc}")
    st.stop()

available_tickers = [col for col in market_data.columns if col != "Fecha"]
with st.sidebar:
    selected_ticker = st.selectbox("Ticker principal", options=available_tickers, index=0)
returns = compute_daily_returns(market_data, periods=1)
base100 = compute_base100(market_data)
rolling_vol = compute_rolling_volatility(returns, window=rolling_window)
distribution_summary = compute_distribution_summary(returns, selected_ticker)
vol_summary = compute_volatility_summary(returns)
confidence = confidence_pct / 100
var_result = compute_historical_var(returns, selected_ticker, confidence, horizon=var_horizon)
var_amount = compute_inventory_var_amount(position_value, var_result["var_return"])
expected_shortfall = compute_expected_shortfall(var_result["scaled_returns"], confidence)
expected_shortfall_amount = compute_inventory_var_amount(position_value, expected_shortfall)
available_set = set(available_tickers)
requested_set = set(tickers)
missing_tickers = sorted(requested_set - available_set)

top1, top2, top3, top4 = st.columns(4)
with top1:
    render_stat_card("Ticker principal", selected_ticker)
with top2:
    render_stat_card("Volatilidad", distribution_summary["volatility"])
with top3:
    render_stat_card("Peor caída observada", distribution_summary["worst_day"])
with top4:
    render_stat_card("Mejor alza observada", distribution_summary["best_day"])

st.markdown(
    f"""
    <div class="callout">
      <strong>Lectura inicial</strong><br/>
      En <strong>{selected_ticker}</strong>, la volatilidad observada en el periodo consultado fue de
      <strong>{distribution_summary['volatility']}</strong>. Esa cifra resume la dispersión de sus retornos diarios
      y permite compararla con otras acciones o con canastas sectoriales.
    </div>
    """,
    unsafe_allow_html=True,
)
if missing_tickers:
    st.warning(
        "No se encontraron datos para estos tickers y fueron excluidos de la comparación: "
        + ", ".join(missing_tickers)
    )

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("### 1. Comparar niveles e historias de precio")
    render_chart_info(
        title="Serie de precios de acciones",
        steps=[
            "Se descargan precios de cierre ajustados para los tickers seleccionados.",
            "Cada línea muestra la trayectoria histórica del precio en el periodo consultado.",
            "Esta vista es útil para ubicar tendencias, pero no basta por sí sola para hablar de riesgo.",
        ],
        formula="Precio_t = cierre ajustado observado en la fecha t",
        note="En acciones conviene usar precios ajustados para reflejar dividendos y eventos corporativos relevantes.",
    )
    st.plotly_chart(
        make_price_chart(market_data, available_tickers, highlight=selected_ticker),
        use_container_width=True,
    )

with col2:
    st.markdown("### 2. Comparar trayectorias relativas")
    render_chart_info(
        title="Series indexadas base 100",
        steps=[
            "Cada ticker se normaliza con respecto a su primer valor disponible.",
            "Todas las series comienzan en 100 para facilitar la comparación relativa.",
            "La distancia entre líneas muestra desempeños acumulados distintos a lo largo del tiempo.",
        ],
        formula="Índice_t = (Precio_t / Precio_inicial) * 100",
        note="Esta vista ayuda a comparar el recorrido relativo de tecnológicas, consumo masivo u otros sectores.",
    )
    st.plotly_chart(
        make_base100_chart(base100, available_tickers, highlight=selected_ticker),
        use_container_width=True,
    )

col3, col4 = st.columns(2, gap="large")
with col3:
    st.markdown("### 3. Retornos: donde realmente aparece el riesgo")
    render_chart_info(
        title="Retornos diarios de acciones",
        steps=[
            "Cada precio se transforma en un cambio porcentual frente al día anterior.",
            "La secuencia resultante permite observar intensidad, frecuencia y dirección de las variaciones.",
            "Sobre esta serie es que se construyen medidas como volatilidad y VaR.",
        ],
        formula="Retorno_t = (Precio_t / Precio_(t-1)) - 1",
        note="Dos acciones pueden tener trayectorias alcistas similares y aun así niveles de riesgo muy diferentes.",
    )
    st.plotly_chart(
        make_return_chart(returns, available_tickers, highlight=selected_ticker, periods=1),
        use_container_width=True,
    )

with col4:
    st.markdown("### 4. Volatilidad rolling por sector o ticker")
    render_chart_info(
        title="Volatilidad rolling en acciones",
        steps=[
            "Se calcula la desviación estándar de una ventana móvil de retornos diarios.",
            "La ventana puede ajustarse para hacer la medida más sensible o más estable.",
            "Esto permite observar cómo cambia el nivel de riesgo en el tiempo.",
        ],
        formula="Volatilidad_t = std(Retorno_(t-n+1), ..., Retorno_t)",
        note="Una tecnológica puede exhibir picos de volatilidad más altos que una acción de consumo masivo, pero eso depende del periodo y del activo concreto.",
    )
    st.plotly_chart(
        make_rolling_vol_chart(rolling_vol, available_tickers, highlight=selected_ticker),
        use_container_width=True,
    )

st.markdown("### Comparación resumida de volatilidad")
st.caption("Esta tabla ayuda a contrastar directamente las acciones seleccionadas por volatilidad, magnitud promedio de cambio y extremos observados.")
display_summary = vol_summary.copy()
display_summary["Volatilidad"] = display_summary["Volatilidad"].map(lambda value: f"{value * 100:.2f}%")
display_summary["Cambio absoluto promedio"] = display_summary["Cambio absoluto promedio"].map(
    lambda value: f"{value * 100:.2f}%"
)
display_summary["Peor caída"] = display_summary["Peor caída"].map(lambda value: f"{value * 100:.2f}%")
display_summary["Mejor alza"] = display_summary["Mejor alza"].map(lambda value: f"{value * 100:.2f}%")
st.dataframe(display_summary, use_container_width=True, hide_index=True)
if not vol_summary.empty and len(vol_summary) >= 2:
    most_volatile = vol_summary.iloc[0]
    least_volatile = vol_summary.iloc[-1]
    st.info(
        f"En esta selección, **{most_volatile['Serie']}** aparece como la acción más volátil y "
        f"**{least_volatile['Serie']}** como la más estable. Ese contraste suele ser útil para discutir diferencias sectoriales entre crecimiento, tecnología, consumo defensivo u otros perfiles."
    )

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("### Cómo usar esta pantalla")
    st.markdown(
        """
        - Cambia la canasta sugerida para comparar sectores distintos.
        - Sustituye los tickers por otros de interés del curso.
        - Observa cómo cambia la lectura cuando pasas de precios a retornos y luego a volatilidad.
        """
    )

with right:
    st.markdown("### Lectura de transición")
    st.markdown(
        """
        Este es el punto en el que la intuición cotidiana se vuelve plenamente financiera.
        Ya no estamos viendo un insumo agrícola, sino acciones. Sin embargo, el razonamiento es el mismo:
        identificar variaciones, medir dispersión y evaluar exposición bajo incertidumbre.
        """
    )

st.markdown("### VaR aplicado a una posición en acciones")
st.markdown(
    f"""
    Si una persona mantiene una posición de **{position_value:,.0f} COP** en **{selected_ticker}**,
    ya no basta con decir que la acción es más o menos volátil. También conviene estimar cuánto podría perder
    bajo un nivel de confianza del **{confidence_pct}%** y un horizonte de **{var_horizon} día(s)**.
    """.replace(",", ".")
)

var1, var2, var3, var4 = st.columns(4)
with var1:
    render_stat_card("VaR estimado", var_amount)
with var2:
    render_stat_card("VaR como retorno", var_result["var_return_pct"])
with var3:
    render_stat_card("Expected Shortfall", expected_shortfall_amount)
with var4:
    render_stat_card("Escenarios utilizados", f"{var_result['scenario_count']}")

st.markdown(
    f"""
    <div class="callout">
      <strong>Lectura del VaR</strong><br/>
      Para <strong>{selected_ticker}</strong>, el VaR histórico estima que, con una confianza del
      <strong>{confidence_pct}%</strong>, la pérdida sobre una posición de
      <strong>{position_value:,.0f} COP</strong> no superaría <strong>{var_amount}</strong> en un horizonte de
      <strong>{var_horizon} día(s)</strong>. Si observamos únicamente los escenarios peores que ese umbral,
      la pérdida promedio sería cercana a <strong>{expected_shortfall_amount}</strong>.
    </div>
    """.replace(",", "."),
    unsafe_allow_html=True,
)

var_col1, var_col2 = st.columns(2, gap="large")
with var_col1:
    st.markdown("### 5. Distribución y percentil adverso")
    render_chart_info(
        title="VaR histórico sobre retornos de la acción",
        steps=[
            "Se calculan los retornos diarios del ticker seleccionado.",
            "Si el horizonte es mayor a un día, se construyen retornos acumulados de ese número de días.",
            "Se identifica el percentil adverso correspondiente al nivel de confianza elegido.",
        ],
        formula="VaR histórico = |percentil_(1-c)(retornos del horizonte)|",
        note="Con 95% de confianza, el umbral se ubica en el 5% más adverso de la distribución observada. Para horizontes mayores a un día, la app usa variaciones acumuladas del horizonte elegido.",
    )
    st.plotly_chart(
        make_var_distribution_chart(var_result["scaled_returns"], confidence, selected_ticker),
        use_container_width=True,
    )

with var_col2:
    st.markdown("### 6. Cola de pérdidas")
    render_chart_info(
        title="Escenarios extremos del ticker",
        steps=[
            "Los retornos históricos se ordenan de peor a mejor.",
            "Se resaltan los escenarios que quedan por debajo del umbral del VaR.",
            "La cola izquierda permite ver que el VaR resume un límite, pero no agota la severidad de las pérdidas.",
        ],
        formula="Cola izquierda = {retornos : retorno <= percentil_(1-c)}",
        note="Esta vista es especialmente útil para contrastar acciones con colas más severas, incluso cuando su volatilidad promedio no parece tan distinta.",
    )
    st.plotly_chart(
        make_var_tail_chart(var_result["scaled_returns"], confidence, selected_ticker),
        use_container_width=True,
    )

nav1, nav2 = st.columns([1, 1])
with nav1:
    st.page_link("pages/04_VaR_Como_Intuicion.py", label="Volver a Pantalla 4", icon="⬅️")
with nav2:
    st.page_link("pages/06_Renta_Fija_Y_MonteCarlo.py", label="Ir a Pantalla 6: renta fija", icon="➡️")
