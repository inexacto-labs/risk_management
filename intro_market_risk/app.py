import streamlit as st


intro = st.Page("introduccion.py", title="Introducción", icon="🏠", default=True)
precios = st.Page("pages/02_Precios_y_Retornos.py", title="Precios y Retornos", icon="📈")
volatilidad = st.Page("pages/03_Volatilidad.py", title="Volatilidad", icon="📊")
var_intuicion = st.Page("pages/04_VaR_Como_Intuicion.py", title="VaR como Intuición", icon="📉")
acciones = st.Page("pages/05_Acciones_Y_Sectores.py", title="Acciones y Sectores", icon="🏦")
renta_fija = st.Page("pages/06_Renta_Fija_Y_MonteCarlo.py", title="Renta Fija y Monte Carlo", icon="💳")

navigation = st.navigation(
    {
        "Riesgo de Mercado para no Financieros": [
            intro,
            precios,
            volatilidad,
            var_intuicion,
            acciones,
            renta_fija,
        ]
    }
)

navigation.run()
