from pathlib import Path

import streamlit as st


def apply_app_style() -> None:
    css_path = Path(__file__).resolve().parent.parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def render_page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_info(title: str, steps: list[str], formula: str | None = None, note: str | None = None) -> None:
    with st.popover("Cómo se calcula", icon="ℹ️"):
        st.markdown(f"**{title}**")
        st.markdown("**Procedimiento**")
        for step in steps:
            st.markdown(f"- {step}")
        if formula:
            st.markdown("**Expresión resumida**")
            st.code(formula, language="text")
        if note:
            st.caption(note)
