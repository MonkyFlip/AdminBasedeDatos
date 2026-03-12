# MÓDULO DE VISUALIZACIÓN PROFESIONAL

import plotly.express as px
import pandas as pd


# GRÁFICA DE DISPERSIÓN
# Permite analizar la relación entre cantidad vendida
# y el ingreso generado.

def grafica_dispersion(df):

    fig = px.scatter(
        df,
        x="cantidad",
        y="ingreso",
        color="producto",
        size="precio",
        hover_data=["precio"],
        template="plotly_dark",
        title="Relación Cantidad vs Ingreso"
    )

    fig.update_layout(
        transition_duration=500
    )

    return fig


# DISTRIBUCIÓN DE INGRESOS

def grafica_distribucion(df):

    fig = px.histogram(
        df,
        x="ingreso",
        color="producto",
        nbins=20,
        template="plotly_dark",
        title="Distribución de Ingresos"
    )

    fig.update_layout(
        transition_duration=500
    )

    return fig


# PRECIO VS INGRESO

def grafica_precio_vs_ingreso(df):

    fig = px.scatter(
        df,
        x="precio",
        y="ingreso",
        color="producto",
        trendline="ols",
        template="plotly_dark",
        title="Relación Precio vs Ingreso"
    )

    fig.update_layout(
        transition_duration=500
    )

    return fig


# COMPARACIÓN DE MODELOS

def grafica_modelos(resultados):

    df_modelos = pd.DataFrame(
        list(resultados.items()),
        columns=["Modelo", "R2"]
    )

    fig = px.bar(
        df_modelos,
        x="Modelo",
        y="R2",
        color="Modelo",
        template="plotly_dark",
        title="Comparación de Modelos ML"
    )

    return fig