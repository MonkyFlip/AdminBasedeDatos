#  APP DE MACHINE LEARNING CON STREAMLIT + PYSPARK
# Este sistema incluye:
# Carga de datos desde MongoDB
# Limpieza de datos
# Pipeline de Machine Learning
# Optimización (Cross Validation)
# Evaluación del modelo
# Dashboard interactivo

#  IMPORTACIONES
import sys
import os

# Agregar ruta raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px

from config.mongo_spark_conexion import get_spark_session

from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#  CONFIGURACIÓN DE LA APP

st.set_page_config(
    page_title="Dashboard ML Ventas",
    layout="wide"
)

st.title(" Sistema Inteligente de Predicción de Ingresos")

#  BOTÓN PARA EJECUTAR EL MODELO
if st.button(" Ejecutar Modelo"):

    #  1. INICIAR SPARK Y CARGAR DATOS
    st.info("Iniciando Spark y cargando datos...")

    spark, df, _ = get_spark_session()

    #  2. LIMPIEZA DE DATOS
    st.info("Limpiando datos...")

    df = df.fillna({
        "cantidad": 0,
        "precio": 0,
        "ingreso": 0
    })

    #  3. VARIABLE OBJETIVO
    df = df.withColumn(
        "categoria",
        when(col("ingreso") > 50000, 1).otherwise(0)
    )

    #  4. DIVISIÓN DE DATOS
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

    #  5. PIPELINE
    assembler = VectorAssembler(
        inputCols=["cantidad", "precio", "ingreso"],
        outputCol="features",
        handleInvalid="skip"
    )

    rf = RandomForestClassifier(
        labelCol="categoria",
        featuresCol="features",
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, rf])

    #  6. OPTIMIZACIÓN
    st.info("Optimizando modelo...")

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 50]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    evaluator = BinaryClassificationEvaluator(
        labelCol="categoria",
        metricName="areaUnderROC"
    )

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3
    )

    model = crossval.fit(train_data)

    #  7. PREDICCIONES
    predictions = model.transform(test_data)

    auc = evaluator.evaluate(predictions)

    st.success(f" Modelo entrenado | AUC: {round(auc, 4)}")

    #  8. CONVERTIR A PANDAS (para gráficas)
    pdf = predictions.select(
        "cantidad", "precio", "ingreso",
        "prediction", "categoria"
    ).toPandas()

    #  9. DASHBOARD INTERACTIVO
    
    st.subheader(" Análisis de Datos")

    #  FILTRO
    categoria = st.selectbox(
        "Filtrar por categoría real:",
        ["Todos", 0, 1]
    )

    if categoria != "Todos":
        pdf = pdf[pdf["categoria"] == categoria]

    #  MÉTRICAS
    col1, col2, col3 = st.columns(3)

    col1.metric("Total registros", len(pdf))
    col2.metric("Ingreso promedio", round(pdf["ingreso"].mean(), 2))
    col3.metric("Predicción promedio", round(pdf["prediction"].mean(), 2))

    #  GRÁFICA 1
    fig1 = px.histogram(
        pdf,
        x="ingreso",
        title="Distribución de Ingresos"
    )
    st.plotly_chart(fig1, use_container_width=True)

    #  GRÁFICA 2 (MATRIZ DE CONFUSIÓN)
    conf_matrix = pdf.groupby(
        ["categoria", "prediction"]
    ).size().reset_index(name="count")

    fig2 = px.density_heatmap(
        conf_matrix,
        x="prediction",
        y="categoria",
        z="count",
        title="Matriz de Confusión"
    )
    st.plotly_chart(fig2, use_container_width=True)

    #  GRÁFICA 3
    fig3 = px.scatter(
        pdf,
        x="precio",
        y="ingreso",
        color="prediction",
        title="Precio vs Ingreso"
    )
    st.plotly_chart(fig3, use_container_width=True)

    #  IMPORTANCIA DE VARIABLES
    st.subheader(" Importancia de Variables")

    rf_model = model.bestModel.stages[-1]

    features = ["cantidad", "precio", "ingreso"]
    importance = list(rf_model.featureImportances)

    fig4 = px.bar(
        x=features,
        y=importance,
        title="Importancia de Variables"
    )
    st.plotly_chart(fig4, use_container_width=True)

    #  TABLA
    st.subheader(" Datos procesados")
    st.dataframe(pdf.head(100))

    #  10. CERRAR SPARK
    spark.stop()
    st.info(" Sesión Spark finalizada")