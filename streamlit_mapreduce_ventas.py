import streamlit as st
from pyspark.sql.functions import sum, count, avg
from spark_connection import get_spark_session

st.title("Análisis de Ventas del Gimnasio (Spark + MongoDB)")
st.write("MapReduce distribuido usando PySpark")

# Obtener Spark y DataFrame desde tu módulo
spark, df, collection = get_spark_session()

st.subheader("Colección analizada")
st.write(collection)

st.subheader("Estructura del DataFrame")
st.write(df.limit(5).toPandas())

# MAPREDUCE (Spark)
resultado = (
    df.groupBy("producto")
      .agg(
          count("*").alias("total_ventas"),
          sum("total").alias("ingresos_totales"),
          avg("total").alias("precio_promedio")
      )
      .orderBy("ingresos_totales", ascending=False)
)

st.subheader("Resultado del MapReduce")
resultado_pd = resultado.toPandas()
st.dataframe(resultado_pd)

# Gráfica
st.subheader("Ingresos por producto")
st.bar_chart(resultado_pd.set_index("producto")["ingresos_totales"])

spark.stop()
