from config.mongo_spark_conexion_sinnulos import get_spark_session
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import avg
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


# FUNCIÓN: INTERPRETACIÓN AUTOMÁTICA DE CLUSTERS
def interpretar_clusters(result_df):

    resumen = result_df.groupBy("prediction").agg(
        avg("precio").alias("precio_promedio"),
        avg("cantidad").alias("cantidad_promedio"),
        avg("ingreso").alias("ingreso_promedio")
    ).orderBy("prediction")

    resumen_pd = resumen.toPandas()

    
    print("INTERPRETACIÓN AUTOMÁTICA")
    

    max_ingreso = resumen_pd["ingreso_promedio"].max()
    max_precio = resumen_pd["precio_promedio"].max()

    for _, row in resumen_pd.iterrows():

        cluster = int(row["prediction"])
        precio = round(row["precio_promedio"], 2)
        cantidad = round(row["cantidad_promedio"], 2)
        ingreso = round(row["ingreso_promedio"], 2)

        print(f"Cluster {cluster}")
        print(f"  Precio promedio: {precio}")
        print(f"  Cantidad promedio: {cantidad}")
        print(f"  Ingreso promedio: {ingreso}")

        if ingreso == max_ingreso:
            print("  → Segmento Estratégico (mayor ingreso)")
        elif precio == max_precio:
            print("  → Productos Premium")
        else:
            print("  → Productos de Rotación / Operativos")

        print()


# FUNCIÓN: GRÁFICA 2D
def graficar_2d(result_df):

    pdf = result_df.select(
        "precio", "ingreso", "prediction"
    ).toPandas()

    plt.figure()

    for cluster in pdf["prediction"].unique():
        subset = pdf[pdf["prediction"] == cluster]
        plt.scatter(
            subset["precio"],
            subset["ingreso"],
            label=f"Cluster {cluster}"
        )

    plt.xlabel("Precio")
    plt.ylabel("Ingreso")
    plt.title("Clustering KMeans 2D")
    plt.legend()
    plt.show()


# FUNCIÓN: GRÁFICA 3D
def graficar_3d(result_df):

    pdf = result_df.select(
        "precio", "cantidad", "ingreso", "prediction"
    ).toPandas()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cluster in pdf["prediction"].unique():
        subset = pdf[pdf["prediction"] == cluster]
        ax.scatter(
            subset["precio"],
            subset["cantidad"],
            subset["ingreso"],
            label=f"Cluster {cluster}"
        )

    ax.set_xlabel("Precio")
    ax.set_ylabel("Cantidad")
    ax.set_zlabel("Ingreso")
    ax.set_title("Clustering KMeans 3D")

    plt.legend()
    plt.show()


# FUNCIÓN PRINCIPAL
def main():

    spark, _, df_vector = get_spark_session()

    
    print("EJECUTANDO KMEANS")
    

    # Entrenamiento
    kmeans = KMeans(k=3, seed=1)
    model = kmeans.fit(df_vector)

    # Predicción
    result = model.transform(df_vector)

    result.select("producto", "prediction").show(10)

    # Evaluación
    evaluator = ClusteringEvaluator()
    score = evaluator.evaluate(result)
    print(f"\nSilhouette Score: {round(score, 4)}")

    # Interpretación automática
    interpretar_clusters(result)

    # Visualizaciones
    graficar_2d(result)
    graficar_3d(result)

    spark.stop()


# EJECUCIÓN
if __name__ == "__main__":
    main()