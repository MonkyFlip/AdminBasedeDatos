# KMEANS DISTRIBUIDO CON SPARK + MONGODB

from config.mongo_spark_conexion_sinnulos import get_spark_session
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def main():
    print("\n")
    print("=== INICIANDO KMEANS DISTRIBUIDO ===\n")
    print("\n")
    # 1.- Obtener sesión y DataFrame
    # Aquí ya tienes:
    # Columna	Significado
    # cantidad	Unidades vendidas
    # precio	Precio unitario
    #ingreso	cantidad × precio
    #features	Vector numérico para ML

    #Ejemplo interno de features:
    #[10.0, 25.0, 250.0]

    spark, df, df_vector = get_spark_session()

    total = df_vector.count()

    if total < 3:
        print("\n")
        print("No hay suficientes datos para clustering.")
        print("\n")
        spark.stop()
        return
    print("\n")
    print(f"Total registros: {total}")
    print("\n")
    # 2.- Definir modelo
    kmeans = KMeans(
        k=3,
        seed=42,
        featuresCol="features",
        predictionCol="cluster"
    )
    #¿Qué significa esto?

    #k=3: quieres 3 grupos es decir cuantos clusters o grupos quieres formar 
    #en este caso se tratara de dividir en 3 grupos distintos.
    #seed=42: reproducibilidad - Es la semilla aleatoria usada para inicializar el algoritmo, 
    #sirve para que los resultados sean reproducibles.
    #si corres el modelo varias veces con la misma semilla, obtendrás los mismos clusters.
    #featuresCol="features" - Especifica la columna que contiene el vector de características 
    #(las variables numéricas que describen cada observación), normalmente, esta columna se crea con un VectorAssembler.
    #predictionCol="cluster" → crea columna cluster - Define el nombre de la columna donde se guardará el resultado del modelo.
    #el número de cluster asignado a cada fila, aquí se llamará "cluster", pero podrías ponerle otro nombre si quisieras.

    #¿Cómo funciona K-Means matemáticamente?
    #Para cada iteración:
    #Se eligen 3 centroides iniciales
    #Se calcula distancia euclidiana
    #Cada punto se asigna al centro más cercano
    #Se recalculan centroides (media del cluster)
    #Se repite hasta converger
    #Todo esto ocurre en modo distribuido.

    # 3.- Entrenamiento
    model = kmeans.fit(df_vector)

    # 4.- Predicción
    result = model.transform(df_vector)

    print("\n")
    print("\n=== RESULTADOS ===")
    print("\n")
    result.select("cantidad", "precio", "ingreso", "cluster").show(10)

    #Ejemplo:
    #cantidad	precio	ingreso	cluster
    # 5	         100	  500	  0
    # 1	          20	  20	  2
    #10	         300	  3000	  1

    #El algoritmo está segmentando productos según:
    #Volumen
    #Precio
    #Ingreso

    #Podría interpretarse como:

    #Cluster	Interpretación posible
    #0	           Ventas medias
    #1	           Alto valor
    #2	           Bajo valor

    # 5.- Evaluación
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette"
    )

    silhouette = evaluator.evaluate(result)

    #Donde:

    #a = distancia promedio dentro del cluster

    #b = distancia al cluster más cercano

    #Valores:

    #Score	    Interpretación
    # > 0.5	    Excelente separación
    # 0.2 – 0.5	Aceptable
    # < 0.2	    Mala segmentación

    #Si obtienes 0.65 → tienes clusters bien definidos.
    print("\n")
    print(f"\nSilhouette Score: {round(silhouette, 4)}")
    print("\n")

    if silhouette > 0.5:
        print(" Buena segmentación")
    elif silhouette > 0.2:
        print(" Segmentación aceptable")
    else:
        print(" Segmentación débil")

    # 6.- Centroides
    print("\n")
    print("\n=== CENTROIDES ===")
    print("\n")
    centers = model.clusterCenters()

    #Interpretación:

    #Cluster 1 → Productos premium

    #Cluster 2 → Productos de bajo impacto

    #Cluster 0 → Intermedios


    for i, center in enumerate(centers):
        print(f"Cluster {i}: {center}")

    # 7.- Conteo por cluster (Analítica básica)
    print("\n")
    print("\n=== DISTRIBUCIÓN DE CLUSTERS ===")
    print("\n")
    result.groupBy("cluster").count().show()

    spark.stop()

    #Te dice el tamaño de cada segmento.

    #Ejemplo:

    #cluster	count
    #0	        2100
    #1	        900
    #2	        2000

    #Esto es análisis de mercado real.



if __name__ == "__main__":
    main()


#K-Means y la distancia euclidiana
#El algoritmo K-Means funciona en pasos:
#- Inicialización
#- Se eligen k puntos iniciales llamados centroides (en tu caso, k=3).
#- Asignación de puntos a clusters
#- Cada observación se asigna al cluster cuyo centroide esté más cercano.
#- Esa cercanía se mide con la distancia euclidiana:
# d(A,B)=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+\dots +(x_n-y_n)^2}- 
#Recalcular centroides
#- Se actualiza la posición de cada centroide como el promedio de los puntos 
#que le fueron asignados.
#- Iteración
#- Se repiten los pasos de asignación y recalculo hasta que los centroides 
#dejan de moverse significativamente o se alcanza un número máximo de iteraciones.

#Ejemplo ilustrativo

#Supongamos que tienes tres puntos en 2D:
#- P_1=(1,2)
#- P_2=(2,3)
#- P_3=(8,9)
#Y dos centroides iniciales:
#- C_1=(0,0)
#- C_2=(10,10)
#Distancias euclidianas:
#- d(P_1,C_1)=\sqrt{(1-0)^2+(2-0)^2}=\sqrt{1+4}=\sqrt{5}
#- d(P_1,C_2)=\sqrt{(1-10)^2+(2-10)^2}=\sqrt{81+64}=\sqrt{145}
#Como \sqrt{5}<\sqrt{145}, el punto P_1 se asigna al cluster de C_1.

#La distancia euclidiana es la regla que usa K-Means para decidir a qué cluster pertenece cada punto:
#- Si está más cerca de un centroide, se va a ese grupo.
#- Así, los clusters se forman como “zonas de influencia” alrededor de cada centroide.


