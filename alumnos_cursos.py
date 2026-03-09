from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("LeerAlumnosMongo") \
    .config("spark.mongodb.read.connection.uri",
            "mongodb+srv://hector1985:Aime131985@utvt.qqqotrr.mongodb.net/ventas_db.alumnos_cursos") \
    .getOrCreate()

df = spark.read.format("mongodb").load()
df.show(20)
