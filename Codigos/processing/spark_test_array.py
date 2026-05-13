from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("EmpresaTest").getOrCreate()
spark.range(5).show()
