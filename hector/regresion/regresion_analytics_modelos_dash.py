# MÓDULO DE MACHINE LEARNING

# Librerías de Spark ML
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def ejecutar_modelos(df):

    # CREAR VECTOR DE FEATURES
    # Spark requiere que las variables de entrada
    # se encuentren dentro de un vector llamado "features"

    if "features" not in df.columns:

        assembler = VectorAssembler(
            inputCols=["cantidad", "precio"],
            outputCol="features"
        )

        df = assembler.transform(df)

    df_vector = df

    # DIVISIÓN DEL DATASET
    # 80% entrenamiento
    # 20% prueba

    train, test = df_vector.randomSplit([0.8, 0.2])

    # Evitar error si dataset queda vacío
    if train.count() == 0 or test.count() == 0:

        return {}, df_vector

    # EVALUADOR DE MODELO
    
    evaluator = RegressionEvaluator(
        labelCol="ingreso",
        predictionCol="prediction",
        metricName="r2"
    )

    resultados = {}

    # MODELO 1
    # REGRESIÓN LINEAL
    
    lr = LinearRegression(
        featuresCol="features",
        labelCol="ingreso"
    )

    model = lr.fit(train)

    pred = model.transform(test)

    resultados["Regresión Lineal"] = evaluator.evaluate(pred)

    # MODELO 2
    # RIDGE REGRESSION
    
    ridge = LinearRegression(
        featuresCol="features",
        labelCol="ingreso",
        regParam=0.5
    )

    model_ridge = ridge.fit(train)

    pred_ridge = model_ridge.transform(test)

    resultados["Ridge"] = evaluator.evaluate(pred_ridge)

    # MODELO 3
    # LASSO REGRESSION
    
    lasso = LinearRegression(
        featuresCol="features",
        labelCol="ingreso",
        elasticNetParam=1
    )

    model_lasso = lasso.fit(train)

    pred_lasso = model_lasso.transform(test)

    resultados["Lasso"] = evaluator.evaluate(pred_lasso)

    return resultados, pred