# Práctica: Análisis de Datos y ETL

## Estructura del proyecto

```
practica_etl/
├── data/                          # Datasets generados y limpios
│   ├── ventas.csv                 # Dataset raw con duplicados y nulos
│   ├── clientes.json              # Datos de clientes (semi-estructurado)
│   ├── comentarios.txt            # Opiniones de clientes (no estructurado)
│   ├── productos.xlsx             # Catálogo de productos
│   ├── ventas_clean.parquet       # Ventas limpias (pandas ETL)
│   ├── productos_clean.parquet    # Productos limpios
│   ├── clientes_clean.parquet     # Clientes limpios
│   └── spark_ventas_clean.parquet # Ventas limpias (PySpark ETL)
│
├── notebooks/
│   ├── etl_pipeline.py            # ETL completo: pandas + scikit-learn
│   └── etl_pyspark.py             # ETL completo: PySpark + MLlib
│
└── docs/
    └── Practica_ETL_Analisis_Datos.docx   # Documento teórico (Partes I-III)
```

## Requisitos

Instalados en tu venv (ya los tienes):
- pandas, numpy, faker, pyarrow, openpyxl
- scikit-learn, pyspark==3.5.1

## Ejecución

```bash
# ETL con pandas + sklearn
python notebooks/etl_pipeline.py

# ETL con PySpark + MLlib (requiere Java 8/11/17)
python notebooks/etl_pyspark.py
```

## Resultados

| Pipeline       | MAE           | R²     |
|---------------|---------------|--------|
| LinearReg (sklearn) | $291,914  | 0.6734 |
| GradBoost (sklearn) | $222,258  | 0.8143 |
| LinearReg (MLlib)   | $165,800  | 0.9358 |
| GBT (MLlib)         | $240,300  | 0.7598 |
