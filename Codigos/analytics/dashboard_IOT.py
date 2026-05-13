# ===============================
# PROYECTO FINAL IoT + DASHBOARD
# ===============================

# ===== LIBRERÍAS =====
import streamlit as st            # Framework para crear dashboards web
import pandas as pd               # Manejo de datos tipo tabla
import numpy as np                # Operaciones numéricas
import time                       # Simulación de tiempo real
import random                     # Generación de datos aleatorios
from sklearn.linear_model import LinearRegression  # Modelo de IA simple

# ===== CONFIGURACIÓN DEL DASHBOARD =====
st.set_page_config(page_title="Sistema IoT Inteligente", layout="wide")

# Título principal
st.title("Sistema IoT Inteligente en Tiempo Real de la UTVT")

# Descripción
st.write("Simulación de sensores + análisis + predicción + alertas + tiempo real")

# ===== VARIABLES GLOBALES =====

# Lista donde guardaremos los datos generados
if "data" not in st.session_state:
    st.session_state.data = []

# ===== FUNCIONES =====

# 🔹 Función para simular sensores IoT
def generar_dato():
    """
    Genera un registro simulado de sensor
    """
    return {
        "temperatura": random.randint(20, 50),  # Temperatura aleatoria
        "humedad": random.randint(40, 90),      # Humedad aleatoria
        "timestamp": time.strftime("%H:%M:%S")  # Hora actual
    }

# 🔹 Función para entrenar modelo predictivo
def entrenar_modelo(df):
    """
    Entrena un modelo simple de regresión
    para predecir temperatura futura
    """
    if len(df) < 5:
        return None  # No hay suficientes datos
    
    # Variable independiente (tiempo)
    X = np.arange(len(df)).reshape(-1, 1)
    
    # Variable dependiente (temperatura)
    y = df["temperatura"].values
    
    # Crear modelo
    modelo = LinearRegression()
    
    # Entrenar modelo
    modelo.fit(X, y)
    
    return modelo

# ===== CONTROLES =====

# Botón para iniciar simulación
iniciar = st.button(" Iniciar Sistema")

# Botón para detener
detener = st.button(" Detener Sistema")

# Variable de control
if "run" not in st.session_state:
    st.session_state.run = False

# Cambiar estado
if iniciar:
    st.session_state.run = True

if detener:
    st.session_state.run = False

# ===== DASHBOARD EN TIEMPO REAL =====

# Contenedor dinámico
placeholder = st.empty()

# Loop principal
while st.session_state.run:

    # Generar nuevo dato
    nuevo_dato = generar_dato()
    
    # Guardar en memoria
    st.session_state.data.append(nuevo_dato)
    
    # Convertir a DataFrame
    df = pd.DataFrame(st.session_state.data)
    
    # ===== VISUALIZACIÓN =====
    with placeholder.container():
        
        st.subheader(" Datos en Tiempo Real")
        
        # Mostrar tabla
        st.dataframe(df.tail(10))
        
        # ===== MÉTRICAS =====
        col1, col2, col3 = st.columns(3)
        
        col1.metric(" Temperatura actual", df["temperatura"].iloc[-1])
        col2.metric(" Humedad actual", df["humedad"].iloc[-1])
        col3.metric(" Total registros", len(df))
        
        # ===== GRÁFICAS =====
        st.subheader(" Tendencia de Temperatura")
        st.line_chart(df["temperatura"])
        
        # ===== ALERTAS =====
        if df["temperatura"].iloc[-1] > 40:
            st.error(" ALERTA: Temperatura crítica detectada")
        
        # ===== ANÁLISIS (BIG DATA SIMPLIFICADO) =====
        st.subheader(" Análisis Estadístico")
        
        promedio = df["temperatura"].mean()
        maximo = df["temperatura"].max()
        minimo = df["temperatura"].min()
        
        st.write(f"Promedio: {promedio:.2f}")
        st.write(f"Máximo: {maximo}")
        st.write(f"Mínimo: {minimo}")
        
        # ===== INTELIGENCIA ARTIFICIAL =====
        st.subheader(" Predicción de Temperatura")
        
        modelo = entrenar_modelo(df)
        
        if modelo:
            # Predecir siguiente valor
            siguiente = modelo.predict([[len(df)]])[0]
            st.success(f"Temperatura estimada siguiente: {siguiente:.2f}°C")
        else:
            st.warning("Esperando más datos para predicción...")
    
    # Esperar 2 segundos (simulación tiempo real)
    time.sleep(2)