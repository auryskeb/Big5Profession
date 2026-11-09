import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuración de la página
st.set_page_config(page_title="Big Five Personalidad & Clústeres", layout="centered")
st.title("🧠 Clasificador de Personalidad Big Five")
st.write("Ingresa tus puntajes del test de personalidad OCEAN para descubrir tu clúster y ver qué famosos comparten tu perfil.")

# 1. Cargar datos y entrenar/cargar modelos en caché para mayor rendimiento
@st.cache_resource
def cargar_modelo_y_datos():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df = pd.read_csv(url)
    dfeatures = df[["op", "co", "ex", "ag", "ne"]].copy()

    # Entrenar en caliente para asegurar la consistencia de los datos
    scaler = StandardScaler()
    dfeatures_scaled = scaler.fit_transform(dfeatures)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(dfeatures_scaled)

    return df, scaler, kmeans

try:
    df, scaler, kmeans = cargar_modelo_y_datos()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# 2. Entrada de datos mediante la barra lateral
st.sidebar.header("🎚️ Tus Puntuaciones OCEAN")
st.sidebar.write("Ajusta tus valores obtenidos en el test:")

op_user = st.sidebar.slider("Apertura a la experiencia (op)", 0.0, 100.0, 50.0, step=0.1)
co_user = st.sidebar.slider("Responsabilidad (co)", 0.0, 100.0, 50.0, step=0.1)
ex_user = st.sidebar.slider("Extroversión (ex)", 0.0, 100.0, 50.0, step=0.1)
ag_user = st.sidebar.slider("Amabilidad (ag)", 0.0, 100.0, 50.0, step=0.1)
ne_user = st.sidebar.slider("Neuroticismo (ne)", 0.0, 100.0, 50.0, step=0.1)

# Crear DataFrame para el usuario en el orden exacto
mis_datos = pd.DataFrame([{
    'op': op_user,
    'co': co_user,
    'ex': ex_user,
    'ag': ag_user,
    'ne': ne_user
}])

# 3. Predicción del clúster del usuario
mis_datos_escalados = scaler.transform(mis_datos)
mi_cluster = kmeans.predict(mis_datos_escalados)[0]

# Mostrar el clúster predicho
st.subheader(f"🎯 Has sido asignado al Clúster {mi_cluster}")

# 4. Mostrar el top 5 de famosos de ese clúster
st.markdown("### 🌟 Famosos en este mismo clúster")
famosos_cluster = df[df["cluster"] == mi_cluster]
st.dataframe(famosos_cluster[["usuario", "op", "co", "ex", "ag", "ne"]].head(5), use_container_width=True)

# 5. Obtener promedios de rasgos del clúster asignado
promedio_cluster = famosos_cluster[["op", "co", "ex", "ag", "ne"]].mean().to_dict()

# 6. Crear gráfico de radar con Plotly
categorias = ["Apertura (op)", "Responsabilidad (co)", "Extroversión (ex)", "Amabilidad (ag)", "Neuroticismo (ne)"]

# Cerrar el ciclo para el radar de ambos perfiles
valores_usuario = [op_user, co_user, ex_user, ag_user, ne_user, op_user]
valores_cluster = [promedio_cluster["op"], promedio_cluster["co"], promedio_cluster["ex"], promedio_cluster["ag"], promedio_cluster["ne"], promedio_cluster["op"]]
categorias_cerradas = categorias + [categorias[0]]

fig = go.Figure()

# Perfil del Clúster
fig.add_trace(go.Scatterpolar(
    r=valores_cluster,
    theta=categorias_cerradas,
    fill='toself',
    name=f'Promedio Clúster {mi_cluster}',
    line=dict(color='#1f77b4', width=2)
))

# Perfil del Usuario
fig.add_trace(go.Scatterpolar(
    r=valores_usuario,
    theta=categorias_cerradas,
    fill='toself',
    name='Tus Resultados',
    line=dict(color='#ff7f0e', width=3, dash='dash')
))

# Estilización del radar interactivo
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    showlegend=True,
    title=dict(
        text=f"Análisis Comparativo de Radar (Tú vs. Clúster {mi_cluster})",
        font=dict(size=16, color="black"),
        x=0.5,
        xanchor="center"
    )
)

st.plotly_chart(fig, use_container_width=True)
