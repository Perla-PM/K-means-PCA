import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Abrir el dataframe
df = pd.read_csv("encuesta.csv")

# Gráfica 1: Sin estandarizar datos, promedio de respuestas por pregunta
X = df.iloc[:, 2:].values
df_preguntas = df.iloc[:, 2:]
df_preguntas_mean = df_preguntas.mean()
plt.figure(figsize=(10, 8))
df_preguntas_mean.plot(kind='barh', color='skyblue')
plt.title('Promedio de Respuestas por Pregunta (Escala Likert)')
plt.xlabel('Promedio de Respuesta')
plt.ylabel('Preguntas (Q1 - Q20)')
plt.show()

# Gráfica 2: Utilizando Estandarización
def PlotLineGraphs(original_data, standardized_data, feature_names, title, ylabel): num_features = original_data.shape[1] 
fig, axes = plt.subplots(num_features, 1, figsize=(12, 3 * num_features), sharex=True)
for i, feature in enumerate(feature_names): axes[i].plot(original_data[:, i],label=f'{feature} (Original)', linestyle='--') 
axes[i].plot(standardized_data[:, i], label=f'{feature} (Estandarizado)') 
axes[i].set_ylabel(ylabel)
axes[i].legend(loc='upper right')
axes[i].grid(True)
plt.xlabel("Índice de punto")
plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.97])
x = df.iloc[:, 2:]
feature_names = x.columns
scaler = StandardScaler()
stdDT = scaler.fit_transform(x)
PlotLineGraphs(x.values, stdDT, feature_names, "Comparación de Características Originales y Estandarizadas", "Valor")
plt.show()

# Gráfica 3: Utilizando Min-Max Scaling
min_max_scaler = MinMaxScaler()
min_max_scaled_data = min_max_scaler.fit_transform(x)
PlotLineGraphs(x.values, min_max_scaled_data, feature_names, "Comparación de Características Originales y Normalizadas (Min-Max)", "Valor Normalizado")
plt.show()

# Gráfica 4: Utilizando PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(stdDT)
plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c='skyblue')
plt.title('PCA - Componentes Principales 1 y 2')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

# Gráfica 5: Utilizando K-Means con centroides
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_data)
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:, 0], pca_data[:, 1],
c=kmeans_labels, cmap='viridis', marker='o',
label='Puntos de datos')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroides')
plt.title('K-Means - Clusters Visualizados en PCA con Centroides')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica 6: Método del Codo
inertia = []
for k in range(1, 11): kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(stdDT)
inertia.append(kmeans.inertia_)
plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Gráfica 7: K-Means con Visualización 3D y centroides
pca_3d = PCA(n_components=3)
pca_3d_data = pca_3d.fit_transform(stdDT)
kmeans_3d = KMeans(n_clusters=3, random_state=42)
kmeans_3d_labels = kmeans_3d.fit_predict(pca_3d_data)
centroids_3d = kmeans_3d.cluster_centers_
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=kmeans_3d_labels, cmap='viridis', marker='o', label='Puntos de datos')
ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2], c='red', marker='x', s=100, label='Centroides')
ax.set_title('K-Means Clustering con Visualización en3D y Centroides')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.legend()
plt.show()