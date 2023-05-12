import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Carga el modelo USE de TensorFlow Hub
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Oraciones a comparar
sentences = [
    'hola como estas',
    'como te va',
    'buenos dias',
    'buenas tardes',
    'mañana ire de compras',
    'Ayer fui al cine',
    'Tengo mucha flojera de salir mañana',
    'Esto es una prueba de analisis',
    'La comida esta muy buena',
    'hay que tirar produccion'
]

embeddings = use_model(sentences)

#se obtienen los puntos de las oraciones
for i in range(len(sentences)):
    print(f"Oracion {i+1}: {sentences[i]}")
    print(f"Vector de incrustacion: {embeddings[i]}")
    print()
########################Similitud Coseno################################
cosine_similarity = tf.keras.losses.CosineSimilarity(axis=-1)

""" similarity_matrix = []
for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
        sim = cosine_similarity(embeddings[i][tf.newaxis, :], embeddings[j][tf.newaxis, :]).numpy()
        row.append(sim)
    similarity_matrix.append(row) """

distance_matrix = np.zeros((len(sentences), len(sentences)))
for i in range(len(sentences)):
    for j in range(len(sentences)):
        distance = 1 - np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        distance_matrix[i][j] = distance

#Graficando el mapa de calor
sns.heatmap(distance_matrix, vmin=0, vmax=1, cmap="coolwarm", annot=True, fmt=".2f")
plt.xticks(ticks=range(len(sentences)), labels=sentences, rotation=90)
plt.yticks(ticks=range(len(sentences)), labels=sentences, rotation =360)
plt.title("Similitud coseno entre las oraciones")
plt.show()
