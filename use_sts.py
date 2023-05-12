import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt

# Se carga el modelo desde tensorhub
# version DAN
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#version transformers
#use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# Oraciones a evaluar
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
    'soy una oracion'
]

#se pasan las oraciones por el modelo para obtener los embeddings
embeddings = use_model(sentences)

#imprime los vectores de las oraciones
for i in range(len(sentences)):
    print(f"Oracion {i+1}: {sentences[i]}")
    print(f"Vector de incrustacion: {embeddings[i]}")
    print()

# se calculan los puntos de las palabras 
similarity_matrix = []
for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
        similarity = tf.reduce_sum(tf.multiply(embeddings[i], embeddings[j])).numpy()
        row.append(similarity)
    similarity_matrix.append(row)

# Se grafican en el mapa de calor
fig = sns.heatmap(similarity_matrix, vmin=0, vmax=1, cmap="coolwarm", annot=True, fmt=".2f")
plt.xticks(ticks=range(len(sentences)), labels=sentences, rotation=90)
plt.yticks(ticks=range(len(sentences)), labels=sentences, rotation=360)
plt.title("Similitud textual Semantica de las oraciones")
plt.show()
