import numpy as np

# Datos de entrada (4 combinaciones posibles para AND)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Salidas esperadas para AND
y = np.array([[0],
              [0],
              [0],
              [1]])

# Inicializar peso y sesgo aleatoriamente
peso = np.random.rand(2,1)
sesgo = np.random.rand(1)

# Función sigmoide (activación)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la sigmoide
def derivada_sigmoide(x):
    return x * (1 - x)

# Tasa de aprendizaje
lr = 0.1

# Entrenamiento (10000 iteraciones)
for i in range(10000):
    # Paso 1: Propagación hacia adelante
    z = np.dot(X, peso) + sesgo
    salida = sigmoid(z)
    
    # Paso 2: Calcular error
    error = y - salida
    
    # Paso 3: Ajustar peso y sesgo con gradiente descendente
    ajuste = error * derivada_sigmoide(salida)
    peso += np.dot(X.T, ajuste) * lr
    sesgo += np.sum(ajuste) * lr

# Probar la red
print("Salida predicha después del entrenamiento:")
print(np.round(salida))
