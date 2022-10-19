[`Computación Flexible`](../README.md) > `Unidad 2. Adaline`

## Unidad 2. Adaline

### Objetivo

En esta clase el alumno conocerá:

* La estructura básica de Adaline. 
* Las funciones de error utilizadas en Adaline. 
* La regla del Gradientes Descendente y Gradientes Descendente Estocástico.
* Estrategia de ajuste de pesos online y banco 
* La estructura básica de Madaline

### 1. Estructura básica de Adaline

* 1943 - McCullock & Pitts (MCP) describen un célula nerviosa como una simple compuerta lógica nerviosa con salidas binarias.
* 1957 - Frank Rosenblantt publicó el primer concepto de las reglas de aprendizaje del perceptrón basado en el modelo neuronal de MCP.
* En 1959, Adaline (ADAptative Linear NEuron) fue publicada por Bernard Widrow, como un mejora del Perceptron de Rosenblatt.

**¿Cuál es la diferencia entre Perceptron y Adaline?**

*¿Qué tienen en común?*

* Son clasificadores binarios
* Ambos usan un limite de decisión lineal
* Ambos aprenden de manera iterativa, ejemplo por ejemplo ( el Perceptron de manera natural y Adaline utiliza el Gradiente Descendente Estocástico)
* Ambos usan una función de umbral.
* Ambos algoritmos calculan la salida ($z$) mediante la combinación lineal de las características (variables $x$) y los pesos del modelo ($w$)

    $z= w_0x_0 + w_1x_1+ \ldots+w_m x_m = w^Tx$
* En el Perceptron y en Adaline, se define una función umbral para realizar una predicción. Por ejemplo, si $z$ es mayor que el umbral $\theta$. Se predice la clase 1, y la clase 0 en el caso contrario.

*Las diferencias.*

* El Perceptron usa las etiquetas de las clases para aprender los coeficientes del modelo.
* Adaline utiliza valores de predicción continuos para aprender los coeficientes del modelo.

La diferencia de Adaline y el perceptrón es la regla de aprendizaje (Widrow-Hoff rule). Los pesos son actualizados en una función de activación lineal en lugar de la función de unidad paso como en el Perceptrón. En Adaline, la función de activación lineal $\phi (z)$￼es simplemente la función identidad de la salida de la red, así que: $\phi(w^Tx) = w^Tx$

### 2. Función de error

Un punto clave de los algoritmos de ML es la definición de la función de costo que es optimizada durante el proceso de aprendizaje. 
La función objetivo es frecuentemente la función de costo que se desea minimizar. En esta caso de Adaline, la función de costo es la suma de los errores cuadrados (SSE, Sum of Square Errors) entre la entrada actual calculada y la etiqueta de clase correcta:

$J(w) = \frac{1}{2} \Sigma_i (y^{(i)}-\phi(z^{(i)}))^2$

### 3. Gradiente y búsquedas de máximos y mínimos

La idea del gradiente descendiente (climbing down a hill) hasta alcanzar el mínimo costo local o global. En cada iteración, se toma un paso en sentido opuesto del gradiente, el tamaño del paso es determinado por el valor de la taza de aprendizaje, así com la pendiente del gradiente:
