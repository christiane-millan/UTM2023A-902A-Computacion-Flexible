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

* Son clasificadores binarios.
* Ambos usan un limite de decisión lineal.
* Ambos aprenden de manera iterativa, ejemplo por ejemplo ( el Perceptron de manera natural y Adaline utiliza el Gradiente Descendente Estocástico).
* Ambos usan una función de umbral.
* Ambos algoritmos calculan la salida ($z$) mediante la combinación lineal de las características (variables $x$) y los pesos del modelo ($w$).

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

![DG](./img/gradiente.png)

Usando el gradiente descendente, se pueden actualizar los pesos con un paso en la dirección opuesta del gradiente $\triangledown J(w)$, de la función de costo, $J(w)$:

$w := w + \Delta w$

El cambio de pesos $\Delta w$, es definido como el gradiente negativo multiplicado por la taza de aprendizaje, $\eta$:

$\Delta w= -\eta \triangledown J(w)$

Para calcular el gradiente de la función de costo, necesitamos calcular la derivada parcial de la función de costo con respecto a cada peso $w_j$:

$\frac{\partial J}{\partial w_j} = - \Sigma_i (y^{(i)} - \phi(z^{(i)}))x_j^{(i)}$

Así que se puede calcular la actualización de los pesos $w_j$  como:

$\Delta w_j = - \eta \frac{\partial J}{ \partial w_j} = \eta \Sigma_i (y^{(i)} - \phi(z^{(i)}))x_j^{(i)}$

Como actualizamos los pesos de manera simultánea, la regla de aprendizaje de Adaline :

$w:= w + \Delta w$

### 4. Ajuste de pesos Adaline

A pesar de que la regla de aprendizaje de Adaline se ve idéntica a la regla del Perceptrón,  debemos notar que $\phi (z^{(i)})$  con $z^{(i)} = w^Tx^{(i)}$ es un número real y no un entero como un la etiqueta de la clase. Además, la actualización de pesos se calcula a partir de todos los ejemplos de la muestra de entrenamiento (en lugar de actualizar los pesos para cada ejemplo de entrenamiento), por lo cual este enfoque es llamado como **Batch Gradiente Descendente**.

Algoritmo de entrenamiento

1. Inicializar los pesos
2. Calcular la entrada a la neurona
3. Calcular la salida con la función de activación
4. Actualizar los pesos
5. Repetir los pasos 2 al 4 hasta que las salidas reales y las deseables sean iguales para todos los vectores del conjunto de entrenamiento.

* [`Ejemplo de aprendizaje de Adaline`](./code/Ejemplo%20Adaline.ipynb)
* [`Práctica de Adaline GD`](./code/AdalineGD.ipynb)
* [`Práctica de Adaline SDG`](./code/AdalineSGD.ipynb)

### 5. Estructura básica de Madaline

### 6. Reglas de entrenamiento de Madaline

### 7. Modelado de funciones booleanas.

[`Anterior`](../README.md) | [`Siguiente`](../L03-perceptron/README.md)
