# Guía Técnica: Regresión Lineal

## ¿Qué es y cuándo usarlo?

La regresión lineal es un algoritmo de **predicción supervisada**. A diferencia del perceptrón, no clasifica en categorías, sino que estima un valor numérico continuo basándose en la relación lineal entre entradas y salida.

**Usá `linear_regression.py` cuando:**

1.  El objetivo (target) es un número real (ej. precio, temperatura).
2.  Existe una relación lineal (proporcional) entre las variables de entrada y la salida.

## Conceptos Técnicos Implementados

### Ecuación Normal

Este script no usa iteraciones (descenso de gradiente), sino la **Ecuación Normal**, una solución analítica cerrada que encuentra el mínimo global de la función de costo instantáneamente (para datasets de tamaño moderado).

La fórmula implementada es:

$$ w = (X^T X)^{-1} X^T y $$

Donde:

- $X$: Matriz de características (con una columna de 1s agregada para el bias).
- $y$: Vector de valores objetivo.
- $w$: Vector de coeficientes resultantes.

Esto garantiza encontrar el óptimo matemático exacto para el dataset entregado, aunque puede ser costoso computacionalmente si el número de características es inmenso (miles).

### Regularización Ridge (L2)

El script solicita un parámetro `Lambda` ($\lambda$) para aplicar regularización. Esto modifica la ecuación para penalizar pesos muy grandes y evitar el sobreajuste o singularidad matemática:

$$ w = (X^T X + \lambda I)^{-1} X^T y $$

- **Si Lambda = 0:** Es una regresión lineal estándar. Puede sufrir de "sobreajuste" (overfitting) si tienes pocos datos y muchas características, o fallar si las características están correlacionadas (matriz singular).
- **Si Lambda > 0:** Es una regresión Ridge. Útil cuando $X^T X$ no es invertible o hay multicolinealidad (características correlacionadas).

## Interpretación de Resultados

Al finalizar, el script devuelve un vector $w$:
- $w_0$ (bias): Valor de la predicción cuando todas las entradas son 0 (intersección con el eje Y). Es el primer valor del vector.
- $w_1 \dots w_n$ (pesos asociados a cada característica): Cuánto aumenta la salida por cada unidad que aumenta esa característica. Son los demás valores del vector.
