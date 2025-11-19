## Fundamento Matemático

La predicción se realiza calculando la suma ponderada de las entradas más un sesgo (bias):

$$ z = \sum_{i=1}^{n} w_i x_i + b $$

La regla de activación es escalón: retorna $1$ si $z \geq 0$ y $-1$ (o $0$) en caso contrario.

### Regla de Aprendizaje

Si el algoritmo comete un error, actualiza los pesos utilizando la siguiente fórmula:

$$ w_{nuevo} = w_{actual} + \eta \cdot (y_{real} - y_{pred}) \cdot x $$

Donde $\eta$ es la tasa de aprendizaje.

## Explicación de Modos de Ejecución

### 1. Modo Binary (Binario)

Utilízalo cuando tu dataset solo tiene dos clases o cuando quieres distinguir una clase específica del resto.

- **Ejemplo:** Detectar si un correo es "Spam" (1) o "No Spam" (-1).
- **Funcionamiento:** Ajusta un vector de pesos que, al multiplicarse por las características, da un resultado positivo o negativo.

### 2. Modo OvR (One-vs-Rest)

Utilízalo cuando tenés **más de dos clases** (Multiclase). Como el perceptrón simple es binario por naturaleza, esta estrategia entrena un clasificador por cada clase.

- **Ejemplo:** Clasificar flores Iris (Setosa, Versicolor, Virginica).
- **Funcionamiento:**
  1.  Entrena "Setosa" vs "Resto".
  2.  Entrena "Versicolor" vs "Resto".
  3.  Entrena "Virginica" vs "Resto".
  4.  Al predecir, gana el clasificador que tenga la mayor activación (confianza).

### 3. Modo Grid (Búsqueda en Grilla)

Este modo es puramente educativo y útil para datasets pequeños. En lugar de usar el algoritmo de descenso de gradiente, prueba todas las combinaciones posibles de pesos dentro de un rango.

- **Uso:** Para visualizar la superficie de error o encontrar soluciones cuando el descenso de gradiente falla por mala configuración.

## Parámetros Clave

- **Tasa de aprendizaje (Learning Rate):** Qué tan rápido cambian los pesos en cada error. Un valor muy alto (ej. 1.0) puede hacer que el modelo oscile y nunca aprenda. Un valor muy bajo (ej. 0.0001) puede hacerlo muy lento. Recomendado: 0.01 a 0.1.
- **Épocas:** Cuántas veces el algoritmo revisará el dataset completo.
