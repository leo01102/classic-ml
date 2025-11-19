# Aprendizaje Automático Clásico

Este repositorio contiene implementaciones interactivas de tres algoritmos fundamentales de Machine Learning desarrollados para la cátedra de Inteligencia Artificial de la carrera de Ingeniería en Sistemas de Información.

El objetivo es proporcionar herramientas claras para entender el funcionamiento interno de estos algoritmos sin depender de librerías de alto nivel como Scikit-Learn para la lógica del modelo.

## Documentación Conceptual

Para entender en detalle cuándo utilizar cada algoritmo, qué significan sus parámetros y cómo interpretar los resultados, consultá las guías ubicadas en la carpeta `docs/`:

- [Guía del Perceptrón y Clasificación](docs/guia_perceptron.md)
- [Guía de Regresión Lineal y Regularización](docs/guia_regresion_lineal.md)
- [Guía de K-Means y Clustering](docs/guia_kmeans.md)

## Requisitos Previos

- Python 3.6 o superior
- `numpy` (Cálculo numérico base)
- `matplotlib` (Visualización de gráficos)
- `rich` (Interfaz interactiva en terminal)

Instalación de dependencias:

```bash
pip install numpy matplotlib rich
```

## Guía Rápida de Ejecución

Los scripts buscan archivos `.csv` en la carpeta `data/` o en la ruta absoluta indicada. Se asume que la última columna es la etiqueta (target).

### 1. Perceptrón (`perceptron.py`)

Clasificador lineal para problemas binarios o multiclase.

```bash
python perceptron.py
```

**Modos disponibles:**

- **Binary:** Para separar dos clases (ej. Aceptado/Rechazado).
- **OvR (One-vs-Rest):** Estrategia para clasificar más de dos clases entrenando múltiples perceptrones.
- **Test:** Verificar manualmente si un conjunto de pesos funciona.
- **Grid:** Búsqueda de fuerza bruta para encontrar pesos (educativo para 2D).

### 2. Regresión Lineal (`linear_regression.py`)

Modelo para predecir valores numéricos continuos utilizando mínimos cuadrados.

```bash
python linear_regression.py
```

**Características:**

- Cálculo mediante Ecuación Normal.
- Soporte para Regularización Ridge (L2) mediante el parámetro Lambda.
- Gráfico automático si el dataset es de una sola variable.

### 3. K-Means (`kmeans.py`)

Algoritmo no supervisado para agrupar datos basándose en similitud (distancia euclidiana).

```bash
python kmeans.py
```

**Configuración:**

- Requiere definir el número de clusters (`k`).
- Detecta automáticamente columnas numéricas y excluye texto.
- Visualización de clusters y centroides en datos 2D.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
