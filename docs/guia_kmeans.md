# Guía Técnica: K-Means Clustering

## ¿Qué es y cuándo usarlo?

K-Means es un algoritmo de **aprendizaje no supervisado**. Significa que el dataset **no tiene etiquetas** (no hay columna target). El algoritmo debe encontrar la estructura por sí mismo.

**Usá `kmeans.py` cuando:**

1.  Tenés un conjunto de datos crudos y querés descubrir patrones.
2.  Necesitás segmentar datos (ej. segmentación de clientes por compras, agrupar zonas geográficas).

## Fundamento Matemático

El objetivo es particionar $n$ observaciones en $k$ clusters, donde cada observación pertenece al cluster cuyo valor medio (centroide) es el más cercano.

### Distancia Euclidiana
Para determinar la similitud entre un punto $x$ y un centroide $c$, el script calcula la distancia euclidiana:

$$ d(x, c) = \sqrt{\sum_{i=1}^{m} (x_i - c_i)^2} $$

El algoritmo itera dos pasos:
1.  **Asignación:** Cada punto se asigna al centroide más cercano.
2.  **Actualización:** Se recalcula la posición del centroide como el promedio de los puntos asignados a él.

## Configuración Crítica

### Selección de `k` (Número de Clusters)

Es el número de grupos a encontrar.

- Si `k` es bajo: Agrupación muy general (sub-ajuste).
- Si `k` es alto: Grupos redundantes (sobre-ajuste).

### Convergencia y Tolerancia

El algoritmo funciona moviendo los centros de los grupos (centroides) iterativamente.

- **Tolerancia:** Si los centroides se mueven menos de esta distancia (ej. 0.0001) entre una iteración y otra, el algoritmo asume que terminó.
- **Max Iteraciones:** Un seguro para evitar que el programa se quede en un bucle infinito si no logra converger.

## Limitaciones a considerar

1.  **Datos Numéricos:** K-Means depende de calcular distancias matemáticas (Euclidiana). No funciona bien con datos de texto categórico (ej. "Rojo", "Azul") a menos que se conviertan a números. El script filtrará automáticamente las columnas de texto.
2.  **Sensibilidad:** El resultado puede cambiar dependiendo de dónde se inicialicen los centroides aleatoriamente al principio.
