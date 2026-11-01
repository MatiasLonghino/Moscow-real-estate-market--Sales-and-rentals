# Moscow Real Estate — Análisis de precios de venta y alquiler

## Primer Parcial — Introducción al Análisis de Datos

### Datos institucionales

- **Universidad:** Universidad Tecnológica Nacional — Facultad Regional Tucumán (UTN-FRT).
- **Carrera:** Ingeniería en Sistemas de Información.
- **Asignatura:** Introducción al Análisis de Datos.
- **Integrantes:**
  - Longhino, Matías.
  - Montilla, Tomás.

## 1. Descripción del proyecto

El proyecto desarrolla un análisis exploratorio, estadístico y predictivo del mercado inmobiliario de Moscú, Rusia, utilizando información de propiedades en venta y alquiler.

El objetivo principal es construir modelos que permitan estimar el precio de una propiedad a partir de sus características físicas, ubicación y contexto inmobiliario.

Se plantean dos preguntas centrales:

1. **Alquiler:** ¿A qué precio mensual conviene ofrecer una propiedad en Moscú?
2. **Venta:** ¿Cuál es el precio de venta estimado de una propiedad en Moscú?

Para responderlas, se realizan procesos de exploración de datos (EDA), limpieza, transformación, ingeniería de características, regresión lineal múltiple y selección de variables mediante diferentes algoritmos.

El trabajo prioriza la interpretación de los factores que explican el precio, la evaluación estadística de los modelos y su capacidad de generalización.

## 2. Dataset utilizado

Se trabaja con el dataset *Moscow Real Estate Market*, que comprende información del mercado inmobiliario de Moscú entre enero de 2020 y abril de 2026.

El conjunto contiene tres segmentos del mercado inmobiliario:

| Archivo | Descripción |
|---|---|
| `rentals.csv` | Propiedades ofrecidas en alquiler. |
| `secondary_market.csv` | Propiedades usadas ofrecidas en venta. |
| `new_builds.csv` | Propiedades de obra nueva. |
| `district_prices_monthly.csv` | Indicadores mensuales de precios inmobiliarios por distrito. |
| `metro_stations.csv` | Información geográfica sobre estaciones del metro. |

Los datos permiten estudiar características como superficie, cantidad de ambientes, estado de renovación, ubicación, proximidad al metro y distancia al centro de Moscú.

**Consideración metodológica:** el dataset combina referencias geográficas y económicas reales con características de anuncios generadas mediante un modelo hedónico. Por lo tanto, los resultados representan relaciones estadísticas dentro de este conjunto de datos y no deben interpretarse como una tasación comercial validada sobre transacciones reales.

## 3. Organización del proyecto

El repositorio se organiza en dos líneas principales de análisis: alquileres y ventas de propiedades usadas.

| Directorio | Contenido |
|---|---|
| `data/` | Datasets originales y archivos procesados de alquiler. |
| `alquiler/` | EDA, limpieza, ingeniería de características, selección de modelos y mejora del modelo de alquiler. |
| `Ventas/Fase1_PropUsadas/` | Primera fase de ventas: EDA, transformación, OLS y selección de características. |
| `Ventas/Fase2_PropUsadas/` | Segunda fase de ventas: segmentación geográfica, modelos OLS y selección de variables por segmento. |
| `Data/Propiedades_Usadas/` | Material y datos transformados complementarios de ventas usadas. |
| `Presentacion/` | Presentaciones del trabajo. |
| `requirements.txt` | Dependencias del entorno Python. |

Se conservan también el dataset y la presentación de propiedades nuevas, aunque el desarrollo principal de ventas actualmente se concentra en propiedades usadas.

---

## 4. Análisis del mercado de alquileres

**Variable objetivo:** `monthly_rent_rub`, correspondiente al alquiler mensual expresado en rublos.

El análisis se desarrolla mediante cinco notebooks consecutivos.

### 4.1. Análisis Exploratorio de Datos

**Notebook:** `alquiler/01_EDA.ipynb`

Se realiza la exploración inicial del mercado de alquileres, identificando distribuciones, relaciones entre las variables y posibles problemas de calidad.

Se estudian especialmente las características de las viviendas y los factores geográficos asociados al precio del alquiler.

### 4.2. Limpieza y preparación

**Notebook:** `alquiler/02_Limpieza.ipynb`

Se aplican reglas de validación, tratamiento de inconsistencias y preparación de los datos.

Entre las decisiones metodológicas se destacan:

- Separación cronológica entre entrenamiento y evaluación.
- Tratamiento de valores atípicos mediante winsorización del objetivo exclusivamente en entrenamiento.
- Integración de indicadores contextuales con rezagos temporales para evitar información futura.
- Conservación de los valores originales del alquiler en test.

La base limpia contiene **19.661 observaciones**, distribuidas en 14.820 para entrenamiento y 4.841 para evaluación.

### 4.3. Ingeniería de características

**Notebook:** `alquiler/03_Feature_Engineering.ipynb`

Se implementa un proceso de extracción, transformación y selección inicial de variables.

Se incorporan codificaciones categóricas, transformaciones funcionales y estandarización de predictores, evaluando alternativas mediante validación temporal *walk-forward*.

Se destaca la incorporación de variables indicadoras por distrito (*one-hot encoding*) y la exclusión de promedios de alquiler derivados de la variable objetivo para evitar relaciones circulares.

La etapa genera una matriz base de **140 características predictoras** para el modelado posterior.

### 4.4. Regresión OLS y selección de modelos

**Notebook:** `alquiler/04_Modelo_OLS.ipynb`

Se comparan diferentes alternativas de modelado y selección de características:

- Regresión lineal múltiple OLS.
- Lasso.
- Best Subset Selection (BSS).
- Stepwise Selection.
- Random Forest como alternativa de comparación.
- Modelos definidos mediante selecciones manuales de características.

El análisis contempla la calidad predictiva, la complejidad y la interpretación de los modelos. La selección Stepwise constituye la referencia utilizada en la etapa siguiente.

### 4.5. Mejora del modelo final

**Notebook:** `alquiler/05_Mejora_Modelo_Final.ipynb`

Se estudian alternativas para mejorar el modelo Stepwise, considerando:

- Transformación logarítmica de la variable objetivo.
- Corrección de retransfomación mediante el factor de Duan.
- Incorporación de términos polinómicos.
- Nuevas selecciones Stepwise sobre la respuesta transformada.
- Validación temporal mediante ventanas expansivas.

El modelo seleccionado mediante el menor MAE promedio en validación temporal es **OLS con objetivo logarítmico, Stepwise sobre logaritmos y términos polinómicos**.

| Indicador de validación temporal | Referencia Stepwise | Modelo mejorado |
|---|---:|---:|
| MAE promedio | 6.697 RUB | 6.154 RUB |
| RMSE promedio | 8.810 RUB | 8.858 RUB |

La mejora del MAE alcanza aproximadamente un **8,1 %**, con un ligero incremento en el RMSE.

Se presentan resultados adicionales en test como comparación exploratoria. Debido a que dicho conjunto había intervenido en una decisión previa del notebook 04, esos resultados no se consideran una evaluación completamente independiente.

---

## 5. Análisis del mercado de ventas — Propiedades usadas

**Variable objetivo:** `price_rub`, correspondiente al precio total de venta de la propiedad en rublos.

El desarrollo se divide en dos fases metodológicas.

## 5.1. Fase 1 — Análisis general y selección de características

**Directorio:** `Ventas/Fase1_PropUsadas/`

### Exploración e integración

**Notebook:** `1_EDA_PropUsadas.ipynb`

Se desarrolla el análisis exploratorio del mercado secundario de Moscú y la integración de información inmobiliaria complementaria.

Se examinan las distribuciones de precios, las características de las propiedades, los factores geográficos, las asociaciones entre variables y los valores atípicos.

El análisis descriptivo del precio se realiza en su escala original, RUB.

### Transformación de datos

**Notebook:** `2_Transformacion_PropUsadas.ipynb`

Se prepara la información para el modelado, contemplando codificación de variables, transformaciones y control de posibles fuentes de fuga de información.

Se documentan las características y su correspondencia con las columnas transformadas, manteniendo separados los datos de entrenamiento y prueba.

Los datasets transformados y sus contratos de reproducibilidad se conservan dentro de la fase.

### Modelos de regresión OLS

**Notebook:** `3_Modelo_OLS.ipynb`

Se construye una línea base OLS y se comparan cuatro experimentos adicionales.

Se evalúan la transformación logarítmica del precio y la exclusión de características potencialmente redundantes, como distancia al centro, superficie habitable y cantidad de ambientes.

La comparación utiliza métricas de ajuste, indicadores de colinealidad y análisis de residuos, manteniendo reservado el conjunto de test.

### Selección de modelos

**Notebook:** `4_Seleccion_Modelos.ipynb`

Se implementan diferentes técnicas para encontrar subconjuntos de características explicativas:

- **Forward Selection:** incorporación progresiva de variables o grupos de variables.
- **Backward Selection:** eliminación progresiva de características desde un modelo completo.
- **Group Lasso + OLS:** selección regularizada por grupos, seguida de un ajuste OLS.
- **Best Subset Selection:** búsqueda exhaustiva de combinaciones de características.
- **BSS con validación temporal:** selección mediante tres ventanas temporales expansivas y aplicación de la regla de una desviación estándar (1-SE).

El proceso incorpora experimentos con universos de 174 y 177 columnas candidatas.

Para BSS se obtiene una unión de 13 características originales, que da lugar a **8.192 subconjuntos posibles** incluyendo el modelo nulo.

La selección temporal permite priorizar modelos parsimoniosos. El notebook documenta un modelo OLS definitivo de **6 características**, seleccionado mediante BSS y validación temporal.

La evaluación estadística y los diagnósticos de los modelos se complementan con registros de experimentación reproducibles.

## 5.2. Fase 2 — Modelado segmentado por ubicación

**Directorio:** `Ventas/Fase2_PropUsadas/`

La segunda fase estudia la heterogeneidad espacial del mercado inmobiliario, buscando determinar si diferentes zonas de Moscú requieren modelos de precios específicos.

### Exploración y preparación

**Notebooks:** `Analisis_Usadas.ipynb` y `Preparacion_OLS_Usadas.ipynb`

Se profundiza el análisis de propiedades usadas y se establece una preparación independiente para modelado.

Se realiza una partición cronológica 80/20, con 40.000 observaciones de entrenamiento y 10.000 de evaluación.

El procesamiento conserva **163 columnas predictoras**, incluyendo codificaciones categóricas y variables temporales.

### Segmentación geográfica

**Notebook:** `Modelo_OLS.ipynb`

Las propiedades se dividen en cuatro segmentos definidos por cuartiles de distancia al centro de Moscú:

| Segmento | Criterio |
|---|---|
| Muy cercanas | Primer cuartil de distancia. |
| Cercanas | Segundo cuartil de distancia. |
| Intermedias | Tercer cuartil de distancia. |
| Lejanas | Cuarto cuartil de distancia. |

Esta estrategia permite analizar diferencias en la estructura de precios entre zonas.

### Modelado y selección

Se implementan regresiones OLS con transformación Box-Cox de la variable objetivo, ajustadas por segmento.

Se desarrollan comparaciones entre modelos segmentados, variantes sin distancia al centro y estrategias de selección de características.

Entre los métodos incorporados se encuentran:

- BSS con conjunto limitado de candidatas.
- BSS exhaustivo por bloques de características.
- Control de multicolinealidad mediante VIF.
- Extensión con GVIF ajustado para características categóricas.
- Selección parsimoniosa considerando error de validación, cantidad de características y parámetros.

El procesamiento contempla métricas en rublos luego de aplicar la transformación inversa, análisis de importancia por permutación y diagnósticos residuales.

Los modelos ajustados y transformadores se conservan como artefactos serializados dentro de `Ventas/Fase2_PropUsadas/data/`.

---

## 6. Evaluación estadística y comparación de modelos

El proyecto utiliza diferentes métricas y criterios para estudiar el desempeño de los modelos.

| Métrica o criterio | Finalidad |
|---|---|
| R² | Proporción de variabilidad explicada. |
| R² ajustado | Evaluación del ajuste considerando complejidad. |
| MAE | Error absoluto promedio en rublos. |
| RMSE | Error cuadrático medio expresado en unidades del precio. |
| AIC y BIC | Comparación penalizada por complejidad. |
| VIF / GVIF | Diagnóstico de multicolinealidad. |
| Análisis residual | Evaluación de patrones, dispersión, normalidad y observaciones influyentes. |
| Validación temporal | Evaluación de modelos respetando el orden cronológico. |

Se busca equilibrar **capacidad predictiva, interpretabilidad, estabilidad y cantidad de características**.

La calidad del ajuste sobre entrenamiento no se considera suficiente por sí sola para justificar un modelo definitivo.

## 7. Reproducibilidad y seguimiento experimental

El proyecto utiliza Python y bibliotecas de análisis de datos, modelado estadístico y aprendizaje automático, entre ellas Pandas, NumPy, Statsmodels, Scikit-learn y SciPy.

Los experimentos se complementan con **MLflow y DagsHub**, utilizados para el seguimiento de configuraciones, métricas y artefactos.

[Consultar experimentos en MLflow / DagsHub](https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow/#/)

La primera fase de ventas incorpora scripts reutilizables, pruebas automatizadas y mecanismos de recuperación y verificación de experimentos.

La segunda fase incorpora registro de modelos segmentados, transformadores, resultados de selección y diagnósticos de evaluación.

## 8. Estado del proyecto

| Línea de trabajo | Desarrollo documentado |
|---|---|
| Alquileres | EDA, limpieza, ingeniería de características, comparación de modelos y mejora final. |
| Ventas — Fase 1 | EDA, preparación, OLS, selección por grupos y validación temporal. |
| Ventas — Fase 2 | Segmentación geográfica, OLS por segmento, BSS y diagnósticos. |
| Propiedades nuevas | Dataset y presentación conservados; fuera del desarrollo principal activo. |
| Experimentos | Integración con MLflow/DagsHub, pruebas y artefactos reproducibles. |

## 9. Alcance y conclusiones

El trabajo permite estudiar cómo las características de una vivienda, su ubicación y el contexto inmobiliario se relacionan con sus precios de alquiler y venta.

La comparación de distintos métodos de selección de variables permite analizar cuánto puede simplificarse un modelo sin sacrificar excesivamente su capacidad predictiva.

La segmentación geográfica agrega una perspectiva adicional al estudio de precios de propiedades usadas y permite contrastar modelos generales con modelos especializados por ubicación.

Los resultados constituyen una aproximación académica a la estimación de precios inmobiliarios. Una recomendación concreta de publicación requiere ingresar las características de la propiedad y utilizar el modelo correspondiente, considerando sus errores y limitaciones.

---

**Universidad Tecnológica Nacional — Facultad Regional Tucumán**

**Introducción al Análisis de Datos — Primer Parcial**
