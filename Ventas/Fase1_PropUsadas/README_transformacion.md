# Datos y resultados de propiedades usadas

Los únicos resultados permanentes del procesamiento local son
`data/train_transformado.csv` y `data/test_transformado.csv`. Ambos conservan los
bytes de la entrega anterior. `scripts/contrato_datasets.py` contiene el esquema y las
huellas del dataset congelado para cargar Train sin red ni ejecutar el EDA.

No redefinir el corte cronológico ni recombinar los CSV. Tukey reduce solo Train;
Test conserva todas sus observaciones. `price_rub` es la respuesta; `id`,
`date_posted_dt` y `dataset_split` son auxiliares, ajenos a X. La selección y
comparación de modelos utilizan exclusivamente Train y su validación interna.

Los modelos, métricas, diagnósticos, tablas y configuraciones permanecen en
[DagsHub/MLflow](https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow).
Los tres artefactos históricos de transformación están en `transformacion/`
del Run Baseline_OLS_01 (`43e383e9b91d43efbdeda5ec1ec28f71`):
`transformadores_prop_usadas.joblib`, `manifiesto_exportacion.json` y
`resumen_final.json`. La reorganización conserva el contrato original y sus
huellas; que el módulo actual tenga otro hash no invalida los CSV congelados.

```python
from pathlib import Path
import sys
base = Path("Ventas/Fase1_PropUsadas").resolve()
sys.path.insert(0, str(base))
from scripts import baseline_ols as b
X_train, y_train, contexto = b.cargar_train(base)
```

Para recuperar transformadores sin conservar copias locales:

```python
import joblib
from scripts.almacenamiento_mlflow import SesionTemporal, RUN_TRANSFORMACION
cliente = b.conectar(base.parents[1])
with SesionTemporal() as sesion:
    ruta = cliente.download_artifacts(
        RUN_TRANSFORMACION, "transformacion/transformadores_prop_usadas.joblib",
        str(sesion.carpeta("lectura")))
    reglas = joblib.load(ruta)
# reglas sigue en memoria; el archivo temporal fue eliminado.
```

`pt.aplicar_flujo(nuevos, reglas)[0]` aplica solamente `transform`, sin Tukey ni
ajustes nuevos. Para observaciones futuras se requieren `id`, `date_posted_dt`
y `dataset_split="test"`; la respuesta puede omitirse. District conserva su
codificador independiente y las predictoras siguen la convención snake_case ASCII.

Los notebooks 2, 3 y 4 usan una `SesionTemporal` fuera del repositorio.
Cada celda se ejecuta dentro de `sesion.paso()`: un error o una interrupción
limpia los temporales y obliga a reiniciar la ejecución desde la primera celda.
La última celda libera la sesión al finalizar correctamente. Los scripts deben
usar `with SesionTemporal() as sesion:` y pasar sus carpetas temporales a las
funciones de exportación, selección y diagnóstico. Estas funciones rechazan
rutas de artefactos dentro del repositorio. No se descargan referencias permanentes.

El notebook 2 conserva transformadores en memoria durante el resumen y publica
su reproducibilidad en el Run existente bajo
`transformacion/<sha256_train>/<sha256_modulo>/`, con verificación de descarga.
Los CSV se escriben directamente en `data`. Si se cambia intencionalmente el
procesamiento, actualizar conjuntamente los dos CSV y su contrato fuente; la
limpieza estructural no modifica los datasets congelados ni sus parámetros.

El resumen de transformación permanece al final del notebook. El script
`scripts/actualizar_indice_transformacion.py --check` verifica su índice y ubicación.
`scripts/verificar_normalizacion_nombres.py` valida las huellas y el esquema contra los
artefactos remotos sin ajustar modelos ni escribir auditorías permanentes.

Los algoritmos, hiperparámetros, criterios de parada y conclusiones no cambian.
Para revisar resultados históricos se descargan los artefactos del Run original;
no se necesita repetir el entrenamiento. No ejecutar los notebooks de ajuste
completos para verificar una reorganización de archivos.

`4_Seleccion_Modelos.ipynb` incorpora Backward Selection mediante
`scripts/backward_ols.py`. Sus exclusiones son vacías: comienza con las 177
columnas de Train y 23 características originales, manteniendo completos los
bloques del linaje. Conserva la partición interna 80/20 de los experimentos
históricos (semilla 42) y utiliza `OLS_02_Log_Target` como referencia.
La trayectoria elimina un grupo por paso hasta conservar uno; luego evalúa
los tamaños de 1 a 12, exigiendo una mejora del RMSE interno en RUB de al menos
0,5 % frente al último aceptado, desde el intercepto. Los diseños de rango
deficiente se documentan y no pueden ser el OLS definitivo. Sin mejora no se
registra un ganador. La evidencia completa, la partición, el modelo y las diez
permutaciones conjuntas por grupo se conservan en MLflow.

Backward utiliza el nombre exclusivo `Backward_03_Log_177_Grupos_OLS`.
`modo="recuperar"` valida estrictamente universo, configuración, partición,
trayectoria y modelo; `modo="nuevo"` exige un nombre de Run libre. Forward y
Group Lasso mantienen sus Runs y sus 174 candidatas. La comparación de los
tres métodos es descriptiva por sus diferentes universos; la validación
interna participa en la selección y no evalúa datos nuevos independientes.

El Forward de 177 candidatas se presenta como un experimento independiente en
el mismo notebook: `Forward_04_Log_177_Grupos`, con referencia
`OLS_02_Log_Target`. `cargar_forward_177(datos)` reutiliza la carga común con
`excluidas=()` y comprueba los índices frente al experimento histórico. El
algoritmo es el mismo `forward_grupos`: intercepto inicial, incorporaciones de
bloques completos, mejora mínima del 0,5 % y máximo de 12 características.
La importancia por permutación conserva cinco repeticiones y semilla 42.
Su carpeta temporal es independiente del Forward de 174 columnas.

Después del primer registro, el notebook utiliza `modo="recuperar"`, que
comprueba también el universo de 177 candidatas, la referencia y las decisiones
del historial. Una ejecución nueva exige un nombre libre. La comparación
específica distingue candidatas y seleccionadas e informa si `to_center_km`,
`living_area` y `rooms` ingresaron, y en qué iteración. La comparación final
incluye los dos Forward, Group Lasso + OLS y Backward, conservando la advertencia
sobre el uso de la validación interna durante la selección.

`GroupLasso_05_Log_177_Grupos_OLS` agrega Group Lasso + OLS sobre las 177
candidatas, con exclusiones vacías y el Run real `Forward_04_Log_177_Grupos`
como referencia obligatoria. Se verifican su estado, universo y partición antes
de seleccionar. El experimento conserva los 40 lambdas y todos los parámetros
históricos; el escalador se ajusta solo en el entrenamiento interno y el OLS
definitivo conserva las escalas exportadas.

Se registra el rango de Gram y de sus bloques, su espectro y condición, sin
eliminar columnas. La recuperación comprueba además el escalador, la trayectoria
de coeficientes, los soportes, los residuos KKT y las decisiones por complejidad,
sin repetir FISTA ni el ajuste OLS. Los artefactos usan una carpeta independiente
`presentacion_lasso_177`; la importancia conjunta mantiene diez repeticiones con
semilla 42. El notebook compara los tres métodos con 177 candidatas y conserva
también los dos experimentos históricos de 174 columnas en la comparación general.

`scripts/bss_ols.py` construye dinámicamente el cuadro de coincidencias y la
unión de las selecciones de Forward, Backward y Group Lasso de 177 candidatas.
Se verifican sus Runs reales, grupos, huella de Train e índices internos antes
de continuar. La unión actual contiene 13 características / 146 columnas;
incluye también las características exclusivas de un solo método.

BSS enumera exhaustivamente los 8.192 subconjuntos, incluido M0 y el modelo de
13 grupos. El ganador de cada tamaño se define por el menor RSS de entrenamiento
interno en escala logarítmica, con desempate por el orden del linaje. Solo los
ganadores por tamaño se evalúan en validación. El modelo definitivo se elige
entre 1 y 12 grupos, con mejora mínima del 0,5 % frente al último aceptado,
sin detenerse ante un rechazo intermedio, y se reajusta sobre todo Train.

QR reducido y Gram/Cholesky aceleran los cálculos sin modificar las columnas
exportadas. El RSS se obtiene por normas de residuos, evitando cancelación;
los diseños singulares y los ajustes casi perfectos utilizan OLS directo.
El historial comprimido registra todas las combinaciones, rangos y estadísticas;
los ganadores se contrastan con OLS directo antes de calcular sus RMSE en RUB.
Su artefacto canónico es `historial_combinaciones.csv.zip`: ZIP evita la
descompresión automática de archivos `.gz` al descargarlos desde DagsHub y
permite conservar la verificación estricta byte a byte.

El Run exclusivo es `BSS_06_Log_Union_177_Grupos_OLS`. La recuperación verifica
los tres orígenes, la unión, la partición, la exhaustividad y las decisiones,
sin repetir la búsqueda ni el ajuste. El notebook conserva las comparaciones
históricas e incorpora las nuevas secciones de características y BSS, con tablas,
gráficos de evolución y diez permutaciones conjuntas por grupo (semilla 42).
La unión ya depende de selecciones realizadas con la misma validación: se
advierte el riesgo de sesgo por selección y optimismo, y se mantiene Test reservado.

`scripts/bss_temporal.py` añade únicamente tres folds temporales expansivos,
una tabla de doce tamaños y una decisión 1-SE. Comparte `MotorRSS` de BSS,
sin AIC, BIC, VIF, gráficos, permutaciones ni historiales de combinaciones.
Cada fold conserva solo sus doce errores; sobre todo Train se buscan únicamente
las combinaciones del tamaño elegido y se ajusta un solo OLS definitivo.

Para evitar reutilizar imputaciones y el filtro Tukey calculados con todo Train,
la CV parte de las 39.998 publicaciones crudas anteriores al corte temporal de
Test, sin ese filtro global. La preparación se ajusta exclusivamente en cada
prefijo; los bloques y el vocabulario OHE heredados permanecen fijos. El modelo
final usa todo ese Train crudo y conserva su preparador para inferencia.
Las diferencias con la población histórica filtrada, la selección previa de la
unión y el carácter práctico de 1-SE con tres folds quedan en un desplegable
cerrado. Los históricos y los CSV exportados no se modifican.

El Run independiente es `BSS_07_Log_Temporal_3F_1SE_OLS`. Solo conserva configuración,
fechas/rangos de los folds, errores, regla 1-SE, modelo y preparador definitivos,
instrucciones de inferencia y las fuentes necesarias. La recuperación verifica
identidad, esquema y selección sin volver a ejecutar los folds ni reajustar OLS.
