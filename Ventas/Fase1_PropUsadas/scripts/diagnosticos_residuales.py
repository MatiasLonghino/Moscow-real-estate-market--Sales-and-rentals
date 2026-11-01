"""Evidencia gráfica de Train para cualquier OLS, sin volver a ajustar el modelo."""
import hashlib
import base64
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import statsmodels.api as sm
from matplotlib.figure import Figure
from statsmodels.nonparametric.smoothers_lowess import lowess
from threadpoolctl import threadpool_limits

ARCHIVOS = ("qq_plot.png", "scale_location.png", "residuals_vs_fitted.png")


def comparar_diagnosticos_residuales(rutas_referencia, carpeta_nuevo, *, nombre_referencia,
                                    nombre_nuevo, unidad_referencia, unidad_nuevo, archivo_html):
    """Comparación autónoma de los PNG existentes; no vuelve a calcular gráficos."""
    from html import escape
    filas = []
    for nombre in ARCHIVOS:
        columnas = []
        for ruta in [Path(rutas_referencia[nombre]), Path(carpeta_nuevo)/nombre]:
            imagen = base64.b64encode(ruta.read_bytes()).decode("ascii")
            columnas.append(f'<td style="width:50%;vertical-align:top"><img alt="{nombre}" style="width:100%" src="data:image/png;base64,{imagen}"></td>')
        filas.append("<tr>"+"".join(columnas)+"</tr>")
    contenido = ('<!doctype html><html lang="es"><meta charset="utf-8"><title>Comparación de residuos OLS</title>'
        '<body><h2>Residuos de Train: comparación de formas en sus respectivas escalas</h2>'
        '<table style="width:100%;table-layout:fixed"><thead><tr>'
        f'<th>{escape(nombre_referencia)} · {escape(unidad_referencia)}</th>'
        f'<th>{escape(nombre_nuevo)} · {escape(unidad_nuevo)}</th></tr></thead><tbody>'
        +"".join(filas)+'</tbody></table></body></html>')
    from .almacenamiento_mlflow import validar_temporal
    validar_temporal(Path(archivo_html).parent)
    Path(archivo_html).write_text(contenido, encoding="utf-8")
    return contenido


def residuos_train(modelo, diseno, y_train):
    """Reconstruye residuos y studentización interna, incluso con remove_data=True.

    r_i = e_i / sqrt((RSS / df_resid) * (1 - h_ii));
    h_ii = x_i' normalized_cov_params x_i. Solo requiere el diseño de Train.
    """
    if list(diseno.columns) != list(modelo.model.exog_names):
        raise ValueError("El orden de columnas debe coincidir con el modelo registrado.")
    if not diseno.index.equals(y_train.index):
        raise ValueError("Los índices de diseño y objetivo de Train deben coincidir.")
    X = np.asarray(diseno, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if y.ndim != 1 or len(y) != int(modelo.nobs):
        raise ValueError("Se requiere la totalidad de las observaciones de Train.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Train contiene valores no finitos; no se descartan filas.")
    ajustados = np.asarray(modelo.predict(diseno), dtype=float)
    residuos = y - ajustados
    with threadpool_limits(limits=1):
        leverage = np.einsum("ij,ij->i", X @ np.asarray(modelo.normalized_cov_params), X)
    sigma2 = float(residuos @ residuos) / float(modelo.df_resid)
    if not np.isfinite(sigma2) or sigma2 <= 0 or np.any(1 - leverage <= 0):
        raise ValueError("Residuos estandarizados indefinidos (varianza nula o leverage >= 1).")
    estandarizados = residuos / np.sqrt(sigma2 * (1 - leverage))
    if not np.isfinite(estandarizados).all():
        raise ValueError("La estandarización produjo valores no finitos; no se filtran filas.")
    return ajustados, residuos, estandarizados


def generar_diagnosticos_residuales(modelo, diseno, y_train, *, nombre_run, carpeta,
                                   unidad_objetivo="RUB"):
    """Genera siempre los tres PNG sobre todas las filas, sin modificar el OLS.

    LOWESS usa frac=0.3, it=3 y delta=1 % del rango de valores ajustados;
    delta interpola el suavizado para reducir costo sin submuestrear observaciones.
    """
    ajustados, residuos, estandarizados = residuos_train(modelo, diseno, y_train)
    from .almacenamiento_mlflow import validar_temporal
    carpeta = validar_temporal(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    sufijo = f"{nombre_run} · Train (n={len(residuos):,})"
    rutas = {}
    estilo = {"s": 9, "alpha": 0.22, "color": "#23678a", "edgecolors": "none"}

    fig = Figure(figsize=(9, 6), layout="constrained")
    ax = fig.subplots()
    sm.qqplot(residuos, line="s", ax=ax, marker="o", markersize=2.5,
              alpha=0.35, color="#23678a")
    ax.set(title=f"Normal Q-Q Plot\n{sufijo}", xlabel="Cuantiles teóricos normales",
           ylabel=f"Cuantiles de los residuos ({unidad_objetivo})")
    ax.grid(alpha=0.18)
    rutas[ARCHIVOS[0]] = carpeta / ARCHIVOS[0]
    fig.savefig(rutas[ARCHIVOS[0]], dpi=160)

    fig = Figure(figsize=(9, 6), layout="constrained")
    ax = fig.subplots()
    escala = np.sqrt(np.abs(estandarizados))
    ax.scatter(ajustados, escala, **estilo)
    curva = lowess(escala, ajustados, frac=0.3, it=3,
                   delta=float(np.ptp(ajustados)) * 0.01, missing="raise")
    ax.plot(curva[:, 0], curva[:, 1], color="#ba3b32", lw=2, label="LOWESS (frac=0.3)")
    ax.set(title=f"Scale-Location Plot\n{sufijo}",
           xlabel=f"Valores ajustados ({unidad_objetivo})",
           ylabel="√|residuos estandarizados (studentización interna)|")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    rutas[ARCHIVOS[1]] = carpeta / ARCHIVOS[1]
    fig.savefig(rutas[ARCHIVOS[1]], dpi=160)

    fig = Figure(figsize=(9, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(ajustados, residuos, **estilo)
    ax.axhline(0, color="#ba3b32", lw=1.7, linestyle="--", label="Residuo = 0")
    ax.set(title=f"Residuals vs Fitted\n{sufijo}",
           xlabel=f"Valores ajustados ({unidad_objetivo})", ylabel=f"Residuos ({unidad_objetivo})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    rutas[ARCHIVOS[2]] = carpeta / ARCHIVOS[2]
    fig.savefig(rutas[ARCHIVOS[2]], dpi=160)
    return rutas


def registrar_diagnosticos_residuales(cliente, run_id, carpeta, *, nombre_run):
    """Añade evidencia al mismo Run y verifica los tres PNG descargándolos.

    No crea Runs ni cambia su estado, modelo, métricas o parámetros.
    """
    run = cliente.get_run(run_id)
    if run.data.tags.get("mlflow.runName") != nombre_run:
        raise ValueError("El Run indicado no corresponde al modelo diagnosticado.")
    from .almacenamiento_mlflow import validar_temporal
    carpeta = validar_temporal(carpeta)
    for nombre in ARCHIVOS:
        if not (carpeta / nombre).is_file():
            raise FileNotFoundError(carpeta / nombre)
    estado = run.info.status
    for nombre in ARCHIVOS:
        cliente.log_artifact(run_id, str(carpeta / nombre), artifact_path="diagnosticos_residuales")
    with TemporaryDirectory(prefix=".verificar_", dir=carpeta.parent) as tmp:
        for nombre in ARCHIVOS:
            remoto = cliente.download_artifacts(run_id, f"diagnosticos_residuales/{nombre}", tmp)
            assert hashlib.sha256(Path(remoto).read_bytes()).digest() == hashlib.sha256((carpeta / nombre).read_bytes()).digest()
    despues = cliente.get_run(run_id)
    assert despues.info.status == estado
    assert despues.data.metrics == run.data.metrics and despues.data.params == run.data.params
    return [f"diagnosticos_residuales/{nombre}" for nombre in ARCHIVOS]
