"""Registro recuperable de Fase 2: lee resultados, nunca ajusta modelos.

MLflow y autenticación siguen la conexión de Fase 1. Los modelos se guardan
individualmente; el preprocesador compartido reside en el run de preparación.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from threadpoolctl import threadpool_limits

EXPERIMENTO = "Fase2_PropUsadas"
TRACKING_URI = "https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow"
VERSION = "fase2-tracking-v1"
SEGMENTOS = ["Muy cercanas", "Cercanas", "Intermedias", "Lejanas"]
CUADERNOS = ["Analisis_Usadas", "Preparacion_OLS_Usadas", "Modelo_OLS"]


def sha(datos):
    return hashlib.sha256(datos).hexdigest()


def serializable(obj):
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return serializable(obj.tolist())
    if isinstance(obj, np.generic):
        return serializable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return obj


def json_bytes(obj):
    return json.dumps(serializable(obj), sort_keys=True, ensure_ascii=False,
                      indent=2).encode("utf-8")


def git(raiz, *args):
    return subprocess.check_output(["git", "-C", str(raiz), *args], stderr=subprocess.PIPE)


def conectar(raiz):
    if str(raiz) not in sys.path:
        sys.path.insert(0, str(raiz))
    # Reutiliza .env, variables de entorno, URI y límites de conexión de Fase 1.
    from Ventas.Fase1_PropUsadas.scripts.baseline_ols import conectar as conexion_fase1
    return conexion_fase1(raiz)


def celdas_cientificas(notebook):
    return [c for c in notebook["cells"] if not c.get("metadata", {}).get("fase2_tracking")]


def evidencia_notebook(notebook):
    celdas = celdas_cientificas(notebook)
    texto = "\n\n".join("".join(c["source"]) for c in celdas
                         if c["cell_type"] == "markdown")
    imagenes = {}
    for i, celda in enumerate(celdas):
        for j, salida in enumerate(celda.get("outputs", [])):
            png = salida.get("data", {}).get("image/png")
            if png:
                imagenes[f"celda_{i:03d}_{j}.png"] = base64.b64decode("".join(png))
    return texto, imagenes


def segmentos(distancia, limites):
    return pd.Series(np.select([distancia.le(limites["Q1"]),
                                distancia.le(limites["Q2"]),
                                distancia.le(limites["Q3"])], SEGMENTOS[:3], default=SEGMENTOS[3]),
                     index=distancia.index)


def bloques_modelo(modelo, registro):
    columnas = set(modelo.model.exog_names) - {"const"}
    bloques = {}
    for feature in registro["entrada_codificador"]:
        candidatas = registro["dummies"].get(feature, [feature])
        elegidas = [c for c in candidatas if c in columnas]
        if elegidas:
            bloques[feature] = elegidas
    assert set(c for cols in bloques.values() for c in cols) == columnas
    return bloques


def diseno(datos, columnas):
    return pd.DataFrame({c: 1.0 if c == "const" else datos[c].astype(float)
                         for c in columnas}, index=datos.index)


def errores(real, pred):
    real, pred = np.asarray(real), np.asarray(pred)
    assert np.isfinite(pred).all() and (pred > 0).all()
    error = real - pred
    return {"r2_rub": float(1 - np.sum(error**2) / np.sum((real-real.mean())**2)),
            "mae_rub": float(np.abs(error).mean()),
            "rmse_rub": float(np.sqrt(np.mean(error**2)))}


def metricas_modelo(modelo, train, test, lam, bloques):
    """Solo predict y diagnósticos de un objeto previamente ajustado."""
    X = diseno(train, modelo.model.exog_names)
    np.testing.assert_array_equal(X.to_numpy(), modelo.model.exog)
    np.testing.assert_array_equal(boxcox(train.price_rub.to_numpy(), lmbda=lam), modelo.model.endog)
    m = {"train_r2_boxcox": modelo.rsquared, "train_r2_adj_boxcox": modelo.rsquared_adj,
         "train_aic_boxcox": modelo.aic, "train_bic_boxcox": modelo.bic,
         "train_f_statistic": modelo.fvalue, "train_f_pvalue": modelo.f_pvalue,
         "train_durbin_watson": durbin_watson(modelo.resid),
         "train_jarque_bera": jarque_bera(modelo.resid)[0],
         "train_jarque_bera_pvalue": jarque_bera(modelo.resid)[1],
         "train_condition_number": modelo.condition_number,
         "n_parameters": len(modelo.params)}
    for nombre, datos in [("train", train), ("test", test)]:
        pred = inv_boxcox(modelo.predict(diseno(datos, modelo.model.exog_names)), lam)
        m.update({f"{nombre}_{k}": v for k, v in errores(datos.price_rub, pred).items()})
    columnas = [c for c in modelo.model.exog_names if c != "const"]
    from bss_features_usadas import vif_features
    principal = vif_features(modelo,bloques)
    vifs = principal.attrs['vif_columnas']
    vif = np.array([vifs[c] for c in columnas])
    m.update(train_vif_max=float(vif.max()), train_parameters_vif_gt5=int((vif > 5).sum()),
             train_features_vif_gt5=sum(any(vifs[c] > 5 for c in cols) for cols in bloques.values()))
    m.update(train_features_fail_vif_gvif=int(principal['Cumple criterio'].ne('Sí').sum()),
             train_features_not_calculable=int(principal['Cumple criterio'].eq('No calculable').sum()))
    cats = principal.loc[principal.Tipo.isin(['Nominal','Ordinal'])]
    nums = principal.loc[~principal.Tipo.isin(['Nominal','Ordinal'])]
    if len(cats):m['train_gvif_adjusted_max_categorical']=cats['Valor'].max()
    if len(nums):m['train_vif_max_numeric_binary']=nums['Valor'].max()
    return {k: float(v) for k, v in m.items() if np.isfinite(v)}


def historico_limitado(raiz):
    """Recupera bytes del historial, sin checkout ni archivos permanentes."""
    ruta = "Fase_2/data/bss_features/modelos_bss_features.joblib"
    revisiones = git(raiz, "log", "--format=%H", "--", ruta).decode().splitlines()
    for revision in revisiones:
        contenido = git(raiz, "show", f"{revision}:{ruta}")
        cache = joblib.load(io.BytesIO(contenido))
        if cache.get("configuracion", {}).get("max_candidatas") == 8:
            return cache, revision, sha(contenido)
    return None, None, None


def _run(nombre, parametros=None, metricas=None, **extra):
    return {"nombre": nombre, "parametros": {"fase": 2, **(parametros or {})},
            "metricas": metricas or {}, "archivos": {}, **extra}


def _tablas_historicas(raiz, revision, segmento):
    if not revision:
        return "No se encontraron modelos ni tablas históricas verificables."
    n = json.loads(git(raiz, "show", f"{revision}:Fase_2/Modelo_OLS.ipynb"))
    return "\n\n".join("".join(c["source"]) for c in n["cells"]
                        if c["cell_type"] == "markdown" and segmento in "".join(c["source"]))


def preparar_registro(raiz):
    """Inventario local íntegro. No conecta, escribe datasets ni invoca fit/BSS."""
    raiz = Path(raiz).resolve()
    if git(raiz, "branch", "--show-current").decode().strip() != "devLonghino":
        raise ValueError("El registro de esta entrega corresponde a devLonghino.")
    carpeta = raiz / "Ventas/Fase2_PropUsadas"
    datos = carpeta / "data"
    fuentes = {p.relative_to(raiz).as_posix(): sha(p.read_bytes())
               for p in datos.rglob("*") if p.is_file()}
    fuentes["Ventas/Fase1_PropUsadas/data/PropUsadas_Integrado.csv"] = sha(
        (raiz / "Ventas/Fase1_PropUsadas/data/PropUsadas_Integrado.csv").read_bytes())
    notebooks = {n: json.loads((carpeta / f"{n}.ipynb").read_text(encoding="utf-8")) for n in CUADERNOS}
    for n, notebook in notebooks.items():
        fuentes[f"Ventas/Fase2_PropUsadas/{n}.ipynb:contenido_cientifico"] = sha(json_bytes(celdas_cientificas(notebook)))
    for n in ["bss_features_usadas.py", "registro_experimentos.py"]:
        fuentes[f"Ventas/Fase2_PropUsadas/{n}"] = sha((carpeta / n).read_bytes())
    transformador = joblib.load(datos / "transformadores_ols_usadas.joblib")
    registro = transformador["registro"]
    originales = joblib.load(datos / "modelos_ols_por_segmento.joblib")
    sin_centro = joblib.load(datos / "modelos_ols_sin_to_center_km.joblib")
    activo = joblib.load(datos / "bss_features/modelos_bss_features.joblib")
    nuevo_gvif = activo if 'gvif' in activo.get('configuracion',{}).get('version_busqueda','') else None
    exhaustivo = activo['historico_vif'] if nuevo_gvif is not None else activo
    limitado, revision, huella_historica = historico_limitado(raiz)
    parametros = originales["parametros_segmentacion_boxcox"]
    train = pd.read_csv(datos / "train_ols_usadas.csv")
    test = pd.read_csv(datos / "test_ols_usadas.csv")
    for nombre, tabla in [("train", train), ("test", test)]:
        if sha((datos / f"{nombre}_ols_usadas.csv").read_bytes()) != parametros["entradas"][f"{nombre}_ols_usadas.csv"]:
            raise ValueError(f"La huella de {nombre} no corresponde al contrato de los modelos.")
    assert len(train) == 40000 and len(test) == 10000
    etiquetas_train = segmentos(train.to_center_km, parametros["cuartiles_distancia_km"])
    etiquetas_test = segmentos(test.to_center_km, parametros["cuartiles_distancia_km"])
    evidencia = {n: evidencia_notebook(nb) for n, nb in notebooks.items()}
    runs = []
    precio = pd.read_csv(raiz / "Ventas/Fase1_PropUsadas/data/PropUsadas_Integrado.csv", usecols=["price_rub"]).price_rub
    # Estos números históricos se verifican contra la tabla conservada y la entrada.
    texto_analisis = evidencia["Analisis_Usadas"][0]
    assert len(precio) == 50000 and precio.gt(0).all() and "47.541" in texto_analisis and "2.459" in texto_analisis
    q1, q3 = precio.quantile([.25, .75]); iqr = q3-q1
    assert int((~precio.between(q1-1.5*iqr, q3+1.5*iqr)).sum()) == 2459
    analisis = _run("F2_00_Analisis_Outliers", {"metodo": "Análisis histórico; Tukey único y reincorporación",
        "tukey_reaplicado": False, "transformaciones_historicas_sobre": "47541 filtradas"},
        {"n_entrada": 50000, "n_filtrado_historico": 47541, "n_reincorporadas": 2459,
         "n_ampliado": 50000, "tukey_q1_rub": q1, "tukey_q3_rub": q3,
         "tukey_limite_inferior_rub": q1-1.5*iqr, "tukey_limite_superior_rub": q3+1.5*iqr})
    analisis["archivos"] = {"resultados/analisis_historico.md": texto_analisis.encode(),
                            **{f"graficos/{k}": v for k, v in evidencia["Analisis_Usadas"][1].items()}}
    runs.append(analisis)
    preparacion = _run("F2_01_Preparacion_Datos", {"split_method": "Temporal 80/20",
        "fit_transformadores": "train exclusivamente", "criterios": registro["correcciones"],
        "imputacion": registro["imputacion"], "variables_eliminadas": registro["eliminadas"]},
        {"n_train": len(train), "n_test": len(test), "n_predictoras_codificadas": train.shape[1]-1,
         "n_features_originales": len(registro["entrada_codificador"])}, transformadores=transformador)
    preparacion["archivos"]["resultados/preparacion.md"] = evidencia["Preparacion_OLS_Usadas"][0].encode()
    runs.append(preparacion)
    segmentacion = _run("F2_02_Segmentacion_BoxCox", {"metodo": "Cuatro segmentos; parámetros conservados",
        "segmentacion": parametros["cuartiles_distancia_km"], "lambda_por_segmento": parametros["lambda_por_segmento"],
        "estimacion_lambda": "MLE train; no reajustada para tracking"}, {"n_train": len(train), "n_test": len(test)})
    for s in SEGMENTOS:
        slug = s.replace(" ", "_")
        segmentacion["metricas"][f"{slug}_n_train"] = int(etiquetas_train.eq(s).sum())
        segmentacion["metricas"][f"{slug}_lambda"] = parametros["lambda_por_segmento"][s]
    segmentacion["archivos"]["contrato_segmentacion.json"] = json_bytes(parametros)
    # Distribuciones originales/transformadas y tablas de los cuatro segmentos.
    nb_modelo = notebooks["Modelo_OLS"]
    prefix = {"cells": celdas_cientificas(nb_modelo)[:24]}
    txt, imagenes = evidencia_notebook(prefix)
    segmentacion["archivos"].update({"resultados/segmentacion.md": txt.encode(),
                                    **{f"graficos/{k}": v for k, v in imagenes.items()}})
    runs.append(segmentacion)
    tablas = []
    por_metodo = {}
    familias = [("03_OLS", "OLS", originales["modelos"], None),
                ("04_OLS_SinCentro", "OLS sin centro", sin_centro["modelos"], None),
                ("05_BSS_Limitado", "BSS limitado", limitado["modelos_finales"] if limitado else {}, limitado),
                ("06_BSS_Exhaustivo", "BSS exhaustivo", exhaustivo["modelos_finales"], exhaustivo)]
    if nuevo_gvif is not None:
        familias.append(('08_BSS_GVIF','BSS GVIF',nuevo_gvif['modelos_finales'],nuevo_gvif))
    for prefijo, metodo, modelos, cache in familias:
        for s in SEGMENTOS:
            nombre = f"F2_{prefijo}_{s.title().replace(' ', '_')}"
            if s not in modelos:
                r = _run(nombre, {"metodo": metodo, "segmento": s, "modelo_disponible": False},
                         disponibilidad="Histórico no recuperable; no se inventan resultados")
                r["archivos"]["resultados/historico.md"] = _tablas_historicas(raiz, revision, s).encode()
                runs.append(r)
                continue
            modelo = modelos[s]
            filas = modelo.model.data.row_labels
            train_s = train.loc[filas]
            assert etiquetas_train.loc[filas].eq(s).all()
            assert len(train_s) == int(etiquetas_train.eq(s).sum())
            test_s = test.loc[etiquetas_test.eq(s)]
            lam = parametros["lambda_por_segmento"][s]
            bloques = bloques_modelo(modelo, registro)
            with threadpool_limits(limits=1):
                metricas = metricas_modelo(modelo, train_s, test_s, lam, bloques)
            config = {"segmento": s, "metodo": metodo, "target": "price_rub",
                "target_transform": "Box-Cox", "boxcox_lambda": lam, "n_train": len(train_s),
                "n_test": len(test_s), "n_features": len(bloques), "n_parameters": len(modelo.params),
                "split_method": "Temporal 80/20", "intercepto": True,
                "selection_method": "none" if prefijo.startswith("03") else "exclusión explícita" if cache is None else
                    "BSS restringido a ocho candidatas" if cache is limitado else "BSS exhaustivo por bloques",
                "excluded_feature": "ninguna" if prefijo.startswith("03") else "to_center_km"}
            r = _run(nombre, config, metricas, modelo=modelo,
                     contrato={"columnas_diseno": modelo.model.exog_names, "features": list(bloques),
                         "feature_columnas": bloques, "boxcox_lambda": lam,
                         "limites_segmentacion": parametros["cuartiles_distancia_km"], "segmento": s,
                         "inversa": "scipy.special.inv_boxcox; sin corrección adicional"},
                     ejemplo=diseno(test_s.head(5), modelo.model.exog_names))
            config = r["parametros"]
            if not prefijo.startswith("03"):
                referencia_prefijo = "03_OLS" if cache is None else "04_OLS_SinCentro"
                ref = por_metodo[(referencia_prefijo, s)]
                r["referencia"] = ref["nombre"]
                for k in ["test_r2_rub", "test_mae_rub", "test_rmse_rub", "train_vif_max"]:
                    metricas[f"delta_{k}"] = metricas[k] - ref["metricas"][k]
                metricas["reduccion_features"] = ref["parametros"]["n_features"] - len(bloques)
                metricas["reduccion_parametros"] = ref["parametros"]["n_parameters"] - len(modelo.params)
            if cache is not None:
                seleccion = cache["seleccion"][s]
                config.update(n_candidatas=len(seleccion.get("candidatas_restringidas", seleccion["candidatas_totales"])),
                    n_candidatas_disponibles=len(seleccion["candidatas_totales"]),
                    features_seleccionadas=seleccion["features"], criterio="RMSE validación RUB; menos features, menos parámetros, RMSE",
                    tolerancia_rmse=cache["configuracion"]["tolerancia_rmse"], vif_estrictamente_menor=5,
                    repeticiones_permutation_importance=cache["configuracion"]["repeticiones"],
                    validacion_interna="Primeras 32000 de train / últimas 8000; segmentación fija",
                    boxcox_lambda_interno=seleccion["lambda_interno"], busqueda_exhaustiva=cache is not limitado)
                interno = cache["modelos_validacion"][s]
                r["modelo_validacion"] = interno
                config["n_subtrain"] = int(interno.nobs)
                config["n_validacion"] = int(etiquetas_train.loc[32000:].eq(s).sum())
                if cache is exhaustivo or cache is nuevo_gvif:
                    prueba = cache["busqueda"][s]
                    config["subconjuntos_considerados"] = prueba["estadisticas"]["subconjuntos_considerados"]
                    for k, v in prueba["estadisticas"].items():
                        if isinstance(v, (int, float)):
                            metricas[f"busqueda_{k}"] = v
                    config["modelos_admisibles"] = prueba["estadisticas"]["modelos_inversa_valida"]
                    metricas.update(validation_min_rmse_rub=prueba["minimo_rmse_factible"],
                                    validation_rmse_rub=prueba["ganador"]["RMSE validación RUB"],
                                    validation_r2_rub=prueba["ganador"]["R2 validación RUB"])
                else:
                    config.update(subconjuntos_considerados=256, revision_historica=revision)
                    r["archivos"]["resultados/historico_verificable.md"] = _tablas_historicas(raiz, revision, s).encode()
                r["cache_bss"] = cache
                if cache is nuevo_gvif:
                    # La preparación interna no cambió: puede compartirse con el run histórico.
                    assert joblib.hash(cache['preparador_validacion']) == joblib.hash(exhaustivo['preparador_validacion'])
                    config.update(version_busqueda=cache['configuracion']['version_busqueda'],
                                  regla_colinealidad=cache['configuracion']['regla_colinealidad'],
                                  selection_method='BSS exhaustivo por bloques con VIF/GVIF',
                                  vif_numericas_binarias_lt=5,gvif_ajustado_categoricas_lt=float(np.sqrt(5)))
                    r['referencia_bss_anterior']=f"F2_06_BSS_Exhaustivo_{s.title().replace(' ','_')}"
                    historico=por_metodo[('06_BSS_Exhaustivo',s)]
                    for k in ['test_r2_rub','test_mae_rub','test_rmse_rub','train_vif_max']:
                        metricas[f'delta_vs_bss_vif_{k}']=metricas[k]-historico['metricas'][k]
                    r['archivos']['resultados/comparacion_bss_anterior.md']=markdown(pd.DataFrame({
                        'Métrica':list(metricas), 'BSS GVIF':list(metricas.values()),
                        'BSS VIF anterior':[historico['metricas'].get(k,np.nan) for k in metricas]})).encode()
            por_metodo[(prefijo, s)] = r
            tablas.append({"Run": nombre, "Segmento": s, "Método": metodo,
                           "Features": len(bloques), **metricas})
            runs.append(r)
    comparacion = _run("F2_07_Comparacion_General", {"metodo": "Comparación; modelos referenciados, sin duplicarlos"})
    comparacion["tabla"] = pd.DataFrame(tablas)
    comparacion["archivos"]["resultados/modelo_ols.md"] = evidencia["Modelo_OLS"][0].encode()
    comparacion["archivos"].update({f"graficos/{k}": v for k, v in evidencia["Modelo_OLS"][1].items()})
    if revision:
        historico_nb = json.loads(git(raiz, "show", f"{revision}:Fase_2/Modelo_OLS.ipynb"))
        _, historico_png = evidencia_notebook(historico_nb)
        comparacion["archivos"].update({f"graficos/bss_limitado_historico/{k}": v for k, v in historico_png.items()})
    preparadores_internos = {}
    for metodo, cache in [("BSS limitado", limitado), ("BSS exhaustivo", exhaustivo)]:
        if cache:
            preparadores_internos[metodo] = {**cache["preparador_validacion"], "registro": registro}
    buffer = io.BytesIO()
    joblib.dump(preparadores_internos, buffer, compress=3)
    segmentacion["archivos"]["preparadores_validacion_interna.joblib"] = buffer.getvalue()
    runs.append(comparacion)
    contexto = {"version_tracking": VERSION, "rama": "devLonghino", "git_commit": git(raiz, "rev-parse", "HEAD").decode().strip(),
                "huellas": fuentes, "revision_bss_limitado": revision, "sha_bss_limitado": huella_historica}
    contexto["versiones_entorno"] = {"python": sys.version.split()[0]}
    for paquete in ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "joblib", "threadpoolctl", "matplotlib", "seaborn", "mlflow-skinny", "python-dotenv"]:
        try:
            contexto["versiones_entorno"][paquete] = version(paquete)
        except PackageNotFoundError:
            contexto["versiones_entorno"][paquete] = "No instalado"
    _completar_validacion(raiz, [r for r in runs if r['nombre'].startswith('F2_08_')] if nuevo_gvif is not None else runs,
                         registro, train, etiquetas_train)
    comparacion["tabla"] = pd.DataFrame([{"Run": r["nombre"], "Segmento": r["parametros"]["segmento"],
        "Método": r["parametros"]["metodo"], "Features": r["parametros"]["n_features"], **r["metricas"]}
        for r in runs if "modelo" in r])
    referencias_esperadas = {}
    if nuevo_gvif is not None:
        referencias_esperadas = {r['nombre']:r for r in runs if r['nombre'] in
            {'F2_01_Preparacion_Datos','F2_02_Segmentacion_BoxCox'} or
            r['nombre'].startswith(('F2_04_','F2_06_'))}
        runs = [r for r in runs if r['nombre'].startswith('F2_08_')]
        # La figura comparativa existente se conserva íntegra como artefacto.
        for r in runs:
            r['archivos'].update({f'graficos/comparativa/{k}':v for k,v in evidencia['Modelo_OLS'][1].items()
                                 if k.startswith('celda_126_')})
    for r in runs:
        r["identidad"] = sha(json_bytes({"contexto": contexto, "nombre": r["nombre"], "parametros": r["parametros"],
                                       "metricas": r["metricas"]}))
    return {"raiz": raiz, "runs": runs, "contexto": contexto, "parametros": parametros,
            "transformadores": transformador, "train": train, "test": test,
            "referencias_esperadas":referencias_esperadas}


def markdown(tabla):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.9g}" if np.isfinite(v) else "No disponible"
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(fmt, tabla.columns)) + " |",
        "| " + " | ".join(["---"] * len(tabla.columns)) + " |"] +
        ["| " + " | ".join(map(fmt, fila)) + " |" for fila in tabla.itertuples(index=False, name=None)])


def _completar_validacion(raiz, runs, registro, train, etiquetas):
    """Recupera validación con transformadores guardados; ningún fit ni búsqueda.

    Solo se reconstruyen las 30 permutaciones determinísticas para exportar los
    paneles individuales que antes estaban guardados en una figura compuesta.
    """
    from bss_features_usadas import entrada_codificador, importancia_permutacion
    import matplotlib.pyplot as plt
    import seaborn as sns
    colores = dict(zip(SEGMENTOS, ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]))
    datos = raiz / "Ventas/Fase2_PropUsadas/data"
    meta = pd.read_csv(datos / "metadatos_train_ols_usadas.csv")
    assert len(meta) == len(train) and meta.id.is_unique
    raw = pd.read_csv(raiz / "Ventas/Fase1_PropUsadas/data/PropUsadas_Integrado.csv").set_index("id", drop=False)
    raw = raw.loc[meta.id].reset_index(drop=True)
    np.testing.assert_array_equal(raw.price_rub, train.price_rub)
    preparados = {}
    for r in runs:
        cache = r.pop("cache_bss", None)
        if cache is None:
            continue
        clave = r["parametros"]["metodo"]
        if clave not in preparados:
            preparador = {**cache["preparador_validacion"], "registro": registro}
            A, y_sub = entrada_codificador(raw.iloc[:32000], preparador)
            V, y_val = entrada_codificador(raw.iloc[32000:], preparador)
            nombres = preparador["encoder"].get_feature_names_out()
            X = pd.DataFrame(preparador["encoder"].transform(A), columns=nombres, index=A.index)
            V = pd.DataFrame(preparador["encoder"].transform(V), columns=nombres, index=V.index)
            preparados[clave] = (X, V, y_sub, y_val)
        X, V, y_sub, y_val = preparados[clave]
        s = r["parametros"]["segmento"]
        interno = r["modelo_validacion"]
        i = X.index[etiquetas.loc[X.index].eq(s)]
        j = V.index[etiquetas.loc[V.index].eq(s)]
        np.testing.assert_array_equal(diseno(X.loc[i], interno.model.exog_names), interno.model.exog)
        lam = r["parametros"]["boxcox_lambda_interno"]
        np.testing.assert_array_equal(boxcox(y_sub.loc[i].to_numpy(), lmbda=lam), interno.model.endog)
        pred = inv_boxcox(interno.predict(diseno(V.loc[j], interno.model.exog_names)), lam)
        met = errores(y_val.loc[j], pred)
        for k, valor in met.items():
            nombre = f"validation_{k}"
            if nombre in r["metricas"]:
                np.testing.assert_allclose(r["metricas"][nombre], valor, rtol=1e-10)
            r["metricas"][nombre] = valor
        if "Limitado" in r["nombre"]:
            r["parametros"].update(modelos_admisibles="No persistido; no se repite la búsqueda",
                rmse_minimo_validacion="No persistido; solo se verifica el modelo seleccionado")
            continue
        b = {"modelo_interno": interno, "columnas_internas": interno.model.exog_names[1:],
             "lambda_interno": lam, "seleccionadas": cache["seleccion"][s]["features"],
             "bloques": {f: [c for c in cols if c in interno.model.exog_names]
                         for f, cols in cache["preparador_validacion"]["mapa"].items()}}
        with threadpool_limits(limits=1):
            detalle, resumen, rmse = importancia_permutacion(b, V.loc[j], y_val.loc[j],
                cache["configuracion"]["semilla"] + SEGMENTOS.index(s), 30)
        np.testing.assert_allclose(rmse, met["rmse_rub"], rtol=1e-10)
        assert detalle["Inversas inválidas"].eq(0).all()
        r["archivos"]["resultados/importancia_agrupada.md"] = markdown(resumen).encode()
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.boxplot(data=detalle, y="Característica", x="ΔRMSE RUB", order=resumen["Característica"].tolist(),
                        color=colores[s], ax=ax, flierprops={"markersize": 3, "alpha": .5})
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.set(title=f"{s} — {r['parametros']['metodo']} — 30 permutaciones por característica",
                   xlabel="Incremento del RMSE de validación (RUB)", ylabel="")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=120)
            plt.close(fig)
        r["archivos"]["graficos/importancia_por_feature.png"] = buffer.getvalue()


class ConflictoRegistro(ValueError):
    """Identidad remota ambigua: se detiene solo el experimento afectado."""


def _buscar_runs(cliente, eid, filtro):
    salida, token = [], None
    while True:
        pagina = cliente.search_runs([eid], filter_string=filtro, max_results=1000, page_token=token)
        salida.extend(pagina)
        token = getattr(pagina, "token", None)
        if not token:
            return salida


def resolver_run(cliente, eid, r):
    existentes = _buscar_runs(cliente, eid, f"tags.fase2_base = '{r['nombre']}'")
    sin_etiqueta = _buscar_runs(cliente, eid, f"tags.mlflow.runName = '{r['nombre']}'")
    existentes = list({x.info.run_id: x for x in existentes + sin_etiqueta}.values())
    if any(not x.data.tags.get("fase2_identidad") for x in existentes):
        raise ConflictoRegistro("Hay un nombre histórico sin identidad verificable.")
    exactos = [x for x in existentes if x.data.tags.get("fase2_identidad") == r["identidad"]]
    if len(exactos) > 1 or (exactos and exactos[0].info.status != "FINISHED"):
        raise ConflictoRegistro("Identidad duplicada o ejecución anterior sin finalizar correctamente.")
    if exactos:
        return exactos[0], exactos[0].data.tags["mlflow.runName"]
    nombre = r["nombre"] if not existentes else f"{r['nombre']}__v_{r['identidad'][:12]}"
    colisiones = _buscar_runs(cliente, eid, f"tags.mlflow.runName = '{nombre}'")
    if colisiones:
        raise ConflictoRegistro("El nombre de la versión ya está ocupado por otro registro.")
    return None, nombre


def pendientes(plan, cliente):
    experimento = cliente.get_experiment_by_name(EXPERIMENTO)
    filas = []
    for r in plan["runs"]:
        try:
            existente, nombre = resolver_run(cliente, experimento.experiment_id, r) if experimento else (None, r["nombre"])
            filas.append({"Run": nombre, "Estado": "Reutilizable" if existente else "Pendiente",
                          "Run ID": existente.info.run_id if existente else ""})
        except ConflictoRegistro as exc:
            filas.append({"Run": r["nombre"], "Estado": f"Conflicto: {exc}", "Run ID": ""})
    return pd.DataFrame(filas)


def _exportar(r, plan, carpeta, ids):
    archivos = dict(r["archivos"])
    parametros = dict(r["parametros"])
    contrato = copy.deepcopy(r.get("contrato", {}))
    if "referencia" in r:
        if r["referencia"] not in ids:
            raise ConflictoRegistro("El modelo de referencia no tiene un run verificado.")
        parametros["reference_run_id"] = ids[r["referencia"]]
    if 'referencia_bss_anterior' in r:
        parametros['previous_bss_run_id']=ids[r['referencia_bss_anterior']]
    if "modelo" in r:
        if "F2_01_Preparacion_Datos" not in ids:
            raise ConflictoRegistro("Falta el preprocesador remoto verificado.")
        contrato["transformadores_run_id"] = ids["F2_01_Preparacion_Datos"]
        contrato["transformadores_artifact"] = "transformadores_ols_usadas.joblib"
        parametros["transformadores_run_id"] = ids["F2_01_Preparacion_Datos"]
        modelo = copy.deepcopy(r["modelo"])
        modelo.remove_data()
        assert modelo.model.exog is None and modelo.model.endog is None
        paquete = {"modelo": modelo, "contrato": contrato}
        buffer = io.BytesIO()
        joblib.dump(paquete, buffer, compress=3)
        archivos["modelo_segmento.joblib"] = buffer.getvalue()
        # Validar la serialización y la inversa sin ajustar coeficientes.
        recuperado = joblib.load(io.BytesIO(buffer.getvalue()))
        ejemplo = r["ejemplo"]
        np.testing.assert_array_equal(recuperado["modelo"].params, r["modelo"].params)
        np.testing.assert_allclose(recuperado["modelo"].predict(ejemplo), r["modelo"].predict(ejemplo), rtol=1e-12)
        archivos["inferencia.json"] = json_bytes(contrato)
        if "modelo_validacion" in r:
            interno = copy.deepcopy(r["modelo_validacion"])
            interno.remove_data()
            buffer = io.BytesIO()
            joblib.dump({"modelo": interno, "lambda_interno": parametros["boxcox_lambda_interno"],
                         "preparadores_run_id": ids["F2_02_Segmentacion_BoxCox"],
                         "preparadores_artifact": "preparadores_validacion_interna.joblib",
                         "clave_preparador": 'BSS exhaustivo' if parametros['metodo']=='BSS GVIF' else parametros["metodo"]}, buffer, compress=3)
            archivos["modelo_validacion_segmento.joblib"] = buffer.getvalue()
    if "transformadores" in r:
        # Única copia remota compartida, con sus reglas aprendidas de train.
        archivos["transformadores_ols_usadas.joblib"] = (plan["raiz"] / "Ventas/Fase2_PropUsadas/data/transformadores_ols_usadas.joblib").read_bytes()
        for nombre in ["registro_experimentos.py", "bss_features_usadas.py"]:
            archivos[f"codigo/{nombre}"] = (plan["raiz"] / "Ventas/Fase2_PropUsadas" / nombre).read_bytes()
    if "tabla" in r:
        archivos["resultados/comparacion_general.md"] = markdown(r["tabla"].T.rename_axis("Métrica").reset_index()).encode()
        archivos["referencias_runs.json"] = json_bytes(ids)
    archivos["procedencia.json"] = json_bytes({**plan["contexto"], "identidad": r["identidad"],
        "parametros": parametros, "metricas": r["metricas"],
        "metricas_no_disponibles": ["BSS limitado: mínimo RMSE y conteo de admisibles no persistidos"]
            if "Limitado" in r["nombre"] else []})
    archivos["resultados/metricas.md"] = markdown(pd.DataFrame(r["metricas"].items(), columns=["Métrica", "Valor"])).encode()
    for ruta, contenido in archivos.items():
        p = carpeta / ruta
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(contenido)
    manifiesto = {ruta: sha(contenido) for ruta, contenido in archivos.items()}
    (carpeta / "manifiesto.json").write_bytes(json_bytes(manifiesto))
    return parametros, manifiesto


def verificar_remoto(cliente, rid, r, parametros, temporal, *, manifiesto=None):
    temporal.mkdir(parents=True, exist_ok=True)
    remoto = cliente.get_run(rid)
    assert remoto.data.tags["fase2_identidad"] == r["identidad"]
    assert remoto.info.status in ("RUNNING", "FINISHED")
    for k, v in r["metricas"].items():
        np.testing.assert_allclose(remoto.data.metrics[k], v, rtol=1e-10, atol=1e-12)
    for k, v in parametros.items():
        texto = valor_parametro(k, v)
        assert remoto.data.params[k] == texto, k
    archivo = cliente.download_artifacts(rid, "manifiesto.json", str(temporal))
    guardado = json.loads(Path(archivo).read_text(encoding="utf-8"))
    if manifiesto is not None:
        assert guardado == manifiesto
    for ruta, huella in guardado.items():
        recibido = Path(cliente.download_artifacts(rid, ruta, str(temporal)))
        assert sha(recibido.read_bytes()) == huella, ruta
    if "modelo" in r:
        paquete = joblib.load(temporal / "modelo_segmento.joblib")
        np.testing.assert_array_equal(paquete["modelo"].params, r["modelo"].params)
        np.testing.assert_allclose(paquete["modelo"].predict(r["ejemplo"]), r["modelo"].predict(r["ejemplo"]), rtol=1e-12)


def valor_parametro(clave, valor):
    texto = json_bytes(valor).decode() if isinstance(valor, (dict, list)) else str(valor)
    # El contenido completo siempre queda en procedencia.json; no truncarlo.
    return texto if len(texto) <= 500 else f"artifact:procedencia.json#parametros/{clave}; sha256={sha(texto.encode())}"


def registrar_experimentos(plan, cliente=None, progreso=print):
    """Publica solo runs pendientes; verifica métricas y cada artefacto descargado."""
    from mlflow.entities import Metric, Param
    raiz = plan["raiz"]
    if cliente is None:
        cliente = conectar(raiz)
    exp = cliente.get_experiment_by_name(EXPERIMENTO)
    if exp is None:
        try:
            eid = cliente.create_experiment(EXPERIMENTO)
        except Exception:
            exp = cliente.get_experiment_by_name(EXPERIMENTO)
            if exp is None:
                raise
            eid = exp.experiment_id
    else:
        if exp.lifecycle_stage != "active":
            raise ConflictoRegistro("El experimento existe archivado; no se modifica automáticamente.")
        eid = exp.experiment_id
    filas, ids = [], {}
    for nombre, esperado in plan.get('referencias_esperadas',{}).items():
        candidatos=_buscar_runs(cliente,eid,f"tags.fase2_base = '{nombre}'")
        candidatos=[r for r in candidatos if r.info.status=='FINISHED']
        if len(candidatos)!=1:
            raise ConflictoRegistro(f'Referencia histórica ambigua o ausente: {nombre}')
        ref=candidatos[0]
        with TemporaryDirectory(prefix='verificar_referencia_f2_') as tmp:
            procedencia=Path(cliente.download_artifacts(ref.info.run_id,'procedencia.json',tmp))
            huellas=json.loads(procedencia.read_text(encoding='utf-8'))['huellas']
            for relativa,huella in plan['contexto']['huellas'].items():
                if relativa.startswith('Ventas/Fase2_PropUsadas/data/') and 'bss_features/' not in relativa:
                    assert huellas.get(relativa, huellas.get(relativa.replace('Ventas/Fase2_PropUsadas/', 'Fase_2/')))==huella,relativa
            if 'modelo' in esperado:
                recuperado=joblib.load(cliente.download_artifacts(ref.info.run_id,'modelo_segmento.joblib',tmp))
                np.testing.assert_array_equal(recuperado['modelo'].params,esperado['modelo'].params)
        ids[nombre]=ref.info.run_id
    for r in plan["runs"]:
        rid = None
        creado = False
        try:
            previo, nombre = resolver_run(cliente, eid, r)
            progreso(f"{nombre}: {'verificando existente' if previo else 'registrando'}", flush=True)
            with TemporaryDirectory(prefix="moscow_fase2_mlflow_") as tmp:
                temporal = Path(tmp)
                if temporal.resolve().is_relative_to(raiz):
                    raise ValueError("Los artefactos temporales deben estar fuera del repositorio.")
                carpeta = temporal / "salida"
                carpeta.mkdir()
                params, manifiesto = _exportar(r, plan, carpeta, ids)
                if previo:
                    rid = previo.info.run_id
                    verificar_remoto(cliente, rid, r, params, temporal / "descarga")
                else:
                    run = cliente.create_run(eid, tags={"mlflow.runName": nombre, "fase2_base": r["nombre"],
                        "fase2_identidad": r["identidad"], "fase2_version": VERSION,
                        "git_commit": plan["contexto"]["git_commit"], "branch": "devLonghino"})
                    rid, creado = run.info.run_id, True
                    timestamp = int(time.time()*1000)
                    params = [Param(k, valor_parametro(k, v)) for k, v in params.items()]
                    mets = [Metric(k, float(v), timestamp, 0) for k, v in r["metricas"].items()]
                    for i in range(0, max(len(params), len(mets)), 100):
                        cliente.log_batch(rid, params=params[i:i+100], metrics=mets[i:i+100])
                    cliente.log_artifacts(rid, str(carpeta))
                    verificar_remoto(cliente, rid, r, {p.key: p.value for p in params}, temporal / "descarga", manifiesto=manifiesto)
                    cliente.set_terminated(rid, "FINISHED")
                    assert cliente.get_run(rid).info.status == "FINISHED"
                ids[r["nombre"]] = rid
                filas.append({"Run": nombre, "Estado": "Reutilizado y verificado" if previo else "Registrado y verificado",
                              "Run ID": rid, "Enlace": f"{TRACKING_URI}/#/experiments/{eid}/runs/{rid}"})
        except ConflictoRegistro as exc:
            filas.append({"Run": r["nombre"], "Estado": f"Conflicto: {exc}", "Run ID": rid or "", "Enlace": ""})
        except Exception as exc:
            if creado:
                cliente.set_terminated(rid, "FAILED")
            # No imprimir excepciones HTTP que pudieran incluir datos de autenticación.
            filas.append({"Run": r["nombre"], "Estado": f"Error de registro/verificación ({type(exc).__name__})",
                          "Run ID": rid or "", "Enlace": ""})
            progreso(f"{r['nombre']}: error {type(exc).__name__}", flush=True)
    for relativa, huella in plan["contexto"]["huellas"].items():
        if ":contenido_cientifico" not in relativa:
            assert sha((raiz / relativa).read_bytes()) == huella, f"Se modificó la entrada {relativa}"
    return pd.DataFrame(filas)


def inventario_artefactos(plan, destino_temporal):
    """Prepara y valida cada paquete local para revisar el contenido a publicar.

    No conecta a DagsHub ni transmite información. Las referencias provisionales
    se sustituyen por IDs verificados exclusivamente durante el registro real.
    """
    carpeta = Path(destino_temporal).resolve()
    if carpeta.is_relative_to(plan["raiz"]):
        raise ValueError("El inventario requiere un temporal fuera del repositorio.")
    ids = {nombre:'ID_REMOTO_PENDIENTE_'+nombre for nombre in plan.get('referencias_esperadas',{})}
    filas = []
    for r in plan["runs"]:
        salida = carpeta / r["nombre"]
        salida.mkdir(parents=True)
        params, manifiesto = _exportar(r, plan, salida, ids)
        for ruta, huella in manifiesto.items():
            assert sha((salida / ruta).read_bytes()) == huella
        filas.append({"Run": r["nombre"], "Artefactos": len(manifiesto)+1,
                      "Tamaño (MB)": sum(p.stat().st_size for p in salida.rglob("*") if p.is_file()) / 1e6,
                      "Métricas": len(r["metricas"]), "Modelo": "modelo" in r,
                      "Estado": "Verificado localmente; pendiente de publicación"})
        ids[r["nombre"]] = "ID_REMOTO_PENDIENTE_" + r["nombre"]
    return pd.DataFrame(filas)


def predecir_paquete(datos, paquete, transformadores):
    """Predicción RUB desde filas crudas, sin fit ni estimación de parámetros."""
    from bss_features_usadas import entrada_codificador
    r = transformadores["registro"]
    configuracion = {"registro": r, "imputacion": {k: v["valor"] for k, v in r["imputacion"].items()},
                     "binarias": {k: [(v["categoria"], v["valor"]) for v in vals] for k, vals in r["binarias"].items()}}
    datos = datos.copy()
    if "price_rub" not in datos:
        datos["price_rub"] = 1.0  # Objetivo ficticio descartado antes de codificar; no es predictor.
    X, _ = entrada_codificador(datos, configuracion)
    assert segmentos(X.to_center_km, paquete["contrato"]["limites_segmentacion"]).eq(paquete["contrato"]["segmento"]).all()
    matriz = transformadores["codificador"].transform(X[r["entrada_codificador"]])
    nombres = transformadores["codificador"].get_feature_names_out()
    X = pd.DataFrame(matriz, columns=nombres, index=X.index)
    pred = inv_boxcox(paquete["modelo"].predict(diseno(X, paquete["contrato"]["columnas_diseno"])),
                     paquete["contrato"]["boxcox_lambda"])
    if not (np.isfinite(pred) & (pred > 0)).all():
        raise ValueError("Predicción fuera del dominio inverso Box-Cox; no se recorta.")
    return pred


COLORES_SEGMENTOS = dict(zip(SEGMENTOS, ['#1f77b4','#2ca02c','#ff7f0e','#d62728']))
VERSION_DIAGNOSTICO_TEST = 'residuos-test-scale-qq-v3'


def diagnosticar_test_bss(modelos, test, parametros, evaluaciones, predicciones=None):
    """Errores fuera de muestra; escala residual aprendida solo en train.

    LOWESS describe los datos, no ajusta el modelo OLS ni cambia su selección.
    No se centran los residuos de test ni se studentizan con leverage de test.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess
    if set(modelos) != set(SEGMENTOS):
        raise ValueError('Se requieren los cuatro modelos BSS definitivos.')
    etiquetas = segmentos(test.to_center_km, parametros['cuartiles_distancia_km'])
    diagnosticos, tablas = {}, {}
    for s in SEGMENTOS:
        modelo=modelos[s]; datos=test.loc[etiquetas.eq(s)].copy()
        lam=float(parametros['lambda_por_segmento'][s])
        assert datos.index.is_unique and len(datos)>0
        M=diseno(datos, modelo.model.exog_names)
        pred_bc=np.asarray(modelo.predict(M),dtype=float)
        assert np.isfinite(pred_bc).all()
        if lam != 0:
            assert (1+lam*pred_bc>0).all(), 'Inversa Box-Cox fuera de dominio; no se recorta.'
        pred_rub=inv_boxcox(pred_bc,lam)
        assert np.isfinite(pred_rub).all() and (pred_rub>0).all()
        if predicciones is not None:
            previo=predicciones[s]
            pd.testing.assert_index_equal(previo.index,datos.index)
            np.testing.assert_array_equal(pred_bc,previo.pred_boxcox.to_numpy())
            np.testing.assert_array_equal(pred_rub,previo.pred_price_rub.to_numpy())
        met=errores(datos.price_rub,pred_rub)
        ref=evaluaciones[s]['test_RUB']
        for nombre,clave in [('r2_rub','R2'),('mae_rub','MAE'),('rmse_rub','RMSE')]:
            np.testing.assert_allclose(met[nombre],ref[clave],rtol=1e-11,atol=1e-8)
        sigma=float(np.sqrt(modelo.mse_resid))
        assert np.isfinite(sigma) and sigma>0 and modelo.df_resid>0
        np.testing.assert_allclose(sigma**2,np.sum(np.asarray(modelo.resid)**2)/modelo.df_resid,rtol=1e-12)
        residuo_rub=datos.price_rub.to_numpy()-pred_rub
        real_bc=boxcox(datos.price_rub.to_numpy(dtype=float),lmbda=lam)
        residuo_bc=real_bc-pred_bc
        z=residuo_bc/sigma
        d=pd.DataFrame({'price_rub':datos.price_rub,'pred_price_rub':pred_rub,
            'residuo_rub':residuo_rub,'price_boxcox':real_bc,'pred_boxcox':pred_bc,
            'residuo_boxcox':residuo_bc,'residuo_escalado_train':z,'scale_location':np.sqrt(np.abs(z))},index=datos.index)
        assert len(d)==len(datos) and np.isfinite(d.to_numpy()).all()
        curvas={}
        for clave,x,y in [('rub',d.pred_price_rub,d.residuo_rub),('scale',d.pred_boxcox,d.scale_location)]:
            curvas[clave]=lowess(y,x,frac=.35,it=3,delta=0,missing='raise',return_sorted=True)
            assert len(curvas[clave])==len(d) and np.isfinite(curvas[clave]).all()
        d.attrs.update(segmento=s,lambda_boxcox=lam,sigma_train_boxcox=sigma,n_train=int(modelo.nobs),
                       lowess_frac=.35,lowess_it=3,curvas=curvas,version=VERSION_DIAGNOSTICO_TEST)
        tablas[s]=pd.DataFrame({'Indicador':['Observaciones evaluadas','Media de residuos (RUB)',
            'Mediana de residuos (RUB)','Desviación estándar residual (RUB)','MAE test (RUB)',
            'RMSE test (RUB)','R² test (RUB)','Porcentaje de residuos positivos','Porcentaje de residuos negativos'],
            'Valor':[len(d),d.residuo_rub.mean(),d.residuo_rub.median(),d.residuo_rub.std(ddof=1),
                     ref['MAE'],ref['RMSE'],ref['R2'],100*d.residuo_rub.gt(0).mean(),100*d.residuo_rub.lt(0).mean()]})
        diagnosticos[s]=d
    indices=pd.Index(np.concatenate([d.index.to_numpy() for d in diagnosticos.values()]))
    assert len(indices)==len(test) and indices.is_unique and set(indices)==set(test.index)
    return diagnosticos,tablas


def _panel_diagnostico_test(ax,datos,tipo):
    from matplotlib.ticker import FuncFormatter,MaxNLocator,ScalarFormatter
    s=datos.attrs['segmento']; color=COLORES_SEGMENTOS[s]
    if tipo=='rub':
        x,y=datos.pred_price_rub,datos.residuo_rub
        ax.axhline(0,color='#555555',linestyle='--',linewidth=1)
        ax.set(xlabel='Precio predicho por BSS (RUB)',ylabel='Residuo: real − predicho (RUB)',
               title=f'{s} — Residuals vs Fitted | test n={len(datos):,}')
        for eje in [ax.xaxis,ax.yaxis]:
            eje.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:,.0f}'.replace(',','.')))
    elif tipo=='scale':
        x,y=datos.pred_boxcox,datos.scale_location
        ax.set(xlabel=f'Valor ajustado (Box-Cox; λ={datos.attrs["lambda_boxcox"]:.4f})',
               ylabel='√|residuo Box-Cox / σ de train|',
               title=f'{s} — Scale–Location | escala residual de train')
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    elif tipo=='qq':
        from scipy.stats import norm
        # Una posición por observación; sin centrar ni reescalar con test.
        y=np.sort(datos.residuo_escalado_train.to_numpy())
        x=norm.ppf((np.arange(len(y))+.5)/len(y))
        ax.scatter(x,y,s=12,alpha=.30,color=color,edgecolors='none',rasterized=True)
        inferior=min(x.min(),y.min());superior=max(x.max(),y.max())
        margen=.05*(superior-inferior)
        limites=(inferior-margen,superior+margen)
        ax.plot(limites,limites,color='black',linestyle='--',linewidth=1.5,label='Normal estándar: y = x')
        ax.set(xlabel='Cuantiles teóricos N(0, 1)',ylabel='Cuantiles de e(test, Box-Cox) / σ de train',
               title=f'{s} — Normal Q-Q | test n={len(datos):,}',xlim=limites,ylim=limites)
        ax.set_aspect('equal',adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc='best',fontsize=8)
        return
    else:
        raise ValueError('Tipo de diagnóstico desconocido.')
    ax.scatter(x,y,s=12,alpha=.30,color=color,edgecolors='none',rasterized=True)
    curva=datos.attrs['curvas'][tipo]
    ax.plot(curva[:,0],curva[:,1],color='black',linewidth=2,label='LOWESS (frac=0,35)')
    ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc='best',fontsize=8)


def figura_residuos_test(diagnosticos, segmentos_mostrar=None, tipo=None):
    """Scale-Location y Q-Q: figura 4×2 y paneles individuales remotos."""
    import matplotlib.pyplot as plt
    nombres=SEGMENTOS if segmentos_mostrar is None else list(segmentos_mostrar)
    tipos=['scale','qq'] if tipo is None else [tipo]
    with plt.style.context('ggplot'), plt.rc_context({'font.family':'DejaVu Sans','font.size':10,
            'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
            'figure.dpi':100,'savefig.dpi':120,'figure.autolayout':False}):
        fig,axes=plt.subplots(len(nombres),len(tipos),figsize=(9*len(tipos),4.5*len(nombres)),squeeze=False)
        for i,s in enumerate(nombres):
            for j,t in enumerate(tipos):_panel_diagnostico_test(axes[i,j],diagnosticos[s],t)
        fig.tight_layout(pad=2.0,h_pad=3.0,w_pad=2.0)
    return fig


def _archivos_diagnosticos_test(diagnosticos,tablas):
    """Exporta los dos diagnósticos visibles sin alterar artefactos históricos."""
    import matplotlib.pyplot as plt
    archivos={s:{} for s in SEGMENTOS}
    for s in SEGMENTOS:
        for tipo,nombre in [('scale','scale_location_test.png'),('qq','normal_qq_test.png')]:
            fig=figura_residuos_test(diagnosticos,[s],tipo)
            b=io.BytesIO();fig.savefig(b,format='png',dpi=120);plt.close(fig)
            archivos[s][nombre]=b.getvalue()
    fig=figura_residuos_test(diagnosticos);b=io.BytesIO();fig.savefig(b,format='png',dpi=120);plt.close(fig)
    archivos[SEGMENTOS[0]]['comparativa_test_4x2.png']=b.getvalue()
    return archivos


def registrar_diagnosticos_test(raiz,modelos,diagnosticos,tablas,cliente=None,progreso=print):
    """Anexa artefactos a runs terminados; no crea runs ni cambia sus métricas.

    Reutiliza el registro/verificación de Fase 1. Cada archivo existente se
    descarga y compara antes de escribir; solo se suben nombres ausentes.
    Una colisión de contenido detiene el proceso antes de la primera subida.
    """
    raiz=Path(raiz).resolve()
    if str(raiz) not in sys.path:sys.path.insert(0,str(raiz))
    from Ventas.Fase1_PropUsadas.scripts.almacenamiento_mlflow import registrar_archivos
    cliente=conectar(raiz) if cliente is None else cliente
    exp=cliente.get_experiment_by_name(EXPERIMENTO)
    if exp is None:raise ConflictoRegistro('Debe existir Fase2_PropUsadas.')
    archivos=_archivos_diagnosticos_test(diagnosticos,tablas)
    planes=[]
    with TemporaryDirectory(prefix='moscow_residuos_test_') as tmp:
        temporal=Path(tmp)
        for s in SEGMENTOS:
            nombre='F2_08_BSS_GVIF_'+s.title().replace(' ','_')
            candidatos=_buscar_runs(cliente,exp.experiment_id,f"tags.fase2_base = '{nombre}'")
            candidatos=[r for r in candidatos if r.info.status=='FINISHED']
            if len(candidatos)!=1:raise ConflictoRegistro(f'Run GVIF ambiguo o ausente: {nombre}')
            run=candidatos[0];rid=run.info.run_id
            descarga=temporal/(s.replace(' ','_')+'_descarga');descarga.mkdir()
            paquete=joblib.load(cliente.download_artifacts(rid,'modelo_segmento.joblib',str(descarga)))
            np.testing.assert_array_equal(paquete['modelo'].params,modelos[s].params)
            assert paquete['contrato']['columnas_diseno']==modelos[s].model.exog_names
            assert paquete['contrato']['boxcox_lambda']==diagnosticos[s].attrs['lambda_boxcox']
            assert int(run.data.params['n_test'])==len(diagnosticos[s])
            met=errores(diagnosticos[s].price_rub,diagnosticos[s].pred_price_rub)
            for k,v in met.items():np.testing.assert_allclose(run.data.metrics['test_'+k],v,rtol=1e-10,atol=1e-8)
            huella=sha(json_bytes({'version':VERSION_DIAGNOSTICO_TEST,'modelo':np.asarray(modelos[s].params),
                'lambda':diagnosticos[s].attrs['lambda_boxcox'],'sigma_train':diagnosticos[s].attrs['sigma_train_boxcox'],
                'datos':sha(pd.util.hash_pandas_object(diagnosticos[s],index=True).to_numpy().tobytes()),
                'lowess_frac':.35,'lowess_it':3,'matplotlib':version('matplotlib')}))[:16]
            destino=f'diagnosticos_test/{VERSION_DIAGNOSTICO_TEST}_{huella}'
            try:
                existentes={p.path:p for p in cliente.list_artifacts(rid,destino)}
            except Exception as exc:
                if getattr(exc,'error_code',None)!='RESOURCE_DOES_NOT_EXIST':raise
                existentes={}
            faltantes={}
            for fichero,contenido in archivos[s].items():
                ruta=destino+'/'+fichero
                if ruta in existentes:
                    recibido=Path(cliente.download_artifacts(rid,ruta,str(descarga)))
                    if sha(recibido.read_bytes())!=sha(contenido):
                        raise ConflictoRegistro(f'Artefacto existente con otro contenido: {ruta}')
                else:faltantes[fichero]=contenido
            planes.append((s,run,destino,faltantes))
        # Todos los runs, contratos y nombres se validan antes de publicar.
        filas=[]
        for s,antes,destino,faltantes in planes:
            rid=antes.info.run_id
            progreso(f'{s}: {len(faltantes)} artefactos nuevos de residuos de test',flush=True)
            if faltantes:
                salida=temporal/s.replace(' ','_');salida.mkdir()
                for nombre,contenido in faltantes.items():(salida/nombre).write_bytes(contenido)
                registrar_archivos(cliente,rid,salida,destino)
            despues=cliente.get_run(rid)
            assert despues.info.status==antes.info.status=='FINISHED'
            assert despues.data.params==antes.data.params and despues.data.metrics==antes.data.metrics
            assert despues.data.tags==antes.data.tags
            filas.append({'Segmento':s,'Run ID':rid,'Estado':'Añadidos y verificados' if faltantes else 'Reutilizados y verificados',
                          'Artefactos nuevos':len(faltantes),'Carpeta remota':destino})
    return pd.DataFrame(filas)
