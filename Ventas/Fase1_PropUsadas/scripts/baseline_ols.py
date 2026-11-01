"""Baseline OLS completa: solo Train transformado; sin selección ni nuevo preprocesamiento."""
import copy
import shutil
from tempfile import TemporaryDirectory
import hashlib
import json
import os
import platform
import warnings
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults
from scipy import linalg
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
from threadpoolctl import threadpool_limits
from .experimentos_ols import evaluar_regresion, tabla_markdown
from .diagnosticos_residuales import ARCHIVOS, generar_diagnosticos_residuales, registrar_diagnosticos_residuales

from .almacenamiento_mlflow import validar_temporal
from .contrato_datasets import CONTRATO, ruta_dataset

EXPERIMENTO = "Modelos Propiedades_Usadas"
RUN = "Baseline_OLS_01"
TRACKING_URI = "https://dagshub.com/matiasgastonlonghino/MOSCOW_IDA.mlflow"

# Una sola definición mantiene sincronizados diseño, presentación y registro.
MODELOS_PRESENTACION = (
    ("Baseline_OLS_01", ()),
    ("OLS_02_Log_Target", ()),
    ("OLS_03_Log_Sin_to_center_km", ("to_center_km",)),
    ("OLS_04_Log_Sin_living_area", ("to_center_km", "living_area")),
    ("OLS_05_Log_Sin_rooms", ("to_center_km", "living_area", "rooms")),
)


def evolucion_variables(X):
    filas = [{"Experimento": f"{i} — {nombre}",
              "Objetivo": "price_rub" if i == 1 else "ln(price_rub)",
              "Predictoras": len(X.columns.difference(excluidas)),
              "Variables eliminadas acumuladas": ", ".join(excluidas) or "Ninguna"}
             for i, (nombre, excluidas) in enumerate(MODELOS_PRESENTACION, 1)]
    return tabla_markdown(pd.DataFrame(filas), decimales=0)


def ajustar_presentacion(X, y, contexto, numero, anterior=None):
    """Ajusta y diagnostica Train; ninguna conexión ni figura durante el ajuste."""
    if numero not in range(1, 6):
        raise ValueError("Número de experimento fuera de 1–5.")
    nombre, excluidas = MODELOS_PRESENTACION[numero-1]
    reducido = X.drop(columns=list(excluidas))
    if numero > 1:
        assert anterior is not None and anterior["numero"] == numero-1
        assert anterior["contexto"]["dataset_sha256"] == contexto["dataset_sha256"]
        assert reducido.index.equals(anterior["diseno"].index)
        esperadas = anterior["contexto"]["predictoras"].copy()
        if numero >= 3:
            esperadas.remove(excluidas[-1])
        assert list(reducido) == esperadas
    objetivo = y if numero == 1 else np.log(y).rename("ln_price_rub")
    modelo, diseno = ajustar(reducido, objetivo)
    metricas, invalidas, vif, notas = diagnosticar(
        modelo, diseno, objetivo, y_original=None if numero == 1 else y,
        resumir_vif=numero >= 3)
    ctx = copy.deepcopy(contexto)
    ctx.update(predictoras=list(reducido), n_predictoras=reducido.shape[1])
    if numero > 1:
        ctx.update(transformacion_objetivo="ln", unidad="ln(RUB)",
                   transformacion_inversa="exp", correccion_sesgo="ninguna")
    if excluidas:
        ctx["predictoras_excluidas"] = list(excluidas)
    relacion = float(X.living_area.corr(X.total_area)) if numero == 4 else None
    return dict(numero=numero, nombre=nombre, modelo=modelo, diseno=diseno, y=objetivo,
                precio=y, contexto=ctx, metricas=metricas, invalidas=invalidas,
                vif=vif, notas=notas, correlacion_superficies=relacion, anterior=anterior)


def _clave_presentacion(resultado, clave):
    if clave in ("mae", "rmse"):
        return "train_"+clave+("_original" if resultado["numero"] > 1 else "")
    if clave in ("vif_max", "condition_number"):
        return "train_"+clave
    return "train_"+clave+("_log" if resultado["numero"] > 1 else "")


def _valor_presentacion(resultado, clave):
    nombre = _clave_presentacion(resultado, clave)
    if nombre in resultado["metricas"]:
        return resultado["metricas"][nombre]
    razon = resultado["invalidas"].get(nombre, "No calculado")
    return "∞" if razon.startswith("Infinito") else "No calculado"


def _formato_presentacion(valor):
    if isinstance(valor, str):
        return valor
    if valor == 0:
        return "0"
    if abs(valor) < .001:
        return f"{valor:.3e}"
    return f"{valor:,.6f}" if abs(valor) < 1 else f"{valor:,.3f}"


def tabla_presentacion(resultado):
    """Delta solo para estadísticas con igual respuesta y escala."""
    previo = resultado["anterior"]
    cruzada = resultado["numero"] == 2
    filas = []
    for clave, etiqueta in [("r2", "R²"), ("r2_adj", "R² ajustado"),
                            ("mae", "MAE (RUB)"), ("rmse", "RMSE (RUB)"),
                            ("aic", "AIC"), ("bic", "BIC"),
                            ("vif_max", "VIF máximo"), ("condition_number", "Número de condición"),
                            ("bp_lm_pvalue", "Breusch–Pagan (p)"),
                            ("jarque_bera_pvalue", "Jarque–Bera (p)")]:
        comparable = not (cruzada and clave not in ("mae", "rmse", "vif_max", "condition_number"))
        nuevo = _valor_presentacion(resultado, clave)
        fila = {"Métrica": etiqueta}
        if clave in ("r2", "r2_adj", "aic", "bic", "bp_lm_pvalue", "jarque_bera_pvalue"):
            fila["Métrica"] += " [RUB → ln(RUB)]" if cruzada else (
                " [ln(RUB)]" if resultado["numero"] > 1 else " [RUB]")
        if previo:
            anterior = _valor_presentacion(previo, clave)
            fila[previo["nombre"]] = _formato_presentacion(anterior)
        fila[resultado["nombre"]] = _formato_presentacion(nuevo)
        if previo:
            fila["Δ nuevo − anterior"] = (_formato_presentacion(nuevo-anterior)
                if comparable and not isinstance(nuevo, str) and not isinstance(anterior, str)
                else "No comparable" if not comparable else "—")
        filas.append(fila)
    texto = tabla_markdown(pd.DataFrame(filas), decimales=6)
    if cruzada:
        texto += "\n\nR², R² ajustado, AIC, BIC y pruebas de residuos describen respuestas en escalas distintas; no se comparan entre estos dos ajustes."
    return texto


def grafico_presentacion(resultado):
    """Una sola figura principal para 1/2. Las ablaciones no generan figuras."""
    from matplotlib.figure import Figure
    if resultado["numero"] not in (1, 2):
        raise ValueError("Los experimentos 3–5 no tienen visualizaciones.")
    fig = Figure(figsize=(8, 4.3), layout="constrained")
    ax = fig.subplots()
    if resultado["numero"] == 1:
        precio = np.asarray(resultado["precio"])
        pred = np.asarray(resultado["modelo"].predict(resultado["diseno"]))
        ax.scatter(precio/1e6, pred/1e6, s=4, alpha=.22, rasterized=True)
        limites = (min(precio.min(), pred.min())/1e6, max(precio.max(), pred.max())/1e6)
        ax.plot(limites, limites, "--", color="black", linewidth=1, label="Predicción = precio")
        ax.set(xlabel="Precio observado (millones de RUB)", ylabel="Precio ajustado (millones de RUB)",
               title="Baseline_OLS_01 · ajuste sobre Train")
    else:
        for r in (resultado["anterior"], resultado):
            pred = np.asarray(r["modelo"].predict(r["diseno"]))
            if r["numero"] > 1:
                pred = np.exp(pred)
            error = np.sort(np.abs(np.asarray(r["precio"])-pred))
            ax.plot(error, np.arange(1, len(error)+1)/len(error), label=r["nombre"])
        ax.set(xscale="symlog", xlabel="Error absoluto (RUB; eje symlog)",
               ylabel="Proporción acumulada de observaciones",
               title="Errores de Train en la misma escala · exp sin corrección de sesgo")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    return fig


def interpretacion_presentacion(resultado):
    m = resultado["metricas"]
    if resultado["numero"] == 1:
        return (f"El ajuste en Train alcanza R²={m['train_r2']:.4f}, con MAE={m['train_mae']:,.0f} RUB y RMSE={m['train_rmse']:,.0f} RUB.  \n"
                f"VIF máximo={m['train_vif_max']:.2f} y número de condición={m['train_condition_number']:.1f}: hay colinealidad y sensibilidad numérica.  \n"
                f"Breusch–Pagan (p={m['train_bp_lm_pvalue']:.2e}) y Jarque–Bera (p={m['train_jarque_bera_pvalue']:.2e}) señalan límites de homocedasticidad y normalidad.  \n"
                "El próximo experimento evalúa ln(price_rub). Estas métricas de Train no establecen capacidad predictiva fuera de muestra.")
    p = resultado["anterior"]["metricas"]
    anterior = resultado["anterior"]
    d_mae = 100*(m["train_mae_original"]/_valor_presentacion(anterior, "mae")-1)
    d_rmse = 100*(m["train_rmse_original"]/_valor_presentacion(anterior, "rmse")-1)
    errores = f"En Train, MAE cambia {d_mae:+.5f} % y RMSE {d_rmse:+.5f} % en RUB frente al modelo anterior."
    if resultado["numero"] == 2:
        return (errores+"  \nLa transformación del objetivo cambia el peso relativo de los errores; VIF y condicionamiento permanecen iguales porque X no cambia.  \n"
                f"Persisten heterocedasticidad y desviaciones de normalidad (BP p={m['train_bp_lm_pvalue_log']:.2e}; JB p={m['train_jarque_bera_pvalue_log']:.2e}).  \n"
                "exp(E[ln(precio)|X]) no equivale a E[precio|X]; la inversa sin corrección puede subestimar el precio esperado.  \n"
                "Se examina a continuación to_center_km para reducir colinealidad. AIC, BIC y R² nativos no se comparan con la baseline en RUB.")
    d_aic = m["train_aic_log"]-p["train_aic_log"]
    d_bic = m["train_bic_log"]-p["train_bic_log"]
    partes = [errores,
              f"VIF máximo: {p['train_vif_max']:.2f} → {m['train_vif_max']:.2f}; número de condición: {p['train_condition_number']:.1f} → {m['train_condition_number']:.1f}.",
              f"ΔR² ajustado log={m['train_r2_adj_log']-p['train_r2_adj_log']:+.7f}; ΔAIC={d_aic:+.3f}; ΔBIC={d_bic:+.3f} (menor AIC/BIC es preferible)."]
    if resultado["numero"] == 3:
        partes.append("Quitar to_center_km reduce la colinealidad, con deterioro del ajuste y del error de Train; la siguiente etapa examina redundancia entre superficies.")
    elif resultado["numero"] == 4:
        partes.append(f"living_area y total_area presentan correlación {resultado['correlacion_superficies']:.4f}; la exclusión reduce colinealidad, con cambios mínimos de error. Se evalúa luego rooms.")
    else:
        partes.append("Quitar rooms simplifica el diseño, pero empeora el ajuste y los errores de Train; la reducción adicional del VIF máximo y del número de condición es mínima.")
    partes.append("La comparación usa las mismas filas; no demuestra equivalencia ni superioridad fuera de muestra.")
    return "  \n".join(partes)


def resumen_presentacion(resultados):
    filas = []
    for r in resultados:
        log = r["numero"] > 1
        filas.append({"Experimento": r["nombre"], "Eliminadas (acumuladas)":
            ", ".join(r["contexto"].get("predictoras_excluidas", [])) or "Ninguna",
            "p": r["contexto"]["n_predictoras"], "Escala del ajuste": "ln(RUB)" if log else "RUB",
            **{etiqueta: _formato_presentacion(_valor_presentacion(r, clave)) for clave, etiqueta in
               [("r2_adj", "R² ajustado¹"), ("aic", "AIC¹"), ("bic", "BIC¹"),
                ("mae", "MAE (RUB)"), ("rmse", "RMSE (RUB)"),
                ("vif_max", "VIF máx."), ("condition_number", "Condición")]}})
    return tabla_markdown(pd.DataFrame(filas), decimales=6)+(
        "\n\n¹ Comparar R² ajustado, AIC y BIC únicamente entre los experimentos 2–5, con la misma respuesta logarítmica. Todos los errores son de Train en RUB.")


def registrar_presentacion(resultado, sesion, cliente=None):
    """Reutiliza Runs históricos sin sobrescribir artefactos; exporta solo si falta el Run.

    El modelo se ajusta en la celda académica. Aquí se verifican el contrato, las
    métricas recalculadas y las predicciones del modelo recuperado de MLflow.
    Las métricas adicionales históricas (p. ej. HC3) permanecen en su Run original.
    """
    from .contrato_datasets import RAIZ, CARPETA_PROYECTO
    cliente = conectar(RAIZ) if cliente is None else cliente
    ctx = resultado["contexto"]
    anterior = resultado["anterior"]
    if anterior:
        clave = "baseline_run_id" if resultado["numero"] == 2 else "reference_run_id"
        ctx[clave] = anterior["registro"]["run_id"]
    exp = cliente.get_experiment_by_name(EXPERIMENTO)
    if exp is None or exp.lifecycle_stage != "active":
        raise ValueError("Debe existir el experimento activo de propiedades usadas.")
    candidatos = cliente.search_runs([exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{resultado['nombre']}'", max_results=100)
    propios = [r for r in candidatos if r.info.status == "FINISHED"
               and r.data.tags.get("dataset_sha256") == ctx["dataset_sha256"]]
    out = sesion.carpeta(resultado["nombre"])
    if propios:
        run = propios[0]
        rid = run.info.run_id
        def descargar(nombre):
            return Path(cliente.download_artifacts(rid, nombre, str(out)))
        contrato = json.loads(descargar("contexto.json").read_text(encoding="utf-8"))
        for clave in ("dataset_sha256", "predictoras", "transformacion_objetivo", "n_train"):
            assert contrato[clave] == ctx[clave], clave
        assert run.data.params["intercepto"].lower() == "true"
        assert run.data.params["n_predictors"] == str(ctx["n_predictoras"])
        assert run.data.params["train_fraction"] == "1.0"
        for clave in ("baseline_run_id", "reference_run_id"):
            if clave in ctx:
                assert run.data.params[clave] == ctx[clave]
        if resultado["numero"] > 1:
            assert run.data.params["target_transform"] == "ln(Target)"
            assert run.data.params["inverse_transform"] == "exp"
            assert run.data.params["bias_correction"] == "none"
        historicas = json.loads(descargar("metricas.json").read_text(encoding="utf-8"))
        for k, v in resultado["metricas"].items():
            np.testing.assert_allclose([run.data.metrics[k], historicas[k]], v, rtol=1e-10, atol=1e-12)
        inferencia = json.loads(descargar("inferencia.json").read_text(encoding="utf-8"))
        assert inferencia["columnas_diseno"] == list(resultado["diseno"])
        recuperado = OLSResults.load(descargar("modelo_statsmodels.pkl"))
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(recuperado.predict(resultado["diseno"]),
                resultado["modelo"].predict(resultado["diseno"]), rtol=1e-10, atol=1e-8)
        despues = cliente.get_run(rid)
        assert despues.data.metrics == run.data.metrics and despues.data.params == run.data.params
        assert despues.info.status == "FINISHED"
        registro = dict(run_id=rid, experiment_id=exp.experiment_id, run_name=resultado["nombre"],
            status="FINISHED", url=f"{TRACKING_URI}/#/experiments/{exp.experiment_id}/runs/{rid}")
    else:
        exportar(CARPETA_PROYECTO, resultado["modelo"], resultado["diseno"], resultado["y"], ctx,
                 resultado["metricas"], resultado["invalidas"], resultado["vif"], resultado["notas"],
                 nombre_run=resultado["nombre"], carpeta_salida=out,
                 generar_graficos=resultado["numero"] <= 2)
        (out/"comparacion_compacta.md").write_text(tabla_presentacion(resultado), encoding="utf-8")
        (out/"interpretacion.md").write_text(interpretacion_presentacion(resultado), encoding="utf-8")
        registro = registrar(cliente, out, ctx, resultado["metricas"], nombre_run=resultado["nombre"],
                             requiere_graficos=resultado["numero"] <= 2)
    assert sha256(CARPETA_PROYECTO/ctx["csv"]) == ctx["dataset_sha256"]
    resultado["registro"] = registro
    return cliente, f"**FINISHED** · [{resultado['nombre']} en DagsHub]({registro['url']})"

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def guardar_json(path, data):
    validar_temporal(Path(path).parent)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def cargar_train(base):
    base = Path(base)
    archivo = ruta_dataset("train_transformado.csv", base)
    contrato = CONTRATO
    huella = sha256(archivo)
    assert huella == contrato["train"]["sha256"], "El CSV difiere del manifiesto."
    datos = pd.read_csv(archivo)
    predictoras = contrato["columnas_X"]
    auxiliares = ["id", "date_posted_dt", "dataset_split"]
    assert datos.columns.is_unique and len(predictoras) == len(set(predictoras))
    assert set(datos) == set(predictoras + auxiliares + ["price_rub"])
    assert len(datos) == contrato["train"]["filas"]
    assert len(predictoras) == contrato["train"]["predictoras"]
    assert datos["dataset_split"].eq("train").all()
    X = datos[predictoras].astype(float)
    y = datos["price_rub"].astype(float)
    assert np.isfinite(X).all().all() and np.isfinite(y).all() and (y > 0).all()
    contexto = {"csv": "data/train_transformado.csv", "dataset_sha256": huella,
        "manifiesto_sha256": contrato["manifiesto_sha256"], "objetivo": "price_rub", "unidad": "RUB",
        "transformacion_objetivo": "ninguna", "predictoras": predictoras, "auxiliares_excluidas": auxiliares,
        "n_train": len(y), "n_predictoras": len(predictoras), "test_utilizado": False,
        "preprocesamiento": "CSV transformado existente; no se ajustan transformadores"}
    return X, y, contexto

def ajustar(X, y):
    diseno = sm.add_constant(X, has_constant="add")
    with threadpool_limits(limits=1):
        modelo = sm.OLS(y, diseno, missing="raise").fit()
    assert int(modelo.nobs) == len(y) and modelo.model.exog_names == list(diseno.columns)
    return modelo, diseno

def comparar_anidados(modelo_completo, modelo_reducido, diseno_completo, y_log, *,
                      variable, rss_completo_esperado, f_clasico_valido=False):
    """Aporte de una restricción; reutiliza el modelo completo sin volver a ajustarlo.

    HC3 es una comprobación de sensibilidad a heterocedasticidad, con inferencia
    asintótica que sigue suponiendo independencia entre observaciones.
    """
    from scipy.stats import chi2, f
    assert list(diseno_completo.columns) == modelo_completo.model.exog_names
    assert [c for c in diseno_completo if c != variable] == modelo_reducido.model.exog_names
    assert int(modelo_completo.nobs) == int(modelo_reducido.nobs) == len(y_log)
    np.testing.assert_allclose(modelo_reducido.model.endog, y_log, rtol=1e-12)
    ajustados=np.asarray(modelo_completo.predict(diseno_completo))
    residuos=np.asarray(y_log)-ajustados
    rss_completo=float(residuos@residuos)
    np.testing.assert_allclose(rss_completo,rss_completo_esperado,rtol=1e-10)
    df_num=float(modelo_reducido.df_resid-modelo_completo.df_resid)
    assert df_num == 1 and rss_completo > 0
    incremento=float(modelo_reducido.ssr)-rss_completo
    assert incremento >= -1e-10*rss_completo
    incremento=max(0.0,incremento)
    f_parcial=(incremento/df_num)/(rss_completo/modelo_completo.df_resid)
    X=diseno_completo.to_numpy(dtype=float)
    bread=np.asarray(modelo_completo.normalized_cov_params)
    with threadpool_limits(limits=1):
        leverage=np.einsum("ij,ij->i",X@bread,X)
        assert np.all(1-leverage > 0)
        scores=X*(residuos/(1-leverage))[:,None]
        cov_hc3=bread@(scores.T@scores)@bread
    j=list(diseno_completo).index(variable)
    coef=float(modelo_completo.params.iloc[j])
    se_hc3=float(np.sqrt(cov_hc3[j,j]))
    wald=(coef/se_hc3)**2
    metricas={"incremental_r2_removed_predictor_log":incremento/float(np.square(y_log-y_log.mean()).sum()),
        "partial_r2_removed_predictor_log":incremento/float(modelo_reducido.ssr),
        "partial_f_descriptive":float(f_parcial), "partial_f_df_num":df_num,
        "partial_f_df_den":float(modelo_completo.df_resid),
        "partial_f_classical_inference_valid":float(f_clasico_valido),
        "removed_predictor_coefficient_log":coef,
        "removed_predictor_t_nonrobust":float(modelo_completo.tvalues.iloc[j]),
        "removed_predictor_pvalue_nonrobust":float(modelo_completo.pvalues.iloc[j]),
        "removed_predictor_se_hc3":se_hc3, "removed_predictor_wald_hc3":float(wald),
        "removed_predictor_pvalue_hc3":float(chi2.sf(wald,1))}
    if f_clasico_valido:
        metricas["partial_f_pvalue"]=float(f.sf(f_parcial,df_num,modelo_completo.df_resid))
    notas={"variable":variable,
        "f_parcial":"Estadístico descriptivo calculado con RSS de modelos anidados; p-value clásico omitido si no se cumplen los supuestos.",
        "hc3":"Wald chi-cuadrado (1 gl) con covarianza HC3 del modelo completo ya registrado. No cambia estimaciones ni predicciones; supone independencia de filas y no corrige dependencia temporal o por grupos."}
    return metricas, notas

def preparar_ablacion(cliente, base, X, y_log, y_original, *, registro_referencia,
                       variable, nombre_run, carpeta_salida, analizar_importancia=True):
    """Recupera un OLS log registrado y prepara una única exclusión sobre X ya cargado."""
    out=validar_temporal(carpeta_salida)
    out.mkdir(exist_ok=True)
    ref=cliente.get_run(registro_referencia["run_id"])
    exp=cliente.get_experiment_by_name(EXPERIMENTO)
    assert exp is not None and exp.lifecycle_stage == "active" and ref.info.experiment_id == exp.experiment_id
    assert ref.info.status == "FINISHED" and ref.data.tags["mlflow.runName"] == registro_referencia["run_name"]
    assert ref.data.params["target_transform"] == "ln(Target)" and ref.data.params["bias_correction"] == "none"
    assert ref.data.params["inverse_transform"] == "exp" and ref.data.params["cov_type"] == "nonrobust"
    assert ref.data.params["solver"] == "pinv"
    carpeta=out/"referencia_modelo"
    carpeta.mkdir(exist_ok=True)
    nombres=["contexto.json"]
    if analizar_importancia:
        nombres += ["modelo_statsmodels.pkl","coeficientes.csv","vif.csv","inferencia.json",
            *["diagnosticos_residuales/"+n for n in ARCHIVOS]]
    archivos={n:Path(cliente.download_artifacts(ref.info.run_id,n,str(carpeta))) for n in nombres}
    contexto_ref=json.loads(archivos["contexto.json"].read_text(encoding="utf-8"))
    assert list(X.columns) == contexto_ref["predictoras"] and variable in X
    assert X.index.equals(y_log.index) and X.index.equals(y_original.index)
    assert len(X) == ref.data.metrics["train_n_obs"] and (y_original > 0).all()
    ruta_csv = Path(base)/"data"/Path(contexto_ref["csv"]).name
    if not ruta_csv.is_file():
        ruta_csv = Path(base)/contexto_ref["csv"]
    assert sha256(ruta_csv) == ref.data.tags["dataset_sha256"]
    np.testing.assert_allclose(y_log,np.log(y_original),rtol=1e-12)
    vif_ref=pd.read_csv(archivos["vif.csv"]).set_index("variable")["vif"] if analizar_importancia else None
    resumen_vif=({"train_vif_mean":float(vif_ref.mean()),"train_vif_gt5_count":int(vif_ref.gt(5).sum()),
        "train_vif_gt10_count":int(vif_ref.gt(10).sum())} if analizar_importancia else {})
    derivadas={k:v for k,v in resumen_vif.items() if k not in ref.data.metrics}
    metricas_ref={**ref.data.metrics,**derivadas}
    guardar_json(out/"referencia_mlflow.json",{"run_id":ref.info.run_id,"run_name":registro_referencia["run_name"],
        "experiment_id":ref.info.experiment_id,"status":ref.info.status,"metrics":ref.data.metrics,
        "params":ref.data.params,"resumen_vif_derivado_del_artefacto":derivadas})
    nuevo=X.drop(columns=[variable])
    assert set(X)-set(nuevo) == {variable} and nuevo.index.equals(X.index)
    contexto={k:v for k,v in contexto_ref.items() if k != "baseline_run_id"}
    contexto["csv"] = ruta_csv.relative_to(Path(base)).as_posix()
    contexto.update(predictoras=list(nuevo.columns),n_predictoras=nuevo.shape[1],
        predictoras_excluidas=[*contexto_ref.get("predictoras_excluidas",[]),variable],
        reference_run_id=ref.info.run_id,nombre_run=nombre_run)
    return {"base":Path(base),"X_referencia":X,"X":nuevo,"y_log":y_log,"y_original":y_original,
        "variable":variable,"nombre_run":nombre_run,"salida":out,"referencia":ref,"archivos":archivos,
        "contexto":contexto,"metricas_referencia":metricas_ref,"vif_referencia":vif_ref,
        "coeficientes_referencia":pd.read_csv(archivos["coeficientes.csv"],index_col="variable") if analizar_importancia else None,
        "modelo_referencia":OLSResults.load(archivos["modelo_statsmodels.pkl"]) if analizar_importancia else None,
        "analizar_importancia":analizar_importancia}

def evaluar_ablacion(datos, modelo, diseno, *, relacionadas=()):
    """Flujo compartido de ablaciones: métricas, aporte, coeficientes, artefactos y tabla."""
    X,variable,out=datos["X_referencia"],datos["variable"],datos["salida"]
    m,invalidas,vif,notas=diagnosticar(modelo,diseno,datos["y_log"],y_original=datos["y_original"],resumir_vif=True)
    analizar_importancia=datos.get("analizar_importancia",True)
    importancia,nota,cambios,relacion={},{},pd.DataFrame(),[]
    if analizar_importancia:
        importancia,nota=comparar_anidados(datos["modelo_referencia"],modelo,sm.add_constant(X,has_constant="add"),
            datos["y_log"],variable=variable,rss_completo_esperado=datos["metricas_referencia"]["train_rss_log"],f_clasico_valido=False)
        coef=datos["coeficientes_referencia"]
        np.testing.assert_allclose([importancia["removed_predictor_coefficient_log"],importancia["removed_predictor_t_nonrobust"],
            importancia["removed_predictor_pvalue_nonrobust"]],coef.loc[variable,["coeficiente","t","p_value"]].to_numpy(),rtol=1e-10,atol=0)
        cambios=pd.DataFrame({"coef_referencia":coef["coeficiente"],"p_referencia_nonrobust":coef["p_value"],
            "coef_nuevo":modelo.params,"p_nuevo_nonrobust":modelo.pvalues}).reindex(modelo.params.index)
        cambios["delta_coef"]=cambios["coef_nuevo"]-cambios["coef_referencia"]
        cambios["significativo_referencia"]=cambios["p_referencia_nonrobust"].le(.05)
        cambios["significativo_nuevo"]=cambios["p_nuevo_nonrobust"].le(.05)
        cambios["cambia"]=cambios["significativo_referencia"] != cambios["significativo_nuevo"]
        importancia.update(n_significance_changes=int(cambios["cambia"].sum()),
            n_significant_to_nonsignificant=int((cambios["significativo_referencia"] & ~cambios["significativo_nuevo"]).sum()),
            n_nonsignificant_to_significant=int((~cambios["significativo_referencia"] & cambios["significativo_nuevo"]).sum()),
            removed_predictor_vif_reference=float(datos["vif_referencia"].loc[variable]))
        relacion=[]
        for nombre in relacionadas:
            assert nombre in datos["X"]
            valores={"correlation_train":float(X[variable].corr(X[nombre])),
                "vif_reference":float(datos["vif_referencia"].loc[nombre]),
                "vif_new":float(vif.set_index("variable").loc[nombre,"vif"]),
                "coefficient_reference_log":float(coef.loc[nombre,"coeficiente"]),"coefficient_new_log":float(modelo.params[nombre]),
                "pvalue_reference_nonrobust":float(coef.loc[nombre,"p_value"]),"pvalue_new_nonrobust":float(modelo.pvalues[nombre])}
            importancia.update({f"related_{nombre}_{k}":v for k,v in valores.items()})
            relacion.append({"eliminada":variable,"predictora_relacionada":nombre,**valores})
    m.update(importancia)
    if analizar_importancia:
        notas["comparacion_anidada"]=nota
    exportar(datos["base"],modelo,diseno,datos["y_log"],datos["contexto"],m,invalidas,vif,notas,
        nombre_run=datos["nombre_run"],carpeta_salida=out)
    if analizar_importancia:
        cambios.to_csv(out/"cambios_significancia.csv",index_label="variable")
        guardar_json(out/f"importancia_{variable}.json",{"metricas":importancia,"notas":nota})
    if relacion:
        pd.DataFrame(relacion).to_csv(out/"relacion_superficies.csv",index=False)
    # Los indicadores de aporte de una ablación previa se refieren a otra restricción:
    # se comparan solo las métricas train_* de los ajustes, no esos contrastes.
    ref=datos["referencia"]
    comparables={k:v for k,v in datos["metricas_referencia"].items() if k.startswith("train_")}
    tabla=tabla_resultados(m,invalidas,notas,nombre_run=datos["nombre_run"],referencia=comparables,
        nombre_referencia=ref.data.tags["mlflow.runName"])
    if analizar_importancia:
        texto=(f"# {datos['nombre_run']} frente a {ref.data.tags['mlflow.runName']} — Train\n\n"
            +f"Referencia MLflow: {ref.info.run_id}. Mismas filas, ln(RUB), exp sin corrección. Δ = nuevo − referencia.\n\n"
            +tabla+f"\n\n## Aporte de {variable}\n\n"
            +tabla_markdown(pd.DataFrame([{"Diagnóstico":k,"Valor":format(v,".12g")} for k,v in importancia.items()]))
            +"\n\nF parcial solo descriptivo: p-value clásico omitido por supuestos no verificados; HC3 supone independencia de filas. Cambios de significancia nominales (nonrobust). Correlaciones, cuando se incluyen, corresponden a las columnas transformadas de Train.\n")
    else:
        texto=tabla.replace("| Diferencia |", "| Δ |")
    (out/"comparacion_referencia.md").write_text(texto,encoding="utf-8")
    return {"metricas":m,"invalidas":invalidas,"vif":vif,"notas":notas,"importancia":importancia,
        "cambios":cambios,"comparacion_md":texto,"relacion":pd.DataFrame(relacion)}

def diagnosticar(modelo, diseno, y, *, y_original=None, resumir_vif=False):
    """Diagnósticos en la escala ajustada; si y=ln(precio), predicción RUB sin corrección."""
    rango = int(modelo.model.rank)
    # Reutiliza las métricas predictivas de referencia, sin su entrenamiento temporal/logarítmico.
    pred = modelo.predict(diseno)
    if y_original is not None:
        assert y.index.equals(y_original.index) and (y_original > 0).all()
        np.testing.assert_allclose(y, np.log(y_original), rtol=1e-12)
    heredadas = evaluar_regresion(y if y_original is None else y_original,
        pred if y_original is None else np.exp(pred),
        es_train=True, n_predictoras_efectivas=rango-1)
    m = {"train_r2": heredadas["r2"], "train_r2_adj": heredadas["r2_ajustado"],
        "train_mae": heredadas["mae_rub"], "train_rmse": heredadas["rmse_rub"],
        "train_mape": heredadas["mape_pct"], "train_rss": heredadas["rss_rub2"],
        "train_mse": heredadas["rss_rub2"]/len(y), "train_aic": float(modelo.aic),
        "train_bic": float(modelo.bic), "train_f_statistic": float(modelo.fvalue),
        "train_f_pvalue": float(modelo.f_pvalue), "train_n_obs": len(y),
        "train_n_predictors": diseno.shape[1]-1, "train_n_parameters_effective": rango,
        "train_n_coefficients": len(modelo.params), "train_df_model": float(modelo.df_model),
        "train_df_resid": float(modelo.df_resid), "train_log_likelihood": float(modelo.llf),
        "train_durbin_watson": float(durbin_watson(modelo.resid)),
        "train_n_coefficients_p_gt_005": int((modelo.pvalues > .05).sum()),
        "train_n_coefficients_p_undefined": int(modelo.pvalues.isna().sum()),
        "train_condition_number": float(modelo.condition_number),
        "train_rank_deficiency": diseno.shape[1]-rango}
    jb, jbp, skew, kurt = jarque_bera(modelo.resid)
    omni, omnip = omni_normtest(modelo.resid)
    m.update(dict(zip(["train_jarque_bera", "train_jarque_bera_pvalue", "train_skewness", "train_kurtosis",
                      "train_omnibus", "train_omnibus_pvalue"], map(float,[jb,jbp,skew,kurt,omni,omnip]))))
    notas = {}
    # QR conserva las normas de los residuos de cada regresión auxiliar.
    # VIF_j = TSS centrado original / RSS auxiliar; el intercepto está en todas las auxiliares.
    filas = []
    with threadpool_limits(limits=1):
        R = np.linalg.qr(diseno.to_numpy(), mode="r").copy()
        for j, nombre in enumerate(diseno.columns[1:], 1):
            tss = float(np.square(diseno.iloc[:,j]-diseno.iloc[:,j].mean()).sum())
            otros = np.delete(R,j,axis=1)
            coef = linalg.lstsq(otros,R[:,j],cond=1e-15)[0]
            rss = float(np.square(R[:,j]-otros@coef).sum())
            if tss == 0:
                valor = np.nan
            elif rss <= tss * 1e-15:
                valor = np.inf
            else:
                valor = tss/rss
            filas.append({"variable": nombre, "vif": valor})
    vif = pd.DataFrame(filas, columns=["variable", "vif"]).astype({"vif": float})
    m["train_vif_max"] = float(vif.vif.max())
    if resumir_vif:
        m["train_vif_mean"] = float(vif.vif.mean())
        m["train_vif_gt5_count"] = int(vif.vif.gt(5).sum())
        m["train_vif_gt10_count"] = int(vif.vif.gt(10).sum())
    m["train_vif_infinite_count"] = int(np.isinf(vif.vif).sum())
    m["train_vif_undefined_count"] = int(vif.vif.isna().sum())
    notas["vif"] = "Incluye intercepto en las regresiones auxiliares y lo excluye del máximo. Constantes: indefinido; dependencia exacta: infinito. No se elimina ninguna columna."
    if rango == diseno.shape[1] and diseno.shape[1] > 1:
        with threadpool_limits(limits=1):
            bp = het_breuschpagan(modelo.resid, diseno, robust=True)
        m.update(dict(zip(["train_bp_lm", "train_bp_lm_pvalue", "train_bp_f", "train_bp_f_pvalue"],map(float,bp))))
        notas["breusch_pagan"] = "Versión robusta de Koenker (robust=True)."
    else:
        notas["breusch_pagan"] = "No calculado: diseño sin predictores o de rango deficiente; los grados de libertad LM nominales no son válidos."
    if y_original is not None:
        # Las métricas heredadas usan precios; los diagnósticos del ajuste permanecen en ln(RUB).
        m.update(train_r2=float(modelo.rsquared), train_r2_adj=float(modelo.rsquared_adj),
            train_mae=float(np.abs(modelo.resid).mean()), train_rss=float(modelo.ssr),
            train_mse=float(modelo.ssr/len(y)), train_rmse=float(np.sqrt(modelo.ssr/len(y))))
        m.pop("train_mape")  # MAPE se registra sobre los precios originales, no sobre sus logaritmos.
        invariantes = {"train_n_obs", "train_n_predictors", "train_n_parameters_effective",
            "train_n_coefficients", "train_df_model", "train_df_resid", "train_condition_number",
            "train_rank_deficiency", "train_vif_max", "train_vif_mean", "train_vif_gt5_count",
            "train_vif_gt10_count", "train_vif_infinite_count", "train_vif_undefined_count"}
        m = {k if k in invariantes else k+"_log":v for k,v in m.items()}
        m.update({f"train_{nombre}_original":heredadas[clave] for nombre,clave in {
            "r2":"r2", "r2_adj":"r2_ajustado", "mae":"mae_rub", "rmse":"rmse_rub",
            "mape":"mape_pct", "rss":"rss_rub2"}.items()})
        m["train_mse_original"] = heredadas["rss_rub2"]/len(y)
        notas["transformacion"] = "ln(price_rub); predicción original = exp(predicción log). Sin Duan ni otra corrección."
        notas["sesgo_inversa"] = "exp(E[ln(precio)|X]) no equivale en general a E[precio|X]; puede subestimar el precio esperado."
        notas["comparabilidad"] = "Comparar con la baseline únicamente métricas predictivas calculadas en RUB sobre las mismas filas. R², AIC, BIC y RSS nativos log no son directamente comparables con los de RUB."
    invalidas = {k:("Infinito: dependencia lineal exacta" if np.isinf(v) else "Indefinido")
                 for k,v in m.items() if not np.isfinite(v)}
    validas = {k:float(v) for k,v in m.items() if np.isfinite(v)}
    notas["metricas_no_finitas_no_registradas"] = invalidas
    notas["aic_bic"] = "Convención statsmodels OLS gaussiana en " + ("RUB" if y_original is None else "ln(RUB)") + ": -2*llf + 2*rank y -2*llf + log(n)*rank; varianza residual no contada como parámetro adicional. Comparar solo misma respuesta, escala, filas y convención."
    notas["inferencia"] = "Covarianza nonrobust; si hay rango deficiente, coeficientes no identificables individualmente. No se eliminan variables."
    notas["durbin_watson"] = "Orden de filas del CSV; no prueba autocorrelación temporal de un panel."
    notas["pvalues_cero"] = "Los p-values devueltos como 0 por statsmodels/SciPy se conservan; pueden corresponder a subdesbordamiento numérico, no a probabilidad matemáticamente nula."
    notas["pvalues"] = "Conteo incluye intercepto; p-values indefinidos se cuentan por separado."
    return validas, invalidas, vif, notas

def tabla_resultados(metricas, invalidas, notas, *, nombre_run=RUN,
                      referencia=None, nombre_referencia=None):
    etiquetas = {"r2":"R²", "r2_adj":"R² ajustado", "mae":"MAE", "mse":"MSE", "rmse":"RMSE",
        "mape":"MAPE", "rss":"RSS", "aic":"AIC", "bic":"BIC", "vif_max":"VIF máximo",
        "durbin_watson":"Durbin-Watson", "f_statistic":"F-statistic", "f_pvalue":"Probabilidad F-statistic",
        "vif_mean":"VIF promedio", "vif_gt5_count":"Cantidad VIF > 5", "vif_gt10_count":"Cantidad VIF > 10"}
    filas=[]
    for k,v in {**metricas, **invalidas}.items():
        clave=k.removeprefix("train_")
        escala = "log" if clave.endswith("_log") else "original" if clave.endswith("_original") else None
        if escala:
            clave=clave.removesuffix("_"+escala)
        unidad={"mae":"RUB", "rmse":"RUB", "mse":"RUB²", "rss":"RUB²", "mape":"%",
                "vif_max":"Sin intercepto", "aic":"Train; likelihood RUB; k=rank", "bic":"Train; likelihood RUB; k=rank"}.get(clave,"Train")
        if escala == "log":
            unidad = unidad.replace("RUB", "ln(RUB)")
            unidad += "; ajuste log" if unidad == "Train" else ""
        columna = "Baseline OLS" if nombre_run == RUN else nombre_run
        etiqueta=etiquetas.get(clave,clave)+(f" ({'ln(RUB)' if escala=='log' else 'precio original'})" if escala else "")
        if referencia is None:
            filas.append({"Métrica":etiqueta,columna:(format(v,".12g") if isinstance(v,(float,int)) else v),"Unidad / Observación":unidad})
        elif k in referencia:
            anterior=referencia[k]
            diferencia=format(v-anterior,".12g") if isinstance(v,(float,int)) and np.isfinite(anterior) else "No calculable"
            filas.append({"Métrica":etiqueta,nombre_referencia:format(anterior,".12g"),
                columna:format(v,".12g") if isinstance(v,(float,int)) else v,"Diferencia":diferencia})
    if referencia is None and not any(k in metricas for k in ["train_bp_lm", "train_bp_lm_log"]):
        filas.append({"Métrica":"Breusch-Pagan",("Baseline OLS" if nombre_run==RUN else nombre_run):"No calculado","Unidad / Observación":notas["breusch_pagan"]})
    return tabla_markdown(pd.DataFrame(filas),decimales=10)

def exportar(base, modelo, diseno, y, contexto, metricas, invalidas, vif, notas,
             *, nombre_run=RUN, carpeta_salida=None, generar_graficos=True):
    import mlflow.statsmodels
    if carpeta_salida is None:
        raise ValueError("Usar exportar dentro de una SesionTemporal y especificar carpeta_salida.")
    out=validar_temporal(carpeta_salida)
    out.mkdir(exist_ok=True)
    temporal_pickle=out/"modelo_statsmodels.tmp"
    copy.deepcopy(modelo).save(temporal_pickle, remove_data=True)
    os.replace(temporal_pickle,out/"modelo_statsmodels.pkl")
    pd.DataFrame({"coeficiente":modelo.params,"error_estandar":modelo.bse,"t":modelo.tvalues,
        "p_value":modelo.pvalues,"ic95_inferior":modelo.conf_int()[0],"ic95_superior":modelo.conf_int()[1]}).to_csv(out/"coeficientes.csv",index_label="variable")
    vif.to_csv(out/"vif.csv",index=False)
    (out/"resumen_ols.txt").write_text(modelo.summary().as_text(),encoding="utf-8")
    (out/"metricas.md").write_text(tabla_resultados(metricas,invalidas,notas,nombre_run=nombre_run),encoding="utf-8")
    guardar_json(out/"metricas.json",metricas)
    guardar_json(out/"diagnosticos.json",notas)
    guardar_json(out/"contexto.json",contexto)
    es_log = contexto["transformacion_objetivo"] == "ln"
    guardar_json(out/"inferencia.json",{"entrada":f"DataFrame de las {len(contexto['predictoras'])} predictoras ya transformadas en el orden declarado",
        "columnas_diseno":list(diseno.columns),"intercepto":1.0,"unidad_salida":"RUB",
        "instruccion":"sm.add_constant(X[columnas_diseno[1:]].astype(float), has_constant='add'); " + ("np.exp(modelo.predict(diseno))" if es_log else "modelo.predict(diseno)"),
        "transformacion_objetivo":contexto["transformacion_objetivo"],
        **({"unidad_salida_modelo":"ln(RUB)", "transformacion_inversa":"exp", "correccion_sesgo":"ninguna"} if es_log else {})})
    paquetes=["numpy","pandas","statsmodels","scipy","scikit-learn","python-dotenv","threadpoolctl","matplotlib"]
    try:
        version("mlflow")
        paquetes.append("mlflow")
    except PackageNotFoundError:
        paquetes.append("mlflow-skinny")
    versiones={k:version(k) for k in paquetes}
    guardar_json(out/"versiones.json",{"python":platform.python_version(),**versiones})
    (out/"requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in versiones.items())+"\n",encoding="utf-8")
    recuperado=OLSResults.load(out/"modelo_statsmodels.pkl")
    np.testing.assert_allclose(recuperado.predict(diseno.iloc[:10]),modelo.predict(diseno.iloc[:10]),rtol=1e-12)
    carpeta_modelo=out/"modelo_mlflow"
    with TemporaryDirectory(prefix=".flavor_", dir=out) as tmp:
        temporal=Path(tmp)/"modelo"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.statsmodels.save_model(copy.deepcopy(modelo),str(temporal),remove_data=True,
                pip_requirements=[f"{k}=={v}" for k,v in versiones.items()])
        shutil.copytree(temporal,carpeta_modelo,dirs_exist_ok=True)
    verificado=mlflow.statsmodels.load_model(str(carpeta_modelo))
    np.testing.assert_allclose(verificado.predict(diseno.iloc[:10]),modelo.predict(diseno.iloc[:10]),rtol=1e-12)
    if generar_graficos:
        generar_diagnosticos_residuales(modelo, diseno, y, nombre_run=nombre_run,
            carpeta=out/"diagnosticos_residuales", unidad_objetivo=contexto["unidad"])
    assert sha256(Path(base)/contexto["csv"])==contexto["dataset_sha256"]
    return out

def conectar(raiz):
    from dotenv import load_dotenv
    import mlflow
    from mlflow.tracking import MlflowClient
    load_dotenv(Path(raiz)/".env",override=False)
    os.environ["MLFLOW_TRACKING_USERNAME"]=os.environ.get("DAGSHUB_USERNAME") or "matiasgastonlonghino"
    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"]=os.environ["DAGSHUB_USER_TOKEN"]
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]="30"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]="1"
    mlflow.set_tracking_uri(TRACKING_URI)
    cliente=MlflowClient(tracking_uri=TRACKING_URI)
    cliente.search_experiments(max_results=1)
    return cliente

def registrar(cliente,out,contexto,metricas, *, nombre_run=RUN, run_id=None, parametros_extra=None,
              requiere_graficos=True):
    if requiere_graficos:
        for nombre in ARCHIVOS:
            assert (Path(out)/"diagnosticos_residuales"/nombre).is_file(), "Generar diagnósticos de Train antes de registrar."
    exp=cliente.get_experiment_by_name(EXPERIMENTO)
    if exp is None or exp.lifecycle_stage != "active":
        raise ValueError("Debe existir el experimento activo; no se crean nuevos experimentos.")
    eid=exp.experiment_id
    existentes=[] if run_id else cliente.search_runs([eid],filter_string=f"tags.mlflow.runName = '{nombre_run}'",max_results=100)
    propios=[r for r in existentes if r.data.tags.get("dataset_sha256")==contexto["dataset_sha256"] and r.info.status=="FINISHED"]
    if propios:
        rid=propios[0].info.run_id
        # Validar el mismo Run sin reemplazar sus artefactos históricos.
        verificar(cliente,rid,eid,out,metricas,nombre_run=nombre_run,reutilizado=True)
        # El Run histórico conserva sus artefactos originales.
    else:
        if run_id:
            run=cliente.get_run(run_id)
            assert run.info.status == "RUNNING" and run.info.experiment_id == eid
            assert run.data.tags["mlflow.runName"] == nombre_run and run.data.tags["dataset_sha256"] == contexto["dataset_sha256"]
        else:
            run=cliente.create_run(eid,tags={"mlflow.runName":nombre_run,"dataset_sha256":contexto["dataset_sha256"],
                "test_utilizado":"false","baseline_version":"1"})
        rid=run.info.run_id
        try:
            parametros={"modelo":"statsmodels.OLS","intercepto":True,"cov_type":"nonrobust","solver":"pinv",
                "train_fraction":1.0,"n_train":contexto["n_train"],"n_predictors":contexto["n_predictoras"],
                "n_parameters_effective":int(metricas["train_n_parameters_effective"]),
                "objetivo":"price_rub","target_unit":contexto["unidad"],
                "target_transform":"ln(Target)" if contexto["transformacion_objetivo"]=="ln" else "none",
                "feature_selection":"none","csv":contexto["csv"],"dataset_sha256":contexto["dataset_sha256"],
                "aic_bic_convention":"statsmodels_gaussian_" + ("log_RUB" if contexto["transformacion_objetivo"]=="ln" else "RUB") + "_k_rank"}
            if contexto["transformacion_objetivo"]=="ln":
                parametros.update(inverse_transform="exp", bias_correction="none")
            for clave in ["baseline_run_id", "reference_run_id"]:
                if clave in contexto:
                    parametros[clave]=contexto[clave]
            if contexto.get("predictoras_excluidas"):
                parametros.update(feature_selection="explicit_exclusion",
                    excluded_predictors=",".join(contexto["predictoras_excluidas"]),
                    exclusion_method="user_specified_single_ablation")
            if contexto.get("seleccion_caracteristicas"):
                parametros["feature_selection"]=contexto["seleccion_caracteristicas"]
                parametros.pop("exclusion_method",None)
            parametros.update(parametros_extra or {})
            for k,v in parametros.items(): cliente.log_param(rid,k,v)
            for k,v in metricas.items(): cliente.log_metric(rid,k,v)
            cliente.log_artifacts(rid,str(out))
            fuentes = ["__init__.py", "preparacion_train_test.py", "baseline_ols.py", "experimentos_ols.py", "diagnosticos_residuales.py",
                       "almacenamiento_mlflow.py", "contrato_datasets.py"]
            if contexto.get("seleccion_caracteristicas"):
                fuentes.append("seleccion_modelos.py")
            if contexto.get("seleccion_caracteristicas") == "group_lasso_then_ols":
                fuentes.append("group_lasso_ols.py")
            with TemporaryDirectory(prefix="verificar_fuentes_") as temporal:
                for nombre in fuentes:
                    fuente = Path(__file__).with_name(nombre)
                    cliente.log_artifact(rid,str(fuente))
                    recibido = cliente.download_artifacts(rid,nombre,temporal)
                    assert sha256(recibido) == sha256(fuente), nombre
            cliente.set_terminated(rid,"FINISHED")
        except BaseException:
            cliente.set_terminated(rid,"FAILED")
            raise
    if not propios:
        verificar(cliente,rid,eid,out,metricas,nombre_run=nombre_run)
    registro={"run_id":rid,"experiment_id":eid,"experiment_name":EXPERIMENTO,"run_name":nombre_run,
        "status":"FINISHED","url":f"{TRACKING_URI}/#/experiments/{eid}/runs/{rid}"}
    guardar_json(out/"registro_dagshub.json",registro)
    return registro

def verificar(cliente,rid,eid,out,metricas, *, nombre_run=RUN, reutilizado=False):
    from tempfile import TemporaryDirectory
    remoto=cliente.get_run(rid)
    assert remoto.info.status=="FINISHED" and remoto.info.experiment_id==eid
    assert remoto.data.tags["mlflow.runName"]==nombre_run
    for k,v in metricas.items(): np.testing.assert_allclose(remoto.data.metrics[k],v,rtol=1e-10,atol=1e-12)
    contexto_local=json.loads((Path(out)/"contexto.json").read_text(encoding="utf-8"))
    assert remoto.data.params["feature_selection"]==contexto_local.get("seleccion_caracteristicas",
        "explicit_exclusion" if contexto_local.get("predictoras_excluidas") else "none")
    assert remoto.data.params["train_fraction"]=="1.0"
    # En un Run existente se valida el contrato y las métricas estables: serializaciones,
    # timestamps y código histórico pueden diferir de una exportación local posterior.
    if reutilizado:
        with TemporaryDirectory(prefix="verificar_contrato_") as tmp:
            recibido=Path(cliente.download_artifacts(rid,"contexto.json",tmp))
            historico=json.loads(recibido.read_text(encoding="utf-8"))
            for clave in ["dataset_sha256", "predictoras", "transformacion_objetivo"]:
                assert historico[clave] == contexto_local[clave], clave
            for nombre in ["metricas.json", "inferencia.json", "metricas.md"]:
                recibido=Path(cliente.download_artifacts(rid,nombre,tmp))
                assert sha256(recibido) == sha256(Path(out)/nombre), nombre
        return
    # En un Run nuevo compara todo el conjunto, incluidos los componentes MLflow.
    with TemporaryDirectory(prefix="verificar_baseline_") as tmp:
        for local in Path(out).rglob("*"):
            if local.is_file() and local.name!="registro_dagshub.json":
                rel=local.relative_to(out).as_posix()
                descargado=cliente.download_artifacts(rid,rel,tmp)
                assert sha256(descargado)==sha256(local), rel
