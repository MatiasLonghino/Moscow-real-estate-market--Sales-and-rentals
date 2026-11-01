"""Evaluación del OLS temporal guardado: sin fit, selección, CV ni uso de Test."""
import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLSResults
from threadpoolctl import threadpool_limits

from . import baseline_ols as b, bss_temporal as bt, seleccion_modelos as s

VERSION = 'evaluacion-ols6-v1'
NOMBRE_RUN = 'Evaluacion_08_OLS_Final_6_Log_Train'
FUENTES = {
    'temporal': ('BSS_07_Log_Temporal_3F_1SE_OLS', '4b3e5cf9cac24ccdbda3d3568e3fb0a5'),
    'ols177': ('OLS_02_Log_Target', '2dfd0a0d8d7a41818305f1137704b26e'),
    'bss12': ('BSS_06_Log_Union_177_Grupos_OLS', '9aee5b7348954fc48bf14562b6987e51'),
}
CARACTERISTICAS = ['total_area', 'metro_distance_type', 'building_type', 'renovation', 'district', 'date_posted_dt']
ARCHIVOS_MODELO = ['modelo_statsmodels.pkl', 'preparador.joblib', 'resultados.json', 'configuracion.json']
ACLARACION = (
    'Los dos modelos históricos utilizan 37.984 observaciones de Train después de Tukey; '
    'el OLS definitivo utiliza 39.998 sin ese filtro. Las métricas de Train, AIC, BIC y R² ajustado '
    'describen los experimentos realizados y no establecen una clasificación bajo condiciones idénticas. '
    'La comparación temporal disponible corresponde a tamaños de BSS en tres folds: '
    'su RMSE no se atribuye al modelo histórico fijo de 12 características. Test permanece reservado.'
)
METRICAS = [
    ('train_r2_log', 'R² [log]'), ('train_r2_adj_log', 'R² ajustado [log]'),
    ('train_mae_original', 'MAE Train (RUB)'), ('train_rmse_original', 'RMSE Train (RUB)'),
    ('train_aic_log', 'AIC [log]'), ('train_bic_log', 'BIC [log]'),
    ('train_vif_max', 'VIF máximo'), ('train_condition_number', 'Número de condición'),
    ('train_bp_lm_pvalue_log', 'Breusch–Pagan (p)'), ('train_jarque_bera_pvalue_log', 'Jarque–Bera (p)'),
]


def restaurar_datos_diagnostico(modelo, diseno, y):
    """Restituye datos en una copia del Results guardado con remove_data=True.

    Conserva parámetros, covarianza normalizada, rango y grados de libertad del
    ajuste original. Invalida las cachés dependientes de datos; nunca llama fit.
    """
    if list(diseno) != modelo.model.exog_names or not diseno.index.equals(y.index):
        raise ValueError('Diseño/objetivo desalineados respecto del modelo guardado.')
    if len(y) != modelo.nobs or not np.isfinite(diseno.to_numpy()).all() or not np.isfinite(y).all():
        raise ValueError('Se requieren todas las filas finitas del Train original.')
    copia = copy.deepcopy(modelo)
    X, objetivo = diseno.to_numpy(dtype=float), y.to_numpy(dtype=float)
    copia.model.exog = copia.model.wexog = X
    copia.model.endog = copia.model.wendog = objetivo
    copia.model.data.exog, copia.model.data.endog = X, objetivo
    copia.model.data.orig_exog, copia.model.data.orig_endog = diseno, y
    copia._results._cache.clear()
    np.testing.assert_array_equal(copia.params, modelo.params)
    np.testing.assert_array_equal(copia.normalized_cov_params, modelo.normalized_cov_params)
    return copia


def evaluar(modelo, preparador, datos, identidad, resultados, configuracion):
    assert identidad == configuracion['identity'] and identidad['test_used'] is False
    assert len(datos) == identidad['n_train'] == modelo.nobs == 39998
    assert resultados['selected_features'] == CARACTERISTICAS
    columnas = s.columnas_grupos(configuracion['groups'], resultados['selected_features'])
    assert columnas == resultados['columns'] and len(columnas) == len(set(columnas)) == 139
    assert preparador.n_fit == len(datos) and preparador.fecha_hasta == identidad['train_end']
    assert preparador.grupos == configuracion['groups']
    assert modelo.model.exog_names == ['const'] + columnas
    assert modelo.model.rank == len(columnas) + 1 and modelo.k_constant == 1
    assert resultados['rule']['k_selected'] == 6 and len(resultados['folds']) == 3
    assert resultados['rule']['selected_mean'] == next(r['mean'] for r in resultados['rows'] if r['k']==6)
    X = preparador.transform(datos).loc[:, columnas]
    diseno = b.sm.add_constant(X, has_constant='add')
    y = np.log(datos.price_rub).rename('ln_price_rub')
    original = np.asarray(modelo.params).copy()
    with threadpool_limits(limits=1):
        reconstruido = restaurar_datos_diagnostico(modelo, diseno, y)
        pred = np.asarray(modelo.predict(diseno))
        np.testing.assert_array_equal(pred, np.asarray(diseno) @ original)
        np.testing.assert_array_equal(pred, reconstruido.fittedvalues)
        metricas, invalidas, vif, notas = b.diagnosticar(reconstruido, diseno, y, y_original=datos.price_rub, resumir_vif=True)
    np.testing.assert_array_equal(modelo.params, original)
    assert all(k in metricas for k, _ in METRICAS), invalidas
    residuos = y.to_numpy() - pred
    rss = float(residuos @ residuos)
    n, rango = len(y), int(modelo.model.rank)
    # Comprobaciones independientes de las estadísticas reconstituidas de statsmodels.
    llf = -.5*n*(np.log(2*np.pi)+1+np.log(rss/n))
    np.testing.assert_allclose(metricas['train_aic_log'], -2*llf+2*rango, rtol=1e-12)
    np.testing.assert_allclose(metricas['train_bic_log'], -2*llf+np.log(n)*rango, rtol=1e-12)
    np.testing.assert_allclose(metricas['train_r2_log'], 1-rss/np.square(y-y.mean()).sum(), rtol=1e-12)
    np.testing.assert_allclose(metricas['train_r2_adj_log'], 1-(1-metricas['train_r2_log'])*(n-1)/(n-rango), rtol=1e-12)
    error = datos.price_rub.to_numpy() - np.exp(pred)
    np.testing.assert_allclose(metricas['train_rmse_original'], np.sqrt(np.mean(error**2)), rtol=1e-12)
    np.testing.assert_allclose(metricas['train_mae_original'], np.mean(np.abs(error)), rtol=1e-12)
    assert 'const' not in set(vif.variable) and len(vif) == 139
    metricas.update(train_n_features_original=6., temporal_rmse_mean=resultados['rule']['selected_mean'], temporal_n_folds=3.)
    comprobaciones = dict(identity=identidad, columns=columnas, selected_features=resultados['selected_features'],
        model_parameters_unchanged=True, design_predictions_match_saved_model=True,
        max_abs_prediction_difference=float(np.max(np.abs(pred-reconstruido.fittedvalues))),
        independently_checked=['R2_log','R2_adjusted_log','AIC_log','BIC_log','MAE_RUB','RMSE_RUB'],
        design_rank=rango, n_coefficients=len(original), vif_excludes_intercept=True,
        no_refit=True, no_selection=True, no_cv=True, no_test=True)
    notas['comparabilidad'] = ACLARACION
    notas['restauracion'] = 'Se reconstruyeron datos y residuos en una copia del Results original; no se ajustó ningún coeficiente.'
    return dict(metricas=metricas, invalidas=invalidas, vif=vif, notas=notas, comprobaciones=comprobaciones)


def _valor(m, clave, invalidas=None):
    if clave in m:
        return b._formato_presentacion(m[clave])
    razon = (invalidas or {}).get(clave)
    if razon:
        return '∞' if razon.startswith('Infinito') else 'Indefinido'
    raise ValueError(f'Métrica histórica ausente: {clave}; no se sustituye por otro modelo.')


def tabla_metricas(r):
    m = r['metricas']
    filas = [{'Métrica': nombre, 'OLS definitivo — 6': _valor(m, clave, r['invalidas'])} for clave, nombre in METRICAS]
    filas += [{'Métrica':'RMSE temporal promedio (RUB)', 'OLS definitivo — 6': _valor(m, 'temporal_rmse_mean')},
              {'Métrica':'Folds / criterio de selección', 'OLS definitivo — 6':'3 / 1-SE'}]
    return b.tabla_markdown(pd.DataFrame(filas))


def tabla_comparativa(r):
    h = r['referencias']
    modelos = [('OLS logarítmico — 177', {**h['ols177']['metrics'], 'train_n_features_original':h['ols177']['n_features_original']}, h['ols177']['invalidas']),
               ('BSS histórico — 12', h['bss12']['metrics'], h['bss12']['invalidas']),
               ('OLS definitivo — 6', r['metricas'], r['invalidas'])]
    filas = []
    for clave, etiqueta in [('train_n_features_original','Características originales'),
                           ('train_n_predictors','Columnas predictoras'), ('train_n_obs','Observaciones Train')] + METRICAS:
        fila = {'Métrica': etiqueta}
        for nombre, metricas, invalidas in modelos:
            fila[nombre] = (f'{int(metricas[clave]):,}'.replace(',', '.') if clave in
                ('train_n_features_original','train_n_predictors','train_n_obs') else _valor(metricas, clave, invalidas))
        filas.append(fila)
    return b.tabla_markdown(pd.DataFrame(filas)) + '\n\n' + ACLARACION


def interpretacion(r):
    m = r['metricas']
    vif_texto = ('no señala multicolinealidad intensa entre las columnas transformadas' if m['train_vif_max']<5
                 else 'señala colinealidad entre algunas columnas transformadas')
    return (f"**Ajuste estadístico.** R² ajustado log={m['train_r2_adj_log']:.6f}; AIC={m['train_aic_log']:,.3f} "
        f"y BIC={m['train_bic_log']:,.3f} describen este Train. El VIF máximo={m['train_vif_max']:.3f} {vif_texto}. "
        f"El número de condición={m['train_condition_number']:.3f} es elevado y depende también de la escala del diseño. "
        f"Breusch–Pagan (p={m['train_bp_lm_pvalue_log']:.3e}) y Jarque–Bera (p={m['train_jarque_bera_pvalue_log']:.3e}) "
        'aportan evidencia contra homocedasticidad y normalidad; la inferencia clásica requiere cautela.\n\n'
        f"**Error predictivo.** MAE Train={m['train_mae_original']:,.3f} RUB y RMSE Train={m['train_rmse_original']:,.3f} RUB. "
        f"El RMSE temporal promedio ya registrado es {m['temporal_rmse_mean']:,.3f} RUB; evalúa ventanas posteriores "
        'y es distinto del error de entrenamiento.\n\n'
        '**Selección del modelo.** 1-SE eligió seis características por parsimonia, aunque doce lograron el menor RMSE '
        'promedio temporal. Esto no demuestra una mejora de precisión por reducir grupos ni que se haya eliminado el overfitting. '
        'La evaluación independiente en Test sigue pendiente.')


def confirmacion(r):
    fuente = r['referencias']['temporal']; run = r['run']
    url = lambda v: f"{b.TRACKING_URI}/#/experiments/{v['experiment_id']}/runs/{v['run_id']}"
    return (f"Modelo recuperado: [{fuente['run_name']}]({url(fuente)}) · **FINISHED**. "
        'Seis características originales, **139 columnas más intercepto**, 39.998 observaciones de Train sin Tukey global. '
        f"Coeficientes y preparador originales conservados. [Run separado de evaluación]({url(run)}) · **{run['status']}**.")


def detalles(r):
    contenido = {k:r[k] for k in ('comprobaciones','notas','invalidas','proveniencia')}
    return ('<details><summary>Detalles de cálculo, verificaciones y metadatos</summary>\n\n'
        'R² y R² ajustado sobre ln(price_rub); errores RUB mediante exp sin corrección. '
        'AIC/BIC gaussianos de statsmodels; VIF sobre columnas transformadas con intercepto en auxiliares, '
        'excluido del máximo. BP robusto de Koenker, igual que en los experimentos anteriores. '
        'No se recalcularon gráficos, importancias ni folds. Los p-values numéricamente cero se conservan sin redondearlos a probabilidades exactas.\n\n'
        '```json\n'+json.dumps(contenido, ensure_ascii=False, indent=2)+'\n```\n\n</details>')


def recuperar_fuentes(cliente, salida):
    """Solo lectura de tres Runs y del linaje histórico; no abre Test ni Train transformado."""
    from .almacenamiento_mlflow import RUN_TRANSFORMACION, leer_json
    salida = b.validar_temporal(salida)
    referencias = {}
    for clave, (nombre, rid) in FUENTES.items():
        run = s.recuperar_experimento(cliente, nombre=nombre, run_id=rid)
        carpeta = salida/clave
        carpeta.mkdir(exist_ok=True)
        archivos = ARCHIVOS_MODELO if clave=='temporal' else ['metricas.json', 'contexto.json']
        for archivo in archivos:
            cliente.download_artifacts(rid, archivo, str(carpeta))
        if clave!='temporal':
            metricas = json.loads((carpeta/'metricas.json').read_text(encoding='utf-8'))
            assert metricas == run['metrics']
            contexto = json.loads((carpeta/'contexto.json').read_text(encoding='utf-8'))
            assert contexto['test_utilizado'] is False and contexto['n_train']==37984
            assert contexto['transformacion_objetivo']=='ln'
            run['contexto'] = contexto
            run['invalidas'] = {}
            if any(k not in metricas for k, _ in METRICAS):
                raise ValueError('Faltan métricas históricas originales; no se ejecutan nuevos experimentos.')
        referencias[clave] = run
    resumen = leer_json(cliente, RUN_TRANSFORMACION, 'transformacion/resumen_final.json')
    grupos = s.grupos_caracteristicas(referencias['ols177']['contexto']['predictoras'], resumen, excluidas=())
    assert len(grupos)==23 and sum(map(len, grupos.values()))==177
    referencias['ols177'].update(n_features_original=len(grupos), groups=grupos,
        lineage_source=dict(run_id=RUN_TRANSFORMACION, artifact='transformacion/resumen_final.json'))
    assert referencias['bss12']['metrics']['train_n_features_original']==12
    assert referencias['ols177']['metrics']['train_n_predictors']==177
    assert referencias['bss12']['metrics']['train_n_predictors']==145
    assert referencias['ols177']['contexto']['dataset_sha256']==referencias['bss12']['contexto']['dataset_sha256']
    return referencias


def preparar_evaluacion(base, salida, referencias):
    salida = b.validar_temporal(salida)
    origen = salida/'temporal'
    leer = lambda nombre: json.loads((origen/nombre).read_text(encoding='utf-8'))
    resultados, cfg = leer('resultados.json'), leer('configuracion.json')
    assert b.sha256(origen/'modelo_statsmodels.pkl')==resultados['model_sha256']
    assert b.sha256(origen/'preparador.joblib')==resultados['preprocessor_sha256']
    assert referencias['temporal']['metrics']['selected_mean']==resultados['rule']['selected_mean']
    assert referencias['temporal']['metrics']['k_selected']==6
    datos, identidad = bt.cargar_train_crudo(base)
    modelo = OLSResults.load(origen/'modelo_statsmodels.pkl')
    preparador = joblib.load(origen/'preparador.joblib')
    r = evaluar(modelo, preparador, datos, identidad, resultados, cfg)
    r['referencias'] = referencias
    r['proveniencia'] = dict(version=VERSION, source_run_id=referencias['temporal']['run_id'],
        model_sha256=resultados['model_sha256'], preprocessor_sha256=resultados['preprocessor_sha256'],
        source_artifact_sha256={n:b.sha256(origen/n) for n in ARCHIVOS_MODELO},
        temporal_rule=resultados['rule'], temporal_folds=resultados['folds'],
        historical_run_ids={k:referencias[k]['run_id'] for k in ('ols177','bss12')},
        code_sha256={n:b.sha256(Path(__file__).with_name(n)) for n in
            ('evaluacion_ols_final.py','baseline_ols.py','bss_temporal.py')})
    destino = salida/'evaluacion'
    destino.mkdir(exist_ok=True)
    b.guardar_json(destino/'metricas.json', r['metricas'])
    b.guardar_json(destino/'evaluacion.json', {k:v for k,v in r.items() if k!='vif'})
    r['vif'].to_csv(destino/'vif.csv', index=False)
    for archivo, texto in [('metricas.md',tabla_metricas(r)), ('comparacion.md',tabla_comparativa(r)), ('interpretacion.md',interpretacion(r))]:
        (destino/archivo).write_text(texto, encoding='utf-8')
    (destino/'evaluacion_ols_final.py').write_bytes(Path(__file__).read_bytes())
    assert b.sha256(origen/'modelo_statsmodels.pkl')==resultados['model_sha256']
    assert b.sha256(origen/'preparador.joblib')==resultados['preprocessor_sha256']
    return r


def _validar_fuentes_intactas(cliente, referencias):
    for r in referencias.values():
        ahora = cliente.get_run(r['run_id'])
        assert ahora.info.status==r['status']=='FINISHED'
        assert ahora.data.metrics==r['metrics'] and ahora.data.params==r['params'] and ahora.data.tags==r['tags']


def registrar(cliente, salida, r, *, nombre_run=NOMBRE_RUN):
    """Publica únicamente métricas/diagnósticos; referencia el modelo, sin copiarlo."""
    from mlflow.entities import Param, Metric
    from tempfile import TemporaryDirectory
    salida = b.validar_temporal(salida)/'evaluacion'
    exp = cliente.get_experiment_by_name(b.EXPERIMENTO)
    assert exp is not None and exp.lifecycle_stage=='active'
    existentes = cliente.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=2)
    if existentes:
        raise ValueError('Run de evaluación existente; recuperar en lugar de duplicar o sobrescribir.')
    assert not any(p.suffix in ('.pkl','.joblib') for p in salida.rglob('*'))
    _validar_fuentes_intactas(cliente, r['referencias'])
    prov = r['proveniencia']
    run = cliente.create_run(exp.experiment_id, tags={'mlflow.runName':nombre_run, 'mlflow.parentRunId':prov['source_run_id'],
        'source_run_id':prov['source_run_id'], 'evaluation_only':'true', 'test_utilizado':'false',
        'training_performed':'false', 'train_sha256':r['comprobaciones']['identity']['train_sha256']})
    rid = run.info.run_id
    try:
        params = dict(evaluation_version=VERSION, source_run_id=prov['source_run_id'],
            model_artifact_uri=f"runs:/{prov['source_run_id']}/modelo_statsmodels.pkl",
            preprocessor_artifact_uri=f"runs:/{prov['source_run_id']}/preparador.joblib",
            model_sha256=prov['model_sha256'], preprocessor_sha256=prov['preprocessor_sha256'],
            train_sha256=r['comprobaciones']['identity']['train_sha256'],
            target_transform='ln(price_rub)', inverse_transform='exp', bias_correction='none',
            selection_rule='1-SE', selected_features=json.dumps(CARACTERISTICAS), n_folds=3,
            train_population='raw chronological Train without global Tukey', n_train=39998,
            no_refit=True, no_selection=True, no_cv=True, test_used=False,
            bp_variant='Koenker robust=True', vif_intercept_excluded_from_max=True,
            aic_bic_convention='statsmodels_gaussian_log_RUB_k_rank',
            historical_ols_run_id=FUENTES['ols177'][1], historical_bss_run_id=FUENTES['bss12'][1])
        cliente.log_batch(rid, params=[Param(k,str(v)) for k,v in params.items()],
            metrics=[Metric(k,float(v),int(time.time()*1000),0) for k,v in r['metricas'].items()])
        cliente.log_artifacts(rid, str(salida))
        with TemporaryDirectory(prefix='verificar_evaluacion_ols6_') as temporal:
            for archivo in salida.iterdir():
                remoto = cliente.download_artifacts(rid, archivo.name, temporal)
                assert b.sha256(remoto)==b.sha256(archivo), archivo.name
        assert cliente.get_run(rid).data.metrics==r['metricas']
        _validar_fuentes_intactas(cliente, r['referencias'])
        cliente.set_terminated(rid,'FINISHED')
    except BaseException:
        cliente.set_terminated(rid,'FAILED')
        raise
    r['run'] = dict(run_id=rid, experiment_id=exp.experiment_id, run_name=nombre_run, status='FINISHED')
    return r['run']


def ejecutar_presentacion(base, sesion, cliente, *, modo='recuperar', nombre_run=NOMBRE_RUN):
    if modo not in ('recuperar','registrar'):
        raise ValueError('Elegir recuperar o registrar; nunca se entrena ni se selecciona.')
    out = sesion.carpeta('evaluacion_ols6')
    if modo=='registrar':
        referencias = recuperar_fuentes(cliente, out)
        r = preparar_evaluacion(base, out, referencias)
        registrar(cliente, out, r, nombre_run=nombre_run)
        return r
    run = s.recuperar_experimento(cliente, nombre=nombre_run,
        parametros={'evaluation_version':VERSION,'source_run_id':FUENTES['temporal'][1]})
    archivo = cliente.download_artifacts(run['run_id'], 'evaluacion.json', str(out))
    r = json.loads(Path(archivo).read_text(encoding='utf-8'))
    assert r['metricas']==run['metrics'] and r['proveniencia']['version']==VERSION
    assert r['proveniencia']['source_run_id']==FUENTES['temporal'][1]
    assert r['metricas']['train_n_obs']==39998 and r['metricas']['train_n_predictors']==139
    assert r['comprobaciones']['no_refit'] and r['comprobaciones']['no_cv'] and r['comprobaciones']['no_test']
    assert run['params']['model_sha256']==r['proveniencia']['model_sha256']
    assert run['params']['preprocessor_sha256']==r['proveniencia']['preprocessor_sha256']
    _validar_fuentes_intactas(cliente, r['referencias'])
    r['run'] = run
    return r
