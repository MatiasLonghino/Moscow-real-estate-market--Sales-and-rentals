"""Forward OLS agrupado; importar no entrena ni registra modelos."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from . import baseline_ols as b
EXCLUIDAS = ('to_center_km', 'living_area', 'rooms')
NOMBRE_RUN = 'Forward_01_Log_Grupos'
NOMBRE_RUN_177 = 'Forward_04_Log_177_Grupos'
REFERENCIA = 'OLS_05_Log_Sin_rooms'
REFERENCIA_177 = 'OLS_02_Log_Target'
CONFIG_FORWARD = {'min_mejora_pct': .5, 'max_features': 12, 'random_state': 42}


def cargar_presentacion(base, sesion, *, excluidas=EXCLUIDAS, config_forward=None,
                        config_lasso=None, cliente=None):
    """Una carga de Train; universo y partición compartidos, sin ajustar transformadores."""
    from . import group_lasso_ols as gl
    from .almacenamiento_mlflow import leer_json, RUN_TRANSFORMACION
    from .contrato_datasets import RAIZ
    X_completo, y, contexto = b.cargar_train(base)
    excluidas = tuple(excluidas)
    if len(set(excluidas)) != len(excluidas) or not set(excluidas) <= set(X_completo):
        raise ValueError('Exclusiones repetidas o ajenas a Train.')
    cliente = b.conectar(RAIZ) if cliente is None else cliente
    X = X_completo.drop(columns=list(excluidas))
    resumen = leer_json(cliente, RUN_TRANSFORMACION, 'transformacion/resumen_final.json')
    assert resumen['columnas_X'] == list(X_completo)
    grupos = grupos_caracteristicas(X.columns, resumen, excluidas=excluidas)
    cfg = {**CONFIG_FORWARD, **(config_forward or {})}
    cfg_gl = {**gl.CONFIG, **(config_lasso or {})}
    if set(cfg) != set(CONFIG_FORWARD) or set(cfg_gl) != set(gl.CONFIG):
        raise ValueError('Parámetros de selección desconocidos.')
    if not 1 <= cfg['max_features'] <= 12 or not np.isfinite(cfg['min_mejora_pct']) or cfg['min_mejora_pct'] < 0:
        raise ValueError('Configuración Forward inválida.')
    fit, val = train_test_split(np.arange(len(y)), test_size=.2, random_state=cfg['random_state'])
    return dict(base=Path(base), sesion=sesion, cliente=cliente, X=X, y=y, contexto=contexto,
                grupos=grupos, excluidas=excluidas, config_forward=cfg, config_lasso=cfg_gl,
                fit=fit, val=val, n_exportadas=X_completo.shape[1])


def _parametros_presentacion(datos, metodo):
    cfg = datos['config_forward'] if metodo in ('forward', 'backward') else datos['config_lasso']
    return {'target_transform': 'ln(Target)', 'inverse_transform': 'exp', 'bias_correction': 'none',
        'intercepto': 'True', 'cov_type': 'nonrobust', 'solver': 'pinv', 'train_fraction': '1.0',
        'n_train': len(datos['y']), 'excluded_predictors': ','.join(datos['excluidas']),
        'max_features': cfg['max_features'], 'min_mejora_pct': cfg['min_mejora_pct'],
        'validation_fraction': .2, 'random_state': datos['config_forward']['random_state'],
        'feature_selection': {'forward': 'forward_stepwise_grouped', 'lasso': 'group_lasso_then_ols',
                              'backward': 'backward_stepwise_grouped'}[metodo],
        'aic_bic_convention': 'statsmodels_gaussian_log_RUB_k_rank'}


def cargar_forward_177(datos):
    """Universo independiente; la carga común mantiene las transformaciones exportadas."""
    completo = cargar_presentacion(datos['base'], datos['sesion'], excluidas=(), cliente=datos['cliente'])
    assert completo['X'].shape[1] == completo['n_exportadas'] == 177
    assert set(EXCLUIDAS) <= set(completo['X']) and completo['excluidas'] == ()
    assert completo['contexto']['dataset_sha256'] == datos['contexto']['dataset_sha256']
    for clave in ('fit', 'val'):
        np.testing.assert_array_equal(completo[clave], datos[clave])
    return completo


def _parametros_forward_177(datos):
    return dict(n_candidate_predictors=177, n_candidate_groups=len(datos['grupos']),
        candidate_columns=json.dumps(list(datos['X'])), reference_run_name=REFERENCIA_177,
        excluded_predictors='', permutation_repeats=5)


def _validar_forward_177(datos, resultado):
    """Audita referencia, universo y cada decisión sin repetir selección ni ajustes."""
    p, sel = resultado['run']['params'], resultado['seleccion']
    assert resultado['contexto']['universo_predictoras'] == list(datos['X'])
    assert resultado['contexto']['n_predictoras_inicial'] == 177
    ref = recuperar_experimento(datos['cliente'], nombre=REFERENCIA_177, run_id=p['reference_run_id'],
        parametros={'n_predictors': 177, 'target_transform': 'ln(Target)',
                    'inverse_transform': 'exp', 'bias_correction': 'none'},
        etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
    assert resultado['contexto']['reference_run_id'] == ref['run_id']
    assert json.loads(p['feature_order']) == sel['seleccionadas']
    assert sel['iteraciones'] == len(sel['historial'])
    assert sel['candidatos_evaluados'] == len(sel['candidatos'])
    elegidas, actual = [], sel['rmse_intercepto_original']
    vistos = 0
    for it, paso in enumerate(sel['historial'], 1):
        candidatos = [c for c in sel['candidatos'] if c['iteracion'] == it]
        assert [c['candidata'] for c in candidatos] == [g for g in datos['grupos'] if g not in elegidas]
        vistos += len(candidatos)
        mejor = min(candidatos, key=lambda c: c['rmse_validation_original']
                    if c['rmse_validation_original'] is not None else np.inf)
        score = mejor['rmse_validation_original']
        mejora = 100*(actual-score)/actual if score is not None and actual > 0 else None
        acepta = mejora is not None and mejora >= datos['config_forward']['min_mejora_pct']
        assert paso['iteracion'] == it and paso['aceptada'] == acepta
        assert paso['mejor_candidata'] == mejor['candidata']
        assert paso['rmse_mejor_candidato'] == score
        if mejora is not None:
            np.testing.assert_allclose(paso['mejora_pct'], mejora, rtol=1e-12)
        if acepta:
            elegidas.append(mejor['candidata'])
            actual = score
        else:
            assert it == len(sel['historial']), 'Forward debe detenerse en el primer rechazo.'
        assert paso['caracteristica_incorporada'] == (mejor['candidata'] if acepta else None)
        assert paso['caracteristicas_acumuladas'] == elegidas
        assert paso['n_predictores'] == len(columnas_grupos(datos['grupos'], elegidas))
        np.testing.assert_allclose(paso['rmse_validacion'], actual)
    assert vistos == len(sel['candidatos']) and elegidas == sel['seleccionadas']
    assert 0 < len(elegidas) <= datos['config_forward']['max_features']
    np.testing.assert_allclose(actual, sel['rmse_validation_original'])
    motivo = ('min_mejora_pct' if not sel['historial'][-1]['aceptada'] else
              'max_features' if len(elegidas) == datos['config_forward']['max_features'] else 'candidatas_agotadas')
    assert sel['motivo_parada'] == p['selection_stop_reason'] == motivo


def _validar_resultado_presentacion(datos, r):
    """Candidatas, configuración, grupos, partición, modelo y escala deben coincidir."""
    p, ctx, sel = r['run']['params'], r['contexto'], r['seleccion']
    for k, v in _parametros_presentacion(datos, r['metodo']).items():
        # Los Runs sin exclusiones no necesitan el parámetro excluded_predictors.
        if p.get(k, '' if k == 'excluded_predictors' else None) != str(v):
            raise ValueError(f'Resultado histórico incompatible: {k}; ejecutar selección nueva con otro nombre de Run.')
    if (r['grupos'] != datos['grupos'] or list(r['grupos']) != list(datos['grupos'])
            or tuple(ctx.get('predictoras_excluidas', ())) != datos['excluidas']):
        raise ValueError('Universo candidato incompatible; no reutilizar resultados históricos.')
    assert ctx['dataset_sha256'] == datos['contexto']['dataset_sha256']
    assert ctx['transformacion_objetivo'] == 'ln' and ctx['n_train'] == len(datos['y'])
    for clave, esperado in [('fit', datos['fit']), ('validation', datos['val'])]:
        if not np.array_equal(r['particion'][clave], esperado):
            raise ValueError('Partición interna incompatible con la configuración solicitada.')
    if r['metodo'] == 'lasso':
        if sel['config'] != datos['config_lasso']:
            raise ValueError('Configuración Group Lasso incompatible; requiere nueva selección.')
        np.testing.assert_allclose(json.loads(p['lambda_grid']), sel['lambdas'], rtol=0, atol=0)
        np.testing.assert_allclose(sel['lambda'], float(p['lambda_selected']), rtol=0, atol=0)
        cfg = sel['config']
        np.testing.assert_allclose(sel['lambdas'], sel['lambda_max']*np.geomspace(
            1., cfg['lambda_min_ratio'], cfg['n_lambdas']), rtol=1e-12)
        for parametro, clave in [('support_tol', 'support_tol'), ('selection_solver_tol', 'solver_tol'),
                                 ('selection_solver_max_iter', 'max_iter')]:
            assert float(p[parametro]) == cfg[clave]
    else:
        assert p['selection_metric'] == 'RMSE validation original RUB; exp sin corrección'
    seleccionadas = sel['seleccionadas']
    assert seleccionadas == json.loads(p['selected_features'])
    columnas = columnas_grupos(datos['grupos'], seleccionadas)
    assert columnas == ctx['predictoras'] == r['modelo'].model.exog_names[1:]
    assert r['modelo'].model.exog_names[0] == 'const'
    assert r['modelo'].model.rank == len(columnas)+1 and r['modelo'].nobs == len(datos['y'])
    m = r['metricas']
    assert m == r['run']['metrics'] and all(np.isfinite(list(m.values())))
    assert m['train_n_features_original'] == len(seleccionadas)
    assert m['train_n_predictors'] == len(columnas) == int(p['n_predictors'])
    np.testing.assert_allclose(m['selection_rmse_validation_original'], sel['rmse_validation_original'])
    with threadpool_limits(limits=1):
        diseno = b.sm.add_constant(datos['X'][columnas], has_constant='add')
        pred_log = np.asarray(r['modelo'].predict(diseno))
    np.testing.assert_allclose(np.sqrt(np.mean((np.asarray(datos['y'])-np.exp(pred_log))**2)),
                               m['train_rmse_original'], rtol=1e-10)
    np.testing.assert_allclose([r['modelo'].rsquared_adj, r['modelo'].aic, r['modelo'].bic],
        [m['train_r2_adj_log'], m['train_aic_log'], m['train_bic_log']], rtol=1e-10)


def ejecutar_presentacion(datos, metodo, *, modo='recuperar', anterior=None, nombre_run=None):
    """Recuperación estricta o selección nueva explícita; nunca una sustitución silenciosa.

    Cambiar exclusiones/configuración requiere modo='nuevo' y un nombre de Run libre.
    Las 177 candidatas se habilitan con excluidas=(); los bloques OHE no cambian.
    """
    from . import group_lasso_ols as gl
    from statsmodels.regression.linear_model import OLSResults
    from . import backward_ols as bw
    if metodo not in ('forward', 'lasso', 'backward') or modo not in ('recuperar', 'nuevo'):
        raise ValueError('Elegir forward/lasso/backward y recuperar/nuevo.')
    if metodo == 'backward':
        if datos['excluidas'] or datos['X'].shape[1] != 177 or datos['config_forward'] != bw.CONFIG:
            raise ValueError('Backward requiere 177 columnas, exclusiones vacías y configuración fija.')
    forward_177 = metodo == 'forward' and datos['excluidas'] == ()
    lasso_177 = metodo == 'lasso' and datos['excluidas'] == ()
    if forward_177 and (datos['X'].shape[1] != 177 or datos['config_forward'] != CONFIG_FORWARD):
        raise ValueError('Forward 177 requiere 177 columnas y configuración fija 80/20, semilla 42, 12 grupos y 0,5 %.')
    if lasso_177 and (datos['X'].shape[1] != 177 or datos['config_lasso'] != gl.CONFIG
                      or datos['config_forward'] != CONFIG_FORWARD):
        raise ValueError('Group Lasso 177 requiere el universo y las configuraciones históricas exactas.')
    nombre = nombre_run or {'forward': NOMBRE_RUN, 'lasso': gl.NOMBRE_RUN, 'backward': bw.NOMBRE_RUN}[metodo]
    if forward_177 and nombre_run is None:
        nombre = NOMBRE_RUN_177
    if lasso_177 and nombre_run is None:
        nombre = gl.NOMBRE_RUN_177
    if "'" in nombre or not nombre.strip():
        raise ValueError('Nombre de Run inválido.')
    if lasso_177:
        gl.validar_referencia_177(datos, anterior)
    elif metodo == 'lasso' and (modo == 'nuevo' or anterior is not None):
        if anterior is None or anterior.get('metodo') != 'forward':
            raise ValueError('Group Lasso requiere la referencia Forward de esta comparación.')
        _validar_resultado_presentacion(datos, anterior)
    cliente, sesion = datos['cliente'], datos['sesion']
    carpeta = ('presentacion_forward_177' if forward_177 else
               'presentacion_lasso_177' if lasso_177 else 'presentacion_'+metodo)
    out = sesion.carpeta(carpeta)
    if modo == 'nuevo':
        exp = cliente.get_experiment_by_name(b.EXPERIMENTO)
        if exp is None or exp.lifecycle_stage != 'active':
            raise ValueError('Debe existir el experimento activo.')
        if cliente.search_runs([exp.experiment_id],
                filter_string=f"tags.mlflow.runName = '{nombre}'", max_results=1):
            raise ValueError('Para una selección nueva utilizar un nombre de Run nuevo, sin sobrescribir el histórico.')
        if metodo == 'forward':
            if datos['excluidas'] == EXCLUIDAS:
                referencia_nombre = REFERENCIA
            elif not datos['excluidas']:
                referencia_nombre = 'OLS_02_Log_Target'
            else:
                raise ValueError('Definir una referencia OLS compatible para estas exclusiones.')
            resultado = ejecutar_experimento(datos['base'], cliente, **datos['config_forward'],
                carpeta_salida=out, excluidas=datos['excluidas'], referencia_nombre=referencia_nombre,
                nombre_run=nombre)
            rid = resultado['registro']['run_id']
        elif metodo == 'backward':
            resultado = bw.ejecutar_experimento(datos, out, nombre_run=nombre)
            rid = resultado['registro']['run_id']
        else:
            if anterior is None:
                raise ValueError('Group Lasso requiere la referencia Forward de esta comparación.')
            seleccion = gl.buscar_group_lasso(datos['X'], datos['y'], datos['grupos'], datos['fit'], datos['val'],
                out, config=datos['config_lasso'], excluidas=datos['excluidas'])
            resultado = gl.entrenar_exportar(datos['base'], datos['X'], datos['y'], datos['grupos'],
                seleccion, datos['contexto'], anterior['run'], out, excluidas=datos['excluidas'], nombre_run=nombre)
            rid = gl.registrar(cliente, resultado, seleccion, anterior['run'], nombre_run=nombre)['run_id']
    else:
        rid = None
    run = recuperar_experimento(cliente, nombre=nombre, run_id=rid,
        etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
    # Antes de descargar, rechaza explícitamente cambios en parámetros de selección.
    for k, v in _parametros_presentacion(datos, metodo).items():
        if run['params'].get(k, '' if k == 'excluded_predictors' else None) != str(v):
            raise ValueError(f'Run incompatible ({k}); ejecutar una selección nueva con otro nombre de Run.')
    if forward_177:
        for k, v in _parametros_forward_177(datos).items():
            if run['params'].get(k) != str(v):
                raise ValueError(f'Run Forward 177 incompatible ({k}).')
    if lasso_177:
        for k, v in gl.parametros_177(datos, anterior).items():
            if run['params'].get(k) != str(v):
                raise ValueError(f'Run Group Lasso 177 incompatible ({k}).')
    if metodo == 'backward':
        for k, esperado in dict(algorithm_version=bw.VERSION, n_candidate_predictors='177',
                               reference_run_name=bw.REFERENCIA, permutation_repeats='10').items():
            if run['params'].get(k) != esperado:
                raise ValueError(f'Run Backward incompatible ({k}).')
        if json.loads(run['params']['candidate_columns']) != list(datos['X']):
            raise ValueError('Universo Backward incompatible.')
    nombres = ['contexto.json', 'seleccion.json', 'metricas.json', 'grupos_caracteristicas.json',
               'particion_interna.npz', 'modelo_statsmodels.pkl', 'inferencia.json']
    if lasso_177:
        nombres += ['scaler_seleccion.joblib', 'ruta_coeficientes_group_lasso.npz', 'estabilidad_numerica.json']
    for archivo in nombres:
        cliente.download_artifacts(run['run_id'], archivo, str(out))
    leer = lambda nombre: json.loads((out/nombre).read_text(encoding='utf-8'))
    grupos = leer('grupos_caracteristicas.json')
    grupos = grupos['grupos'] if metodo == 'forward' else grupos
    with np.load(out/'particion_interna.npz') as particion:
        indices = {k: particion[k].copy() for k in ('fit', 'validation')}
    r = dict(metodo=metodo, modo=modo, run=run, salida=out, contexto=leer('contexto.json'),
             seleccion=leer('seleccion.json'), metricas=leer('metricas.json'), grupos=grupos,
             modelo=OLSResults.load(out/'modelo_statsmodels.pkl'), particion=indices)
    _validar_resultado_presentacion(datos, r)
    if metodo == 'backward':
        bw.validar_recuperacion(datos, r)
    if forward_177:
        _validar_forward_177(datos, r)
    if lasso_177:
        gl.validar_recuperacion_177(datos, r)
    assert leer('inferencia.json')['columnas_diseno'] == r['modelo'].model.exog_names
    if metodo == 'lasso':
        if anterior is None or run['params']['reference_run_id'] != anterior['run']['run_id']:
            raise ValueError('Group Lasso no corresponde al Forward de esta comparación.')
    if modo == 'recuperar':
        for nombre_archivo in ['importancia_caracteristicas.csv', 'importancia_metodologia.json',
                               'importancia_caracteristicas_boxplot.png']:
            cliente.download_artifacts(run['run_id'], nombre_archivo, str(out))
        importancia = pd.read_csv(out/'importancia_caracteristicas.csv')
        meta = leer('importancia_metodologia.json')
        repeticiones = 5 if metodo == 'forward' else 10
        assert meta['n_repeticiones'] == repeticiones and meta['random_state'] == 42
        assert meta['dataset_sha256'] == datos['contexto']['dataset_sha256']
        if 'modelo_final_sha256' in meta:
            assert meta['modelo_final_sha256'] == b.sha256(out/'modelo_statsmodels.pkl')
    else:
        repeticiones = 5 if metodo == 'forward' else 10
        importancia = importancia_permutacion_grupos(datos['X'], datos['y'], datos['grupos'],
            r['seleccion']['seleccionadas'], datos['fit'], datos['val'], n_repeticiones=repeticiones,
            random_state=42, rmse_esperado=r['metricas']['selection_rmse_validation_original'])
        importancia.to_csv(out/'importancia_caracteristicas.csv', index=False)
        guardar_boxplot_importancia(importancia, out/'importancia_caracteristicas_boxplot.png')
        b.guardar_json(out/'importancia_metodologia.json', dict(n_repeticiones=repeticiones, random_state=42,
            dataset_sha256=datos['contexto']['dataset_sha256'],
            formula='100 * (RMSE permutado - RMSE original) / RMSE original',
            evaluacion='validación interna de Train utilizada en selección; descriptiva',
            modelo_final_sha256=b.sha256(out/'modelo_statsmodels.pkl')))
    assert set(importancia.caracteristica) == set(r['seleccion']['seleccionadas'])
    assert importancia.groupby('caracteristica').size().eq(repeticiones).all()
    assert np.isfinite(importancia.select_dtypes(include='number')).all().all()
    np.testing.assert_allclose(importancia.rmse_original, r['metricas']['selection_rmse_validation_original'], rtol=1e-10)
    np.testing.assert_allclose(importancia.importancia_pct,
        100*(importancia.rmse_permutado-importancia.rmse_original)/importancia.rmse_original, rtol=1e-10, atol=1e-10)
    r['importancias'] = importancia
    r['caracteristicas_md'] = resumen_caracteristicas(r['seleccion']['seleccionadas'], grupos, r['modelo'], mostrar_orden=forward_177)
    if metodo == 'backward' or forward_177 or lasso_177:
        filas = [{'Característica original': g, 'Columna transformada': c,
                  'Coeficiente OLS [ln(RUB)]': float(r['modelo'].params[c])}
                 for g in r['seleccion']['seleccionadas'] for c in grupos[g]]
        r['caracteristicas_md'] += '\n\n'+b.tabla_markdown(pd.DataFrame(filas))
    if forward_177:
        r['etiqueta'] = 'Forward Stepwise Selection — 177 predictoras'
    if lasso_177:
        r['etiqueta'] = 'Group Lasso + OLS — 177 predictoras'
    if modo == 'nuevo':
        (out/'caracteristicas_seleccionadas.md').write_text(r['caracteristicas_md'], encoding='utf-8')
        registrar_presentacion(cliente, run, out, ['caracteristicas_seleccionadas.md',
            'importancia_caracteristicas.csv', 'importancia_caracteristicas_boxplot.png', 'importancia_metodologia.json'])
    despues = cliente.get_run(run['run_id'])
    assert despues.info.status == 'FINISHED' and despues.data.metrics == run['metrics'] and despues.data.params == run['params']
    assert b.sha256(datos['base']/datos['contexto']['csv']) == datos['contexto']['dataset_sha256']
    return r


def tabla_metricas_presentacion(resultado, anterior=None):
    filas = []
    for clave, etiqueta, evaluacion in [
        ('train_r2_log', 'R² [ln(RUB)]', 'Train'),
        ('train_r2_adj_log', 'R² ajustado [ln(RUB)]', 'Train'),
        ('train_mae_original', 'MAE (RUB)', 'Train'), ('train_rmse_original', 'RMSE (RUB)', 'Train'),
        ('train_aic_log', 'AIC [ln(RUB)]', 'Train'), ('train_bic_log', 'BIC [ln(RUB)]', 'Train'),
        ('train_vif_max', 'VIF máximo', 'Train'), ('train_condition_number', 'Número de condición', 'Train'),
        ('train_bp_lm_pvalue_log', 'Breusch–Pagan (p)', 'Train'),
        ('train_jarque_bera_pvalue_log', 'Jarque–Bera (p)', 'Train'),
        ('selection_rmse_validation_original', 'RMSE (RUB)', 'Validación interna')]:
        nuevo = resultado['metricas'].get(clave)
        fila = {'Evaluación': evaluacion, 'Métrica': etiqueta}
        if anterior:
            previo = anterior['metricas'].get(clave)
            fila[anterior.get('etiqueta', 'Forward')] = b._formato_presentacion(previo) if previo is not None else 'N/D'
        etiqueta_modelo = resultado.get('etiqueta') or ('Backward Selection' if resultado.get('metodo') == 'backward'
                          else 'Group Lasso + OLS' if anterior else 'Forward')
        fila[etiqueta_modelo] = b._formato_presentacion(nuevo) if nuevo is not None else 'N/D'
        if anterior:
            fila['Δ nuevo − anterior'] = b._formato_presentacion(nuevo-previo) if nuevo is not None and previo is not None else 'N/D'
        filas.append(fila)
    return b.tabla_markdown(pd.DataFrame(filas))


def interpretar_presentacion(resultado, anterior=None):
    m = resultado['metricas']
    seleccionadas = resultado['seleccion']['seleccionadas']
    partes = [f"El OLS utiliza {len(seleccionadas)} características originales y {int(m['train_n_predictors'])} columnas predictoras, más el intercepto.",
        f"R² ajustado log={m['train_r2_adj_log']:.5f}; RMSE Train={m['train_rmse_original']:,.0f} RUB y validación interna={m['selection_rmse_validation_original']:,.0f} RUB."]
    if anterior:
        p = anterior['metricas']
        d = 100*(m['selection_rmse_validation_original']/p['selection_rmse_validation_original']-1)
        cambio = len(seleccionadas)-len(anterior['seleccion']['seleccionadas'])
        partes.append(f"Frente a Forward, cambia {cambio:+d} características y el RMSE interno {d:+.3f} %; "
            +( "no simplifica el modelo." if cambio > 0 else "reduce su tamaño." if cambio < 0 else "mantiene su tamaño."))
    partes += [f"Persisten límites de homocedasticidad (BP p={m['train_bp_lm_pvalue_log']:.2e}) y normalidad (JB p={m['train_jarque_bera_pvalue_log']:.2e}).",
        "La validación se utilizó para seleccionar: sus métricas no son una evaluación independiente. exp sin corrección puede sesgar el precio esperado."]
    return '  \n'.join(partes)


def interpretar_importancia(resultado):
    orden = resultado['importancias'].groupby('caracteristica').importancia_pct.median().sort_values(ascending=False)
    top = ', '.join(f"`{g}` ({v:.1f} %)" for g, v in orden.head(3).items())
    repeticiones = resultado['importancias'].groupby('caracteristica').size().iloc[0]
    return (f"Mayores medianas de incremento del RMSE: {top}; {repeticiones} permutaciones por grupo.  \n"
            "Las columnas de cada grupo se permutan conjuntamente respecto de price_rub; no mide efectos causales ni magnitudes de coeficientes OLS.  \n"
            "Se reutiliza la validación de selección; correlaciones entre características pueden afectar la importancia.")


def confirmacion_presentacion(resultado):
    run = resultado['run']
    accion = 'Recuperado de MLflow; sin nueva selección ni ajuste final' if resultado['modo'] == 'recuperar' else 'Selección y ajuste OLS nuevos'
    return f"{accion} · **FINISHED** · [{run['run_name']}]({b.TRACKING_URI}/#/experiments/{run['experiment_id']}/runs/{run['run_id']})"


def comparacion_final_presentacion(forward, lasso, backward=None, forward_177=None, lasso_177=None, bss=None):
    filas = []
    resultados = [('Forward Selection', forward), ('Group Lasso + OLS', lasso)]
    if backward is not None:
        resultados.append(('Backward Selection', backward))
    if forward_177 is not None:
        resultados[0] = ('Forward Selection — 174 predictoras', forward)
        resultados.append(('Forward Selection — 177 predictoras', forward_177))
    if lasso_177 is not None:
        resultados[1] = ('Group Lasso + OLS — 174 predictoras', lasso)
        resultados.append(('Group Lasso + OLS — 177 predictoras', lasso_177))
    if bss is not None:
        resultados.append(('Best Subset Selection — unión', bss))
    for nombre, r in resultados:
        m = r['metricas']
        filas.append({'Método': nombre, 'Características': len(r['seleccion']['seleccionadas']),
            'Columnas OLS': int(m['train_n_predictors']), 'R² ajustado Train [log]': f"{m['train_r2_adj_log']:.6f}",
            **{etiqueta: b._formato_presentacion(m[k]) for k, etiqueta in [
                ('train_rmse_original', 'RMSE Train (RUB)'), ('selection_rmse_validation_original', 'RMSE interno (RUB)'),
                ('train_aic_log', 'AIC [log]'), ('train_bic_log', 'BIC [log]')]}})
        if forward_177 is not None or lasso_177 is not None:
            filas[-1].update({'Columnas candidatas': sum(map(len, r['grupos'].values())),
                              'Grupos candidatos': len(r['grupos'])})
    a, bgr = set(forward['seleccion']['seleccionadas']), set(lasso['seleccion']['seleccionadas'])
    lista = lambda conjunto: ', '.join(f'`{g}`' for g in sorted(conjunto)) or 'Ninguna'
    coincidencias = f"Coinciden: {lista(a & bgr)}.  \nExclusivas de Forward: {lista(a-bgr)}.  \nExclusivas de Group Lasso: {lista(bgr-a)}."
    if backward is not None:
        conjuntos = {nombre: set(r['seleccion']['seleccionadas']) for nombre, r in resultados}
        comunes = set.intersection(*conjuntos.values())
        partes = [f'Comunes a los tres: {lista(comunes)}.']
        for nombre, conjunto in conjuntos.items():
            otros = set.union(*(v for k, v in conjuntos.items() if k != nombre))
            partes.append(f'Exclusivas de {nombre}: {lista(conjunto-otros)}.')
        for i, (nombre, conjunto) in enumerate(conjuntos.items()):
            for otro, segundo in list(conjuntos.items())[i+1:]:
                partes.append(f'Compartidas solo por {nombre} y {otro}: {lista((conjunto & segundo)-comunes)}.')
        coincidencias = '  \n'.join(partes)
    if forward_177 is not None or lasso_177 is not None:
        conjuntos = {nombre: set(r['seleccion']['seleccionadas']) for nombre, r in resultados}
        firmas = {}
        for g in sorted(set.union(*conjuntos.values())):
            firma = tuple(nombre for nombre, conjunto in conjuntos.items() if g in conjunto)
            firmas.setdefault(firma, set()).add(g)
        todos = tuple(conjuntos)
        partes = [f"Comunes a los {len(resultados)}: {lista(firmas.get(todos, set()))}."]
        for nombre in conjuntos:
            partes.append(f"Exclusivas de {nombre}: {lista(firmas.get((nombre,), set()))}.")
        for firma, nombres in firmas.items():
            if 1 < len(firma) < len(resultados):
                partes.append(f"Compartidas solo por {' y '.join(firma)}: {lista(nombres)}.")
        coincidencias = '  \n'.join(partes)
    return b.tabla_markdown(pd.DataFrame(filas)), coincidencias


def conclusion_presentacion(forward, lasso, backward=None, forward_177=None, lasso_177=None, bss=None):
    if backward is not None or forward_177 is not None or lasso_177 is not None:
        resultados = [('Forward Selection', forward), ('Group Lasso + OLS', lasso)]
        if backward is not None:
            resultados.append(('Backward Selection', backward))
        if forward_177 is not None:
            resultados[0] = ('Forward Selection — 174 predictoras', forward)
            resultados.append(('Forward Selection — 177 predictoras', forward_177))
        if lasso_177 is not None:
            resultados[1] = ('Group Lasso + OLS — 174 predictoras', lasso)
            resultados.append(('Group Lasso + OLS — 177 predictoras', lasso_177))
        if bss is not None:
            resultados.append(('Best Subset Selection — unión', bss))
        minimo = min(r['metricas']['selection_rmse_validation_original'] for _, r in resultados)
        menor = min(len(r['seleccion']['seleccionadas']) for _, r in resultados)
        ganadores = ', '.join(n for n, r in resultados if r['metricas']['selection_rmse_validation_original'] == minimo)
        compactos = ', '.join(n for n, r in resultados if len(r['seleccion']['seleccionadas']) == menor)
        reducciones = '; '.join(f"{n}: {len(r['grupos'])} → {len(r['seleccion']['seleccionadas'])} grupos "
            f"({100*(1-len(r['seleccion']['seleccionadas'])/len(r['grupos'])):.1f} % de reducción), "
            f"{int(r['metricas']['train_n_predictors'])} columnas finales" for n, r in resultados)
        mayor = max(1-len(r['seleccion']['seleccionadas'])/len(r['grupos']) for _, r in resultados)
        mas_reducidos = ', '.join(n for n, r in resultados
            if np.isclose(1-len(r['seleccion']['seleccionadas'])/len(r['grupos']), mayor))
        universos = ('Forward histórico y Group Lasso parten de 174 columnas; Forward 177 y Backward, de 177. '
                     if forward_177 is not None else
                     'Forward y Group Lasso parten de 174 columnas; Backward parte de 177. ')
        if lasso_177 is not None:
            universos = ('Forward y Group Lasso históricos parten de 174 columnas; los nuevos Forward y Group Lasso '
                         'y Backward parten de 177. ')
        if bss is not None:
            universos += (f"BSS parte de una unión derivada de {len(bss['grupos'])} características / "
                f"{sum(map(len, bss['grupos'].values()))} columnas, no del mismo universo inicial. "
                'Esta unión fue construida usando selecciones que ya consultaron la validación; '
                'reutilizarla introduce riesgo de sesgo por selección y optimismo. ')
        return (f"{ganadores} obtuvo el menor RMSE interno ({minimo:,.0f} RUB). "
            f"El menor número de características finales corresponde a {compactos} ({menor} grupos); "
            f"la mayor reducción proporcional corresponde a {mas_reducidos}. {reducciones}. "
            "Menos grupos favorecen la interpretación, aunque los bloques OHE pueden conservar muchas columnas.  \n"
            + universos + "La comparación es descriptiva: "
            "las diferencias no pueden atribuirse exclusivamente al algoritmo. AIC, BIC y R² ajustado "
            "corresponden al objetivo logarítmico sobre las mismas filas de Train.  \n"
            "La validación interna participa en la selección y no constituye una evaluación independiente "
            "sobre datos nuevos. Test no se utilizó.")
    f, g = forward['metricas'], lasso['metricas']
    ganador = 'Group Lasso + OLS' if g['selection_rmse_validation_original'] < f['selection_rmse_validation_original'] else 'Forward Selection'
    compacto = 'Forward Selection' if len(forward['seleccion']['seleccionadas']) < len(lasso['seleccion']['seleccionadas']) else 'Group Lasso + OLS'
    return (f"{ganador} obtuvo el menor RMSE en la validación interna compartida; {compacto} logró la mayor reducción de características.  \n"
            "AIC y BIC son comparables porque ambos OLS ajustan ln(price_rub) sobre las mismas filas con la misma convención.  \n"
            "La comparación reutiliza la validación empleada en la selección; no prueba superioridad fuera de muestra ni define el modelo final. Test no se utilizó.")


def comparacion_177_presentacion(forward, lasso, backward=None):
    """Comparación descriptiva con universo, partición y escalas constantes."""
    tabla, coincidencias = comparacion_final_presentacion(forward, lasso, backward)
    resultados = [('Forward — 177', forward), ('Group Lasso + OLS — 177', lasso)]
    if backward is not None:
        resultados.append(('Backward — 177', backward))
    especiales = []
    for g in EXCLUIDAS:
        estados = '; '.join(f"{nombre}: {'sí' if g in r['seleccion']['seleccionadas'] else 'no'}"
                            for nombre, r in resultados)
        especiales.append(f'`{g}` — {estados}.')
    menor = min(r['metricas']['selection_rmse_validation_original'] for _, r in resultados)
    ganadores = ', '.join(nombre for nombre, r in resultados
                         if r['metricas']['selection_rmse_validation_original'] == menor)
    analisis = '  \n'.join(especiales + [f'{ganadores} obtuvo el menor RMSE interno ({menor:,.0f} RUB). '
        'El universo inicial de 177 columnas y la partición son compartidos. '
        'La validación participó en la selección; esta comparación no constituye una evaluación independiente sobre datos nuevos.'])
    return tabla, coincidencias, analisis


def comparacion_forward_presentacion(historico, nuevo):
    """Mismo algoritmo y partición; distingue el universo habilitado del modelo final."""
    tabla, _ = comparacion_final_presentacion(historico, nuevo)
    tabla = tabla.replace('Forward Selection', 'Forward Stepwise Selection — 174 predictoras').replace(
        'Group Lasso + OLS', 'Forward Stepwise Selection — 177 predictoras')
    a, n = set(historico['seleccion']['seleccionadas']), set(nuevo['seleccion']['seleccionadas'])
    lista = lambda valores: ', '.join(f'`{g}`' for g in sorted(valores)) or 'Ninguna'
    lineas = [f"Universos iniciales: 174 columnas / {len(historico['grupos'])} grupos frente a "
        f"177 columnas / {len(nuevo['grupos'])} grupos. La tabla cuenta las características y columnas finales.",
        f"Comunes: {lista(a & n)}. Exclusivas del histórico: {lista(a-n)}. Exclusivas del nuevo: {lista(n-a)}."]
    for variable in EXCLUIDAS:
        paso = next((p['iteracion'] for p in nuevo['seleccion']['historial']
                     if p['caracteristica_incorporada'] == variable), None)
        lineas.append(f"`{variable}`: " + (f"incorporada en la iteración {paso}." if paso else
                                          "disponible entre las candidatas, no seleccionada."))
    previo = historico['metricas']['selection_rmse_validation_original']
    actual = nuevo['metricas']['selection_rmse_validation_original']
    diferencias = [len(n)-len(a),
        nuevo['metricas']['train_n_predictors']-historico['metricas']['train_n_predictors']]
    diferencias += [nuevo['metricas'][k]-historico['metricas'][k] for k in
        ('train_r2_adj_log', 'train_rmse_original', 'selection_rmse_validation_original',
         'train_aic_log', 'train_bic_log')]
    tabla += '\n| Δ 177 − 174 | ' + ' | '.join(b._formato_presentacion(v) for v in diferencias) + ' |'
    delta, pct = actual-previo, 100*(actual/previo-1)
    lineas.append(f"Cambio del RMSE interno: {delta:+,.3f} RUB ({pct:+.3f} %). "
        + ('Habilitar las tres variables reduce el error observado de selección. ' if delta < -1e-6 else
           'Habilitar las tres variables aumenta el error observado de selección. ' if delta > 1e-6 else
           'Habilitar las tres variables no modifica el error observado de selección. ')
        + ('El conjunto seleccionado cambia.' if a != n else 'El conjunto seleccionado se mantiene.'))
    lineas.append('La validación interna se utilizó para seleccionar los modelos; no es una evaluación independiente sobre datos nuevos. Test no se utilizó.')
    return tabla, '  \n'.join(lineas)


def recuperar_experimento(cliente, *, nombre=None, run_id=None,
                         experimento=b.EXPERIMENTO, parametros=None, etiquetas=None):
    """Recupera todas las métricas remotas; ante ambigüedad exige un ID explícito."""
    exp = cliente.get_experiment_by_name(experimento)
    if exp is None or exp.lifecycle_stage != 'active' or not (nombre or run_id):
        raise ValueError('Se requiere experimento existente activo y nombre o Run ID.')
    if run_id:
        candidatos = [cliente.get_run(run_id)]
    else:
        candidatos, token = [], None
        while True:
            pagina = cliente.search_runs([exp.experiment_id], max_results=1000, page_token=token)
            candidatos.extend(r for r in pagina if r.data.tags.get('mlflow.runName') == nombre)
            token = pagina.token
            if not token:
                break
    validos = [r for r in candidatos if r.info.experiment_id == exp.experiment_id
               and r.info.status == 'FINISHED'
               and (nombre is None or r.data.tags.get('mlflow.runName') == nombre)
               and all(r.data.params.get(k) == str(v) for k, v in (parametros or {}).items())
               and all(r.data.tags.get(k) == str(v) for k, v in (etiquetas or {}).items())]
    if len(validos) != 1:
        detalle = [{'run_id': r.info.run_id, 'status': r.info.status, 'params': r.data.params}
                   for r in candidatos]
        raise ValueError(f'Referencia inexistente, incompatible o ambigua; indicar Run ID: {detalle}')
    r = validos[0]
    return {'run_id': r.info.run_id, 'experiment_id': exp.experiment_id,
            'run_name': r.data.tags['mlflow.runName'], 'status': r.info.status,
            'metrics': dict(r.data.metrics), 'params': dict(r.data.params), 'tags': dict(r.data.tags)}


def grupos_caracteristicas(columnas, resumen, *, excluidas=EXCLUIDAS):
    """Linaje real: bloques One-Hot e indicador de imputación junto con su origen.

    months_since_start corresponde a date_posted_dt; derivadas multientrada
    como surface_corrected conservan su identidad exportada.
    """
    columnas = list(columnas)
    if len(columnas) != len(set(columnas)) or set(excluidas) & set(columnas):
        raise ValueError('Columnas duplicadas o exclusiones no respetadas.')
    mapa = {c: g['variable'] for g in resumen['categorias'] for c in g['columnas']}
    grupos = {}
    for c in columnas:
        info = resumen['linaje'][c]
        if info['tipo'] == 'onehot' and c not in mapa:
            raise ValueError(f'Faltan metadatos One-Hot: {c}')
        origen = info['entradas'][0] if len(info['entradas']) == 1 else info['variable']
        grupos.setdefault(mapa.get(c, origen), []).append(c)
    for g in resumen['categorias']:
        presentes = set(g['columnas']) & set(columnas)
        if presentes and (presentes != set(g['columnas']) or
                          g['referencia'] not in g['categorias'] or
                          len(g['columnas']) != len(g['categorias'])-1):
            raise ValueError(f"One-Hot parcial o sin referencia compatible: {g['variable']}")
    assert len(columnas) == sum(map(len, grupos.values()))
    assert set(columnas) == {c for cols in grupos.values() for c in cols}
    return grupos


def columnas_grupos(grupos, seleccionadas):
    if len(set(seleccionadas)) != len(seleccionadas):
        raise ValueError('Grupos repetidos.')
    return [c for g in seleccionadas for c in grupos[g]]


def forward_grupos(X, y_log, y_original, grupos, fit, val, *,
                   min_mejora_pct=0.5, max_features=12):
    """Una partición fija; OLS pinv con intercepto; exp sin corrección en validación."""
    if (not isinstance(max_features, int) or not 1 <= max_features <= 12 or
            not np.isfinite(min_mejora_pct) or min_mejora_pct < 0):
        raise ValueError('Mejora mínima no negativa y máximo absoluto de 12 características.')
    assert X.index.equals(y_log.index) and X.index.equals(y_original.index)
    np.testing.assert_allclose(y_log, np.log(y_original), rtol=1e-12)
    assert not set(fit) & set(val) and set(fit) | set(val) == set(range(len(X)))
    assert len(fit) > 1 and len(val) > 1
    assert len(X.columns) == sum(map(len, grupos.values()))
    assert set(X) == set(columnas_grupos(grupos, list(grupos)))
    posiciones = {c: i+1 for i, c in enumerate(X)}
    ids = {g: [posiciones[c] for c in cols] for g, cols in grupos.items()}
    matriz = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    ajuste, validacion = matriz[fit], matriz[val]
    objetivo, real = y_log.to_numpy()[fit], y_original.to_numpy()[val]
    seleccionadas, historial, candidatos = [], [], []
    evaluados = 0

    def evaluar(nombres):
        indices = [0] + [i for g in nombres for i in ids[g]]
        modelo = b.sm.OLS(objetivo, ajuste[:, indices], missing='raise').fit()
        with np.errstate(over='ignore'):
            pred = np.exp(modelo.predict(validacion[:, indices]))
            error = np.sqrt(np.mean(np.square(real-pred)))
        return float(error) if np.isfinite(error) else np.inf, int(modelo.model.rank)

    with threadpool_limits(limits=1):
        actual, _ = evaluar([])
        inicial = actual
        motivo = 'candidatas_agotadas'
        for iteracion in range(1, min(max_features, len(grupos))+1):
            mejores = []
            for g in grupos:
                if g in seleccionadas:
                    continue
                score, rango = evaluar(seleccionadas+[g])
                evaluados += 1
                mejores.append((score, g))
                candidatos.append({'iteracion': iteracion, 'candidata': g,
                    'rmse_validation_original': score if np.isfinite(score) else None,
                    'rango_efectivo': rango})
            score, g = min(mejores, key=lambda par: par[0])  # Empates: orden del linaje.
            mejora = 100*(actual-score)/actual if actual > 0 else -np.inf
            aceptar = bool(np.isfinite(score) and mejora >= min_mejora_pct)
            acumuladas = seleccionadas+[g] if aceptar else seleccionadas.copy()
            historial.append({'iteracion': iteracion, 'caracteristica_incorporada': g if aceptar else None,
                'mejor_candidata': g, 'aceptada': aceptar, 'caracteristicas_acumuladas': acumuladas,
                'rmse_validacion': score if aceptar else actual,
                'rmse_mejor_candidato': score if np.isfinite(score) else None,
                'mejora_pct': mejora if np.isfinite(mejora) else None,
                'n_predictores': len(columnas_grupos(grupos, acumuladas))})
            if not aceptar:
                motivo = 'min_mejora_pct'
                break
            seleccionadas, actual = acumuladas, score
            if len(seleccionadas) == max_features:
                motivo = 'max_features'
    return {'seleccionadas': seleccionadas, 'historial': historial, 'candidatos': candidatos,
            'rmse_validation_original': actual, 'rmse_intercepto_original': inicial,
            'motivo_parada': motivo, 'iteraciones': len(historial), 'candidatos_evaluados': evaluados}


def tabla_comparativa(referencia, metricas, invalidas=(), *,
                      nombre_referencia=REFERENCIA, nombre_run=NOMBRE_RUN):
    """Solo Train comparable; N/D sin derivar ni completar métricas históricas."""
    etiquetas = {'train_n_features_original': 'Cantidad de características',
                 'train_n_predictors': 'Cantidad de predictores',
                 'train_condition_number': 'Condition Number'}
    filas = []
    for k in dict.fromkeys([*metricas, *invalidas, *referencia]):
        if not k.startswith('train_'):
            continue
        etiqueta = etiquetas.get(k)
        if etiqueta is None:
            dummy = b.tabla_resultados({k: 0.0}, {}, {'breusch_pagan': 'N/D'}, nombre_run=NOMBRE_RUN)
            etiqueta = dummy.splitlines()[2].split('|')[1].strip()
        anterior, nuevo = referencia.get(k), metricas.get(k)
        a = anterior is not None and np.isfinite(anterior)
        n = nuevo is not None and np.isfinite(nuevo)
        filas.append({'Métrica': etiqueta, nombre_referencia: repr(anterior) if a else 'N/D',
                      nombre_run: repr(nuevo) if n else 'N/D',
                      'Δ': repr(nuevo-anterior) if a and n else 'N/D'})
    return b.tabla_markdown(pd.DataFrame(filas))


def preparar_experimento(base, cliente, *, reference_run_id=None, carpeta_salida,
                        excluidas=EXCLUIDAS, referencia_nombre=REFERENCIA):
    """Recupera y verifica la referencia ANTES del primer ajuste Forward."""
    base = Path(base)
    X, y, contexto = b.cargar_train(base)
    X = X.drop(columns=list(excluidas))
    ref = recuperar_experimento(cliente, nombre=referencia_nombre, run_id=reference_run_id,
        parametros={'target_transform': 'ln(Target)', 'inverse_transform': 'exp',
                    'bias_correction': 'none', 'train_fraction': '1.0', 'n_train': len(y),
                    'intercepto': 'True', 'cov_type': 'nonrobust', 'solver': 'pinv',
                    **({'excluded_predictors': ','.join(excluidas)} if excluidas else {}),
                    'aic_bic_convention': 'statsmodels_gaussian_log_RUB_k_rank'},
        etiquetas={'dataset_sha256': contexto['dataset_sha256']})
    out = b.validar_temporal(carpeta_salida)
    out.mkdir(exist_ok=True)
    ref_dir = out/'referencia_modelo'
    ref_dir.mkdir(exist_ok=True)
    ctx = json.loads(Path(cliente.download_artifacts(ref['run_id'], 'contexto.json', str(ref_dir))).read_text(encoding='utf-8'))
    assert ctx['predictoras'] == list(X) and ref['metrics']['train_n_obs'] == len(y)
    assert ctx['dataset_sha256'] == contexto['dataset_sha256'] and ctx['transformacion_objetivo'] == 'ln'
    from .almacenamiento_mlflow import leer_json, RUN_TRANSFORMACION
    resumen = leer_json(cliente, RUN_TRANSFORMACION, 'transformacion/resumen_final.json')
    assert resumen['columnas_X'] == contexto['predictoras']
    grupos = grupos_caracteristicas(X.columns, resumen, excluidas=excluidas)
    for g in resumen['categorias']:
        bloque = X[g['columnas']]
        assert bloque.isin([0, 1]).all().all() and bloque.sum(axis=1).le(1).all()
    contexto.update(transformacion_objetivo='ln', unidad='ln(RUB)', reference_run_id=ref['run_id'],
                    predictoras_excluidas=list(excluidas), seleccion_caracteristicas='forward_stepwise_grouped')
    b.guardar_json(out/'referencia_mlflow.json', ref)
    b.guardar_json(out/'comparabilidad.json', {'dataset_sha256': contexto['dataset_sha256'],
        'reference_run_id': ref['run_id'], 'mismas_filas_train': True, 'target': 'ln(Target)',
        'inversa': 'exp sin corrección', 'aic_bic': 'statsmodels_gaussian_log_RUB_k_rank',
        'evaluacion': 'Train completo; validación de selección separada', 'metricas_historicas': 'MLflow exclusivamente'})
    b.guardar_json(out/'grupos_caracteristicas.json', {'grupos': grupos, 'categorias': resumen['categorias'],
        'linaje': {c: resumen['linaje'][c] for c in X}})
    return X, y, contexto, ref, grupos, out


def ejecutar_experimento(base, cliente, *, reference_run_id=None,
                        min_mejora_pct=0.5, max_features=12, random_state=42, carpeta_salida=None,
                        excluidas=EXCLUIDAS, referencia_nombre=REFERENCIA, nombre_run=NOMBRE_RUN):
    if carpeta_salida is None:
        from .almacenamiento_mlflow import SesionTemporal
        with SesionTemporal() as sesion:
            resultado = ejecutar_experimento(base, cliente, reference_run_id=reference_run_id,
                min_mejora_pct=min_mejora_pct, max_features=max_features, random_state=random_state,
                carpeta_salida=sesion.carpeta('forward'), excluidas=excluidas,
                referencia_nombre=referencia_nombre, nombre_run=nombre_run)
            resultado.pop('salida')  # El llamador recibe resultados en memoria, nunca una ruta borrada.
            return resultado
    opciones = {} if tuple(excluidas) == EXCLUIDAS and referencia_nombre == REFERENCIA else {
        'excluidas': excluidas, 'referencia_nombre': referencia_nombre}
    X, y, contexto, ref, grupos, out = preparar_experimento(base, cliente,
        reference_run_id=reference_run_id, carpeta_salida=carpeta_salida, **opciones)
    forward_177 = tuple(excluidas) == ()
    if forward_177:
        if X.shape[1] != 177 or referencia_nombre != REFERENCIA_177:
            raise ValueError('Forward sin exclusiones requiere 177 columnas y referencia OLS_02_Log_Target.')
        if nombre_run == NOMBRE_RUN:
            raise ValueError('El Forward de 177 requiere un nombre de Run independiente del histórico.')
        if cliente.search_runs([ref['experiment_id']],
                filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=1):
            raise ValueError('El Run ya existe; utilizar un nombre de Run nuevo.')
    y_log = np.log(y).rename('ln_price_rub')
    fit, val = train_test_split(np.arange(len(y)), test_size=0.2, random_state=random_state)
    np.savez_compressed(out/'particion_interna.npz', fit=fit, validation=val)
    seleccion = forward_grupos(X, y_log, y, grupos, fit, val,
                               min_mejora_pct=min_mejora_pct, max_features=max_features)
    if forward_177 and not seleccion['seleccionadas']:
        raise RuntimeError('Ninguna incorporación supera la mejora mínima; no registrar un ganador arbitrario.')
    columnas = columnas_grupos(grupos, seleccion['seleccionadas'])
    modelo, diseno = b.ajustar(X[columnas], y_log)
    assert modelo.model.rank == diseno.shape[1], 'Diseño final no identificable; no registrar.'
    assert len(y) == modelo.nobs and len(seleccion['seleccionadas']) <= 12
    metricas, invalidas, vif, notas = b.diagnosticar(modelo, diseno, y_log, y_original=y, resumir_vif=True)
    metricas.update(train_n_features_original=float(len(seleccion['seleccionadas'])),
                    selection_rmse_validation_original=seleccion['rmse_validation_original'],
                    selection_rmse_intercept_original=seleccion['rmse_intercepto_original'],
                    selection_iterations=float(seleccion['iteraciones']),
                    selection_candidates_evaluated=float(seleccion['candidatos_evaluados']))
    contexto.update(predictoras=columnas, n_predictoras=len(columnas),
                    caracteristicas_originales=seleccion['seleccionadas'])
    if forward_177:
        contexto.update(universo_predictoras=list(X), n_predictoras_inicial=177)
    b.exportar(base, modelo, diseno, y_log, contexto, metricas, invalidas, vif, notas,
               nombre_run=nombre_run, carpeta_salida=out)
    tabla = tabla_comparativa(ref['metrics'], metricas, invalidas,
        nombre_referencia=referencia_nombre, nombre_run=nombre_run)
    (out/'comparacion_referencia.md').write_text(tabla, encoding='utf-8')
    b.guardar_json(out/'seleccion.json', seleccion)
    b.guardar_json(out/'historial_seleccion.json', seleccion['historial'])
    b.guardar_json(out/'caracteristicas_seleccionadas.json', {g: grupos[g] for g in seleccion['seleccionadas']})
    parametros = {'algorithm': 'Forward Stepwise Selection agrupado', 'min_mejora_pct': min_mejora_pct,
        'max_features': max_features, 'random_state': random_state, 'validation_fraction': 0.2,
        'selection_metric': 'RMSE validation original RUB; exp sin corrección',
        'selection_stop_reason': seleccion['motivo_parada'], 'selection_iterations': seleccion['iteraciones'],
        'n_features_original': len(seleccion['seleccionadas']),
        'selected_features': json.dumps(seleccion['seleccionadas']),
        'feature_order': json.dumps(seleccion['seleccionadas']),
        'reference_run_name': referencia_nombre, 'reference_run_id': ref['run_id'],
        'experiment_id': ref['experiment_id'], 'notebook': '4_Seleccion_Modelos.ipynb'}
    if forward_177:
        parametros.update(_parametros_forward_177({'X': X, 'grupos': grupos}))
        b.guardar_json(out/'candidatos_evaluados.json', seleccion['candidatos'])
    b.guardar_json(out/'configuracion.json', parametros)
    # Evita elegir arbitrariamente el primero de varios Runs en el registrador histórico.
    existentes = cliente.search_runs([ref['experiment_id']],
        filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=2)
    if len(existentes) > 1 or (existentes and existentes[0].data.tags.get('dataset_sha256') != contexto['dataset_sha256']):
        raise ValueError('Runs Forward ambiguos; resolver sus IDs antes de registrar.')
    registro = b.registrar(cliente, out, contexto, metricas, nombre_run=nombre_run, parametros_extra=parametros)
    remoto = cliente.get_run(registro['run_id'])
    assert remoto.data.params['reference_run_id'] == ref['run_id']
    assert remoto.data.params['notebook'] == '4_Seleccion_Modelos.ipynb'
    assert all(remoto.data.metrics[k] == v for k, v in metricas.items())
    despues = cliente.get_run(ref['run_id'])
    assert despues.data.metrics == ref['metrics'] and despues.data.params == ref['params']
    return {'salida': out, 'comparacion_md': tabla, 'registro': registro,
            'seleccion': seleccion, 'metricas': metricas}


def restaurar_forward(cliente, run, salida, contexto_train, referencia):
    """Recupera evidencia del Run terminado y comprueba las copias locales existentes."""
    from tempfile import TemporaryDirectory
    out = b.validar_temporal(salida)
    out.mkdir(exist_ok=True)
    assert run['params']['reference_run_id'] == referencia['run_id']
    assert run['tags']['dataset_sha256'] == contexto_train['dataset_sha256']
    assert run['params']['target_transform'] == 'ln(Target)'
    assert run['params']['inverse_transform'] == 'exp' and run['params']['bias_correction'] == 'none'
    assert run['params']['excluded_predictors'] == ','.join(EXCLUIDAS)
    nombres = ['contexto.json', 'seleccion.json', 'grupos_caracteristicas.json', 'metricas.json',
               'particion_interna.npz', 'modelo_statsmodels.pkl', 'comparacion_referencia.md',
               *['diagnosticos_residuales/'+n for n in b.ARCHIVOS]]
    with TemporaryDirectory(prefix='forward_evidencia_') as temporal:
        for nombre in nombres:
            remoto = Path(cliente.download_artifacts(run['run_id'], nombre, temporal))
            local = out/nombre
            if local.exists():
                assert b.sha256(local) == b.sha256(remoto), f'Artefacto local distinto: {nombre}'
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remoto, local)
    leer = lambda nombre: json.loads((out/nombre).read_text(encoding='utf-8'))
    estado = {k: leer(k+'.json') for k in ['contexto', 'seleccion', 'grupos_caracteristicas', 'metricas']}
    assert estado['metricas'] == run['metrics']
    assert estado['contexto']['dataset_sha256'] == contexto_train['dataset_sha256']
    assert run['metrics']['train_n_obs'] == contexto_train['n_train']
    assert estado['seleccion']['seleccionadas'] == json.loads(run['params']['selected_features'])
    return estado


def resumen_caracteristicas(seleccionadas, grupos, modelo, *, mostrar_orden=True):
    """Una fila por característica original; cuenta predictores sin el intercepto."""
    columnas = columnas_grupos(grupos, seleccionadas)
    assert modelo.model.exog_names == ['const']+columnas
    efectivos = int(modelo.model.rank)-1
    assert efectivos == len(columnas), 'La tabla requiere un diseño final de rango completo.'
    filas = [{'Orden': i, 'Característica seleccionada': f'`{g}`',
              'Cantidad de predictores': len(grupos[g])} for i, g in enumerate(seleccionadas, 1)]
    filas.append({'Orden': 'Total', 'Característica seleccionada': f'{len(seleccionadas)} características',
                  'Cantidad de predictores': f'{efectivos} predictores'})
    if not mostrar_orden:
        filas[-1]['Característica seleccionada'] = f'Total: {len(seleccionadas)} características'
    tabla = pd.DataFrame(filas)
    return b.tabla_markdown(tabla if mostrar_orden else tabla.drop(columns='Orden'))


def importancia_permutacion_grupos(X, y_original, grupos, seleccionadas, fit, val, *,
                                   n_repeticiones=5, random_state=42, rmse_esperado=None):
    """Un ajuste interno; cada repetición permuta conjuntamente todas las columnas del grupo.

    Importancia descriptiva sobre la validación usada para seleccionar. No modifica
    el modelo final ni reajusta entre permutaciones. La correlación puede afectarla.
    """
    assert isinstance(n_repeticiones, int) and n_repeticiones > 0 and len(seleccionadas) > 0
    assert X.index.equals(y_original.index)
    assert not set(fit) & set(val) and set(fit) | set(val) == set(range(len(X)))
    assert len(fit)+len(val) == len(X)
    columnas = columnas_grupos(grupos, seleccionadas)
    modelo, _ = b.ajustar(X[columnas].iloc[fit], np.log(y_original.iloc[fit]))
    validacion = b.sm.add_constant(X[columnas].iloc[val], has_constant='add').to_numpy()
    real = y_original.iloc[val].to_numpy()
    rng, filas = np.random.default_rng(random_state), []
    posiciones = {c: i+1 for i, c in enumerate(columnas)}
    with threadpool_limits(limits=1):
        rmse = float(np.sqrt(np.mean((real-np.exp(modelo.predict(validacion)))**2)))
        assert np.isfinite(rmse) and rmse > 0
        if rmse_esperado is not None:
            np.testing.assert_allclose(rmse, rmse_esperado, rtol=1e-10, atol=1e-6)
        for g in seleccionadas:
            indices = [posiciones[c] for c in grupos[g]]
            permutada = validacion.copy()
            for repeticion in range(1, n_repeticiones+1):
                orden = rng.permutation(len(val))
                permutada[:, indices] = validacion[orden][:, indices]
                pred = np.exp(modelo.predict(permutada))
                score = float(np.sqrt(np.mean((real-pred)**2)))
                assert np.isfinite(score)
                filas.append({'caracteristica': g, 'repeticion': repeticion,
                    'rmse_original': rmse, 'rmse_permutado': score,
                    'incremento_rmse': score-rmse, 'importancia_pct': 100*(score-rmse)/rmse})
    return pd.DataFrame(filas)


def guardar_boxplot_importancia(importancias, archivo):
    b.validar_temporal(Path(archivo).parent)
    from matplotlib.figure import Figure
    orden = importancias.groupby('caracteristica')['importancia_pct'].median().sort_values(
        ascending=False, kind='stable').index.tolist()
    valores = [importancias.loc[importancias.caracteristica.eq(g), 'importancia_pct'].to_numpy() for g in orden]
    fig = Figure(figsize=(10, max(4, .55*len(orden)+1.6)), layout='constrained')
    ax = fig.subplots()
    ax.boxplot(valores, vert=False, tick_labels=orden, patch_artist=True,
               boxprops={'facecolor': '#b9dce9'}, medianprops={'color': '#b12c2c', 'linewidth': 1.7})
    ax.invert_yaxis()
    ax.axvline(0, color='#555555', linestyle='--', linewidth=1)
    ax.set(title='Permutation Importance agrupada · Validación interna de Train',
           xlabel='Incremento del RMSE (%)', ylabel='Característica original')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(archivo, dpi=160)


def registrar_presentacion(cliente, run, salida, archivos):
    """Agrega y verifica artefactos al mismo Run, conservando métricas, parámetros y modelo."""
    from tempfile import TemporaryDirectory
    antes = cliente.get_run(run['run_id'])
    assert antes.info.status == 'FINISHED' and antes.data.metrics == run['metrics']
    for nombre in archivos:
        cliente.log_artifact(run['run_id'], str(Path(salida)/nombre))
    with TemporaryDirectory(prefix='forward_presentacion_') as temporal:
        for nombre in archivos:
            remoto = cliente.download_artifacts(run['run_id'], nombre, temporal)
            assert b.sha256(remoto) == b.sha256(Path(salida)/nombre), nombre
        modelo = cliente.download_artifacts(run['run_id'], 'modelo_statsmodels.pkl', temporal)
        assert b.sha256(modelo) == b.sha256(Path(salida)/'modelo_statsmodels.pkl')
    despues = cliente.get_run(run['run_id'])
    assert despues.info.status == 'FINISHED'
    assert despues.data.metrics == antes.data.metrics and despues.data.params == antes.data.params
