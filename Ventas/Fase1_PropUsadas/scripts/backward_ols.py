"""Backward OLS por bloques completos; trayectoria y selección son etapas separadas."""
import json
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import baseline_ols as b
from . import seleccion_modelos as s

EXCLUIDAS = ()
NOMBRE_RUN = 'Backward_03_Log_177_Grupos_OLS'
REFERENCIA = 'OLS_02_Log_Target'
CONFIG = dict(s.CONFIG_FORWARD)
VERSION = 'backward_grouped_path_v1'


def backward_grupos(X, y, grupos, fit, val, *, min_mejora_pct=.5, max_features=12):
    """OLS sobre fit; se elimina el mejor bloque aun si aumenta el RMSE en RUB.

    Se conserva hasta el modelo de un grupo, sin detenerse en un mínimo local.
    Solo diseños internos identificables pueden ganar la selección por complejidad.
    No se descartan columnas para corregir colinealidad durante la trayectoria.
    """
    from .group_lasso_ols import elegir_por_complejidad
    if (not isinstance(max_features, int) or not 1 <= max_features <= 12
            or not np.isfinite(min_mejora_pct) or min_mejora_pct < 0):
        raise ValueError('Criterios de selección inválidos.')
    assert X.index.equals(y.index) and np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y).all() and (y > 0).all()
    assert len(fit) > 1 and len(val) > 1
    assert len(fit)+len(val) == len(X) and not set(fit) & set(val)
    assert set(fit) | set(val) == set(range(len(X)))
    columnas = s.columnas_grupos(grupos, list(grupos))
    assert len(columnas) == len(set(columnas)) == X.shape[1] and set(columnas) == set(X)
    posiciones = {c: i+1 for i, c in enumerate(X)}
    matriz = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    interno, validacion = matriz[fit], matriz[val]
    objetivo, real = np.log(y.to_numpy()[fit]), y.to_numpy()[val]

    def evaluar(nombres):
        indices = [0]+[posiciones[c] for c in s.columnas_grupos(grupos, nombres)]
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            modelo = b.sm.OLS(objetivo, interno[:, indices], missing='raise').fit()
            with np.errstate(over='ignore', invalid='ignore'):
                score = float(np.sqrt(np.mean((real-np.exp(modelo.predict(validacion[:, indices])))**2)))
        rango = int(modelo.model.rank)
        notas = [str(w.message) for w in avisos]
        if rango < len(indices):
            notas.append('Diseño de rango deficiente; OLS pinv conserva todas las columnas.')
        if not np.isfinite(score):
            notas.append('RMSE no finito; candidato no elegible para selección final.')
        return dict(seleccionadas=list(nombres), n_caracteristicas=len(nombres),
            n_predictores=len(indices)-1, rango_efectivo=rango,
            identificable=rango == len(indices), advertencias=notas,
            rmse_validation_original=score if np.isfinite(score) else None)

    with threadpool_limits(limits=1):
        intercepto = evaluar([])
        if intercepto['rmse_validation_original'] is None or intercepto['rmse_validation_original'] <= 0:
            raise ValueError('RMSE del intercepto inválido para calcular mejoras relativas.')
        presentes = list(grupos)
        trayectoria = [evaluar(presentes)]
        candidatos, historial = [], []
        for it in range(1, len(grupos)):
            opciones = []
            for g in presentes:
                candidato = dict(evaluar([n for n in presentes if n != g]),
                                 iteracion=it, caracteristica_eliminada=g)
                candidatos.append(candidato)
                opciones.append(candidato)
            elegido = min(opciones, key=lambda c: c['rmse_validation_original']
                          if c['rmse_validation_original'] is not None else np.inf)
            presentes = elegido['seleccionadas']
            historial.append(elegido.copy())
            trayectoria.append(elegido.copy())
    elegibles = [dict(c, **{'lambda': None}) for c in trayectoria
                 if c['identificable'] and c['rmse_validation_original'] is not None]
    ganador, decisiones = elegir_por_complejidad(elegibles, intercepto['rmse_validation_original'],
        min_mejora_pct=min_mejora_pct, max_features=max_features)
    # No se inventa un ganador si ningún tamaño alcanza el umbral.
    return dict(seleccionadas=ganador['seleccionadas'],
        rmse_validation_original=ganador['rmse_validation_original'],
        rmse_intercepto_original=intercepto['rmse_validation_original'],
        inicial=trayectoria[0], trayectoria=trayectoria, historial=historial, candidatos=candidatos,
        decisiones_complejidad=decisiones, iteraciones=len(historial),
        candidatos_evaluados=len(candidatos),
        motivo_parada='trayectoria_completa_y_criterio_complejidad' if ganador['seleccionadas']
                      else 'ningun_modelo_supera_umbral',
        config=dict(min_mejora_pct=min_mejora_pct, max_features=max_features, random_state=42))


def cargar_datos(datos):
    """Misma partición histórica, universo completo independiente de las exclusiones."""
    completo = s.cargar_presentacion(datos['base'], datos['sesion'], excluidas=EXCLUIDAS,
                                     cliente=datos['cliente'])
    assert completo['X'].shape[1] == completo['n_exportadas'] == 177
    for c in ('to_center_km', 'living_area', 'rooms'):
        assert c in s.columnas_grupos(completo['grupos'], list(completo['grupos']))
    for clave in ('fit', 'val'):
        np.testing.assert_array_equal(completo[clave], datos[clave])
    assert completo['contexto']['dataset_sha256'] == datos['contexto']['dataset_sha256']
    return completo


def ejecutar_experimento(datos, salida, nombre_run=NOMBRE_RUN):
    """Valida OLS_02 y el universo antes del primer ajuste; registra un Run nuevo."""
    assert datos['excluidas'] == EXCLUIDAS and datos['X'].shape[1] == 177
    X, y, ctx, ref, grupos, out = s.preparar_experimento(datos['base'], datos['cliente'],
        carpeta_salida=salida, excluidas=EXCLUIDAS, referencia_nombre=REFERENCIA)
    assert list(X) == list(datos['X']) and grupos == datos['grupos']
    cfg = datos['config_forward']
    if cfg != CONFIG:
        raise ValueError('Backward requiere la configuración fija 80/20, semilla 42, 12 grupos y 0,5 %.')
    seleccion = backward_grupos(X, y, grupos, datos['fit'], datos['val'],
        min_mejora_pct=cfg['min_mejora_pct'], max_features=cfg['max_features'])
    b.guardar_json(out/'seleccion.json', seleccion)
    for nombre, valor in [('historial_eliminaciones', seleccion['historial']),
                          ('candidatos_evaluados', seleccion['candidatos']),
                          ('decisiones_complejidad', seleccion['decisiones_complejidad'])]:
        b.guardar_json(out/(nombre+'.json'), valor)
    if not seleccion['seleccionadas']:
        raise RuntimeError('Ningún modelo supera la mejora mínima; no se registra un ganador.')
    columnas = s.columnas_grupos(grupos, seleccion['seleccionadas'])
    modelo, diseno = b.ajustar(X[columnas], np.log(y))
    assert modelo.model.rank == diseno.shape[1], 'OLS definitivo no identificable; no registrar.'
    metricas, invalidas, vif, notas = b.diagnosticar(modelo, diseno, np.log(y),
                                                  y_original=y, resumir_vif=True)
    metricas.update(train_n_features_original=float(len(seleccion['seleccionadas'])),
        selection_rmse_validation_original=seleccion['rmse_validation_original'],
        selection_rmse_intercept_original=seleccion['rmse_intercepto_original'],
        selection_iterations=float(seleccion['iteraciones']),
        selection_candidates_evaluated=float(seleccion['candidatos_evaluados']))
    ctx.update(predictoras=columnas, n_predictoras=len(columnas),
        caracteristicas_originales=seleccion['seleccionadas'],
        seleccion_caracteristicas='backward_stepwise_grouped',
        universo_predictoras=list(X), n_predictoras_inicial=177)
    b.exportar(datos['base'], modelo, diseno, np.log(y), ctx, metricas, invalidas, vif, notas,
               nombre_run=nombre_run, carpeta_salida=out)
    # Formato común de recuperación para Lasso y Backward: diccionario de grupos.
    b.guardar_json(out/'grupos_caracteristicas.json', grupos)
    b.guardar_json(out/'universo_linaje.json', dict(columnas=list(X), grupos=grupos))
    b.guardar_json(out/'caracteristicas_seleccionadas.json',
                   {g: grupos[g] for g in seleccion['seleccionadas']})
    np.savez_compressed(out/'particion_interna.npz', fit=datos['fit'], validation=datos['val'])
    (out/'comparacion_referencia.md').write_text(s.tabla_comparativa(ref['metrics'], metricas, invalidas,
        nombre_referencia=REFERENCIA, nombre_run=nombre_run), encoding='utf-8')
    params = {**s._parametros_presentacion(datos, 'backward'),
        'algorithm': 'Backward Stepwise Selection agrupado', 'algorithm_version': VERSION,
        'n_candidate_predictors': 177, 'n_candidate_groups': len(grupos),
        'candidate_columns': json.dumps(list(X)),
        'reference_run_name': REFERENCIA, 'reference_run_id': ref['run_id'],
        'selected_features': json.dumps(seleccion['seleccionadas']),
        'selection_metric': 'RMSE validation original RUB; exp sin corrección',
        'selection_stop_reason': seleccion['motivo_parada'], 'permutation_repeats': 10,
        'notebook': '4_Seleccion_Modelos.ipynb'}
    b.guardar_json(out/'configuracion.json', params)
    # El código específico forma parte de los artefactos verificados del Run.
    from pathlib import Path
    (out/'backward_ols.py').write_bytes(Path(__file__).read_bytes())
    registro = b.registrar(datos['cliente'], out, ctx, metricas, nombre_run=nombre_run,
                            parametros_extra=params)
    despues = datos['cliente'].get_run(ref['run_id'])
    assert despues.data.metrics == ref['metrics'] and despues.data.params == ref['params']
    return dict(registro=registro, seleccion=seleccion, metricas=metricas)


def validar_recuperacion(datos, resultado):
    """Comprueba identidad del universo, referencia, trayectoria y criterio de ganador."""
    from .group_lasso_ols import elegir_por_complejidad
    p, sel = resultado['run']['params'], resultado['seleccion']
    assert p['algorithm_version'] == VERSION and p['reference_run_name'] == REFERENCIA
    assert json.loads(p['candidate_columns']) == list(datos['X'])
    assert int(p['n_candidate_groups']) == len(datos['grupos'])
    ref = s.recuperar_experimento(datos['cliente'], nombre=REFERENCIA, run_id=p['reference_run_id'],
        etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
    assert ref['params']['n_predictors'] == '177'
    assert resultado['contexto']['reference_run_id'] == ref['run_id']
    assert resultado['contexto']['universo_predictoras'] == list(datos['X'])
    assert sel['config'] == CONFIG
    ruta = sel['trayectoria']
    assert ruta[0] == sel['inicial'] and ruta[0]['seleccionadas'] == list(datos['grupos'])
    assert ruta[0]['n_predictores'] == 177 and len(ruta) == len(datos['grupos'])
    assert sel['historial'] == ruta[1:] and sel['iteraciones'] == len(ruta)-1
    candidatos = sel['candidatos']
    assert len(candidatos) == sel['candidatos_evaluados'] == sum(range(2, len(ruta)+1))
    for it, (previo, actual) in enumerate(zip(ruta, ruta[1:]), 1):
        opciones = [c for c in candidatos if c['iteracion'] == it]
        assert [c['caracteristica_eliminada'] for c in opciones] == previo['seleccionadas']
        for c in opciones:
            assert c['seleccionadas'] == [g for g in previo['seleccionadas'] if g != c['caracteristica_eliminada']]
            assert c['n_predictores'] == len(s.columnas_grupos(datos['grupos'], c['seleccionadas']))
        assert actual == min(opciones, key=lambda c: c['rmse_validation_original']
                             if c['rmse_validation_original'] is not None else np.inf)
    elegibles = [dict(c, **{'lambda': None}) for c in ruta
                 if c['identificable'] and c['rmse_validation_original'] is not None]
    ganador, decisiones = elegir_por_complejidad(elegibles, sel['rmse_intercepto_original'],
        min_mejora_pct=CONFIG['min_mejora_pct'], max_features=CONFIG['max_features'])
    assert decisiones == sel['decisiones_complejidad'] and ganador['seleccionadas'] == sel['seleccionadas']
    assert ganador['seleccionadas'] and ganador['rmse_validation_original'] == sel['rmse_validation_original']


def resumen_trayectoria(resultado):
    sel = resultado['seleccion']
    inicial, ruta = sel['inicial'], sel['trayectoria']
    eliminadas = ', '.join('`'+c['caracteristica_eliminada']+'`' for c in sel['historial'])
    errores = [c['rmse_validation_original'] for c in ruta if c['rmse_validation_original'] is not None]
    return (f"Inicio: {inicial['n_caracteristicas']} características originales, {inicial['n_predictores']} columnas "
        f"más intercepto; rango efectivo {inicial['rango_efectivo']}/{inicial['n_predictores']+1}. "
        f"Advertencias iniciales: {'; '.join(inicial['advertencias']) or 'ninguna'}.  \n"
        f"Trayectoria completa: {sel['iteraciones']} eliminaciones y {sel['candidatos_evaluados']} candidatos. "
        f"Orden de eliminación: {eliminadas}.  \n"
        f"RMSE interno: inicial {errores[0]:,.0f}, mínimo de trayectoria {min(errores):,.0f}, "
        f"un grupo {errores[-1]:,.0f} RUB. Seleccionado: {sel['rmse_validation_original']:,.0f} RUB.  \n"
        "Se evaluaron los tamaños identificables de 1 a 12 en orden creciente, desde el intercepto; "
        "cada aceptación exige una mejora ≥0,5 % frente al último aceptado. Un rechazo no detiene la evaluación.")


def tabla_trayectoria(resultado):
    """Resumen compacto de los tamaños elegibles; el detalle completo queda en MLflow."""
    filas = [dict(Grupos=c['n_caracteristicas'], Columnas=c['n_predictores'],
                  **{'RMSE interno (RUB)': b._formato_presentacion(c['rmse_validation_original']),
                     'Rango': c['rango_efectivo'],
                     'Aceptado': next((d['aceptado'] for d in resultado['seleccion']['decisiones_complejidad']
                                       if d['n_caracteristicas'] == c['n_caracteristicas']), False)})
             for c in reversed(resultado['seleccion']['trayectoria']) if c['n_caracteristicas'] <= 12]
    return b.tabla_markdown(pd.DataFrame(filas))
