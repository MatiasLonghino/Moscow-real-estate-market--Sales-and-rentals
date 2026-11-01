"""Group Lasso ponderado para seleccionar bloques; el estimador final es OLS sin penalizar.

Objetivo: ||y_c-Zb||²/(2n) + lambda * sum_g sqrt(rank(Z_g))*||b_g||₂.
Z se estandariza exclusivamente en fit; intercepto no penalizado mediante centrado.
FISTA opera sobre Gram (p x p), con reinicio adaptativo y control de estacionariedad.
Referencia: https://group-lasso.readthedocs.io/en/latest/maths.html
"""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from . import baseline_ols as b
from . import seleccion_modelos as s
NOMBRE_RUN = 'GroupLasso_02_Log_Grupos_OLS'
NOMBRE_RUN_177 = 'GroupLasso_05_Log_177_Grupos_OLS'
CONFIG = {'min_mejora_pct': .5, 'max_features': 12, 'support_tol': 1e-6,
          'solver_tol': 1e-7, 'max_iter': 20000, 'n_lambdas': 40, 'lambda_min_ratio': .001}


def parametros_177(datos, referencia):
    return dict(n_candidate_predictors=177, n_candidate_groups=len(datos['grupos']),
        candidate_columns=json.dumps(list(datos['X'])), excluded_predictors='',
        reference_run_name=s.NOMBRE_RUN_177, reference_run_id=referencia['run']['run_id'],
        n_lambdas=40, lambda_min_ratio=.001, permutation_repeats=10)


def validar_referencia_177(datos, referencia):
    """El Forward real de 177 debe estar terminado y compartir universo y partición."""
    if referencia is None or referencia.get('metodo') != 'forward':
        raise ValueError('Group Lasso 177 requiere Forward_04_Log_177_Grupos terminado.')
    s._validar_resultado_presentacion(datos, referencia)
    for k, v in s._parametros_forward_177(datos).items():
        if referencia['run']['params'].get(k) != str(v):
            raise ValueError(f'Referencia Forward 177 incompatible ({k}).')
    s._validar_forward_177(datos, referencia)
    remoto = s.recuperar_experimento(datos['cliente'], nombre=s.NOMBRE_RUN_177,
        run_id=referencia['run']['run_id'], parametros=s._parametros_forward_177(datos),
        etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
    assert remoto['params'] == referencia['run']['params'] and remoto['metrics'] == referencia['run']['metrics']


def diagnostico_gram(gram, bloques, columnas):
    """Audita el diseño sin modificarlo ni ajustar el umbral histórico de rango."""
    valores, vectores = np.linalg.eigh(gram)
    rango = int(np.linalg.matrix_rank(gram, tol=1e-10))
    return dict(n_columnas=len(columnas), rango_gram=rango, tolerancia_rango=1e-10,
        autovalor_min=float(valores[0]), autovalor_max=float(valores[-1]),
        condicion_gram=float(valores[-1]/valores[0]) if valores[0] > 0 else None,
        rangos_grupos=[int(np.linalg.matrix_rank(gram[np.ix_(i, i)], tol=1e-10)) for i in bloques],
        direccion_menos_estable={columnas[i]: float(vectores[i, 0])
            for i in np.argsort(abs(vectores[:, 0]))[-8:][::-1]},
        advertencias=[] if rango == len(columnas) else
            ['Gram de rango deficiente; no excluir columnas ni continuar con resultados incompletos.'])


def validar_recuperacion_177(datos, resultado):
    """Audita escalador, 40 soluciones KKT, bloques y decisiones; no reentrena."""
    sel, out = resultado['seleccion'], resultado['salida']
    X, y, grupos, fit = datos['X'], datos['y'], datos['grupos'], datos['fit']
    assert resultado['contexto']['universo_predictoras'] == list(X)
    assert resultado['contexto']['n_predictoras_inicial'] == X.shape[1] == 177
    scaler = joblib.load(out/'scaler_seleccion.joblib')
    assert list(scaler.feature_names_in_) == list(X) and scaler.n_samples_seen_ == len(fit)
    np.testing.assert_allclose(scaler.mean_, X.iloc[fit].mean(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, X.iloc[fit].var(ddof=0), rtol=1e-12, atol=1e-12)
    with np.load(out/'ruta_coeficientes_group_lasso.npz') as ruta:
        assert ruta['columnas'].tolist() == list(X)
        coef, ids = ruta['coeficientes'].copy(), ruta['group_ids'].copy()
        np.testing.assert_array_equal(ruta['lambdas'], sel['lambdas'])
    nombres = list(grupos)
    posiciones = {c: i for i, c in enumerate(X)}
    bloques = [np.array([posiciones[c] for c in grupos[g]]) for g in nombres]
    for i, bloque in enumerate(bloques):
        assert (ids[bloque] == i).all()
    assert coef.shape == (40, 177) and np.isfinite(coef).all()
    assert len(sel['historial']) == len(sel['lambdas']) == 40 and sel['config'] == CONFIG
    candidatos, cache = [], {}
    with threadpool_limits(limits=1):
        Z = scaler.transform(X.iloc[fit])
        yc = np.log(y.iloc[fit]).to_numpy()-np.log(y.iloc[fit]).mean()
        gram, xy = Z.T@Z/len(fit), Z.T@yc/len(fit)
        diag = diagnostico_gram(gram, bloques, list(X))
        meta = json.loads((out/'estabilidad_numerica.json').read_text(encoding='utf-8'))
        assert diag['rango_gram'] == meta['rango_gram'] == 177
        assert diag['rangos_grupos'] == meta['rangos_grupos'] == [len(i) for i in bloques]
        np.testing.assert_allclose([meta[k] for k in ('autovalor_min', 'autovalor_max', 'condicion_gram')],
            [diag[k] for k in ('autovalor_min', 'autovalor_max', 'condicion_gram')], rtol=1e-10)
        pesos = np.sqrt(diag['rangos_grupos'])
        for g, r, w in zip(nombres, diag['rangos_grupos'], pesos):
            assert sel['ponderacion'][g] == dict(p_g=r, peso=float(w))
        np.testing.assert_allclose(sel['lambda_max'], max(np.linalg.norm(xy[i])/w for i, w in zip(bloques, pesos)))
        L = float(np.linalg.eigvalsh(gram)[-1])
        escala = max(float(np.linalg.norm(xy, ord=np.inf)), 1e-12)
        for lam, beta, paso in zip(sel['lambdas'], coef, sel['historial']):
            normas = [float(np.linalg.norm(beta[i])) for i in bloques]
            activas = [g for g, norma in zip(nombres, normas) if norma > CONFIG['support_tol']]
            assert paso['lambda'] == lam and paso['seleccionadas'] == activas
            np.testing.assert_allclose(list(paso['normas_grupos'].values()), normas, rtol=1e-12, atol=1e-14)
            assert paso['n_caracteristicas'] == len(activas)
            assert paso['n_predictores'] == len(s.columnas_grupos(grupos, activas))
            valido = 0 < len(activas) <= 12
            assert paso['valido'] == valido and paso['convergencia']
            assert paso['motivo'] == ('evaluado' if valido else 'solo_intercepto' if not activas else 'max_features')
            assert 1 <= paso['iteraciones_solver'] <= CONFIG['max_iter']
            v = beta-(gram@beta-xy)/L
            n = np.sqrt(np.bincount(ids, weights=v*v, minlength=len(pesos)))
            prox = v*np.maximum(0., 1-lam*pesos/L/np.maximum(n, np.finfo(float).tiny))[ids]
            residuo = float(L*np.linalg.norm(beta-prox, ord=np.inf))
            assert residuo <= CONFIG['solver_tol']*escala*(1+1e-8)
            np.testing.assert_allclose(paso['residuo_kkt'], residuo, rtol=1e-7, atol=1e-14)
            np.testing.assert_allclose(paso['residuo_kkt_relativo'], residuo/escala, rtol=1e-7, atol=1e-14)
            if valido:
                score = paso['rmse_validation_original']
                assert np.isfinite(score)
                clave = tuple(activas)
                if clave in cache:
                    assert cache[clave] == score
                cache[clave] = score
                candidatos.append(dict(seleccionadas=activas, **{'lambda': lam}, rmse_validation_original=score))
            else:
                assert paso['rmse_validation_original'] is None
    real = y.iloc[datos['val']].to_numpy()
    basal = float(np.sqrt(np.mean((real-np.exp(np.log(y.iloc[fit]).mean()))**2)))
    np.testing.assert_allclose(sel['rmse_intercepto_original'], basal, rtol=1e-12)
    np.testing.assert_allclose(sel['target_mean_fit'], np.log(y.iloc[fit]).mean(), rtol=1e-12)
    assert sel['ols_provisionales_unicos'] == len(cache)
    ganador, decisiones = elegir_por_complejidad(candidatos, sel['rmse_intercepto_original'])
    assert decisiones == sel['decisiones_complejidad']
    assert ganador['seleccionadas'] == sel['seleccionadas'] and ganador['lambda'] == sel['lambda']
    assert ganador['rmse_validation_original'] == sel['rmse_validation_original']


def resumen_numerico_presentacion(resultado):
    sel = resultado['seleccion']
    meta = json.loads((resultado['salida']/'estabilidad_numerica.json').read_text(encoding='utf-8'))
    return (f"Universo: 177 columnas / {len(resultado['grupos'])} grupos; Gram de rango "
        f"{meta['rango_gram']}/177, condición {meta['condicion_gram']:.1f}. "
        f"Se evaluaron las 40 lambdas: {sum(h['n_caracteristicas'] > 12 for h in sel['historial'])} "
        f"configuraciones con más de 12 grupos se conservaron como no elegibles. "
        f"Todas convergieron; máximo residuo KKT relativo "
        f"{max(h['residuo_kkt_relativo'] for h in sel['historial']):.2e} (tolerancia 1e-7). "
        "No se excluyeron columnas ni se modificaron hiperparámetros; el OLS definitivo usa las escalas exportadas.")


def resolver_group_lasso(gram, xy, ids, pesos, lam, *, inicial=None, tol=1e-7, max_iter=20000):
    """Proximal por bloques; devuelve solo soluciones convergentes y su residuo KKT."""
    L = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(lam) or lam <= 0 or L <= 0:
        raise ValueError('Lambda y Lipschitz deben ser positivos.')
    beta = np.zeros(len(xy)) if inicial is None else inicial.copy()
    z, t = beta.copy(), 1.
    umbrales = lam*pesos/L
    def prox(v):
        normas = np.sqrt(np.bincount(ids, weights=v*v, minlength=len(pesos)))
        factores = np.maximum(0., 1-umbrales/np.maximum(normas, np.finfo(float).tiny))
        return v*factores[ids]
    escala = max(float(np.linalg.norm(xy, ord=np.inf)), 1e-12)
    for it in range(1, max_iter+1):
        nuevo = prox(z-(gram@z-xy)/L)
        residuo = float(L*np.linalg.norm(nuevo-prox(nuevo-(gram@nuevo-xy)/L), ord=np.inf))
        if residuo <= tol*escala:
            return nuevo, {'iteraciones_solver': it, 'residuo_kkt': residuo,
                           'residuo_kkt_relativo': residuo/escala, 'convergencia': True}
        tn = (1+np.sqrt(1+4*t*t))/2
        if np.dot(z-nuevo, nuevo-beta) > 0:
            z, t = nuevo.copy(), 1.
        else:
            z, t = nuevo+(t-1)/tn*(nuevo-beta), tn
        beta = nuevo
    raise RuntimeError(f'Group Lasso no convergió: lambda={lam}, residuo relativo={residuo/escala:g}')


def elegir_por_complejidad(candidatos, rmse_intercepto, *, min_mejora_pct=.5, max_features=12):
    """Evalúa todos los tamaños disponibles; un rechazo no detiene tamaños posteriores."""
    if not 1 <= max_features <= 12 or min_mejora_pct < 0 or not np.isfinite(min_mejora_pct):
        raise ValueError('Criterios de selección inválidos.')
    aceptado = {'seleccionadas': [], 'lambda': None, 'rmse_validation_original': rmse_intercepto}
    decisiones = []
    for k in sorted({len(c['seleccionadas']) for c in candidatos if 0 < len(c['seleccionadas']) <= max_features}):
        candidato = min((c for c in candidatos if len(c['seleccionadas']) == k),
                        key=lambda c: c['rmse_validation_original'])
        mejora = 100*(aceptado['rmse_validation_original']-candidato['rmse_validation_original'])/aceptado['rmse_validation_original']
        aceptar = bool(np.isfinite(mejora) and mejora >= min_mejora_pct)
        decisiones.append({'n_caracteristicas': k, 'lambda': candidato['lambda'],
            'seleccionadas': candidato['seleccionadas'], 'rmse_referencia': aceptado['rmse_validation_original'],
            'rmse_validation_original': candidato['rmse_validation_original'],
            'mejora_pct': mejora, 'aceptado': aceptar})
        if aceptar:
            aceptado = candidato.copy()
    return aceptado, decisiones


def buscar_group_lasso(X, y, grupos, fit, val, salida, *, lambdas=None, config=None,
                       excluidas=s.EXCLUIDAS):
    cfg = {**CONFIG, **(config or {})}
    if not 1 <= cfg['max_features'] <= 12 or cfg['min_mejora_pct'] < 0:
        raise ValueError('Máximo absoluto 12 y mejora mínima no negativa.')
    assert X.index.equals(y.index) and not set(excluidas) & set(X)
    assert not set(fit) & set(val) and len(fit)+len(val) == len(X)
    assert set(fit) | set(val) == set(range(len(X)))
    assert set(s.columnas_grupos(grupos, list(grupos))) == set(X)
    out = b.validar_temporal(salida)
    out.mkdir(exist_ok=True)
    nombres = list(grupos)
    posiciones = {c: i for i, c in enumerate(X)}
    bloques = [np.array([posiciones[c] for c in grupos[g]]) for g in nombres]
    ids = np.empty(X.shape[1], dtype=int)
    for i, bloque in enumerate(bloques): ids[bloque] = i
    scaler = StandardScaler().fit(X.iloc[fit])
    Z = scaler.transform(X.iloc[fit])
    # Se aplica el mismo escalador en validación; OLS provisional usa las escalas exportadas.
    assert np.isfinite(scaler.transform(X.iloc[val])).all()
    y_log = np.log(y)
    yc = y_log.iloc[fit].to_numpy()-y_log.iloc[fit].mean()
    historial, candidatos, cache, coeficientes = [], [], {}, []
    with threadpool_limits(limits=1):
        gram, xy = Z.T@Z/len(fit), Z.T@yc/len(fit)
        rangos = [int(np.linalg.matrix_rank(gram[np.ix_(i, i)], tol=1e-10)) for i in bloques]
        if not excluidas:
            b.guardar_json(out/'estabilidad_numerica.json', diagnostico_gram(gram, bloques, list(X)))
        assert all(r == len(i) for r, i in zip(rangos, bloques)), 'Grupo interno no identificable.'
        assert np.linalg.matrix_rank(gram, tol=1e-10) == X.shape[1], 'Diseño interno no identificable.'
        pesos = np.sqrt(rangos)
        lambda_max = max(np.linalg.norm(xy[i])/w for i, w in zip(bloques, pesos))
        grilla = np.asarray(lambdas if lambdas is not None else
            lambda_max*np.geomspace(1., cfg['lambda_min_ratio'], cfg['n_lambdas']), dtype=float)
        if grilla.ndim != 1 or not len(grilla) or not np.isfinite(grilla).all() or (grilla <= 0).any():
            raise ValueError('Definir una grilla finita de lambdas positivos.')
        grilla = np.sort(np.unique(grilla))[::-1]
        beta = np.zeros(X.shape[1])
        interno = np.column_stack([np.ones(len(fit)), X.to_numpy()[fit]])
        validacion = np.column_stack([np.ones(len(val)), X.to_numpy()[val]])
        real = y.to_numpy()[val]
        def evaluar(seleccion):
            indices = [0]+[posiciones[c]+1 for g in seleccion for c in grupos[g]]
            modelo = b.sm.OLS(y_log.iloc[fit].to_numpy(), interno[:, indices]).fit()
            assert modelo.model.rank == len(indices), 'OLS provisional de rango deficiente.'
            score = float(np.sqrt(np.mean((real-np.exp(modelo.predict(validacion[:, indices])))**2)))
            assert np.isfinite(score)
            return score
        rmse_intercepto = evaluar([])
        for lam in grilla:
            beta, info = resolver_group_lasso(gram, xy, ids, pesos, float(lam), inicial=beta,
                tol=cfg['solver_tol'], max_iter=cfg['max_iter'])
            coeficientes.append(beta.copy())
            normas = [float(np.linalg.norm(beta[i])) for i in bloques]
            seleccionadas = [g for g, norma in zip(nombres, normas) if norma > cfg['support_tol']]
            valido = 0 < len(seleccionadas) <= cfg['max_features']
            score = None
            if valido:
                clave = tuple(seleccionadas)
                if clave not in cache: cache[clave] = evaluar(seleccionadas)
                score = cache[clave]
                candidatos.append({'lambda': float(lam), 'seleccionadas': seleccionadas,
                                   'rmse_validation_original': score})
            historial.append({'lambda': float(lam), 'seleccionadas': seleccionadas,
                'n_caracteristicas': len(seleccionadas), 'n_predictores': sum(len(grupos[g]) for g in seleccionadas),
                'valido': valido, 'rmse_validation_original': score,
                'motivo': 'evaluado' if valido else 'solo_intercepto' if not seleccionadas else 'max_features',
                'normas_grupos': dict(zip(nombres, normas)), **info})
    elegido, decisiones = elegir_por_complejidad(candidatos, rmse_intercepto,
        min_mejora_pct=cfg['min_mejora_pct'], max_features=cfg['max_features'])
    if not elegido['seleccionadas']:
        raise RuntimeError('Ningún subconjunto mejora al intercepto según el umbral; no registrar un modelo arbitrario.')
    resultado = {**elegido, 'rmse_intercepto_original': rmse_intercepto, 'historial': historial,
        'decisiones_complejidad': decisiones, 'config': cfg, 'lambda_max': float(lambda_max),
        'lambdas': grilla.tolist(), 'motivo_parada': 'grilla_completa_y_criterio_complejidad',
        'ponderacion': {g: {'p_g': r, 'peso': float(w)} for g, r, w in zip(nombres, rangos, pesos)},
        'ols_provisionales_unicos': len(cache), 'target_mean_fit': float(y_log.iloc[fit].mean())}
    joblib.dump(scaler, out/'scaler_seleccion.joblib')
    np.savez_compressed(out/'ruta_coeficientes_group_lasso.npz', lambdas=grilla,
                        coeficientes=np.asarray(coeficientes), group_ids=ids, columnas=np.asarray(list(X)))
    np.savez_compressed(out/'particion_interna.npz', fit=fit, validation=val)
    b.guardar_json(out/'seleccion.json', resultado)
    b.guardar_json(out/'historial_lambdas.json', historial)
    b.guardar_json(out/'decisiones_complejidad.json', decisiones)
    b.guardar_json(out/'grupos_caracteristicas.json', grupos)
    return resultado


def entrenar_exportar(base, X, y, grupos, seleccion, contexto_train, referencia, salida,
                     *, excluidas=s.EXCLUIDAS, nombre_run=NOMBRE_RUN):
    columnas = s.columnas_grupos(grupos, seleccion['seleccionadas'])
    modelo, diseno = b.ajustar(X[columnas], np.log(y))
    assert modelo.model.rank == len(columnas)+1 and modelo.nobs == len(y)
    metricas, invalidas, vif, notas = b.diagnosticar(modelo, diseno, np.log(y), y_original=y, resumir_vif=True)
    metricas.update(train_n_features_original=float(len(seleccion['seleccionadas'])),
        selection_rmse_validation_original=seleccion['rmse_validation_original'],
        selection_rmse_intercept_original=seleccion['rmse_intercepto_original'],
        selection_lambda=seleccion['lambda'])
    contexto = {**contexto_train, 'predictoras': columnas, 'n_predictoras': len(columnas),
        'transformacion_objetivo': 'ln', 'unidad': 'ln(RUB)', 'predictoras_excluidas': list(excluidas),
        'seleccion_caracteristicas': 'group_lasso_then_ols', 'reference_run_id': referencia['run_id'],
        'caracteristicas_originales': seleccion['seleccionadas']}
    if not excluidas:
        contexto.update(universo_predictoras=list(X), n_predictoras_inicial=X.shape[1])
    out = b.exportar(base, modelo, diseno, np.log(y), contexto, metricas, invalidas, vif, notas,
                     nombre_run=nombre_run, carpeta_salida=salida)
    tabla = s.tabla_comparativa(referencia['metrics'], metricas, invalidas,
                                nombre_referencia=referencia['run_name'], nombre_run=nombre_run)
    resumen = s.resumen_caracteristicas(seleccion['seleccionadas'], grupos, modelo, mostrar_orden=False)
    (out/'comparacion_referencia.md').write_text(tabla, encoding='utf-8')
    (out/'caracteristicas_seleccionadas.md').write_text(resumen, encoding='utf-8')
    b.guardar_json(out/'referencia_mlflow.json', referencia)
    b.guardar_json(out/'caracteristicas_seleccionadas.json', {g: grupos[g] for g in seleccion['seleccionadas']})
    return {'modelo': modelo, 'metricas': metricas, 'contexto': contexto, 'salida': out,
            'comparacion_md': tabla, 'caracteristicas_md': resumen}


def registrar(cliente, resultado, seleccion, referencia, *, nombre_run=NOMBRE_RUN):
    cfg = seleccion['config']
    assert not cliente.search_runs([referencia['experiment_id']],
        filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=1), 'El Run ya existe; no sobrescribir.'
    params = {'algorithm': 'Group Lasso + OLS', 'min_mejora_pct': cfg['min_mejora_pct'],
        'max_features': cfg['max_features'], 'lambda_selected': seleccion['lambda'],
        'lambda_grid': json.dumps(seleccion['lambdas']), 'group_weight': 'sqrt(effective_group_rank)',
        'scaler': 'StandardScaler fit internal_train only; OLS in original transformed columns',
        'support_tol': cfg['support_tol'], 'solver': 'pinv', 'selection_solver': 'FISTA Gram; adaptive restart',
        'selection_solver_tol': cfg['solver_tol'], 'selection_solver_max_iter': cfg['max_iter'],
        'objective': '||yc-Zb||^2/(2n) + lambda * sum(sqrt(p_g)*||b_g||_2)',
        'n_features_original': len(seleccion['seleccionadas']), 'selected_features': json.dumps(seleccion['seleccionadas']),
        'reference_run_name': referencia['run_name'], 'reference_run_id': referencia['run_id'],
        'experiment_id': referencia['experiment_id'], 'notebook': '4_Seleccion_Modelos.ipynb',
        'validation_fraction': .2, 'random_state': referencia['params']['random_state'], 'permutation_repeats': 10,
        'selection_stop_reason': seleccion['motivo_parada']}
    if resultado['contexto'].get('n_predictoras_inicial') == 177:
        assert referencia['run_name'] == s.NOMBRE_RUN_177 and cfg == CONFIG
        params.update(n_candidate_predictors=177,
            n_candidate_groups=len(json.loads((resultado['salida']/'grupos_caracteristicas.json').read_text(encoding='utf-8'))),
            candidate_columns=json.dumps(resultado['contexto']['universo_predictoras']),
            excluded_predictors='', n_lambdas=40, lambda_min_ratio=.001)
    b.guardar_json(resultado['salida']/'configuracion.json', params)
    registro = b.registrar(cliente, resultado['salida'], resultado['contexto'], resultado['metricas'],
                           nombre_run=nombre_run, parametros_extra=params)
    actual = cliente.get_run(referencia['run_id'])
    assert actual.data.metrics == referencia['metrics'] and actual.data.params == referencia['params']
    assert actual.info.status == 'FINISHED'
    return registro
