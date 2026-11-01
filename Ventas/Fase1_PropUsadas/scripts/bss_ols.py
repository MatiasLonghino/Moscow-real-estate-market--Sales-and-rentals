"""BSS exhaustivo por RSS interno; validación exclusivamente entre ganadores por tamaño."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from . import baseline_ols as b, seleccion_modelos as s, group_lasso_ols as gl, backward_ols as bw

NOMBRE_RUN = 'BSS_06_Log_Union_177_Grupos_OLS'
METODO = 'best_subset_grouped_train_rss_union'
VERSION = 'bss_rss_exhaustive_v1'
HISTORIAL = 'historial_combinaciones.csv.zip'
CONFIG = dict(min_mejora_pct=.5, max_features=12, random_state=42)
ADVERTENCIA = ('La unión se construyó con métodos que ya utilizaron esta validación. '
    'BSS reutiliza información que influyó en sus candidatas: existe riesgo de sesgo por selección '
    'y optimismo. La comparación es exploratoria, no una evaluación independiente sobre datos nuevos. '
    'Test queda reservado para la evaluación final, una vez cerradas las decisiones metodológicas.')


def cuadro_selecciones(grupos, selecciones):
    """Unión dinámica en orden de linaje; jamás columnas OHE como unidades de selección."""
    conjuntos = {nombre: set(v) for nombre, v in selecciones.items()}
    assert len(conjuntos) == 3
    for nombre, valores in selecciones.items():
        assert len(valores) == len(conjuntos[nombre]) and conjuntos[nombre] <= set(grupos)
    union = [g for g in grupos if any(g in v for v in conjuntos.values())]
    assert union
    filas = []
    for g in union:
        miembros = [nombre for nombre, v in conjuntos.items() if g in v]
        filas.append({'Característica original': g,
            **{nombre: '✓' if g in v else '—' for nombre, v in conjuntos.items()},
            'Número de métodos': len(miembros),
            'Clasificación de coincidencia': 'Los 3' if len(miembros) == 3 else
                ' y '.join(miembros) if len(miembros) == 2 else 'Solo '+miembros[0]})
    tabla = pd.DataFrame(filas).sort_values(['Número de métodos', 'Característica original'],
                                           ascending=[False, True], kind='stable').reset_index(drop=True)
    meta = dict(union=union, interseccion=[g for g in union if all(g in v for v in conjuntos.values())],
        exactamente_dos=[g for g in union if sum(g in v for v in conjuntos.values()) == 2],
        exclusivas=[g for g in union if sum(g in v for v in conjuntos.values()) == 1],
        selecciones={k: list(v) for k, v in selecciones.items()},
        grupos={g: grupos[g] for g in union})
    columnas = s.columnas_grupos(meta['grupos'], union)
    assert len(columnas) == len(set(columnas))
    meta['columnas'] = columnas
    assert sum(len(meta[k]) for k in ('interseccion', 'exactamente_dos', 'exclusivas')) == len(union)
    assert tabla['Número de métodos'].sum() == sum(map(len, selecciones.values()))
    return tabla, meta


def construir_universo(datos, forward, backward, lasso):
    """Verifica los tres Runs reales, sus selecciones, grupos, huella e índices."""
    assert datos['excluidas'] == () and datos['X'].shape[1] == 177
    assert datos['config_forward'] == CONFIG and datos['config_lasso'] == gl.CONFIG
    resultados = [('Forward 177', 'forward', s.NOMBRE_RUN_177, forward),
                  ('Backward 177', 'backward', bw.NOMBRE_RUN, backward),
                  ('Group Lasso 177', 'lasso', gl.NOMBRE_RUN_177, lasso)]
    origenes = {}
    for etiqueta, metodo, nombre, r in resultados:
        assert r['metodo'] == metodo and r['run']['run_name'] == nombre
        s._validar_resultado_presentacion(datos, r)
        vivo = s.recuperar_experimento(datos['cliente'], nombre=nombre, run_id=r['run']['run_id'],
            parametros={'n_candidate_predictors': 177},
            etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
        assert vivo['params'] == r['run']['params'] and vivo['metrics'] == r['run']['metrics']
        origenes[etiqueta] = dict(run=vivo, seleccionadas=r['seleccion']['seleccionadas'])
    gl.validar_referencia_177(datos, forward)
    bw.validar_recuperacion(datos, backward)
    gl.validar_recuperacion_177(datos, lasso)
    tabla, universo = cuadro_selecciones(datos['grupos'],
        {etiqueta: r['seleccion']['seleccionadas'] for etiqueta, _, _, r in resultados})
    universo.update(origenes=origenes, dataset_sha256=datos['contexto']['dataset_sha256'])
    assert set(universo['columnas']) <= set(datos['X'])
    return tabla, universo


def cuadro_presentacion(tabla, universo):
    vista = tabla.copy()
    vista['Característica original'] = [f'**`{g}`**' if n == 3 else f'`{g}`'
        for g, n in zip(tabla['Característica original'], tabla['Número de métodos'])]
    resumen = '; '.join(f'{nombre}: {len(v)} características' for nombre, v in universo['selecciones'].items())
    resumen += (f". Coinciden en los tres: {len(universo['interseccion'])}; en exactamente dos: "
        f"{len(universo['exactamente_dos'])}; exclusivas: {len(universo['exclusivas'])}. "
        f"Unión: {len(universo['union'])} características / {len(universo['columnas'])} columnas. "
        'Se incluyen todas las características elegidas por al menos un método, sin priorizar las comunes '
        'ni excluir las exclusivas. BSS estudia exhaustivamente esa unión, no solo la intersección.')
    return b.tabla_markdown(vista), resumen


def _estadisticos(rss, tss, n, rango):
    r2 = 1-rss/tss if tss > 0 else None
    return dict(r2_train_log=r2, r2_adj_train_log=1-(1-r2)*(n-1)/(n-rango)
        if r2 is not None and n > rango else None,
        aic_train_log=n*(np.log(2*np.pi)+1+np.log(rss/n))+2*rango if rss > 0 and n > rango else None,
        bic_train_log=n*(np.log(2*np.pi)+1+np.log(rss/n))+np.log(n)*rango if rss > 0 and n > rango else None)


class MotorRSS:
    """Álgebra OLS compartida; QR/Gram se preparan una vez por entrenamiento."""
    def __init__(self, A, z, *, fallback_statsmodels=False):
        self.A, self.z = np.asarray(A,dtype=float), np.asarray(z,dtype=float)
        self.n = len(z)
        self.raw = np.column_stack([np.ones(self.n),self.A])
        self.rango = int(np.linalg.matrix_rank(self.raw))
        self.centro, escala = self.A.mean(axis=0), self.A.std(axis=0)
        self.escala = np.where(escala > 0,escala,1.)
        C, yc = (self.A-self.centro)/self.escala, self.z-self.z.mean()
        Q,self.R = np.linalg.qr(C,mode='reduced')
        self.qy = Q.T@yc
        fuera = yc-Q@self.qy
        self.rss_fuera, self.tss = float(fuera@fuera), float(yc@yc)
        self.gram,self.xy = self.R.T@self.R,self.R.T@self.qy
        self.fallback_statsmodels = fallback_statsmodels

    def resolver(self, indices):
        if not indices:
            return self.tss,1,np.empty(0),float(self.z.mean()),'QR_Gram_Cholesky'
        fallback = self.rango != self.raw.shape[1]
        if not fallback:
            try:
                factor = cho_factor(self.gram[np.ix_(indices,indices)],lower=True,check_finite=False)
                beta = cho_solve(factor,self.xy[indices],check_finite=False)
                residuo = self.qy-self.R[:,indices]@beta
                rss,rango = self.rss_fuera+float(residuo@residuo),len(indices)+1
                if rss <= np.finfo(float).eps*self.n*max(self.tss,1.):
                    fallback = True
            except np.linalg.LinAlgError:
                fallback = True
        if fallback:
            if self.fallback_statsmodels:
                modelo = b.sm.OLS(self.z,self.raw[:,[0]+[i+1 for i in indices]],missing='raise').fit()
                return float(modelo.ssr),int(modelo.model.rank),np.asarray(modelo.params[1:]),float(modelo.params[0]),'pinv'
            # SVD solo de la matriz QR reducida: nunca statsmodels por combinación.
            beta,_,rango,_ = np.linalg.lstsq(self.R[:,indices],self.qy,rcond=1e-12)
            residuo = self.qy-self.R[:,indices]@beta
            rss,rango = self.rss_fuera+float(residuo@residuo),int(rango)+1
        coef = beta/self.escala[indices]
        return float(rss),rango,coef,float(self.z.mean()-self.centro[indices]@coef),'QR_SVD' if fallback else 'QR_Gram_Cholesky'


def buscar_rss_minimo(X, y_log, grupos, tamanos):
    """Ruta mínima: solo tamaños solicitados, ganadores y coeficientes; sin diagnósticos/historial."""
    assert X.index.equals(y_log.index) and np.isfinite(X.to_numpy()).all() and np.isfinite(y_log).all()
    assert set(X)==set(s.columnas_grupos(grupos,list(grupos)))
    assert X.shape[1]==sum(map(len,grupos.values())) and X.columns.is_unique
    tamanos=sorted(set(tamanos))
    assert tamanos and all(1<=k<=len(grupos) for k in tamanos)
    posiciones={c:i for i,c in enumerate(X)}
    ids={g:[posiciones[c] for c in cols] for g,cols in grupos.items()}
    ganadores,evaluados,no_identificables={},0,0
    with threadpool_limits(limits=1):
        motor=MotorRSS(X.to_numpy(),y_log.to_numpy())
        for k in tamanos:
            mejor=None
            for combinacion in itertools.combinations(grupos,k):
                indices=[i for g in combinacion for i in ids[g]]
                rss,rango,coef,intercepto,_=motor.resolver(indices)
                evaluados+=1
                if rango != len(indices)+1:
                    no_identificables+=1
                    continue
                if mejor is None or rss<mejor['rss']:
                    mejor=dict(seleccionadas=list(combinacion),columnas=[X.columns[i] for i in indices],
                        rss=rss,coef=coef,intercepto=intercepto,rango=rango)
            if mejor is None:
                raise ValueError(f'No hay un subconjunto identificable de tamaño {k}; no completar con ceros ni omitir el tamaño.')
            ganadores[k]=mejor
    assert evaluados==sum(math.comb(len(grupos),k) for k in tamanos)
    return ganadores,dict(evaluados=evaluados,no_identificables=no_identificables,rango_union=motor.rango)


def buscar_bss(X, y, grupos, fit, val, *, min_mejora_pct=.5, max_features=12):
    """Enumera 2**m diseños. QR/Gram solo aceleran OLS; no transforman el dataset.

    Centrado y escala internos son precondicionamiento algebraico reversible. El
    RSS se calcula como norma de residuo en coordenadas QR, sin restar y'y-beta'X'y.
    Un diseño completo de rango pleno garantiza el rango de sus subdiseños; en el
    caso singular se usa OLS pinv explícito y se documenta cada diseño no identificable.
    Empates exactos de RSS: primera combinación en orden del linaje, sin validación.
    """
    if not isinstance(max_features, int) or not 1 <= max_features <= 12 or not np.isfinite(min_mejora_pct) or min_mejora_pct < 0:
        raise ValueError('Máximo absoluto 12 y mejora mínima finita no negativa.')
    assert X.index.equals(y.index) and (y > 0).all() and np.isfinite(y).all()
    assert np.isfinite(X.to_numpy()).all() and grupos
    assert len(fit) > 1 and len(val) > 1 and len(fit)+len(val) == len(X)
    assert not set(fit) & set(val) and set(fit) | set(val) == set(range(len(X)))
    columnas = s.columnas_grupos(grupos, list(grupos))
    assert len(columnas) == len(set(columnas)) == X.shape[1] and set(columnas) == set(X)
    nombres = list(grupos)
    posiciones = {c: i for i, c in enumerate(X)}
    ids = {g: [posiciones[c] for c in grupos[g]] for g in nombres}
    A, z = X.iloc[fit].to_numpy(dtype=float), np.log(y.iloc[fit]).to_numpy()
    n, m = len(fit), len(nombres)
    historial, mejores = [], []
    with threadpool_limits(limits=1):
        motor = MotorRSS(A,z,fallback_statsmodels=True)
        rango_completo,tss = motor.rango,motor.tss
        for k in range(m+1):
            mejor = None
            for combinacion in itertools.combinations(nombres, k):
                indices = [i for g in combinacion for i in ids[g]]
                rss,rango,_,_,solver = motor.resolver(indices)
                fila = dict(orden=len(historial), id_subconjunto=sum(1 << nombres.index(g) for g in combinacion),
                    n_caracteristicas=k, n_predictores=len(indices), rango_efectivo=rango,
                    identificable=rango == len(indices)+1, grupos=json.dumps(list(combinacion)),
                    rss_train_log=float(rss), solver=solver,
                    **_estadisticos(rss, tss, n, rango))
                historial.append(fila)
                if fila['identificable'] and (mejor is None or rss < mejor['rss_train_log']):
                    mejor = fila.copy()  # Estrictamente menor: desempate por primera combinación.
            if mejor is not None:
                mejores.append(mejor)
        assert len(historial) == 2**m
        assert len({h['id_subconjunto'] for h in historial}) == 2**m
        discrepancias = []
        for mejor in mejores:
            seleccionadas = json.loads(mejor['grupos'])
            cols = s.columnas_grupos(grupos, seleccionadas)
            modelo, diseno = b.ajustar(X[cols].iloc[fit], np.log(y.iloc[fit]))
            assert modelo.model.rank == diseno.shape[1]
            np.testing.assert_allclose(mejor['rss_train_log'], modelo.ssr, rtol=1e-8, atol=1e-8)
            discrepancias.append(abs(mejor['rss_train_log']-modelo.ssr))
            mejor['rss_refit_log'] = float(modelo.ssr)
            for clave, valor in [('r2_train_log', modelo.rsquared), ('r2_adj_train_log', modelo.rsquared_adj),
                                  ('aic_train_log', modelo.aic), ('bic_train_log', modelo.bic)]:
                if mejor[clave] is not None:
                    np.testing.assert_allclose(mejor[clave], valor, rtol=1e-8, atol=1e-8)
            with np.errstate(over='ignore', invalid='ignore'):
                pred = np.exp(modelo.predict(b.sm.add_constant(X[cols].iloc[val], has_constant='add')))
                error = float(np.sqrt(np.mean((y.iloc[val].to_numpy()-pred)**2)))
            mejor.update(seleccionadas=seleccionadas, rmse_validation_original=error if np.isfinite(error) else None)
    basal = next(c['rmse_validation_original'] for c in mejores if c['n_caracteristicas'] == 0)
    if basal is None or basal <= 0:
        raise ValueError('RMSE del intercepto inválido para comparar mejoras relativas.')
    candidatos = [dict(c, **{'lambda': None}) for c in mejores
                  if c['n_caracteristicas'] > 0 and c['rmse_validation_original'] is not None]
    ganador, decisiones = gl.elegir_por_complejidad(candidatos, basal,
        min_mejora_pct=min_mejora_pct, max_features=max_features)
    optimos = {}
    for clave, minimizar in [('rss_train_log', True), ('r2_adj_train_log', False),
                             ('aic_train_log', True), ('bic_train_log', True), ('rmse_validation_original', True)]:
        validos = [c for c in mejores if c.get(clave) is not None and np.isfinite(c[clave])]
        if validos:
            elegido = min(validos, key=lambda c: c[clave] if minimizar else -c[clave])
            optimos[clave] = dict(k=elegido['n_caracteristicas'], valor=elegido[clave])
    seleccion = dict(seleccionadas=ganador['seleccionadas'],
        rmse_validation_original=ganador['rmse_validation_original'], rmse_intercepto_original=basal,
        mejores_por_tamano=mejores, decisiones_complejidad=decisiones, optimos=optimos,
        n_grupos_candidatos=m, n_columnas_candidatas=len(columnas), combinaciones_posibles=2**m,
        combinaciones_evaluadas=len(historial), combinaciones_no_nulas=2**m-1,
        combinaciones_elegibles=sum(math.comb(m, k) for k in range(1, min(m, max_features)+1)),
        modelos_identificables=sum(h['identificable'] for h in historial),
        n_modelos_validados=len(mejores), rango_diseno_union=rango_completo,
        max_diferencia_rss_refit=max(discrepancias),
        config=dict(min_mejora_pct=min_mejora_pct, max_features=max_features, random_state=42),
        motivo_parada='busqueda_exhaustiva_completa' if ganador['seleccionadas'] else 'ningun_modelo_supera_umbral',
        desempate='Primera combinación en orden del linaje cuando RSS es exactamente igual; sin validación',
        optimizacion='Centrado y escala solo algebraicos; QR reducido y Gram/Cholesky; pinv si el diseño completo no tiene rango pleno')
    return seleccion, pd.DataFrame(historial)


def _parametros(datos, universo):
    return {**s._parametros_presentacion(datos, 'forward'), 'feature_selection': METODO,
        'n_candidate_predictors': len(universo['columnas']), 'n_candidate_groups': len(universo['union']),
        'source_universe_predictors': 177, 'candidate_features': json.dumps(universo['union']),
        'candidate_columns': json.dumps(universo['columnas']), 'algorithm_version': VERSION,
        'history_artifact': HISTORIAL,
        'reference_run_ids': json.dumps({k: v['run']['run_id'] for k, v in universo['origenes'].items()}),
        'within_size_metric': 'RSS internal_train ln(RUB)', 'permutation_repeats': 10,
        'selection_metric': 'RMSE validation original RUB; exp sin corrección'}


def validar_fuentes_actuales(datos, universo):
    """Una fuente eliminada, incompleta o modificada no permite recuperar/publicar BSS."""
    esperados = {'Forward 177': s.NOMBRE_RUN_177, 'Backward 177': bw.NOMBRE_RUN,
                 'Group Lasso 177': gl.NOMBRE_RUN_177}
    assert set(universo['origenes']) == set(esperados)
    for etiqueta, nombre in esperados.items():
        previo = universo['origenes'][etiqueta]['run']
        vivo = s.recuperar_experimento(datos['cliente'], nombre=nombre, run_id=previo['run_id'],
            parametros={'n_candidate_predictors': 177},
            etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
        assert vivo['params'] == previo['params'] and vivo['metrics'] == previo['metrics']


def guardar_graficos(seleccion, archivo):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 10), layout='constrained')
    axes = fig.subplots(3, 2).ravel()
    elegida = len(seleccion['seleccionadas'])
    for ax, (clave, titulo) in zip(axes, [('rss_train_log', 'RSS entrenamiento interno [ln(RUB)]²'),
        ('r2_adj_train_log', 'R² ajustado entrenamiento interno [log]'), ('aic_train_log', 'AIC entrenamiento interno [log]'),
        ('bic_train_log', 'BIC entrenamiento interno [log]'), ('rmse_validation_original', 'RMSE validación interna (RUB)')]):
        filas = [c for c in seleccion['mejores_por_tamano'] if c.get(clave) is not None]
        ax.plot([c['n_caracteristicas'] for c in filas], [c[clave] for c in filas], 'o-', markersize=4)
        ax.axvline(elegida, color='#b12c2c', linestyle='--', label=f'Seleccionado: k={elegida}')
        opt = seleccion['optimos'].get(clave)
        if opt:
            ax.plot(opt['k'], opt['valor'], '*', color='#24814b', markersize=12, label=f'Óptimo del criterio: k={opt["k"]}')
        ax.set(title=titulo, xlabel='Características originales (k)', xticks=range(seleccion['n_grupos_candidatos']+1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].axis('off')
    axes[-1].text(0, .9, 'Un ganador por tamaño: mínimo RSS interno.\n\nSelección final: RMSE en RUB, máximo 12 grupos,\nmejora ≥0,5 % frente al último aceptado.\n\n'
        f"Se incluye el tamaño máximo k={seleccion['n_grupos_candidatos']}.\n"
        'La validación se reutilizó para formar la unión;\nla evaluación es exploratoria.', va='top', fontsize=11)
    fig.savefig(archivo, dpi=150)


def preparar_experimento(datos, tabla, universo, salida):
    """Ajusta una sola búsqueda nueva y el OLS definitivo; no escribe en MLflow."""
    assert datos['config_forward'] == CONFIG
    out = b.validar_temporal(salida)
    out.mkdir(exist_ok=True)
    grupos = universo['grupos']
    X, y = datos['X'][universo['columnas']], datos['y']
    sel, historial = buscar_bss(X, y, grupos, datos['fit'], datos['val'])
    b.guardar_json(out/'seleccion.json', sel)
    historial.to_csv(out/HISTORIAL, index=False,
        compression={'method':'zip','archive_name':'historial_combinaciones.csv'})
    if not sel['seleccionadas']:
        raise RuntimeError('BSS no encontró un subconjunto aceptable; no registrar un ganador arbitrario.')
    columnas = s.columnas_grupos(grupos, sel['seleccionadas'])
    modelo, diseno = b.ajustar(X[columnas], np.log(y))
    assert modelo.model.rank == diseno.shape[1] and modelo.nobs == len(y)
    metricas, invalidas, vif, notas = b.diagnosticar(modelo, diseno, np.log(y), y_original=y, resumir_vif=True)
    metricas.update(train_n_features_original=float(len(sel['seleccionadas'])),
        selection_rmse_validation_original=sel['rmse_validation_original'],
        selection_rmse_intercept_original=sel['rmse_intercepto_original'],
        selection_subsets_evaluated=float(sel['combinaciones_evaluadas']))
    ctx = {**datos['contexto'], 'predictoras': columnas, 'n_predictoras': len(columnas),
        'transformacion_objetivo': 'ln', 'unidad': 'ln(RUB)', 'predictoras_excluidas': [],
        'seleccion_caracteristicas': METODO, 'caracteristicas_originales': sel['seleccionadas'],
        'universo_predictoras': universo['columnas'], 'n_predictoras_inicial': len(universo['columnas']),
        'n_predictoras_origen': 177}
    b.exportar(datos['base'], modelo, diseno, np.log(y), ctx, metricas, invalidas, vif, notas,
               nombre_run=NOMBRE_RUN, carpeta_salida=out)
    for nombre, valor in [('universo_bss', universo), ('grupos_caracteristicas', grupos),
        ('mejores_por_tamano', sel['mejores_por_tamano']), ('criterios_seleccion', sel['optimos'])]:
        b.guardar_json(out/(nombre+'.json'), valor)
    tabla.to_csv(out/'cuadro_caracteristicas.csv', index=False)
    md, resumen = cuadro_presentacion(tabla, universo)
    (out/'cuadro_caracteristicas.md').write_text(md+'\n\n'+resumen, encoding='utf-8')
    np.savez_compressed(out/'particion_interna.npz', fit=datos['fit'], validation=datos['val'])
    for nombre in ('bss_ols.py','group_lasso_ols.py','backward_ols.py'):
        (out/nombre).write_bytes(Path(__file__).with_name(nombre).read_bytes())
    guardar_graficos(sel, out/'evolucion_bss.png')
    (out/'criterios_seleccion.md').write_text(criterios_presentacion(sel), encoding='utf-8')
    importancia = s.importancia_permutacion_grupos(X, y, grupos, sel['seleccionadas'], datos['fit'], datos['val'],
        n_repeticiones=10, random_state=42, rmse_esperado=sel['rmse_validation_original'])
    importancia.to_csv(out/'importancia_caracteristicas.csv', index=False)
    s.guardar_boxplot_importancia(importancia, out/'importancia_caracteristicas_boxplot.png')
    b.guardar_json(out/'importancia_metodologia.json', dict(n_repeticiones=10, random_state=42,
        dataset_sha256=ctx['dataset_sha256'], modelo_final_sha256=b.sha256(out/'modelo_statsmodels.pkl'),
        formula='100 * (RMSE permutado - RMSE original) / RMSE original', evaluacion=ADVERTENCIA))
    r = dict(metodo='bss', etiqueta='Best Subset Selection', salida=out, modelo=modelo,
             metricas=metricas, seleccion=sel, contexto=ctx, grupos=grupos, universo=universo,
             importancias=importancia, particion={'fit': datos['fit'], 'validation': datos['val']})
    r['caracteristicas_md'] = caracteristicas_presentacion(r)
    (out/'caracteristicas_seleccionadas.md').write_text(r['caracteristicas_md'], encoding='utf-8')
    return r


def caracteristicas_presentacion(r):
    filas = [{'Característica original': g, 'Columna transformada': c,
              'Coeficiente OLS [ln(RUB)]': float(r['modelo'].params[c])}
        for g in r['seleccion']['seleccionadas'] for c in r['grupos'][g]]
    return s.resumen_caracteristicas(r['seleccion']['seleccionadas'], r['grupos'], r['modelo'],
        mostrar_orden=False)+'\n\n'+b.tabla_markdown(pd.DataFrame(filas))


def registrar(datos, universo, resultado, nombre_run=NOMBRE_RUN):
    if "'" in nombre_run or not nombre_run.strip():
        raise ValueError('Nombre de Run inválido.')
    cliente = datos['cliente']
    exp = cliente.get_experiment_by_name(b.EXPERIMENTO)
    if exp is None or exp.lifecycle_stage != 'active':
        raise ValueError('Debe existir el experimento activo.')
    if cliente.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=1):
        raise ValueError('El Run ya existe; no sobrescribir. Recuperarlo o utilizar un nombre nuevo.')
    validar_fuentes_actuales(datos, universo)
    validar_resultado(datos, universo, resultado)
    parametros = {**_parametros(datos, universo), 'algorithm': 'Best Subset Selection exhaustivo agrupado',
        'selected_features': json.dumps(resultado['seleccion']['seleccionadas']),
        'selection_stop_reason': resultado['seleccion']['motivo_parada'], 'notebook': '4_Seleccion_Modelos.ipynb',
        'subsets_evaluated': resultado['seleccion']['combinaciones_evaluadas'],
        'tie_break': resultado['seleccion']['desempate']}
    b.guardar_json(resultado['salida']/'configuracion.json', parametros)
    return b.registrar(cliente, resultado['salida'], resultado['contexto'], resultado['metricas'],
                        nombre_run=nombre_run, parametros_extra=parametros)


def validar_resultado(datos, universo, r):
    """Audita exhaustividad y ganadores sin repetir los ajustes de la búsqueda."""
    sel, out = r['seleccion'], r['salida']
    assert r['universo'] == universo and r['grupos'] == universo['grupos']
    assert r['contexto']['dataset_sha256'] == datos['contexto']['dataset_sha256']
    assert r['contexto']['universo_predictoras'] == universo['columnas']
    for clave, esperado in [('fit', datos['fit']), ('validation', datos['val'])]:
        np.testing.assert_array_equal(r['particion'][clave], esperado)
    assert sel['config'] == CONFIG
    m = len(universo['union'])
    h = pd.read_csv(out/HISTORIAL, float_precision='round_trip')
    assert len(h) == sel['combinaciones_evaluadas'] == sel['combinaciones_posibles'] == 2**m
    assert set(h.id_subconjunto) == set(range(2**m)) and h.orden.tolist() == list(range(2**m))
    for k in range(m+1):
        assert h.n_caracteristicas.eq(k).sum() == math.comb(m, k)
    assert sel['modelos_identificables'] == int(h.identificable.sum())
    assert sel['combinaciones_no_nulas'] == 2**m-1
    assert sel['combinaciones_elegibles'] == sum(math.comb(m,k) for k in range(1,min(m,12)+1))
    assert sel['n_grupos_candidatos'] == m and sel['n_columnas_candidatas'] == len(universo['columnas'])
    for fila in h.itertuples():
        nombres = json.loads(fila.grupos)
        assert nombres == [g for i, g in enumerate(universo['union']) if fila.id_subconjunto & (1 << i)]
        assert fila.n_caracteristicas == len(nombres)
        assert fila.n_predictores == len(s.columnas_grupos(r['grupos'], nombres))
        assert fila.identificable == (fila.rango_efectivo == fila.n_predictores+1)
        assert np.isfinite(fila.rss_train_log) and fila.rss_train_log >= 0
    for mejor in sel['mejores_por_tamano']:
        candidatos = h[h.n_caracteristicas.eq(mejor['n_caracteristicas']) & h.identificable]
        ganador = candidatos.sort_values(['rss_train_log', 'orden'], kind='stable').iloc[0]
        assert mejor['id_subconjunto'] == ganador.id_subconjunto
        assert mejor['rss_train_log'] == ganador.rss_train_log
        assert mejor['seleccionadas'] == json.loads(ganador.grupos)
        for clave in ('r2_train_log','r2_adj_train_log','aic_train_log','bic_train_log'):
            if mejor[clave] is not None:
                np.testing.assert_allclose(mejor[clave], ganador[clave], rtol=1e-12, atol=1e-12)
    assert {c['n_caracteristicas'] for c in sel['mejores_por_tamano']} == set(h.loc[h.identificable, 'n_caracteristicas'])
    candidatos = [dict(c, **{'lambda': None}) for c in sel['mejores_por_tamano']
                  if c['n_caracteristicas'] > 0 and c['rmse_validation_original'] is not None]
    ganador, decisiones = gl.elegir_por_complejidad(candidatos, sel['rmse_intercepto_original'])
    assert decisiones == sel['decisiones_complejidad'] and ganador['seleccionadas'] == sel['seleccionadas']
    assert ganador['seleccionadas'] and len(ganador['seleccionadas']) <= 12
    assert ganador['rmse_validation_original'] == sel['rmse_validation_original']
    basal = float(np.sqrt(np.mean((datos['y'].iloc[datos['val']]-np.exp(np.log(datos['y'].iloc[datos['fit']]).mean()))**2)))
    np.testing.assert_allclose(sel['rmse_intercepto_original'], basal, rtol=1e-12)
    columnas = s.columnas_grupos(r['grupos'], sel['seleccionadas'])
    assert r['modelo'].model.exog_names == ['const']+columnas == ['const']+r['contexto']['predictoras']
    assert r['modelo'].model.rank == len(columnas)+1 and r['modelo'].nobs == len(datos['y'])
    with threadpool_limits(limits=1):
        pred = np.exp(r['modelo'].predict(b.sm.add_constant(datos['X'][columnas], has_constant='add')))
    np.testing.assert_allclose(np.sqrt(np.mean((datos['y']-pred)**2)), r['metricas']['train_rmse_original'], rtol=1e-10)
    np.testing.assert_allclose([r['modelo'].rsquared_adj, r['modelo'].aic, r['modelo'].bic],
        [r['metricas'][k] for k in ('train_r2_adj_log', 'train_aic_log', 'train_bic_log')], rtol=1e-10)
    assert r['metricas']['train_n_predictors'] == len(columnas)
    assert r['metricas']['train_n_features_original'] == len(sel['seleccionadas'])
    assert r['metricas']['selection_rmse_validation_original'] == sel['rmse_validation_original']


def ejecutar_presentacion(datos, universo, *, modo='recuperar', nombre_run=NOMBRE_RUN):
    from statsmodels.regression.linear_model import OLSResults
    if modo not in ('recuperar', 'nuevo') or "'" in nombre_run or not nombre_run.strip():
        raise ValueError('Elegir recuperar/nuevo y nombre de Run válido.')
    cliente, out = datos['cliente'], datos['sesion'].carpeta('presentacion_bss')
    if modo == 'nuevo':
        exp = cliente.get_experiment_by_name(b.EXPERIMENTO)
        if exp is None or exp.lifecycle_stage != 'active':
            raise ValueError('Debe existir el experimento activo.')
        if cliente.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{nombre_run}'", max_results=1):
            raise ValueError('El Run ya existe; utilizar un nombre nuevo.')
        tabla, _ = cuadro_selecciones(datos['grupos'], universo['selecciones'])
        resultado = preparar_experimento(datos, tabla, universo, out)
        rid = registrar(datos, universo, resultado, nombre_run)['run_id']
    else:
        rid = None
    validar_fuentes_actuales(datos, universo)
    run = s.recuperar_experimento(cliente, nombre=nombre_run, run_id=rid,
        parametros=_parametros(datos, universo), etiquetas={'dataset_sha256': datos['contexto']['dataset_sha256']})
    archivos = ['contexto.json', 'seleccion.json', 'metricas.json', 'grupos_caracteristicas.json',
        'universo_bss.json', 'modelo_statsmodels.pkl', 'particion_interna.npz', HISTORIAL,
        'inferencia.json', 'evolucion_bss.png', 'importancia_caracteristicas.csv',
        'importancia_caracteristicas_boxplot.png', 'importancia_metodologia.json']
    for nombre in archivos:
        cliente.download_artifacts(run['run_id'], nombre, str(out))
    leer = lambda nombre: json.loads((out/nombre).read_text(encoding='utf-8'))
    with np.load(out/'particion_interna.npz') as p:
        particion = {k: p[k].copy() for k in ('fit', 'validation')}
    r = dict(metodo='bss', etiqueta='Best Subset Selection', modo=modo, run=run, salida=out,
        seleccion=leer('seleccion.json'), contexto=leer('contexto.json'), metricas=leer('metricas.json'),
        grupos=leer('grupos_caracteristicas.json'), universo=leer('universo_bss.json'),
        modelo=OLSResults.load(out/'modelo_statsmodels.pkl'), particion=particion,
        importancias=pd.read_csv(out/'importancia_caracteristicas.csv'))
    assert r['metricas'] == run['metrics']
    assert json.loads(run['params']['selected_features']) == r['seleccion']['seleccionadas']
    validar_resultado(datos, universo, r)
    assert leer('inferencia.json')['columnas_diseno'] == r['modelo'].model.exog_names
    meta = leer('importancia_metodologia.json')
    assert meta['n_repeticiones'] == 10 and meta['random_state'] == 42
    assert meta['dataset_sha256'] == datos['contexto']['dataset_sha256']
    assert meta['modelo_final_sha256'] == b.sha256(out/'modelo_statsmodels.pkl')
    imp = r['importancias']
    assert set(imp.caracteristica) == set(r['seleccion']['seleccionadas'])
    assert imp.groupby('caracteristica').size().eq(10).all()
    assert np.isfinite(imp.select_dtypes(include='number')).all().all()
    np.testing.assert_allclose(imp.rmse_original, r['seleccion']['rmse_validation_original'], rtol=1e-10)
    np.testing.assert_allclose(imp.importancia_pct, 100*(imp.rmse_permutado-imp.rmse_original)/imp.rmse_original, atol=1e-10)
    r['caracteristicas_md'] = caracteristicas_presentacion(r)
    assert b.sha256(datos['base']/datos['contexto']['csv']) == datos['contexto']['dataset_sha256']
    return r


def evolucion_presentacion(r):
    return b.tabla_markdown(pd.DataFrame([{'k': c['n_caracteristicas'], 'Columnas': c['n_predictores'],
        'RSS interno [log]²': b._formato_presentacion(c['rss_train_log']),
        'R² ajustado interno [log]': b._formato_presentacion(c['r2_adj_train_log']),
        'AIC interno [log]': b._formato_presentacion(c['aic_train_log']),
        'BIC interno [log]': b._formato_presentacion(c['bic_train_log']),
        'RMSE validación (RUB)': b._formato_presentacion(c['rmse_validation_original'])}
        for c in r['seleccion']['mejores_por_tamano']]))


def criterios_presentacion(sel):
    filas = [{'Criterio': clave, 'k favorecido': opt['k'], 'Valor': b._formato_presentacion(opt['valor']),
              'Tamaño elegible (1–12)': 1 <= opt['k'] <= 12} for clave, opt in sel['optimos'].items()]
    filas.append({'Criterio': 'Decisión principal: RMSE y mejora ≥0,5 %',
        'k favorecido': len(sel['seleccionadas']), 'Valor': b._formato_presentacion(sel['rmse_validation_original']),
        'Tamaño elegible (1–12)': 1 <= len(sel['seleccionadas']) <= 12})
    return b.tabla_markdown(pd.DataFrame(filas))


def interpretar_bss(r):
    sel, u = r['seleccion'], r['universo']
    excluidas = [g for g in u['union'] if g not in sel['seleccionadas']]
    opt = '; '.join(f"{nombre}: k={sel['optimos'][clave]['k']}" for clave, nombre in
        [('aic_train_log', 'AIC'), ('bic_train_log', 'BIC'), ('r2_adj_train_log', 'R² ajustado')]
        if clave in sel['optimos'])
    return (f"Se evaluaron {sel['combinaciones_evaluadas']:,}/{sel['combinaciones_posibles']:,} combinaciones "
        f"(incluido M0), {sel['modelos_identificables']:,} identificables, sobre {len(u['union'])} grupos / "
        f"{len(u['columnas'])} columnas. Solo {sel['n_modelos_validados']} ganadores por RSS se compararon en validación. "
        f"Elegido: {len(sel['seleccionadas'])} grupos. Excluidas de la unión: {', '.join(excluidas) or 'ninguna'}. "
        f"Óptimos alternativos entre mejores por tamaño (incluye M0 y el tamaño máximo): {opt}. "
        'Los criterios de evolución corresponden al entrenamiento interno; la tabla de métricas del '
        'OLS definitivo corresponde a todo Train. '
        f"Máxima discrepancia RSS optimizado/OLS directo: {sel['max_diferencia_rss_refit']:.2e}.\n\n"
        +s.interpretar_presentacion(r)+'\n\n'+ADVERTENCIA)


def comparar_177(forward, backward, lasso, bss):
    resultados = [('Forward 177', forward), ('Backward 177', backward), ('Group Lasso 177', lasso), ('BSS unión', bss)]
    filas = []
    for nombre, r in resultados:
        met = r['metricas']
        filas.append({'Método': nombre, 'Características candidatas': len(r['grupos']),
            'Columnas candidatas': sum(map(len, r['grupos'].values())),
            'Características seleccionadas': len(r['seleccion']['seleccionadas']),
            'Columnas OLS': int(met['train_n_predictors']), 'R² ajustado Train [log]': f"{met['train_r2_adj_log']:.6f}",
            **{nombre_m: b._formato_presentacion(met[clave]) for clave, nombre_m in
                [('train_rmse_original', 'RMSE Train (RUB)'), ('selection_rmse_validation_original', 'RMSE interno (RUB)'),
                 ('train_aic_log', 'AIC [log]'), ('train_bic_log', 'BIC [log]')]}})
    seleccionadas = set(bss['seleccion']['seleccionadas'])
    comunes = set(bss['universo']['interseccion'])
    lista = lambda c: ', '.join(f'`{g}`' for g in sorted(c)) or 'ninguna'
    textos = [f'Comunes a los tres orígenes conservadas por BSS: {lista(comunes & seleccionadas)}.',
        f'Comunes descartadas por BSS: {lista(comunes-seleccionadas)}.',
        f"Compartidas por exactamente dos que BSS conservó: {lista(set(bss['universo']['exactamente_dos']) & seleccionadas)}.",
        f"Exclusivas de un origen que BSS conservó: {lista(set(bss['universo']['exclusivas']) & seleccionadas)}."]
    for nombre, r in resultados[:-1]:
        conjunto = set(r['seleccion']['seleccionadas'])
        textos.append(f'BSS frente a {nombre}: comparte {lista(conjunto & seleccionadas)}; '
            f'añade {lista(seleccionadas-conjunto)}; omite {lista(conjunto-seleccionadas)}.')
    minimo = min(r['metricas']['selection_rmse_validation_original'] for _, r in resultados)
    ganador = ', '.join(nombre for nombre, r in resultados if r['metricas']['selection_rmse_validation_original'] == minimo)
    cercanos = [(nombre, r) for nombre, r in resultados if r['metricas']['selection_rmse_validation_original'] <= minimo*1.005]
    clave_equilibrio = lambda r: (len(r['seleccion']['seleccionadas']),
        r['metricas']['train_n_predictors'], r['metricas']['selection_rmse_validation_original'])
    mejor_equilibrio = min(clave_equilibrio(r) for _,r in cercanos)
    equilibrio = ', '.join(nombre for nombre,r in cercanos if clave_equilibrio(r) == mejor_equilibrio)
    textos += [f'Menor RMSE interno: {ganador} ({minimo:,.0f} RUB). '
        f'Equilibrio descriptivo entre error y parsimonia: {equilibrio}, el modelo más compacto entre los que '
        'quedan a ≤0,5 % del menor RMSE. Esta regla describe el compromiso; no reemplaza la selección BSS.',
        f"Los tres orígenes partieron de 177 columnas; BSS parte de su unión de {len(bss['grupos'])} grupos / "
        f"{len(bss['universo']['columnas'])} columnas. Los cuatro universos candidatos no son idénticos. "
        'Examinar más combinaciones no garantiza superioridad.', ADVERTENCIA]
    return b.tabla_markdown(pd.DataFrame(filas)), '  \n'.join(textos)
