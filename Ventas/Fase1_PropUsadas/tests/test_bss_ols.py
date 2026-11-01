"""BSS clásico: exhaustividad por grupos, RSS interno, recuperación y trazabilidad."""
import copy
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bss_ols as bs, seleccion_modelos as s, group_lasso_ols as gl, backward_ols as bw


def datos_pequenos():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(180, 3)), columns=['a', 'b', 'c'])
    cat = np.tile([0, 1, 2], 60)
    X['d_a'], X['d_b'] = (cat == 1).astype(float), (cat == 2).astype(float)
    y = np.exp(10+.4*X.a+.15*X.b+.2*X.d_a+rng.normal(0, .04, len(X)))
    grupos = {'a': ['a'], 'b': ['b'], 'c': ['c'], 'district': ['d_a', 'd_b']}
    fit, val = train_test_split(np.arange(len(X)), test_size=.2, random_state=42)
    return X, y, grupos, fit, val


def test_todos_los_subconjuntos_equivalen_a_ols_directo_y_mejores_por_rss():
    X, y, grupos, fit, val = datos_pequenos()
    sel, h = bs.buscar_bss(X, y, grupos, fit, val)
    assert len(h) == 16 and set(h.id_subconjunto) == set(range(16)) and h.identificable.all()
    for fila in h.itertuples():
        seleccionadas = json.loads(fila.grupos)
        cols = s.columnas_grupos(grupos, seleccionadas)
        modelo, _ = s.b.ajustar(X[cols].iloc[fit], np.log(y.iloc[fit]))
        np.testing.assert_allclose(fila.rss_train_log, modelo.ssr, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose([fila.r2_train_log, fila.r2_adj_train_log, fila.aic_train_log, fila.bic_train_log],
            [modelo.rsquared, modelo.rsquared_adj, modelo.aic, modelo.bic], rtol=1e-9, atol=1e-9)
        assert not ({'d_a', 'd_b'} & set(cols)) or {'d_a', 'd_b'} <= set(cols)
    for ganador in sel['mejores_por_tamano']:
        candidatos = h[h.n_caracteristicas == ganador['n_caracteristicas']]
        assert ganador['id_subconjunto'] == candidatos.sort_values(['rss_train_log', 'orden']).iloc[0].id_subconjunto
    assert sel['n_modelos_validados'] == 5
    assert sel['decisiones_complejidad'][0]['rmse_referencia'] == sel['rmse_intercepto_original']


def test_validacion_no_elige_ni_desempata_el_mejor_de_un_tamano():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(240, 2)), columns=['a', 'b'])
    fit, val = np.arange(180), np.arange(180, 240)
    log = pd.Series(10+.7*X.a)
    log.iloc[val] = 10+.7*X.b.iloc[val]
    y = np.exp(log)
    sel, h = bs.buscar_bss(X, y, {'a': ['a'], 'b': ['b']}, fit, val)
    assert next(c for c in sel['mejores_por_tamano'] if c['n_caracteristicas'] == 1)['seleccionadas'] == ['a']
    errores = {}
    for g in ('a', 'b'):
        modelo, _ = s.b.ajustar(X[[g]].iloc[fit], np.log(y.iloc[fit]))
        pred = np.exp(modelo.predict(s.b.sm.add_constant(X[[g]].iloc[val], has_constant='add')))
        errores[g] = np.sqrt(np.mean((y.iloc[val]-pred)**2))
    assert errores['b'] < errores['a']  # Aun así, BSS debe conservar a como ganador de tamaño 1.
    y_alternativo = y.copy()
    y_alternativo.iloc[val] *= 1.8
    otro, hh = bs.buscar_bss(X, y_alternativo, {'a': ['a'], 'b': ['b']}, fit, val)
    pd.testing.assert_frame_equal(h, hh)
    assert [c['id_subconjunto'] for c in otro['mejores_por_tamano']] == [c['id_subconjunto'] for c in sel['mejores_por_tamano']]


def test_13_grupos_8192_combinaciones_modelo_13_no_elegible_y_todos_los_tamanos():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(160, 13)), columns=[f'g{i}' for i in range(13)])
    y = np.exp(10+.3*X.g0+.06*X.iloc[:, 1:].sum(axis=1)+rng.normal(0, .05, len(X)))
    fit, val = train_test_split(np.arange(len(X)), test_size=.2, random_state=42)
    sel, h = bs.buscar_bss(X, y, {c: [c] for c in X}, fit, val)
    assert sel['combinaciones_evaluadas'] == sel['combinaciones_posibles'] == 8192
    assert sel['combinaciones_no_nulas'] == 8191 and sel['combinaciones_elegibles'] == 8190
    assert sel['modelos_identificables'] == 8192 and sel['n_modelos_validados'] == 14
    assert [c['n_caracteristicas'] for c in sel['mejores_por_tamano']] == list(range(14))
    assert [d['n_caracteristicas'] for d in sel['decisiones_complejidad']] == list(range(1,13))
    assert len(sel['seleccionadas']) <= 12 and h.n_caracteristicas.eq(13).sum() == 1
    for k in range(14): assert h.n_caracteristicas.eq(k).sum() == math.comb(13, k)
    assert any(not d['aceptado'] for d in sel['decisiones_complejidad'])


def test_rango_deficiente_no_es_ganador_y_desempate_exacto_por_linaje():
    rng = np.random.default_rng(4)
    X = pd.DataFrame({'primera': rng.normal(size=120)})
    X['copia'] = X.primera
    y = np.exp(10+.4*X.primera+rng.normal(0, .03, len(X)))
    sel, h = bs.buscar_bss(X, y, {c: [c] for c in X}, np.arange(90), np.arange(90,120))
    assert h.iloc[-1].n_predictores == 2 and h.iloc[-1].rango_efectivo == 2
    assert not h.iloc[-1].identificable and sel['modelos_identificables'] == 3
    assert [c['n_caracteristicas'] for c in sel['mejores_por_tamano']] == [0,1]
    assert sel['seleccionadas'] == ['primera']
    assert h[h.n_caracteristicas.eq(1)].rss_train_log.nunique() == 1


def test_sin_mejora_no_se_inventa_un_ganador():
    X, y, grupos, fit, val = datos_pequenos()
    sel, h = bs.buscar_bss(X, y, grupos, fit, val, min_mejora_pct=100)
    assert len(h) == 16 and sel['seleccionadas'] == [] and sel['motivo_parada'] == 'ningun_modelo_supera_umbral'


def test_union_dinamica_13_grupos_146_columnas_y_cuadro_de_coincidencias():
    comunes = ['building_type','date_posted_dt','district','metro_distance_min','metro_distance_type','renovation','total_area']
    parciales = ['ceiling_height','has_balcony','surface_corrected','to_center_km']
    grupos = {g: [g] for g in comunes+parciales+['rooms','living_area','nunca_elegida']}
    grupos.update(district=[f'd{i}' for i in range(128)], building_type=[f'b{i}' for i in range(5)],
                  renovation=[f'r{i}' for i in range(3)])
    selecciones = {'Forward 177': comunes+['rooms'], 'Backward 177': comunes+parciales+['rooms'],
                   'Group Lasso 177': comunes+parciales+['living_area']}
    tabla, u = bs.cuadro_selecciones(grupos, selecciones)
    assert len(u['union']) == 13 and len(u['columnas']) == 146
    assert len(u['interseccion']) == 7 and len(u['exactamente_dos']) == 5 and len(u['exclusivas']) == 1
    assert 'nunca_elegida' not in u['union'] and 'living_area' in u['union']
    assert tabla['Número de métodos'].sum() == 32
    md, texto = bs.cuadro_presentacion(tabla,u)
    assert '**`district`**' in md and 'Unión: 13 características / 146 columnas' in texto
    # La implementación no depende de estos nombres concretos.
    t, nuevo = bs.cuadro_selecciones({'nuevo': ['n']}, {'A': ['nuevo'], 'B': [], 'C': []})
    assert nuevo['union'] == ['nuevo'] and t.iloc[0]['Clasificación de coincidencia'] == 'Solo A'


def test_recuperacion_de_los_tres_origenes_verifica_nombres_ids_y_universo(monkeypatch):
    X = pd.DataFrame(np.zeros((10,177)), columns=[f'x{i}' for i in range(177)])
    grupos = {g: [g] for g in X}
    datos = dict(X=X, grupos=grupos, excluidas=(), config_forward=bs.CONFIG, config_lasso=gl.CONFIG,
                 cliente=Mock(), contexto={'dataset_sha256': 'h'})
    resultados = []
    for metodo, nombre, seleccion in [('forward',s.NOMBRE_RUN_177,['x0']),
        ('backward',bw.NOMBRE_RUN,['x0','x1']),('lasso',gl.NOMBRE_RUN_177,['x0','x2'])]:
        resultados.append(dict(metodo=metodo, run=dict(run_name=nombre, run_id=metodo, params={}, metrics={}),
                               seleccion={'seleccionadas': seleccion}))
    validar = Mock()
    monkeypatch.setattr(s,'_validar_resultado_presentacion',validar)
    monkeypatch.setattr(gl,'validar_referencia_177',lambda *a: None)
    monkeypatch.setattr(gl,'validar_recuperacion_177',lambda *a: None)
    monkeypatch.setattr(bw,'validar_recuperacion',lambda *a: None)
    recuperar = Mock(side_effect=lambda *a, **k: next(r['run'] for r in resultados if r['run']['run_id']==k['run_id']))
    monkeypatch.setattr(s,'recuperar_experimento',recuperar)
    _, u = bs.construir_universo(datos,*resultados)
    assert u['union']==['x0','x1','x2'] and validar.call_count==3
    assert [c.kwargs['nombre'] for c in recuperar.call_args_list]==[s.NOMBRE_RUN_177,bw.NOMBRE_RUN,gl.NOMBRE_RUN_177]
    assert all(c.kwargs['parametros']=={'n_candidate_predictors':177} for c in recuperar.call_args_list)


def test_preparacion_reajuste_final_persistencia_y_auditoria(monkeypatch):
    X, y, grupos, fit, val = datos_pequenos()
    datos = dict(X=X,y=y,grupos=grupos,fit=fit,val=val,config_forward=bs.CONFIG,excluidas=(),
        contexto=dict(dataset_sha256='h',csv='data/train_transformado.csv',n_train=len(y)),base='base',cliente=Mock())
    tabla, u = bs.cuadro_selecciones(grupos,{'Forward': ['a','district'], 'Backward': ['a','b'], 'Lasso': ['a','b','district']})
    u.update(origenes={k:{'run':{'run_id':k}} for k in ('Forward','Backward','Lasso')},dataset_sha256='h')
    # Los diagnósticos, flavors y serialización completos se prueban en el exportador común.
    def exportar(base, modelo, diseno, objetivo, contexto, metricas, *a, carpeta_salida, **kw):
        modelo.save(carpeta_salida/'modelo_statsmodels.pkl')
    monkeypatch.setattr(bs.b,'exportar',exportar)
    with TemporaryDirectory() as tmp:
        out=Path(tmp)
        r=bs.preparar_experimento(datos,tabla,u,out)
        cols=s.columnas_grupos(r['grupos'],r['seleccion']['seleccionadas'])
        directo,_=s.b.ajustar(X[cols],np.log(y))
        np.testing.assert_allclose(r['modelo'].params,directo.params)
        assert r['modelo'].nobs==len(y) and r['modelo'].model.rank==len(cols)+1
        assert r['importancias'].groupby('caracteristica').size().eq(10).all()
        with patch.object(bs,'buscar_bss',side_effect=AssertionError('No repetir búsqueda')), \
             patch.object(bs.b,'ajustar',side_effect=AssertionError('No ajustar')):
            bs.validar_resultado(datos,u,r)
        assert (out/'evolucion_bss.png').is_file() and (out/'cuadro_caracteristicas.csv').is_file()
        assert (out/'bss_ols.py').read_bytes()==Path(bs.__file__).read_bytes()
        cliente=datos['cliente']
        cliente.get_experiment_by_name.return_value=SimpleNamespace(experiment_id='5',lifecycle_stage='active')
        cliente.search_runs.return_value=[]
        def registrar(c, carpeta, ctx, met, *, nombre_run, parametros_extra):
            assert nombre_run==bs.NOMBRE_RUN and parametros_extra['within_size_metric']=='RSS internal_train ln(RUB)'
            assert parametros_extra['source_universe_predictors']==177
            assert json.loads(parametros_extra['candidate_features'])==u['union']
            assert json.loads(parametros_extra['reference_run_ids'])=={k:k for k in ('Forward','Backward','Lasso')}
            with np.load(carpeta/'particion_interna.npz') as p:
                np.testing.assert_array_equal(p['fit'],fit)
                np.testing.assert_array_equal(p['validation'],val)
            return {'run_id':'bss'}
        monkeypatch.setattr(bs.b,'registrar',registrar)
        monkeypatch.setattr(bs,'validar_fuentes_actuales',lambda *a: None)
        assert bs.registrar(datos,u,r)['run_id']=='bss'
        h=pd.read_csv(out/bs.HISTORIAL)
        h.loc[0,'id_subconjunto']=1
        h.to_csv(out/bs.HISTORIAL,index=False,compression='zip')
        with pytest.raises(AssertionError): bs.validar_resultado(datos,u,r)


def test_nombre_existente_y_referencia_distinta_no_se_reutilizan(tmp_path,monkeypatch):
    cliente=Mock()
    cliente.get_experiment_by_name.return_value=SimpleNamespace(experiment_id='5',lifecycle_stage='active')
    cliente.search_runs.return_value=[object()]
    datos=dict(cliente=cliente,sesion=SimpleNamespace(carpeta=lambda _:tmp_path))
    preparar=Mock()
    monkeypatch.setattr(bs,'preparar_experimento',preparar)
    with pytest.raises(ValueError,match='nombre nuevo'): bs.ejecutar_presentacion(datos,{},modo='nuevo')
    preparar.assert_not_called()
    cliente.create_run.assert_not_called()


def test_recuperacion_exige_origenes_union_y_configuracion_antes_de_descargar(tmp_path,monkeypatch):
    X,y,grupos,fit,val=datos_pequenos()
    _,u=bs.cuadro_selecciones(grupos,{'F':['a'],'B':['a','b'],'G':['b']})
    u['origenes']={k:{'run':{'run_id':k}} for k in ('F','B','G')}
    datos=dict(X=X,y=y,config_forward=bs.CONFIG,excluidas=(),contexto={'dataset_sha256':'h'},
        cliente=Mock(),sesion=SimpleNamespace(carpeta=lambda _:tmp_path))
    def recuperar(cliente,**kw):
        assert json.loads(kw['parametros']['reference_run_ids'])=={'F':'F','B':'B','G':'G'}
        assert json.loads(kw['parametros']['candidate_features'])==['a','b']
        assert kw['parametros']['algorithm_version']==bs.VERSION
        raise ValueError('Referencia incompatible')
    monkeypatch.setattr(s,'recuperar_experimento',recuperar)
    monkeypatch.setattr(bs,'validar_fuentes_actuales',lambda *a: None)
    with pytest.raises(ValueError,match='incompatible'): bs.ejecutar_presentacion(datos,u)
    datos['cliente'].download_artifacts.assert_not_called()


def test_fuentes_actuales_rechazan_cambios_en_un_run(monkeypatch):
    origenes = {k:{'run':dict(run_id=k,params={'n_candidate_predictors':'177'},metrics={'m':1.})}
                for k in ('Forward 177','Backward 177','Group Lasso 177')}
    datos = dict(cliente=Mock(),contexto={'dataset_sha256':'h'})
    vivo = Mock(side_effect=lambda c,**kw: copy.deepcopy(origenes[kw['run_id']]['run']))
    monkeypatch.setattr(s,'recuperar_experimento',vivo)
    bs.validar_fuentes_actuales(datos,{'origenes':origenes})
    assert vivo.call_count==3
    monkeypatch.setattr(s,'recuperar_experimento',lambda *a,**kw: {'params':{},'metrics':{}})
    with pytest.raises(AssertionError): bs.validar_fuentes_actuales(datos,{'origenes':origenes})


def test_comparacion_bss_reconoce_empates_y_universo_derivado():
    metricas = dict(train_n_predictors=2,train_r2_adj_log=.9,train_rmse_original=2,
        selection_rmse_validation_original=3,train_aic_log=4,train_bic_log=5)
    origen = dict(metricas=metricas,grupos={'a':['a'],'b':['b'],'c':['c']},seleccion={'seleccionadas':['a','b']})
    r = dict(metricas=metricas,grupos={'a':['a'],'b':['b']},seleccion={'seleccionadas':['a','b']},
        universo=dict(interseccion=['a'],exactamente_dos=['b'],exclusivas=[],columnas=['a','b']))
    tabla,texto = bs.comparar_177(origen,origen,origen,r)
    assert 'Características candidatas' in tabla and 'BSS unión' in tabla
    assert 'Equilibrio descriptivo entre error y parsimonia: Forward 177, Backward 177, Group Lasso 177, BSS unión' in texto
    assert 'universos candidatos no son idénticos' in texto and 'sesgo por selección' in texto
    general, coincidencias = s.comparacion_final_presentacion(origen,origen,origen,origen,origen,bss=r)
    assert 'Best Subset Selection' in general and 'Comunes a los 6' in coincidencias
