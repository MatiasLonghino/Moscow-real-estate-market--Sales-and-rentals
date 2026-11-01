"""Trayectoria completa, bloques OHE, partición, identificabilidad y trazabilidad."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import backward_ols as bw, seleccion_modelos as s
from scripts.group_lasso_ols import elegir_por_complejidad


def sinteticos():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(240, 175)), columns=[f'x{i}' for i in range(175)])
    X = X.rename(columns={'x0': 'rooms', 'x1': 'living_area', 'x2': 'to_center_km'})
    cat = np.tile([0, 1, 2], 80)
    X['d_a'], X['d_b'] = (cat == 1).astype(float), (cat == 2).astype(float)
    grupos = {c: [c] for c in list(X)[:14]}
    # Bloques grandes para verificar 177 columnas sin miles de ajustes en la prueba.
    grupos['restantes'] = list(X)[14:175]
    grupos['district'] = ['d_a', 'd_b']
    y = np.exp(10 + .6*X.rooms + .2*X.d_a + rng.normal(0, .1, len(X)))
    fit, val = train_test_split(np.arange(len(X)), test_size=.2, random_state=42)
    return X, y, grupos, fit, val


def test_177_columnas_ohe_y_trayectoria_completa(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    original = bw.b.sm.OLS
    ajustes = []
    def ols(objetivo, diseno, **kw):
        ajustes.append((np.asarray(objetivo), diseno.copy()))
        return original(objetivo, diseno, **kw)
    monkeypatch.setattr(bw.b.sm, 'OLS', ols)
    r = bw.backward_grupos(X, y, grupos, fit, val)
    assert r['inicial']['n_predictores'] == 177
    assert r['inicial']['seleccionadas'] == list(grupos)
    np.testing.assert_allclose(ajustes[1][1][:, 1:], X.iloc[fit])
    assert all(len(a) == len(fit) for a, _ in ajustes)
    np.testing.assert_allclose(ajustes[1][0], np.log(y.iloc[fit]))
    assert r['iteraciones'] == len(grupos)-1 > 12
    assert r['candidatos_evaluados'] == sum(range(2, len(grupos)+1))
    for previo, actual in zip(r['trayectoria'], r['trayectoria'][1:]):
        assert len(previo['seleccionadas'])-len(actual['seleccionadas']) == 1
        assert actual['n_predictores'] == len(s.columnas_grupos(grupos, actual['seleccionadas']))
    assert r['trayectoria'][-1]['n_caracteristicas'] == 1
    assert 1 <= len(r['seleccionadas']) <= 12
    columnas = s.columnas_grupos(grupos, r['seleccionadas'])
    modelo, _ = bw.b.ajustar(X[columnas], np.log(y))
    assert modelo.model.rank == len(columnas)+1
    interno, _ = bw.b.ajustar(X[columnas].iloc[fit], np.log(y.iloc[fit]))
    pred = np.exp(interno.predict(bw.b.sm.add_constant(X[columnas].iloc[val], has_constant='add')))
    np.testing.assert_allclose(r['rmse_validation_original'], np.sqrt(np.mean((y.iloc[val]-pred)**2)))
    candidatos = [dict(c, **{'lambda': None}) for c in r['trayectoria'] if c['identificable']]
    elegido, decisiones = elegir_por_complejidad(candidatos, r['rmse_intercepto_original'])
    assert elegido['seleccionadas'] == r['seleccionadas'] and decisiones == r['decisiones_complejidad']


def test_colinealidad_se_registra_sin_eliminar_columnas():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({'a': rng.normal(size=120)})
    X['duplicada'] = X.a
    y = np.exp(10 + .4*X.a + rng.normal(0, .05, len(X)))
    r = bw.backward_grupos(X, y, {'a': ['a'], 'duplicada': ['duplicada']}, np.arange(90), np.arange(90, 120))
    assert r['inicial']['n_predictores'] == 2 and r['inicial']['rango_efectivo'] == 2
    assert r['inicial']['advertencias'] and not r['inicial']['identificable']
    assert len(r['seleccionadas']) == 1


def test_sin_mejora_no_hay_ganador():
    X, y, grupos, fit, val = sinteticos()
    r = bw.backward_grupos(X, y, grupos, fit, val, min_mejora_pct=100)
    assert r['seleccionadas'] == [] and r['motivo_parada'] == 'ningun_modelo_supera_umbral'
    assert len(r['trayectoria']) == len(grupos)


def test_rechazo_intermedio_no_detiene_tamanos_posteriores():
    candidatos = [dict(seleccionadas=list('abc')[:k], **{'lambda': None}, rmse_validation_original=error)
                  for k, error in [(1, 90), (2, 89.9), (3, 89)]]
    elegido, decisiones = elegir_por_complejidad(candidatos, 100)
    assert [d['aceptado'] for d in decisiones] == [True, False, True]
    assert elegido['seleccionadas'] == list('abc')


def test_universo_independiente_y_misma_particion(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    anterior = dict(base='base', sesion=object(), cliente=object(), fit=fit, val=val,
                    contexto={'dataset_sha256': 'huella'})
    completo = dict(X=X, y=y, grupos=grupos, n_exportadas=177, fit=fit.copy(), val=val.copy(),
                    contexto=anterior['contexto'])
    cargar = Mock(return_value=completo)
    monkeypatch.setattr(s, 'cargar_presentacion', cargar)
    assert bw.cargar_datos(anterior) is completo
    assert cargar.call_args.kwargs['excluidas'] == () and s.EXCLUIDAS == ('to_center_km', 'living_area', 'rooms')
    completo['val'] = val[::-1]
    with pytest.raises(AssertionError):
        bw.cargar_datos(anterior)


def test_run_incompatible_se_rechaza_antes_de_descargar(tmp_path, monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    datos = dict(X=X, y=y, grupos=grupos, fit=fit, val=val, excluidas=(), config_forward=bw.CONFIG,
        cliente=Mock(), sesion=SimpleNamespace(carpeta=lambda _: tmp_path), contexto={'dataset_sha256': 'h'})
    params = {k: str(v) for k, v in s._parametros_presentacion(datos, 'backward').items()}
    params.update(algorithm_version=bw.VERSION, n_candidate_predictors='174')
    monkeypatch.setattr(s, 'recuperar_experimento', lambda *a, **k: {'params': params})
    with pytest.raises(ValueError, match='incompatible'):
        s.ejecutar_presentacion(datos, 'backward')
    datos['cliente'].download_artifacts.assert_not_called()
    datos['cliente'].create_run.assert_not_called()


def test_recuperacion_valida_trayectoria_y_referencia(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    sel = bw.backward_grupos(X, y, grupos, fit, val)
    datos = dict(X=X, grupos=grupos, cliente=Mock(), contexto={'dataset_sha256': 'h'})
    params = dict(algorithm_version=bw.VERSION, reference_run_name=bw.REFERENCIA,
        candidate_columns=json.dumps(list(X)), n_candidate_groups=str(len(grupos)), reference_run_id='ref')
    r = dict(run={'params': params}, seleccion=sel,
             contexto={'reference_run_id': 'ref', 'universo_predictoras': list(X)})
    monkeypatch.setattr(s, 'recuperar_experimento', lambda *a, **k: {'run_id': 'ref', 'params': {'n_predictors': '177'}})
    bw.validar_recuperacion(datos, r)
    alterado = copy.deepcopy(r)
    alterado['seleccion']['candidatos'][0]['seleccionadas'] = []
    with pytest.raises(AssertionError):
        bw.validar_recuperacion(datos, alterado)


def test_comparacion_tres_metodos_y_limitaciones():
    m = dict(train_n_predictors=2, train_r2_adj_log=.8, train_rmse_original=2,
             selection_rmse_validation_original=3, train_aic_log=4, train_bic_log=5)
    f = dict(metricas=m, grupos={c: [c] for c in 'abcd'}, seleccion={'seleccionadas': ['a', 'b']})
    g = dict(metricas=m, grupos=f['grupos'], seleccion={'seleccionadas': ['a', 'c']})
    w = dict(metricas={**m, 'selection_rmse_validation_original': 2.5},
             grupos={c: [c] for c in 'abcde'}, seleccion={'seleccionadas': ['a']})
    tabla, coincidencias = s.comparacion_final_presentacion(f, g, w)
    assert 'Backward Selection' in tabla and 'Comunes a los tres: `a`' in coincidencias
    texto = s.conclusion_presentacion(f, g, w)
    assert 'Backward Selection obtuvo el menor RMSE' in texto
    assert '174' in texto and '177' in texto and 'descriptiva' in texto and 'independiente' in texto


def test_exportacion_registra_configuracion_modelo_particion_y_evidencia(monkeypatch):
    from tempfile import TemporaryDirectory
    X, y, grupos, fit, val = sinteticos()
    cliente = Mock()
    ref = dict(run_id='ref177', experiment_id='5', metrics={}, params={})
    cliente.get_run.return_value = SimpleNamespace(data=SimpleNamespace(metrics={}, params={}))
    ctx = dict(csv='data/train_transformado.csv', dataset_sha256='huella', n_train=len(y),
               reference_run_id='ref177', transformacion_objetivo='ln', predictoras_excluidas=[])
    datos = dict(X=X, y=y, grupos=grupos, fit=fit, val=val, cliente=cliente,
                 config_forward=dict(bw.CONFIG), base='base', excluidas=())
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        monkeypatch.setattr(s, 'preparar_experimento', lambda *a, **k: (X, y, ctx, ref, grupos, out))
        # El exportador OLS común tiene pruebas de serialización propias; aquí se verifica su contrato.
        exportar = Mock()
        monkeypatch.setattr(bw.b, 'exportar', exportar)
        def registrar(client, carpeta, contexto, metricas, *, nombre_run, parametros_extra):
            assert client is cliente and nombre_run == bw.NOMBRE_RUN
            assert parametros_extra['reference_run_id'] == 'ref177'
            assert parametros_extra['reference_run_name'] == bw.REFERENCIA
            assert parametros_extra['n_candidate_predictors'] == 177
            assert parametros_extra['excluded_predictors'] == ''
            assert parametros_extra['permutation_repeats'] == 10
            assert json.loads(parametros_extra['candidate_columns']) == list(X)
            seleccion = json.loads((carpeta/'seleccion.json').read_text(encoding='utf-8'))
            assert seleccion['inicial']['n_predictores'] == 177
            assert json.loads((carpeta/'candidatos_evaluados.json').read_text(encoding='utf-8')) == seleccion['candidatos']
            assert (carpeta/'backward_ols.py').read_bytes() == Path(bw.__file__).read_bytes()
            with np.load(carpeta/'particion_interna.npz') as particion:
                np.testing.assert_array_equal(particion['fit'], fit)
                np.testing.assert_array_equal(particion['validation'], val)
            assert contexto['universo_predictoras'] == list(X) and contexto['n_predictoras_inicial'] == 177
            assert metricas['train_n_features_original'] <= 12
            return {'run_id': 'backward'}
        guardar = Mock(side_effect=registrar)
        monkeypatch.setattr(bw.b, 'registrar', guardar)
        resultado = bw.ejecutar_experimento(datos, out)
        assert resultado['registro']['run_id'] == 'backward' and guardar.call_count == 1
        args = exportar.call_args.args
        modelo, diseno = args[1:3]
        assert modelo.model.rank == diseno.shape[1] and modelo.nobs == len(y)
        assert list(diseno.columns) == ['const']+s.columnas_grupos(grupos, resultado['seleccion']['seleccionadas'])
