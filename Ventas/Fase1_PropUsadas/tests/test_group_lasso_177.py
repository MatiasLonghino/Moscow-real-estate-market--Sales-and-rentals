"""Group Lasso 177: universo, grilla completa, KKT, referencia y persistencia."""
import copy
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import group_lasso_ols as gl, seleccion_modelos as s


@pytest.fixture(scope='module')
def experimento():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(360, 175)), columns=[f'x{i}' for i in range(175)])
    X = X.rename(columns={'x0': 'rooms', 'x1': 'living_area', 'x2': 'to_center_km'})
    cat = np.tile([0, 1, 2], 120)
    X['d_a'], X['d_b'] = (cat == 1).astype(float), (cat == 2).astype(float)
    grupos = {c: [c] for c in list(X)[:23]}
    grupos.update(district=['d_a', 'd_b'], otras=list(X)[23:175])
    y = np.exp(10+.5*X.rooms+.25*X.to_center_km+.2*X.d_a+
               .025*X.iloc[:, 3:23].sum(axis=1)+rng.normal(0, .04, len(X)))
    fit, val = train_test_split(np.arange(len(X)), test_size=.2, random_state=42)
    llamadas = []
    resolver = gl.resolver_group_lasso
    def resolver_auditado(*a, **kw):
        llamadas.append(kw.copy())
        return resolver(*a, **kw)
    with TemporaryDirectory() as tmp:
        with patch.object(gl, 'resolver_group_lasso', side_effect=resolver_auditado):
            sel = gl.buscar_group_lasso(X, y, grupos, fit, val, tmp, excluidas=())
        yield dict(X=X, y=y, grupos=grupos, fit=fit, val=val, seleccion=sel, salida=Path(tmp), llamadas=llamadas)


def test_177_columnas_40_lambdas_parametros_y_grupos_completos(experimento):
    e, sel = experimento, experimento['seleccion']
    assert e['X'].shape[1] == sum(map(len, e['grupos'].values())) == 177
    assert set(s.EXCLUIDAS) <= set(e['X']) and s.EXCLUIDAS == ('to_center_km', 'living_area', 'rooms')
    assert sel['config'] == gl.CONFIG and len(sel['historial']) == len(e['llamadas']) == 40
    assert all(c['tol'] == 1e-7 and c['max_iter'] == 20000 for c in e['llamadas'])
    np.testing.assert_allclose(sel['lambdas'], sel['lambda_max']*np.geomspace(1, .001, 40))
    assert any(h['n_caracteristicas'] > 12 for h in sel['historial'])
    for paso in sel['historial']:
        assert paso['valido'] == (0 < paso['n_caracteristicas'] <= 12)
        cols = s.columnas_grupos(e['grupos'], paso['seleccionadas'])
        assert not ({'d_a', 'd_b'} & set(cols)) or {'d_a', 'd_b'} <= set(cols)
        assert paso['convergencia'] and paso['residuo_kkt_relativo'] <= 1e-7
    assert 1 <= len(sel['seleccionadas']) <= 12


def test_escalador_particion_rangos_y_seleccion_por_complejidad(experimento):
    e, sel = experimento, experimento['seleccion']
    scaler = joblib.load(e['salida']/'scaler_seleccion.joblib')
    np.testing.assert_allclose(scaler.mean_, e['X'].iloc[e['fit']].mean())
    assert not np.allclose(scaler.mean_, e['X'].mean())
    with np.load(e['salida']/'particion_interna.npz') as p:
        fit, val = train_test_split(np.arange(len(e['y'])), test_size=.2, random_state=42)
        np.testing.assert_array_equal(p['fit'], fit)
        np.testing.assert_array_equal(p['validation'], val)
    diag = json.loads((e['salida']/'estabilidad_numerica.json').read_text(encoding='utf-8'))
    assert diag['rango_gram'] == diag['n_columnas'] == 177 and not diag['advertencias']
    candidatos = [dict(seleccionadas=h['seleccionadas'], **{'lambda': h['lambda']},
        rmse_validation_original=h['rmse_validation_original']) for h in sel['historial'] if h['valido']]
    elegido, decisiones = gl.elegir_por_complejidad(candidatos, sel['rmse_intercepto_original'])
    assert decisiones == sel['decisiones_complejidad'] and elegido['seleccionadas'] == sel['seleccionadas']


def test_recuperacion_kkt_sin_entrenar_y_coeficientes_alterados(experimento):
    e = experimento
    ctx = dict(universo_predictoras=list(e['X']), n_predictoras_inicial=177)
    r = dict(seleccion=e['seleccion'], salida=e['salida'], contexto=ctx)
    with patch.object(gl, 'resolver_group_lasso', side_effect=AssertionError('No optimizar')), \
         patch.object(gl.b, 'ajustar', side_effect=AssertionError('No ajustar')):
        gl.validar_recuperacion_177(e, r)
    with TemporaryDirectory() as tmp:
        carpeta = Path(tmp)/'alterado'
        shutil.copytree(e['salida'], carpeta)
        archivo = carpeta/'ruta_coeficientes_group_lasso.npz'
        with np.load(archivo) as ruta:
            contenido = {k: ruta[k].copy() for k in ruta.files}
        contenido['coeficientes'][0, 0] += .1
        np.savez_compressed(archivo, **contenido)
        with pytest.raises(AssertionError):
            gl.validar_recuperacion_177(e, {**r, 'salida': carpeta})


def test_rango_deficiente_se_documenta_sin_eliminar_columnas(experimento):
    e = experimento
    X = e['X'].copy()
    X['rooms'] = X.to_center_km
    with TemporaryDirectory() as tmp:
        with pytest.raises(AssertionError, match='Diseño interno no identificable'):
            gl.buscar_group_lasso(X, e['y'], e['grupos'], e['fit'], e['val'], tmp, excluidas=())
        diag = json.loads((Path(tmp)/'estabilidad_numerica.json').read_text(encoding='utf-8'))
        assert diag['n_columnas'] == 177 and diag['rango_gram'] == 176 and diag['advertencias']


def test_referencia_obligatoria_no_sustituye_forward_historico(experimento, monkeypatch):
    e = experimento
    with pytest.raises(ValueError, match='Forward_04'):
        gl.validar_referencia_177(e, None)
    validar = Mock()
    monkeypatch.setattr(s, '_validar_resultado_presentacion', validar)
    ref = dict(metodo='forward', run={'params': {k: str(v) for k, v in s._parametros_forward_177(e).items()}})
    ref['run']['params']['n_candidate_predictors'] = '174'
    with pytest.raises(ValueError, match='incompatible'):
        gl.validar_referencia_177(e, ref)
    validar.assert_called_once()


@pytest.mark.parametrize('clave,valor', [('n_candidate_predictors', '174'), ('candidate_columns', '[]'),
    ('reference_run_name', s.NOMBRE_RUN), ('reference_run_id', 'hist174'), ('n_lambdas', '39')])
def test_incompatibilidad_antes_de_descargar(tmp_path, experimento, monkeypatch, clave, valor):
    e = experimento
    carpetas = []
    datos = {**e, 'excluidas': (), 'config_forward': s.CONFIG_FORWARD, 'config_lasso': gl.CONFIG,
        'contexto': {'dataset_sha256': 'h'}, 'cliente': Mock(),
        'sesion': SimpleNamespace(carpeta=lambda n: carpetas.append(n) or tmp_path)}
    anterior = dict(run={'run_id': 'forward177'})
    params = {k: str(v) for k, v in {**s._parametros_presentacion(datos, 'lasso'),
                                   **gl.parametros_177(datos, anterior)}.items()}
    params[clave] = valor
    monkeypatch.setattr(gl, 'validar_referencia_177', lambda *a: None)
    monkeypatch.setattr(s, 'recuperar_experimento', lambda *a, **kw: {'params': params})
    with pytest.raises(ValueError, match='incompatible'):
        s.ejecutar_presentacion(datos, 'lasso', anterior=anterior, nombre_run=gl.NOMBRE_RUN_177)
    assert carpetas == ['presentacion_lasso_177']
    datos['cliente'].download_artifacts.assert_not_called()
    datos['cliente'].create_run.assert_not_called()


def test_persistencia_ols_no_penalizado_y_metadatos(experimento, monkeypatch):
    e = experimento
    cliente = Mock()
    cliente.search_runs.return_value = []
    ref = dict(run_name=s.NOMBRE_RUN_177, run_id='forward177', experiment_id='5', metrics={}, params={'random_state': '42'})
    cliente.get_run.return_value = SimpleNamespace(info=SimpleNamespace(status='FINISHED'),
        data=SimpleNamespace(metrics={}, params=ref['params']))
    with TemporaryDirectory() as tmp:
        out = Path(tmp)/'nuevo'
        shutil.copytree(e['salida'], out)
        monkeypatch.setattr(gl.b, 'exportar', lambda *a, **k: out)
        r = gl.entrenar_exportar('base', e['X'], e['y'], e['grupos'], e['seleccion'],
            {'dataset_sha256': 'h'}, ref, out, excluidas=(), nombre_run=gl.NOMBRE_RUN_177)
        cols = s.columnas_grupos(e['grupos'], e['seleccion']['seleccionadas'])
        directo, _ = gl.b.ajustar(e['X'][cols], np.log(e['y']))
        np.testing.assert_allclose(r['modelo'].params, directo.params)
        assert r['modelo'].model.rank == len(cols)+1 and r['modelo'].nobs == len(e['y'])
        def registrar(c, carpeta, ctx, metricas, *, nombre_run, parametros_extra):
            assert c is cliente and nombre_run == gl.NOMBRE_RUN_177
            esperado = gl.parametros_177(e, {'run': ref})
            for k, v in esperado.items(): assert parametros_extra[k] == v
            assert (carpeta/'ruta_coeficientes_group_lasso.npz').is_file()
            assert ctx['universo_predictoras'] == list(e['X']) and ctx['n_predictoras_inicial'] == 177
            return {'run_id': 'nuevo'}
        monkeypatch.setattr(gl.b, 'registrar', registrar)
        assert gl.registrar(cliente, r, e['seleccion'], ref, nombre_run=gl.NOMBRE_RUN_177)['run_id'] == 'nuevo'


def test_comparacion_177_y_general_cinco_modelos():
    m = dict(train_n_predictors=3, train_r2_adj_log=.8, train_rmse_original=2,
             selection_rmse_validation_original=3, train_aic_log=4, train_bic_log=5)
    f = dict(metricas=m, grupos={c: [c] for c in ('rooms', 'living_area', 'to_center_km')},
             seleccion={'seleccionadas': ['rooms']})
    g = {**f, 'metricas': {**m, 'selection_rmse_validation_original': 2.5},
         'seleccion': {'seleccionadas': ['rooms', 'living_area']}}
    tabla, coincidencias, analisis = s.comparacion_177_presentacion(f, g, f)
    assert 'RMSE interno (RUB)' in tabla and 'living_area' in coincidencias
    assert 'Group Lasso + OLS — 177: sí' in analisis and 'no constituye una evaluación independiente' in analisis
    tabla, texto = s.comparacion_final_presentacion(f, f, f, forward_177=f, lasso_177=g)
    assert 'Comunes a los 5' in texto and 'Group Lasso + OLS — 174' in tabla and 'Group Lasso + OLS — 177' in tabla
    assert 'Group Lasso + OLS — 177 predictoras obtuvo el menor RMSE' in s.conclusion_presentacion(f, f, f, forward_177=f, lasso_177=g)
