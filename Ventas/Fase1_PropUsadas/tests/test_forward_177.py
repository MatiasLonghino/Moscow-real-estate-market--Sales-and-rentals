"""Forward independiente de 177 columnas: algoritmo común y persistencia estricta."""
import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import seleccion_modelos as s


def sinteticos():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(260, 175)), columns=[f'x{i}' for i in range(175)])
    X = X.rename(columns={'x0': 'rooms', 'x1': 'to_center_km', 'x2': 'living_area'})
    cat = np.tile([0, 1, 2, 0], 65)
    X['d_a'], X['d_b'] = (cat == 1).astype(float), (cat == 2).astype(float)
    grupos = {c: [c] for c in list(X)[:3]}
    grupos.update(district=['d_a', 'd_b'], otras=list(X)[3:175])
    y = np.exp(10+.5*X.rooms+.25*X.to_center_km+.25*X.d_a+rng.normal(0, .03, len(X)))
    fit, val = train_test_split(np.arange(len(X)), test_size=.2, random_state=42)
    return X, y, grupos, fit, val


def test_carga_explicita_sin_exclusiones_y_particion_identica(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    previos = dict(base='base', sesion=object(), cliente=object(), fit=fit, val=val,
                   contexto={'dataset_sha256': 'h'})
    completos = dict(X=X, y=y, grupos=grupos, fit=fit.copy(), val=val.copy(), n_exportadas=177,
                     excluidas=(), contexto=previos['contexto'])
    cargar = Mock(return_value=completos)
    monkeypatch.setattr(s, 'cargar_presentacion', cargar)
    assert s.cargar_forward_177(previos) is completos
    assert cargar.call_args.kwargs['excluidas'] == ()
    assert s.EXCLUIDAS == ('to_center_km', 'living_area', 'rooms')
    completos['fit'] = fit[::-1]
    with pytest.raises(AssertionError):
        s.cargar_forward_177(previos)


def test_177_candidatas_grupos_completos_y_orden_sin_eliminaciones(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    ols = s.b.sm.OLS
    tamanos = []
    def ajustar(objetivo, diseno, **kwargs):
        assert len(objetivo) == len(fit)
        tamanos.append(diseno.shape[1])
        return ols(objetivo, diseno, **kwargs)
    monkeypatch.setattr(s.b.sm, 'OLS', ajustar)
    sel = s.forward_grupos(X, np.log(y), y, grupos, fit, val)
    assert tamanos[0] == 1  # Solo intercepto.
    assert [c['candidata'] for c in sel['candidatos'] if c['iteracion'] == 1] == list(grupos)
    assert sum(map(len, grupos.values())) == 177
    anteriores = []
    for paso in sel['historial']:
        actuales = paso['caracteristicas_acumuladas']
        assert actuales[:len(anteriores)] == anteriores
        assert len(actuales)-len(anteriores) == int(paso['aceptada'])
        assert paso['n_predictores'] == len(s.columnas_grupos(grupos, actuales))
        if paso['aceptada']:
            assert paso['mejora_pct'] >= .5
        anteriores = actuales
    assert 0 < len(sel['seleccionadas']) <= 12
    cols = s.columnas_grupos(grupos, sel['seleccionadas'])
    if 'district' in sel['seleccionadas']:
        assert {'d_a', 'd_b'} <= set(cols)
    assert sel['historial'][-1]['aceptada'] is False or sel['motivo_parada'] == 'candidatas_agotadas'


def test_recuperacion_audita_historial_y_referencia_sin_entrenar(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    sel = s.forward_grupos(X, np.log(y), y, grupos, fit, val)
    datos = dict(X=X, grupos=grupos, config_forward=s.CONFIG_FORWARD,
                 cliente=Mock(), contexto={'dataset_sha256': 'h'})
    resultado = dict(seleccion=sel, contexto=dict(universo_predictoras=list(X), n_predictoras_inicial=177,
        reference_run_id='ref'), run={'params': dict(reference_run_id='ref', feature_order=json.dumps(sel['seleccionadas']),
                                                    selection_stop_reason=sel['motivo_parada'])})
    recuperar = Mock(return_value={'run_id': 'ref'})
    monkeypatch.setattr(s, 'recuperar_experimento', recuperar)
    monkeypatch.setattr(s.b, 'ajustar', Mock(side_effect=AssertionError('No reentrenar')))
    s._validar_forward_177(datos, resultado)
    assert recuperar.call_args.kwargs['nombre'] == 'OLS_02_Log_Target'
    assert recuperar.call_args.kwargs['parametros']['n_predictors'] == 177
    alterado = copy.deepcopy(resultado)
    alterado['seleccion']['historial'][0]['caracteristicas_acumuladas'] = []
    with pytest.raises(AssertionError):
        s._validar_forward_177(datos, alterado)


@pytest.mark.parametrize('clave,valor', [('n_candidate_predictors', '174'), ('excluded_predictors', 'rooms'),
    ('reference_run_name', 'OLS_05_Log_Sin_rooms'), ('candidate_columns', '[]'), ('permutation_repeats', '10')])
def test_rechaza_run_incompatible_antes_de_descargar(tmp_path, monkeypatch, clave, valor):
    X, y, grupos, fit, val = sinteticos()
    carpetas = []
    datos = dict(X=X, y=y, grupos=grupos, fit=fit, val=val, excluidas=(), config_forward=s.CONFIG_FORWARD,
        cliente=Mock(), sesion=SimpleNamespace(carpeta=lambda n: carpetas.append(n) or tmp_path),
        contexto={'dataset_sha256': 'h'})
    params = {k: str(v) for k, v in {**s._parametros_presentacion(datos, 'forward'),
                                    **s._parametros_forward_177(datos)}.items()}
    params[clave] = valor
    monkeypatch.setattr(s, 'recuperar_experimento', lambda *a, **k: {'params': params})
    with pytest.raises(ValueError, match='incompatible'):
        s.ejecutar_presentacion(datos, 'forward', nombre_run=s.NOMBRE_RUN_177)
    assert carpetas == ['presentacion_forward_177']
    datos['cliente'].download_artifacts.assert_not_called()
    datos['cliente'].create_run.assert_not_called()


def test_persistencia_nueva_reutiliza_algoritmo_y_ols_comunes(monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    cliente = Mock()
    cliente.search_runs.return_value = []
    ref = dict(run_id='ref', experiment_id='5', metrics={}, params={})
    ctx = dict(dataset_sha256='h', reference_run_id='ref', predictoras_excluidas=[])
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        preparar = Mock(return_value=(X, y, ctx, ref, grupos, out))
        monkeypatch.setattr(s, 'preparar_experimento', preparar)
        exportar = Mock()
        monkeypatch.setattr(s.b, 'exportar', exportar)
        def registrar(client, carpeta, contexto, metricas, *, nombre_run, parametros_extra):
            assert nombre_run == s.NOMBRE_RUN_177 and client is cliente
            assert parametros_extra['n_candidate_predictors'] == 177
            assert parametros_extra['reference_run_name'] == 'OLS_02_Log_Target'
            assert parametros_extra['reference_run_id'] == 'ref'
            assert parametros_extra['excluded_predictors'] == '' and parametros_extra['permutation_repeats'] == 5
            assert json.loads(parametros_extra['candidate_columns']) == list(X)
            assert contexto['universo_predictoras'] == list(X) and contexto['n_predictoras_inicial'] == 177
            with np.load(carpeta/'particion_interna.npz') as particion:
                np.testing.assert_array_equal(particion['fit'], fit)
                np.testing.assert_array_equal(particion['validation'], val)
            sel = json.loads((carpeta/'seleccion.json').read_text(encoding='utf-8'))
            assert json.loads((carpeta/'historial_seleccion.json').read_text(encoding='utf-8')) == sel['historial']
            assert json.loads((carpeta/'candidatos_evaluados.json').read_text(encoding='utf-8')) == sel['candidatos']
            cliente.get_run.side_effect = [SimpleNamespace(data=SimpleNamespace(params=parametros_extra, metrics=metricas)),
                                           SimpleNamespace(data=SimpleNamespace(params={}, metrics={}))]
            return {'run_id': 'nuevo'}
        monkeypatch.setattr(s.b, 'registrar', registrar)
        r = s.ejecutar_experimento('base', cliente, carpeta_salida=out, excluidas=(),
            referencia_nombre=s.REFERENCIA_177, nombre_run=s.NOMBRE_RUN_177)
        assert preparar.call_args.kwargs['excluidas'] == ()
        assert r['registro']['run_id'] == 'nuevo'
        modelo, diseno = exportar.call_args.args[1:3]
        assert modelo.model.rank == diseno.shape[1] and modelo.nobs == len(y)


def test_no_sobrescribe_nombre_existente_ni_entrena(tmp_path, monkeypatch):
    X, y, grupos, fit, val = sinteticos()
    cliente = Mock()
    cliente.get_experiment_by_name.return_value = SimpleNamespace(experiment_id='5', lifecycle_stage='active')
    cliente.search_runs.return_value = [object()]
    datos = dict(X=X, y=y, grupos=grupos, excluidas=(), config_forward=s.CONFIG_FORWARD, cliente=cliente,
                 sesion=SimpleNamespace(carpeta=lambda _: tmp_path))
    ejecutar = Mock()
    monkeypatch.setattr(s, 'ejecutar_experimento', ejecutar)
    with pytest.raises(ValueError, match='nombre de Run nuevo'):
        s.ejecutar_presentacion(datos, 'forward', modo='nuevo', nombre_run=s.NOMBRE_RUN_177)
    ejecutar.assert_not_called()


def test_comparaciones_distinguen_universos_e_ingreso_de_variables():
    m = dict(train_n_predictors=3, train_r2_adj_log=.8, train_rmse_original=2,
             selection_rmse_validation_original=3, train_aic_log=4, train_bic_log=5)
    f = dict(metricas=m, grupos={'a': ['a']}, seleccion={'seleccionadas': ['a']})
    n = dict(metricas={**m, 'selection_rmse_validation_original': 2.5},
        grupos={'a': ['a'], 'rooms': ['rooms']}, seleccion={'seleccionadas': ['a', 'rooms'],
            'historial': [{'iteracion': 1, 'caracteristica_incorporada': 'a'},
                          {'iteracion': 2, 'caracteristica_incorporada': 'rooms'}]})
    tabla, texto = s.comparacion_forward_presentacion(f, n)
    assert '174 predictoras' in tabla and '177 predictoras' in tabla and 'Δ 177 − 174' in tabla
    assert '`rooms`: incorporada en la iteración 2' in texto
    assert '`living_area`: disponible entre las candidatas, no seleccionada' in texto
    assert 'reduce el error' in texto and 'no es una evaluación independiente' in texto
    tabla, coincidencias = s.comparacion_final_presentacion(f, f, f, forward_177=n)
    assert 'Columnas candidatas' in tabla and 'Comunes a los 4' in coincidencias
    assert 'Exclusivas de Forward Selection — 177 predictoras: `rooms`' in coincidencias
    assert 'Forward Selection — 177 predictoras obtuvo el menor RMSE' in s.conclusion_presentacion(f, f, f, forward_177=n)
