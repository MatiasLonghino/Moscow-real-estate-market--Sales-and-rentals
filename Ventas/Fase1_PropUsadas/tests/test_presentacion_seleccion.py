"""Reutilización estricta de Runs y separación de evaluación en la presentación."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import seleccion_modelos as s, group_lasso_ols as gl


def datos_minimos(tmp_path):
    return dict(cliente=Mock(), sesion=SimpleNamespace(carpeta=lambda nombre: tmp_path),
        contexto={'dataset_sha256': 'huella'}, excluidas=s.EXCLUIDAS,
        config_forward=dict(s.CONFIG_FORWARD), config_lasso=dict(gl.CONFIG), y=np.ones(100))


@pytest.mark.parametrize('metodo,clave,valor', [
    ('forward', 'excluded_predictors', ''), ('lasso', 'excluded_predictors', ''),
    ('forward', 'min_mejora_pct', '.7'), ('forward', 'max_features', '11'),
    ('lasso', 'random_state', '10'), ('lasso', 'target_transform', 'none')])
def test_rechaza_cambios_antes_de_descargar_o_entrenar(tmp_path, monkeypatch, metodo, clave, valor):
    datos = datos_minimos(tmp_path)
    params = {k: str(v) for k, v in s._parametros_presentacion(datos, metodo).items()}
    params[clave] = valor
    monkeypatch.setattr(s, 'recuperar_experimento', lambda *a, **kw: {'params': params})
    with pytest.raises(ValueError, match='incompatible'):
        s.ejecutar_presentacion(datos, metodo)
    datos['cliente'].download_artifacts.assert_not_called()
    datos['cliente'].create_run.assert_not_called()


def test_no_sobrescribe_run_en_modo_nuevo(tmp_path):
    datos = datos_minimos(tmp_path)
    datos['cliente'].get_experiment_by_name.return_value = SimpleNamespace(experiment_id='5', lifecycle_stage='active')
    datos['cliente'].search_runs.return_value = [object()]
    with pytest.raises(ValueError, match='nombre de Run nuevo'):
        s.ejecutar_presentacion(datos, 'forward', modo='nuevo')
    datos['cliente'].create_run.assert_not_called()


def test_177_candidatas_no_heredan_exclusion_global():
    columnas = ['to_center_km', 'living_area', 'rooms', 'district_a', 'district_b']
    resumen = {'linaje': {c: {'tipo': 'onehot' if c.startswith('district') else 'numerica',
                              'variable': c, 'entradas': [c]} for c in columnas},
        'categorias': [{'variable': 'district', 'columnas': columnas[-2:],
                        'referencia': 'ref', 'categorias': ['ref', 'a', 'b']}]}
    with pytest.raises(ValueError):
        s.grupos_caracteristicas(columnas, resumen)
    grupos = s.grupos_caracteristicas(columnas, resumen, excluidas=())
    assert grupos['district'] == columnas[-2:]
    assert set(s.columnas_grupos(grupos, list(grupos))) == set(columnas)


def test_particion_historica_distinta_se_rechaza(tmp_path):
    datos = datos_minimos(tmp_path)
    datos.update(grupos={'area': ['area']}, fit=np.arange(80), val=np.arange(80, 100))
    params = {k: str(v) for k, v in s._parametros_presentacion(datos, 'forward').items()}
    r = dict(metodo='forward', run={'params': params}, grupos=datos['grupos'], seleccion={},
        contexto={'predictoras_excluidas': list(s.EXCLUIDAS), 'dataset_sha256': 'huella',
                  'transformacion_objetivo': 'ln', 'n_train': 100},
        particion={'fit': np.arange(1, 81), 'validation': np.r_[0, np.arange(81, 100)]})
    with pytest.raises(ValueError, match='Partición interna incompatible'):
        s._validar_resultado_presentacion(datos, r)


def test_tablas_sinteticas_separan_train_y_validacion():
    m = dict(train_r2_log=.9, train_r2_adj_log=.89, train_aic_log=10, train_bic_log=12,
        train_mae_original=1, train_rmse_original=2, selection_rmse_validation_original=3,
        train_vif_max=2, train_condition_number=10, train_bp_lm_pvalue_log=.001,
        train_jarque_bera_pvalue_log=.002, train_n_predictors=2)
    f = dict(metricas=m, seleccion={'seleccionadas': ['a']}, run={'run_name': s.NOMBRE_RUN,
        'run_id': 'f', 'experiment_id': '5'}, modo='recuperar')
    g = dict(metricas={**m, 'selection_rmse_validation_original': 2.5, 'train_n_predictors': 3},
             seleccion={'seleccionadas': ['a', 'b']})
    tabla = s.tabla_metricas_presentacion(g, anterior=f)
    assert '| Validación interna | RMSE (RUB) |' in tabla
    assert '| Train | RMSE (RUB) |' in tabla
    assert 'R² ajustado [ln(RUB)]' in tabla and len(tabla.splitlines()) == 13
    assert len(s.interpretar_presentacion(g, f).splitlines()) <= 6
    final, coincidencias = s.comparacion_final_presentacion(f, g)
    assert 'RMSE interno (RUB)' in final and 'Exclusivas de Group Lasso: `b`' in coincidencias
    assert 'sin nueva selección ni ajuste final' in s.confirmacion_presentacion(f)
    assert 'Group Lasso + OLS obtuvo el menor RMSE' in s.conclusion_presentacion(f, g)
    assert 'Forward Selection logró la mayor reducción' in s.conclusion_presentacion(f, g)


def test_group_lasso_admite_rooms_con_exclusiones_explicitas_vacias():
    rng = np.random.default_rng(49)
    X = pd.DataFrame(rng.normal(size=(160, 2)), columns=['rooms', 'area'])
    y = np.exp(10 + .4*X.rooms + .2*X.area + rng.normal(0, .03, len(X)))
    with TemporaryDirectory() as tmp:
        resultado = gl.buscar_group_lasso(X, y, {c: [c] for c in X}, np.arange(120), np.arange(120, 160),
            tmp, lambdas=[.1, .03, .01], excluidas=())
    assert 'rooms' in resultado['seleccionadas']
