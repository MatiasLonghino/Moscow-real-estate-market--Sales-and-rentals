"""Verifica escalas separadas y la inversa exp sin corrección de sesgo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.regression.linear_model import OLSResults

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import baseline_ols as b
def ejemplo():
    rng = np.random.default_rng(28)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["area", "rooms", "to_center_km"])
    # Incluye logaritmos negativos para comprobar que se valida positividad en el precio.
    y = pd.Series(np.exp(-2 + .3 * X.area + rng.normal(0, .2, len(X))), name="price_rub")
    y_log = np.log(y).rename("ln_price_rub")
    modelo, diseno = b.ajustar(X, y_log)
    return y, y_log, modelo, diseno


def test_metricas_nativas_y_originales_independientes():
    y, y_log, modelo, diseno = ejemplo()
    original = y.copy()
    m, invalidas, vif, notas = b.diagnosticar(modelo, diseno, y_log, y_original=y)
    p = np.exp(modelo.predict(diseno))
    np.testing.assert_allclose([m["train_r2_log"], m["train_r2_adj_log"], m["train_aic_log"], m["train_bic_log"], m["train_rss_log"]],
        [modelo.rsquared, modelo.rsquared_adj, modelo.aic, modelo.bic, modelo.ssr])
    np.testing.assert_allclose([m["train_r2_original"], m["train_mae_original"], m["train_mse_original"], m["train_rmse_original"]],
        [r2_score(y,p), mean_absolute_error(y,p), mean_squared_error(y,p), np.sqrt(mean_squared_error(y,p))])
    assert "train_aic_original" not in m and "train_aic" not in m
    assert "train_mape_log" not in m and not invalidas
    assert m["train_n_predictors"] == 3 and m["train_n_parameters_effective"] == 4
    pd.testing.assert_series_equal(y, original)
    tabla = b.tabla_resultados(m, invalidas, notas, nombre_run="OLS_02_Log_Target")
    assert "OLS_02_Log_Target" in tabla and "ln(RUB)²" in tabla and "precio original" in tabla


def test_exportacion_contrato_log_e_inversa(tmp_path):
    y, y_log, modelo, diseno = ejemplo()
    archivo = tmp_path / "train.csv"
    y.to_csv(archivo, index=False)
    contexto = {"csv":"train.csv", "dataset_sha256":b.sha256(archivo),
        "transformacion_objetivo":"ln", "unidad":"ln(RUB)",
        "predictoras":list(diseno.columns[1:]), "baseline_run_id":"baseline_existente"}
    m, invalidas, vif, notas = b.diagnosticar(modelo, diseno, y_log, y_original=y)
    out = b.exportar(tmp_path, modelo, diseno, y_log, contexto, m, invalidas, vif, notas,
        nombre_run="OLS_02_Log_Target", carpeta_salida=tmp_path/"log")
    import json
    inferencia = json.loads((out/"inferencia.json").read_text(encoding="utf-8"))
    assert inferencia["transformacion_inversa"] == "exp" and inferencia["correccion_sesgo"] == "ninguna"
    assert inferencia["unidad_salida_modelo"] == "ln(RUB)" and inferencia["unidad_salida"] == "RUB"
    cargado = OLSResults.load(out/"modelo_statsmodels.pkl")
    assert cargado.model.exog is None and cargado.model.endog is None
    np.testing.assert_allclose(np.exp(cargado.predict(diseno)), np.exp(modelo.predict(diseno)), rtol=1e-12)
    assert all((out/"diagnosticos_residuales"/f).is_file() for f in b.ARCHIVOS)
