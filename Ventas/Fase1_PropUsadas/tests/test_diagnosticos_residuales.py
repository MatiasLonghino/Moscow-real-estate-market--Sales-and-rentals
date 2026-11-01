"""Compara la estandarización con statsmodels y verifica el destino de los PNG."""
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnosticos_residuales import (
    ARCHIVOS, generar_diagnosticos_residuales,
    registrar_diagnosticos_residuales, residuos_train,
)


def ejemplo():
    rng = np.random.default_rng(41)
    X = sm.add_constant(pd.DataFrame(rng.normal(size=(140, 3)), columns=list("abc")))
    y = pd.Series(8e6 + 4e5 * X.a + rng.normal(0, 2e5, len(X)))
    return sm.OLS(y, X).fit(), X, y


def test_estandarizacion_y_modelo_sin_datos(tmp_path):
    modelo, X, y = ejemplo()
    antes = modelo.params.copy()
    ajustados, residuos, estandarizados = residuos_train(modelo, X, y)
    np.testing.assert_allclose(estandarizados, modelo.get_influence().resid_studentized_internal)
    np.testing.assert_allclose(residuos, modelo.resid)
    archivo = tmp_path / "modelo.pkl"
    modelo.save(archivo, remove_data=True)
    from statsmodels.regression.linear_model import OLSResults
    cargado = OLSResults.load(archivo)
    reconstruidos = residuos_train(cargado, X, y)
    for a, b in zip(reconstruidos, (ajustados, residuos, estandarizados)):
        np.testing.assert_allclose(a, b)
    pd.testing.assert_series_equal(antes, cargado.params)


def test_tres_png_y_registro_en_mismo_run(tmp_path):
    modelo, X, y = ejemplo()
    parametros = modelo.params.copy()
    rutas = generar_diagnosticos_residuales(modelo, X, y,
        nombre_run="OLS_prueba", carpeta=tmp_path / "graficos")
    assert set(rutas) == set(ARCHIVOS)
    for ruta in rutas.values():
        with Image.open(ruta) as png:
            assert png.format == "PNG" and png.size == (1440, 960)
    pd.testing.assert_series_equal(parametros, modelo.params)

    class Cliente:
        def __init__(self):
            self.logs = []
        def get_run(self, rid):
            assert rid == "id_existente"
            return SimpleNamespace(info=SimpleNamespace(status="FINISHED"),
                data=SimpleNamespace(tags={"mlflow.runName": "OLS_prueba"}, metrics={"r2": .8}, params={"intercepto": "True"}))
        def log_artifact(self, rid, ruta, artifact_path):
            self.logs.append((rid, Path(ruta).name, artifact_path))
        def download_artifacts(self, rid, ruta, destino):
            archivo = Path(destino) / Path(ruta).name
            shutil.copyfile(rutas[archivo.name], archivo)
            return str(archivo)

    cliente = Cliente()
    registrados = registrar_diagnosticos_residuales(cliente, "id_existente", tmp_path / "graficos", nombre_run="OLS_prueba")
    assert set(registrados) == {f"diagnosticos_residuales/{nombre}" for nombre in ARCHIVOS}
    assert cliente.logs == [("id_existente", n, "diagnosticos_residuales") for n in ARCHIVOS]
    with pytest.raises(ValueError, match="no corresponde"):
        registrar_diagnosticos_residuales(cliente, "id_existente", tmp_path / "graficos", nombre_run="Otro_OLS")


def test_no_descarta_filas_invalidas():
    modelo, X, y = ejemplo()
    y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="no se descartan filas"):
        residuos_train(modelo, X, y)
