"""Escalas comparables, exclusiones acumulativas y ausencia de gráficos en ablaciones."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import baseline_ols as b


@pytest.fixture
def secuencia(monkeypatch):
    rng = np.random.default_rng(628)
    X = pd.DataFrame(rng.normal(size=(120, 5)),
                     columns=["total_area", "living_area", "rooms", "to_center_km", "otra"])
    y = pd.Series(np.exp(10 + .2*X.total_area + rng.normal(0, .1, len(X))))
    ctx = {"dataset_sha256": "prueba", "predictoras": list(X), "n_predictoras": 5,
           "n_train": len(X), "transformacion_objetivo": "ninguna"}
    def prohibido(*args, **kwargs):
        raise AssertionError("No generar gráficos ni usar MLflow durante los ajustes.")
    monkeypatch.setattr(b, "generar_diagnosticos_residuales", prohibido)
    monkeypatch.setattr(b, "conectar", prohibido)
    resultados = []
    for i in range(1, 6):
        resultados.append(b.ajustar_presentacion(X, y, ctx, i, resultados[-1] if resultados else None))
    return X, y, ctx, resultados


def test_mismos_datos_exclusiones_y_objetivo(secuencia):
    X, y, ctx, resultados = secuencia
    assert ctx["transformacion_objetivo"] == "ninguna" and list(X) == ctx["predictoras"]
    for i, r in enumerate(resultados, 1):
        assert r["diseno"].index.equals(y.index)
        assert r["diseno"].columns[0] == "const"
        assert r["contexto"]["n_predictoras"] == 5-max(0, i-2)
        np.testing.assert_allclose(r["y"], y if i == 1 else np.log(y))
        pred = r["modelo"].predict(r["diseno"])
        if i > 1:
            pred = np.exp(pred)
        np.testing.assert_allclose(b._valor_presentacion(r, "rmse"), np.sqrt(np.mean((y-pred)**2)))
    assert "| 5 — OLS_05_Log_Sin_rooms | ln(price_rub) | 2 |" in b.evolucion_variables(X)


def test_comparacion_no_resta_escalas_distintas(secuencia):
    *_, resultados = secuencia
    tabla = b.tabla_presentacion(resultados[1])
    for linea in tabla.splitlines():
        if any(f"| {nombre}" in linea for nombre in ("AIC", "BIC", "R²")):
            assert "No comparable" in linea
        if "| MAE (RUB)" in linea or "| RMSE (RUB)" in linea:
            assert "No comparable" not in linea
    for r in resultados[2:]:
        assert "No comparable" not in b.tabla_presentacion(r)
        with pytest.raises(ValueError, match="no tienen visualizaciones"):
            b.grafico_presentacion(r)
    assert all(len(b.interpretacion_presentacion(r).splitlines()) <= (8 if r["numero"] <= 2 else 6)
               for r in resultados)


def test_exportacion_sin_graficos_es_opcional(secuencia, tmp_path, monkeypatch):
    *_, resultados = secuencia
    r = resultados[-1]
    monkeypatch.setattr(b, "sha256", lambda path: "prueba")
    monkeypatch.setattr(b, "validar_temporal", lambda path: Path(path))
    # El fixture ya prohíbe generar_diagnosticos_residuales: la exportación debe omitirlo.
    carpeta = b.exportar(tmp_path, r["modelo"], r["diseno"], r["y"],
        {**r["contexto"], "csv": "train.csv"}, r["metricas"], r["invalidas"], r["vif"], r["notas"],
        nombre_run=r["nombre"], carpeta_salida=tmp_path, generar_graficos=False)
    assert not (carpeta/"diagnosticos_residuales").exists()
    assert (carpeta/"modelo_statsmodels.pkl").is_file()
