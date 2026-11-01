"""ValidaciÃ³n independiente de unidades, VIF e integridad del contrato."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.outliers_influence import variance_inflation_factor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import baseline_ols as b
def ejemplo(redundante=False):
    rng=np.random.default_rng(49)
    X=pd.DataFrame(rng.normal(size=(120,3)),columns=["a","b","c"])
    if redundante: X["doble_a"]=2*X.a
    y=pd.Series(5e6+1e5*X.a+2e5*X.b+rng.normal(0,1e4,120))
    modelo,diseno=b.ajustar(X,y)
    return modelo,diseno,y

def test_metricas_unidades_y_vif():
    modelo,diseno,y=ejemplo()
    m,invalidas,vif,notas=b.diagnosticar(modelo,diseno,y)
    resid=y-modelo.predict(diseno)
    assert not invalidas
    np.testing.assert_allclose(m["train_mse"],np.mean(resid**2))
    np.testing.assert_allclose(m["train_mape"],np.mean(np.abs(resid)/y)*100)
    np.testing.assert_allclose([m["train_aic"],m["train_bic"]],[modelo.aic,modelo.bic])
    esperado=[variance_inflation_factor(diseno.to_numpy(),j) for j in range(1,4)]
    np.testing.assert_allclose(vif.vif,esperado,rtol=1e-10)
    assert "train_bp_lm" in m

def test_singularidad_no_elimina_columnas():
    modelo,diseno,y=ejemplo(True)
    m,invalidas,vif,notas=b.diagnosticar(modelo,diseno,y)
    assert len(modelo.params)==5 and m["train_rank_deficiency"]==1
    assert m["train_vif_infinite_count"]==2
    assert "train_vif_max" in invalidas and "train_vif_max" not in m
    assert "train_bp_lm" not in m
    assert all(np.isfinite(v) for v in m.values())

def test_train_exportado_conservado():
    base=Path(__file__).resolve().parents[1]
    archivo=base/"data/train_transformado.csv"
    antes=b.sha256(archivo)
    X,y,contexto=b.cargar_train(base)
    assert contexto["test_utilizado"] is False
    assert len(y)==37984 and X.shape==(37984,177)
    assert not set(["id","price_rub","dataset_split","date_posted_dt"]) & set(X)
    assert b.sha256(archivo)==antes==contexto["dataset_sha256"]
