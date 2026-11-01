"""Anidamiento, HC3 sin reajuste y resúmenes de VIF del experimento de ablación."""
import copy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLSResults

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import baseline_ols as b
from scripts import diagnosticos_residuales as dr
def test_anidados_hc3_y_modelo_completo_serializado(tmp_path):
    rng=np.random.default_rng(73)
    X=pd.DataFrame(rng.normal(size=(180,3)),columns=["area","to_center_km","rooms"])
    y=pd.Series(2+.4*X.area-.2*X.to_center_km+rng.normal(0,.2,len(X)))
    full,D=b.ajustar(X,y)
    red,_=b.ajustar(X.drop(columns="to_center_km"),y)
    esperado_f=full.compare_f_test(red)
    esperado_se=np.sqrt(full.get_robustcov_results(cov_type="HC3").cov_params()[2,2])
    esperado_delta=full.rsquared-red.rsquared
    _=full.pvalues
    archivo=tmp_path/"full.pkl"
    copy.deepcopy(full).save(archivo,remove_data=True)
    cargado=OLSResults.load(archivo)
    m,notas=b.comparar_anidados(cargado,red,D,y,variable="to_center_km",rss_completo_esperado=full.ssr)
    np.testing.assert_allclose(m["partial_f_descriptive"],esperado_f[0],rtol=1e-10)
    np.testing.assert_allclose(m["removed_predictor_se_hc3"],esperado_se,rtol=1e-10)
    np.testing.assert_allclose(m["incremental_r2_removed_predictor_log"],esperado_delta,rtol=1e-10)
    assert "partial_f_pvalue" not in m and m["partial_f_classical_inference_valid"]==0
    assert cargado.model.exog is None
    validas,_=b.comparar_anidados(full,red,D,y,variable="to_center_km",rss_completo_esperado=full.ssr,f_clasico_valido=True)
    np.testing.assert_allclose(validas["partial_f_pvalue"],esperado_f[1],rtol=1e-10)


def test_vif_resumido_y_tabla_misma_escala():
    rng=np.random.default_rng(13)
    X=pd.DataFrame(rng.normal(size=(120,3)),columns=list("abc"))
    y=pd.Series(np.exp(2+.2*X.a+rng.normal(0,.1,len(X))))
    log=np.log(y)
    modelo,D=b.ajustar(X,log)
    m,invalidas,vif,notas=b.diagnosticar(modelo,D,log,y_original=y,resumir_vif=True)
    np.testing.assert_allclose(m["train_vif_mean"],vif.vif.mean())
    assert m["train_vif_gt5_count"]==int((vif.vif>5).sum())
    assert m["train_vif_gt10_count"]==int((vif.vif>10).sum())
    tabla=b.tabla_resultados(m,invalidas,notas,nombre_run="Nuevo",referencia=m,nombre_referencia="Anterior")
    assert "| Métrica | Anterior | Nuevo | Diferencia |" in tabla
    assert "R² (ln(RUB))" in tabla and "VIF promedio" in tabla


def test_no_crea_experimento_si_falta(tmp_path):
    carpeta=tmp_path/"diagnosticos_residuales"
    carpeta.mkdir()
    for nombre in dr.ARCHIVOS:
        (carpeta/nombre).write_bytes(b"PNG")
    class Cliente:
        def get_experiment_by_name(self,nombre):
            return None
        def create_experiment(self,*args):
            raise AssertionError("No debe crear un experimento")
    with pytest.raises(ValueError,match="no se crean"):
        b.registrar(Cliente(),tmp_path,{}, {},nombre_run="OLS_03_Log_Sin_to_center_km")


def test_flujo_compartido_exclusiones_acumuladas_y_comparacion(tmp_path):
    rng=np.random.default_rng(9)
    total=rng.normal(size=120)
    X=pd.DataFrame({"total_area":total,"living_area":.8*total+rng.normal(0,.2,120),"rooms":rng.normal(size=120)})
    log=pd.Series(2+.4*X.total_area+rng.normal(0,.1,120))
    y=np.exp(log)
    y.to_csv(tmp_path/"train.csv",index=False)
    ctx={"csv":"train.csv","dataset_sha256":b.sha256(tmp_path/"train.csv"),"predictoras":list(X),
        "predictoras_excluidas":["to_center_km"],"n_train":len(X),"n_predictoras":3,
        "unidad":"ln(RUB)","transformacion_objetivo":"ln"}
    full,D=b.ajustar(X,log)
    m,inv,vif,notas=b.diagnosticar(full,D,log,y_original=y,resumir_vif=True)
    origen=b.exportar(tmp_path,full,D,log,ctx,m,inv,vif,notas,nombre_run="Previo",carpeta_salida=tmp_path/"origen")
    huella=b.sha256(origen/"modelo_statsmodels.pkl")
    # Este indicador anterior mide otra restricción y no admite una diferencia entre ablaciones.
    ref=SimpleNamespace(info=SimpleNamespace(run_id="anterior",status="FINISHED",experiment_id="1"),
        data=SimpleNamespace(tags={"mlflow.runName":"Previo","dataset_sha256":ctx["dataset_sha256"]},
            metrics={**m,"partial_f_descriptive":9999.0},params={"target_transform":"ln(Target)",
                "bias_correction":"none","inverse_transform":"exp","cov_type":"nonrobust","solver":"pinv"}))
    class Cliente:
        def get_run(self,rid):
            assert rid=="anterior"
            return ref
        def get_experiment_by_name(self,nombre):
            return SimpleNamespace(experiment_id="1",lifecycle_stage="active")
        def download_artifacts(self,rid,nombre,destino):
            nuevo=Path(destino)/Path(nombre).name
            shutil.copyfile(origen/nombre,nuevo)
            return str(nuevo)
    datos=b.preparar_ablacion(Cliente(),tmp_path,X,log,y,
        registro_referencia={"run_id":"anterior","run_name":"Previo"},variable="living_area",
        nombre_run="Nuevo",carpeta_salida=tmp_path/"nuevo")
    assert datos["contexto"]["predictoras_excluidas"]==["to_center_km","living_area"]
    red,Dr=b.ajustar(datos["X"],log)
    resultado=b.evaluar_ablacion(datos,red,Dr,relacionadas=("total_area",))
    assert resultado["metricas"]["train_n_predictors"]==2
    np.testing.assert_allclose(resultado["metricas"]["related_total_area_correlation_train"],X.living_area.corr(X.total_area))
    assert "partial_f_descriptive" not in resultado["comparacion_md"].split("## Aporte")[0]
    assert datos["modelo_referencia"].model.exog is None and b.sha256(origen/"modelo_statsmodels.pkl")==huella

    class ClienteLigero(Cliente):
        def __init__(self):
            self.descargas=[]
        def download_artifacts(self,rid,nombre,destino):
            self.descargas.append(nombre)
            return super().download_artifacts(rid,nombre,destino)
    cliente_ligero=ClienteLigero()
    ligero=b.preparar_ablacion(cliente_ligero,tmp_path,X,log,y,
        registro_referencia={"run_id":"anterior","run_name":"Previo"},variable="rooms",
        nombre_run="NuevoLigero",carpeta_salida=tmp_path/"ligero",analizar_importancia=False)
    assert cliente_ligero.descargas==["contexto.json"] and ligero["modelo_referencia"] is None
    modelo_ligero,diseno_ligero=b.ajustar(ligero["X"],log)
    evaluacion=b.evaluar_ablacion(ligero,modelo_ligero,diseno_ligero)
    assert set(evaluacion["metricas"])=={k for k in ref.data.metrics if k.startswith("train_")}
    assert evaluacion["comparacion_md"].startswith("| Métrica | Previo | NuevoLigero | Δ |")
    assert evaluacion["comparacion_md"].count("| Métrica |")==1
    assert evaluacion["comparacion_md"].rstrip().endswith("|")
    assert not any((ligero["salida"]/nombre).exists() for nombre in
        ["interpretacion.md","importancia_rooms.json","cambios_significancia.csv"])
