"""Verifica la migración de nombres frente a la referencia anterior, sin ajustar objetos."""
import hashlib
import json
import warnings
from pathlib import Path

# Entrada CLI por ruta absoluta: la resolución del paquete no depende del cwd.
if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "scripts"

import joblib
import pandas as pd

from . import preparacion_train_test as pt
def parametros_aprendidos(objetos):
    preparacion = objetos["preparacion"]
    cat = objetos["base"].named_transformers_["cat"]
    distrito = objetos["district"].named_steps["onehot"]
    temporal = objetos["extraccion"]
    escala = objetos["escalado"].estandarizador_
    return {
        "imputadores": {k: getattr(preparacion, k).statistics_.tolist()
                        for k in ["numerico_", "mensual_", "binario_", "categorico_"]},
        "categorias": {c: v.tolist() for c, v in zip(cat.feature_names_in_, cat.categories_)},
        "district": distrito.categories_[0].tolist(), "drop_district": distrito.drop_idx_.tolist(),
        "drop_cat": cat.drop_idx_.tolist(), "mes": temporal.mes_.categories_[0].tolist(),
        "drop_mes": temporal.mes_.drop_idx_.tolist(), "origen_mes": temporal.origen_mes_,
        "media_escalado": escala.mean_.tolist(), "varianza_escalado": escala.var_.tolist(),
        "escala": escala.scale_.tolist(),
        "tukey": objetos["configuracion"]["tratamiento_objetivo"],
    }


def verificar(raiz):
    """Contrasta el contrato congelado y artefactos remotos; no entrena ni escribe resultados."""
    from . import baseline_ols as b
    from .almacenamiento_mlflow import SesionTemporal, RUN_TRANSFORMACION
    from .contrato_datasets import CONTRATO, ruta_dataset
    carpeta = Path(raiz) / "Ventas/Fase1_PropUsadas"
    cliente = b.conectar(raiz)
    with SesionTemporal() as sesion:
        temporal = sesion.carpeta("verificacion")
        descargar = lambda nombre: Path(cliente.download_artifacts(
            RUN_TRANSFORMACION, "transformacion/" + nombre, str(temporal)))
        manifiesto = json.loads(descargar("manifiesto_exportacion.json").read_text(encoding="utf-8"))
        resumen = json.loads(descargar("resumen_final.json").read_text(encoding="utf-8"))
        objetos = joblib.load(descargar("transformadores_prop_usadas.joblib"))
        assert manifiesto["columnas_X"] == resumen["columnas_X"] == objetos["columnas_X"] == CONTRATO["columnas_X"]
        csv = {}
        for conjunto in ["train", "test"]:
            ruta = ruta_dataset(f"{conjunto}_transformado.csv", carpeta)
            assert pt.huella_archivo(ruta) == manifiesto[conjunto]["sha256"] == CONTRATO[conjunto]["sha256"]
            frame = pd.read_csv(ruta)
            pt.validar_matriz(pt.obtener_X(frame), frame, columnas=objetos["columnas_X"])
            assert frame.dataset_split.eq(conjunto).all()
            csv[conjunto] = {"filas": len(frame), "sha256": pt.huella_archivo(ruta)}
        assert pt.registro_escalado(objetos["escalado"]) == resumen["escalado"]
        return {"csv": csv, "esquema_y_huellas_correctos": True, "artefactos_remotos_legibles": True}


if __name__ == "__main__":
    reporte = verificar(Path(__file__).resolve().parents[3])
    print("CSV, esquema y artefactos remotos verificados; sin ajustes ni archivos auxiliares locales.")
