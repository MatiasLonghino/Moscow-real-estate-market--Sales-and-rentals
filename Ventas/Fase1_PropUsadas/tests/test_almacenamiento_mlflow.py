"""Contratos de persistencia: temporales, excepciones y verificación de registro."""
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.almacenamiento_mlflow import SesionTemporal, validar_temporal, registrar_archivos, RAIZ


def test_limpieza_al_finalizar_y_ante_interrupcion():
    with SesionTemporal() as sesion:
        carpeta = sesion.carpeta('modelo')
        (carpeta / 'modelo.pkl').write_bytes(b'contenido')
        assert not carpeta.is_relative_to(RAIZ)
    assert not carpeta.exists()
    sesion = SesionTemporal()
    carpeta = sesion.carpeta('referencia')
    with pytest.raises(KeyboardInterrupt):
        with sesion.paso():
            (carpeta / 'metadatos.json').write_text('{}')
            raise KeyboardInterrupt()
    assert not carpeta.exists()


def test_rechaza_artefactos_permanentes():
    with pytest.raises(ValueError):
        validar_temporal(RAIZ / 'Ventas/Fase1_PropUsadas/resultados')


def test_ejecucion_forward_libera_temporales_si_falla_la_preparacion(monkeypatch):
    from scripts import seleccion_modelos as s
    carpetas = []

    def fallar(base, cliente, **opciones):
        carpeta = opciones['carpeta_salida']
        carpetas.append(carpeta)
        (carpeta / 'configuracion.json').write_text('{}')
        raise RuntimeError('fallo de recuperación')

    monkeypatch.setattr(s, 'preparar_experimento', fallar)
    with pytest.raises(RuntimeError, match='fallo de recuperación'):
        s.ejecutar_experimento(None, None)
    assert len(carpetas) == 1 and not carpetas[0].exists()


def test_registro_temporal_verifica_bytes_sin_tocar_metricas(tmp_path):
    class Cliente:
        def get_run(self, rid):
            assert rid == 'existente'
            return SimpleNamespace(info=SimpleNamespace(status='FINISHED'),
                data=SimpleNamespace(metrics={'rmse': 1.5}, params={'solver': 'pinv'}))

        def log_artifacts(self, rid, carpeta, artifact_path):
            assert rid == 'existente' and artifact_path == 'transformacion'
            shutil.copytree(carpeta, tmp_path / artifact_path)

        def download_artifacts(self, rid, nombre, destino):
            salida = Path(destino) / Path(nombre).name
            shutil.copyfile(tmp_path / nombre, salida)
            return str(salida)

    with SesionTemporal() as sesion:
        carpeta = sesion.carpeta('registro')
        (carpeta / 'contrato.json').write_text('{"filas": 4}', encoding='utf-8')
        registrar_archivos(Cliente(), 'existente', carpeta, 'transformacion')
    assert not carpeta.exists()
