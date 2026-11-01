"""Archivos de trabajo fuera del repositorio y recuperación histórica reutilizable."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

from .contrato_datasets import RAIZ
RUN_TRANSFORMACION = '43e383e9b91d43efbdeda5ec1ec28f71'


def validar_temporal(carpeta):
    ruta = Path(carpeta).resolve()
    if ruta.is_relative_to(RAIZ) or not ruta.is_relative_to(Path(gettempdir()).resolve()):
        raise ValueError('Los artefactos requieren un directorio temporal del sistema fuera del repositorio.')
    return ruta


class SesionTemporal:
    """Una sesión de notebook; paso() limpia también ante errores o interrupciones."""
    def __init__(self):
        self._temporal = TemporaryDirectory(prefix='moscow_eda_')
        self.raiz = validar_temporal(self._temporal.name)

    def carpeta(self, nombre):
        ruta = validar_temporal(self.raiz / nombre)
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    def cerrar(self):
        self._temporal.cleanup()

    @contextmanager
    def paso(self):
        try:
            yield self
        except BaseException:
            self.cerrar()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()


def leer_json(cliente, run_id, nombre):
    with TemporaryDirectory(prefix='leer_mlflow_') as temporal:
        archivo = cliente.download_artifacts(run_id, nombre, temporal)
        return json.loads(Path(archivo).read_text(encoding='utf-8'))


def registrar_archivos(cliente, run_id, carpeta, destino, *, fuentes=()):
    """Registra y compara cada byte descargado antes de liberar los temporales."""
    import hashlib
    carpeta = validar_temporal(carpeta)
    huella = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    antes = cliente.get_run(run_id)
    cliente.log_artifacts(run_id, str(carpeta), artifact_path=destino)
    for fuente in fuentes:
        cliente.log_artifact(run_id, str(fuente), artifact_path=destino)
    with TemporaryDirectory(prefix='verificar_mlflow_') as temporal:
        for archivo in carpeta.rglob('*'):
            if archivo.is_file():
                nombre = destino + '/' + archivo.relative_to(carpeta).as_posix()
                recibido = cliente.download_artifacts(run_id, nombre, temporal)
                assert huella(recibido) == huella(archivo), nombre
        for fuente in fuentes:
            nombre = destino + '/' + Path(fuente).name
            recibido = cliente.download_artifacts(run_id, nombre, temporal)
            assert huella(recibido) == huella(fuente), nombre
    despues = cliente.get_run(run_id)
    assert antes.data.metrics == despues.data.metrics
    assert antes.data.params == despues.data.params
    assert antes.info.status == despues.info.status
