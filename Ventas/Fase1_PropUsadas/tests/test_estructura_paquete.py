"""Rutas y compatibilidad de serialización tras reorganizar los módulos."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import preparacion_train_test as pt
from scripts.actualizar_indice_transformacion import NOTEBOOK
from scripts.contrato_datasets import CARPETA_PROYECTO, DATA_DIR, RAIZ, ruta_dataset


def test_rutas_independientes_del_directorio_actual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ruta_dataset('PropUsadas_Integrado.csv') == DATA_DIR / 'PropUsadas_Integrado.csv'
    assert ruta_dataset('PropUsadas_OLS.csv').is_file()
    assert NOTEBOOK == CARPETA_PROYECTO / '2_Transformacion_PropUsadas.ipynb'
    assert NOTEBOOK.is_file() and (RAIZ / 'requirements.txt').is_file()


def test_pickle_historico_resuelve_el_modulo_antes_del_traslado(monkeypatch):
    objeto = pt.PreparacionInicial()
    with monkeypatch.context() as contexto:
        contexto.setattr(pt.PreparacionInicial, '__module__', 'preparacion_train_test')
        historico = pickle.dumps(objeto)
    recuperado = pickle.loads(historico)
    assert type(recuperado) is pt.PreparacionInicial
    assert type(recuperado).__module__ == 'scripts.preparacion_train_test'
    assert sys.modules['preparacion_train_test'] is pt


def test_la_raiz_solo_conserva_notebooks_documentacion_y_carpetas():
    assert not list(CARPETA_PROYECTO.glob('*.py'))
    assert not list(CARPETA_PROYECTO.glob('*.csv'))
    assert {p.name for p in DATA_DIR.iterdir()} == {
        'PropUsadas_Integrado.csv', 'PropUsadas_OLS.csv',
        'train_transformado.csv', 'test_transformado.csv',
    }
