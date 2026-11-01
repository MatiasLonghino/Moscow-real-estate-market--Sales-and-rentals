"""Verifica contratos de preparación sin entrenar estimadores."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experimentos_ols import crear_preprocesador, tabla_markdown, validacion_temporal


class PreparacionOLS(unittest.TestCase):
    def test_preprocesador_es_una_plantilla_sin_ajustar(self):
        prepro=crear_preprocesador(numericas=['lat'],categoricas=['district_group'])
        self.assertFalse(hasattr(prepro,'transformers_'))
        self.assertEqual(prepro.transformers[0][2],['lat'])
        self.assertEqual(prepro.transformers[1][2],['district_group'])

    def test_cv_no_acepta_mapa_de_train_completo(self):
        with self.assertRaises(ValueError):
            validacion_temporal(pd.DataFrame({'district_group':['G01']}),
                predictoras=['district_group'],numericas=[],categoricas=['district_group'])

    def test_markdown_no_requiere_tabulate(self):
        tabla=pd.DataFrame({'Nombre':['x|y'],'Valor':[1.25]})
        texto=tabla_markdown(tabla,decimales=2)
        self.assertIn('x\\|y',texto)
        self.assertIn('1.25',texto)
        self.assertEqual(len(texto.splitlines()),3)


if __name__=='__main__':
    unittest.main()
