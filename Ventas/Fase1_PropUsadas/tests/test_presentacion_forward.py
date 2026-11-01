"""Contratos de la presentación por grupos y su importancia por permutación."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import seleccion_modelos as s
class PresentacionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        categoria = np.tile([0, 1, 2], 60)
        self.X = pd.DataFrame({'d_a': (categoria == 1).astype(float),
                               'd_b': (categoria == 2).astype(float), 'area': rng.normal(size=180)})
        self.y = np.exp(15 + .5*self.X.d_a + .2*self.X.d_b + .15*self.X.area)
        self.grupos = {'district': ['d_a', 'd_b'], 'area': ['area']}
        self.fit, self.val = np.arange(120), np.arange(120, 180)

    def test_permutacion_conjunta_un_ajuste_y_formula(self):
        ajustar_original = s.b.ajustar
        matrices = []
        def ajustar(*args):
            modelo, diseno = ajustar_original(*args)
            predict_original = modelo.predict
            def predecir(matriz):
                matrices.append(np.asarray(matriz).copy())
                return predict_original(matriz)
            modelo.predict = predecir
            return modelo, diseno
        # Añade ruido para un RMSE basal positivo y estable.
        self.y *= np.exp(np.random.default_rng(10).normal(0, .01, len(self.y)))
        with patch.object(s.b, 'ajustar', side_effect=ajustar) as fit:
            resultado = s.importancia_permutacion_grupos(
                self.X, self.y, self.grupos, ['district', 'area'], self.fit, self.val)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(resultado), 10)
        self.assertEqual(len(matrices), 11)
        for matriz in matrices:
            self.assertTrue(np.all(matriz[:, 0] == 1))
            self.assertTrue(np.all(matriz[:, 1:3].sum(axis=1) <= 1))
            self.assertEqual(sorted(map(tuple, matriz[:, 1:3])),
                             sorted(map(tuple, matrices[0][:, 1:3])))
        np.testing.assert_allclose(resultado.importancia_pct,
            100*(resultado.rmse_permutado-resultado.rmse_original)/resultado.rmse_original)
        self.assertEqual(resultado.groupby('caracteristica').size().to_dict(), {'area': 5, 'district': 5})
        segundo = s.importancia_permutacion_grupos(
            self.X, self.y, self.grupos, ['district', 'area'], self.fit, self.val)
        pd.testing.assert_frame_equal(resultado, segundo)

    def test_resumen_sin_columnas_o_intercepto(self):
        modelo, _ = s.b.ajustar(self.X, np.log(self.y))
        tabla = s.resumen_caracteristicas(['district', 'area'], self.grupos, modelo)
        self.assertIn('| 1 | `district` | 2 |', tabla)
        self.assertIn('| Total | 2 características | 3 predictores |', tabla)
        self.assertNotIn('d_a', tabla)
        self.assertNotIn('d_b', tabla)
        self.assertNotIn('const', tabla)


if __name__ == '__main__':
    unittest.main()
