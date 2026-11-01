"""Verificación matemática y del protocolo Group Lasso + OLS, sin MLflow remoto."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import group_lasso_ols as gl
from scripts import seleccion_modelos as s
class GroupLassoTests(unittest.TestCase):
    def test_resumen_dos_columnas_con_total(self):
        modelo = SimpleNamespace(model=SimpleNamespace(exog_names=['const', 'a'], rank=2))
        tabla = s.resumen_caracteristicas(['area'], {'area': ['a']}, modelo, mostrar_orden=False)
        self.assertIn('| Total: 1 características | 1 predictores |', tabla)
        self.assertNotIn('Orden', tabla)

    def test_solucion_ortonormal_por_bloques(self):
        xy = np.array([3., 4., .1])
        beta, info = gl.resolver_group_lasso(np.eye(3), xy, np.array([0, 0, 1]), np.array([np.sqrt(2), 1]), 1.)
        esperado = np.array([3., 4., 0.])
        esperado[:2] *= 1-np.sqrt(2)/5
        np.testing.assert_allclose(beta, esperado, atol=1e-12)
        self.assertTrue(info['convergencia'])
        self.assertLessEqual(info['residuo_kkt_relativo'], 1e-7)

    def test_no_se_detiene_en_primer_rechazo(self):
        candidatos = [{'seleccionadas': list(range(k)), 'lambda': 1/k,
                       'rmse_validation_original': score} for k, score in [(1, 90), (2, 89.8), (3, 88), (13, 10)]]
        elegido, pasos = gl.elegir_por_complejidad(candidatos, 100)
        self.assertEqual(len(elegido['seleccionadas']), 3)
        self.assertEqual([p['aceptado'] for p in pasos], [True, False, True])
        self.assertTrue(all(p['n_caracteristicas'] <= 12 for p in pasos))

    def test_el_mejor_de_cada_complejidad(self):
        candidatos = [{'seleccionadas': ['a'], 'lambda': .1, 'rmse_validation_original': 99.9},
                      {'seleccionadas': ['b'], 'lambda': .2, 'rmse_validation_original': 90}]
        elegido, _ = gl.elegir_por_complejidad(candidatos, 100)
        self.assertEqual(elegido['seleccionadas'], ['b'])
        rechazado, _ = gl.elegir_por_complejidad(candidatos[:1], 100)
        self.assertEqual(rechazado['seleccionadas'], [])

    def test_no_convergencia_no_se_acepta(self):
        with self.assertRaises(RuntimeError):
            gl.resolver_group_lasso(np.array([[1., .8], [.8, 2.]]), np.array([1., 2.]),
                np.array([0, 1]), np.ones(2), .01, max_iter=1, tol=1e-15)

    def test_escalador_solo_fit_historial_y_grupos_completos(self):
        rng = np.random.default_rng(3)
        c = np.tile([0, 1, 2], 60)
        X = pd.DataFrame({'d_a': (c == 1).astype(float), 'd_b': (c == 2).astype(float),
                          'area': rng.normal(size=180)})
        X.loc[120:, 'area'] += 2
        y = np.exp(15 + .5*X.d_a + .2*X.d_b + .3*X.area + rng.normal(0, .02, len(X)))
        grupos = {'district': ['d_a', 'd_b'], 'area': ['area']}
        with TemporaryDirectory() as tmp:
            resultado = gl.buscar_group_lasso(X, y, grupos, np.arange(120), np.arange(120, 180), tmp,
                                              lambdas=[.1, .03, .01])
            scaler = joblib.load(Path(tmp)/'scaler_seleccion.joblib')
            np.testing.assert_allclose(scaler.mean_, X.iloc[:120].mean())
            self.assertFalse(np.allclose(scaler.mean_, X.mean()))
            self.assertEqual(len(resultado['historial']), 3)
            self.assertLessEqual(len(resultado['seleccionadas']), 12)
            self.assertEqual(resultado['ponderacion']['district']['p_g'], 2)
            self.assertAlmostEqual(resultado['ponderacion']['district']['peso'], np.sqrt(2))
            columnas = s.columnas_grupos(grupos, resultado['seleccionadas'])
            if 'district' in resultado['seleccionadas']:
                self.assertTrue({'d_a', 'd_b'} <= set(columnas))
            importancia = s.importancia_permutacion_grupos(X, y, grupos, resultado['seleccionadas'],
                np.arange(120), np.arange(120, 180), n_repeticiones=10,
                rmse_esperado=resultado['rmse_validation_original'])
            self.assertTrue(importancia.groupby('caracteristica').size().eq(10).all())


if __name__ == '__main__':
    unittest.main()
