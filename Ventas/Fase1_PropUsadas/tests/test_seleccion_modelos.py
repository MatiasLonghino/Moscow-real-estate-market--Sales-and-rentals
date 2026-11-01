"""Pruebas del contrato Forward y la recuperación histórica, sin conexión remota."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import seleccion_modelos as s
class ForwardTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(180, 15)), columns=[f'x{i}' for i in range(15)])
        self.log = (15 + self.X.x0*.25 + self.X.x1*.18 + rng.normal(0, .005, len(self.X))).rename('log')
        self.y = np.exp(self.log)
        self.grupos = {c: [c] for c in self.X}
        self.fit, self.val = np.arange(140), np.arange(140, 180)

    def buscar(self, **kw):
        return s.forward_grupos(self.X, self.log, self.y, self.grupos, self.fit, self.val, **kw)

    def test_umbral_desde_intercepto(self):
        r = self.buscar(min_mejora_pct=100)
        self.assertEqual(r['seleccionadas'], [])
        self.assertEqual(r['iteraciones'], 1)
        self.assertEqual(r['candidatos_evaluados'], 15)
        self.assertEqual(r['motivo_parada'], 'min_mejora_pct')
        self.assertEqual(r['rmse_validation_original'], r['rmse_intercepto_original'])

    def test_maximo_sin_evaluaciones_posteriores(self):
        r = self.buscar(max_features=1)
        self.assertEqual(len(r['seleccionadas']), 1)
        self.assertEqual(r['motivo_parada'], 'max_features')
        self.assertEqual(r['candidatos_evaluados'], 15)
        with self.assertRaises(ValueError):
            self.buscar(max_features=13)

    def test_bloques_y_no_eliminacion(self):
        self.grupos = {'bloque': ['x0', 'x1'], **{c: [c] for c in self.X.columns[2:]}}
        r = self.buscar()
        self.assertEqual(r['seleccionadas'][0], 'bloque')
        self.assertEqual(r['historial'][0]['n_predictores'], 2)
        for h in r['historial']:
            self.assertEqual(h['caracteristicas_acumuladas'][:1], ['bloque'])
        cols = s.columnas_grupos(self.grupos, r['seleccionadas'])
        modelo, _ = s.b.ajustar(self.X.loc[self.fit, cols], self.log.iloc[self.fit])
        pred = np.exp(modelo.predict(s.b.sm.add_constant(self.X.loc[self.val, cols])))
        esperado = np.sqrt(np.mean((self.y.iloc[self.val]-pred)**2))
        np.testing.assert_allclose(r['rmse_validation_original'], esperado, atol=1e-7)

    def test_limite_absoluto_12(self):
        rng = np.random.default_rng(8)
        z = rng.normal(size=(64, 15))
        z -= z.mean(axis=0)
        q = np.linalg.qr(z)[0]
        X = pd.DataFrame(np.vstack([q, q]), columns=[f'x{i}' for i in range(15)])
        log = 15 + .3*X.sum(axis=1)
        r = s.forward_grupos(X, log, np.exp(log), {c: [c] for c in X},
                             np.arange(64), np.arange(64, 128), min_mejora_pct=0)
        self.assertEqual(len(r['seleccionadas']), 12)
        self.assertEqual(r['motivo_parada'], 'max_features')
        self.assertEqual(r['candidatos_evaluados'], sum(range(4, 16)))
        self.assertEqual(max(c['iteracion'] for c in r['candidatos']), 12)

    def test_diagnostico_solo_intercepto(self):
        modelo, diseno = s.b.ajustar(self.X[[]], self.log)
        m, invalidas, vif, notas = s.b.diagnosticar(
            modelo, diseno, self.log, y_original=self.y, resumir_vif=True)
        self.assertEqual(m['train_n_parameters_effective'], 1)
        self.assertEqual(m['train_n_predictors'], 0)
        self.assertTrue(vif.empty)
        self.assertNotIn('train_bp_lm_log', m)
        self.assertIn('train_vif_max', invalidas)

    def test_ohe_parcial_rechazado(self):
        resumen = {'linaje': {'d_a': {'tipo': 'onehot', 'variable': 'd', 'entradas': ['d']}},
                   'categorias': [{'variable': 'd', 'columnas': ['d_a', 'd_b'],
                                   'categorias': ['ref', 'a', 'b'], 'referencia': 'ref'}]}
        with self.assertRaises(ValueError):
            s.grupos_caracteristicas(['d_a'], resumen)

    def test_tabla_ausencias_y_escalas(self):
        tabla = s.tabla_comparativa({'train_rmse_log': 2.1},
            {'train_rmse_log': 1.9, 'train_vif_mean': 1.2, 'selection_rmse_validation_original': 100})
        self.assertIn('N/D', tabla)
        self.assertNotIn('selection_rmse', tabla)
        self.assertIn(repr(1.9-2.1), tabla)


class RecuperacionTests(unittest.TestCase):
    def run_mock(self, rid='a', status='FINISHED', target='ln(Target)'):
        return NS(info=NS(run_id=rid, status=status, experiment_id='5'),
                  data=NS(tags={'mlflow.runName': 'otro_modelo'},
                          params={'target': target}, metrics={'train_rmse_log': 2.123456789012345}))

    def cliente(self, runs):
        cliente = Mock()
        cliente.get_experiment_by_name.return_value = NS(experiment_id='5', lifecycle_stage='active')
        class Pagina(list):
            token = None
        cliente.search_runs.return_value = Pagina(runs)
        cliente.get_run.return_value = runs[0]
        return cliente

    def test_por_nombre_o_id_sin_modificar_metricas(self):
        run = self.run_mock()
        cliente = self.cliente([run])
        for kw in ({'nombre': 'otro_modelo'}, {'run_id': 'a'}):
            ref = s.recuperar_experimento(cliente, **kw)
            self.assertEqual(ref['metrics'], run.data.metrics)
            self.assertEqual(ref['run_id'], 'a')

    def test_ambiguedad_y_parametros(self):
        cliente = self.cliente([self.run_mock(), self.run_mock('b', target='none')])
        with self.assertRaises(ValueError):
            s.recuperar_experimento(cliente, nombre='otro_modelo')
        ref = s.recuperar_experimento(cliente, nombre='otro_modelo', parametros={'target': 'ln(Target)'})
        self.assertEqual(ref['run_id'], 'a')

    def test_rechaza_no_finished(self):
        with self.assertRaises(ValueError):
            s.recuperar_experimento(self.cliente([self.run_mock(status='FAILED')]), run_id='a')


if __name__ == '__main__':
    unittest.main()
