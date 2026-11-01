"""Restitución de datos en un modelo guardado, sin cambiar el ajuste original."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluacion_ols_final as e, baseline_ols as b


class EvaluacionOLSFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ajuste sintético exclusivo de la prueba; nunca recupera modelos reales ni usa red.
        rng = np.random.default_rng(237)
        X = pd.DataFrame(rng.normal(size=(160, 3)), columns=['a','b','c'])
        cls.D = sm.add_constant(X)
        cls.y = pd.Series(12 + X.to_numpy()@np.array([.3,-.2,.1]) + rng.normal(0,.08,len(X)), name='ln_price_rub')
        cls.original = sm.OLS(cls.y, cls.D).fit()
        cls.metricas = b.diagnosticar(cls.original, cls.D, cls.y, y_original=np.exp(cls.y), resumir_vif=True)[0]

    def recuperar(self):
        with tempfile.TemporaryDirectory() as carpeta:
            p = Path(carpeta)/'modelo.pkl'
            self.original.save(p, remove_data=True)
            return OLSResults.load(p)

    def test_diagnosticos_de_copia_reconstruida_equivalen_al_ajuste(self):
        guardado = self.recuperar()
        params = guardado.params.copy()
        with patch.object(sm.OLS, 'fit', side_effect=AssertionError('No reajustar el modelo')):
            copia = e.restaurar_datos_diagnostico(guardado, self.D, self.y)
        metricas, invalidas, vif, _ = b.diagnosticar(copia, self.D, self.y, y_original=np.exp(self.y), resumir_vif=True)
        self.assertFalse(invalidas)
        for clave, valor in self.metricas.items():
            np.testing.assert_allclose(metricas[clave], valor, rtol=1e-10, atol=1e-12, err_msg=clave)
        np.testing.assert_array_equal(guardado.params, params)
        self.assertIsNone(guardado.model.exog)
        self.assertIsNone(guardado.model.endog)
        self.assertNotIn('const', vif.variable.tolist())

    def test_rechaza_orden_de_columnas_o_filas_alterados(self):
        guardado = self.recuperar()
        for D, y in [(self.D.iloc[:,::-1],self.y), (self.D.iloc[:-1],self.y.iloc[:-1]),
                     (self.D,self.y.set_axis(np.arange(1,len(self.y)+1)))]:
            with self.assertRaises(ValueError):
                e.restaurar_datos_diagnostico(guardado, D, y)

    def test_no_sustituye_metricas_historicas_ausentes(self):
        with self.assertRaisesRegex(ValueError, 'Métrica histórica ausente'):
            e._valor({}, 'train_aic_log')


if __name__=='__main__':
    unittest.main()
