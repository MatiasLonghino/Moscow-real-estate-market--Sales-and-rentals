"""Pruebas sin conexión ni ajustes de modelos: python -m unittest discover -s .../tests."""

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.transformaciones_experimentales import (
    ARTEFACTO_MAPAS, aplicar_agrupamiento, contexto_entrenamiento, crear_artefacto,
    generar_representacion, huella_json, recuperar_agrupamiento,
    preparar_df_seleccionado,
)


class ReutilizacionTest(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({'id': range(6),
            'date_posted_dt': pd.date_range('2020-01-01', periods=6),
            'dataset_split': ['train']*6, 'district': ['a','b','c']*2,
            'lat': [1.,2.,3.]*2, 'price_rub': [100.,200.,300.]*2})
        self.valid = self.train.iloc[:2].copy()
        self.valid['id'] += 100
        self.valid['date_posted_dt'] += pd.Timedelta(days=31)
        self.valid['dataset_split'] = 'validation'
        kwargs = dict(dataset_sha256='datos_inmutables', caracteristicas=['lat','district'],
                      numericas=['lat'], categoricas=['district'], preprocesamiento={'escalar': False})
        self.ctx = contexto_entrenamiento(self.train, **kwargs)
        self.fold = contexto_entrenamiento(self.train.iloc[:3], fold=1,
            periodo_validacion={'desde':'2020-01-04','hasta':'2020-01-06'}, **kwargs)
        self.rid = 'run_prueba'
        self.mapa = {'a':'G01','b':'G02','c':'G03'}
        self.a = crear_artefacto('Ward',3,self.rid,
            {'train_completo':self.mapa,'fold_1':self.mapa},
            {'train_completo':self.train,'fold_1':self.train.iloc[:3]},
            {'train_completo':self.ctx,'fold_1':self.fold})
        self.cache = {(self.rid,ARTEFACTO_MAPAS):self.a}
        self.cliente = SimpleNamespace(
            get_experiment_by_name=lambda _: SimpleNamespace(experiment_id='1',lifecycle_stage='active'),
            get_run=lambda _: SimpleNamespace(info=SimpleNamespace(experiment_id='1',status='FINISHED')))
        self.config = {'pasos':[{'tipo':'agrupamiento','algoritmo':'Ward','K':3,'run_id':self.rid}]}

    def recuperar(self, ctx=None, fold=None):
        return recuperar_agrupamiento(self.cliente,algoritmo='Ward',k=3,run_id=self.rid,
            contexto_esperado=self.ctx if ctx is None else ctx,fold=fold,cache=self.cache)

    def test_original_y_agrupado_no_modifican_datos(self):
        antes = self.valid.copy(deep=True)
        original = generar_representacion(self.valid,{'pasos':[]},cliente=self.cliente,
            contexto_esperado=self.ctx,uso='validation',cache=self.cache)
        grupo = generar_representacion(self.valid,self.config,cliente=self.cliente,
            contexto_esperado=self.ctx,uso='validation',cache=self.cache)
        self.assertIn('district',original['X'])
        self.assertNotIn('district',grupo['X'])
        self.assertIn('district',grupo['datos'])
        grupo['datos'].loc[:,'lat'] = -999
        pd.testing.assert_frame_equal(self.valid,antes)

    def test_rechaza_cambios_estadisticos(self):
        for campo in ['dataset_sha256','entrenamiento_sha256','caracteristicas_efectos',
                      'preprocesamiento','versiones','hasta','ols_preliminar']:
            with self.subTest(campo=campo):
                ctx = copy.deepcopy(self.ctx)
                ctx[campo] = 'cambio'
                with self.assertRaises(ValueError):
                    self.recuperar(ctx)

    def test_no_reutiliza_train_completo_en_fold(self):
        with self.assertRaises(ValueError):
            self.recuperar(self.fold)
        with self.assertRaises(ValueError):
            self.recuperar(self.fold,fold=2)
        t = self.recuperar()
        with self.assertRaises(ValueError):
            aplicar_agrupamiento(self.train.iloc[:3],t,contexto_esperado=self.ctx,uso='train')

    def test_rechaza_otro_periodo_y_test(self):
        t = self.recuperar(self.fold,fold=1)
        with self.assertRaises(ValueError):
            aplicar_agrupamiento(self.valid,t,contexto_esperado=self.fold,uso='validation')
        tabla_test = self.valid.assign(dataset_split='test')
        with self.assertRaises(ValueError):
            aplicar_agrupamiento(tabla_test,self.recuperar(),contexto_esperado=self.ctx,uso='validation')

    def test_fallback_reproducible_solo_train(self):
        tabla = self.valid.assign(district='desconocido')
        resultado,_,n = aplicar_agrupamiento(tabla,self.recuperar(),contexto_esperado=self.ctx,uso='validation')
        self.assertEqual(n,2)
        self.assertTrue(resultado['district_group'].eq('G01').all())

    def test_seleccion_independiente_y_combinada(self):
        for metodo in ['forward','backward']:
            for combinado in [False,True]:
                with self.subTest(metodo=metodo,combinado=combinado):
                    entradas = ['lat','district_group'] if combinado else ['lat','district']
                    seleccion = {'tipo':'seleccion','metodo':metodo,'caracteristicas':['lat'],
                        'columnas_entrada':entradas,'criterio':'contrato de prueba',
                        'contexto_sha256':huella_json(self.ctx),
                        'origen_transformaciones':[f'{self.rid}:train_completo'] if combinado else []}
                    pasos = self.config['pasos']+[seleccion] if combinado else [seleccion]
                    resultado = generar_representacion(self.valid,{'pasos':pasos},cliente=self.cliente,
                        contexto_esperado=self.ctx,uso='validation',cache=self.cache)
                    self.assertEqual(resultado['X'].columns.tolist(),['lat'])
                    seleccion['columnas_entrada'] = ['price_rub']
                    with self.assertRaises(ValueError):
                        generar_representacion(self.valid,{'pasos':pasos},cliente=self.cliente,
                            contexto_esperado=self.ctx,uso='validation',cache=self.cache)

    def test_mapa_incompleto_o_alterado(self):
        with self.assertRaises(ValueError):
            crear_artefacto('Ward',3,self.rid,{'train_completo':{'a':'G01'}},
                {'train_completo':self.train},{'train_completo':self.ctx})
        self.cache[(self.rid,ARTEFACTO_MAPAS)]['agrupamientos']['train_completo']['mapa']['a']='otro'
        with self.assertRaises(ValueError):
            self.recuperar()

    def test_exclusion_explicita_sin_algoritmo_de_seleccion(self):
        config = {'pasos':[{'tipo':'seleccion','metodo':'exclusion_explicita',
            'caracteristicas':['district'],'columnas_entrada':['lat','district'],
            'criterio':'Excluir exclusivamente lat por configuración',
            'contexto_sha256':huella_json(self.ctx),'origen_transformaciones':[]}]}
        antes = self.train.copy(deep=True)
        r = generar_representacion(self.train,config,cliente=self.cliente,
            contexto_esperado=self.ctx,uso='train',cache=self.cache)
        self.assertEqual(r['X'].columns.tolist(),['district'])
        pd.testing.assert_frame_equal(self.train,antes)

    def test_preparacion_reservada_no_aprende_de_test(self):
        test = self.valid.copy()
        test['id'] += 100
        test['date_posted_dt'] += pd.Timedelta(days=31)
        test['dataset_split'] = 'test'
        test['district'] = 'desconocido'
        base = pd.concat([self.train,self.valid,test],ignore_index=True)
        antes = base.copy(deep=True)
        t = self.recuperar()
        resultado,columnas = preparar_df_seleccionado(base,t,['lat'],
            predictoras_base=['lat','district'],uso='preparacion_reservada')
        self.assertEqual(columnas,['district_group'])
        self.assertNotIn('lat',resultado)
        self.assertIn('district',resultado)
        self.assertTrue(resultado.loc[resultado['dataset_split'].eq('test'),'district_group'].eq('G01').all())
        self.assertEqual(str(resultado['district_group'].dtype),'category')
        pd.testing.assert_series_equal(resultado['price_rub'],base['price_rub'])
        pd.testing.assert_frame_equal(base,antes)
        self.assertFalse(resultado.attrs['transformacion']['evaluada_conjuntamente'])

    def test_preparacion_rechaza_contexto_o_eliminaciones_distintas(self):
        with self.assertRaises(ValueError):
            preparar_df_seleccionado(self.train,self.recuperar(self.fold,fold=1),[],
                predictoras_base=['lat','district'],uso='preparacion_reservada')
        with self.assertRaises(ValueError):
            preparar_df_seleccionado(self.train,self.recuperar(),[],
                predictoras_base=['district'],uso='train')
        with self.assertRaises(ValueError):
            preparar_df_seleccionado(self.train,self.recuperar(),['price_rub'],
                predictoras_base=['lat','district'],uso='train')


if __name__ == '__main__':
    unittest.main()
