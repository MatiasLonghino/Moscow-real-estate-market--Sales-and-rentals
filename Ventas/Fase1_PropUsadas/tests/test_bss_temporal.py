"""CV mínima: cronología, preparación sin futuro, RSS por tamaño y regla 1-SE."""
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import bss_temporal as t,bss_ols as bs,baseline_ols as b,preparacion_train_test as pt


def muestra():
    rng=np.random.default_rng(42); n=800
    d=pd.DataFrame({'id':[f'{i:05}' for i in range(n)],'date_posted_dt':pd.date_range('2020-01-01',periods=n),
        'dataset_split':'train','total_area':rng.uniform(100,200,n),'living_area':rng.uniform(40,60,n),
        'kitchen_area':rng.uniform(10,20,n),'rooms':rng.integers(1,6,n),'floor':rng.integers(1,11,n),
        'total_floors':20.,'building_year':rng.integers(1980,2010,n),'ceiling_height':rng.uniform(2.5,3.5,n),
        'metro_distance_min':rng.uniform(1,20,n),'to_center_km':rng.uniform(1,30,n),
        'metro_to_center_km':rng.uniform(1,35,n),'mortgage_key_rate_at_listing':12.,
        'secondary_mom_change_pct':.1,'has_balcony':rng.choice(['true','false'],n),
        'metro_distance_type':rng.choice(['walk','transport'],n),'seller_type':'owner',
        'building_type':rng.choice([f'b{i}' for i in range(6)],n),
        'renovation':rng.choice([f'r{i}' for i in range(4)],n),'metro_line':'line',
        'district':[f'd{i%129}' for i in range(n)]})
    d['price_rub']=np.exp(10+.004*d.total_area+.025*d.rooms+rng.normal(0,.08,n))
    nombres=['total_area','living_area','rooms','ceiling_height','metro_distance_min','to_center_km',
             'surface_corrected','has_balcony','metro_distance_type','date_posted_dt']
    grupos={g:[g] for g in nombres}
    grupos['date_posted_dt']=['months_since_start']
    grupos.update(building_type=[f'building_type_b{i}' for i in range(1,6)],
        renovation=[f'renovation_r{i}' for i in range(1,4)],district=[f'district_d{i}' for i in range(1,129)])
    return d,grupos


def test_motor_minimo_equivale_a_ols_y_no_calcula_diagnosticos():
    rng=np.random.default_rng(4)
    X=pd.DataFrame(rng.normal(size=(100,5)),columns=list('abcde'))
    z=10+.4*X.a+.2*X.b+rng.normal(0,.02,len(X))
    with patch.object(bs,'_estadisticos',side_effect=AssertionError('No estadísticas extra')), \
         patch.object(bs.b.sm,'OLS',side_effect=AssertionError('No statsmodels por combinación')):
        ganadores,control=bs.buscar_rss_minimo(X,z,{c:[c] for c in X},[1,2,3])
    assert control['evaluados']==sum(math.comb(5,k) for k in (1,2,3))
    for k,g in ganadores.items():
        cols=g['columnas']
        m,_=b.ajustar(X[cols],z)
        np.testing.assert_allclose(m.params.iloc[1:],g['coef'],atol=1e-10)
        np.testing.assert_allclose(m.ssr,g['rss'],atol=1e-10)
        assert len(g['seleccionadas'])==k
    unico,control=bs.buscar_rss_minimo(X,z,{c:[c] for c in X},[2])
    assert list(unico)==[2] and control['evaluados']==math.comb(5,2)


def test_folds_expansivos_por_dias_completos_sin_solapamiento():
    d,_=muestra()
    d['date_posted_dt']=pd.Timestamp('2020-01-01')+pd.to_timedelta(np.arange(len(d))//7,unit='D')
    folds=t.folds_expansivos(d)
    assert len(folds)==3 and folds[-1]['val_stop']==len(d)
    for f in folds:
        assert f['fit_start']==0 and f['fit_stop']==f['val_start']
        assert d.date_posted_dt.iloc[f['fit_stop']-1]<d.date_posted_dt.iloc[f['val_start']]
    assert folds[0]['fit_stop']<folds[1]['fit_stop']<folds[2]['fit_stop']
    assert folds[0]['val_stop']==folds[1]['val_start'] and folds[1]['val_stop']==folds[2]['val_start']


def test_preparacion_ajustada_solo_en_pasado_y_bloques_fijos():
    d,g=muestra()
    d.loc[:199,'rooms']=1.
    d.loc[200:,'rooms']=100.
    preparador=t.PreparadorTemporal(g).fit(d.iloc[:200])
    indice=pt.NUMERICAS.index('rooms')
    assert preparador.preparacion.numerico_.statistics_[indice]==1.
    valid=d.iloc[200:250].copy(); valid['rooms']=np.nan
    V=preparador.transform(valid)
    assert V.rooms.eq(1).all() and preparador.n_fit==200
    assert V.shape[1]==146 and all(c in V for c in g['district'])
    assert preparador.fecha_hasta<str(valid.date_posted_dt.min().date())


def test_1se_usa_error_estandar_del_minimo_y_no_mejora_05():
    e=np.full((12,3),120.)
    e[0]=130.; e[1]=105.; e[2]=101.; e[7]=[90.,100.,110.]
    filas,r=t.elegir_1se(e)
    assert len(filas)==12 and r['k_min']==8 and r['k_selected']==2
    np.testing.assert_allclose(r['threshold'],100+np.std([90,100,110],ddof=1)/np.sqrt(3))
    assert filas[1]['se']==0. and r['se_at_minimum']>0
    assert 'min_mejora_pct' not in t.CONFIG


@pytest.fixture(scope='module')
def experimento():
    d,g=muestra()
    fits=[]
    original=b.ajustar
    def ajustar(X,y):
        fits.append(len(X))
        return original(X,y)
    with patch.object(b,'ajustar',side_effect=ajustar), \
         patch.object(b,'diagnosticar',side_effect=AssertionError('No diagnósticos')), \
         patch.object(bs,'buscar_bss',side_effect=AssertionError('No ruta histórica')), \
         patch.object(bs,'_estadisticos',side_effect=AssertionError('No AIC/BIC')):
        r=t.ejecutar_cv(d,g)
    return d,g,r,fits


def test_cv_3_folds_12_rmse_y_un_solo_ols_definitivo(experimento):
    d,g,r,fits=experimento
    assert len(g)==13 and sum(map(len,g.values()))==146
    assert fits==[len(d)]
    assert len(r['rows'])==12 and [v['k'] for v in r['rows']]==list(range(1,13))
    for f,c in zip(r['folds'],r['checks']):
        assert c['preprocessor_fit_rows']==f['fit_stop']
        assert c['evaluados']==8190
    assert r['final_check']['evaluados']==math.comb(13,r['rule']['k_selected'])
    assert r['model'].model.rank==len(r['columns'])+1
    assert r['model'].nobs==len(d)
    assert len(r['selected_features'])==r['rule']['k_selected']


def test_tabla_unica_de_12_filas_y_detalles_cerrados(experimento):
    d,g,res,_=experimento
    r=dict(resultados=res,configuracion=dict(identity={'n_train':len(d)}),modo='recuperar',
        run=dict(run_id='id',experiment_id='5',run_name=t.NOMBRE_RUN,status='FINISHED'))
    tabla=t.tabla_presentacion(r)
    assert len(tabla.splitlines())==14 and '**' in tabla and 'RMSE Fold 3' in tabla
    assert 'características' in t.modelo_presentacion(r)
    oculto=t.detalles_presentacion(r)
    assert oculto.startswith('<details>') and '<details open' not in oculto
    assert 'Tukey' in oculto and 'no una garantía estadística' in oculto


def test_lector_no_usa_valores_de_test(tmp_path):
    d,_=muestra()
    futuro=d.iloc[:2].copy(); futuro['id']=['t1','t2']; futuro['date_posted_dt']=pd.Timestamp('2026-01-01'); futuro['price_rub']=1e15
    p=tmp_path/'data'; p.mkdir()
    fuente=pd.concat([d,futuro],ignore_index=True)
    fuente.to_csv(p/'PropUsadas_Integrado.csv',index=False)
    a,identidad=t.cargar_train_crudo(tmp_path)
    assert len(a)==len(d) and a.dataset_split.eq('train').all()
    fuente.loc[fuente.id.isin(['t1','t2']),'price_rub']=1.
    fuente.to_csv(p/'PropUsadas_Integrado.csv',index=False)
    aa,otra=t.cargar_train_crudo(tmp_path)
    pd.testing.assert_frame_equal(a,aa)
    assert identidad['train_sha256']==otra['train_sha256']


def test_persistencia_minima_y_registro_por_lote(experimento):
    d,g,res,_=experimento
    cfg=dict(version=t.VERSION,algorithm=t.CONFIG,source_bss_run_id='bss',groups=g,folds=res['folds'],
        preprocessing='past_only',identity={'train_sha256':'hash','n_train':len(d)},limitations=t.LIMITACIONES)
    with TemporaryDirectory() as tmp:
        out=Path(tmp)
        # El cálculo ya está comprobado: no repetir CV para probar la persistencia.
        with patch.object(t,'ejecutar_cv',return_value=res):
            r=t.preparar_local(d,cfg,out)
        assert set(p.name for p in out.iterdir())=={'modelo_statsmodels.pkl','preparador.joblib','configuracion.json',
            'resultados.json','inferencia.json','bss_temporal.py','bss_ols.py','preparacion_train_test.py'}
        t.validar_resultados(r,cfg)
        cliente=Mock()
        cliente.get_experiment_by_name.return_value=SimpleNamespace(experiment_id='5',lifecycle_stage='active')
        cliente.search_runs.return_value=[]
        cliente.create_run.return_value=SimpleNamespace(info=SimpleNamespace(run_id='cv'))
        archivos={}
        def log_artifacts(rid,path): archivos.update({p.name:p.read_bytes() for p in Path(path).iterdir()})
        def download(rid,nombre,destino):
            p=Path(destino)/nombre; p.write_bytes(archivos[nombre]); return str(p)
        cliente.log_artifacts.side_effect=log_artifacts; cliente.download_artifacts.side_effect=download
        assert t.registrar(cliente,cfg,out)['run_id']=='cv'
        assert cliente.log_batch.call_count==1
        assert not any('aic' in m.key or 'vif' in m.key for m in cliente.log_batch.call_args.kwargs['metrics'])
        cliente.set_terminated.assert_called_once_with('cv','FINISHED')
        assert not list(out.glob('*.png'))
