"""CV temporal mínima: 36 errores, regla 1-SE y un único OLS definitivo."""
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import baseline_ols as b,bss_ols as bs,seleccion_modelos as s,preparacion_train_test as pt
from .contrato_datasets import CONTRATO,RAIZ

NOMBRE_RUN='BSS_07_Log_Temporal_3F_1SE_OLS'
VERSION='temporal_bss_minimal_1se_v1'
CONFIG=dict(n_folds=3,max_features=12,rule='one_standard_error',target='ln(price_rub)',inverse='exp',bias_correction='none')
LIMITACIONES=('La unión y el vocabulario OHE están fijados por el BSS previo, construido con todo Train: '
    'esta CV es condicional a esa selección y no es una evaluación independiente. '
    'Se utiliza Train crudo anterior a Test, sin el filtro Tukey global del dataset histórico; '
    'por ello las filas y los errores no son directamente comparables con los históricos. '
    'Imputación y origen temporal se ajustan únicamente en el pasado de cada fold. '
    'El esquema OHE heredado es fijo; no se aprenden categorías con la validación. '
    'Con tres folds expansivos correlacionados, 1-SE es una regla práctica de parsimonia, '
    'no una garantía estadística de superioridad. Test no se utiliza.')


class PreparadorTemporal:
    """Imputación solo en fit; bloques del esquema heredado sin aprender niveles futuros."""
    def __init__(self,grupos): self.grupos=copy.deepcopy(grupos)

    def fit(self,datos):
        assert datos['dataset_split'].eq('train').all()
        self.preparacion=pt.PreparacionInicial().fit(datos)
        fechas=pd.to_datetime(datos.date_posted_dt)
        self.origen_mes=int(fechas.min().year*12+fechas.min().month-1)
        self.n_fit=len(datos)
        self.fecha_hasta=str(fechas.max().date())
        return self

    def transform(self,datos):
        preparado=self.preparacion.transform(datos)
        salida={}
        for g,cols in self.grupos.items():
            if g in ('district','building_type','renovation'):
                texto=preparado[g].astype(str).map(lambda v: pt.nombre_predictor(g+'_'+v))
                for c in cols: salida[c]=texto.eq(c).astype(float)
            elif g=='date_posted_dt':
                assert cols==['months_since_start']
                fechas=pd.to_datetime(datos.date_posted_dt)
                salida[cols[0]]=(fechas.dt.year*12+fechas.dt.month-1-self.origen_mes).astype(float)
            else:
                for c in cols:
                    assert c in preparado, f'Columna del esquema no reproducible: {c}'
                    salida[c]=preparado[c].astype(float)
        X=pd.DataFrame(salida,index=datos.index)
        assert list(X)==s.columnas_grupos(self.grupos,list(self.grupos))
        assert np.isfinite(X.to_numpy()).all()
        return X


def cargar_train_crudo(base):
    """No utiliza valores de Test ni la selección de filas Tukey ajustada con el futuro."""
    base=Path(base)
    ruta=base/'data/PropUsadas_Integrado.csv'
    encabezado=pd.read_csv(ruta,nrows=0).columns
    columnas=['id','date_posted_dt','price_rub']+pt.ENTRADAS_X
    columnas=[('mortgage_rate_at_listing' if c=='mortgage_key_rate_at_listing' and c not in encabezado else c) for c in columnas]
    corte=pd.Timestamp(CONTRATO['train']['fecha_max'])
    trozos=[]
    for parte in pd.read_csv(ruta,usecols=columnas,dtype={'id':'string'},chunksize=20000):
        fechas=pd.to_datetime(parte.date_posted_dt,errors='raise')
        pasada=parte.loc[fechas.le(corte)].copy()
        pasada['date_posted_dt']=fechas.loc[pasada.index]
        trozos.append(pasada)
    datos=pd.concat(trozos,ignore_index=True).sort_values(['date_posted_dt','id'],kind='stable').reset_index(drop=True)
    datos['dataset_split']='train'
    datos=pt.seleccionar_variables(datos)
    assert datos.id.is_unique and datos.date_posted_dt.is_monotonic_increasing
    assert datos.date_posted_dt.max()<=corte and datos.date_posted_dt.max()<pd.Timestamp(CONTRATO['test']['fecha_min'])
    assert np.isfinite(datos.price_rub).all() and datos.price_rub.gt(0).all()
    huella=hashlib.sha256(pd.util.hash_pandas_object(datos,index=False).to_numpy().tobytes()).hexdigest()
    return datos,dict(train_sha256=huella,n_train=len(datos),train_end=str(corte.date()),
        raw_file_sha256=b.sha256(ruta),source='chronological raw Train; no global Tukey filtering',test_used=False)


def folds_expansivos(datos):
    fechas=pd.to_datetime(datos.date_posted_dt)
    assert fechas.is_monotonic_increasing and datos.id.is_unique
    conteos=fechas.value_counts().sort_index().cumsum()
    assert len(conteos)>=4
    limites=[]
    for q in (.25,.5,.75):
        disponibles=conteos.iloc[:-1]
        if limites: disponibles=disponibles.loc[disponibles>limites[-1]]
        fecha=(disponibles-len(datos)*q).abs().idxmin()
        limites.append(int(conteos.loc[fecha]))
    assert 0<limites[0]<limites[1]<limites[2]<len(datos)
    cortes=limites+[len(datos)]
    folds=[]
    for i in range(3):
        fin=cortes[i]; hasta=cortes[i+1]
        assert fechas.iloc[fin-1]<fechas.iloc[fin]
        folds.append(dict(fold=i+1,fit_start=0,fit_stop=fin,val_start=fin,val_stop=hasta,
            fit_from=str(fechas.iloc[0].date()),fit_until=str(fechas.iloc[fin-1].date()),
            val_from=str(fechas.iloc[fin].date()),val_until=str(fechas.iloc[hasta-1].date())))
    return folds


def elegir_1se(errores):
    valores=np.asarray(errores,dtype=float)
    assert valores.shape==(12,3) and np.isfinite(valores).all() and (valores>=0).all()
    medias=valores.mean(axis=1)
    se=valores.std(axis=1,ddof=1)/np.sqrt(3)
    mejor=int(np.argmin(medias))
    umbral=float(medias[mejor]+se[mejor])
    elegido=int(np.flatnonzero(medias<=umbral)[0])
    filas=[dict(k=k+1,fold_1=float(v[0]),fold_2=float(v[1]),fold_3=float(v[2]),
                mean=float(medias[k]),se=float(se[k])) for k,v in enumerate(valores)]
    return filas,dict(k_min=mejor+1,k_selected=elegido+1,threshold=umbral,
                     selected_mean=float(medias[elegido]),se_at_minimum=float(se[mejor]))


def ejecutar_cv(datos,grupos):
    assert len(grupos)==13 and sum(map(len,grupos.values()))==146
    folds=folds_expansivos(datos)
    errores=np.empty((12,3))
    auditoria=[]
    for i,fold in enumerate(folds):
        fit=datos.iloc[:fold['fit_stop']]
        valid=datos.iloc[fold['val_start']:fold['val_stop']]
        assert fit.date_posted_dt.max()<valid.date_posted_dt.min()
        preparador=PreparadorTemporal(grupos).fit(fit)
        X=preparador.transform(fit)
        V=preparador.transform(valid)
        ganadores,control=bs.buscar_rss_minimo(X,np.log(fit.price_rub),grupos,range(1,13))
        for k,ganador in ganadores.items():
            with np.errstate(over='raise',invalid='raise'):
                pred=np.exp(ganador['intercepto']+V[ganador['columnas']].to_numpy()@ganador['coef'])
            errores[k-1,i]=float(np.sqrt(np.mean((valid.price_rub.to_numpy()-pred)**2)))
        assert preparador.n_fit==len(fit) and pd.Timestamp(preparador.fecha_hasta)<valid.date_posted_dt.min()
        auditoria.append({**control,'preprocessor_fit_rows':len(fit),'preprocessor_fit_until':preparador.fecha_hasta})
        del X,V,ganadores,preparador
    filas,regla=elegir_1se(errores)
    # Solo C(13,k elegido) combinaciones en todo Train; no repetir la búsqueda por todos los tamaños.
    preparador=PreparadorTemporal(grupos).fit(datos)
    X=preparador.transform(datos)
    ganadores,control=bs.buscar_rss_minimo(X,np.log(datos.price_rub),grupos,[regla['k_selected']])
    elegido=ganadores[regla['k_selected']]
    modelo,diseno=b.ajustar(X[elegido['columnas']],np.log(datos.price_rub))
    assert modelo.model.rank==diseno.shape[1]
    np.testing.assert_allclose(modelo.params.iloc[1:],elegido['coef'],rtol=1e-7,atol=1e-7)
    return dict(rows=filas,rule=regla,folds=folds,checks=auditoria,final_check=control,
        selected_features=elegido['seleccionadas'],columns=elegido['columnas'],
        model=modelo,preprocessor=preparador)


def cargar_contexto(base,cliente,*,cache_universo=None):
    """Solo el esquema y la procedencia del BSS previo; no descarga ni reajusta modelos históricos."""
    from tempfile import TemporaryDirectory
    run=s.recuperar_experimento(cliente,nombre=bs.NOMBRE_RUN,
        parametros={'algorithm_version':bs.VERSION,'source_universe_predictors':177})
    if cache_universo is None:
        with TemporaryDirectory() as tmp:
            archivo=cliente.download_artifacts(run['run_id'],'universo_bss.json',tmp)
            universo=json.loads(Path(archivo).read_text(encoding='utf-8'))
    else: universo=cache_universo
    assert universo['dataset_sha256']==CONTRATO['train']['sha256']
    assert run['tags']['dataset_sha256']==universo['dataset_sha256']
    assert json.loads(run['params']['candidate_features'])==universo['union']
    assert json.loads(run['params']['candidate_columns'])==universo['columnas']
    assert len(universo['union'])==13 and len(universo['columnas'])==146
    datos,identidad=cargar_train_crudo(base)
    configuracion=dict(version=VERSION,algorithm=CONFIG,identity=identidad,source_bss_run_id=run['run_id'],
        groups=universo['grupos'],folds=folds_expansivos(datos),limitations=LIMITACIONES,
        preprocessing='prefix-only imputation; frozen OHE schema; prefix month origin; no global Tukey/scaler')
    from importlib.metadata import version
    configuracion['versions']={k:version(k) for k in ('numpy','pandas','scipy','statsmodels','scikit-learn','joblib')}
    configuracion['code_sha256']={k:b.sha256(Path(__file__).with_name(k))
        for k in ('bss_temporal.py','bss_ols.py','preparacion_train_test.py')}
    return datos,configuracion


def preparar_local(datos,configuracion,salida):
    salida=b.validar_temporal(salida)
    salida.mkdir(exist_ok=True)
    r=ejecutar_cv(datos,configuracion['groups'])
    r['model'].save(salida/'modelo_statsmodels.pkl',remove_data=True)
    joblib.dump(r['preprocessor'],salida/'preparador.joblib')
    resultado={k:v for k,v in r.items() if k not in ('model','preprocessor')}
    resultado['model_sha256']=b.sha256(salida/'modelo_statsmodels.pkl')
    resultado['preprocessor_sha256']=b.sha256(salida/'preparador.joblib')
    b.guardar_json(salida/'configuracion.json',configuracion)
    b.guardar_json(salida/'resultados.json',resultado)
    b.guardar_json(salida/'inferencia.json',dict(input='Raw rows with date_posted_dt; apply preparador.transform, select columns, add const, exp(model.predict).',
        columns=resultado['columns'],target='ln(price_rub)',inverse='exp',bias_correction='none'))
    for nombre in ('bss_temporal.py','bss_ols.py','preparacion_train_test.py'):
        (salida/nombre).write_bytes(Path(__file__).with_name(nombre).read_bytes())
    return resultado


def parametros(configuracion):
    return dict(algorithm_version=VERSION,n_folds=3,max_features=12,selection_rule='1-SE',
        source_bss_run_id=configuracion['source_bss_run_id'],train_sha256=configuracion['identity']['train_sha256'],
        n_train=configuracion['identity']['n_train'],n_candidate_groups=13,n_candidate_columns=146,
        preprocessing=configuracion['preprocessing'],target_transform='ln(price_rub)',inverse_transform='exp',bias_correction='none')


def registrar(cliente,configuracion,salida,*,nombre_run=NOMBRE_RUN):
    from mlflow.entities import Param,Metric
    salida=Path(salida)
    exp=cliente.get_experiment_by_name(b.EXPERIMENTO)
    if exp is None or exp.lifecycle_stage!='active': raise ValueError('Debe existir el experimento activo.')
    if "'" in nombre_run or not nombre_run.strip(): raise ValueError('Nombre de Run inválido.')
    if cliente.search_runs([exp.experiment_id],filter_string=f"tags.mlflow.runName = '{nombre_run}'",max_results=1):
        raise ValueError('Run existente; recuperar o utilizar un nombre nuevo.')
    r=json.loads((salida/'resultados.json').read_text(encoding='utf-8'))
    assert json.loads((salida/'configuracion.json').read_text(encoding='utf-8'))==configuracion
    validar_resultados(r,configuracion)
    assert b.sha256(salida/'modelo_statsmodels.pkl')==r['model_sha256']
    assert b.sha256(salida/'preparador.joblib')==r['preprocessor_sha256']
    run=cliente.create_run(exp.experiment_id,tags={'mlflow.runName':nombre_run,'test_utilizado':'false',
        'train_sha256':configuracion['identity']['train_sha256']})
    rid=run.info.run_id
    try:
        vals={**r['rule'],'n_predictors':len(r['columns'])}
        for row in r['rows']:
            vals.update({f'k{row["k"]}_{k}':v for k,v in row.items() if k!='k'})
        cliente.log_batch(rid,params=[Param(k,str(v)) for k,v in parametros(configuracion).items()],
            metrics=[Metric(k,float(v),int(time.time()*1000),0) for k,v in vals.items()])
        cliente.log_artifacts(rid,str(salida))
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            for archivo in salida.iterdir():
                remoto=cliente.download_artifacts(rid,archivo.name,tmp)
                assert b.sha256(remoto)==b.sha256(archivo),archivo.name
        cliente.set_terminated(rid,'FINISHED')
    except BaseException:
        cliente.set_terminated(rid,'FAILED')
        raise
    return dict(run_id=rid,experiment_id=exp.experiment_id,run_name=nombre_run,status='FINISHED')


def validar_resultados(r,cfg):
    errores=np.array([[row[k] for k in ('fold_1','fold_2','fold_3')] for row in r['rows']])
    filas,regla=elegir_1se(errores)
    assert filas==r['rows'] and regla==r['rule']
    assert r['folds']==cfg['folds'] and len(r['checks'])==3
    assert len(r['selected_features'])==regla['k_selected']
    assert r['columns']==s.columnas_grupos(cfg['groups'],r['selected_features'])
    import math
    for fold,check in zip(r['folds'],r['checks']):
        assert pd.Timestamp(fold['fit_until'])<pd.Timestamp(fold['val_from'])
        assert check['preprocessor_fit_rows']==fold['fit_stop']
        assert check['preprocessor_fit_until']==fold['fit_until']
        assert check['evaluados']==8190
    assert r['final_check']['evaluados']==math.comb(13,regla['k_selected'])


def ejecutar_presentacion(base,sesion,cliente,*,modo='recuperar',nombre_run=NOMBRE_RUN):
    from statsmodels.regression.linear_model import OLSResults
    if modo not in ('nuevo','recuperar'): raise ValueError('Elegir recuperar/nuevo.')
    datos,cfg=cargar_contexto(base,cliente)
    out=sesion.carpeta('bss_temporal')
    if modo=='nuevo':
        exp=cliente.get_experiment_by_name(b.EXPERIMENTO)
        if exp is None or exp.lifecycle_stage!='active': raise ValueError('Experimento inexistente.')
        if cliente.search_runs([exp.experiment_id],filter_string=f"tags.mlflow.runName = '{nombre_run}'",max_results=1):
            raise ValueError('Run existente; no repetir el procesamiento.')
        preparar_local(datos,cfg,out)
        rid=registrar(cliente,cfg,out,nombre_run=nombre_run)['run_id']
    else: rid=None
    run=s.recuperar_experimento(cliente,nombre=nombre_run,run_id=rid,parametros=parametros(cfg))
    for nombre in ('configuracion.json','resultados.json','modelo_statsmodels.pkl','preparador.joblib'):
        cliente.download_artifacts(run['run_id'],nombre,str(out))
    assert json.loads((out/'configuracion.json').read_text(encoding='utf-8'))==cfg
    r=json.loads((out/'resultados.json').read_text(encoding='utf-8'))
    validar_resultados(r,cfg)
    assert b.sha256(out/'modelo_statsmodels.pkl')==r['model_sha256']
    assert b.sha256(out/'preparador.joblib')==r['preprocessor_sha256']
    modelo=OLSResults.load(out/'modelo_statsmodels.pkl')
    assert modelo.nobs==len(datos) and modelo.model.rank==len(r['columns'])+1
    assert modelo.model.exog_names==['const']+r['columns']
    assert run['metrics']['k_selected']==r['rule']['k_selected']
    assert run['metrics']['selected_mean']==r['rule']['selected_mean']
    return dict(resultados=r,configuracion=cfg,run=run,modo=modo)


def tabla_presentacion(r):
    resultados=r['resultados']
    filas=[]
    for row in resultados['rows']:
        elegido=row['k']==resultados['rule']['k_selected']
        valores=[str(row['k'])]+[f'{row[k]:,.0f}' for k in ('fold_1','fold_2','fold_3','mean','se')]
        if elegido: valores=[f'**{v}**' for v in valores]
        filas.append(dict(zip(['Características','RMSE Fold 1','RMSE Fold 2','RMSE Fold 3','RMSE promedio','Error estándar'],valores)))
    return b.tabla_markdown(pd.DataFrame(filas))


def modelo_presentacion(r):
    res=r['resultados']; regla=res['rule']
    return (f"Menor RMSE promedio: **{regla['k_min']} características**. Regla 1-SE: **{regla['k_selected']} características**, "
        f"RMSE temporal promedio **{regla['selected_mean']:,.0f} RUB**. OLS definitivo: "
        f"{', '.join('`'+g+'`' for g in res['selected_features'])}; **{len(res['columns'])} columnas transformadas**, más intercepto.")


def detalles_presentacion(r):
    run=r['run']
    url=f"{b.TRACKING_URI}/#/experiments/{run['experiment_id']}/runs/{run['run_id']}"
    cfg=r['configuracion']; res=r['resultados']
    datos=json.dumps(dict(configuracion=cfg,comprobaciones=res['checks'],comprobacion_final=res['final_check'],regla=res['rule']),ensure_ascii=False,indent=2)
    return ('<details><summary>Detalles técnicos y limitaciones (cerrado por defecto)</summary>\n\n'
        +LIMITACIONES+f"\n\nMLflow: [{run['run_name']}]({url}) · **{run['status']}** · {r['modo']}. "
        f"Train: {cfg['identity']['n_train']:,} publicaciones.\n\n```json\n{datos}\n```\n\n</details>")
