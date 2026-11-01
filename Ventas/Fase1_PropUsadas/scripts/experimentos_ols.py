"""Procedimientos OLS existentes, centralizados; importar no entrena ni registra Runs."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def crear_preprocesador(*, numericas, categoricas, escalar=False):
    categoricas = list(categoricas)
    numericas = list(numericas)
    numerico = Pipeline([
        ("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("escala", StandardScaler() if escalar else "passthrough"),
    ])
    categorico = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                 sparse_output=False, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("numericas", numerico, numericas),
        ("categoricas", categorico, categoricas),
    ], remainder="drop", verbose_feature_names_out=True).set_output(transform="pandas")

def evaluar_regresion(y_real, pred_rub, *, es_train=False,
                      n_predictoras_efectivas=None, ajuste_verosimilitud=None):
    real = np.asarray(y_real, dtype=float)
    pred = np.asarray(pred_rub, dtype=float)
    assert real.ndim == pred.ndim == 1 and real.shape == pred.shape and len(real) > 1
    assert np.isfinite(real).all() and np.isfinite(pred).all()
    assert (real > 0).all(), "MAPE requiere precios reales estrictamente positivos."
    n = len(real)
    errores = real - pred
    rss = float(errores @ errores)
    tss = float(np.square(real - real.mean()).sum())
    assert tss > 0, "R² no está definido para un objetivo constante."
    r2 = 1 - rss / tss
    resultado = {
        "r2": r2, "r2_ajustado": None, "rss_rub2": rss, "aic_rub": None, "bic_rub": None,
        "mae_rub": float(np.abs(errores).mean()), "rmse_rub": float(np.sqrt(rss / n)),
        "mape_pct": float((np.abs(errores) / real).mean() * 100),
    }
    if es_train and n_predictoras_efectivas is not None:
        p = int(n_predictoras_efectivas)
        assert p == n_predictoras_efectivas and p >= 0 and n > p + 1
        resultado["r2_ajustado"] = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    if ajuste_verosimilitud is not None:
        assert es_train, "AIC y BIC se reservan para el modelo ajustado sobre Train."
        assert ajuste_verosimilitud["n_train"] == n
        ll = float(ajuste_verosimilitud["loglike_rub"])
        k = int(ajuste_verosimilitud["k_likelihood"])
        assert np.isfinite(ll) and k > 0
        resultado["aic_rub"] = -2 * ll + 2 * k
        resultado["bic_rub"] = -2 * ll + np.log(n) * k
    assert all(np.isfinite(v) for v in resultado.values() if v is not None)
    return resultado

etiquetas_metricas = {
    "r2": "R²", "r2_ajustado": "R² ajustado Train", "rss_rub2": "RSS (RUB²)",
    "aic_rub": "AIC Train (y=RUB)", "bic_rub": "BIC Train (y=RUB)",
    "mae_rub": "MAE (RUB)", "rmse_rub": "RMSE (RUB)", "mape_pct": "MAPE (%)",
}

def tabla_metricas(resultados):
    return pd.DataFrame.from_dict(resultados, orient="index").rename(columns=etiquetas_metricas)

def ajustar_ols(train, validation, *, predictoras, numericas, categoricas,
                transformacion="log", escalar=False):
    if transformacion not in {"original", "log"}:
        raise ValueError("Objetivo no admitido.")
    if not train['dataset_split'].eq('train').all() or train['date_posted_dt'].max() >= validation['date_posted_dt'].min():
        raise ValueError("Requiere entrenamiento temporal anterior a la evaluación.")
    if not validation['dataset_split'].isin(['train','validation']).all():
        raise ValueError("Test permanece reservado.")
    if set(predictoras) != set(numericas) | set(categoricas):
        raise ValueError("Especificar el tratamiento de todas y solo las predictoras.")
    prepro=crear_preprocesador(numericas=numericas,categoricas=categoricas,escalar=escalar)
    matrices={'train':prepro.fit_transform(train[predictoras])}
    matrices['validation']=prepro.transform(validation[predictoras])
    disenos={s:sm.add_constant(m.astype(float),has_constant='add') for s,m in matrices.items()}
    objetivo=np.log(train['price_rub']) if transformacion=='log' else train['price_rub']
    modelo=sm.OLS(objetivo,disenos['train'],missing='raise').fit()
    rango=int(modelo.model.rank)
    duan=float(np.exp(modelo.resid).mean()) if transformacion=='log' else 1.0
    ajuste={'loglike_rub':float(modelo.llf-(np.log(train['price_rub']).sum() if transformacion=='log' else 0)),
            'k_likelihood':rango+1,'n_train':len(train)}
    metricas={}
    for s,tabla in [('train',train),('validation',validation)]:
        pred=modelo.predict(disenos[s])
        pred=np.exp(pred)*duan if transformacion=='log' else pred
        metricas[s]=evaluar_regresion(tabla['price_rub'],pred,es_train=s=='train',
            n_predictoras_efectivas=rango-1 if s=='train' else None,
            ajuste_verosimilitud=ajuste if s=='train' else None)
    diagnosticos_log=None
    if transformacion=='log':
        diagnosticos_log={'r2_ajustado_log':float(modelo.rsquared_adj),'rss_log':float(modelo.ssr),
            'aic_log':float(-2*modelo.llf+2*(rango+1)),
            'bic_log':float(-2*modelo.llf+np.log(len(train))*(rango+1))}
    return {'modelo':modelo,'preprocesador':prepro,'metricas':metricas,'rango':rango,'duan':duan,
        'ajuste':ajuste,'diagnosticos_log':diagnosticos_log,'transformacion':transformacion,
        'caracteristicas':list(predictoras),'n_columnas':matrices['train'].shape[1],
        'ejemplo_diseno':disenos['validation'].iloc[:5].copy()}


def validacion_temporal(datos_base, *, predictoras, numericas, categoricas,
        preparar_fold=None, n_folds=5, meses_validacion=6):
    from sklearn.model_selection import TimeSeriesSplit
    if 'district_group' in datos_base:
        raise ValueError("No reutilizar el mapa de Train completo en CV; usar la base y preparar_fold.")
    train=datos_base.loc[datos_base['dataset_split'].eq('train')]
    meses=np.sort(train['year_month'].unique())
    filas=[]
    for fold,(i_fit,i_valid) in enumerate(TimeSeriesSplit(n_splits=n_folds,test_size=meses_validacion).split(meses),1):
        fit=train.loc[train['year_month'].isin(meses[i_fit])]
        valid=train.loc[train['year_month'].isin(meses[i_valid])]
        if preparar_fold is not None:
            preparados=preparar_fold(fold)
            for nombre,esperado in [('train',fit),('validation',valid)]:
                pd.testing.assert_series_equal(preparados[nombre]['id'],esperado['id'])
            if preparados['trazabilidad']['fold'] != fold:
                raise ValueError("Mapa de otro fold.")
            fit,valid=preparados['train'],preparados['validation']
            if preparados['predictoras'] != list(predictoras):
                raise ValueError("Predictoras incompatibles con el fold.")
        resultado=ajustar_ols(fit,valid,predictoras=predictoras,numericas=numericas,categoricas=categoricas)
        filas.append({'fold':fold,'n_train':len(fit),'n_valid':len(valid),
            'train_desde':fit['date_posted_dt'].min(),'train_hasta':fit['date_posted_dt'].max(),
            'valid_desde':valid['date_posted_dt'].min(),'valid_hasta':valid['date_posted_dt'].max(),
            **{f'{s}_{m}':v for s,valores in resultado['metricas'].items() for m,v in valores.items()}})
    tabla=pd.DataFrame(filas)
    return tabla,{'cv_mae_mean':float(tabla['validation_mae_rub'].mean()),
                  'cv_mae_std':float(tabla['validation_mae_rub'].std(ddof=1))}


def estimar_efectos_distrito(train, *, predictoras, numericas, categoricas):
    prepro=crear_preprocesador(numericas=numericas,categoricas=categoricas)
    diseno=sm.add_constant(prepro.fit_transform(train[predictoras]).astype(float),has_constant='add')
    modelo=sm.OLS(np.log(train['price_rub']),diseno,missing='raise').fit()
    if int(modelo.model.rank) != diseno.shape[1]:
        raise ValueError("Efectos no identificables.")
    encoder=prepro.named_transformers_['categoricas'].named_steps['onehot']
    posicion=list(categoricas).index('district')
    categorias=encoder.categories_[posicion]
    referencia=categorias[encoder.drop_idx_[posicion]]
    efectos=pd.DataFrame({'district':categorias})
    efectos['referencia']=efectos['district'].eq(referencia)
    efectos['n_anuncios']=efectos['district'].map(train['district'].value_counts()).astype(int)
    efectos['efecto_log_ajustado']=[0.0 if d==referencia else float(modelo.params[f'categoricas__district_{d}']) for d in categorias]
    efectos['error_estandar']=[np.nan if d==referencia else float(modelo.bse[f'categoricas__district_{d}']) for d in categorias]
    efectos['soporte_escaso']=efectos['n_anuncios'].lt(30)
    return efectos


def agrupar_efectos(efectos, *, algoritmo, ks=tuple(range(3,11))):
    if algoritmo not in {'KMeans','Ward'}:
        raise ValueError("Agrupamiento no admitido.")
    resultados={}
    tabla=efectos.sort_values('district',kind='stable').reset_index(drop=True) if algoritmo=='Ward' else efectos.copy()
    valores=tabla[['efecto_log_ajustado']].to_numpy(dtype=float)
    if algoritmo=='Ward':
        from scipy.cluster.hierarchy import linkage,cut_tree
        jerarquia=linkage(valores,method='ward',metric='euclidean',optimal_ordering=True)
        cortes=cut_tree(jerarquia,n_clusters=list(ks))
    for i,k in enumerate(ks):
        if algoritmo=='KMeans':
            from sklearn.cluster import KMeans
            estimador=KMeans(n_clusters=k,random_state=42,n_init=50,init='k-means++',algorithm='lloyd',max_iter=300,tol=1e-4)
            etiquetas=estimador.fit_predict(valores)
            centros=pd.Series(estimador.cluster_centers_.ravel())
            orden=np.argsort(centros.to_numpy())
        else:
            etiquetas=cortes[:,i]
            centros=tabla.assign(_cluster=etiquetas).groupby('_cluster')['efecto_log_ajustado'].mean()
            orden=centros.sort_values(kind='stable').index.to_numpy()
        nombres={int(c):f'G{j+1:02d}' for j,c in enumerate(orden)}
        asignacion=tabla.copy()
        asignacion['district_group']=[nombres[int(e)] for e in etiquetas]
        asignacion['centroide_log']=[float(centros.loc[e]) for e in etiquetas]
        soporte=asignacion.groupby('district_group')['n_anuncios'].sum().sort_index()
        resultados[k]={'asignacion':asignacion,'fallback':soporte.idxmax()}
        resultados[k]['kmeans' if algoritmo=='KMeans' else 'linkage']=estimador if algoritmo=='KMeans' else jerarquia
    return resultados


def registrar_experimento(nombre, resultado, *, cliente, contexto, parametros, nota, cv=None, artefactos=()):
    import mlflow
    import mlflow.statsmodels
    exp=cliente.get_experiment_by_name('Ventas_Propiedades_Usadas')
    if exp is None or exp.lifecycle_stage!='active' or mlflow.active_run() is not None:
        raise ValueError("Usar el experimento existente, sin anidar ni reutilizar Runs.")
    if cliente.tracking_uri.rstrip('/') != mlflow.get_tracking_uri().rstrip('/'):
        raise ValueError("Conexiones MLflow distintas.")
    metricas={f'{s}_{m}':v for s,valores in resultado['metricas'].items() for m,v in valores.items() if v is not None}
    if cv is not None:
        metricas.update({'cv_mae_mean':float(cv['validation_mae_rub'].mean()),'cv_mae_std':float(cv['validation_mae_rub'].std(ddof=1))})
    with mlflow.start_run(experiment_id=exp.experiment_id,run_name=nombre) as run:
        mlflow.set_tags({'mlflow.note.content':nota,'dataset_sha256':contexto['dataset_sha256'],'test_evaluado':'false'})
        mlflow.log_params({**parametros,'n_columnas_predictoras':resultado['n_columnas'],
            'n_caracteristicas_originales':len(resultado['caracteristicas']),'n_predictoras_efectivas':resultado['rango']-1,
            'duan_factor_train':resultado['duan'],'transformacion_objetivo':resultado['transformacion']})
        mlflow.log_metrics(metricas)
        mlflow.log_dict(contexto,'preparacion/contexto.json')
        mlflow.log_dict({'predictoras':resultado['caracteristicas'],
            'columnas_transformadas':resultado['preprocesador'].get_feature_names_out().tolist()},'preparacion/caracteristicas.json')
        mlflow.log_dict({'transformacion_objetivo':resultado['transformacion'],'intercepto':True,
            'prediccion_rub':'exp(prediccion_nativa) * duan_factor_train' if resultado['transformacion']=='log' else 'prediccion_nativa',
            'duan_factor_train':resultado['duan'],'columnas_diseno':resultado['ejemplo_diseno'].columns.tolist()},'preparacion/inferencia.json')
        with TemporaryDirectory(prefix='ols_registro_') as temporal:
            carpeta=Path(temporal)
            joblib.dump(resultado['preprocesador'],carpeta/'preprocesador.joblib')
            mlflow.log_artifact(str(carpeta/'preprocesador.joblib'),artifact_path='preparacion')
            modelo=copy.deepcopy(resultado['modelo'])
            modelo.remove_data()
            mlflow.statsmodels.save_model(modelo,path=str(carpeta/'modelo'),
                pip_requirements=[f'{p}=={version(p)}' for p in ['statsmodels','numpy','pandas','scipy','patsy']])
            recargado=mlflow.statsmodels.load_model(str(carpeta/'modelo'))
            np.testing.assert_allclose(recargado.predict(resultado['ejemplo_diseno']),
                resultado['modelo'].predict(resultado['ejemplo_diseno']),rtol=1e-12)
            mlflow.log_artifacts(str(carpeta/'modelo'),artifact_path='modelo')
            if cv is not None:
                (carpeta/'cv.json').write_text(cv.to_json(orient='records',date_format='iso'),encoding='utf-8')
                mlflow.log_artifact(str(carpeta/'cv.json'),artifact_path='resultados')
            # Confirmar el contenido remoto antes de liberar la serialización temporal.
            import hashlib
            huella = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            verificaciones = [('preparacion/preprocesador.joblib', carpeta/'preprocesador.joblib')]
            verificaciones += [('modelo/'+p.relative_to(carpeta/'modelo').as_posix(), p)
                               for p in (carpeta/'modelo').rglob('*') if p.is_file()]
            if cv is not None:
                verificaciones.append(('resultados/cv.json', carpeta/'cv.json'))
            with TemporaryDirectory(prefix='ols_verificacion_') as descarga:
                for nombre, local in verificaciones:
                    recibido = cliente.download_artifacts(run.info.run_id, nombre, descarga)
                    assert huella(recibido) == huella(local), nombre
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta),artifact_path='resultados')
            with TemporaryDirectory(prefix='ols_artefacto_') as descarga:
                recibido = cliente.download_artifacts(run.info.run_id, 'resultados/'+Path(ruta).name, descarga)
                assert huella(recibido) == huella(ruta)
        return run.info.run_id


def recuperar_runs(cliente, referencias, dataset_sha256):
    exp=cliente.get_experiment_by_name('Ventas_Propiedades_Usadas')
    if exp is None or exp.lifecycle_stage!='active':
        raise ValueError("El experimento existente no está disponible.")
    def leer(item):
        nombre,run_id=item
        run=cliente.get_run(run_id)
        if run.info.experiment_id!=exp.experiment_id or run.info.status!='FINISHED':
            raise ValueError(f'Run no disponible: {nombre}')
        if run.data.tags.get('dataset_sha256')!=dataset_sha256:
            raise ValueError(f'Dataset distinto: {nombre}')
        return nombre,run
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(leer,referencias.items()))


def tabla_markdown(tabla, *, indice=False, decimales=4):
    def formato(v):
        if pd.isna(v):
            return '—'
        if isinstance(v,(float,np.floating)):
            return f'{v:,.{decimales}f}'
        return str(v).replace('|',r'\|').replace('\n',' ')
    datos=tabla.reset_index() if indice else tabla
    lineas=['| '+' | '.join(map(str,datos.columns))+' |','|'+'---|'*len(datos.columns)]
    lineas.extend('| '+' | '.join(formato(v) for v in fila)+' |' for fila in datos.itertuples(index=False,name=None))
    return '\n'.join(lineas)
