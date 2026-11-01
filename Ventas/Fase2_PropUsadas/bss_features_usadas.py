"""BSS exhaustivo por características completas; sin usar test."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def limpiar(datos, registro):
    d = datos.copy(deep=True)
    d['date_posted_dt'] = pd.to_datetime(d.date_posted, errors='raise')
    for c in registro['columnas_numericas_entrada']:
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if not (np.isfinite(d.price_rub) & d.price_rub.gt(0)).all():
        raise ValueError('Objetivos inválidos: no se imputan ni se eliminan automáticamente.')
    areas = d[['total_area', 'living_area', 'kitchen_area']]
    m = (np.isfinite(areas) & areas.gt(0)).all(axis=1) & d.total_area.lt(d.living_area+d.kitchen_area)
    d.loc[m, 'total_area'] = d.loc[m, 'living_area']+d.loc[m, 'kitchen_area']
    t, f, y = d.total_floors, d.floor, d.building_year
    tv = np.isfinite(t) & t.gt(0) & t.mod(1).eq(0)
    fv = np.isfinite(f) & f.gt(0) & f.mod(1).eq(0)
    d.loc[t.notna() & ~tv, 'total_floors'] = np.nan
    d.loc[f.notna() & (~fv | (tv & f.gt(t))), 'floor'] = np.nan
    d.loc[y.notna() & (~np.isfinite(y) | y.mod(1).ne(0) | y.lt(1000) | y.gt(d.date_posted_dt.dt.year)), 'building_year'] = np.nan
    return d


def binaria(c, s):
    v = s.dropna()
    if c == 'district': return False
    if pd.api.types.is_bool_dtype(s) or c == 'has_balcony': return True
    if pd.api.types.is_numeric_dtype(s): return len(v)>0 and v.isin([0, 1]).all()
    return v.nunique()==2


def entrada_codificador(datos, parametros):
    d = limpiar(datos, parametros['registro'])
    for c, v in parametros['imputacion'].items(): d[c] = d[c].fillna(v)
    assert not d.isna().any().any()
    assert (d.floor.gt(0) & d.total_floors.gt(0) & d.floor.le(d.total_floors)).all()
    assert d[['floor', 'total_floors', 'building_year']].mod(1).eq(0).all().all()
    assert (d.building_year.ge(1000) & d.building_year.le(d.date_posted_dt.dt.year)).all()
    d = d.drop(columns=parametros['registro']['eliminadas']).rename(columns=parametros['registro']['alias'])
    X = d.drop(columns=['id', 'price_rub', 'date_posted_dt']).copy()
    X['anio_publicacion'], X['mes_publicacion'] = d.date_posted_dt.dt.year, d.date_posted_dt.dt.month
    for c, pares in parametros.get('binarias', {}).items():
        X[c] = X[c].map(dict(pares)).fillna(0).astype(float)
    return X, d.price_rub


def preparar_interno(ajuste_crudo, validacion_cruda, registro):
    """Fit solo en ajuste_crudo; validación recibe transform, nunca fit."""
    d = limpiar(ajuste_crudo, registro)
    valores = {}
    for c in d:
        if c in ['id', 'price_rub', 'date_posted', 'date_posted_dt']: continue
        s = d[c]
        if binaria(c, s):
            moda = s.mode(dropna=True)
            v = moda.iloc[0] if len(moda) else np.nan
        elif pd.api.types.is_numeric_dtype(s):
            v = s.mean() if c == 'secondary_mom_change_pct' else s.median()
        else: v = '__FALTANTE__'
        if pd.isna(v): raise ValueError(f'No se puede imputar {c} usando entrenamiento interno.')
        valores[c] = v.item() if isinstance(v, np.generic) else v
    parametros = {'registro': registro, 'imputacion': valores}
    X, y = entrada_codificador(ajuste_crudo, parametros)
    binarias = [c for c in X if binaria(c, X[c])]
    nominales = [c for c in X if c not in binarias and not pd.api.types.is_numeric_dtype(X[c])]
    numericas = [c for c in X if c not in binarias+nominales]
    mapas = {}
    for c in binarias:
        v = sorted(X[c].unique().tolist())
        mapas[c] = [(x, int(x)) for x in v] if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]) else list(zip(v, range(len(v))))
    parametros['binarias'] = mapas
    X, y = entrada_codificador(ajuste_crudo, parametros)
    V, yv = entrada_codificador(validacion_cruda, parametros)
    bloques = [('num', 'passthrough', numericas), ('bin', 'passthrough', binarias)]
    for i,c in enumerate(nominales):
        bloques.append((f'cat{i}', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False,dtype=float),[c]))
    encoder = ColumnTransformer(bloques,verbose_feature_names_out=False)
    A = encoder.fit_transform(X)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Found unknown categories',category=UserWarning)
        B = encoder.transform(V)
    nombres = encoder.get_feature_names_out().tolist()
    mapa = {c:[c] for c in numericas+binarias if c!='to_center_km'}
    for i,c in enumerate(nominales): mapa[c] = encoder.named_transformers_[f'cat{i}'].get_feature_names_out([c]).tolist()
    parametros['encoder'], parametros['mapa'] = encoder, mapa
    parametros['columnas'] = nombres
    parametros['categorias'] = {c:encoder.named_transformers_[f'cat{i}'].categories_[0].tolist() for i,c in enumerate(nominales)}
    parametros['referencias'] = {c:v[0] for c,v in parametros['categorias'].items()}
    return pd.DataFrame(A,index=X.index,columns=nombres),pd.DataFrame(B,index=V.index,columns=nombres),y,yv,parametros


def inversa(pred, lam):
    pred=np.asarray(pred,dtype=float)
    valida=np.isfinite(pred)
    if lam: valida &= 1+lam*pred>0
    out=np.full(pred.shape,np.nan)
    with np.errstate(over='ignore',invalid='ignore'): out[valida]=inv_boxcox(pred[valida],lam)
    valida &= np.isfinite(out) & (out>0)
    return out,valida


def errores(real,pred):
    real,pred=np.asarray(real),np.asarray(pred)
    mse=np.mean((real-pred)**2)
    return {'R2':float(1-np.sum((real-pred)**2)/np.sum((real-real.mean())**2)),
            'MAE':float(np.mean(np.abs(real-pred))),'MSE':float(mse),'RMSE':float(np.sqrt(mse))}


def base_estimable(X,mapa):
    columnas=[c for cols in mapa.values() for c in cols]
    constantes=[c for c in columnas if X[c].nunique()<=1]
    candidatas=[c for c in columnas if c not in constantes]
    Z=(X[candidatas]-X[candidatas].mean())/X[candidatas].std(ddof=0)
    _,R,piv=qr(Z.to_numpy(),mode='economic',pivoting=True,check_finite=False)
    tol=np.finfo(float).eps*max(Z.shape)*np.abs(np.diag(R)).max()
    rango=int((np.abs(np.diag(R))>tol).sum())
    conservar=[candidatas[i] for i in sorted(piv[:rango].tolist())]
    bloques={f:[c for c in conservar if c in cols] for f,cols in mapa.items()}
    return bloques,{'constantes':constantes,'dependencias':[c for c in candidatas if c not in conservar],'tolerancia':float(tol)}


VERSION_BUSQUEDA = 'exhaustivo-features-gvif-v3'
REGLA_COLINEALIDAD = {'vif_numericas_binarias_lt': 5.0, 'gvif_ajustado_categoricas_lt': float(np.sqrt(5)),
    'ajuste': 'GVIF^(1/(2*df))', 'poda': 'Solo VIF de numéricas/binarias; sin poda GVIF',
    'fundamento_umbral': 'Heurística por dimensión; no umbral universal de GVIF'}


def tipos_features(bloques, tipos=None):
    """La declaración explícita permite nominales de un solo parámetro estimable."""
    nominales = {'district', 'building_type', 'renovation', 'metro_line'}
    binarias = {'has_balcony', 'metro_distance_type', 'seller_type'}
    salida = {}
    for f, cs in bloques.items():
        tipo = (tipos or {}).get(f, 'Nominal' if f in nominales or len(cs)>1 else 'Binaria' if f in binarias else 'Numérica')
        if tipo not in {'Nominal', 'Ordinal', 'Binaria', 'Numérica'}:
            raise ValueError(f'Tipo no reconocido para {f}: {tipo}')
        if tipo in {'Numérica','Binaria'} and len(cs)>1:
            raise ValueError(f'{f}: varios parámetros requieren declarar un bloque categórico.')
        salida[f] = tipo
    return salida


def _logdet_spd(matriz):
    """Log-determinante de una matriz SPD; singularidad explícita, sin pseudoinversa."""
    L = np.linalg.cholesky(matriz)
    return float(2*np.log(np.diag(L)).sum())


def _medidas_bloques(K, grupos, tipos, logdet_C):
    """Núcleo único para búsqueda y diagnóstico posterior.

    C es la correlación de predictores centrados y K=C^-1. Por la identidad
    de Jacobi, GVIF(g)=det(C_g)*det(K_g), equivalente al cociente de
    determinantes en la correlación de coeficientes de OLS con intercepto.
    logdet_C contiene los log-determinantes de los bloques fijos de C.
    """
    diagonal = np.diag(K)
    valores, ajustados = {}, {}
    for f, ids in grupos.items():
        df = len(ids)
        if tipos[f] in {'Nominal','Ordinal'}:
            loggvif = logdet_C[f]+_logdet_spd(K[np.ix_(ids,ids)])
            ajustados[f] = float(np.exp(loggvif/(2*df)))
            valores[f] = ajustados[f]
        else:
            valores[f] = float(diagonal[ids[0]])
    cumple = {f: bool(np.isfinite(v) and v < (np.sqrt(5) if tipos[f] in {'Nominal','Ordinal'} else 5))
              for f,v in valores.items()}
    return valores, ajustados, cumple


def vif_features(modelo, bloques, tipos=None):
    """Una fila por característica; acepta OLS ajustado o DataFrame de diseño.

    Los VIF de dummies son secundarios. No ordena indicadores de escalas
    distintas. Singularidad, constancia y valores no finitos producen una
    indicación explícita de no calculable, nunca una sustitución arbitraria.
    """
    if hasattr(modelo, 'model'):
        X = pd.DataFrame(modelo.model.exog, columns=modelo.model.exog_names)
    else:
        X = pd.DataFrame(modelo).copy()
    X = X.drop(columns=['const','Intercept'], errors='ignore')
    bloques = {f:[c for c in cs if c in X] for f,cs in bloques.items()}
    bloques = {f:cs for f,cs in bloques.items() if cs}
    columnas = [c for cs in bloques.values() for c in cs]
    if len(columnas)!=len(set(columnas)) or set(columnas)!=set(X):
        raise ValueError('El mapa debe cubrir todas las columnas estimables una sola vez.')
    tipos = tipos_features(bloques, tipos)
    grupos = {f:np.array([X.columns.get_loc(c) for c in cs],dtype=int) for f,cs in bloques.items()}
    error = None
    try:
        A = X.to_numpy(dtype=float)
        if not np.isfinite(A).all() or (X.std(ddof=0)<=0).any():
            raise ValueError('Valores no finitos o columnas constantes.')
        Z=(A-A.mean(axis=0))/A.std(axis=0)
        if np.linalg.matrix_rank(Z)!=X.shape[1]:
            raise ValueError('Diseño singular o no estimable.')
        C=np.atleast_2d(Z.T@Z/len(Z))
        from scipy.linalg import cho_solve
        K=cho_solve((np.linalg.cholesky(C),True),np.eye(len(C)))
        logC={f:_logdet_spd(C[np.ix_(ids,ids)]) for f,ids in grupos.items()}
        valores, ajustados, cumple=_medidas_bloques(K,grupos,tipos,logC)
        diagonal=np.diag(K)
    except (ValueError,np.linalg.LinAlgError) as exc:
        error=str(exc);valores={f:np.nan for f in grupos};ajustados={};cumple={}
        diagonal=np.full(X.shape[1],np.nan)
    filas=[]
    for f,ids in grupos.items():
        categorica=tipos[f] in {'Nominal','Ordinal'}
        gv=ajustados.get(f,np.nan)
        filas.append({'Característica':f,'Tipo':tipos[f],'Parámetros':len(ids),
            'Indicador principal':'GVIF ajustado' if categorica else 'VIF', 'Valor':valores[f],
            'Cumple criterio':'No calculable' if error else 'Sí' if cumple[f] else 'No',
            'VIF máximo':float(diagonal[ids].max()),'Columnas VIF >5':int((diagonal[ids]>5).sum()) if not error else np.nan,
            'GVIF':float(np.exp(2*len(ids)*np.log(gv))) if categorica and np.isfinite(gv) else valores[f] if not categorica else np.nan,
            'df':len(ids),'GVIF ajustado':gv,'Umbral':np.sqrt(5) if categorica else 5,
            'Estado cálculo':error or 'Calculable'})
    tabla=pd.DataFrame(filas,columns=['Característica','Tipo','Parámetros','Indicador principal','Valor','Cumple criterio',
        'VIF máximo','Columnas VIF >5','GVIF','df','GVIF ajustado','Umbral','Estado cálculo'])
    tabla.attrs['vif_columnas']=dict(zip(X.columns,diagonal))
    return tabla


def firma_busqueda(X, V, y, yv, mapa, nombre, tolerancia_rmse):
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps({'version':VERSION_BUSQUEDA, 'segmento':nombre, 'mapa':mapa,
        'columnas':X.columns.tolist(), 'tolerancia':tolerancia_rmse, 'colinealidad':REGLA_COLINEALIDAD,
        'tipos':tipos_features(mapa)}, sort_keys=True).encode())
    for objeto in [X, V, y, yv]:
        h.update(pd.util.hash_pandas_object(objeto,index=True).to_numpy().tobytes())
    return h.hexdigest()


def _ranking_parcial(C, d, yty, n, bloques_indices):
    K = np.linalg.inv(C)
    beta = K @ d
    rss = yty - n * d @ beta
    filas = []
    for feature, ids in bloques_indices.items():
        extra = n * beta[ids] @ np.linalg.solve(K[np.ix_(ids,ids)], beta[ids])
        filas.append({'Característica':feature, 'R2 parcial subtrain':extra/(rss+extra), 'Parámetros':len(ids)})
    return pd.DataFrame(filas).sort_values(['R2 parcial subtrain','Característica'],ascending=[False,True],kind='stable')


def buscar_segmento(X,V,y,yv,mapa,nombre,tolerancia_rmse=.01,lote=128,progreso=None):
    """BSS completo con VIF/GVIF por tipo; poda solo VIF numérico/binario.

    Una base completa de columnas independientes implica independencia de
    cualquier subconjunto. Con filas y codificaciones fijas, añadir columnas
    no reduce los VIF existentes de términos numéricos/binarios. No se poda
    por GVIF ni por VIF de dummies. Se conserva el mínimo RMSE por pareja de
    complejidad (features, parámetros); el resto está dominado para la regla
    de parsimonia. No hay cribado por ranking ni máximo de features.
    """
    import time
    from scipy.linalg import cho_solve
    from threadpoolctl import threadpool_limits
    bloques,estructura=base_estimable(X,mapa)
    candidatas=[f for f,cols in bloques.items() if cols]
    columnas=[c for f in candidatas for c in bloques[f]]
    if not columnas: raise ValueError('No hay características estimables.')
    ybox,lam=boxcox(np.asarray(y,dtype=float))
    n=len(y);nv=len(yv);P=len(columnas);m=len(candidatas)
    mu=X[columnas].mean();sd=X[columnas].std(ddof=0)
    Z=np.ascontiguousarray(((X[columnas]-mu)/sd).to_numpy())
    Zv=np.ascontiguousarray(((V[columnas]-mu)/sd).to_numpy())
    C=Z.T@Z/n;d=Z.T@(ybox-ybox.mean())/n
    yty=float(np.sum((ybox-ybox.mean())**2));media_y=float(ybox.mean())
    np.linalg.cholesky(C)  # base completa independiente; submatrices principales SPD
    indices={f:np.asarray([columnas.index(c) for c in bloques[f]],dtype=int) for f in candidatas}
    tipos=tipos_features(bloques)
    logC={f:_logdet_spd(C[np.ix_(ids,ids)]) for f,ids in indices.items()}
    ranking=_ranking_parcial(C,d,yty,n,indices)  # únicamente diagnóstico
    # Bloques grandes primero: se reutilizan sus inversas en descendientes.
    recorrido=sorted(candidatas,key=lambda f:(-len(indices[f]),f))
    posicion={f:i for i,f in enumerate(candidatas)}
    total=1<<m
    estad={'subconjuntos_total':total,'subconjuntos_considerados':0,'evaluados_explicitamente':0,
        'podados_vif':0,'rechazados_vif_explicitos':0,'modelos_cumplen_vif':0,
        'modelos_inversa_valida':0,'modelos_inversa_invalida':0,'disenos_validos':total,
        'fallback_numerico':0,'rechazados_gvif_explicitos':0,
        'poda':REGLA_COLINEALIDAD['poda']}
    frontera={};pendientes=[];coeficientes=np.zeros((P,lote))
    inicio=time.monotonic();ultimo_aviso=inicio
    real=np.asarray(yv,dtype=float);sst=float(np.sum((real-real.mean())**2))

    def vaciar():
        if not pendientes:return
        q=len(pendientes)
        pred=media_y+Zv@coeficientes[:,:q]
        rub,validos=inversa(pred,lam)
        completas=validos.all(axis=0)
        estad['modelos_inversa_valida']+=int(completas.sum())
        estad['modelos_inversa_invalida']+=int((~completas).sum())
        for j in np.flatnonzero(completas):
            mask,nfeatures,k,rss,vmax=pendientes[j]
            sse=float(np.sum((real-rub[:,j])**2));rmse=float(np.sqrt(sse/nv))
            clave=(nfeatures,k)
            anterior=frontera.get(clave)
            if anterior is None or (rmse,mask)<(anterior['RMSE validación RUB'],anterior['máscara']):
                ll=-n/2*(np.log(2*np.pi)+1+np.log(rss/n))
                fs=[f for f in candidatas if mask & (1<<posicion[f])]
                frontera[clave]={'máscara':mask,'características':json.dumps(fs,ensure_ascii=False),
                    'n_features':nfeatures,'n_parámetros':k,'rango_válido':True,
                    'R2 ajustado':1-(rss/(n-k))/(yty/(n-1)),'AIC':-2*ll+2*k,'BIC':-2*ll+k*np.log(n),
                    'RMSE validación RUB':rmse,'R2 validación RUB':1-sse/sst,'VIF máximo':vmax,'inversas inválidas':0}
        pendientes.clear()

    def visitar(start,mascara,ids,K,nfeatures,grupos):
        nonlocal ultimo_aviso
        estad['evaluados_explicitamente']+=1
        vmax=float(np.diag(K).max()) if len(ids) else np.nan
        numericos=[ix[0] for f,ix in grupos.items() if tipos[f] in {'Numérica','Binaria'}]
        vmax_num=float(np.diag(K)[numericos].max()) if numericos else 0.0
        if vmax_num>=5:
            estad['rechazados_vif_explicitos']+=1
            # Banda conservadora: cerca del umbral no se poda descendencia.
            if vmax_num>5+1e-8:
                estad['podados_vif']+=(1<<(m-start))-1
                return
        else:
            try:
                valores,ajustados,cumple=_medidas_bloques(K,grupos,tipos,logC)
            except np.linalg.LinAlgError:
                # Recalcular la inversa; no descartar silenciosamente por redondeo.
                K=np.linalg.inv(C[np.ix_(ids,ids)])
                valores,ajustados,cumple=_medidas_bloques(K,grupos,tipos,logC)
            admisible=all(cumple.values())
            if not admisible:estad['rechazados_gvif_explicitos']+=1
        if vmax_num<5 and admisible:
            estad['modelos_cumplen_vif']+=1
            beta=K@d[ids] if len(ids) else np.empty(0)
            rss=yty-n*float(d[ids]@beta) if len(ids) else yty
            if rss<=0 or not np.isfinite(rss):
                raise FloatingPointError('RSS no positivo; búsqueda no declarada completa.')
            j=len(pendientes);coeficientes[:,j]=0
            if len(ids):coeficientes[ids,j]=beta
            pendientes.append((mascara,nfeatures,len(ids)+1,rss,vmax))
            if len(pendientes)==lote:vaciar()
        if estad['evaluados_explicitamente']%2048==0 and progreso and time.monotonic()-ultimo_aviso>20:
            progreso({'segmento':nombre,'explicitos':estad['evaluados_explicitamente'],
                'podados':estad['podados_vif'],'total':total,'segundos':time.monotonic()-inicio})
            ultimo_aviso=time.monotonic()
        for j in range(start,m):
            f=recorrido[j];nuevos=indices[f]
            if len(ids):
                B=C[np.ix_(ids,nuevos)];U=K@B
                S=C[np.ix_(nuevos,nuevos)]-B.T@U
            else:S=C[np.ix_(nuevos,nuevos)]
            try:
                L=np.linalg.cholesky(S)
                D=cho_solve((L,True),np.eye(len(nuevos)),check_finite=False)
                if len(ids):
                    UD=U@D
                    nuevoK=np.block([[K+UD@U.T,-UD],[-UD.T,D]])
                else:nuevoK=D
            except np.linalg.LinAlgError:
                estad['fallback_numerico']+=1
                todos=np.concatenate([ids,nuevos])
                # No se descarta un candidato por un fallo de actualización.
                nuevoK=np.linalg.inv(C[np.ix_(todos,todos)])
            visitar(j+1,mascara|(1<<posicion[f]),np.concatenate([ids,nuevos]),nuevoK,nfeatures+1,
                {**grupos,f:np.arange(len(ids),len(ids)+len(nuevos))})

    with threadpool_limits(limits=1,user_api='blas'):
        visitar(0,0,np.empty(0,dtype=int),np.empty((0,0)),0,{})
        vaciar()
    estad['subconjuntos_considerados']=estad['evaluados_explicitamente']+estad['podados_vif']
    assert estad['subconjuntos_considerados']==total
    assert estad['modelos_cumplen_vif']==estad['modelos_inversa_valida']+estad['modelos_inversa_invalida']
    estad['segundos']=time.monotonic()-inicio
    resultados=pd.DataFrame(frontera.values())
    if resultados.empty:raise ValueError('Sin modelos admisibles; no existe una búsqueda completa utilizable.')
    minimo=float(resultados['RMSE validación RUB'].min())
    equivalentes=resultados.loc[resultados['RMSE validación RUB'].le(minimo*(1+tolerancia_rmse))]
    ganador=equivalentes.sort_values(['n_features','n_parámetros','RMSE validación RUB','máscara'],kind='stable').iloc[0]
    seleccionadas=json.loads(ganador['características'])
    elegidas=[c for f in seleccionadas for c in bloques[f]]
    inner=sm.OLS(pd.Series(ybox,index=X.index,name='boxcox_price_rub'),sm.add_constant(X[elegidas],has_constant='add')).fit()
    pv=inner.predict(sm.add_constant(V[elegidas],has_constant='add'))
    rub,ok=inversa(pv,lam);assert ok.all()
    diagnostico=vif_features(inner,{f:bloques[f] for f in seleccionadas})
    assert diagnostico['Cumple criterio'].eq('Sí').all()
    np.testing.assert_allclose(errores(yv,rub)['RMSE'],ganador['RMSE validación RUB'],rtol=1e-6)
    return {'nombre':nombre,'candidatas_totales':candidatas,'candidatas_evaluadas':candidatas,'ranking':ranking,
        'bloques':bloques,'estructura':estructura,'resultados':resultados,'seleccionadas':seleccionadas,
        'lambda_interno':float(lam),'modelo_interno':inner,'ganador':ganador.to_dict(),'columnas_internas':elegidas,
        'n_subtrain':len(y),'n_validacion':len(yv),'minimo_rmse_factible':minimo,'estadisticas':estad,
        'version':VERSION_BUSQUEDA,'firma':firma_busqueda(X,V,y,yv,mapa,nombre,tolerancia_rmse),'completa':True}


def cache_busqueda_valida(cache, contexto):
    if cache is None:return False
    config=cache.get('configuracion',{})
    if config.get('version_busqueda')!=VERSION_BUSQUEDA or config.get('firma_contexto')!=contexto:return False
    for s,seleccion in cache.get('seleccion',{}).items():
        evidencia=cache.get('busqueda',{}).get(s,{})
        if not evidencia.get('completa') or evidencia.get('version')!=VERSION_BUSQUEDA:return False
        if seleccion.get('candidatas_totales')!=seleccion.get('candidatas_evaluadas'):return False
        if evidencia.get('estadisticas',{}).get('subconjuntos_considerados')!=2**len(seleccion['candidatas_totales']):return False
    return len(cache.get('seleccion',{}))==4


def recuperar_busqueda(cache, X, V, y, yv, segmento, tolerancia_rmse=.01):
    config=cache['seleccion'][segmento];evidencia=cache['busqueda'][segmento]
    firma=firma_busqueda(X,V,y,yv,cache['preparador_validacion']['mapa'],segmento,tolerancia_rmse)
    if evidencia['firma']!=firma or not evidencia['completa']:raise ValueError('Caché BSS incompatible: datos o candidatas cambiaron.')
    modelo=cache['modelos_validacion'][segmento]
    bloques,estructura=base_estimable(X,cache['preparador_validacion']['mapa'])
    candidatas=[f for f,cs in bloques.items() if cs]
    if candidatas!=config['candidatas_totales']:raise ValueError('Caché inválida por cambio de características.')
    columnas=modelo.model.exog_names[1:]
    A=sm.add_constant(X[columnas],has_constant='add')
    np.testing.assert_array_equal(A.to_numpy(),modelo.model.exog)
    np.testing.assert_allclose(boxcox(np.asarray(y,dtype=float),lmbda=config['lambda_interno']),modelo.model.endog,rtol=1e-13,atol=1e-13)
    assert columnas==[c for f in config['features'] for c in bloques[f]]
    # Ranking diagnóstico regenerado con productos de subtrain, sin filtrar.
    cols=[c for f in candidatas for c in bloques[f]]
    Z=((X[cols]-X[cols].mean())/X[cols].std(ddof=0)).to_numpy()
    ybox=np.asarray(modelo.model.endog)
    # endog del modelo recuperado contiene todas las filas del segmento.
    C=Z.T@Z/len(X);d=Z.T@(ybox-ybox.mean())/len(X)
    ids={f:np.array([cols.index(c) for c in bloques[f]]) for f in candidatas}
    ranking=_ranking_parcial(C,d,float(np.sum((ybox-ybox.mean())**2)),len(X),ids)
    return {'nombre':segmento,'candidatas_totales':candidatas,'candidatas_evaluadas':candidatas,
        'seleccionadas':config['features'],'lambda_interno':config['lambda_interno'],'modelo_interno':modelo,
        'columnas_internas':columnas,'bloques':bloques,'estructura':estructura,'resultados':None,'ranking':ranking,
        'ganador':evidencia['ganador'],'minimo_rmse_factible':evidencia['minimo_rmse_factible'],
        'estadisticas':evidencia['estadisticas'],'version':VERSION_BUSQUEDA,'firma':firma,'completa':True,
        'n_subtrain':len(y),'n_validacion':len(yv)}


def importancia_permutacion(busqueda,V,yv,semilla=20261028,repeticiones=30):
    modelo=busqueda['modelo_interno'];cols=busqueda['columnas_internas'];lam=busqueda['lambda_interno']
    A=sm.add_constant(V[cols],has_constant='add')[modelo.model.exog_names].to_numpy()
    beta=np.asarray(modelo.params);base,ok=inversa(A@beta,lam);assert ok.all()
    rmse=errores(yv,base)['RMSE'];filas=[]
    for i,f in enumerate(busqueda['seleccionadas']):
        ind=[modelo.model.exog_names.index(c) for c in busqueda['bloques'][f]]
        for repeticion in range(repeticiones):
            rng=np.random.default_rng(np.random.SeedSequence([semilla,i,repeticion]))
            orden=rng.permutation(len(A));B=A.copy();B[:,ind]=A[orden][:,ind]
            pred,valida=inversa(B@beta,lam)
            delta=errores(yv,pred)['RMSE']-rmse if valida.all() else np.nan
            filas.append({'Característica':f,'Repetición':repeticion+1,'ΔRMSE RUB':delta,'Inversas inválidas':int((~valida).sum())})
    detalle=pd.DataFrame(filas)
    resumen=detalle.groupby('Característica',sort=False).agg(
        media=('ΔRMSE RUB','mean'),mediana=('ΔRMSE RUB','median'),desvio=('ΔRMSE RUB','std'),repeticiones_validas=('ΔRMSE RUB','count')).sort_values('mediana',ascending=False)
    resumen['Posición']=range(1,len(resumen)+1)
    return detalle,resumen.reset_index(),rmse


