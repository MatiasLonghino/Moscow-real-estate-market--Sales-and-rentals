"""Representaciones en memoria y contratos de reutilización; no ajusta modelos."""

import copy
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

EXPERIMENTO = "Ventas_Propiedades_Usadas"
ARTEFACTO_MAPAS = "transformaciones/agrupamientos_v1.json"


def huella_json(valor):
    return hashlib.sha256(json.dumps(valor, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def huella_tabla(tabla):
    esquema = [(str(c), str(t)) for c, t in tabla.dtypes.items()]
    filas = pd.util.hash_pandas_object(tabla, index=False).to_numpy().tobytes()
    return hashlib.sha256(huella_json(esquema).encode() + filas).hexdigest()


def contexto_entrenamiento(tabla, *, dataset_sha256, caracteristicas, numericas,
                           categoricas, preprocesamiento, fold=None, periodo_validacion=None):
    if tabla.empty or not tabla['dataset_split'].eq('train').all():
        raise ValueError('El aprendizaje de transformaciones requiere exclusivamente Train.')
    if not tabla['id'].is_unique or not tabla['date_posted_dt'].is_monotonic_increasing:
        raise ValueError('Identificadores duplicados o entrenamiento sin orden temporal.')
    if fold is not None and (int(fold) < 1 or periodo_validacion is None):
        raise ValueError('Un fold debe identificar su ventana de validación.')
    if fold is None and periodo_validacion is not None:
        raise ValueError('No asociar una ventana de fold a Train completo.')
    if 'district' not in caracteristicas or 'price_rub' in caracteristicas:
        raise ValueError('El OLS preliminar debe incluir district y excluir el objetivo.')
    contexto = {
        'dataset_sha256': dataset_sha256, 'entrenamiento_sha256': huella_tabla(tabla),
        'particion': 'train', 'fold': None if fold is None else int(fold),
        'n_train': len(tabla), 'desde': str(tabla['date_posted_dt'].min().date()),
        'hasta': str(tabla['date_posted_dt'].max().date()),
        'periodo_validacion': periodo_validacion,
        'caracteristicas_efectos': list(caracteristicas),
        'ols_preliminar': {'modelo': 'statsmodels.OLS', 'objetivo': 'log(price_rub)',
                           'intercepto': True, 'missing': 'raise',
                           'efecto_referencia': 0.0, 'peso_distrito': 'igual'},
        'preprocesamiento': copy.deepcopy(preprocesamiento),
        'numericas': list(numericas), 'categoricas': list(categoricas),
        'versiones': {p: version(p) for p in ['pandas', 'numpy', 'scikit-learn', 'statsmodels', 'scipy']},
    }
    if periodo_validacion is not None and contexto['hasta'] >= periodo_validacion['desde']:
        raise ValueError('El entrenamiento debe terminar antes de la validación del fold.')
    return contexto


def validar_contexto(guardado, esperado):
    if guardado != esperado:
        cambios = [k for k in set(guardado) | set(esperado) if guardado.get(k) != esperado.get(k)]
        raise ValueError(f'Agrupamiento incompatible ({sorted(cambios)}). Recalcular con este entrenamiento.')


def crear_artefacto(algoritmo, k, run_id, mapas, entrenamientos, contextos):
    if algoritmo not in {'KMeans', 'Ward'} or int(k) not in range(3, 11):
        raise ValueError('Algoritmo o K no admitidos.')
    if set(mapas) != set(contextos) or set(mapas) != set(entrenamientos):
        raise ValueError('Cada contexto necesita mapa y entrenamiento.')
    agrupamientos = {}
    for clave, mapa in mapas.items():
        tabla, contexto = entrenamientos[clave], contextos[clave]
        if clave != ('train_completo' if contexto['fold'] is None else f"fold_{contexto['fold']}"):
            raise ValueError('El identificador no corresponde al contexto.')
        if huella_tabla(tabla) != contexto['entrenamiento_sha256']:
            raise ValueError('El entrenamiento no coincide con su huella.')
        mapa = {str(d): str(g) for d, g in dict(mapa).items()}
        if set(mapa) != set(tabla['district'].astype(str).unique()) or len(set(mapa.values())) != k:
            raise ValueError('El mapa debe cubrir todos los distritos y exactamente K grupos.')
        soporte = tabla['district'].astype(str).map(mapa).value_counts().sort_index()
        fallback = soporte.idxmax()
        agrupamientos[clave] = {'contexto': copy.deepcopy(contexto), 'mapa': mapa,
            'fallback': fallback, 'regla_desconocidos': 'mayor_soporte_train_empate_etiqueta_ascendente',
            'id_transformacion': f'{run_id}:{clave}', 'mapa_sha256': huella_json(mapa)}
    return {'schema_version': 1, 'algoritmo': algoritmo, 'K': int(k), 'run_id': run_id,
            'experimento': EXPERIMENTO, 'agrupamientos': agrupamientos}


def leer_json_remoto(cliente, run_id, ruta, cache=None):
    clave = (run_id, ruta)
    if cache is not None and clave in cache:
        return copy.deepcopy(cache[clave])
    with TemporaryDirectory(prefix='transformacion_') as carpeta:
        archivo = cliente.download_artifacts(run_id, ruta, carpeta)
        contenido = json.loads(Path(archivo).read_text(encoding='utf-8'))
    if cache is not None:
        cache[clave] = copy.deepcopy(contenido)
    return contenido


def recuperar_agrupamiento(cliente, *, algoritmo, k, run_id, contexto_esperado, fold=None, cache=None):
    experimento = cliente.get_experiment_by_name(EXPERIMENTO)
    run = cliente.get_run(run_id)
    if experimento is None or experimento.lifecycle_stage != 'active' or run.info.experiment_id != experimento.experiment_id:
        raise ValueError('El Run debe pertenecer al experimento existente Ventas_Propiedades_Usadas.')
    if run.info.status != 'FINISHED':
        raise ValueError('El Run de origen no está finalizado.')
    artefacto = leer_json_remoto(cliente, run_id, ARTEFACTO_MAPAS, cache)
    if (artefacto['schema_version'], artefacto['algoritmo'], artefacto['K'], artefacto['run_id']) != (1, algoritmo, k, run_id):
        raise ValueError('Identidad del agrupamiento incorrecta.')
    clave = 'train_completo' if fold is None else f'fold_{int(fold)}'
    if clave not in artefacto['agrupamientos'] or contexto_esperado['fold'] != fold:
        raise ValueError('No reutilizar Train completo ni otro fold en este contexto.')
    agrupamiento = artefacto['agrupamientos'][clave]
    validar_contexto(agrupamiento['contexto'], contexto_esperado)
    if huella_json(agrupamiento['mapa']) != agrupamiento['mapa_sha256']:
        raise ValueError('Mapa alterado.')
    if len(set(agrupamiento['mapa'].values())) != k or agrupamiento['fallback'] not in agrupamiento['mapa'].values():
        raise ValueError('Cantidad de grupos o fallback incorrectos.')
    return {**agrupamiento, 'algoritmo': algoritmo, 'K': k, 'run_id': run_id, 'reutilizada': True}


def validar_uso(tabla, contexto_esperado, uso):
    if uso == 'train':
        if huella_tabla(tabla) != contexto_esperado['entrenamiento_sha256']:
            raise ValueError('Usar exactamente el entrenamiento del agrupamiento.')
    elif uso == 'validation':
        split = 'validation' if contexto_esperado['fold'] is None else 'train'
        if tabla.empty or not tabla['dataset_split'].eq(split).all():
            raise ValueError('Partición de evaluación incompatible; Test permanece reservado.')
        fechas = tabla['date_posted_dt']
        ventana = contexto_esperado['periodo_validacion']
        if fechas.min() <= pd.Timestamp(contexto_esperado['hasta']):
            raise ValueError('La evaluación debe ser posterior al entrenamiento.')
        if ventana and not fechas.between(pd.Timestamp(ventana['desde']), pd.Timestamp(ventana['hasta'])).all():
            raise ValueError('La evaluación está fuera del fold correspondiente.')
    else:
        raise ValueError('uso debe ser train o validation.')


def aplicar_agrupamiento(tabla, agrupamiento, *, contexto_esperado, uso):
    validar_contexto(agrupamiento['contexto'], contexto_esperado)
    validar_uso(tabla, contexto_esperado, uso)
    salida = tabla.copy(deep=True)
    grupos = salida['district'].astype(str).map(agrupamiento['mapa'])
    salida['district_group'] = grupos.fillna(agrupamiento['fallback'])
    caracteristicas = ['district_group' if c == 'district' else c
                       for c in contexto_esperado['caracteristicas_efectos']]
    return salida, caracteristicas, int(grupos.isna().sum())


def generar_representacion(tabla, configuracion, *, cliente, contexto_esperado, uso, cache=None):
    """Lista explícita de pasos: independiente por defecto, combinada solo al indicarlo."""
    validar_uso(tabla, contexto_esperado, uso)
    trabajo = tabla.copy(deep=True)
    caracteristicas = list(contexto_esperado['caracteristicas_efectos'])
    procedencia = []
    agrupado = False
    for paso in configuracion.get('pasos', []):
        if paso['tipo'] == 'agrupamiento':
            if agrupado or procedencia:
                raise ValueError('Aplicar un único agrupamiento antes de seleccionar o extraer.')
            transformacion = recuperar_agrupamiento(cliente, algoritmo=paso['algoritmo'], k=paso['K'],
                run_id=paso['run_id'], fold=paso.get('fold'), contexto_esperado=contexto_esperado, cache=cache)
            trabajo, caracteristicas, desconocidos = aplicar_agrupamiento(trabajo, transformacion,
                contexto_esperado=contexto_esperado, uso=uso)
            procedencia.append({'tipo': 'agrupamiento', 'algoritmo': paso['algoritmo'], 'K': paso['K'],
                'run_id_origen': paso['run_id'], 'id_transformacion': transformacion['id_transformacion'],
                'reutilizada': True, 'desconocidos': desconocidos})
            agrupado = True
        elif paso['tipo'] == 'seleccion':
            # El algoritmo de selección se ajustará en su propia etapa, exclusivamente con Train.
            if paso['metodo'] not in {'forward', 'backward', 'exclusion_explicita'}:
                raise ValueError('Método de selección no admitido.')
            if paso['contexto_sha256'] != huella_json(contexto_esperado):
                raise ValueError('La selección fue ajustada con otro entrenamiento o fold.')
            if paso['columnas_entrada'] != caracteristicas or paso['origen_transformaciones'] != procedencia_ids(procedencia):
                raise ValueError('La selección pertenece a otra representación de entrada.')
            elegidas = list(paso['caracteristicas'])
            if not elegidas or len(set(elegidas)) != len(elegidas) or not set(elegidas) <= set(caracteristicas):
                raise ValueError('Selección inválida o incluye variables auxiliares/objetivo.')
            if not paso.get('criterio'):
                raise ValueError('Registrar el criterio de selección.')
            caracteristicas = elegidas
            procedencia.append(copy.deepcopy(paso))
        else:
            raise ValueError('Transformación no implementada: agregar un adaptador que valide el contexto de ajuste.')
    return {'datos': trabajo, 'X': trabajo[caracteristicas].copy(),
            'caracteristicas': caracteristicas, 'procedencia': procedencia,
            'configuracion': copy.deepcopy(configuracion), 'contexto': copy.deepcopy(contexto_esperado)}


def procedencia_ids(procedencia):
    return [p['id_transformacion'] for p in procedencia if 'id_transformacion' in p]


def registrar_procedencia(cliente, run_id, representacion, *, resumen, estado='reutilizada'):
    experimento = cliente.get_experiment_by_name(EXPERIMENTO)
    run = cliente.get_run(run_id)
    if experimento is None or run.info.experiment_id != experimento.experiment_id:
        raise ValueError('No registrar fuera del experimento existente.')
    if estado not in {'reutilizada', 'calculada'}:
        raise ValueError('Estado de transformación inválido.')
    manifiesto = {k: representacion[k] for k in ['configuracion', 'contexto', 'caracteristicas', 'procedencia']}
    cliente.log_dict(run_id, manifiesto, 'transformaciones/representacion.json')
    fuentes = sorted({p['run_id_origen'] for p in representacion['procedencia'] if 'run_id_origen' in p})
    cliente.set_tag(run_id, 'transformaciones.runs_origen', json.dumps(fuentes))
    cliente.set_tag(run_id, 'transformaciones.estado', estado)
    nota = '\n\nGestión de transformaciones: ' + estado + '. ' + resumen
    anterior = run.data.tags.get('mlflow.note.content', '')
    if nota not in anterior:
        cliente.set_tag(run_id, 'mlflow.note.content', anterior + nota)


def preparar_df_seleccionado(tabla, agrupamiento, excluir, *, predictoras_base, uso):
    """Aplica reglas ya aprendidas; preparación de filas reservadas no implica evaluarlas."""
    contexto = agrupamiento['contexto']
    if list(predictoras_base) != contexto['caracteristicas_efectos']:
        raise ValueError('No cambiar las características del OLS que originó el mapa.')
    if len(set(excluir)) != len(excluir) or not set(excluir) <= set(predictoras_base):
        raise ValueError('Las exclusiones deben ser predictoras originales distintas.')
    if 'district' in excluir:
        raise ValueError('district se reemplaza por su grupo, no se elimina como regla adicional.')
    if uso == 'preparacion_reservada':
        if contexto['fold'] is not None:
            raise ValueError('La representación general requiere el mapa de Train completo.')
        if not tabla['dataset_split'].isin(['train','validation','test']).all():
            raise ValueError('Particiones desconocidas.')
        fit = tabla.loc[tabla['dataset_split'].eq('train')]
        if huella_tabla(fit) != contexto['entrenamiento_sha256']:
            raise ValueError('El DataFrame no contiene exactamente el Train del agrupamiento.')
        trabajo = tabla.copy(deep=True)
        grupos = trabajo['district'].astype(str).map(agrupamiento['mapa'])
        desconocidos = int(grupos.isna().sum())
        trabajo['district_group'] = grupos.fillna(agrupamiento['fallback'])
        columnas = ['district_group' if c == 'district' else c for c in predictoras_base]
    else:
        trabajo,columnas,desconocidos = aplicar_agrupamiento(tabla,agrupamiento,
            contexto_esperado=contexto,uso=uso)
    trabajo = trabajo.drop(columns=list(excluir))
    columnas = [c for c in columnas if c not in excluir]
    categorias = sorted(set(agrupamiento['mapa'].values()))
    trabajo['district_group'] = pd.Categorical(trabajo['district_group'],categories=categorias)
    if trabajo['district_group'].isna().any() or len(trabajo) != len(tabla):
        raise ValueError('Grupos nulos inesperados o cambio de cantidad de filas.')
    pd.testing.assert_series_equal(trabajo['price_rub'],tabla['price_rub'])
    pd.testing.assert_series_equal(trabajo['dataset_split'],tabla['dataset_split'])
    trabajo.attrs['transformacion'] = {
        'id':agrupamiento['id_transformacion'],'run_id':agrupamiento['run_id'],
        'algoritmo':agrupamiento['algoritmo'],'K':agrupamiento['K'],'fold':contexto['fold'],
        'excluidas':list(excluir),'predictoras':columnas,'solo_preparacion':True,
        'evaluada_conjuntamente':False,'desconocidos':desconocidos,
    }
    return trabajo,columnas


def preparar_fold_seleccionado(datos_base, fold, *, cliente, run_id, excluir,
        dataset_sha256, predictoras_base, numericas, categoricas, preprocesamiento,
        n_folds=5, meses_validacion=6, cache=None):
    from sklearn.model_selection import TimeSeriesSplit

    if 'district_group' in datos_base:
        raise ValueError('Para CV usar el DataFrame base, no el mapa de Train completo.')
    train = datos_base.loc[datos_base['dataset_split'].eq('train')]
    meses = sorted(train['year_month'].unique())
    cortes = list(TimeSeriesSplit(n_splits=n_folds,test_size=meses_validacion).split(meses))
    if fold not in range(1,n_folds+1):
        raise ValueError('Fold inválido.')
    i_fit,i_valid = cortes[fold-1]
    fit = train.loc[train['year_month'].isin([meses[i] for i in i_fit])]
    valid = train.loc[train['year_month'].isin([meses[i] for i in i_valid])]
    periodo = {'desde':str(valid['date_posted_dt'].min().date()),'hasta':str(valid['date_posted_dt'].max().date())}
    contexto = contexto_entrenamiento(fit,dataset_sha256=dataset_sha256,
        caracteristicas=predictoras_base,numericas=numericas,categoricas=categoricas,
        preprocesamiento=preprocesamiento,fold=fold,periodo_validacion=periodo)
    agrupamiento = recuperar_agrupamiento(cliente,algoritmo='KMeans',k=8,run_id=run_id,
        contexto_esperado=contexto,fold=fold,cache=cache)
    datos_fit,columnas = preparar_df_seleccionado(fit,agrupamiento,excluir,
        predictoras_base=predictoras_base,uso='train')
    datos_valid,columnas_valid = preparar_df_seleccionado(valid,agrupamiento,excluir,
        predictoras_base=predictoras_base,uso='validation')
    assert columnas == columnas_valid
    return {'train':datos_fit,'validation':datos_valid,'predictoras':columnas,
            'trazabilidad':copy.deepcopy(datos_fit.attrs['transformacion'])}
