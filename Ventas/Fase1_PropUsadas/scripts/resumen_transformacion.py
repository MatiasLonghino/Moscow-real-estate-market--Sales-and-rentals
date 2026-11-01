"""Auditoría documental de la entrega real; no ajusta ni transforma datos."""
import json
from pathlib import Path

import joblib
import pandas as pd

from . import preparacion_train_test as pt
def ejemplos(valores, n=3):
    valores = list(map(str, valores))
    return ", ".join(valores[:n]) + (", …" if len(valores) > n else "")


def grupos_onehot(encoder, entradas, nombres_salida):
    """Distribuye las salidas por categorías/drop reales, nunca por prefijos similares."""
    if encoder.min_frequency is not None or encoder.max_categories is not None:
        raise ValueError("Actualizar la auditoría para agrupación de categorías infrecuentes.")
    resultado, inicio = [], 0
    for i, variable in enumerate(entradas):
        categorias = encoder.categories_[i].tolist()
        omitida = None if encoder.drop_idx_ is None else encoder.drop_idx_[i]
        cantidad = len(categorias) - int(omitida is not None)
        columnas = list(nombres_salida[inicio:inicio + cantidad])
        resultado.append({"variable": variable, "columnas": columnas,
                          "categorias": categorias,
                          "referencia": None if omitida is None else str(categorias[int(omitida)]),
                          "desconocidas": encoder.handle_unknown})
        inicio += cantidad
    if inicio != len(nombres_salida):
        raise ValueError("Las categorías del codificador no explican todas sus salidas.")
    return resultado


def generar_resumen(carpeta, train_crudo, test_crudo, *, manifiesto, objetos):
    carpeta = Path(carpeta)
    hashes = {k:v for k,v in manifiesto["archivos_sha256"].items() if k.endswith(".csv")}
    for nombre, huella in hashes.items():
        if pt.huella_archivo(carpeta / nombre) != huella:
            raise ValueError(f"Entrega inconsistente: cambió {nombre}. Regenerar los tres archivos.")
    if pt.huella_archivo(Path(pt.__file__)) != manifiesto["modulo_sha256"]:
        raise ValueError("El módulo cambió desde la exportación; regenerar antes del resumen.")
    huella_objetos = joblib.hash(objetos)
    contrato = objetos["configuracion"]
    columnas = list(objetos["columnas_X"])
    if len(columnas) != len(set(columnas)) or any(c != pt.nombre_predictor(c) for c in columnas):
        raise ValueError("El esquema final no respeta la convención única de nombres.")
    if columnas != manifiesto["columnas_X"] or columnas != contrato["columnas_X"]:
        raise ValueError("El paquete, contrato y manifiesto discrepan sobre el orden de X.")
    datos = {}
    for conjunto in ["train", "test"]:
        frame = pd.read_csv(carpeta / f"{conjunto}_transformado.csv", dtype={"id": "string"},
                            parse_dates=["date_posted_dt"], float_precision="round_trip")
        pt.verificar_excluidas(frame)
        pt.validar_matriz(pt.obtener_X(frame), frame, columnas=columnas)
        if pt.OBJETIVO not in frame or frame[pt.OBJETIVO].isna().any():
            raise ValueError("Falta la respuesta exportada.")
        datos[conjunto] = frame
    train, test = datos["train"], datos["test"]
    if train.columns.tolist() != test.columns.tolist():
        raise ValueError("Los CSV difieren en columnas u orden.")
    for actual, original in [(train, train_crudo[train_crudo.id.isin(train.id)]), (test, test_crudo)]:
        pd.testing.assert_frame_equal(actual[[*pt.AUXILIARES, pt.OBJETIVO]].reset_index(drop=True),
                                      original[[*pt.AUXILIARES, pt.OBJETIVO]].reset_index(drop=True), check_dtype=False)

    preparacion, base = objetos["preparacion"], objetos["base"]
    extraccion, filtro, escala = objetos["extraccion"], objetos["filtro"], objetos["escalado"]
    # Un registro completo de procedencia exige que cada salida tenga un productor conocido.
    linaje, grupos = {}, []
    nombres_base = base.get_feature_names_out().tolist()
    for bloque, transformador, entradas in base.transformers_:
        if bloque == "remainder":
            if transformador != "drop":
                raise ValueError("Documentar las nuevas salidas remainder antes de continuar.")
            continue
        salidas = nombres_base[base.output_indices_[bloque]]
        if bloque == "cat":
            grupos.extend(grupos_onehot(transformador, entradas, salidas))
        else:
            if len(entradas) != len(salidas):
                raise ValueError("El bloque no categórico dejó de ser una correspondencia uno a uno.")
            for entrada, salida in zip(entradas, salidas):
                derivada = preparacion.derivadas_.get(entrada)
                linaje[salida] = {"variable": entrada, "entradas": [entrada] if derivada is None else derivada["entradas"],
                                  "tipo": "binaria" if bloque == "bin" else "numerica",
                                  "derivada": derivada}
                if derivada:
                    linaje[salida]["tipo"] = derivada["tipo"]
    distrito = objetos["district"].named_steps["onehot"]
    grupos.extend(grupos_onehot(distrito, distrito.feature_names_in_.tolist(), distrito.get_feature_names_out().tolist()))
    mes = extraccion.mes_
    grupos.extend(grupos_onehot(mes, mes.feature_names_in_.tolist(), mes.get_feature_names_out().tolist()))
    for grupo in grupos:
        fuentes = ["date_posted_dt"] if grupo["variable"] in mes.feature_names_in_ else [grupo["variable"]]
        for salida in grupo["columnas"]:
            if salida in linaje:
                raise ValueError("Dos productores declaran la misma salida.")
            linaje[salida] = {"variable": grupo["variable"], "entradas": fuentes, "tipo": "onehot", "derivada": None}
    for salida, derivada in extraccion.derivadas_.items():
        linaje[salida] = {"variable": ", ".join(derivada["entradas"]), "entradas": derivada["entradas"],
                          "tipo": derivada["tipo"], "derivada": derivada}
    normalizador = filtro.named_steps["nombres"]
    correspondencia = normalizador.correspondencia_
    if set(correspondencia) != set(linaje):
        raise ValueError("El normalizador no corresponde al linaje de las salidas candidatas.")
    linaje = {correspondencia[c]: info for c, info in linaje.items()}
    if any(info["tipo"] != normalizador.roles_salida_[c] for c, info in linaje.items()):
        raise ValueError("El linaje y los roles aprendidos discrepan.")
    for grupo in grupos:
        grupo["columnas"] = [correspondencia[c] for c in grupo["columnas"]]
    constantes = filtro.named_steps["constantes"]
    entradas_filtro = constantes.feature_names_in_.tolist()
    if set(entradas_filtro) != set(linaje):
        raise ValueError("Hay salidas sin procedencia documentada o procedencias inexistentes.")
    if filtro.get_feature_names_out().tolist() != columnas:
        raise ValueError("Las columnas del filtro final no coinciden con las exportadas.")
    if escala.get_feature_names_out().tolist() != columnas:
        raise ValueError("El escalador no corresponde a la estructura final.")
    finales = set(columnas)
    seleccion = []
    for variable in train_crudo.columns:
        if variable in [*pt.AUXILIARES, pt.OBJETIVO]:
            continue
        canonico = pt.ALIAS_HIPOTECARIO.get(variable, variable)
        if canonico in contrato["entradas_seleccionadas"]:
            reemplazos = [c for c in columnas if canonico in linaje[c]["entradas"]]
            seleccion.append({"Variable original": variable, "Acción": "Reemplazada por representación transformada" if reemplazos else "Sin representación final",
                              "Motivo / representación": ejemplos(reemplazos) or "Retirada por el filtro final"})
        else:
            if variable in train:
                raise ValueError(f"Selección no aplicada: {variable}.")
            motivo = pt.MOTIVOS_EXCLUSION.get(variable, "Fuera de la lista blanca de entradas del flujo; no se utiliza en etapas posteriores.")
            seleccion.append({"Variable original": variable, "Acción": "Excluida completamente de X", "Motivo / representación": motivo})
    duplicadas = filtro.named_steps["duplicadas"].duplicadas_
    for variable in entradas_filtro:
        if variable in finales:
            continue
        motivo = (f"Duplicada exacta en Train de {duplicadas[variable]}" if variable in duplicadas else
                  f"Varianza Train <= {constantes.threshold}")
        seleccion.append({"Variable original": variable, "Acción": "Salida intermedia eliminada", "Motivo / representación": motivo})

    transformaciones = []
    def agregar(variables, metodo, ajuste, regla, resultado):
        transformaciones.append({"Variables": variables, "Método": metodo, "Aprendizaje": ajuste,
                                 "Regla / parámetro": regla, "Resultado": resultado})
    for nombre in ["numerico_", "mensual_", "binario_", "categorico_"]:
        imputador = getattr(preparacion, nombre)
        for variable, valor in zip(imputador.feature_names_in_, imputador.statistics_):
            agregar(variable, f"Imputación {imputador.strategy}", "Configuración fija" if imputador.strategy == "constant" else contrato["momento_ajuste_imputacion"],
                    str(valor), "Sin faltantes en entradas preparadas; indicadores según imputador ajustado")
    agregar(", ".join(preparacion.numerico_.feature_names_in_), "Corrección de dominio", "Determinística; corregir_dominio",
            "No finitos → NaN; total/altura > 0; conteos/distancias/tasa >= 0; conteos enteros; piso <= pisos; año entero entre 1 y año de publicación",
            "Valores incoherentes → NaN antes de imputar; las superficies individuales observadas se conservan")
    agregar("total_area, living_area, kitchen_area", "Corrección de superficie", "Determinística; corregir_superficie_total",
            "total_area = max(total_area, living_area + kitchen_area), cuando son finitas; antes y después de imputar",
            f"{contrato['auditoria_train']['superficies_corregidas']} señales iniciales de corrección en Train antes de Tukey")
    tukey = contrato["tratamiento_objetivo"]
    if tukey["filas_finales"] != len(train) or tukey["filas_iniciales"] != len(train_crudo) or len(test) != len(test_crudo):
        raise ValueError("Los recuentos de Tukey o Test no coinciden con la entrega.")
    ids_esperados = train_crudo.loc[train_crudo[pt.OBJETIVO].between(
        tukey["limite_inferior"], tukey["limite_superior"], inclusive="both"), "id"].tolist()
    if train["id"].tolist() != ids_esperados:
        raise ValueError("Las observaciones finales no coinciden con la regla Tukey persistida.")
    agregar(pt.OBJETIVO, tukey["metodo"], "Cuartiles de Train preparado; no se aplica a Test",
            f"Q1={tukey['q1']}; Q3={tukey['q3']}; IQR={tukey['iqr']}; factor={tukey['factor_iqr']}; límites inclusivos [{tukey['limite_inferior']}, {tukey['limite_superior']}]",
            f"Train: {len(train_crudo)-len(train)} eliminadas; Test: 0 eliminadas")
    for variable, mapa in contrato["mapas_binarios"].items():
        agregar(variable, "Mapeo binario", "Mapa fijo; moda Train si falta / etiqueta futura desconocida",
                "NFKC, strip, minúsculas; " + json.dumps(mapa, ensure_ascii=False), "Una columna 0/1; sin One-Hot; etiqueta Train desconocida causa error")
    for grupo in grupos:
        retenidas = [c for c in grupo["columnas"] if c in finales]
        agregar(grupo["variable"], "One-Hot Encoding", "Categorías de Train retenido",
                f"{len(grupo['categorias'])} categorías; referencia={grupo['referencia']}; desconocidas={grupo['desconocidas']} (ignore → todo 0); ejemplos: {ejemplos(grupo['categorias'])}",
                f"{len(grupo['columnas'])} salidas del encoder; {len(retenidas)} columnas finales")
    agregar(", ".join(mes.feature_names_in_), "Mes calendario previo al One-Hot", "Determinística",
            "date_posted_dt.dt.month.astype(str)", "Entrada del codificador temporal; las dummies se contabilizan solo como One-Hot")
    agregar("Matriz codificada y derivadas", "Filtro final de constantes y duplicadas", "fit solo Train retenido",
            f"VarianceThreshold({constantes.threshold}); igualdad exacta de columnas; conservar primera aparición",
            f"{int((~constantes.get_support()).sum())} constantes retiradas; {len(duplicadas)} duplicadas retiradas")
    agregar("Todas las salidas candidatas", "Normalización centralizada de nombres", "Solo esquema; sin estadísticas",
            "snake_case ASCII, minúsculas, sin prefijos técnicos; colisiones y nombres reservados causan error",
            f"{len(correspondencia)} nombres candidatos; correspondencia completa persistida")
    registro = pt.registro_escalado(escala)
    for variable in registro["estandarizadas"]:
        agregar(variable, "StandardScaler", f"fit con {registro['filas_ajuste']} filas Train retenido",
                f"(x - {registro['media_train'][variable]}) / {registro['escala_standard_scaler'][variable]}; SD poblacional={registro['desviacion_train_ddof0'][variable]}", "Una columna; mismo orden")
    for variable in registro["normalizadas"]:
        agregar(variable, "MinMaxScaler", "fit solo Train retenido",
                f"mín={registro['minimo_train'][variable]}; máx={registro['maximo_train'][variable]}; intervalo={registro['intervalo_minmax']}; clip={registro['recortar_test']}", "Una columna; mismo orden")
    if not registro["normalizadas"]:
        agregar("—", "Normalización Min-Max", "No aplicada", "No hay variables asignadas al normalizador", "0 columnas normalizadas")
    sin_escala = [c for c in columnas if c not in registro["estandarizadas"] + registro["normalizadas"]]
    originales = [c for c in sin_escala if linaje[c]["tipo"] == "numerica"]
    agregar(", ".join(originales), "Conservación de escala numérica", "Sin fit adicional", "Passthrough después de las etapas anteriores", f"{len(originales)} numéricas; {len(sin_escala)} predictoras totales sin escalado")
    agregar(", ".join(preparacion.categorico_.feature_names_in_), "Normalización de texto", "Determinística", "Unicode NFKC; quitar espacios extremos; vacío → faltante", "Categorías consistentes antes de imputar/codificar")

    inventario, extracciones, cubiertas = [], [], []
    for grupo in grupos:
        salidas = [c for c in grupo["columnas"] if c in finales]
        if salidas:
            inventario.append({"Variable original": grupo["variable"], "Características finales": ejemplos(salidas),
                               "Cantidad": len(salidas), "Referencia omitida": grupo["referencia"] or "Sin drop",
                               "Tipo": "One-Hot"})
            cubiertas.extend(salidas)
    for columna in columnas:
        info = linaje[columna]
        if info["tipo"] != "onehot":
            inventario.append({"Variable original": ", ".join(info["entradas"]), "Características finales": columna,
                               "Cantidad": 1, "Referencia omitida": "—", "Tipo": info["tipo"]})
            cubiertas.append(columna)
        if info["derivada"]:
            d = info["derivada"]
            extracciones.append({"Característica": columna, "Entradas": ", ".join(d["entradas"]),
                                  "Fórmula": d["formula"], "Finalidad": d["finalidad"]})
    if len(cubiertas) != len(columnas) or set(cubiertas) != finales:
        raise ValueError("El inventario no cubre cada predictora exactamente una vez.")
    conteos = {tipo: sum(linaje[c]["tipo"] == tipo for c in columnas) for tipo in ["numerica", "binaria", "onehot"]}
    if sum(conteos.values()) != len(columnas):
        raise ValueError("Las clases de columnas no suman el total de X.")
    originales_representadas = sorted({v for c in columnas for v in linaje[c]["entradas"]})
    cantidades = {
        "Registros originales (Train + Test)": len(train_crudo) + len(test_crudo),
        "Train antes de Tukey": len(train_crudo), "Filas eliminadas por Tukey": len(train_crudo)-len(train),
        "Train final": len(train), "Test final (sin filtrado)": len(test),
        "Columnas predictoras reales": len(columnas), "Numéricas (excluye binarias y One-Hot)": conteos["numerica"],
        "Binarias (incluye indicadores, excluye One-Hot)": conteos["binaria"], "Columnas One-Hot": conteos["onehot"],
        "Derivadas finales (subconjunto de numéricas/binarias)": len(extracciones),
        "Entradas originales distintas representadas (incluye fecha)": len(originales_representadas),
        "Filas del inventario agrupado (no variables originales)": len(inventario),
        "Columnas totales CSV (X + objetivo + auxiliares)": train.shape[1],
    }
    if train.shape[1] != len(columnas) + len(pt.AUXILIARES) + 1:
        raise ValueError("Las dimensiones exportadas no concuerdan con X, respuesta y auxiliares.")
    reglas = [
        {"Regla": "Entradas de características", "Detalle": ", ".join(contrato["entradas_seleccionadas"])},
        {"Regla": "Auxiliares / respuesta", "Detalle": f"Auxiliares: {', '.join(pt.AUXILIARES)}; respuesta separada: {pt.OBJETIVO} (opcional en inferencia nueva)"},
        {"Regla": "Alias de entrada", "Detalle": json.dumps(pt.ALIAS_HIPOTECARIO, ensure_ascii=False)},
        {"Regla": "Secuencia", "Detalle": " → ".join(contrato["orden_train"])},
        {"Regla": "Nuevas observaciones / Test", "Detalle": "aplicar_flujo(nuevos, reglas): solo transform; sin fit ni Tukey; todos los parámetros provienen de Train"},
        {"Regla": "Categorías y orden completo de X", "Detalle": "Categorías completas en resumen_final.json; orden exacto en columnas_X del paquete y del manifiesto, contrastado con CSV (no se imprime el listado One-Hot)"},
        {"Regla": "Persistencia", "Detalle": "transformadores_prop_usadas.joblib; district independiente del codificador base; regenerar entrega y resumen ante cambios"},
    ]
    roles = [{"Variable / rol": pt.OBJETIVO, "Uso": "Objetivo; fuera de X", "Tipo Train": str(train[pt.OBJETIVO].dtype)},
             {"Variable / rol": ", ".join(pt.AUXILIARES), "Uso": "Auxiliares de trazabilidad; fuera de X", "Tipo Train": "Identidad, fecha, conjunto"}]
    tablas = {
        "Feature Selection: exclusiones y reemplazos": pd.DataFrame(seleccion),
        "Feature Transformation: operaciones ejecutadas": pd.DataFrame(transformaciones),
        "Feature Extraction: derivadas presentes (no One-Hot)": pd.DataFrame(extracciones, columns=["Característica", "Entradas", "Fórmula", "Finalidad"]),
        "Inventario agrupado de predictoras": pd.DataFrame(inventario),
        "Objetivo y auxiliares": pd.DataFrame(roles),
        "Dimensiones finales: clases disjuntas y procedencia": pd.DataFrame(cantidades.items(), columns=["Medida", "Cantidad"]),
        "Reglas para futuras observaciones": pd.DataFrame(reglas),
    }
    if joblib.hash(objetos) != huella_objetos or any(pt.huella_archivo(carpeta / n) != h for n, h in hashes.items()):
        raise ValueError("La auditoría modificó los datos o los transformadores.")
    return {"tablas": tablas, "columnas_X": columnas, "linaje": {c: linaje[c] for c in columnas},
            "categorias": grupos, "entradas_originales_representadas": originales_representadas,
            "archivos_sha256": hashes, "escalado": registro, "solo_documentacion": True,
            "generador_sha256": pt.huella_archivo(Path(__file__)),
            "convencion_nombres": "snake_case_ascii_v1",
            "correspondencia_nombres_finales": {a: b for a, b in correspondencia.items() if b in finales}}


def guardar_resumen(resumen, archivo):
    contenido = {k: v for k, v in resumen.items() if k != "tablas"}
    contenido["tablas"] = {k: v.to_dict(orient="records") for k, v in resumen["tablas"].items()}
    pt.guardar_json(archivo, contenido)
