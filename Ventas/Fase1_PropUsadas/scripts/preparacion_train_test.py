"""Flujo v2 acumulativo: selección → preparación → Tukey (solo Train) →
codificación → extracción → filtro de constantes/duplicadas → escalado.
Cada etapa recibe el DataFrame vigente. Las copias crudas son solo trazabilidad.
"""

import hashlib
import csv
import json
import re
import shutil
import tempfile
import warnings
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


NUMERICAS = [
    "total_area", "living_area", "kitchen_area", "rooms", "floor",
    "total_floors", "building_year", "ceiling_height", "metro_distance_min",
    "to_center_km", "metro_to_center_km",
    "mortgage_key_rate_at_listing",
]
CATEGORICAS = [
    "building_type", "renovation", "metro_line",
]
MAPAS_BINARIOS = {
    "has_balcony": {"false": 0, "true": 1, "no": 0, "yes": 1,
                    "0": 0, "1": 1, "0.0": 0, "1.0": 1},
    "metro_distance_type": {"walk": 0, "transport": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1},
    "seller_type": {"owner": 0, "agency": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1},
}
BINARIAS = list(MAPAS_BINARIOS)
CAMBIOS_MENSUALES = ["secondary_mom_change_pct"]
MARCAS_DOMINIO = ["surface_corrected", "building_year_inconsistent"]
ENTRADAS_X = NUMERICAS + CAMBIOS_MENSUALES + CATEGORICAS + BINARIAS + ["district"]
AUXILIARES = ["id", "date_posted_dt", "dataset_split"]
OBJETIVO = "price_rub"
EXCLUIDAS_MODELO = [
    "lat", "lon", "cbr_key_rate_pct", "avg_mortgage_rate_pct",
    "price_per_sqm", "secondary_price_per_sqm", "newbuild_price_per_sqm",
    "metro_lat", "metro_lon", "metro_year_opened", "newbuild_mom_change_pct",
]
ALIAS_HIPOTECARIO = {"mortgage_rate_at_listing": "mortgage_key_rate_at_listing"}


def nombre_predictor(nombre):
    """Convención única de salida; los nombres técnicos internos se conservan como origen."""
    texto = re.sub(r"^(?:(?:num|mom|bin|cat)__)+", "", str(nombre))
    texto = unicodedata.normalize("NFKD", texto).lower()
    texto = "".join(c if c.isascii() else (f"_u{ord(c):04x}_" if c.isalnum() else "_")
                    for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    if not texto:
        raise ValueError(f"Nombre de característica vacío al normalizar {nombre!r}.")
    return "variable_" + texto if texto[0].isdigit() else texto


class NormalizadorNombres(TransformerMixin, BaseEstimator):
    """Renombra sin modificar valores; aprende solo el esquema y verifica colisiones."""
    def __init__(self, roles=None):
        self.roles = roles

    def fit(self, X, y=None):
        if not X.columns.is_unique:
            raise ValueError("Nombres de entrada duplicados.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.correspondencia_ = {c: nombre_predictor(c) for c in X.columns}
        inverso = {}
        for anterior, nuevo in self.correspondencia_.items():
            if nuevo in AUXILIARES + [OBJETIVO]:
                raise ValueError(f"Nombre predictor reservado: {anterior!r} → {nuevo!r}.")
            if nuevo in inverso:
                raise ValueError(f"Colisión de nombres: {inverso[nuevo]!r} y {anterior!r} → {nuevo!r}.")
            inverso[nuevo] = anterior
        roles = self.roles if self.roles is not None else dict.fromkeys(X.columns, "numerica")
        if set(roles) != set(X.columns):
            raise ValueError("Los roles no cubren exactamente las salidas candidatas.")
        self.roles_salida_ = {self.correspondencia_[c]: roles[c] for c in X.columns}
        return self

    def transform(self, X):
        check_is_fitted(self, "correspondencia_")
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("Cambió el esquema anterior a la normalización de nombres.")
        return X.rename(columns=self.correspondencia_).copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "correspondencia_")
        return np.asarray(list(self.correspondencia_.values()), dtype=object)


def roles_candidatas(preparacion, base, district, extraccion):
    """Tipos semánticos desde productores ajustados, sin inferirlos de los nombres."""
    roles = {}
    nombres = base.get_feature_names_out().tolist()
    for bloque, _, entradas in base.transformers_:
        if bloque == "remainder":
            continue
        salidas = nombres[base.output_indices_[bloque]]
        if bloque == "cat":
            roles.update(dict.fromkeys(salidas, "onehot"))
        else:
            for entrada, salida in zip(entradas, salidas):
                roles[salida] = preparacion.derivadas_.get(entrada, {}).get(
                    "tipo", "binaria" if bloque == "bin" else "numerica")
    roles.update(dict.fromkeys(district.get_feature_names_out().tolist(), "onehot"))
    roles.update(dict.fromkeys(extraccion.mes_.get_feature_names_out().tolist(), "onehot"))
    roles.update({c: d["tipo"] for c, d in extraccion.derivadas_.items()})
    return roles

# Motivos de la selección explícita, junto a su configuración ejecutable.
MOTIVOS_EXCLUSION = {
    **dict.fromkeys(["lat", "lon", "metro_lat", "metro_lon"], "Coordenadas descartadas; se priorizan otras medidas de ubicación y accesibilidad."),
    **dict.fromkeys(["cbr_key_rate_pct", "avg_mortgage_rate_pct"], "Redundancia financiera; se conserva mortgage_key_rate_at_listing."),
    "price_per_sqm": "Precio por superficie derivado de la respuesta; riesgo de fuga de información.",
    **dict.fromkeys(["secondary_price_per_sqm", "newbuild_price_per_sqm"], "Se predice precio total; se descartan precios por unidad de superficie."),
    "metro_year_opened": "Año de apertura fuera del enfoque actual.",
    "newbuild_mom_change_pct": "Se conserva únicamente el cambio mensual del mercado secundario.",
}




def verificar_excluidas(datos):
    nombres = {nombre_predictor(c) for c in datos.columns}
    for variable in EXCLUIDAS_MODELO:
        prohibidos = {variable, f"missingindicator_{variable}"}
        if nombres.intersection(prohibidos) or any(c.startswith(f"{variable}_") for c in nombres):
            raise ValueError(f"Reapareció una variable excluida: {variable}.")


def seleccionar_variables(datos):
    """Primera etapa después de dividir: preserva auxiliares/objetivo y solo las entradas vigentes."""
    datos = datos.copy()
    for original, canonico in ALIAS_HIPOTECARIO.items():
        if original in datos and canonico in datos:
            a, b = pd.to_numeric(datos[original]), pd.to_numeric(datos[canonico])
            if not (a.eq(b) | (a.isna() & b.isna())).all():
                raise ValueError("Los alias hipotecarios contienen valores distintos.")
        elif original in datos:
            datos = datos.rename(columns={original: canonico})
    requeridas = [*AUXILIARES, *ENTRADAS_X]
    if not set(requeridas).issubset(datos):
        raise ValueError(f"Faltan entradas: {sorted(set(requeridas) - set(datos))}")
    columnas = [*AUXILIARES, *([OBJETIVO] if OBJETIVO in datos else []), *ENTRADAS_X]
    resultado = datos.loc[:, columnas].copy()
    verificar_excluidas(resultado)
    return resultado


def obtener_X(datos):
    return datos.drop(columns=[*AUXILIARES, OBJETIVO], errors="ignore").copy()


def reemplazar_X(datos, matriz):
    if not datos.index.equals(matriz.index):
        raise ValueError("La matriz perdió alineación con sus anuncios.")
    auxiliares = [c for c in [*AUXILIARES, OBJETIVO] if c in datos]
    return integrar_caracteristicas(datos[auxiliares], matriz)


def registrar_etapa(etapa, antes, despues, registros, permite_filtrar=False):
    """Comprueba continuidad de identidad, orden y respuesta; no consulta distribuciones."""
    if not despues.index.is_unique or not despues["id"].is_unique:
        raise ValueError(f"Identidad duplicada en {etapa}.")
    if permite_filtrar:
        esperados = antes.index[antes.index.isin(despues.index)]
    else:
        esperados = antes.index
    if not despues.index.equals(esperados):
        raise ValueError(f"Se perdieron, reincorporaron o reordenaron filas en {etapa}.")
    identidad = [c for c in [*AUXILIARES, OBJETIVO] if c in antes]
    pd.testing.assert_frame_equal(despues[identidad], antes.loc[despues.index, identidad])
    verificar_excluidas(despues)
    if not despues["date_posted_dt"].is_monotonic_increasing:
        raise ValueError(f"Cronología inválida en {etapa}.")
    huella = hashlib.sha256(pd.util.hash_pandas_object(despues["id"], index=False).to_numpy().tobytes()).hexdigest()
    registros.append({"etapa": etapa, "filas_entrada": len(antes), "filas_salida": len(despues),
                      "filas_eliminadas": len(antes) - len(despues),
                      "predictoras": obtener_X(despues).shape[1], "ids_sha256": huella})


def corregir_dominio(datos):
    verificar_excluidas(datos)
    salida = datos.copy()
    fechas = pd.to_datetime(salida["date_posted_dt"], errors="raise")
    if fechas.isna().any():
        raise ValueError("Fecha ausente.")
    for c in NUMERICAS + CAMBIOS_MENSUALES:
        salida[c] = pd.to_numeric(salida[c], errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)
    salida["surface_corrected"] = (
        salida[["total_area", "living_area", "kitchen_area"]].notna().all(axis=1)
        & salida.total_area.lt(salida.living_area + salida.kitchen_area)).astype(float)
    salida = corregir_superficie_total(salida)
    invalido = salida.building_year.notna() & (
        salida.building_year.le(0) | salida.building_year.gt(fechas.dt.year) | salida.building_year.mod(1).ne(0))
    salida["building_year_inconsistent"] = invalido.astype(float)
    salida.loc[invalido, "building_year"] = np.nan
    for c in ["total_area", "ceiling_height"]:
        salida.loc[salida[c].le(0), c] = np.nan
    for c in ["rooms", "floor", "total_floors", "metro_distance_min", "to_center_km",
              "metro_to_center_km", "mortgage_key_rate_at_listing"]:
        salida.loc[salida[c].lt(0), c] = np.nan
    for c in ["rooms", "floor", "total_floors"]:
        salida.loc[salida[c].notna() & salida[c].mod(1).ne(0), c] = np.nan
    salida.loc[salida.floor.gt(salida.total_floors), "floor"] = np.nan
    for c, mapa in MAPAS_BINARIOS.items():
        salida[c] = normalizar_binaria(salida[c]).map(mapa).astype(float)
    for c in CATEGORICAS + ["district"]:
        texto = salida[c].astype("string").str.normalize("NFKC").str.strip().replace("", pd.NA)
        salida[c] = texto.astype(object).where(texto.notna(), np.nan)
    return salida


class PreparacionInicial(TransformerMixin, BaseEstimator):
    """Ajusta imputadores una sola vez con el Train seleccionado, antes de Tukey."""

    def fit(self, X, y=None):
        datos = corregir_dominio(X)
        for c, mapa in MAPAS_BINARIOS.items():
            etiquetas = normalizar_binaria(X[c])
            if not etiquetas.dropna().isin(mapa).all() or datos[c].notna().sum() == 0:
                raise ValueError(f"Binaria de Train inválida o vacía: {c}.")
        for c in CAMBIOS_MENSUALES:
            if not datos[c].notna().any():
                raise ValueError(f"No hay valores de Train para estimar la media: {c}.")
        self.numerico_ = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True).set_output(transform="pandas").fit(datos[NUMERICAS])
        self.mensual_ = SimpleImputer(strategy="mean", keep_empty_features=True).set_output(transform="pandas").fit(datos[CAMBIOS_MENSUALES])
        self.binario_ = SimpleImputer(strategy="most_frequent", keep_empty_features=True).set_output(transform="pandas").fit(datos[BINARIAS])
        self.categorico_ = SimpleImputer(strategy="constant", fill_value="__FALTANTE__", keep_empty_features=True).set_output(transform="pandas").fit(datos[CATEGORICAS + ["district"]])
        self.columnas_numericas_ = [*self.numerico_.get_feature_names_out().tolist(), *MARCAS_DOMINIO]
        self.derivadas_ = {
            "surface_corrected": {"entradas": ["total_area", "living_area", "kitchen_area"],
                "formula": "1 si total_area < living_area + kitchen_area con superficies finitas antes de corregir; 0 en otro caso",
                "finalidad": "Identificar corrección de superficie total", "tipo": "binaria"},
            "building_year_inconsistent": {"entradas": ["building_year", "date_posted_dt"],
                "formula": "1 si año observado <= 0, posterior a la publicación o no entero; 0 en otro caso",
                "finalidad": "Identificar año incoherente", "tipo": "binaria"},
        }
        for indice in self.numerico_.indicator_.features_:
            variable = self.numerico_.feature_names_in_[indice]
            self.derivadas_["missingindicator_" + variable] = {
                "entradas": [str(variable)], "formula": "1 si falta después de corregir dominio y antes de imputar; 0 en otro caso",
                "finalidad": "Conservar señal de imputación", "tipo": "binaria"}
        return self

    def transform(self, X):
        check_is_fitted(self, "numerico_")
        salida = corregir_dominio(X)
        for imputador, columnas in [(self.numerico_, NUMERICAS), (self.mensual_, CAMBIOS_MENSUALES),
                                    (self.binario_, BINARIAS), (self.categorico_, CATEGORICAS + ["district"])]:
            valores = imputador.transform(salida[columnas])
            for c in valores:
                salida[c] = valores[c]
        salida = corregir_superficie_total(salida)
        validar_superficies(salida)
        if salida[ENTRADAS_X].isna().any().any():
            raise ValueError("Persisten faltantes después de la preparación inicial.")
        return salida


def validar_superficies(datos, exigir_columnas=True):
    nombres = {nombre_predictor(c): c for c in datos.columns}
    areas = ["total_area", "living_area", "kitchen_area"]
    if not set(areas).issubset(nombres):
        if exigir_columnas:
            raise ValueError("Faltan superficies para comprobar la restricción.")
        return  # La restricción ya se verificó antes de retirar columnas constantes/duplicadas.
    total, sala, cocina = [nombres[c] for c in areas]
    if not datos[total].ge(datos[sala] + datos[cocina]).all():
        raise ValueError("Se perdió la corrección de superficies.")


def auditar_preparacion(antes, despues, preparador):
    """Compara la etapa ya ejecutada, sin reconstruir ni transformar desde una copia cruda."""
    for c in ["living_area", "kitchen_area"]:
        observado = np.isfinite(pd.to_numeric(antes[c]).astype(float))
        np.testing.assert_array_equal(despues.loc[observado, c], antes.loc[observado, c])
    resumen = {"filas": len(despues), "superficies_corregidas": int(despues.surface_corrected.sum()), "imputaciones_mom": {}}
    cambios = []
    for c in ["total_area", *CAMBIOS_MENSUALES]:
        original = pd.to_numeric(antes[c]).astype(float)
        mascara = ~np.isfinite(original) | original.ne(despues[c])
        if c in CAMBIOS_MENSUALES:
            observados = np.isfinite(original)
            np.testing.assert_array_equal(despues.loc[observados, c], original.loc[observados])
            resumen["imputaciones_mom"][c] = {
                "imputados": int((~observados).sum()), "faltantes_finales": int(despues[c].isna().sum()),
                "media_train": float(preparador.mensual_.statistics_[CAMBIOS_MENSUALES.index(c)])}
        for indice in antes.index[mascara]:
            cambios.append({"id": antes.at[indice, "id"], "variable": c, "valor_anterior": original.at[indice],
                            "valor_nuevo": despues.at[indice, c],
                            "motivo": "correccion_total_area" if c == "total_area" else "imputacion_media_train"})
    return resumen, pd.DataFrame(cambios, columns=["id", "variable", "valor_anterior", "valor_nuevo", "motivo"])


def crear_codificador_base(columnas_numericas):
    """Recibe columnas ya corregidas e imputadas; nunca vuelve a imputar desde los originales."""
    return ColumnTransformer([
        ("num", "passthrough", columnas_numericas),
        ("mom", "passthrough", CAMBIOS_MENSUALES),
        ("bin", "passthrough", BINARIAS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ], remainder="drop", verbose_feature_names_out=True).set_output(transform="pandas")


def nombre_columna_mes(variable, categoria):
    return f"cat__month_of_year_{categoria}"


class ExtraccionTemporal(TransformerMixin, BaseEstimator):
    """Agrega las derivadas temporales al DataFrame ya codificado."""

    def fit(self, X, y=None):
        fechas = pd.to_datetime(X["date_posted_dt"])
        self.origen_mes_ = int(fechas.dt.year.min() * 12 + fechas.loc[fechas.idxmin()].month - 1)
        self.derivadas_ = {"num__months_since_start": {
            "entradas": ["date_posted_dt"],
            "formula": f"12 * año + mes - 1 - {self.origen_mes_}",
            "finalidad": "Tendencia mensual desde el origen aprendido con Train", "tipo": "numerica"}}
        self.mes_ = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False,
                                 feature_name_combiner=nombre_columna_mes).set_output(transform="pandas")
        self.mes_.fit(pd.DataFrame({"month_of_year": fechas.dt.month.astype(str)}, index=X.index))
        return self

    def transform(self, X):
        check_is_fitted(self, "mes_")
        fechas = pd.to_datetime(X["date_posted_dt"])
        tendencia = pd.DataFrame({"num__months_since_start": fechas.dt.year * 12 + fechas.dt.month - 1 - self.origen_mes_}, index=X.index)
        mes = self.mes_.transform(pd.DataFrame({"month_of_year": fechas.dt.month.astype(str)}, index=X.index))
        return integrar_caracteristicas(X, integrar_caracteristicas(tendencia, mes))


def crear_filtro_final(roles=None):
    return Pipeline([("nombres", NormalizadorNombres(roles)), ("constantes", VarianceThreshold(0)),
                     ("duplicadas", EliminarDuplicadas())]).set_output(transform="pandas")


def aplicar_flujo(datos, objetos):
    """Aplicación acumulativa a futuro/Test: no contiene fit ni filtro del objetivo."""
    registros = []
    actual = seleccionar_variables(datos)
    registrar_etapa("seleccion", datos, actual, registros)
    anterior = actual
    actual = objetos["preparacion"].transform(actual)
    registrar_etapa("correccion_imputacion", anterior, actual, registros)
    resumen, auditoria = auditar_preparacion(anterior, actual, objetos["preparacion"])
    anterior = actual
    matriz = integrar_caracteristicas(objetos["base"].transform(actual), objetos["district"].transform(actual[["district"]]))
    actual = reemplazar_X(actual, matriz)
    registrar_etapa("codificacion", anterior, actual, registros)
    anterior = actual
    actual = objetos["extraccion"].transform(actual)
    registrar_etapa("extraccion", anterior, actual, registros)
    validar_superficies(actual)
    anterior = actual
    actual = reemplazar_X(actual, objetos["filtro"].transform(obtener_X(actual)))
    registrar_etapa("filtro_final", anterior, actual, registros)
    anterior = actual
    actual = reemplazar_X(actual, objetos["escalado"].transform(obtener_X(actual)))
    registrar_etapa("escalado", anterior, actual, registros)
    validar_matriz(obtener_X(actual), datos, columnas=objetos["columnas_X"])
    validar_superficies(actual, exigir_columnas=False)
    return actual, resumen, auditoria, pd.DataFrame(registros)


def huella_archivo(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def guardar_json(path, contenido):
    from .almacenamiento_mlflow import validar_temporal
    validar_temporal(Path(path).parent)
    Path(path).write_text(json.dumps(contenido, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def cargar_y_dividir(entrada, carpeta):
    """Lee y divide antes de transformar; revalida la asignación persistida por ID."""
    from .almacenamiento_mlflow import validar_temporal
    entrada, carpeta = Path(entrada), validar_temporal(carpeta)
    datos = pd.read_csv(entrada, dtype={"id": "string"})
    # Solo controles estructurales, lectura de fechas y orden para poder particionar.
    if datos.empty or not datos.columns.is_unique:
        raise ValueError("La entrada debe tener filas y columnas únicas.")
    if not {"id", "date_posted_dt"}.issubset(datos):
        raise ValueError("Faltan id o date_posted_dt.")
    if datos["id"].isna().any() or not datos["id"].is_unique:
        raise ValueError("Los ID deben ser únicos y no nulos.")
    datos["date_posted_dt"] = pd.to_datetime(datos["date_posted_dt"], format="%Y-%m-%d", errors="raise")
    if datos["date_posted_dt"].isna().any():
        raise ValueError("No se puede ordenar una fecha ausente.")
    datos = datos.sort_values(["date_posted_dt", "id"], kind="stable").reset_index(drop=True)
    conteos = datos.groupby("date_posted_dt", sort=True).size()
    if len(conteos) < 2:
        raise ValueError("Se necesitan al menos dos fechas distintas para separar pasado y futuro.")
    # Corte más cercano al 80 % entre días completos; empate: menor tamaño de Train.
    candidatos = conteos.cumsum().iloc[:-1]
    fecha_corte = (candidatos * 5 - len(datos) * 4).abs().idxmin()
    datos["dataset_split"] = np.where(datos["date_posted_dt"].le(fecha_corte), "train", "test")
    asignacion = datos[AUXILIARES].copy()
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo_asignacion = carpeta / "particion_fija.csv"
    archivo_contrato = carpeta / "particion_fija.json"
    contrato = {
        "entrada_sha256": huella_archivo(entrada), "fraccion_train_objetivo": .8,
        "criterio": "fecha completa más cercana al 80%; empate: corte anterior",
        "ultima_fecha_train": fecha_corte.strftime("%Y-%m-%d"),
        "filas_entrada": len(datos),
    }
    if archivo_asignacion.exists() or archivo_contrato.exists():
        if not (archivo_asignacion.exists() and archivo_contrato.exists()):
            raise ValueError("Partición persistida incompleta; revisar los artefactos existentes.")
        previo = json.loads(archivo_contrato.read_text(encoding="utf-8"))
        if previo != contrato:
            raise ValueError("Cambió la fuente o el contrato. No se redefine la partición congelada.")
        persistida = pd.read_csv(archivo_asignacion, dtype={"id": "string"}, parse_dates=["date_posted_dt"])
        pd.testing.assert_frame_equal(asignacion, persistida, check_dtype=False)
    else:
        asignacion.to_csv(archivo_asignacion, index=False, date_format="%Y-%m-%d")
        guardar_json(archivo_contrato, contrato)
    train = datos.loc[datos["dataset_split"].eq("train")].copy()
    test = datos.loc[datos["dataset_split"].eq("test")].copy()
    if train["date_posted_dt"].max() >= test["date_posted_dt"].min():
        raise ValueError("La frontera no separa estrictamente pasado y futuro.")
    resumen = datos.groupby("dataset_split").agg(
        registros=("id", "size"), desde=("date_posted_dt", "min"), hasta=("date_posted_dt", "max")
    ).reindex(["train", "test"])
    resumen["porcentaje"] = 100 * resumen["registros"] / len(datos)
    resumen["desviacion_pp"] = resumen["porcentaje"] - [80, 20]
    return train, test, resumen, contrato


def filtrar_outliers_tukey_train(train, objetivo=OBJETIVO):
    """Selecciona filas completas de Train; no debe aplicarse a Test o validación."""
    if (train.empty or "dataset_split" not in train or train["dataset_split"].isna().any()
            or not train["dataset_split"].eq("train").all()):
        raise ValueError("Tukey requiere un conjunto Train no vacío y exclusivamente de entrenamiento.")
    precio = pd.to_numeric(train[objetivo], errors="raise")
    if precio.isna().any() or not np.isfinite(precio).all():
        raise ValueError("El precio debe ser finito y no nulo; no se imputa para calcular Tukey.")
    q1, q3 = precio.quantile([.25, .75], interpolation="linear")
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier = precio.lt(inferior) | precio.gt(superior)
    filtrado, eliminados = train.loc[~outlier].copy(), train.loc[outlier].copy()
    resumen = {
        "metodo": "Tukey", "objetivo": objetivo, "factor_iqr": 1.5, "interpolacion": "linear",
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "limite_inferior": float(inferior), "limite_superior": float(superior),
        "filas_iniciales": len(train), "outliers_eliminados": len(eliminados),
        "porcentaje_eliminado": 100 * len(eliminados) / len(train), "filas_finales": len(filtrado),
    }
    if not pd.to_numeric(filtrado[objetivo]).between(inferior, superior, inclusive="both").all():
        raise ValueError("Persisten precios fuera de los límites de Tukey de Train.")
    return filtrado, eliminados, resumen


def normalizar_binaria(serie):
    return serie.astype("string").str.normalize("NFKC").str.strip().str.lower().replace("", pd.NA)


def corregir_superficie_total(datos, prefijo=""):
    """Corrige por fila; no altera las superficies de sala de estar y cocina."""
    salida = datos.copy()
    total, sala, cocina = [prefijo + c for c in ["total_area", "living_area", "kitchen_area"]]
    suma = salida[sala] + salida[cocina]
    finitas = np.isfinite(salida[[total, sala, cocina]]).all(axis=1)
    corregir = finitas & salida[total].lt(suma)
    salida.loc[corregir, total] = suma.loc[corregir]
    return salida


class EliminarDuplicadas(TransformerMixin, BaseEstimator):
    """Retira columnas idénticas en Train; conserva el orden y la primera aparición."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.columnas_ = []
        self.duplicadas_ = {}
        vistas = {}
        for nombre in X.columns:
            valores = X[nombre].to_numpy(dtype=float).copy()
            valores[valores == 0] = 0.0  # Unificar los bytes de +0 y -0.
            huella = hashlib.sha256(valores.tobytes()).hexdigest()
            anterior = vistas.get(huella)
            if anterior is not None and np.array_equal(X[anterior], X[nombre]):
                self.duplicadas_[nombre] = anterior
            else:
                vistas[huella] = nombre
                self.columnas_.append(nombre)
        return self

    def transform(self, X):
        check_is_fitted(self, "columnas_")
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("Cambió el esquema anterior al filtro de duplicadas.")
        return X.loc[:, self.columnas_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columnas_")
        return np.asarray(self.columnas_, dtype=object)


def validar_matriz(matriz, datos, columnas=None):
    if not matriz.index.equals(datos.index) or len(matriz) != len(datos):
        raise ValueError("Se alteró la identidad, el orden o la cantidad de filas.")
    if not matriz.columns.is_unique or matriz.shape[1] == 0:
        raise ValueError("Esquema vacío o duplicado.")
    if set(EXCLUIDAS_MODELO).intersection(c.split("__")[-1] for c in matriz.columns):
        raise ValueError("La matriz conserva variables excluidas del modelado.")
    if columnas is not None and matriz.columns.tolist() != list(columnas):
        raise ValueError("Test no conserva el esquema y orden aprendidos en Train.")
    if not np.isfinite(matriz.to_numpy(dtype=float)).all():
        raise ValueError("La matriz contiene faltantes o valores no finitos.")
    for columna in matriz.columns:
        canonico = nombre_predictor(columna)
        if (canonico in BINARIAS + MARCAS_DOMINIO or canonico.startswith("missingindicator_")) and not matriz[columna].isin([0, 1]).all():
            raise ValueError(f"Representación binaria inválida: {columna}.")


def normalizar_district(datos):
    """Normalización local del bloque district, independiente del pipeline base."""
    texto = datos["district"].astype("string").str.normalize("NFKC").str.strip().replace("", pd.NA)
    return pd.DataFrame({"district": texto.astype(object).where(texto.notna(), np.nan)}, index=datos.index)


def nombre_columna_district(variable, categoria):
    return f"district__{categoria}"


def crear_codificador_district():
    """Estrategia provisional independiente; no usa el objetivo ni implementa Target Encoding."""
    return Pipeline([
        ("normalizar", FunctionTransformer(normalizar_district, feature_names_out="one-to-one")),
        ("imputar", SimpleImputer(strategy="constant", fill_value="__FALTANTE__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False,
                                 feature_name_combiner=nombre_columna_district)),
    ]).set_output(transform="pandas")


def integrar_caracteristicas(base, adicionales):
    """Une bloques sin depender del tipo de codificador ni de sus categorías."""
    if not base.index.equals(adicionales.index):
        raise ValueError("Los bloques de características no tienen el mismo índice y orden.")
    if not base.columns.is_unique or not adicionales.columns.is_unique or set(base).intersection(adicionales):
        raise ValueError("Los bloques contienen nombres de columnas duplicados.")
    return pd.concat([base, adicionales], axis=1)


def columnas_para_escala(matriz, roles):
    """Descubre numéricas en la matriz vigente; nunca reconstruye columnas desde listas de entrada."""
    verificar_excluidas(matriz)
    auxiliares = set(AUXILIARES + [OBJETIVO])
    candidatas = [c for c in matriz.select_dtypes(include="number").columns if c not in auxiliares]
    if not set(candidatas).issubset(roles):
        raise ValueError("Hay predictoras sin rol documentado para analizar escala.")
    return [c for c in candidatas if roles[c] == "numerica"]


def verificar_estado_analizado(matriz, huella, columnas_evaluadas, roles):
    if joblib.hash(matriz) != huella:
        raise ValueError("El DataFrame cambió desde el análisis: no ajustar sobre otro estado ni escalar dos veces.")
    if columnas_para_escala(matriz, roles) != list(columnas_evaluadas):
        raise ValueError("Las variables del análisis no coinciden con las numéricas actuales.")


class EscaladoSelectivo(TransformerMixin, BaseEstimator):
    """Escala solo columnas seleccionadas; conserva nombres, orden y demás valores."""

    def __init__(self, columnas_estandarizar=(), columnas_normalizar=()):
        self.columnas_estandarizar = columnas_estandarizar
        self.columnas_normalizar = columnas_normalizar

    def fit(self, X, y=None):
        estandarizar, normalizar = list(self.columnas_estandarizar), list(self.columnas_normalizar)
        seleccionadas = estandarizar + normalizar
        if len(seleccionadas) != len(set(seleccionadas)):
            raise ValueError("Una variable no puede recibir dos escalados.")
        if not set(seleccionadas).issubset(X.columns):
            raise ValueError("Faltan columnas seleccionadas para el escalado.")
        if not X.columns.is_unique or not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("La entrada del escalado debe ser finita y tener columnas únicas.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.filas_ajuste_ = len(X)
        self.huella_entrada_train_ = joblib.hash(X)
        self.columnas_estandarizar_ = estandarizar
        self.columnas_normalizar_ = normalizar
        self.estandarizador_ = StandardScaler().fit(X[estandarizar]) if estandarizar else None
        self.normalizador_ = MinMaxScaler(clip=False).fit(X[normalizar]) if normalizar else None
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if X.columns.tolist() != self.feature_names_in_.tolist():
            raise ValueError("El esquema del escalado difiere del ajustado con Train.")
        salida = X.copy()
        if self.estandarizador_ is not None:
            salida[self.columnas_estandarizar_] = self.estandarizador_.transform(X[self.columnas_estandarizar_])
        if self.normalizador_ is not None:
            salida[self.columnas_normalizar_] = self.normalizador_.transform(X[self.columnas_normalizar_])
        return salida

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


def registro_escalado(escalador):
    check_is_fitted(escalador, "feature_names_in_")
    estandar, normalizar = escalador.estandarizador_, escalador.normalizador_
    return {
        "filas_ajuste": escalador.filas_ajuste_,
        "huella_entrada_train": escalador.huella_entrada_train_,
        "estandarizadas": escalador.columnas_estandarizar_,
        "normalizadas": escalador.columnas_normalizar_,
        "media_train": {} if estandar is None else dict(zip(escalador.columnas_estandarizar_, estandar.mean_.tolist())),
        "desviacion_train_ddof0": {} if estandar is None else dict(zip(escalador.columnas_estandarizar_, np.sqrt(estandar.var_).tolist())),
        "escala_standard_scaler": {} if estandar is None else dict(zip(escalador.columnas_estandarizar_, estandar.scale_.tolist())),
        "minimo_train": {} if normalizar is None else dict(zip(escalador.columnas_normalizar_, normalizar.data_min_.tolist())),
        "maximo_train": {} if normalizar is None else dict(zip(escalador.columnas_normalizar_, normalizar.data_max_.tolist())),
        "intervalo_minmax": [0, 1], "recortar_test": False,
    }


def exportar_conjunto(datos, matriz, archivo):
    validar_matriz(matriz, datos)
    if any(c != nombre_predictor(c) for c in matriz):
        raise ValueError("Las predictoras finales no cumplen la convención de nombres.")
    objetivo = pd.to_numeric(datos[OBJETIVO], errors="raise")
    if not (objetivo.notna().all() and np.isfinite(objetivo).all() and objetivo.gt(0).all()):
        raise ValueError("Objetivo ausente, no finito o no positivo: no se imputa ni se elimina silenciosamente.")
    exportado = pd.concat([datos[AUXILIARES], objetivo, matriz], axis=1)
    exportado.to_csv(archivo, index=False, date_format="%Y-%m-%d")
    recarga = pd.read_csv(archivo, dtype={"id": "string"}, parse_dates=["date_posted_dt"],
                          float_precision="round_trip")
    pd.testing.assert_frame_equal(exportado.reset_index(drop=True), recarga,
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    areas = ["total_area", "living_area", "kitchen_area"]
    if set(areas).issubset(recarga):
        if not recarga[areas[0]].ge(recarga[areas[1]] + recarga[areas[2]]).all():
            raise ValueError("El CSV recargado presenta superficies inconsistentes.")
    return {"filas": len(exportado), "predictoras": matriz.shape[1], "sha256": huella_archivo(archivo)}


def estado_aprendido(objetos):
    """Parámetros y esquemas persistidos, sin ejecutar fit ni transform."""
    preparacion, base = objetos["preparacion"], objetos["base"]
    distrito, temporal = objetos["district"], objetos["extraccion"]
    filtro, escala = objetos["filtro"], objetos["escalado"]
    codificadores = {"base": base.named_transformers_["cat"],
                    "district": distrito.named_steps["onehot"], "mes": temporal.mes_}
    return {
        "columnas_X": objetos["columnas_X"],
        "imputadores": {k: getattr(preparacion, k).statistics_ for k in
                        ("numerico_", "mensual_", "binario_", "categorico_")},
        "indicadores": preparacion.numerico_.indicator_.features_,
        "categorias": {k: {"categorias": e.categories_, "referencias": e.drop_idx_,
                           "salidas": e.get_feature_names_out(), "opciones": e.get_params()}
                       for k, e in codificadores.items()},
        "imputador_district": distrito.named_steps["imputar"].statistics_,
        "origen_mes": temporal.origen_mes_,
        "correspondencia": filtro.named_steps["nombres"].correspondencia_,
        "roles": filtro.named_steps["nombres"].roles_salida_,
        "varianzas": filtro.named_steps["constantes"].variances_,
        "duplicadas": filtro.named_steps["duplicadas"].duplicadas_,
        "seleccionadas": filtro.get_feature_names_out(),
        "escala": registro_escalado(escala),
    }


def exportar_entrega_final(train, test, train_crudo, test_crudo, objetos, contrato, carpeta, *, carpeta_artefactos):
    """Publica datasets separados y reglas Train verificadas; manifiesto escrito al final."""
    columnas = list(objetos["columnas_X"])
    if set(columnas).intersection([*AUXILIARES, OBJETIVO]):
        raise ValueError("La respuesta y los auxiliares no pueden ser predictores.")
    for datos in (train, test):
        verificar_excluidas(datos)
        validar_matriz(obtener_X(datos), datos, columnas=columnas)
    if not train["id"].is_unique or not test["id"].is_unique:
        raise ValueError("Los identificadores deben ser únicos por conjunto.")
    if set(train_crudo["id"]) & set(test_crudo["id"]):
        raise ValueError("Train y Test no pueden compartir observaciones.")
    if not train_crudo["date_posted_dt"].max() < test_crudo["date_posted_dt"].min():
        raise ValueError("Test debe ser cronológicamente posterior a Train.")
    retenidos = train_crudo.loc[train_crudo["id"].isin(train["id"])]
    for actual, original, etiqueta in ((train, retenidos, "train"), (test, test_crudo, "test")):
        if not actual["dataset_split"].eq(etiqueta).all():
            raise ValueError("Asignación Train/Test inconsistente.")
        pd.testing.assert_frame_equal(actual[[*AUXILIARES, OBJETIVO]].reset_index(drop=True),
                                      original[[*AUXILIARES, OBJETIVO]].reset_index(drop=True))

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    # No se persisten DataFrames ni referencias al kernel. district queda independiente.
    paquete = dict(objetos, configuracion=contrato)
    huella_antes = joblib.hash(paquete)
    from .almacenamiento_mlflow import validar_temporal
    carpeta_artefactos = validar_temporal(carpeta_artefactos)
    carpeta_artefactos.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="entrega_") as temporal:
        staging = Path(temporal)
        ruta_objetos = staging / "transformadores_prop_usadas.joblib"
        joblib.dump(paquete, ruta_objetos, compress=3)
        recuperados = joblib.load(ruta_objetos)
        huella_recuperada = joblib.hash(recuperados)
        # La exportación no reconstruye ni vuelve a transformar los datos acumulados.
        np.testing.assert_equal(estado_aprendido(paquete), estado_aprendido(recuperados))
        np.testing.assert_equal(contrato, recuperados["configuracion"])
        if joblib.hash(recuperados) != huella_recuperada or joblib.hash(paquete) != huella_antes:
            raise ValueError("La reutilización modificó los parámetros ajustados con Train.")
        resumen = {}
        for etiqueta, datos in (("train", train), ("test", test)):
            nombre = f"{etiqueta}_transformado.csv"
            resumen[etiqueta] = dict(exportar_conjunto(datos, obtener_X(datos), staging / nombre),
                                    archivo=nombre, columnas=len(datos.columns),
                                    fecha_min=str(datos["date_posted_dt"].min().date()),
                                    fecha_max=str(datos["date_posted_dt"].max().date()))
        nombres = ["train_transformado.csv", "test_transformado.csv", ruta_objetos.name]
        manifiesto = {
            "version_flujo": contrato["version_flujo"], "columnas_X": columnas,
            "train": resumen["train"], "test": resumen["test"],
            "filas_train_original": len(train_crudo), "filas_test_original": len(test_crudo),
            "test_sin_filtrado_objetivo": True, "exportacion_desde_dataframes_finales": True,
            "parametros_recuperados_sin_cambios": True,
            "modulo_sha256": huella_archivo(Path(__file__)),
            "archivos_sha256": {nombre: huella_archivo(staging / nombre) for nombre in nombres},
        }
        guardar_json(staging / "manifiesto_exportacion.json", manifiesto)
        # La entrega congelada del repositorio no puede cambiar por una limpieza.
        from .almacenamiento_mlflow import RAIZ
        if carpeta.resolve().is_relative_to(RAIZ):
            from .contrato_datasets import CONTRATO, DATA_DIR
            if carpeta.resolve() != DATA_DIR:
                raise ValueError('Usar la ubicación única de datasets: Propiedades_Usadas/data.')
            if columnas != CONTRATO["columnas_X"]:
                raise ValueError("Las predictoras difieren del contrato de datos.")
            for conjunto in ('train', 'test'):
                if any(resumen[conjunto][k] != v for k, v in CONTRATO[conjunto].items()):
                    raise ValueError('La entrega difiere del contrato congelado; revisar antes de actualizar datasets.')
        # Solo los dos datasets son permanentes; reproducibilidad en el temporal.
        for nombre in nombres[:2]:
            shutil.copyfile(staging / nombre, carpeta / nombre)
        for nombre in [ruta_objetos.name, "manifiesto_exportacion.json"]:
            shutil.copyfile(staging / nombre, carpeta_artefactos / nombre)
    return manifiesto


def validar_entrega_para_ols(train, test, carpeta):
    """Recarga la entrega definitiva y prueba el mismo lector que usa 3_Modelo_OLS."""
    from .contrato_datasets import CONTRATO, ruta_dataset
    from . import baseline_ols
    carpeta = Path(carpeta)
    resultados, errores, recargados = [], [], {}
    for conjunto, esperado in (("train", train), ("test", test)):
        archivo = carpeta / f"{conjunto}_transformado.csv"
        fila = {"Archivo exportado": archivo.name, "Observaciones": None,
                "Variables predictoras": None, "Columnas totales": None, "Estado": "Error"}
        try:
            if archivo.resolve() != ruta_dataset(archivo.name, carpeta.parent).resolve():
                raise ValueError("El archivo no está en la ruta data/ esperada por OLS.")
            with archivo.open(encoding="utf-8", newline="") as fuente:
                cabecera = next(csv.reader(fuente))
            if len(cabecera) != len(set(cabecera)):
                raise ValueError("El CSV contiene encabezados duplicados.")
            datos = pd.read_csv(archivo, dtype={"id": "string"}, parse_dates=["date_posted_dt"],
                                float_precision="round_trip")
            X = obtener_X(datos)
            fila.update({"Observaciones": len(datos), "Variables predictoras": X.shape[1],
                         "Columnas totales": datos.shape[1]})
            contrato = CONTRATO[conjunto]
            if (cabecera != [*AUXILIARES, OBJETIVO, *CONTRATO["columnas_X"]]
                    or X.columns.tolist() != CONTRATO["columnas_X"]):
                raise ValueError("Los nombres u orden de columnas difieren del contrato.")
            if (huella_archivo(archivo) != contrato["sha256"] or len(datos) != contrato["filas"]
                    or X.shape[1] != contrato["predictoras"] or datos.shape[1] != contrato["columnas"]
                    or archivo.name != contrato["archivo"]
                    or str(datos.date_posted_dt.min().date()) != contrato["fecha_min"]
                    or str(datos.date_posted_dt.max().date()) != contrato["fecha_max"]):
                raise ValueError("La huella, dimensiones o fechas difieren del contrato.")
            if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
                raise ValueError("Hay predictoras no numéricas.")
            validar_matriz(X, datos, columnas=CONTRATO["columnas_X"])
            verificar_excluidas(datos)
            if not datos.dataset_split.eq(conjunto).all():
                raise ValueError("Asignación Train/Test incorrecta.")
            pd.testing.assert_frame_equal(datos, esperado.reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
            recargados[conjunto] = datos
            fila["Estado"] = "Correcto"
        except (AssertionError, ValueError, OSError, KeyError, StopIteration) as exc:
            errores.append(f"{archivo.name}: {exc}")
        resultados.append(fila)
    if "train" in recargados:
        try:
            X_ols, y_ols, contexto = baseline_ols.cargar_train(carpeta.parent)
            pd.testing.assert_frame_equal(X_ols, obtener_X(recargados["train"]), check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
            pd.testing.assert_series_equal(y_ols, recargados["train"][OBJETIVO], check_dtype=False,
                                           check_exact=True)
            if contexto["test_utilizado"] or set(X_ols).intersection([*AUXILIARES, OBJETIVO]):
                raise ValueError("OLS incorporó Test, la respuesta o auxiliares como predictores.")
        except (AssertionError, ValueError, OSError, KeyError) as exc:
            resultados[0]["Estado"] = "Error"
            errores.append(f"cargar_train(): {exc}")
    tabla = pd.DataFrame(resultados)
    tabla.attrs["errores"] = errores
    return tabla

