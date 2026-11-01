"""Pruebas del flujo acumulativo v2 y de aislamiento Train/Test."""
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import preparacion_train_test as pt
from scripts import resumen_transformacion as rt
from scripts.actualizar_indice_transformacion import actualizar_indice
from scripts.preparacion_train_test import (cargar_y_dividir, filtrar_outliers_tukey_train,
    EscaladoSelectivo, registro_escalado, crear_codificador_district)


def ejemplo():
    n = 16
    datos = pd.DataFrame({c: np.arange(n, dtype=float) + 1 for c in pt.NUMERICAS})
    for c in pt.CATEGORICAS:
        datos[c] = ["a", "b"] * 8
    datos["has_balcony"] = [False, True] * 8
    datos["metro_distance_type"] = ["walk", "walk", "transport", "transport"] * 4
    datos["seller_type"] = ["owner", "agency", "agency", "owner"] * 4
    datos["district"] = ["distrito_a", "distrito_b", "distrito_c", "distrito_a"] * 4
    datos["date_posted_dt"] = pd.date_range("2020-01-01", periods=n)
    datos["dataset_split"] = "train"
    datos["id"] = [f"id{i}" for i in range(n)]
    datos["total_area"] = np.arange(n, dtype=float) + 30
    datos["living_area"] = np.arange(n, dtype=float) + 12
    datos["kitchen_area"] = np.arange(n, dtype=float) / 10 + 5.
    datos.loc[0, "total_area"] = 10.  # Se debe corregir a 17.
    datos["building_year"] = np.arange(n, dtype=float) + 1980
    datos.loc[1, "building_year"] = 2025.  # Se marca e imputa antes de Tukey.
    datos["floor"] = 1.
    datos["total_floors"] = 10.
    datos["price_rub"] = [100.] * 15 + [10000.]
    datos["secondary_mom_change_pct"] = [np.nan] + list(np.arange(1, 15, dtype=float)) + [100.]
    for c in pt.EXCLUIDAS_MODELO:
        datos[c] = "valor descartado que no debe procesarse"
    return datos


def ajustar_ejemplo(datos):
    actual = pt.seleccionar_variables(datos)
    preparador = pt.PreparacionInicial()
    preparado = preparador.fit_transform(actual)
    actual, eliminados, resumen = pt.filtrar_outliers_tukey_train(preparado)
    base = pt.crear_codificador_base(preparador.columnas_numericas_)
    distrito = pt.crear_codificador_district()
    features = pt.integrar_caracteristicas(base.fit_transform(actual), distrito.fit_transform(actual[["district"]]))
    actual = pt.reemplazar_X(actual, features)
    extraccion = pt.ExtraccionTemporal()
    actual = extraccion.fit_transform(actual)
    filtro = pt.crear_filtro_final(pt.roles_candidatas(preparador, base, distrito, extraccion))
    actual = pt.reemplazar_X(actual, filtro.fit_transform(pt.obtener_X(actual)))
    escalador = pt.EscaladoSelectivo(["building_year"], [])
    actual = pt.reemplazar_X(actual, escalador.fit_transform(pt.obtener_X(actual)))
    objetos = {"preparacion": preparador, "base": base, "district": distrito,
               "extraccion": extraccion, "filtro": filtro, "escalado": escalador,
               "columnas_X": pt.obtener_X(actual).columns.tolist()}
    return actual, objetos, preparado, eliminados, resumen


class PreparacionTrainTest(unittest.TestCase):
    def test_exportacion_no_vuelve_a_transformar_datos_finales(self):
        train, test, crudo, futuro, objetos = self.preparar_entrega()
        with tempfile.TemporaryDirectory() as carpeta, tempfile.TemporaryDirectory() as reproducibilidad:
            with patch.object(pt, "aplicar_flujo", side_effect=AssertionError("No repetir transformaciones")):
                m = pt.exportar_entrega_final(train, test, crudo, futuro, objetos,
                    {"version_flujo": 3}, carpeta, carpeta_artefactos=reproducibilidad)
            self.assertTrue(m["exportacion_desde_dataframes_finales"])
            self.assertTrue(m["parametros_recuperados_sin_cambios"])
            for nombre, esperado in [("train", train), ("test", test)]:
                actual = pd.read_csv(Path(carpeta) / f"{nombre}_transformado.csv", dtype={"id": "string"},
                                     parse_dates=["date_posted_dt"], float_precision="round_trip")
                pd.testing.assert_frame_equal(actual, esperado.reset_index(drop=True), check_dtype=False)

    def test_compatibilidad_exportacion_con_lector_ols_real(self):
        from scripts import baseline_ols as b, contrato_datasets as cd
        train, test, crudo, futuro, objetos = self.preparar_entrega()
        with tempfile.TemporaryDirectory() as carpeta, tempfile.TemporaryDirectory() as reproducibilidad:
            data = Path(carpeta) / "data"
            m = pt.exportar_entrega_final(train, test, crudo, futuro, objetos,
                {"version_flujo": 3}, data, carpeta_artefactos=reproducibilidad)
            contrato = {"train": m["train"], "test": m["test"], "columnas_X": m["columnas_X"],
                        "manifiesto_sha256": "referencia de prueba"}
            with patch.object(cd, "CONTRATO", contrato), patch.object(b, "CONTRATO", contrato):
                tabla = pt.validar_entrega_para_ols(train, test, data)
                self.assertEqual(tabla.Estado.tolist(), ["Correcto", "Correcto"])
                self.assertEqual(tabla.attrs["errores"], [])
                # Una exportación que se haya saltado el escalado no coincide con el acumulado.
                incorrecto = train.copy()
                incorrecto["building_year"] += 1
                tabla = pt.validar_entrega_para_ols(incorrecto, test, data)
                self.assertEqual(tabla.Estado.tolist(), ["Error", "Correcto"])
                self.assertTrue(tabla.attrs["errores"])
                # Una cabecera duplicada no debe ser aceptada por la lectura de pandas.
                ruta = data / "test_transformado.csv"
                texto = ruta.read_text(encoding="utf-8")
                ruta.write_text(texto.replace("price_rub", "id", 1), encoding="utf-8")
                tabla = pt.validar_entrega_para_ols(train, test, data)
                self.assertEqual(tabla.Estado.tolist(), ["Correcto", "Error"])
                self.assertIn("duplicados", "; ".join(tabla.attrs["errores"]))

    def test_normalizador_convencion_central_y_valores_exactos(self):
        anteriores = ["num__total_area", "mom__secondary_mom_change_pct", "bin__has_balcony",
                      "cat__building_type_panel", "district__Alekseevsky",
                      "cat__metro_line_Kaluzhsko-Rizhskaya", "cat__month_of_year_10",
                      "num__Nueva ÁREA (%)-Test"]
        esperados = ["total_area", "secondary_mom_change_pct", "has_balcony", "building_type_panel",
                     "district_alekseevsky", "metro_line_kaluzhsko_rizhskaya", "month_of_year_10", "nueva_area_test"]
        entrada = pd.DataFrame(np.arange(24).reshape(3, 8), columns=anteriores, index=[4, 5, 8])
        normalizador = pt.NormalizadorNombres().fit(entrada)
        salida = normalizador.transform(entrada)
        self.assertEqual(salida.columns.tolist(), esperados)
        pd.testing.assert_frame_equal(salida.set_axis(anteriores, axis=1), entrada)
        self.assertEqual([pt.nombre_predictor(c) for c in esperados], esperados)
        self.assertEqual(normalizador.get_feature_names_out().tolist(), esperados)

    def test_normalizador_detiene_colisiones_y_reservados(self):
        for columnas in [["cat__x_A-B", "cat__x_A B"], ["num__rooms", "rooms"], ["num__price_rub"], ["__"]]:
            with self.assertRaises(ValueError):
                pt.NormalizadorNombres().fit(pd.DataFrame([[1, 2][:len(columnas)]], columns=columnas))

    def test_normalizador_serializado_reutiliza_esquema_y_roles(self):
        entrada = pd.DataFrame({"cat__a_A": [0., 1.], "num__x": [30., 50.]})
        roles = {"cat__a_A": "onehot", "num__x": "numerica"}
        normalizador = pt.NormalizadorNombres(roles).fit(entrada)
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "nombres.joblib"
            joblib.dump(normalizador, ruta)
            recuperado = joblib.load(ruta)
            estado = joblib.hash(recuperado)
            futuro = entrada * 2
            pd.testing.assert_frame_equal(recuperado.transform(futuro), futuro.rename(columns=normalizador.correspondencia_))
            self.assertEqual(pt.columnas_para_escala(recuperado.transform(futuro), recuperado.roles_salida_), ["x"])
            self.assertEqual(estado, joblib.hash(recuperado))
            with self.assertRaisesRegex(ValueError, "Cambió el esquema"):
                recuperado.transform(futuro.iloc[:, ::-1])

    def test_resumen_desde_entrega_cubre_esquema_sin_modificar_datos(self):
        crudo = ejemplo()
        train, objetos, preparado, eliminados, tukey = ajustar_ejemplo(crudo)
        futuro = crudo.iloc[:3].copy()
        futuro["id"] = ["f0", "f1", "f2"]
        futuro["date_posted_dt"] += pd.Timedelta(days=400)
        futuro["dataset_split"] = "test"
        test = pt.aplicar_flujo(futuro, objetos)[0]
        contrato = {"version_flujo": 2, "columnas_X": objetos["columnas_X"],
                    "entradas_seleccionadas": pt.ENTRADAS_X,
                    "momento_ajuste_imputacion": "Train antes de Tukey",
                    "auditoria_train": {"superficies_corregidas": int(preparado.surface_corrected.sum())},
                    "tratamiento_objetivo": tukey, "mapas_binarios": pt.MAPAS_BINARIOS,
                    "orden_train": ["seleccion", "correccion_imputacion", "tukey", "codificacion", "extraccion", "filtro_final", "escalado"]}
        with tempfile.TemporaryDirectory() as carpeta, tempfile.TemporaryDirectory() as reproducibilidad:
            m = pt.exportar_entrega_final(train, test, crudo, futuro, objetos, contrato, carpeta, carpeta_artefactos=reproducibilidad)
            resumen = rt.generar_resumen(carpeta, crudo, futuro, manifiesto=m, objetos=dict(objetos, configuracion=contrato))
            inventario = resumen["tablas"]["Inventario agrupado de predictoras"]
            self.assertEqual(inventario.Cantidad.sum(), len(objetos["columnas_X"]))
            self.assertEqual(list(resumen["linaje"]), objetos["columnas_X"])
            self.assertEqual(len(inventario[inventario["Variable original"] == "district"]), 1)
            extraidas = resumen["tablas"]["Feature Extraction: derivadas presentes (no One-Hot)"]
            self.assertTrue(set(extraidas.Característica).issubset(train))
            self.assertFalse(any(resumen["linaje"][c]["tipo"] == "onehot" for c in extraidas.Característica))
            for nombre, huella in m["archivos_sha256"].items():
                self.assertEqual(pt.huella_archivo(Path(carpeta if nombre.endswith(".csv") else reproducibilidad) / nombre), huella)
            rt.guardar_resumen(resumen, Path(reproducibilidad) / "resumen_final.json")
            ruta = Path(carpeta) / "train_transformado.csv"
            ruta.write_text(ruta.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Entrega inconsistente"):
                rt.generar_resumen(carpeta, crudo, futuro, manifiesto=m, objetos=dict(objetos, configuracion=contrato))

    def test_grupos_ohe_proceden_de_encoder_no_de_prefijos(self):
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(drop="first", sparse_output=False).fit(
            pd.DataFrame({"a": ["x", "x_1", "y"], "a_x": ["1", "2", "3"]}))
        salidas = encoder.get_feature_names_out().tolist()
        grupos = rt.grupos_onehot(encoder, encoder.feature_names_in_.tolist(), salidas)
        self.assertEqual(grupos[0]["columnas"], salidas[:2])
        self.assertEqual(grupos[1]["columnas"], salidas[2:])
        self.assertEqual(grupos[0]["referencia"], "x")

    def test_indice_reubica_y_renumera_resumen_al_final(self):
        notebook = {"cells": [
            {"id": "indice", "cell_type": "markdown", "metadata": {"indice_transformacion": True}, "source": []},
            {"id": "resumen", "cell_type": "markdown", "metadata": {"resumen_final_transformacion": True}, "source": ["## 9. Resumen final\n"]},
            {"id": "resumen-code", "cell_type": "code", "metadata": {"resumen_final_transformacion": True}, "source": []},
            {"id": "nueva", "cell_type": "markdown", "metadata": {}, "source": ["## 10. Nueva operación\n"]},
        ]}
        actualizar_indice(notebook)
        self.assertEqual(notebook["cells"][-1]["id"], "resumen-code")
        self.assertIn("## 11. Resumen final", "".join(notebook["cells"][-2]["source"]))
        self.assertLess("".join(notebook["cells"][0]["source"]).index("10. Nueva"),
                        "".join(notebook["cells"][0]["source"]).index("11. Resumen"))

    def preparar_entrega(self):
        crudo = ejemplo()
        train, objetos, *_ = ajustar_ejemplo(crudo)
        futuro = crudo.iloc[:3].copy()
        futuro["id"] = ["test0", "test1", "test2"]
        futuro["date_posted_dt"] += pd.Timedelta(days=400)
        futuro["dataset_split"] = "test"
        futuro["price_rub"] = [100., 1e12, 200.]
        test = pt.aplicar_flujo(futuro, objetos)[0]
        return train, test, crudo, futuro, objetos

    def test_entrega_csv_separados_y_paquete_recuperable(self):
        train, test, crudo, futuro, objetos = self.preparar_entrega()
        with tempfile.TemporaryDirectory() as carpeta, tempfile.TemporaryDirectory() as reproducibilidad:
            manifiesto = pt.exportar_entrega_final(train, test, crudo, futuro, objetos,
                                                   {"version_flujo": 2}, carpeta, carpeta_artefactos=reproducibilidad)
            for nombre, huella in manifiesto["archivos_sha256"].items():
                self.assertEqual(pt.huella_archivo(Path(carpeta if nombre.endswith(".csv") else reproducibilidad) / nombre), huella)
            self.assertEqual({p.suffix for p in Path(carpeta).iterdir()}, {".csv"})
            recuperados = joblib.load(Path(reproducibilidad) / "transformadores_prop_usadas.joblib")
            pd.testing.assert_frame_equal(pt.aplicar_flujo(futuro, recuperados)[0], test)
            archivos = [pd.read_csv(Path(carpeta) / f"{c}_transformado.csv") for c in ["train", "test"]]
            self.assertEqual(archivos[0].columns.tolist(), archivos[1].columns.tolist())
            self.assertFalse(any(c.startswith("Unnamed:") for c in archivos[0]))
            self.assertEqual(archivos[1].price_rub.max(), 1e12)
            self.assertEqual(manifiesto["test"]["filas"], len(futuro))
            self.assertLess(manifiesto["train"]["filas"], len(crudo))
            regenerado = pt.exportar_entrega_final(train, test, crudo, futuro, objetos,
                                                   {"version_flujo": 2}, carpeta, carpeta_artefactos=reproducibilidad)
            self.assertEqual(manifiesto, regenerado)

    def test_entrega_rechaza_test_recortado_o_reordenado(self):
        train, test, crudo, futuro, objetos = self.preparar_entrega()
        with tempfile.TemporaryDirectory() as carpeta, tempfile.TemporaryDirectory() as reproducibilidad:
            for incorrecto in [test.iloc[:-1], test.iloc[::-1]]:
                with self.assertRaises(AssertionError):
                    pt.exportar_entrega_final(train, incorrecto, crudo, futuro, objetos,
                                               {"version_flujo": 2}, carpeta, carpeta_artefactos=reproducibilidad)
            self.assertEqual(list(Path(carpeta).iterdir()), [])

    def test_escala_descubre_solo_numericas_predictoras_actuales(self):
        matriz = pd.DataFrame({"total_area": [40., 60.], "secondary_mom_change_pct": [1., 2.],
            "surface_corrected": [0., 1.], "missingindicator_building_year": [0., 1.],
            "seller_type": [0., 1.], "building_type_brick": [0., 1.],
            "district_a": [1., 0.], "price_rub": [10., 20.], "id": [1, 2]})
        roles = {c: "binaria" for c in matriz}
        roles.update(dict.fromkeys(["total_area", "secondary_mom_change_pct"], "numerica"))
        self.assertEqual(pt.columnas_para_escala(matriz, roles), ["total_area", "secondary_mom_change_pct"])
        self.assertEqual(pt.columnas_para_escala(matriz.drop(columns="total_area"), roles), ["secondary_mom_change_pct"])

    def test_verificacion_de_escala_rechaza_otro_estado_o_variables(self):
        matriz = pd.DataFrame({"x": [1., 2., 3.], "b": [0., 1., 0.]})
        roles = {"x": "numerica", "b": "binaria"}
        huella = joblib.hash(matriz)
        pt.verificar_estado_analizado(matriz, huella, ["x"], roles)
        with self.assertRaisesRegex(ValueError, "DataFrame cambió"):
            pt.verificar_estado_analizado(matriz.iloc[::-1], huella, ["x"], roles)
        with self.assertRaisesRegex(ValueError, "no coinciden"):
            pt.verificar_estado_analizado(matriz, huella, [], roles)
        escalado = pt.EscaladoSelectivo(["x"], []).fit_transform(matriz)
        with self.assertRaisesRegex(ValueError, "escalar dos veces"):
            pt.verificar_estado_analizado(escalado, huella, ["x"], roles)

    def test_parametros_escalador_registran_exactamente_su_train(self):
        matriz = pd.DataFrame({"num__year": [1980., 1990., 2000.], "bin__b": [0., 1., 0.]})
        escalador = pt.EscaladoSelectivo(["num__year"], []).fit(matriz)
        registro = pt.registro_escalado(escalador)
        self.assertEqual(registro["huella_entrada_train"], joblib.hash(matriz))
        self.assertEqual(registro["filas_ajuste"], len(matriz))
        self.assertEqual(escalador.estandarizador_.feature_names_in_.tolist(), ["num__year"])
        self.assertEqual(registro["media_train"]["num__year"], matriz.num__year.mean())

    def test_escalado_selectivo_estandariza_solo_columnas_elegidas(self):
        train = pd.DataFrame({"year": [1980., 1990., 2000., 2010.],
                              "distance": [0., 10., 20., 30.], "dummy": [0., 1., 0., 1.]})
        escalador = EscaladoSelectivo(["year"], [])
        resultado = escalador.fit_transform(train)
        self.assertAlmostEqual(resultado.year.mean(), 0.)
        self.assertAlmostEqual(resultado.year.std(ddof=0), 1.)
        pd.testing.assert_frame_equal(resultado[["distance", "dummy"]], train[["distance", "dummy"]])
        self.assertEqual(resultado.columns.tolist(), train.columns.tolist())
        registro = registro_escalado(escalador)
        self.assertEqual(registro["media_train"]["year"], 1995.)
        futuro = train.iloc[:2].copy()
        futuro["year"] = [2020., 2100.]
        estado = joblib.hash(escalador)
        transformado = escalador.transform(futuro)
        np.testing.assert_allclose(transformado.year, (futuro.year - 1995.) / train.year.std(ddof=0))
        self.assertEqual(estado, joblib.hash(escalador))
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "escalador.joblib"
            joblib.dump(escalador, archivo)
            pd.testing.assert_frame_equal(joblib.load(archivo).transform(futuro), transformado)


    def test_minmax_reutiliza_extremos_train_sin_recortar_futuro(self):
        train = pd.DataFrame({"x": [0., 10., 20.], "dummy": [0., 1., 0.]})
        escalador = EscaladoSelectivo([], ["x"])
        resultado = escalador.fit_transform(train)
        np.testing.assert_array_equal(resultado.x, [0., .5, 1.])
        futuro = pd.DataFrame({"x": [-10., 40.], "dummy": [1., 0.]}, index=[10, 20])
        estado = joblib.hash(escalador)
        transformado = escalador.transform(futuro)
        np.testing.assert_array_equal(transformado.x, [-.5, 2.])
        pd.testing.assert_series_equal(transformado.dummy, futuro.dummy)
        self.assertEqual(estado, joblib.hash(escalador))
        self.assertEqual(registro_escalado(escalador)["minimo_train"]["x"], 0.)


    def test_escalado_rechaza_doble_transformacion_y_esquema_distinto(self):
        datos = pd.DataFrame({"x": [1., 2.], "y": [3., 4.]})
        with self.assertRaisesRegex(ValueError, "dos escalados"):
            EscaladoSelectivo(["x"], ["x"]).fit(datos)
        escalador = EscaladoSelectivo(["x"], []).fit(datos)
        with self.assertRaisesRegex(ValueError, "esquema"):
            escalador.transform(datos[["y", "x"]])


    def test_tukey_elimina_filas_completas_y_conserva_fronteras(self):
        datos = pd.DataFrame({"id": [f"p{i}" for i in range(12)],
                              "price_rub": [-16., -15., 0., 0., 0., 0., 10., 10., 10., 10., 25., 26.],
                              "predictora": np.arange(12), "dataset_split": "train"},
                             index=np.arange(12) * 3)
        original = datos.copy(deep=True)
        filtrado, eliminados, registro = filtrar_outliers_tukey_train(datos)
        self.assertEqual(registro["q1"], 0.)
        self.assertEqual(registro["q3"], 10.)
        self.assertEqual(registro["iqr"], 10.)
        self.assertEqual(registro["limite_inferior"], -15.)
        self.assertEqual(registro["limite_superior"], 25.)
        self.assertEqual(eliminados.id.tolist(), ["p0", "p11"])
        self.assertTrue({"p1", "p10"}.issubset(filtrado.id))
        self.assertEqual(registro["filas_finales"], 10)
        self.assertAlmostEqual(registro["porcentaje_eliminado"], 100 * 2 / 12)
        pd.testing.assert_frame_equal(filtrado, original.loc[filtrado.index])
        pd.testing.assert_frame_equal(datos, original)


    def test_tukey_rechaza_test_y_precios_no_finitos(self):
        datos = pd.DataFrame({"price_rub": [1., 2., 3.], "dataset_split": "test"})
        with self.assertRaisesRegex(ValueError, "exclusivamente de entrenamiento"):
            filtrar_outliers_tukey_train(datos)
        datos["dataset_split"] = "train"
        datos.loc[0, "price_rub"] = np.nan
        with self.assertRaisesRegex(ValueError, "finito y no nulo"):
            filtrar_outliers_tukey_train(datos)


    def test_tukey_iqr_cero_preserva_todos_los_precios_iguales(self):
        datos = pd.DataFrame({"price_rub": [100.] * 8, "dataset_split": "train"})
        filtrado, eliminados, registro = filtrar_outliers_tukey_train(datos)
        pd.testing.assert_frame_equal(filtrado, datos)
        self.assertTrue(eliminados.empty)
        self.assertEqual(registro["iqr"], 0.)
        self.assertEqual(registro["porcentaje_eliminado"], 0.)


    def test_district_onehot_train_y_futuro_sin_reajuste(self):
        train = pd.DataFrame({"district": [" A ", "B", "C", "A"]}, index=[4, 8, 12, 16])
        original = train.copy(deep=True)
        codificador = crear_codificador_district()
        resultado = codificador.fit_transform(train)
        self.assertEqual(resultado.columns.tolist(), ["district__B", "district__C"])
        np.testing.assert_array_equal(resultado.to_numpy(), [[0, 0], [1, 0], [0, 1], [0, 0]])
        futuro = pd.DataFrame({"district": ["D", "B", None]}, index=[20, 24, 28])
        estado = joblib.hash(codificador)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            transformado = codificador.transform(futuro)
        self.assertEqual(transformado.columns.tolist(), resultado.columns.tolist())
        np.testing.assert_array_equal(transformado.to_numpy(), [[0, 0], [1, 0], [0, 0]])
        self.assertEqual(estado, joblib.hash(codificador))
        self.assertNotIn("D", codificador.named_steps["onehot"].categories_[0])
        pd.testing.assert_frame_equal(train, original)
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "district.joblib"
            joblib.dump(codificador, archivo)
            pd.testing.assert_frame_equal(joblib.load(archivo).transform(train), resultado)


    def test_corte_por_dia_completo_y_empate_anterior(self):
        datos = pd.DataFrame({"id": [f"i{i}" for i in range(10)],
                              "date_posted_dt": ["2020-01-01"] * 7 + ["2020-01-02"] * 2 + ["2020-01-03"]})
        with tempfile.TemporaryDirectory() as carpeta:
            entrada = Path(carpeta) / "fuente.csv"
            datos.iloc[::-1].to_csv(entrada, index=False)
            train, test, resumen, contrato = cargar_y_dividir(entrada, Path(carpeta) / "salida")
            self.assertEqual(len(train), 7)
            self.assertEqual(len(test), 3)
            self.assertLess(train.date_posted_dt.max(), test.date_posted_dt.min())
            repetidos = cargar_y_dividir(entrada, Path(carpeta) / "salida")
            pd.testing.assert_frame_equal(train, repetidos[0])
            self.assertEqual(contrato["ultima_fecha_train"], "2020-01-01")


    def test_fuente_modificada_no_redefine_particion(self):
        with tempfile.TemporaryDirectory() as carpeta:
            entrada = Path(carpeta) / "fuente.csv"
            datos = ejemplo()
            datos.to_csv(entrada, index=False)
            cargar_y_dividir(entrada, Path(carpeta) / "salida")
            datos.loc[0, "price_rub"] = 12345.
            datos.to_csv(entrada, index=False)
            with self.assertRaisesRegex(ValueError, "No se redefine"):
                cargar_y_dividir(entrada, Path(carpeta) / "salida")


    def test_una_sola_fecha_no_simula_separacion_temporal(self):
        with tempfile.TemporaryDirectory() as carpeta:
            entrada = Path(carpeta) / "fuente.csv"
            pd.DataFrame({"id": ["a", "b"], "date_posted_dt": ["2020-01-01"] * 2}).to_csv(entrada, index=False)
            with self.assertRaisesRegex(ValueError, "dos fechas distintas"):
                cargar_y_dividir(entrada, Path(carpeta) / "salida")


    def test_seleccion_excluye_newbuild_y_conserva_indicador_hipotecario(self):
        datos = ejemplo().rename(columns={"mortgage_key_rate_at_listing": "mortgage_rate_at_listing"})
        seleccionado = pt.seleccionar_variables(datos)
        self.assertEqual(pt.CAMBIOS_MENSUALES, ["secondary_mom_change_pct"])
        self.assertFalse(set(pt.EXCLUIDAS_MODELO).intersection(seleccionado))
        np.testing.assert_array_equal(seleccionado.mortgage_key_rate_at_listing, datos.mortgage_rate_at_listing)
        self.assertIn("district", seleccionado)

    def test_imputacion_antes_tukey_se_acumula_hasta_el_final(self):
        datos = ejemplo()
        final, objetos, preparado, eliminados, resumen = ajustar_ejemplo(datos)
        media_completa = datos.secondary_mom_change_pct.mean()
        self.assertEqual(objetos["preparacion"].mensual_.statistics_[0], media_completa)
        self.assertEqual(preparado.loc[0, "secondary_mom_change_pct"], media_completa)
        self.assertEqual(final.loc[0, "secondary_mom_change_pct"], media_completa)
        self.assertEqual(final.loc[0, "total_area"], 17.)
        self.assertEqual(final.loc[0, "living_area"], 12.)
        self.assertEqual(final.loc[0, "kitchen_area"], 5.)
        self.assertEqual(eliminados.id.tolist(), ["id15"])
        self.assertNotIn(15, final.index)
        self.assertEqual(len(final), resumen["filas_finales"])
        pd.testing.assert_series_equal(final.price_rub, datos.loc[final.index, "price_rub"])
        self.assertFalse(any("newbuild_mom_change_pct" in c for c in final))
        self.assertTrue(any(c.startswith("district_") for c in final))
        pt.validar_superficies(final)

    def test_auditoria_compara_resultados_sin_volver_a_transformar(self):
        datos = pt.seleccionar_variables(ejemplo())
        preparador = pt.PreparacionInicial()
        preparado = preparador.fit_transform(datos)
        resumen, auditoria = pt.auditar_preparacion(datos, preparado, preparador)
        self.assertEqual(resumen["superficies_corregidas"], 1)
        self.assertEqual(resumen["imputaciones_mom"]["secondary_mom_change_pct"]["imputados"], 1)
        self.assertEqual(auditoria.variable.tolist(), ["total_area", "secondary_mom_change_pct"])
        self.assertFalse(any("newbuild" in c for c in resumen["imputaciones_mom"]))

    def test_futuro_conserva_todas_filas_y_no_ajusta_ningun_objeto(self):
        datos = ejemplo()
        final, objetos, preparado, eliminados, resumen = ajustar_ejemplo(datos)
        test = ejemplo().iloc[:3].copy()
        test["dataset_split"] = "test"
        test["date_posted_dt"] = pd.date_range("2021-02-01", periods=3)
        test["price_rub"] = [1e10, 1., 1e9]
        test["secondary_mom_change_pct"] = [np.nan, 500., -500.]
        test["district"] = "nuevo"
        test["seller_type"] = "vendedor_nuevo"
        estado = joblib.hash(objetos)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            aplicado, calidad, auditoria, trazabilidad = pt.aplicar_flujo(test, objetos)
        self.assertEqual(joblib.hash(objetos), estado)
        self.assertEqual(len(aplicado), len(test))
        self.assertEqual(pt.obtener_X(aplicado).columns.tolist(), objetos["columnas_X"])
        pd.testing.assert_series_equal(aplicado.price_rub, test.price_rub)
        self.assertEqual(aplicado.loc[0, "secondary_mom_change_pct"], objetos["preparacion"].mensual_.statistics_[0])
        self.assertEqual(trazabilidad.filas_eliminadas.sum(), 0)
        self.assertNotIn("tukey", trazabilidad.etapa.tolist())
        self.assertTrue(aplicado[[c for c in pt.BINARIAS if c in aplicado]].isin([0, 1]).all().all())

    def test_serializacion_reproduce_flujo_completo(self):
        datos = ejemplo()
        final, objetos, preparado, eliminados, resumen = ajustar_ejemplo(datos)
        with tempfile.TemporaryDirectory() as carpeta:
            archivo = Path(carpeta) / "flujo.joblib"
            joblib.dump(objetos, archivo)
            recargados = joblib.load(archivo)
            entrada = datos.loc[final.index]
            esperado = pt.aplicar_flujo(entrada, objetos)[0]
            actual = pt.aplicar_flujo(entrada, recargados)[0]
            pd.testing.assert_frame_equal(actual, final)
            pd.testing.assert_frame_equal(actual, esperado)

    def test_trazabilidad_detiene_reordenamientos_y_reincorporaciones(self):
        datos = pt.seleccionar_variables(ejemplo())
        with self.assertRaisesRegex(ValueError, "reordenaron"):
            pt.registrar_etapa("incorrecta", datos, datos.iloc[::-1], [])
        filtrados = datos.iloc[:3]
        with self.assertRaisesRegex(ValueError, "reincorporaron"):
            pt.registrar_etapa("incorrecta", filtrados, datos, [])
        rotos = filtrados.copy()
        rotos.loc[0, "price_rub"] = 999.
        with self.assertRaises(AssertionError):
            pt.registrar_etapa("incorrecta", filtrados, rotos, [])

    def test_control_rechaza_descartadas_prefijadas(self):
        for nombre in ["newbuild_mom_change_pct", "mom__newbuild_mom_change_pct", "num__lat"]:
            with self.assertRaisesRegex(ValueError, "excluida"):
                pt.verificar_excluidas(pd.DataFrame({nombre: [1.]}))

    def test_faltantes_sin_datos_train_no_inventan_media(self):
        datos = pt.seleccionar_variables(ejemplo())
        datos["secondary_mom_change_pct"] = np.nan
        with self.assertRaisesRegex(ValueError, "estimar la media"):
            pt.PreparacionInicial().fit(datos)

    def test_conflicto_alias_hipotecario_detiene_la_seleccion(self):
        datos = ejemplo()
        datos["mortgage_rate_at_listing"] = 12345.
        with self.assertRaisesRegex(ValueError, "alias hipotecarios"):
            pt.seleccionar_variables(datos)


if __name__ == "__main__":
    unittest.main()
