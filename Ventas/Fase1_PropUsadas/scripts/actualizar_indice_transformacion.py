"""Sincroniza el índice del notebook sin ejecutar sus celdas ni tocar los datos."""

import argparse
import json
import re
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "2_Transformacion_PropUsadas.ipynb"


def actualizar_indice(notebook):
    finales = [c for c in notebook["cells"] if c.get("metadata", {}).get("resumen_final_transformacion")]
    if finales:
        if len(finales) != 2 or [c["cell_type"] for c in finales] != ["markdown", "code"]:
            raise ValueError("El resumen final debe tener un encabezado y una celda de generación.")
        restantes = [c for c in notebook["cells"] if not c.get("metadata", {}).get("resumen_final_transformacion")]
        numeros = [int(m.group(1)) for c in restantes if c["cell_type"] == "markdown"
                   for m in re.finditer(r"^## (\d+)\.", "".join(c["source"]), flags=re.M)]
        titulo = re.sub(r"^## \d+\.", f"## {max(numeros, default=0) + 1}.",
                        "".join(finales[0]["source"]), flags=re.M)
        finales[0]["source"] = titulo.splitlines(keepends=True)
        notebook["cells"] = [*restantes, *finales]
    indices = [c for c in notebook["cells"] if c.get("metadata", {}).get("indice_transformacion")]
    if len(indices) != 1:
        raise ValueError("Debe existir una única celda marcada como indice_transformacion.")
    enlaces = []
    for celda in notebook["cells"]:
        if celda["cell_type"] != "markdown" or celda is indices[0]:
            continue
        lineas = "".join(celda["source"]).splitlines(keepends=True)
        nuevas = []
        for linea in lineas:
            encabezado = re.match(r"^(#{2,6})\s+(.+?)\s*#*\s*$", linea.rstrip())
            if encabezado:
                nivel, titulo = encabezado.groups()
                anterior = next((s.strip() for s in reversed(nuevas) if s.strip()), "")
                ancla = re.fullmatch(r'<a id="([^"]+)"></a>', anterior)
                if ancla:
                    destino = ancla.group(1)
                else:
                    destino = f"seccion-{celda['id']}-{len(enlaces)}"
                    nuevas.append(f'<a id="{destino}"></a>\n\n')
                enlaces.append(f"{'  ' * (len(nivel) - 2)}- [{titulo}](#{destino})\n")
            nuevas.append(linea)
        celda["source"] = "".join(nuevas).splitlines(keepends=True)
    indices[0]["source"] = [
        '<a id="indice"></a>\n', "\n", "## Índice\n", "\n", *enlaces, "\n",
        "Actualizar después de agregar, eliminar, renombrar o mover secciones: "
        "`python Ventas/Fase1_PropUsadas/scripts/actualizar_indice_transformacion.py` "
        "desde la raíz del proyecto. `--check` verifica la sincronización sin escribir.\n",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    anterior = json.dumps(notebook, ensure_ascii=False)
    actualizar_indice(notebook)
    cambiado = anterior != json.dumps(notebook, ensure_ascii=False)
    if args.check:
        if cambiado:
            parser.exit(1, "Índice desactualizado: ejecutar el script sin --check.\n")
        print("Índice sincronizado.")
    elif cambiado:
        NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("Índice actualizado.")
    else:
        print("Índice sincronizado.")


if __name__ == "__main__":
    main()
