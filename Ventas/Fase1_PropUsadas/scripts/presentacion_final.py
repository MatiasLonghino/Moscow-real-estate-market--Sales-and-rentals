"""Comparación final de resultados existentes: presentación pura, sin ajuste ni selección."""
import html
import json

import numpy as np
import pandas as pd

from . import baseline_ols as b

NOMBRES={'forward':'Forward Selection — 177','backward':'Backward Selection — 177',
         'lasso':'Group Lasso + OLS — 177','bss':'BSS histórico','temporal':'BSS temporal — modelo final'}
INTERPRETACIONES={'total_area':'Superficie total de la propiedad',
    'metro_distance_type':'Modalidad de acceso al metro','building_type':'Tipo de construcción del edificio',
    'renovation':'Estado de renovación','district':'Distrito de Moscú',
    'date_posted_dt':'Evolución temporal, representada por months_since_start',
    'rooms':'Cantidad de habitaciones','ceiling_height':'Altura del techo',
    'metro_distance_min':'Tiempo de acceso al metro','has_balcony':'Disponibilidad de balcón',
    'surface_corrected':'Indicador de corrección de superficie','to_center_km':'Distancia al centro',
    'living_area':'Superficie habitable'}


def rub(valor): return f'{float(valor):,.0f}'.replace(',','.')


def preparar(forward,backward,lasso,bss,temporal):
    modelos=dict(forward=forward,backward=backward,lasso=lasso,bss=bss,temporal=temporal)
    universo=list(bss['universo']['union'])
    assert len(universo)==len(set(universo))
    selecciones={k:list(r['seleccion']['seleccionadas']) for k,r in modelos.items() if k!='temporal'}
    res=temporal['resultados']
    selecciones['temporal']=list(res['selected_features'])
    for seleccion in selecciones.values():
        assert len(seleccion)==len(set(seleccion)) and set(seleccion)<=set(universo)
    assert set(universo)==set.union(*(set(selecciones[k]) for k in ('forward','backward','lasso')))
    filas=res['rows']; regla=res['rule']
    assert [r['k'] for r in filas]==list(range(1,13))
    menor=min(filas,key=lambda r:r['mean'])
    assert menor['k']==regla['k_min']
    np.testing.assert_allclose(regla['threshold'],menor['mean']+menor['se'],rtol=1e-12)
    assert min(r['k'] for r in filas if r['mean']<=regla['threshold'])==regla['k_selected']
    assert len(selecciones['temporal'])==regla['k_selected']
    columnas=[c for g in selecciones['temporal'] for c in temporal['configuracion']['groups'][g]]
    assert columnas==res['columns'] and len(columnas)==len(set(columnas))
    elegido=next(r for r in filas if r['k']==regla['k_selected'])
    np.testing.assert_allclose(elegido['mean'],regla['selected_mean'])
    return dict(modelos=modelos,universo=universo,selecciones=selecciones,menor=menor,elegido=elegido)


def matriz_html(datos):
    selecciones=datos['selecciones']; claves=list(NOMBRES)
    conjuntos={k:set(v) for k,v in selecciones.items()}
    comunes=set.intersection(*(conjuntos[k] for k in ('forward','backward','lasso')))
    def orden(g):
        cuenta=sum(g in conjuntos[k] for k in ('forward','backward','lasso'))
        return (-cuenta,-int(g in conjuntos['forward']),g)
    filas=sorted(datos['universo'],key=orden)
    cabeceras=['Forward Selection<br>177','Backward Selection<br>177','Group Lasso + OLS<br>177',
               'BSS<br>histórico','BSS temporal<br><strong>modelo final</strong>']
    estilos=('style="border-collapse:collapse;width:100%;font-size:14px;line-height:1.4"')
    partes=[f'<table {estilos}><thead><tr><th style="text-align:left;padding:9px">Característica original</th>']
    for i,titulo in enumerate(cabeceras):
        color='background:#DDEEE7;border-left:2px solid #176B70;' if i==4 else 'background:#F3F5F4;'
        partes.append(f'<th style="padding:9px;text-align:center;{color}">{titulo}</th>')
    partes.append('</tr></thead><tbody>')
    separado=False
    for g in filas:
        separador=''
        if g not in comunes and not separado:
            separador='border-top:2px solid #93A99F;'; separado=True
        color='background:#EAF4EE;' if g in conjuntos['temporal'] else ''
        partes.append(f'<tr><td style="padding:6px 9px;{separador}{color}"><code>{html.escape(g)}</code></td>')
        for k in claves:
            marca='✓' if g in conjuntos[k] else '—'
            estilo=separador+'padding:6px;text-align:center;'
            if k=='temporal':
                estilo+='background:#DDEEE7;border-left:2px solid #176B70;color:#155749;'
                if marca=='✓': marca='<strong>✓</strong>'
            partes.append(f'<td style="{estilo}">{marca}</td>')
        partes.append('</tr>')
    partes.append('</tbody><tfoot><tr><th style="padding:9px;text-align:left;border-top:2px solid #93A99F">Total</th>')
    for k in claves:
        estilo='padding:9px;border-top:2px solid #93A99F;'
        if k=='temporal': estilo+='background:#DDEEE7;border-left:2px solid #176B70;'
        partes.append(f'<th style="{estilo}">{len(selecciones[k])}</th>')
    partes.append('</tr></tfoot></table>')
    return ''.join(partes)


def observaciones(datos):
    s=datos['selecciones']
    minimo=min(len(s[k]) for k in ('forward','backward','lasso'))
    selectivos=', '.join(NOMBRES[k] for k in ('forward','backward','lasso') if len(s[k])==minimo)
    iguales=set(s['backward'])==set(s['bss'])
    cambio='redujo la selección final a' if len(s['temporal'])<len(s['bss']) else 'definió una selección final de'
    return (f'- {selectivos} fue el método más selectivo entre los tres algoritmos originales ({minimo} características).\n'
        + (f"- Backward y BSS histórico terminaron con exactamente las mismas {len(s['bss'])} características.\n" if iguales else
           '- Backward y BSS histórico conservaron subconjuntos diferentes.\n')
        + f"- La validación temporal {cambio} {len(s['temporal'])} características originales.")


def tabla_historica(datos):
    filas=[]
    for k in ('forward','backward','lasso','bss'):
        r=datos['modelos'][k]
        filas.append({'Modelo':NOMBRES[k],'Características finales':len(datos['selecciones'][k]),
            'Columnas OLS':int(r['metricas']['train_n_predictors']),
            'RMSE interno (RUB)':rub(r['metricas']['selection_rmse_validation_original'])})
    return b.tabla_markdown(pd.DataFrame(filas))


def tabla_temporal(datos):
    a,e=datos['menor'],datos['elegido']
    return b.tabla_markdown(pd.DataFrame([
        {'Criterio temporal':'Menor RMSE','Características':str(a['k']),'RMSE promedio temporal (RUB)':rub(a['mean'])},
        {'Criterio temporal':'**Regla 1-SE — elegido**','Características':f"**{e['k']}**",
         'RMSE promedio temporal (RUB)':f"**{rub(e['mean'])}**"}]))


def interpretar_comparaciones(datos):
    s=datos['selecciones']; m=datos['modelos']
    iguales=set(s['backward'])==set(s['bss']) and np.isclose(
        m['backward']['metricas']['selection_rmse_validation_original'],m['bss']['metricas']['selection_rmse_validation_original'])
    texto=('BSS histórico alcanzó el mismo resultado que Backward. Para el criterio y el universo considerados, '
        'la búsqueda exhaustiva no mejoró el subconjunto obtenido mediante eliminación progresiva. ' if iguales else '')
    return (texto+'Los RMSE históricos provienen de la validación interna utilizada en la selección y no de Test.\n\n'
        'El BSS temporal utilizó tres folds cronológicos y Train crudo, con preparación dentro de cada ventana. '
        'Las dos tablas corresponden a poblaciones y procedimientos de evaluación diferentes: sus RMSE no son directamente comparables. '
        'La primera resume el recorrido histórico; la segunda fundamenta la complejidad del modelo definitivo.')


def modelo_elegido(datos):
    t=datos['modelos']['temporal']; r=t['resultados']; regla=r['rule']
    seleccion=datos['selecciones']['temporal']; columnas=len(r['columns'])
    filas=[{'Característica final':f'`{g}`','Interpretación':INTERPRETACIONES.get(g,'Característica original del esquema')} for g in seleccion]
    tabla=b.tabla_markdown(pd.DataFrame(filas))
    resumen=(f"**Modelo final: {len(seleccion)} características originales, {columnas} columnas transformadas más intercepto "
        f"y RMSE temporal promedio de {rub(regla['selected_mean'])} RUB.**")
    if 'district' in seleccion:
        resumen+=f" La característica `district` aporta {len(t['configuracion']['groups']['district'])} columnas One-Hot."
    justificacion=(f"El modelo de {regla['k_min']} características obtuvo el menor RMSE temporal promedio "
        f"({rub(datos['menor']['mean'])} RUB). La regla 1-SE fija el margen en ese mínimo más su error estándar "
        f"({rub(datos['menor']['se'])} RUB), con un umbral de {rub(regla['threshold'])} RUB. "
        f"El menor tamaño dentro de ese margen es {regla['k_selected']}. La elección aplica el criterio de parsimonia "
        'definido previamente y acepta un error temporal algo mayor a cambio de menos grupos.\n\n'
        'La validación temporal orienta la selección hacia publicaciones posteriores. Con tres folds, 1-SE es una '
        'regla práctica: no garantiza eliminar el overfitting ni que este modelo sea superior en Test.')
    return tabla,resumen,justificacion


def conclusion(datos):
    final=len(datos['selecciones']['temporal']); historico=len(datos['selecciones']['bss'])
    return ('El EDA y las transformaciones permitieron preparar las variables y reconocer sus relaciones con el precio. '
        'OLS, Forward, Backward y Group Lasso aportaron distintas estrategias de modelado y selección. '
        f'BSS exploró exhaustivamente la unión y obtuvo un modelo histórico de {historico} características. '
        f'La validación temporal y la regla 1-SE permitieron elegir finalmente un **OLS de {final} características originales**, '
        'priorizando el equilibrio entre error temporal y complejidad por grupos. '
        'La evidencia sobre períodos posteriores orienta esta decisión; la evaluación final independiente sobre Test sigue pendiente.')


def anexos(datos,tabla_historica_completa,coincidencias_historicas):
    refs={NOMBRES[k]:r.get('run',{}) for k,r in datos['modelos'].items()}
    return ('<details><summary>Anexo: resultados históricos completos y referencias</summary>\n\n'
        'Esta comparación ampliada conserva los resultados históricos y no define el modelo final temporal.\n\n'
        'R² ajustado, AIC y BIC corresponden al objetivo logarítmico sobre las mismas filas históricas de Train. '
        'Los RMSE distinguen Train de validación interna. Los experimentos de 174 columnas conservan sus exclusiones históricas; '
        'los de 177 utilizan el universo ampliado. BSS explora la unión derivada de las selecciones anteriores, '
        'que ya consultaron la validación interna. La comparación es descriptiva y puede presentar optimismo por selección. '
        'La validación temporal mantiene fija esa unión elegida en Train; tampoco constituye una evaluación independiente sobre Test.\n\n'
        +tabla_historica_completa+'\n\n'+coincidencias_historicas+'\n\n'
        '<details><summary>Referencias de MLflow</summary>\n\n```json\n'
        +json.dumps(refs,ensure_ascii=False,indent=2)+'\n```\n\n</details>\n\n</details>')
