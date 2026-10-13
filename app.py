import io

import altair as alt
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="Precio de la gasolina", page_icon="⛽", layout="wide")

# ---------------------------------------------------------------- cargar modelo
@st.cache_resource
def cargar_paquete():
    return joblib.load("modelo_gasolina.pkl")

try:
    paq = cargar_paquete()
except FileNotFoundError:
    st.error("No encuentro 'modelo_gasolina.pkl'. Ejecuta primero TODO el notebook "
             "(hasta el Paso 11) y deja ese archivo en la misma carpeta que app.py.")
    st.stop()

modelo = paq["modelo"]
scaler = paq["scaler"]
variables = paq["variables"]
rangos = paq["rangos"]
metricas = paq["metricas"]
n_train = paq.get("n_train", int(modelo.nobs))
n_test = paq.get("n_test")
medianas = paq.get("medianas", {v: (rangos[v][0] + rangos[v][1]) / 2 for v in rangos})

ETIQUETAS = {
    "Precio_Petroleo_USD_Barril": "Precio del petróleo (USD/barril)",
    "Tipo_Cambio_USD_COP": "Tipo de cambio (COP por USD)",
    "Demanda_miles_barriles_dia": "Demanda (miles de barriles/día)",
    "Produccion_OPEP_millones_dia": "Producción OPEP (millones/día)",
    "Inflacion_Porcentaje": "Inflación (%)",
    "PIB_Porcentaje": "Crecimiento del PIB (%)",
    "Eventos_Geopoliticos": "¿Evento geopolítico relevante?",
    "Impuestos_Combustibles_Porcentaje": "Impuestos a combustibles (%)",
}
ORDEN = list(ETIQUETAS.keys())
variables_orden = [v for v in ORDEN if v in variables]


def predecir(df_entrada, nivel):
    """Recibe un DataFrame con las columnas de `variables` y devuelve predicción + intervalo."""
    X = pd.DataFrame(scaler.transform(df_entrada[variables]), columns=variables)
    Xc = sm.add_constant(X, has_constant="add")
    res = modelo.get_prediction(Xc).summary_frame(alpha=1 - nivel)
    return pd.DataFrame({
        "Precio_predicho": res["mean"].values,
        "Minimo": res["obs_ci_lower"].values,
        "Maximo": res["obs_ci_upper"].values,
    })


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Sobre el modelo")
    if n_test:
        st.write(f"Entrenado con **{n_train}** meses y probado con **{n_test}**.")
    else:
        st.write(f"Entrenado con **{n_train}** meses.")
    st.metric("R² (prueba)", f"{metricas['R²']:.3f}")
    st.metric("RMSE (prueba)", f"{metricas['RMSE']:.3f} USD/galón")
    st.metric("MAE (prueba)", f"{metricas['MAE']:.3f} USD/galón")
    nivel = st.selectbox("Nivel de confianza", [0.90, 0.95, 0.99], index=1,
                         format_func=lambda x: f"{int(round(x * 100))}%")
    st.caption("Variables usadas: " + ", ".join(ETIQUETAS[v] for v in variables_orden))

# ---------------------------------------------------------------- encabezado
st.title("⛽ Predicción del precio mensual de la gasolina")
st.caption("Regresión lineal entrenada con variables económicas, energéticas y geopolíticas. "
           "Salida en USD/galón.")

tab1, tab2, tab3 = st.tabs(["Predicción individual", "Predicción por lote (CSV)",
                            "Coeficientes del modelo"])

# ================================================================ TAB 1
with tab1:
    st.subheader("Ingresa las condiciones del mes")
    valores = {}
    cols = st.columns(3)
    for i, v in enumerate(variables_orden):
        with cols[i % 3]:
            if v == "Eventos_Geopoliticos":
                opcion = st.selectbox(ETIQUETAS[v], ["No", "Sí"])
                valores[v] = 1 if opcion == "Sí" else 0
            else:
                mn, mx = rangos[v]
                paso = round((mx - mn) / 100, 2) or 0.1
                valores[v] = st.number_input(
                    ETIQUETAS[v], value=float(round(medianas[v], 2)), step=float(paso),
                    format="%.2f",
                    help=f"Rango de los datos de entrenamiento: {mn:.2f} a {mx:.2f}")

    fuera = [ETIQUETAS[v] for v in variables_orden
             if v != "Eventos_Geopoliticos" and not (rangos[v][0] <= valores[v] <= rangos[v][1])]
    if fuera:
        st.warning("⚠️ Estos valores están fuera del rango con el que se entrenó el modelo, "
                   "así que la predicción es menos confiable: " + ", ".join(fuera))

    if st.button("Predecir precio", type="primary"):
        r = predecir(pd.DataFrame([valores]), nivel).iloc[0]
        pct = int(round(nivel * 100))

        st.subheader("Resultado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precio estimado", f"{r['Precio_predicho']:.2f} USD/galón")
        c2.metric(f"Mínimo ({pct}%)", f"{r['Minimo']:.2f} USD/galón")
        c3.metric(f"Máximo ({pct}%)", f"{r['Maximo']:.2f} USD/galón")

        d = pd.DataFrame({"y": ["Precio"], "Minimo": [r["Minimo"]],
                          "Prediccion": [r["Precio_predicho"]], "Maximo": [r["Maximo"]]})
        base = alt.Chart(d)
        barra = base.mark_rule(size=10, opacity=0.5).encode(
            x=alt.X("Minimo:Q", title="USD/galón", scale=alt.Scale(zero=False)),
            x2="Maximo:Q", y=alt.Y("y:N", axis=None))
        punto = base.mark_point(size=300, filled=True, color="#FF4B4B").encode(
            x="Prediccion:Q", y=alt.Y("y:N", axis=None))
        st.altair_chart((barra + punto).properties(width="container", height=90))

        st.caption(f"Con {pct}% de confianza, el precio real estaría entre "
                   f"{r['Minimo']:.2f} y {r['Maximo']:.2f} USD/galón.")

# ================================================================ TAB 2
with tab2:
    st.subheader("Predicción por lote")
    st.write("Sube un archivo **CSV** con una fila por mes. Debe tener estas columnas "
             "(con estos nombres exactos):")
    st.code(", ".join(variables_orden))

    plantilla = pd.DataFrame([{v: round(medianas[v], 2) for v in variables_orden}])
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8"),
                       file_name="plantilla_prediccion.csv", mime="text/csv")

    archivo = st.file_uploader("Sube tu CSV", type=["csv"])
    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No pude leer el archivo: {e}")
            st.stop()

        faltan = [v for v in variables_orden if v not in datos.columns]
        if faltan:
            st.error("Faltan estas columnas en tu CSV: " + ", ".join(faltan))
        else:
            numericas = [v for v in variables_orden]
            datos[numericas] = datos[numericas].apply(pd.to_numeric, errors="coerce")
            if datos[numericas].isnull().any().any():
                st.warning("Hay celdas vacías o con texto en las variables; esas filas se omiten.")
                datos = datos.dropna(subset=numericas).reset_index(drop=True)

            if datos.empty:
                st.error("No quedaron filas válidas para predecir.")
            else:
                pct = int(round(nivel * 100))
                pred = predecir(datos, nivel)
                salida = datos.copy()
                salida["Precio_predicho_USD_galon"] = pred["Precio_predicho"].round(3)
                salida[f"Minimo_{pct}%"] = pred["Minimo"].round(3)
                salida[f"Maximo_{pct}%"] = pred["Maximo"].round(3)

                st.success(f"✅ Se predijeron {len(salida)} filas.")
                st.dataframe(salida)
                st.download_button("Descargar resultados (CSV)",
                                   salida.to_csv(index=False).encode("utf-8"),
                                   file_name="predicciones_gasolina.csv", mime="text/csv")

# ================================================================ TAB 3
with tab3:
    st.subheader("Coeficientes del modelo")
    st.write("Un **coeficiente** dice cuánto cambia el precio (USD/galón) cuando esa variable "
             "sube **1 unidad**, manteniendo las demás igual. Si es positivo, el precio sube; "
             "si es negativo, el precio baja.")

    b = modelo.params
    media, escala = scaler.mean_, scaler.scale_
    coef_orig = pd.Series(b[variables].values / escala, index=variables)
    intercepto = float(b["const"] - np.sum(b[variables].values * media / escala))

    tabla = pd.DataFrame({
        "Variable": [ETIQUETAS[v] for v in variables_orden],
        "Coeficiente (por unidad)": [coef_orig[v] for v in variables_orden],
        "Efecto estandarizado": [b[v] for v in variables_orden],
        "p-valor": [modelo.pvalues[v] for v in variables_orden],
    })
    tabla["¿Significativa? (p < 0.05)"] = np.where(tabla["p-valor"] < 0.05, "✅ Sí", "❌ No")
    st.dataframe(tabla.round(4), hide_index=True)
    st.write(f"**Intercepto (valor base):** {intercepto:.4f}")

    st.markdown("**Importancia relativa de cada variable** "
                "(efecto estandarizado: mientras más grande la barra, más influye)")
    graf = tabla.set_index("Variable")[["Efecto estandarizado"]]
    st.bar_chart(graf)

    ecuacion = " + ".join(f"({coef_orig[v]:.4f} × {v})" for v in variables_orden)
    st.markdown("**Ecuación del modelo:**")
    st.code(f"Precio = {intercepto:.4f} + {ecuacion}")
