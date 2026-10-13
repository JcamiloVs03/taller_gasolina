"""Lógica de predicción compartida por el notebook y la app (sin dependencia de Streamlit)."""
import numpy as np
import pandas as pd
from scipy import stats


def predecir(artefacto, datos, nivel=0.95):
    """Predice el precio (USD/galón) con intervalos del `nivel` indicado.

    Devuelve un DataFrame con:
      prediccion          -> valor estimado
      ic_inf / ic_sup     -> intervalo de confianza del PRECIO PROMEDIO esperado
      ip_inf / ip_sup     -> intervalo de predicción del precio de UN mes concreto
    """
    feats = artefacto["features"]
    X = pd.DataFrame(datos)[feats].astype(float)
    Xs = (X.values - np.array(artefacto["media"])) / np.array(artefacto["escala"])
    A = np.column_stack([np.ones(len(Xs)), Xs])

    beta, XtXi = np.array(artefacto["beta"]), np.array(artefacto["XtXi"])
    s2, gl = artefacto["s2"], artefacto["gl"]
    t = stats.t.ppf(1 - (1 - nivel) / 2, gl)

    y_hat = A @ beta
    h = np.einsum("ij,jk,ik->i", A, XtXi, A)      # apalancamiento a' (X'X)^-1 a
    se_media = np.sqrt(s2 * h)
    se_pred = np.sqrt(s2 * (1 + h))

    return pd.DataFrame({
        "prediccion": y_hat,
        "ic_inf": y_hat - t * se_media, "ic_sup": y_hat + t * se_media,
        "ip_inf": y_hat - t * se_pred, "ip_sup": y_hat + t * se_pred,
    }, index=X.index)


def fuera_de_rango(artefacto, datos):
    """Lista de variables cuyo valor cae fuera del rango visto en el entrenamiento."""
    X = pd.DataFrame(datos)[artefacto["features"]].astype(float)
    avisos = []
    for c in artefacto["features"]:
        lo, hi = artefacto["rangos"][c]["min"], artefacto["rangos"][c]["max"]
        if ((X[c] < lo) | (X[c] > hi)).any():
            avisos.append(c)
    return avisos
