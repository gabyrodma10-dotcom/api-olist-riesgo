
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

# Cargar el modelo final entrenado
modelo = joblib.load(BASE_DIR / "modelo_riesgo.joblib")

app = FastAPI(
    title="API de Riesgo de Malas Reseñas - Olist",
    version="1.1"
)


# =========================================================
# 1. CONTRATO DE ENTRADA
# =========================================================

class Pedido(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    order_purchase_timestamp: datetime
    order_approved_at: Optional[datetime] = None
    order_delivered_carrier_date: Optional[datetime] = None
    order_delivered_customer_date: Optional[datetime] = None
    order_estimated_delivery_date: datetime

    estado_cliente: str
    n_items: int = Field(ge=0)
    n_vendedores: int = Field(ge=0)
    precio_total: float = Field(ge=0)
    flete_total: float = Field(ge=0)
    peso_total_g: Optional[float] = Field(default=None, ge=0)
    fotos_promedio: Optional[float] = Field(default=None, ge=0)

    categoria: Optional[str] = None
    estado_vendedor: Optional[str] = None
    tipo_pago: Optional[str] = None
    cuotas_max: Optional[float] = Field(default=None, ge=0)


# =========================================================
# 2. UMBRAL DE RIESGO ALTO
# =========================================================

# Percentil 90 de las probabilidades generadas por
# el modelo final sobre sus datos de entrenamiento.
#
# Valor calculado en Google Colab:
# 0.262102 = 26.21 %
#
# Se utiliza para identificar pedidos que superan
# el punto de corte de priorizacion.

UMBRAL_RIESGO = 0.262102


# =========================================================
# 3. ENDPOINT DE INICIO
# =========================================================

@app.get("/")
def inicio():
    return {
        "mensaje": "API de Olist funcionando"
    }


# =========================================================
# 4. ENDPOINT DE PREDICCION DE RIESGO
# =========================================================

@app.post("/riesgo")
def predecir_riesgo(pedido: Pedido):

    # Convertir los datos recibidos en un diccionario
    datos = pedido.model_dump()

    # Crear el DataFrame con los datos del pedido
    df = pd.DataFrame([datos])

    # Identificar las columnas de fechas
    columnas_fecha = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    # Convertir las fechas al formato esperado
    for columna in columnas_fecha:
        df[columna] = pd.to_datetime(df[columna])

    # Calcular la probabilidad de mala resena
    # utilizando el pipeline XGBoost entrenado
    probabilidad = float(
        modelo.predict_proba(df)[0, 1]
    )

    # Determinar si supera el umbral de riesgo alto
    riesgo_alto = bool(
        probabilidad >= UMBRAL_RIESGO
    )

    # Devolver la prediccion a Streamlit
    return {
        "probabilidad": round(probabilidad, 4),
        "umbral_riesgo": UMBRAL_RIESGO,
        "riesgo_alto": riesgo_alto
    }
