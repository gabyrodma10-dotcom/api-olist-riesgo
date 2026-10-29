
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

# Cargar modelo entrenado
modelo = joblib.load(BASE_DIR / "modelo_riesgo.joblib")

app = FastAPI(
    title="API de Riesgo de Malas Reseñas - Olist",
    version="1.0"
)

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

# Pendiente: sustituir por el umbral Top 10%
# calculado exclusivamente con datos de entrenamiento.

UMBRAL_RIESGO = 0.2704


@app.get("/")
def inicio():
    return {"mensaje": "API de Olist funcionando"}

@app.post("/riesgo")
def predecir_riesgo(pedido: Pedido):
    datos = pedido.model_dump()

    df = pd.DataFrame([datos])

    columnas_fecha = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for columna in columnas_fecha:
        df[columna] = pd.to_datetime(df[columna])

    probabilidad = float(
        modelo.predict_proba(df)[0, 1]
    )

    return {
        "probabilidad": round(probabilidad, 4),
        "riesgo_alto": (
            probabilidad >= UMBRAL_RIESGO
            if UMBRAL_RIESGO is not None
            else None
        )
    }
