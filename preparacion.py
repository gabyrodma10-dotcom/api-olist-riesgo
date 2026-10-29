
import numpy as np
import pandas as pd


def crear_variables(df):
    datos = df.copy()

    fechas = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for columna in fechas:
        datos[columna] = pd.to_datetime(
            datos[columna], errors="coerce"
        )

    variables = pd.DataFrame(index=datos.index)

    variables["dias_entrega"] = (
        datos["order_delivered_customer_date"]
        - datos["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    variables["dias_retraso"] = (
        datos["order_delivered_customer_date"]
        - datos["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    variables["dias_preparacion"] = (
        datos["order_delivered_carrier_date"]
        - datos["order_approved_at"]
    ).dt.total_seconds() / 86400

    variables["n_items"] = datos["n_items"]
    variables["n_vendedores"] = datos["n_vendedores"]
    variables["precio_total"] = datos["precio_total"]
    variables["flete_total"] = datos["flete_total"]
    variables["peso_total_g"] = datos["peso_total_g"]
    variables["fotos_promedio"] = datos["fotos_promedio"]
    variables["cuotas_max"] = datos["cuotas_max"]

    variables["flete_relativo"] = (
        datos["flete_total"]
        / datos["precio_total"].clip(lower=0.01)
    )

    variables["mes_compra"] = (
        datos["order_purchase_timestamp"].dt.month
    )

    variables["dia_semana_compra"] = (
        datos["order_purchase_timestamp"].dt.dayofweek
    )

    variables["estado_cliente"] = datos["estado_cliente"]
    variables["estado_vendedor"] = datos["estado_vendedor"]
    variables["categoria"] = datos["categoria"]
    variables["tipo_pago"] = datos["tipo_pago"]

    return variables
